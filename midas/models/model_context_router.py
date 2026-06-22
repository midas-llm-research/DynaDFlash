"""
DFlash draft model with a learned per-token context router replacing the
static concat+FC fusion. Inherits everything from `DFlashDraftModel` and only
overrides the target-hidden fusion hooks.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from midas.config import RouterConfig
from midas.models.model import DFlashDraftModel


# ---------------------------------------------------------------------------
# Context router
# ---------------------------------------------------------------------------


class ContextRouter(nn.Module):
    """
    Per-token attention-style router over N target layer hidden states.

    Query from the deepest selected layer, keys from each layer (with a
    learnable per-layer embedding added), softmax-mixed with a static log prior
    and learnable temperature. Result goes through a linear projection + RMSNorm
    so the downstream attention receives the same [B, T, H] shape it would have
    received from the static fc+hidden_norm fusion.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.rank = config.rank

        self.layer_norms = nn.ModuleList(
            [nn.RMSNorm(config.hidden_size) for _ in range(self.num_layers)]
        )
        self.W_q = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.W_k = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.layer_embeddings = nn.Parameter(torch.zeros(self.num_layers, config.rank))
        self.log_prior = nn.Parameter(torch.zeros(self.num_layers))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(config.gamma_init)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(config.temperature_init)))
        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output_norm = nn.RMSNorm(config.hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.W_q.weight, gain=0.1)
        nn.init.xavier_normal_(self.W_k.weight, gain=0.1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.layer_embeddings)

    def forward(
        self, layer_hiddens: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            layer_hiddens: list of N tensors, each [B, T, H]

        Returns:
            context: [B, T, H] routed context vector
            routing_weights: [B, T, N] for logging/analysis
        """
        # Per-layer normalization
        normed = [self.layer_norms[i](layer_hiddens[i]) for i in range(self.num_layers)]

        # Stack for batch operations: [B, T, N, H]
        x = torch.stack(normed, dim=2)

        # Query from deepest selected layer: [B, T, r]
        q = self.W_q(normed[-1])

        # Keys for all layers: [B, T, N, r]
        k = self.W_k(x) + self.layer_embeddings

        # Routing scores: [B, T, N]
        scores = torch.einsum("btr,btnr->btn", q, k) * (self.rank ** -0.5)

        # Combine static prior with dynamic scores
        gamma = self.log_gamma.exp()
        tau = self.log_tau.exp()
        logits = self.log_prior + gamma * scores

        # Routing weights
        alpha = F.softmax(logits / tau, dim=-1)

        # Weighted sum over layers: [B, T, H]
        context = torch.einsum("btn,btnh->bth", alpha, x)

        # Cross-dimension mixing + normalize
        context = self.output_norm(self.fc(context))

        return context, alpha


# ---------------------------------------------------------------------------
# Routed draft model
# ---------------------------------------------------------------------------


class DFlashDraftModelRouted(DFlashDraftModel):
    """
    DFlash draft model with `ContextRouter` replacing the static fc+hidden_norm
    fusion. All other components — attention, decoder layers, generation —
    inherit unchanged from `DFlashDraftModel`.
    """

    def __init__(self, config, router_config: RouterConfig):
        # Stash router_config before super().__init__ so the base __init__'s
        # call to `_build_target_hidden_fusion` can use it.
        self._router_config = router_config
        super().__init__(config)
        self.last_routing_weights: Optional[torch.Tensor] = None

    def _build_target_hidden_fusion(self, config) -> None:
        self.context_router = ContextRouter(self._router_config)

    def _apply_target_hidden_fusion(self, target_hidden: torch.Tensor) -> torch.Tensor:
        # target_hidden is the concat tensor [B, T, N*H] from the base
        # `extract_target_hidden`. Chunk back into N per-layer views (zero-copy)
        # so the router can do its per-layer attention.
        layer_hiddens = list(
            torch.chunk(target_hidden, self.context_router.num_layers, dim=-1)
        )
        context, routing_weights = self.context_router(layer_hiddens)
        self.last_routing_weights = routing_weights
        return context

    def routing_aux_loss(self) -> Optional[torch.Tensor]:
        """Routing entropy. Higher = more spread across layers. Encourage by subtracting."""
        if self.last_routing_weights is None:
            return None
        rw = self.last_routing_weights
        return -(rw * rw.clamp(min=1e-8).log()).sum(dim=-1).mean()

