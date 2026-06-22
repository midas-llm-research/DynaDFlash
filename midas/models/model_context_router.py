"""
DFlash draft model with a learned per-token context router replacing the
static concat+FC fusion. Inherits everything from `DFlashDraftModel` and only
overrides the fusion hooks.
"""

import math
from typing import Any, Optional

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
            context: [B, T, H]
            routing_weights: [B, T, N]
        """
        normed = [self.layer_norms[i](layer_hiddens[i]) for i in range(self.num_layers)]
        x = torch.stack(normed, dim=2)  # [B, T, N, H]

        q = self.W_q(normed[-1])
        k = self.W_k(x) + self.layer_embeddings
        scores = torch.einsum("btr,btnr->btn", q, k) * (self.rank ** -0.5)

        gamma = self.log_gamma.exp()
        tau = self.log_tau.exp()
        logits = self.log_prior + gamma * scores
        alpha = F.softmax(logits / tau, dim=-1)

        context = torch.einsum("btn,btnh->bth", alpha, x)
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
        # Stash router_config before super().__init__ so _build_fusion can use it.
        self._router_config = router_config
        super().__init__(config)
        self._last_routing_weights: Optional[torch.Tensor] = None

    def _build_fusion(self, config) -> None:
        self.context_router = ContextRouter(self._router_config)

    def _apply_fusion(self, target_hidden: list[torch.Tensor]) -> torch.Tensor:
        context, routing_weights = self.context_router(target_hidden)
        self._last_routing_weights = routing_weights
        return context

    def extract_target_hidden(
        self, hidden_states: tuple[torch.Tensor, ...]
    ) -> list[torch.Tensor]:
        """Select per-layer hidden states for the router (z-lab offset = +1)."""
        return [hidden_states[layer_id + 1] for layer_id in self.target_layer_ids]

    def routing_aux_loss(self) -> Optional[torch.Tensor]:
        """Routing entropy. Higher = more spread across layers. Encourage by subtracting."""
        if self._last_routing_weights is None:
            return None
        rw = self._last_routing_weights
        return -(rw * rw.clamp(min=1e-8).log()).sum(dim=-1).mean()

    @property
    def last_routing_weights(self) -> Optional[torch.Tensor]:
        return self._last_routing_weights

    @classmethod
    def from_pretrained_dflash(
        cls,
        dflash_checkpoint: str,
        router_config: RouterConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "DFlashDraftModelRouted":
        """Load a pretrained DFlash checkpoint and swap fc+hidden_norm for the router."""
        from transformers import AutoModel

        original = AutoModel.from_pretrained(
            dflash_checkpoint,
            trust_remote_code=True,
            attn_implementation="sdpa",
            **kwargs,
        )
        config = original.config
        config._attn_implementation = "sdpa"

        model = cls(config, router_config)

        original_state = original.state_dict()
        model_state = model.state_dict()
        loaded = []
        for key, value in original_state.items():
            if key.startswith("fc.") or key.startswith("hidden_norm."):
                continue
            if key in model_state:
                model_state[key] = value
                loaded.append(key)
        model.load_state_dict(model_state)

        if device is None:
            device = next(original.parameters()).device
        model = model.to(device=device, dtype=dtype)
        del original

        print(
            f"Loaded {len(loaded)} params from DFlash checkpoint. "
            f"Router params initialized fresh. Model on {device}, {dtype}."
        )
        return model
