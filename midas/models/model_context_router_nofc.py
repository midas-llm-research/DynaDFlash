"""
Router variant without the FC cross-dimension mixing layer (ablation).
Identical to ContextRouter except the weighted sum goes directly through
RMSNorm with no linear projection.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from midas.config import RouterConfig
from midas.models.model import DFlashDraftModel


class ContextRouterNoFC(nn.Module):
    """ContextRouter without the fc layer — weighted sum + norm only."""

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
        self.output_norm = nn.RMSNorm(config.hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.W_q.weight, gain=0.1)
        nn.init.xavier_normal_(self.W_k.weight, gain=0.1)
        nn.init.zeros_(self.layer_embeddings)

    def forward(
        self, layer_hiddens: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = [self.layer_norms[i](layer_hiddens[i]) for i in range(self.num_layers)]
        x = torch.stack(normed, dim=2)

        q = self.W_q(normed[-1])
        k = self.W_k(x) + self.layer_embeddings

        scores = torch.einsum("btr,btnr->btn", q, k) * (self.rank ** -0.5)

        gamma = self.log_gamma.exp()
        tau = self.log_tau.exp()
        logits = self.log_prior + gamma * scores
        alpha = F.softmax(logits / tau, dim=-1)

        context = torch.einsum("btn,btnh->bth", alpha, x)
        context = self.output_norm(context)

        return context, alpha


class DFlashDraftModelRoutedNoFC(DFlashDraftModel):
    """Router variant without FC — for ablation against the full router."""

    def __init__(self, config, router_config: RouterConfig):
        self._router_config = router_config
        super().__init__(config)
        self.last_routing_weights: Optional[torch.Tensor] = None

    def _build_target_hidden_fusion(self, config) -> None:
        self.context_router = ContextRouterNoFC(self._router_config)

    def _apply_target_hidden_fusion(self, target_hidden: torch.Tensor) -> torch.Tensor:
        layer_hiddens = list(
            torch.chunk(target_hidden, self.context_router.num_layers, dim=-1)
        )
        context, routing_weights = self.context_router(layer_hiddens)
        self.last_routing_weights = routing_weights
        return context

    def log_metrics(self, **base) -> dict:
        metrics = super().log_metrics(**base)
        if self.last_routing_weights is not None:
            mean_weights = self.last_routing_weights.mean(dim=(0, 1)).tolist()
            for i, w in enumerate(mean_weights):
                metrics[f"routing_weight_layer_{i}"] = w
        metrics["gamma"] = self.context_router.log_gamma.exp().item()
        metrics["tau"] = self.context_router.log_tau.exp().item()
        return metrics
