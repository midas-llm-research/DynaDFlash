import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from midas.config import RouterConfig


class ContextRouter(nn.Module):
    """
    Dynamic attention-style router over target model hidden states.

    Replaces DFlash's static concat+FC fusion with per-token routing weights
    over N selected target layers, followed by FC for cross-dimension mixing.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.rank = config.rank

        # Per-layer RMSNorm
        self.layer_norms = nn.ModuleList(
            [nn.RMSNorm(config.hidden_size) for _ in range(self.num_layers)]
        )

        # Query projection (from deepest layer)
        self.W_q = nn.Linear(config.hidden_size, config.rank, bias=False)

        # Key projection (shared across layers)
        self.W_k = nn.Linear(config.hidden_size, config.rank, bias=False)

        # Learnable layer identity embeddings added to keys
        self.layer_embeddings = nn.Parameter(torch.zeros(self.num_layers, config.rank))

        # Static prior over layers (log-space, init uniform)
        self.log_prior = nn.Parameter(torch.full((self.num_layers,), 0.0))

        # Dynamic strength (init near zero so routing starts static)
        self.log_gamma = nn.Parameter(torch.tensor(math.log(config.gamma_init)))

        # Softmax temperature (learnable)
        self.log_tau = nn.Parameter(torch.tensor(math.log(config.temperature_init)))

        # Cross-dimension mixing after weighted sum
        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Output normalization
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
