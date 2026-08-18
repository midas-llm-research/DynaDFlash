"""
DFlare with a hybrid shared-and-specialized target conditioning.

Follows the decomposition AngelSpec/DFly introduces (Eq. 12): a *shared* branch
that models cross-layer feature interactions is added to DFlare's *layer-specific*
weighted sum, and the sum is normalized per draft layer:

    g^(i)_t = RMSNorm( c_t + f^(i)_t )

DFly uses DFlash's concat+FC projection for the shared branch c_t. Here we
substitute our attention-style ContextRouter, which mixes the target layers
per *token* rather than with one static projection:

    c_t     = ContextRouter(h^(1..T)_t)          shared across draft layers
    f^(i)_t = sum_j softmax(W_fuse)[i,j] h^(j)_t  DFlare, per draft layer i

Everything else is DFlare unchanged: heterogeneous KV projections, per-layer
target context, progressive position-weighted loss.

Note the RMSNorm is applied *after* the addition (Eq. 12), so DFlare's
`output_norms` are reused in that position rather than normalizing the weighted
sum on its own. No new normalization parameters are introduced.
"""

from typing import Optional

import torch

from midas.config import DFlareConfig, RouterConfig
from midas.models.model_context_router import ContextRouter
from midas.models.model_dflare import DFlareDraftModel


class DFlareRouterDraftModel(DFlareDraftModel):
    """DFlare + a shared ContextRouter branch added to the per-layer fusion."""

    def __init__(
        self,
        config,
        dflare_config: DFlareConfig,
        router_config: RouterConfig,
    ):
        if router_config.num_layers != dflare_config.num_target_layers:
            raise ValueError(
                f"router.num_layers ({router_config.num_layers}) must equal "
                f"dflare.num_target_layers ({dflare_config.num_target_layers}) — "
                f"both branches read the same extracted target layers."
            )
        if router_config.hidden_size != dflare_config.hidden_size:
            raise ValueError(
                f"router.hidden_size ({router_config.hidden_size}) must equal "
                f"dflare.hidden_size ({dflare_config.hidden_size})."
            )
        super().__init__(config, dflare_config)
        self.context_router = ContextRouter(router_config)
        self.post_init()
        self.last_routing_weights: Optional[torch.Tensor] = None

    def _build_layer_contexts(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """target_hidden [T, B, L, H] -> per-draft-layer contexts [D, B, L, H]."""
        # Shared branch: per-token routed mix over target layers -> [B, L, H].
        # unbind gives T zero-copy [B, L, H] views, the format ContextRouter wants.
        context, routing_weights = self.context_router(list(target_hidden.unbind(0)))
        self.last_routing_weights = routing_weights

        # Specialized branch: DFlare's raw weighted sum, left unnormalized so the
        # RMSNorm lands after the addition as in Eq. 12.
        fusion = self.adaptive_layer_fusion
        weights = fusion.fusion_weights().to(dtype=target_hidden.dtype)
        fused = torch.einsum("dt,tblh->dblh", weights, target_hidden)

        return torch.stack(
            [
                fusion.output_norms[i](fused[i] + context)
                for i in range(fusion.num_draft_layers)
            ],
            dim=0,
        )

    def log_metrics(self, **base) -> dict:
        metrics = super().log_metrics(**base)
        if self.last_routing_weights is not None:
            for i, w in enumerate(self.last_routing_weights.mean(dim=(0, 1)).tolist()):
                metrics[f"routing_weight_layer_{i}"] = w
        metrics["router_gamma"] = self.context_router.log_gamma.exp().item()
        metrics["router_tau"] = self.context_router.log_tau.exp().item()
        return metrics
