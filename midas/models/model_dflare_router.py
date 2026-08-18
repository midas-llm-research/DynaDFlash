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

The RMSNorm is applied *after* the addition (Eq. 12), so DFlare's `output_norms`
are reused in that position rather than normalizing the weighted sum on its own.

One deviation from a literal Eq. 12, forced by the router: AngelSpec's two
branches both read *raw* target hidden states and so are naturally on the same
scale. Our router normalizes each target layer before mixing, so its output is
unit RMS while the raw weighted sum measures ~6.3 RMS on Qwen3-4B — a plain sum
would hand the router only ~14% of the signal. We therefore rescale the weighted
sum to unit RMS before adding. This is parameterless (the learnable gain comes
from output_norms[i] on the sum), so no new parameters are introduced and the
context reaching the KV projections keeps the same unit scale stock DFlare
produces.
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

        # Specialized branch: DFlare's weighted sum over raw target hidden states.
        fusion = self.adaptive_layer_fusion
        weights = fusion.fusion_weights().to(dtype=target_hidden.dtype)
        fused = torch.einsum("dt,tblh->dblh", weights, target_hidden)

        # Scale-match the two branches before adding. Measured on real Qwen3-4B
        # hidden states, the raw weighted sum has RMS ~6.3 while the router's
        # output is exactly unit RMS (it normalizes each target layer before
        # mixing), so a plain sum would give the router only ~14% of the signal.
        # AngelSpec's branches are naturally balanced because both read raw
        # hidden states; ours are not, so we normalize this one to unit RMS.
        # Parameterless on purpose — the learnable gain is already provided by
        # output_norms[i] applied to the sum, so this adds no parameters and
        # leaves the context reaching the KV projections at the same unit scale
        # stock DFlare produces.
        fused = fused * torch.rsqrt(
            fused.float().pow(2).mean(-1, keepdim=True) + 1e-6
        ).to(fused.dtype)

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
