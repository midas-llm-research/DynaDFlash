from midas.models.model import (
    DFlashDraftModel,
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    build_target_layer_ids,
    dflash_generate,
    sample,
)
from midas.models.model_context_router import (
    ContextRouter,
    DFlashDraftModelRouted,
)
from midas.models.model_dflare import (
    AdaptiveLayerFusion,
    DFlareDraftModel,
    Qwen3DFlareAttention,
    Qwen3DFlareDecoderLayer,
    build_dflare_target_layer_ids,
    compute_position_weights,
    compute_progressive_gamma,
)

# Backward-compat aliases — use ContextRouter(config) with use_fc=False instead
ContextRouterNoFC = ContextRouter
DFlashDraftModelRoutedNoFC = DFlashDraftModelRouted

__all__ = [
    "ContextRouter",
    "ContextRouterNoFC",
    "AdaptiveLayerFusion",
    "DFlashDraftModel",
    "DFlashDraftModelRouted",
    "DFlashDraftModelRoutedNoFC",
    "DFlareDraftModel",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "Qwen3DFlareAttention",
    "Qwen3DFlareDecoderLayer",
    "build_target_layer_ids",
    "build_dflare_target_layer_ids",
    "compute_position_weights",
    "compute_progressive_gamma",
    "dflash_generate",
    "sample",
]
