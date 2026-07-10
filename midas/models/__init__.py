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
    compute_progressive_gamma,
)
from midas.models.model_split_query import (
    DFlashDraftModelSplitQuery,
    Qwen3DFlashSplitQueryAttention,
    Qwen3DFlashSplitQueryDecoderLayer,
)
from midas.models.model_causal_block import DFlashDraftModelCausalBlock
from midas.models.model_token_mix import (
    DFlashDraftModelTokenMix,
    Qwen3DFlashTokenMixDecoderLayer,
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
    "DFlashDraftModelCausalBlock",
    "DFlashDraftModelSplitQuery",
    "DFlashDraftModelTokenMix",
    "DFlareDraftModel",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "Qwen3DFlashSplitQueryAttention",
    "Qwen3DFlashSplitQueryDecoderLayer",
    "Qwen3DFlareAttention",
    "Qwen3DFlareDecoderLayer",
    "build_target_layer_ids",
    "build_dflare_target_layer_ids",
    "compute_progressive_gamma",
    "dflash_generate",
    "sample",
]
