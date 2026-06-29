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
from midas.models.model_skip import (
    BlockAggregator,
    DFlashDraftModelSkip,
)

# Backward-compat aliases — use ContextRouter(config) with use_fc=False instead
ContextRouterNoFC = ContextRouter
DFlashDraftModelRoutedNoFC = DFlashDraftModelRouted

__all__ = [
    "BlockAggregator",
    "ContextRouter",
    "ContextRouterNoFC",
    "DFlashDraftModel",
    "DFlashDraftModelRouted",
    "DFlashDraftModelRoutedNoFC",
    "DFlashDraftModelSkip",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "build_target_layer_ids",
    "dflash_generate",
    "sample",
]
