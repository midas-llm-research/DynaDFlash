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
from midas.models.model_split_query import (
    DFlashDraftModelSplitQuery,
    Qwen3DFlashSplitQueryAttention,
    Qwen3DFlashSplitQueryDecoderLayer,
)

# Backward-compat aliases — use ContextRouter(config) with use_fc=False instead
ContextRouterNoFC = ContextRouter
DFlashDraftModelRoutedNoFC = DFlashDraftModelRouted

__all__ = [
    "ContextRouter",
    "ContextRouterNoFC",
    "DFlashDraftModel",
    "DFlashDraftModelRouted",
    "DFlashDraftModelRoutedNoFC",
    "DFlashDraftModelSplitQuery",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "Qwen3DFlashSplitQueryAttention",
    "Qwen3DFlashSplitQueryDecoderLayer",
    "build_target_layer_ids",
    "dflash_generate",
    "sample",
]
