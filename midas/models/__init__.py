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

# Backward-compat aliases — use ContextRouter(config) with use_fc=False instead
ContextRouterNoFC = ContextRouter
DFlashDraftModelRoutedNoFC = DFlashDraftModelRouted

__all__ = [
    "ContextRouter",
    "ContextRouterNoFC",
    "DFlashDraftModel",
    "DFlashDraftModelRouted",
    "DFlashDraftModelRoutedNoFC",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "build_target_layer_ids",
    "dflash_generate",
    "sample",
]
