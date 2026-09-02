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
from midas.models.model_dflash2 import (
    BlockLocalDynamicConv,
    CandidateSelector,
    DFlash2DraftModel,
    Qwen3DFlash2DecoderLayer,
)
from midas.models.model_correction import (
    CorrectionBlock,
    CorrectionDraftModel,
    RecycledInputMixer,
    build_correction_block,
    positions_to_redraft,
    should_correct,
    simulate_acceptance,
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

# Backward-compat aliases — use ContextRouter(config) with use_fc=False instead
ContextRouterNoFC = ContextRouter
DFlashDraftModelRoutedNoFC = DFlashDraftModelRouted

__all__ = [
    "ContextRouter",
    "ContextRouterNoFC",
    "AdaptiveLayerFusion",
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "CorrectionBlock",
    "CorrectionDraftModel",
    "RecycledInputMixer",
    "build_correction_block",
    "positions_to_redraft",
    "should_correct",
    "simulate_acceptance",
    "BlockLocalDynamicConv",
    "CandidateSelector",
    "Qwen3DFlash2DecoderLayer",
    "DFlashDraftModelRouted",
    "DFlashDraftModelRoutedNoFC",
    "DFlashDraftModelSplitQuery",
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
