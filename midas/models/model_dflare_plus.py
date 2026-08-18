"""
DFlare+ draft model: DFlare with token mixing + configurable MLP width.

Inherits DFlare's adaptive layer fusion and heterogeneous KV projections. Adds:
1. Per-layer token mixing (block_size x block_size learned position mixing after MLP)
2. Configurable intermediate_size for MLP width (enables more layers at iso-compute)

Deliberately does NOT use a causal-within-block mask. Attention stays
bidirectional within the block, exactly as in DFlare and DFlareDeep, so that
this model is a strict superset of DFlareDeepDraftModel: identical architecture
plus the zero-initialized token mixers. That makes dflare-plus vs dflare-deep a
clean one-variable ablation on token mixing alone.

(The causal-within-block mechanism lives in DFlashDraftModelCausalBlock and in
`prepare_draft_inputs(causal_block=True)`; it was measured at +0.3% standalone
and would additionally be un-attributable here, since an unmasked token mixer
can reintroduce the late->early information flow the mask removes.)
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
)
from transformers.cache_utils import Cache
from typing_extensions import Unpack

from midas.models.model_dflare import (
    DFlareDraftModel,
    Qwen3DFlareAttention,
)


class Qwen3DFlarePlusDecoderLayer(GradientCheckpointingLayer):
    """DFlare decoder layer with token mixing after MLP."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_size = config.block_size
        self.self_attn = Qwen3DFlareAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.token_mixer = nn.Linear(config.block_size, config.block_size, bias=False)
        nn.init.zeros_(self.token_mixer.weight)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Token mixing across positions within each block. Positions are laid out
        # block-major ([blk0_pos0..blk0_pos15, blk1_pos0..], see prepare_draft_inputs),
        # so this reshape groups by block and never mixes across block boundaries.
        # Assert rather than skip: silently disabling the mixer would make the
        # inference path diverge from what was trained.
        B, N, H = hidden_states.shape
        assert N % self.block_size == 0, (
            f"sequence length {N} is not a multiple of block_size {self.block_size}; "
            f"token mixing would silently be skipped"
        )
        num_blocks = N // self.block_size
        blocks = hidden_states.reshape(B * num_blocks, self.block_size, H)
        mixed = self.token_mixer(blocks.transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states + mixed.reshape(B, N, H)

        hidden_states = residual + hidden_states
        return hidden_states


class DFlarePlusDraftModel(DFlareDraftModel):
    """DFlare + per-layer token mixing + configurable MLP width.

    Architecturally identical to DFlareDeepDraftModel (same layer count, same
    shrunk intermediate_size, same adaptive fusion and heterogeneous KV, same
    bidirectional within-block attention) plus one zero-initialized
    block_size x block_size mixing matrix per layer.

    No forward() override: it uses DFlare's forward unchanged, so the training
    and inference paths are exactly those already validated by dflare and
    dflare-deep.
    """

    _no_split_modules = ["Qwen3DFlarePlusDecoderLayer"]

    def __init__(self, config: Qwen3Config, dflare_config):
        # Override intermediate_size before super().__init__ builds layers.
        # Copy first — super() deep-copies, but only after we'd have mutated the
        # caller's config, which leaks into any model built from it afterwards.
        # (DFlareDeepDraftModel does the same.)
        if hasattr(dflare_config, "intermediate_size") and dflare_config.intermediate_size is not None:
            config = copy.deepcopy(config)
            config.intermediate_size = dflare_config.intermediate_size
        super().__init__(config, dflare_config)
        # Replace DFlare layers with our custom layers (which add token mixing)
        self.layers = nn.ModuleList(
            [
                Qwen3DFlarePlusDecoderLayer(self.config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.post_init()
        # post_init re-randomizes the freshly built layers, including the mixers;
        # re-zero them last so warm-start from dflare-deep behaviour is exact.
        for layer in self.layers:
            nn.init.zeros_(layer.token_mixer.weight)
