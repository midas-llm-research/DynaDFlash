"""
DFlash draft model with per-layer token mixing across block positions.

Adds a learnable 16×16 position-mixing matrix per layer that provides
unconditional cross-position communication within blocks. This complements
attention (which struggles in early layers when all mask positions have
near-identical content and produce uniform attention weights).

Inherits from DFlashDraftModel and only replaces the decoder layers with
a variant that applies token mixing after the MLP.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from typing_extensions import Unpack
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
)

from midas.models.model import DFlashDraftModel, Qwen3DFlashAttention


class Qwen3DFlashTokenMixDecoderLayer(GradientCheckpointingLayer):
    """Standard DFlash decoder layer + token mixing after MLP."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_size = config.block_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
    ) -> torch.FloatTensor:
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

        # Token mixing: cross-position communication within each block
        B, N, H = hidden_states.shape
        if N % self.block_size == 0:
            num_blocks = N // self.block_size
            blocks = hidden_states.reshape(B * num_blocks, self.block_size, H)
            mixed = self.token_mixer(blocks.transpose(1, 2)).transpose(1, 2)
            hidden_states = hidden_states + mixed.reshape(B, N, H)

        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModelTokenMix(DFlashDraftModel):
    """
    DFlash draft model with per-layer token mixing.

    Each layer applies a learned 16×16 mixing matrix over block positions
    after the MLP, enabling direct position-to-position communication that
    doesn't depend on attention query/key content quality.

    256 params per layer (16×16), 1280 total for 5 layers. Zero-initialized.
    """

    _no_split_modules = ["Qwen3DFlashTokenMixDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3DFlashTokenMixDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()
        # Re-zero the token mixers after post_init (which may re-initialize)
        for layer in self.layers:
            nn.init.zeros_(layer.token_mixer.weight)
