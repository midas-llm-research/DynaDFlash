"""
DFlare+ draft model: DFlare with causal block mask + token mixing + configurable MLP.

Inherits DFlare's adaptive layer fusion and heterogeneous KV projections. Adds:
1. Causal attention within blocks (position k attends only to 0..k)
2. Per-layer token mixing (16×16 learned position mixing after MLP)
3. Configurable intermediate_size for MLP width (enables more layers at iso-compute)
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
)
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

        B, N, H = hidden_states.shape
        if N % self.block_size == 0:
            num_blocks = N // self.block_size
            blocks = hidden_states.reshape(B * num_blocks, self.block_size, H)
            mixed = self.token_mixer(blocks.transpose(1, 2)).transpose(1, 2)
            hidden_states = hidden_states + mixed.reshape(B, N, H)

        hidden_states = residual + hidden_states
        return hidden_states


class DFlarePlusDraftModel(DFlareDraftModel):
    """DFlare + causal block mask + token mixing + configurable MLP width.

    Inherits adaptive layer fusion and heterogeneous KV from DFlare.
    Adds causal-within-block attention, per-layer token mixing, and
    supports shrunk intermediate_size for more layers at iso-compute.
    """

    _no_split_modules = ["Qwen3DFlarePlusDecoderLayer"]
    causal_block = True

    def __init__(self, config: Qwen3Config, dflare_config):
        # Override intermediate_size before super().__init__ builds layers
        # (super deep-copies config, so we modify the original first)
        if hasattr(dflare_config, "intermediate_size") and dflare_config.intermediate_size is not None:
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
        for layer in self.layers:
            nn.init.zeros_(layer.token_mixer.weight)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if use_cache and attention_mask is None:
            B, N, H = noise_embedding.shape
            fused_contexts = self.adaptive_layer_fusion(target_hidden)
            ctx_len = fused_contexts.shape[2]
            cache_len = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            total_kv = cache_len + ctx_len + N

            mask = torch.zeros(
                B, 1, N, total_kv,
                device=noise_embedding.device,
                dtype=noise_embedding.dtype,
            )
            causal = torch.triu(
                torch.full(
                    (N, N),
                    float("-inf"),
                    device=noise_embedding.device,
                    dtype=noise_embedding.dtype,
                ),
                diagonal=1,
            )
            mask[:, :, :, -N:] = causal

            hidden_states = noise_embedding
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer_idx, layer in enumerate(self.layers):
                hidden_states = layer(
                    hidden_states=hidden_states,
                    target_hidden=fused_contexts[layer_idx],
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            return self.norm(hidden_states)

        return super().forward(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
