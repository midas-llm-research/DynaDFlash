"""
DFlash draft model ablation: separate query projections for target-KV and
draft-KV attention, with gated mixing of the two attention outputs.

Each source gets its own Q projection and independent flash-attention call.
A lightweight learned gate (sigmoid, per-head, per-token) controls the mix,
replacing the joint-softmax competition of the naive formulation while
remaining fully compatible with SDPA / flash attention backends.

Inherits from DFlashDraftModel and only overrides the attention layers.
Everything else (target fusion, decoder structure, generation) is inherited.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Unpack
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3RMSNorm,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.cache_utils import Cache

from midas.models.model import DFlashDraftModel


class Qwen3DFlashSplitQueryAttention(nn.Module):
    """
    DFlash attention with separate query projections for target-KV and draft-KV.

    Q_t = q_proj_target(H_d)  — used to attend over target context KV
    Q_d = q_proj_draft(H_d)   — used to attend over draft/noise KV

    Two independent SDPA calls (flash-compatible), mixed via a learned gate.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj_target = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_proj_draft = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm_target = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_draft = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        # Separate query projections with independent norms
        q_t = self.q_proj_target(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q_t = self.q_norm_target(q_t).transpose(1, 2)
        q_d = self.q_proj_draft(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        q_d = self.q_norm_draft(q_d).transpose(1, 2)

        # Shared K/V projections (same as baseline)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        cos_unsq = cos.unsqueeze(1)
        sin_unsq = sin.unsqueeze(1)
        cos_q = cos_unsq[..., -q_len:, :]
        sin_q = sin_unsq[..., -q_len:, :]
        q_t = (q_t * cos_q) + (rotate_half(q_t) * sin_q)
        q_d = (q_d * cos_q) + (rotate_half(q_d) * sin_q)
        k = (k * cos_unsq) + (rotate_half(k) * sin_unsq)

        # Cache update
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Split K/V into target-context and draft portions
        total_kv_len = k.shape[2]
        target_kv_len = total_kv_len - q_len
        k_t, k_d = k[..., :target_kv_len, :], k[..., target_kv_len:, :]
        v_t, v_d = v[..., :target_kv_len, :], v[..., target_kv_len:, :]

        # Split attention mask
        mask_t = attention_mask[..., :target_kv_len] if attention_mask is not None else None
        mask_d = attention_mask[..., target_kv_len:] if attention_mask is not None else None

        # Attention dispatch (uses SDPA/flash backend, handles GQA internally)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        dropout = 0.0 if not self.training else self.attention_dropout

        out_t, _ = attn_fn(
            self, q_t, k_t, v_t, mask_t,
            dropout=dropout, scaling=self.scaling, **kwargs,
        )
        out_d, _ = attn_fn(
            self, q_d, k_d, v_d, mask_d,
            dropout=dropout, scaling=self.scaling, **kwargs,
        )

        # Learned gate: per-token, per-head mixing of target vs draft attention
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [B, q_len, num_heads]
        gate = gate.unsqueeze(-1)  # [B, q_len, num_heads, 1]

        attn_output = gate * out_t + (1.0 - gate) * out_d
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen3DFlashSplitQueryDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashSplitQueryAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModelSplitQuery(DFlashDraftModel):
    """
    DFlash draft model with separate query projections for target-KV and
    draft-KV attention. Tests whether giving the model independent query
    spaces for the two KV sources improves block-diffusion drafting.

    Uses two independent SDPA calls (flash-compatible) with a learned gate
    to mix outputs — architecturally distinct from DFlare's heterogeneous
    KV projections (which split on the key side).

    Only the attention layers are replaced; everything else (target fusion,
    loss, generation) is inherited from DFlashDraftModel.
    """

    _no_split_modules = ["Qwen3DFlashSplitQueryDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3DFlashSplitQueryDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()
        # Zero-init gate after post_init so it starts at sigmoid(0) = 0.5
        for layer in self.layers:
            nn.init.zeros_(layer.self_attn.gate_proj.weight)
            nn.init.zeros_(layer.self_attn.gate_proj.bias)

    def log_metrics(self, **base) -> dict:
        metrics = super().log_metrics(**base)
        attn = self.layers[0].self_attn
        norm_t = attn.q_proj_target.weight.norm().item()
        norm_d = attn.q_proj_draft.weight.norm().item()
        metrics["q_norm_target_L0"] = norm_t
        metrics["q_norm_draft_L0"] = norm_d
        metrics["q_norm_ratio_L0"] = norm_t / max(norm_d, 1e-8)
        gate_bias = attn.gate_proj.bias.detach()
        metrics["gate_mean_L0"] = gate_bias.sigmoid().mean().item()
        return metrics
