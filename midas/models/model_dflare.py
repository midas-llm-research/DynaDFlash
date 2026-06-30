"""DFlare draft model components.

This module implements the paper-specific DFlare changes on top of the local
DFlash draft model shape:

1. Adaptive layer fusion: each draft layer learns softmax weights over selected
   target hidden layers and receives its own fused target context.
2. Heterogeneous KV projections: target context has dedicated K/V projections,
   while block positions use the draft K/V projections.
3. Progressive position-weighted loss support via a small gamma schedule helper.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)
from typing_extensions import Unpack

from midas.config import DFlareConfig
from midas.models.model import (
    apply_rotary_pos_emb,
    compute_position_weighted_ce_loss,
    dflash_generate,
)

if TYPE_CHECKING:
    from midas.data import DraftTrainingBatch


def build_dflare_target_layer_ids(
    num_target_layers: int,
    num_fused_target_layers: int,
) -> list[int]:
    """Uniformly select target layers from second through third-to-last."""
    if num_fused_target_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_fused_target_layers - 1)))
        for i in range(num_fused_target_layers)
    ]


def compute_progressive_gamma(config: DFlareConfig, epoch: int) -> float:
    return config.gamma_initial + config.gamma_increment_per_epoch * epoch


class AdaptiveLayerFusion(nn.Module):
    """DFlare adaptive layer fusion from paper Section 4.1."""

    def __init__(self, config: DFlareConfig):
        super().__init__()
        self.num_draft_layers = config.num_draft_layers
        self.num_target_layers = config.num_target_layers
        self.hidden_size = config.hidden_size
        self.fusion_logits = nn.Parameter(
            torch.zeros(config.num_draft_layers, config.num_target_layers)
        )
        self.output_norms = nn.ModuleList(
            [nn.RMSNorm(config.hidden_size) for _ in range(config.num_draft_layers)]
        )

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits, dim=-1)

    def forward(self, layer_hiddens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_hiddens: [T, B, L, H] selected target hidden states.

        Returns:
            [D, B, L, H] fused target context, one tensor per draft layer.
        """
        if layer_hiddens.shape[0] != self.num_target_layers:
            raise ValueError(
                f"Expected {self.num_target_layers} target layers, "
                f"got {layer_hiddens.shape[0]}"
            )
        weights = self.fusion_weights().to(dtype=layer_hiddens.dtype)
        fused = torch.einsum("dt,tblh->dblh", weights, layer_hiddens)
        return torch.stack(
            [self.output_norms[i](fused[i]) for i in range(self.num_draft_layers)],
            dim=0,
        )


class Qwen3DFlareAttention(nn.Module):
    """DFlare attention with separate target-context K/V projections."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.target_k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.target_v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

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

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        k_ctx = self.target_k_proj(target_hidden)
        v_ctx = self.target_v_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlareDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlareAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
        hidden_states = residual + hidden_states
        return hidden_states


class DFlareDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlareDecoderLayer"]

    def __init__(self, config: Qwen3Config, dflare_config: DFlareConfig):
        config = copy.deepcopy(config)
        config.num_hidden_layers = dflare_config.num_draft_layers
        super().__init__(config)
        if dflare_config.hidden_size != config.hidden_size:
            raise ValueError(
                f"DFlare hidden_size={dflare_config.hidden_size} must match "
                f"model hidden_size={config.hidden_size}"
            )
        self.config = config
        self.dflare_config = dflare_config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlareDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = build_dflare_target_layer_ids(
            config.num_target_layers,
            dflare_config.num_target_layers,
        )
        self.adaptive_layer_fusion = AdaptiveLayerFusion(dflare_config)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def extract_target_hidden(
        self, hidden_states: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        offset = 1
        return torch.stack(
            [hidden_states[layer_id + offset] for layer_id in self.target_layer_ids],
            dim=0,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        fused_contexts = self.adaptive_layer_fusion(target_hidden)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=fused_contexts[layer_idx],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def compute_loss(
        self,
        draft_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        batch: "DraftTrainingBatch",
        lm_head: nn.Module,
        *,
        gamma: float = 4.5,
        **_,
    ) -> torch.Tensor:
        return compute_position_weighted_ce_loss(
            draft_hidden, input_ids, batch, lm_head, gamma=gamma
        )

    def log_metrics(self, **base) -> dict:
        metrics = dict(base)
        weights = self.adaptive_layer_fusion.fusion_weights().detach()
        for draft_layer in range(weights.shape[0]):
            for target_layer in range(weights.shape[1]):
                metrics[f"fusion_d{draft_layer}_t{target_layer}"] = weights[
                    draft_layer, target_layer
                ].item()
        return metrics

    @classmethod
    def from_pretrained_dflash(
        cls,
        dflash_checkpoint: str,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        **constructor_kwargs,
    ) -> "DFlareDraftModel":
        original = AutoModel.from_pretrained(
            dflash_checkpoint,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        config = original.config
        config._attn_implementation = "sdpa"
        model = cls(config, **constructor_kwargs)
        model_state = model.state_dict()
        loaded = 0
        for key, value in original.state_dict().items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
        model.load_state_dict(model_state)
        if device is None:
            device = next(original.parameters()).device
        model = model.to(device=device, dtype=dtype)
        del original
        print(
            f"Loaded {loaded} shape-compatible params from {dflash_checkpoint} "
            f"into {cls.__name__}. Model on {device}, {dtype}."
        )
        return model

    @classmethod
    def from_config_random(
        cls,
        dflash_checkpoint: str,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        **constructor_kwargs,
    ) -> "DFlareDraftModel":
        config = AutoConfig.from_pretrained(dflash_checkpoint, trust_remote_code=True)
        config._attn_implementation = "sdpa"
        model = cls(config, **constructor_kwargs)
        if device is not None:
            model = model.to(device=device, dtype=dtype)
        print(
            f"Created {cls.__name__} with random weights. "
            f"{sum(p.numel() for p in model.parameters()):,} params on {device}, {dtype}."
        )
        return model

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
