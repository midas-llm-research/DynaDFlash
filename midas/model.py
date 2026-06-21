"""
Minimal fork of z-lab/dflash model.py.

Only change: DFlashDraftModel's concat+FC fusion replaced with ContextRouter.
All other components (attention, decoder layers, generation) are identical to upstream.
"""

import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from typing_extensions import Unpack
from transformers import DynamicCache
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
    rotate_half,
)

from midas.config import RouterConfig
from midas.router import ContextRouter


# --------------------------------------------------------------------------
# Utilities (unchanged from z-lab/dflash)
# --------------------------------------------------------------------------


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature_separated(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int],
) -> list[torch.Tensor]:
    """Return individual layer hidden states as a list (no concatenation)."""
    offset = 1
    return [hidden_states[layer_id + offset] for layer_id in layer_ids]


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


# --------------------------------------------------------------------------
# Attention and Decoder (unchanged from z-lab/dflash)
# --------------------------------------------------------------------------


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
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

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
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

        attn_fn = eager_attention_forward
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


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
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


# --------------------------------------------------------------------------
# DFlash Draft Model with Context Router (our modification)
# --------------------------------------------------------------------------


class DFlashDraftModelRouted(Qwen3PreTrainedModel):
    """
    DFlash draft model with dynamic context routing.

    Identical to z-lab's DFlashDraftModel except:
    - self.fc + self.hidden_norm replaced by self.context_router
    - extract_context_feature_separated returns list (not concat)
    """

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config: Qwen3Config, router_config: RouterConfig):
        super().__init__(config)
        self.config = config

        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.target_layer_ids = config.dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        # THE CHANGE: dynamic router replaces static fc + hidden_norm
        self.context_router = ContextRouter(router_config)

        self.block_size = config.block_size
        self.mask_token_id = config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[list[torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        return_routing_weights: bool = False,
        **kwargs,
    ):
        hidden_states = noise_embedding

        # Dynamic routing over target layer hidden states
        target_context, routing_weights = self.context_router(target_hidden)

        # Position embeddings must cover ctx_len + noise_len for attention keys.
        # In attention, keys = cat([k_ctx, k_noise]) so RoPE needs positions for both.
        # Context and noise represent the SAME sequence positions, so we repeat:
        # [0, 1, ..., T-1, 0, 1, ..., T-1] — both halves get correct position encoding.
        ctx_len = target_context.shape[1]
        noise_len = hidden_states.shape[1]
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            # Training mode (no cache): need positions for full ctx+noise
            ctx_positions = torch.arange(ctx_len, device=position_ids.device)
            full_position_ids = torch.cat([ctx_positions, position_ids[0]]).unsqueeze(0).expand(
                hidden_states.shape[0], -1
            )
            position_embeddings = self.rotary_emb(hidden_states, full_position_ids)
        else:
            # Inference mode (with cache): position_ids already covers ctx+noise
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_context,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        output = self.norm(hidden_states)

        if return_routing_weights:
            return output, routing_weights
        return output

    @classmethod
    def from_pretrained_dflash(
        cls,
        dflash_checkpoint: str,
        router_config: RouterConfig,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "DFlashDraftModelRouted":
        """Load pre-trained DFlash draft, replace fusion with router."""
        from transformers import AutoModel

        # Load the original DFlash model to get config and weights
        original = AutoModel.from_pretrained(
            dflash_checkpoint, trust_remote_code=True,
            attn_implementation="sdpa", **kwargs
        )
        config = original.config
        config._attn_implementation = "sdpa"

        # Create our routed model
        model = cls(config, router_config)

        # Copy all shared weights (layers, norm, rotary_emb)
        original_state = original.state_dict()
        model_state = model.state_dict()

        loaded_keys = []
        for key in original_state:
            # Skip the old fusion params
            if key.startswith("fc.") or key.startswith("hidden_norm."):
                continue
            if key in model_state:
                model_state[key] = original_state[key]
                loaded_keys.append(key)

        model.load_state_dict(model_state)

        # Move entire model to correct device and dtype
        if device is None:
            device = next(original.parameters()).device
        model = model.to(device=device, dtype=dtype)

        # Free original model memory
        del original

        print(
            f"Loaded {len(loaded_keys)} params from DFlash checkpoint. "
            f"Router params initialized fresh. Model on {device}, {dtype}."
        )
        return model


# --------------------------------------------------------------------------
# Generation (adapted from z-lab/dflash for separated hidden states)
# --------------------------------------------------------------------------


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModelRouted,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    return_stats: bool = False,
    return_routing_weights: bool = False,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits, temperature
    )
    if block_size > 1:
        target_hidden = extract_context_feature_separated(
            output.hidden_states, model.target_layer_ids
        )
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    all_routing_weights = []
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_out = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                return_routing_weights=return_routing_weights,
            )
            if return_routing_weights:
                draft_hidden, rw = draft_out
                all_routing_weights.append(rw)
            else:
                draft_hidden = draft_out

            draft_logits = target.lm_head(draft_hidden[:, 1 - block_size :, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if block_size > 1:
            target_hidden = extract_context_feature_separated(
                output.hidden_states, model.target_layer_ids
            )
            target_hidden = [h[:, : acceptance_length + 1, :] for h in target_hidden]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids_t
        ).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start

    result = SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
    )
    if return_routing_weights:
        result.routing_weights = all_routing_weights
    return result
