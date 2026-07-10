"""
DFlash draft model with causal attention within blocks.

Position k in a block can only attend to positions 0..k (not k+1..15).
This prevents unreliable late positions from contaminating reliable early
positions. Since acceptance is cumprod (first error kills everything),
protecting early position accuracy directly improves acceptance length.

Zero extra parameters. Inherits everything from DFlashDraftModel.
Only difference: the attention mask enforces causal ordering within blocks.
"""

from typing import Optional

import torch
from transformers.cache_utils import Cache

from midas.models.model import DFlashDraftModel


class DFlashDraftModelCausalBlock(DFlashDraftModel):
    """DFlash with causal attention within blocks.

    At training: expects the attention_mask from prepare_draft_inputs
    to already enforce causal-within-block (via causal_block=True flag).

    At inference: constructs the causal mask internally when use_cache=True.
    """

    causal_block = True

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        if use_cache and attention_mask is None:
            B, N, H = noise_embedding.shape
            target_hidden_fused = self._apply_target_hidden_fusion(target_hidden)
            ctx_len = target_hidden_fused.shape[1]
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
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    target_hidden=target_hidden_fused,
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
