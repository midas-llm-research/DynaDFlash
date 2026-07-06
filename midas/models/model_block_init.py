"""
DFlash draft model with anchor-conditioned block initialization.

Replaces the uninformative embed_tokens(MASK_ID) starting point for masked
positions with a signal derived from the target model's fused hidden state at
the anchor position (the generation boundary). Adds a learned position-in-block
embedding so each mask has a unique identity from layer 0.

Inherits everything from DFlashDraftModel and only overrides forward() to
insert the initialization between fusion and the layer loop.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from midas.models.model import DFlashDraftModel


class AnchorBlockInitializer(nn.Module):
    """
    Produces an additive initialization signal for masked block positions:
        init_k = anchor_proj(anchor_hidden) + block_pos_embed(k)

    Applied only at mask positions (position_in_block > 0); anchors keep
    their real token embedding unchanged.

    Both projections are zero-initialized so warm-starting from a DFlash
    checkpoint is lossless (output is identically zero at init).
    """

    def __init__(self, hidden_size: int, block_size: int = 16):
        super().__init__()
        self.block_size = block_size
        self.anchor_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.block_pos_embed = nn.Embedding(block_size, hidden_size)
        nn.init.zeros_(self.anchor_proj.weight)
        nn.init.zeros_(self.block_pos_embed.weight)

    def forward(
        self,
        noise_embedding: torch.Tensor,
        anchor_hidden: torch.Tensor,
        position_in_block: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noise_embedding: [B, N, H] — embed_tokens output for the noise sequence
            anchor_hidden: [B, A, H] — fused target hidden at each block's anchor
            position_in_block: [B, N] — 0 for anchors, 1..block_size-1 for masks

        Returns:
            [B, N, H] — noise_embedding with initialization added at mask positions
        """
        B, N, H = noise_embedding.shape
        assert N % self.block_size == 0, (
            f"Noise sequence length {N} must be divisible by block_size {self.block_size}"
        )
        num_anchors = N // self.block_size

        content = self.anchor_proj(anchor_hidden)  # [B, A, H]
        content = (
            content.unsqueeze(2)
            .expand(B, num_anchors, self.block_size, H)
            .reshape(B, N, H)
        )

        pos = self.block_pos_embed(position_in_block)  # [B, N, H]

        is_mask = (position_in_block > 0).unsqueeze(-1).to(noise_embedding.dtype)
        return noise_embedding + (content + pos) * is_mask


class DFlashDraftModelBlockInit(DFlashDraftModel):
    """
    DFlash draft model with anchor-conditioned initialization of mask positions.

    At each block, the fused target hidden at the anchor position is projected
    and added to mask embeddings before the transformer layers run. This gives
    the model an informative starting point (the target's prediction state at
    the generation boundary) rather than identical MASK embeddings.

    Inherits all other behavior from DFlashDraftModel: same fusion, same
    attention, same generation, same loss, same checkpoint loading.
    """

    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.block_initializer = AnchorBlockInitializer(
            config.hidden_size, config.block_size
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
    ):
        hidden_states = noise_embedding
        target_hidden = self._apply_target_hidden_fusion(target_hidden)

        B, N, H = hidden_states.shape
        noise_block_anchor = kwargs.get("noise_block_anchor")
        position_in_block = kwargs.get("position_in_block")

        if noise_block_anchor is not None:
            anchor_indices = noise_block_anchor[:, :: self.block_size]  # [B, A]
        else:
            num_anchors = N // self.block_size
            anchor_indices = torch.full(
                (B, num_anchors),
                target_hidden.shape[1] - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )

        anchor_h = target_hidden.gather(
            1, anchor_indices.unsqueeze(-1).expand(-1, -1, H)
        )

        if position_in_block is None:
            num_anchors = N // self.block_size
            position_in_block = (
                torch.arange(self.block_size, device=hidden_states.device)
                .repeat(num_anchors)
                .unsqueeze(0)
                .expand(B, -1)
            )

        hidden_states = self.block_initializer(hidden_states, anchor_h, position_in_block)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)
