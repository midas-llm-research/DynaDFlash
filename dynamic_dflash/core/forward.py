from __future__ import annotations
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

def forward_fused(self, fused_context: Tensor, noise_embedding: Tensor, position_ids: Tensor, attention_mask: Tensor | None=None) -> Tensor:
    hidden = noise_embedding
    position_embeddings = self.draft.rotary_emb(hidden, position_ids)
    for layer in self.draft.layers:

        def run(block_hidden: Tensor, context: Tensor, block=layer) -> Tensor:
            return block(hidden_states=block_hidden, target_hidden=context, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, position_embeddings=position_embeddings)
        if self.training and self.activation_checkpointing:
            hidden = checkpoint(run, hidden, fused_context, use_reentrant=False)
        else:
            hidden = run(hidden, fused_context)
    return self.draft.norm(hidden)
