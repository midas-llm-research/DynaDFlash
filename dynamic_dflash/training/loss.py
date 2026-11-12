from __future__ import annotations
from typing import Any
import torch
from dynamic_dflash.core.blocks import BLOCK_SIZE, GAMMA, anchor_positions, build_blocks, weighted_block_loss
from dynamic_dflash.core.model import extract_decoder_states
from dynamic_dflash.training.common import _autocast

def one_sequence_loss(target: Any, model: Any, record: dict, device: torch.device, *, anchor_limit: int, anchor_chunk: int, accumulation: int) -> float:
    ids = torch.tensor(record['input_ids'], device=device, dtype=torch.long)
    anchors = anchor_positions(int(record['prompt_length']), ids.numel(), record['prompt_hash'], 0, max_anchors=anchor_limit)
    if not anchors:
        raise ValueError('prepared record has no valid anchor')
    with torch.no_grad():
        output = target.model(input_ids=ids[None], output_hidden_states=True, use_cache=False, return_dict=True)
        states = extract_decoder_states(output.hidden_states, model.projector.config)
        chunks = [anchors[i:i + anchor_chunk] for i in range(0, len(anchors), anchor_chunk)]
        batches = [build_blocks(ids, chunk, model.draft.mask_token_id, dtype=torch.bfloat16) for chunk in chunks]
        denominator = sum((float(batch.loss_denominator) for batch in batches))
    if denominator <= 0:
        raise RuntimeError('no valid masked-block labels')
    total_numerator = 0.0
    for batch in batches:
        with torch.no_grad():
            noise = target.model.embed_tokens(batch.token_ids)
        with _autocast(device):
            context = model.projector(states)
            hidden = model.forward_fused(context, noise, batch.position_ids, batch.attention_mask)
            logits = target.lm_head(hidden)
            numerator, _ = weighted_block_loss(logits, batch)
            loss = numerator / (denominator * accumulation)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite weighted block loss')
        loss.backward()
        total_numerator += float(numerator.detach())
    return total_numerator / denominator
PAPER_BLOCK_SIZE = BLOCK_SIZE
PAPER_GAMMA = GAMMA
