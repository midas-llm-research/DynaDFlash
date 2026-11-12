from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass
import torch
from torch import Tensor
BLOCK_SIZE = 16
GAMMA = 7.0
MAX_ANCHORS = 512

def anchor_positions(prompt_length: int, total_length: int, record_hash: str, epoch: int, *, max_anchors: int=MAX_ANCHORS) -> list[int]:
    if not 0 <= prompt_length < total_length:
        raise ValueError('invalid prompt and sequence lengths')
    candidates = range(prompt_length, total_length - 1)
    seed = int.from_bytes(hashlib.sha256(f'router-anchor-v1\x00{record_hash}\x00{epoch}'.encode()).digest()[:8], 'big')
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, min(max_anchors, len(candidates))))

@dataclass
class BlockBatch:
    token_ids: Tensor
    position_ids: Tensor
    attention_mask: Tensor
    target_ids: Tensor
    valid: Tensor
    position_weights: Tensor

    @property
    def loss_denominator(self) -> Tensor:
        return (self.valid.float() * self.position_weights.float()).sum()

def build_blocks(input_ids: Tensor, anchors: list[int], mask_token_id: int, *, dtype: torch.dtype=torch.bfloat16) -> BlockBatch:
    if input_ids.ndim != 1 or not anchors:
        raise ValueError('expected a 1D sequence and at least one anchor')
    device = input_ids.device
    length = input_ids.numel()
    anchor_tensor = torch.tensor(anchors, device=device, dtype=torch.long)
    if torch.any(anchor_tensor < 0) or torch.any(anchor_tensor >= length - 1):
        raise ValueError('anchors must each have at least one future label')
    count = len(anchors)
    offsets = torch.arange(BLOCK_SIZE, device=device)
    synthetic_positions = anchor_tensor[:, None] + offsets[None, :]
    synthetic_ids = torch.full((count, BLOCK_SIZE), mask_token_id, device=device, dtype=input_ids.dtype)
    synthetic_ids[:, 0] = input_ids[anchor_tensor]
    position_ids = torch.cat([torch.arange(length, device=device), synthetic_positions.flatten()])[None]
    query_anchor = anchor_tensor.repeat_interleave(BLOCK_SIZE)
    base_positions = torch.arange(length, device=device)
    base_allowed = base_positions[None, :] < query_anchor[:, None]
    block_ids = torch.arange(count, device=device).repeat_interleave(BLOCK_SIZE)
    noise_allowed = block_ids[:, None] == block_ids[None, :]
    allowed = torch.cat([base_allowed, noise_allowed], dim=1)
    minimum = torch.finfo(dtype).min
    attention_mask = torch.zeros((1, 1, count * BLOCK_SIZE, length + count * BLOCK_SIZE), dtype=dtype, device=device).masked_fill_(~allowed[None, None], minimum)
    target_positions = anchor_tensor[:, None] + offsets[None, 1:]
    valid = target_positions < length
    target_ids = input_ids[target_positions.clamp(max=length - 1)]
    weights = torch.exp(-torch.arange(BLOCK_SIZE - 1, device=device).float() / GAMMA)
    return BlockBatch(token_ids=synthetic_ids.flatten()[None], position_ids=position_ids, attention_mask=attention_mask, target_ids=target_ids, valid=valid, position_weights=weights[None].expand(count, -1))

def weighted_block_loss(logits: Tensor, batch: BlockBatch) -> tuple[Tensor, Tensor]:
    import torch.nn.functional as F
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] % BLOCK_SIZE:
        raise ValueError('logits must cover complete 16-position blocks')
    count = logits.shape[1] // BLOCK_SIZE
    predicted = logits.reshape(count, BLOCK_SIZE, -1)[:, 1:, :]
    if predicted.shape[:2] != batch.target_ids.shape:
        raise ValueError('logit and target block shapes differ')
    loss = F.cross_entropy(predicted.float().reshape(-1, predicted.shape[-1]), batch.target_ids.reshape(-1), reduction='none').reshape_as(batch.target_ids)
    weights = batch.position_weights.float() * batch.valid.float()
    return ((loss * weights).sum(), weights.sum())
