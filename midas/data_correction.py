"""
Round-2 batch construction for the correction drafter.

The correction model (`midas/models/model_correction.py`) re-drafts a rejected
block suffix from the residuals that suffix already produced, so a training step
needs two rounds rather than one:

    round 1   frozen DFlash 2 drafter over the standard `prepare_draft_inputs`
              blocks, with `return_prenorm=True`     -> residuals + proposed tokens
    simulate  which slots the target would have rejected, per block  -> a
    round 2   the trainable correction model over the rejected suffix, anchored on
              the corrected token, fed round 1's residuals

Only round 2 carries gradient. No extra target forward is needed: round 2's ctx is
a prefix of the same `target_hidden` tensor round 1 used, so the frozen target runs
once per step exactly as in plain DFlash training.

The narrow variant is what is built here: the round-2 block covers
`s+a+1 … s+15` and never extends past the round-1 block, so every supervised slot
is a recycled one and the supervised width varies with `a`. See
papers/hidden_state_recycling.md for the widened alternative.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from midas.data import DraftTrainingBatch
from midas.models.model_correction import (
    CorrectionBlock,
    build_correction_block,
    positions_to_redraft,
    simulate_acceptance,
)


@dataclass
class CorrectionStep:
    """Everything one round-2 training step needs.

    `batch` plugs into `CorrectionDraftModel.compute_loss` unchanged; `recycled`
    and `is_recycled` are the two extra `forward` arguments.
    """

    batch: DraftTrainingBatch
    recycled_hidden: torch.Tensor
    is_recycled: torch.Tensor
    metrics: dict


@torch.no_grad()
def run_round1(
    parent: nn.Module,
    batch: DraftTrainingBatch,
    target_hidden: torch.Tensor,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    *,
    block_size: int,
    chunk_blocks: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen drafter pass. -> (pre-norm residuals [B, N, H], proposed tokens [B, N]).

    Two things this does that a naive call would get wrong:

    1. `propose` is run PER BLOCK. DFlash 2's selector walks a path left to right,
       chaining each choice into the next position's predecessor score; run over
       the flat `[B, N]` sequence it would chain block i's last token into block
       i+1's anchor, across a boundary the attention mask exists to prevent. The
       reshape to `[B*num_blocks, block_size, H]` mirrors inference, where the walk
       only ever sees one block.
    2. It is chunked over blocks. The walk materializes `lm_head(hidden)` to take
       its top-K, which at V=151936 and the default 128 anchors is ~2.3 GB of
       logits in one shot — the same cost `DFlash2DraftModel.compute_loss` is
       written out to avoid paying twice.
    """
    normed, prenorm = parent(
        position_ids=batch.position_ids,
        attention_mask=batch.attention_mask,
        noise_embedding=embed_tokens(batch.noise_token_ids),
        target_hidden=target_hidden,
        return_prenorm=True,
    )

    B, N, H = normed.shape
    num_blocks = N // block_size
    flat = normed.reshape(B * num_blocks, block_size, H)
    anchor_ids = batch.noise_token_ids.view(B, num_blocks, block_size)[:, :, 0].reshape(-1)

    proposed = batch.noise_token_ids.view(B, num_blocks, block_size).clone().reshape(
        B * num_blocks, block_size
    )
    for start in range(0, flat.shape[0], chunk_blocks):
        sl = slice(start, start + chunk_blocks)
        proposed[sl, 1:] = parent.propose(
            flat[sl, 1:], anchor_ids[sl], lm_head, embed_tokens
        )
    return prenorm, proposed.view(B, N)


def build_round2_attention_mask(
    block: CorrectionBlock,
    context_length: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Additive mask [B, 1, N, T + N] for the round-2 block.

    Differs from `prepare_draft_inputs`' round-1 mask in exactly two ways:

    - **ctx cutoff moves to `s+a+1`** (the round-2 anchor) rather than `s`. That
      opens up the target hidden states for the newly confirmed tokens `s..s+a`,
      which is the whole point of a second round — and it stops there, because at
      inference the verify pass computed the hidden state at `s+a+1` from the
      *rejected* token, so it is invalid (`model.py:229` truncates it away).
    - **pad slots are hidden as keys.** They carry a duplicated position and a mask
      embedding; a recycled slot attending to one would be reading noise. They are
      left free as queries, which costs nothing (their output is masked out of the
      loss) and avoids an all `-inf` softmax row — every block keeps at least its
      anchor visible.
    """
    device = block.position_in_block.device
    B, N = block.position_in_block.shape
    num_blocks = N // block.block_size

    # ctx half [B, N, T]: every slot in a block shares its block's cutoff.
    cutoff = block.anchor_positions.repeat_interleave(block.block_size, dim=1)  # [B, N]
    ctx_visible = torch.arange(context_length, device=device).view(1, 1, -1) < cutoff.unsqueeze(2)

    # noise half [B, N, N]: same block only, and never a pad key.
    block_idx = torch.arange(num_blocks, device=device).repeat_interleave(block.block_size)
    same_block = (block_idx.unsqueeze(1) == block_idx.unsqueeze(0)).unsqueeze(0)
    noise_visible = same_block & ~block.is_pad.unsqueeze(1)

    visible = torch.cat([ctx_visible, noise_visible.expand(B, N, N)], dim=2)
    return torch.where(
        visible,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float("-inf"), device=device, dtype=dtype),
    ).unsqueeze(1)


def prepare_correction_batch(
    parent: nn.Module,
    input_ids: torch.Tensor,
    batch: DraftTrainingBatch,
    target_hidden: torch.Tensor,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    *,
    block_size: int,
    mask_token_id: int,
    min_remaining: int = 7,
    below_trigger_keep_prob: float = 0.1,
    chunk_blocks: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    verifier_tokens: Optional[torch.Tensor] = None,
) -> CorrectionStep:
    """One round-2 training batch from one round-1 batch.

    `verifier_tokens` [B, N] is what the target is taken to have committed at each
    round-1 slot; it defaults to ground truth, which is itself a T=0.7 draw from
    the target (the SDD corpus was sampled at 0.7), so the default simulates a
    T=0.7 verifier for free and keeps the anchor and every label on the one path
    we have labels for. See `simulate_acceptance` for the alternatives.

    Two gates on which blocks contribute loss, both applied through `loss_mask`
    rather than by dropping rows, so shapes stay rectangular:

    - **round-1 sentinels.** `prepare_draft_inputs` pads short rows with duplicate
      anchors and gates them out of its own loss. Those duplicates would look like
      perfectly good round-2 blocks, so their round-1 gate has to be carried
      forward or the same block gets trained on repeatedly.
    - **the deployment trigger.** Blocks with fewer than `min_remaining` slots left
      are not ones the correction model will ever be asked about, so they are
      excluded — except a `below_trigger_keep_prob` fraction, kept so the model
      does not fall over if the real acceptance distribution runs narrower than
      the simulated one.
    """
    B, N = batch.noise_token_ids.shape
    num_blocks = N // block_size
    context_length = input_ids.shape[1]

    if verifier_tokens is None:
        verifier_tokens = input_ids.gather(1, batch.noise_original_positions)

    prenorm, proposed = run_round1(
        parent, batch, target_hidden, embed_tokens, lm_head,
        block_size=block_size, chunk_blocks=chunk_blocks,
    )
    acceptance = simulate_acceptance(proposed, verifier_tokens, block_size)  # [B, num_blocks]

    block = build_correction_block(
        prenorm,
        verifier_tokens,
        acceptance,
        batch.noise_original_positions.view(B, num_blocks, block_size)[:, :, 0],
        block_size=block_size,
        mask_token_id=mask_token_id,
    )

    # Round-1 sentinel blocks contribute no supervised slot of their own.
    round1_active = batch.loss_mask.view(B, num_blocks, block_size)[:, :, 1:].any(dim=-1)
    remaining = positions_to_redraft(acceptance, block_size)
    triggered = remaining >= min_remaining
    keep = triggered
    if below_trigger_keep_prob > 0.0:
        sampled = torch.rand(triggered.shape, device=triggered.device) < below_trigger_keep_prob
        keep = triggered | (sampled & (remaining > 0))
    keep = keep & round1_active

    loss_mask = block.valid_mask & keep.repeat_interleave(block_size, dim=1)

    round2 = DraftTrainingBatch(
        noise_token_ids=block.noise_token_ids,
        position_ids=torch.cat(
            [batch.position_ids[:, :context_length], block.noise_original_positions], dim=1
        ),
        attention_mask=build_round2_attention_mask(block, context_length, dtype=dtype),
        noise_original_positions=block.noise_original_positions,
        loss_mask=loss_mask,
        position_in_block=block.position_in_block,
    )

    supervised = loss_mask.sum().item()
    active = round1_active.sum().clamp(min=1)
    metrics = {
        "round1_acceptance": (acceptance * round1_active).sum().item() / active.item(),
        "trigger_rate": (triggered & round1_active).sum().item() / active.item(),
        "blocks_kept": keep.sum().item() / active.item(),
        "supervised_positions": supervised,
        "supervised_width": supervised / max(1, keep.sum().item()),
    }
    return CorrectionStep(
        batch=round2,
        recycled_hidden=block.recycled_hidden,
        is_recycled=block.is_recycled,
        metrics=metrics,
    )
