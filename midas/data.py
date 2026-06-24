"""
Data pipeline for DFlash training.

- `DFlashDataset` loads pre-generated self-distillation JSONL (sdd/generate.py output).
- `DFlashDataProcessor` wraps the dataset in a DataLoader; train.py only needs
  to instantiate this.
- `OnlineHiddenStateExtractor` runs the frozen target and returns its full
  per-layer hidden states + logits.
- `prepare_draft_inputs` samples anchors and builds the concatenated-blocks
  noise sequence + attention mask used to train the draft.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer

from midas.config import TrainConfig


# ---------------------------------------------------------------------------
# Dataset + dataloader
# ---------------------------------------------------------------------------


class DFlashDataset(Dataset):
    """
    Tokenized self-distilled sequences for DFlash training.

    Loads records from a JSONL file produced by `sdd/generate.py`, applies the
    target's chat template to each (prompt, response) pair, and tokenizes the
    full conversation. Sequences match the target model's own output
    distribution because the responses were sampled from it.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        max_seq_len: int = 3072,
        max_samples: Optional[int] = None,
        min_seq_len: int = 64,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sequences: list[torch.Tensor] = []
        self._load(Path(data_path), max_samples, min_seq_len, enable_thinking)

    def _load(
        self,
        data_path: Path,
        max_samples: Optional[int],
        min_seq_len: int,
        enable_thinking: bool,
    ) -> None:
        if not data_path.exists():
            raise FileNotFoundError(
                f"SDD data not found at {data_path}. "
                "Run sdd/generate.py (or scripts/modal_generate.py) first."
            )

        kept = 0
        skipped_short = 0
        skipped_long = 0
        with open(data_path) as f:
            for line in f:
                if max_samples is not None and kept >= max_samples:
                    break
                record = json.loads(line)
                messages = [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["response"]},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    enable_thinking=enable_thinking,
                )
                tokens = self.tokenizer(text, return_tensors="pt")
                ids = tokens["input_ids"].squeeze(0)
                if ids.shape[0] > self.max_seq_len:
                    skipped_long += 1
                    continue
                if ids.shape[0] < min_seq_len:
                    skipped_short += 1
                    continue
                self.sequences.append(ids)
                kept += 1

        print(
            f"Loaded {kept} sequences from {data_path} "
            f"(skipped {skipped_short} below {min_seq_len} tokens, "
            f"{skipped_long} above {self.max_seq_len} tokens)"
        )
        if self.sequences:
            lens = torch.tensor(
                [int(seq.shape[0]) for seq in self.sequences], dtype=torch.float
            )
            qs = torch.quantile(
                lens, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            ).long().tolist()
            print(
                f"  Token length distribution: "
                f"min={qs[0]}  q25={qs[1]}  median={qs[2]}  q75={qs[3]}  max={qs[4]}"
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(
    batch: list[torch.Tensor], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Right-pad variable-length sequences to the batch's max length and also
    return the per-row real lengths so downstream block construction and
    attention masking can ignore the padding.
    """
    lengths = torch.tensor([seq.shape[0] for seq in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : seq.shape[0]] = seq
    return padded, lengths


class LengthInterleavedBatchSampler(Sampler[list[int]]):
    """
    Length-bucketed batching with interleaved batch order.

    Construction:
      1. Sort sample indices by sequence length (descending).
      2. Chunk into batches of size `batch_size`.
      3. Yield in interleaved order [0, n-1, 1, n-2, 2, n-3, ...] so the first
         few batches mix long-and-short, keeping memory peak roughly steady
         across the epoch instead of front-loading the largest batches.

    Why: within each batch, sequences have similar lengths → padding waste in
    `collate_fn` is minimized. Without bucketing, a single long sequence forces
    every other sequence in the batch to pad to its length.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        # 1. sort indices by descending length
        sorted_indices = sorted(
            range(len(self.lengths)), key=lambda i: -self.lengths[i]
        )
        # 2. chunk into contiguous batches
        batches: list[list[int]] = []
        for i in range(0, len(sorted_indices), self.batch_size):
            chunk = sorted_indices[i : i + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                continue
            batches.append(chunk)
        # 3. interleave order: [0, n-1, 1, n-2, ...]
        n = len(batches)
        for i in range((n + 1) // 2):
            yield batches[i]
            if i != n - 1 - i:
                yield batches[n - 1 - i]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class DFlashDataProcessor:
    """
    One-stop data class: builds `DFlashDataset` from the config and wraps it in
    a DataLoader with the length-bucketed interleaved batch sampler. Train loop
    iterates over `self` directly.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: TrainConfig,
        num_workers: int = 4,
        drop_last: bool = False,
    ):
        self.tokenizer = tokenizer
        self.dataset = DFlashDataset(
            tokenizer,
            data_path=config.data_path,
            max_seq_len=config.max_seq_len,
        )
        pad_id = tokenizer.pad_token_id or 0
        sampler = LengthInterleavedBatchSampler(
            lengths=[int(seq.shape[0]) for seq in self.dataset.sequences],
            batch_size=config.batch_size,
            drop_last=drop_last,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=sampler,
            collate_fn=lambda batch: collate_fn(batch, pad_id),
            num_workers=num_workers,
            pin_memory=True,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


# ---------------------------------------------------------------------------
# Online target-model hidden state extraction
# ---------------------------------------------------------------------------


class OnlineHiddenStateExtractor:
    """
    Runs frozen target model and returns the full per-layer hidden states tuple
    plus logits. Layer selection is delegated to the draft model so subclasses
    can pick any subset they want from the full stack.
    """

    def __init__(
        self,
        target_model: nn.Module,
        device: torch.device,
    ):
        self.target_model = target_model
        self.device = device
        self.target_model.eval()

    @torch.no_grad()
    def extract(
        self, input_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Args:
            input_ids: [B, T]
        Returns:
            hidden_states: tuple of (num_layers + 1) tensors, each [B, T, H]
            logits: [B, T, V]
        """
        input_ids = input_ids.to(self.device)
        position_ids = (
            torch.arange(input_ids.shape[1], device=self.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )
        output = self.target_model(
            input_ids,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        return output.hidden_states, output.logits


# ---------------------------------------------------------------------------
# Block-diffusion training input builder
# ---------------------------------------------------------------------------


@dataclass
class DraftTrainingBatch:
    """All tensors needed for one training step beyond `target_hidden` itself.

    Shapes (with B = batch, T = original sequence length,
    N = num_anchors * block_size):

    - `noise_token_ids` [B, N]: per-anchor blocks concatenated. Each block is
      [t_anchor, MASK, MASK, ..., MASK] of length `block_size`.
    - `position_ids` [B, T + N]: rotary positions for the context+noise key
      stream. First T entries are [0..T-1] (target's actual positions); last N
      entries are the original sequence positions of each noise token (so each
      block's tokens get RoPE for [a, a+1, ..., a+block_size-1]).
    - `attention_mask` [B, 1, N, T + N]: additive mask. Each noise query can
      attend to (a) target hidden states at original positions < its block's
      anchor, and (b) other noise tokens in the same block only.
    - `noise_original_positions` [B, N]: the original position each noise token
      corresponds to; used at loss time to gather target logits.
    - `loss_mask` [B, N]: True at the masked positions where loss is computed.
    - `position_in_block` [B, N]: 0 for the anchor, 1..block_size-1 for masks.
    """
    noise_token_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    noise_original_positions: torch.Tensor
    loss_mask: torch.Tensor
    position_in_block: torch.Tensor


def prepare_draft_inputs(
    input_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    block_size: int,
    num_anchors: int,
    mask_token_id: int,
    min_anchor: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> DraftTrainingBatch:
    """
    Sample anchors and build the concatenated-blocks noise sequence used for
    block-diffusion training of the draft.

    For each sequence in the batch, sample `num_anchors` anchor positions in
    [min_anchor, real_len - block_size]. For each anchor `a` we build a block

        block = [t_a, MASK, MASK, ..., MASK]    (length `block_size`)

    where the anchor token comes from the original sequence and the rest are
    mask tokens. All `num_anchors` blocks are concatenated into a single
    `[B, N]` noise sequence where `N = num_anchors * block_size`.

    Each block is independent at training time:
    - It conditions only on target hidden states at original positions < `a`.
    - Its tokens only attend to other tokens in the same block (block-diagonal
      noise-to-noise mask).

    Overlap in original sequence positions between blocks is harmless because
    blocks live at different indices in the noise sequence and the attention
    mask isolates them.

    Padding handling: anchors per row are bounded by that row's real length
    (`seq_lengths[b]`), so noise tokens are never pad tokens. Context positions
    in the padding region (>= real_len) are masked out of every block's
    cross-attention as a side-effect — the anchor < real_len bound makes
    `ctx_positions < anchor` automatically False for any pad position.
    """
    B, T = input_ids.shape
    device = input_ids.device
    seq_lengths = seq_lengths.to(device=device, dtype=torch.long)

    # Sample anchors per row, sorted, bounded by each row's real length.
    anchors_rows = []
    for b in range(B):
        real_len = int(seq_lengths[b].item())
        valid_high = real_len - block_size + 1
        if valid_high <= min_anchor:
            raise ValueError(
                f"Row {b} too short: real_len={real_len}, block_size={block_size}, "
                f"min_anchor={min_anchor}. Need real_len >= min_anchor + block_size."
            )
        if valid_high - min_anchor < num_anchors:
            raise ValueError(
                f"Row {b}: cannot sample {num_anchors} distinct anchors from "
                f"[{min_anchor}, {valid_high}) — only {valid_high - min_anchor} "
                "positions available. Lower num_anchors or filter shorter sequences."
            )
        anchors_rows.append(
            sorted(random.sample(range(min_anchor, valid_high), num_anchors))
        )
    anchors = torch.tensor(anchors_rows, device=device, dtype=torch.long)

    N = num_anchors * block_size
    # block_offsets[k] = k. position_in_block[b, i*block_size + k] = k.
    block_offsets = torch.arange(block_size, device=device, dtype=torch.long)

    # Original sequence position for each noise token.
    # noise_original_positions[b, i*block_size + k] = anchors[b, i] + k.
    noise_original_positions = (
        anchors.unsqueeze(2) + block_offsets.view(1, 1, -1)
    ).reshape(B, N)  # ends up being B x original position of every element in each block

    # Per-noise-position metadata.
    # B x [0,1,...,num_anchors-1,0,1,...,num_anchors-1,...,num_anchors-2,num_anchors-1]
    position_in_block = block_offsets.repeat(num_anchors).unsqueeze(0).expand(B, -1)

    # B x [0,...,0,1,..,1,...,num_anchors-1,...,num_anchors-1] (each one repeated "block_size" number of times)
    noise_block_idx = (
        torch.arange(num_anchors, device=device, dtype=torch.long)
        .repeat_interleave(block_size)
        .unsqueeze(0)
        .expand(B, -1)
    )  

    # Anchor position for the block each noise token belongs to.
    # B x [a_0,...,a_0,a_1,...,a_1,...,a_n,...,a_n]
    noise_block_anchor = torch.gather(anchors, 1, noise_block_idx)

    # Build noise_token_ids: anchor positions take the original token, the
    # block_size-1 trailing positions get the mask token.
    noise_token_ids = torch.gather(input_ids, 1, noise_original_positions)
    is_mask = position_in_block > 0
    noise_token_ids = torch.where(
        is_mask,
        torch.tensor(mask_token_id, device=device, dtype=noise_token_ids.dtype),
        noise_token_ids,
    )

    # Attention mask: [B, 1, N, T + N]
    # ctx half [B, N, T]: noise q can attend to context k iff k < q's anchor.
    ctx_positions = torch.arange(T, device=device).view(1, 1, T)
    ctx_visible = ctx_positions < noise_block_anchor.unsqueeze(2)
    ctx_mask = torch.where(
        ctx_visible,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float("-inf"), device=device, dtype=dtype),
    )

    # noise half [B, N, N]: block-diagonal — same-block visibility only.
    same_block = noise_block_idx.unsqueeze(2) == noise_block_idx.unsqueeze(1)
    noise_mask = torch.where(
        same_block,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float("-inf"), device=device, dtype=dtype),
    )

    attention_mask = torch.cat([ctx_mask, noise_mask], dim=2).unsqueeze(1)

    # Rotary position_ids for cat([k_ctx, k_noise]) of total length T + N.
    # First T entries: target's actual positions [0..T-1].
    # Last N entries: each noise token's original sequence position (so each
    # block's RoPE encodes positions [a, a+1, ..., a+block_size-1]).
    ctx_pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    position_ids = torch.cat([ctx_pos_ids, noise_original_positions], dim=1)

    return DraftTrainingBatch(
        # [B, N] of actual token ids (whether actual token or mask token)
        noise_token_ids=noise_token_ids,  

        # [B, T+N] position indices:
        # - if token is in original sequence T, then it's just its original index
        # - if token is an anchor token in N suffix, then it's its corresponding original index
        # - if token is a mask token in N suffix, then it's the index of the corresponding token in original sequence that it SHOULD be after prediction
        position_ids=position_ids,  

        # [B, 1, N, T + N]
        # for each anchor/mask token in N suffix, masks attention correctly to 
        # - tokens in the same block (which all must obv be in N suffix too)
        # - tokens in original sequence (this masking is the masking used on the target hidden states in the T prefix)
        attention_mask=attention_mask,

        # [B, N]
        # the suffix of position_ids
        noise_original_positions=noise_original_positions,

        # [B, N]
        # - 0 if anchor token
        # - 1 if mask token
        loss_mask=is_mask,

        # [B, N]
        # the index in [0, N-1] which is the index within its block for each anchor/mask token in each block
        position_in_block=position_in_block,
    )
