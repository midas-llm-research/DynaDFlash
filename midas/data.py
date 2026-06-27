"""
Data pipeline for DFlash training.

- `DFlashDataset` loads pre-generated self-distillation JSONL (sdd/generate_sdd.py output).
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

    Loads records from a JSONL file produced by `sdd/generate_sdd.py`, applies the
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
        self.response_starts: list[int] = []
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
                "Run sdd/generate_sdd.py (or scripts/modal_generate_sdd.py) first."
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
                # Find where response starts by tokenizing just the prompt.
                prompt_messages = [{"role": "user", "content": record["prompt"]}]
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=enable_thinking,
                )
                prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze(0)
                prompt_len = prompt_ids.shape[0]
                assert torch.equal(ids[:prompt_len], prompt_ids), \
                    "Prompt tokenization mismatch — response_start boundary is unreliable"
                self.sequences.append(ids)
                self.response_starts.append(prompt_len)
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
        return self.sequences[idx], self.response_starts[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Right-pad variable-length sequences to the batch's max length.
    Returns (padded_ids, lengths, response_starts).
    """
    sequences = [item[0] for item in batch]
    response_starts = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = seq
    return padded, lengths, response_starts


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

    Filters out sequences too short to fit even one anchor (`real_len <
    min_anchor_pos + block_size`) — `prepare_draft_inputs` cannot do anything
    useful with them. The actual number of anchors used for each step is
    determined *per batch* by `prepare_draft_inputs` itself (see that
    function's docstring), so this class only handles loading + batching.
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

        # Filter sequences that can't fit even one anchor in the response.
        keep_idx = [
            i
            for i, (seq, rs) in enumerate(zip(self.dataset.sequences, self.dataset.response_starts))
            if int(seq.shape[0]) - config.block_size + 1 - rs > 0
        ]
        n_dropped = len(self.dataset.sequences) - len(keep_idx)
        if n_dropped:
            self.dataset.sequences = [self.dataset.sequences[i] for i in keep_idx]
            self.dataset.response_starts = [self.dataset.response_starts[i] for i in keep_idx]
            print(
                f"DataProcessor: dropped {n_dropped} sequences with 0 valid "
                f"anchors (response too short for block_size={config.block_size})"
            )

        if not self.dataset.sequences:
            raise ValueError(
                f"No sequence in the dataset can fit a single anchor with "
                f"block_size={config.block_size}. "
                f"Filter to longer sequences or use shorter block_size."
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
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            input_ids: [B, T]
        Returns:
            hidden_states: tuple of (num_layers + 1) tensors, each [B, T, H]
        """
        input_ids = input_ids.to(self.device)
        position_ids = (
            torch.arange(input_ids.shape[1], device=self.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )
        output = self.target_model.model(
            input_ids,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        return output.hidden_states


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
    max_num_anchors: int,
    mask_token_id: int,
    response_starts: Optional[torch.Tensor] = None,
    min_anchor_pos: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> DraftTrainingBatch:
    """
    Sample anchors from the response portion and build concatenated-blocks
    noise sequence for block-diffusion training.

    Anchors are sampled from [response_start, real_len - block_size + 1) per
    row. If response_starts is None, falls back to min_anchor_pos.
    """
    B, T = input_ids.shape
    device = input_ids.device
    seq_lengths = seq_lengths.to(device=device, dtype=torch.long)

    # Per-row anchor start position.
    if response_starts is not None:
        response_starts = response_starts.to(device=device, dtype=torch.long)
        anchor_starts = [int(response_starts[b].item()) for b in range(B)]
    else:
        anchor_starts = [min_anchor_pos] * B

    # Per-row real anchor capacity = max anchors that fit in this row.
    row_max = [
        max(0, int(seq_lengths[b].item()) - block_size + 1 - anchor_starts[b])
        for b in range(B)
    ]
    if min(row_max) == 0:
        bad = [b for b, m in enumerate(row_max) if m == 0]
        raise ValueError(
            f"Rows {bad}: real_len too short to fit any anchor "
            f"(anchor_start={[anchor_starts[b] for b in bad]}, block_size={block_size}). "
            f"DFlashDataProcessor should have filtered these — did you bypass it?"
        )

    # Per-batch effective N: the longest row's capacity, capped by the cap.
    num_anchors = min(max_num_anchors, max(row_max))

    # Sample real anchors per row; pad with duplicates of the last real anchor
    # to reach `num_anchors`. Track per-row real anchor counts so the loss can
    # gate out the sentinel slots.
    anchors_rows: list[list[int]] = []
    n_real_per_row: list[int] = []
    for b in range(B):
        real_len = int(seq_lengths[b].item())
        valid_high = real_len - block_size + 1
        n_real = min(row_max[b], num_anchors)
        real_anchors = sorted(random.sample(range(anchor_starts[b], valid_high), n_real))
        # Sentinels duplicate the last real anchor — produces identical compute
        # to the real block, but loss_mask gates them out below.
        sentinels = [real_anchors[-1]] * (num_anchors - n_real)
        anchors_rows.append(real_anchors + sentinels)
        n_real_per_row.append(n_real)
    anchors = torch.tensor(anchors_rows, device=device, dtype=torch.long)
    n_real_tensor = torch.tensor(n_real_per_row, device=device, dtype=torch.long)

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

    # block_active[b, i] = True iff block i in row b is a REAL block (not sentinel).
    block_idx_range = torch.arange(num_anchors, device=device, dtype=torch.long)
    block_active = block_idx_range.unsqueeze(0) < n_real_tensor.unsqueeze(1)  # [B, num_anchors]
    # Expand to per-noise-position: True for every position in a real block.
    block_active_per_noise = block_active.repeat_interleave(block_size, dim=1)  # [B, N]
    # Loss is computed only at masked positions within REAL (non-sentinel) blocks.
    loss_mask = is_mask & block_active_per_noise

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
        # - 1 if mask or sentinal token
        loss_mask=loss_mask,

        # [B, N]
        # the index in [0, N-1] which is the index within its block for each anchor/mask token in each block
        position_in_block=position_in_block,
    )
