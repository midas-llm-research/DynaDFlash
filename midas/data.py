"""
Data pipeline for DFlash training.

Loads pre-generated self-distillation data (sdd/generate.py output: a JSONL of
(prompt, response) records where each response was sampled from the frozen
target model). Also provides online target hidden state extraction and
block-diffusion masking with random anchor sampling.
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    """Pad sequences to same length in batch."""
    max_len = max(seq.shape[0] for seq in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : seq.shape[0]] = seq
    return padded


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
        Run target model, return full hidden-states tuple and logits.

        Args:
            input_ids: [B, T]

        Returns:
            hidden_states: tuple of (num_layers + 1) tensors, each [B, T, H]
                (index 0 = embedding output, then one per layer)
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


def create_masked_blocks(
    input_ids: torch.Tensor,
    block_size: int,
    num_anchors: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create block-diffusion training samples by randomly masking blocks.

    For each sequence, sample anchor positions. Each anchor is the first token
    of a block; remaining positions in the block are masked for prediction.

    Args:
        input_ids: [B, T] original token ids
        block_size: number of tokens per block
        num_anchors: number of anchor positions to sample per sequence
        mask_token_id: token id used for masking

    Returns:
        masked_ids: [B, T] with blocks masked
        loss_mask: [B, T] binary mask indicating positions to compute loss on
        position_in_block: [B, T] position within block (1-indexed) for loss weighting
        block_ids: [B, T] block assignment per position (0-indexed, -1 = not in any block)
        block_anchors: [B, T] anchor position for each token's block (-1 if not in block)
    """
    B, T = input_ids.shape
    masked_ids = input_ids.clone()
    loss_mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
    position_in_block = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
    block_ids = torch.full((B, T), -1, dtype=torch.long, device=input_ids.device)
    block_anchors = torch.full((B, T), -1, dtype=torch.long, device=input_ids.device)

    # Minimum anchor position: at inference, blocks start after the prompt.
    # Setting min=64 ensures all blocks have meaningful context to attend to.
    min_anchor = min(64, T // 4)

    for b in range(B):
        # Sample anchor positions (between min_anchor and T-block_size)
        valid_start = min_anchor
        valid_end = max(valid_start + 1, T - block_size)
        valid_range = valid_end - valid_start
        n_anchors = min(num_anchors, valid_range)
        anchors = sorted(random.sample(range(valid_start, valid_end), n_anchors))

        for block_idx, anchor in enumerate(anchors):
            end = min(anchor + block_size, T)
            block_ids[b, anchor:end] = block_idx
            block_anchors[b, anchor:end] = anchor
            masked_ids[b, anchor + 1 : end] = mask_token_id
            loss_mask[b, anchor + 1 : end] = True
            for k in range(1, end - anchor):
                position_in_block[b, anchor + k] = k

    return masked_ids, loss_mask, position_in_block, block_ids, block_anchors


def build_block_attention_mask(
    block_ids: torch.Tensor,
    block_anchors: torch.Tensor,
    ctx_len: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Build sparse attention mask enforcing causal consistency and inter-block
    isolation (DFlash paper Section 4.2, Figure 4).

    Rules:
    - Queries in block b (anchor at position a) can attend to context positions < a ONLY
      (causal consistency: no future context, matching inference behavior)
    - Queries in a block can attend to noise positions in the SAME block only
    - Queries not in any block cannot attend to any noise position

    Args:
        block_ids: [B, T] block assignment (-1 = not in block)
        block_anchors: [B, T] anchor position of the block each position belongs to (-1 if not in block)
        ctx_len: number of context positions (prepended to keys)
        dtype: mask dtype

    Returns:
        mask: [B, 1, T, ctx_len + T] additive attention mask (0 = attend, -inf = block)
    """
    B, T = block_ids.shape
    device = block_ids.device

    # --- Noise-to-noise mask: [B, T, T] ---
    q_ids = block_ids.unsqueeze(2)  # [B, T, 1]
    k_ids = block_ids.unsqueeze(1)  # [B, 1, T]
    same_block = (q_ids == k_ids) & (q_ids != -1)
    noise_mask = torch.where(same_block, 0.0, float("-inf"))  # [B, T, T]

    # --- Context mask: [B, T, ctx_len] ---
    # Each query position can only attend to context at positions BEFORE its block's anchor.
    # This enforces causal consistency: at inference, context only covers verified prefix.
    ctx_positions = torch.arange(ctx_len, device=device).view(1, 1, ctx_len)  # [1, 1, ctx_len]
    anchor_positions = block_anchors.unsqueeze(2)  # [B, T, 1]

    # Query can attend to context[j] if j < anchor_position of its block
    # For non-block positions (anchor=-1), block everything (they don't contribute to loss)
    ctx_visible = ctx_positions < anchor_positions  # [B, T, ctx_len]
    ctx_mask = torch.where(ctx_visible, 0.0, float("-inf"))  # [B, T, ctx_len]

    # Full mask: [B, 1, T, ctx_len + T]
    full_mask = torch.cat([ctx_mask.to(dtype), noise_mask.to(dtype)], dim=2)

    return full_mask.unsqueeze(1)  # [B, 1, T, ctx_len + T]
