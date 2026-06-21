"""
Data pipeline for DFlash router training.

Loads pre-existing Qwen3-generated training data from HuggingFace,
then provides online hidden state extraction from the frozen target model
and block-diffusion masking with random anchor sampling.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DFlashDataset(Dataset):
    """
    Dataset of tokenized sequences for DFlash training.

    Uses Qwen3-generated text from public datasets (AM-Qwen3-Distilled)
    to match the target model's output distribution.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 3072,
        num_samples: int = 100_000,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sequences = []
        self._load_data(num_samples)

    def _load_data(self, num_samples: int):
        from datasets import load_dataset

        # Nemotron Post-Training Dataset V2: same source as DFlash paper
        # Splits: math, code, stem, chat — diverse model-generated responses
        # We use the full conversations (prompt + response) as training text.
        # The target model (Qwen3-4B) provides hidden states + logits ONLINE.
        splits = ["math", "code", "stem", "chat"]
        per_split = num_samples // len(splits)

        for split_name in splits:
            try:
                ds = load_dataset(
                    "nvidia/Nemotron-Post-Training-Dataset-v2",
                    split=split_name,
                    streaming=True,
                )
                count = 0
                for sample in ds:
                    if count >= per_split:
                        break

                    # Nemotron format: messages list with role/content
                    messages = sample.get("messages", [])
                    parts = []
                    for msg in messages:
                        content = msg.get("content", "")
                        if content:
                            parts.append(content)
                    text = "\n".join(parts)

                    if not text or len(text) < 50:
                        continue

                    tokens = self.tokenizer(
                        text,
                        max_length=self.max_seq_len,
                        truncation=True,
                        return_tensors="pt",
                    )
                    if tokens["input_ids"].shape[1] > 64:
                        self.sequences.append(tokens["input_ids"].squeeze(0))
                        count += 1

                print(f"Loaded {count} sequences from Nemotron/{split_name}")
            except Exception as e:
                print(f"Warning: Could not load Nemotron/{split_name}: {e}")

        print(f"Total: {len(self.sequences)} training sequences")

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
    Runs frozen target model to extract hidden states from selected layers.
    Used during training to provide context features for the draft model.
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_layer_ids: list[int],
        device: torch.device,
    ):
        self.target_model = target_model
        self.target_layer_ids = target_layer_ids
        self.device = device
        self.target_model.eval()

    @torch.no_grad()
    def extract(
        self, input_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Run target model, return per-layer hidden states and logits.

        Args:
            input_ids: [B, T]

        Returns:
            layer_hiddens: list of N tensors, each [B, T, H]
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

        # Extract selected layer hidden states (offset by 1 as in z-lab's code)
        offset = 1
        layer_hiddens = [
            output.hidden_states[layer_id + offset]
            for layer_id in self.target_layer_ids
        ]

        return layer_hiddens, output.logits


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

    for b in range(B):
        # Sample anchor positions (avoid last block_size positions)
        valid_range = max(1, T - block_size)
        n_anchors = min(num_anchors, valid_range)
        anchors = sorted(random.sample(range(valid_range), n_anchors))

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
