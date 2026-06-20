"""
Data pipeline for DFlash router training.

Handles:
1. Loading prompts from Nemotron + CodeAlpaca
2. Online hidden state extraction from frozen target model
3. Block-diffusion masking with random anchor sampling
"""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class TrainingBatch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    target_hidden_states: list[torch.Tensor]
    target_logits: torch.Tensor
    masked_ids: torch.Tensor
    block_positions: torch.Tensor
    loss_mask: torch.Tensor


class DFlashDataset(Dataset):
    """Dataset of tokenized sequences for DFlash training."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 3072,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sequences = []
        self._load_data(split)

    def _load_data(self, split: str):
        from datasets import load_dataset, concatenate_datasets

        # Nemotron post-training data
        nemotron = load_dataset(
            "nvidia/Nemotron-CrossThink-Dataset",
            split=split,
            streaming=False,
        )

        # CodeAlpaca
        code_alpaca = load_dataset(
            "sahil2801/CodeAlpaca-20k",
            split=split,
            streaming=False,
        )

        # Tokenize and store
        for sample in nemotron:
            text = sample.get("text", sample.get("content", ""))
            if not text:
                continue
            tokens = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            if tokens["input_ids"].shape[1] > 64:
                self.sequences.append(tokens["input_ids"].squeeze(0))

        for sample in code_alpaca:
            text = sample.get("output", "")
            if not text:
                continue
            prompt = sample.get("instruction", "")
            full_text = f"{prompt}\n{text}" if prompt else text
            tokens = self.tokenizer(
                full_text,
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            if tokens["input_ids"].shape[1] > 64:
                self.sequences.append(tokens["input_ids"].squeeze(0))

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
        position_ids = torch.arange(
            input_ids.shape[1], device=self.device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)

        output = self.target_model(
            input_ids,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        # Extract selected layer hidden states (offset by 1 as in z-lab's code)
        offset = 1
        layer_hiddens = [
            output.hidden_states[layer_id + offset] for layer_id in self.target_layer_ids
        ]

        return layer_hiddens, output.logits


def create_masked_blocks(
    input_ids: torch.Tensor,
    block_size: int,
    num_anchors: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create block-diffusion training samples by randomly masking blocks.

    For each sequence, sample anchor positions. Each anchor is the first token
    of a block; remaining positions in the block are masked.

    Args:
        input_ids: [B, T] original token ids
        block_size: number of tokens per block
        num_anchors: number of anchor positions to sample per sequence
        mask_token_id: token id used for masking

    Returns:
        masked_ids: [B, T] with blocks masked
        loss_mask: [B, T] binary mask indicating positions to compute loss on
        anchor_positions: [B, num_anchors] sampled anchor indices
    """
    B, T = input_ids.shape
    masked_ids = input_ids.clone()
    loss_mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)

    all_anchors = []
    for b in range(B):
        # Sample anchor positions (avoid last block_size positions)
        valid_range = max(1, T - block_size)
        n_anchors = min(num_anchors, valid_range)
        anchors = sorted(random.sample(range(valid_range), n_anchors))
        all_anchors.append(anchors)

        # Mask positions after each anchor (positions 1..block_size-1 in each block)
        for anchor in anchors:
            end = min(anchor + block_size, T)
            masked_ids[b, anchor + 1 : end] = mask_token_id
            loss_mask[b, anchor + 1 : end] = True

    return masked_ids, loss_mask, all_anchors
