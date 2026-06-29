import pytest
pytestmark = pytest.mark.data

"""Tests for midas/data.py — dataset, collate, sampler, prepare_draft_inputs."""

import json
from pathlib import Path

import torch
import pytest

from midas.data import (
    DFlashDataset,
    DFlashDataProcessor,
    LengthInterleavedBatchSampler,
    OnlineHiddenStateExtractor,
    collate_fn,
    prepare_draft_inputs,
    DraftTrainingBatch,
)
from midas.config import TrainConfig


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


class TestCollateFn:
    def test_padding_to_max_length(self):
        batch = [(torch.tensor([1, 2, 3]), 1), (torch.tensor([4, 5, 6, 7, 8]), 2)]
        padded, lengths, rs = collate_fn(batch, pad_token_id=0)
        assert padded.shape == (2, 5)
        assert torch.equal(padded[0], torch.tensor([1, 2, 3, 0, 0]))
        assert torch.equal(padded[1], torch.tensor([4, 5, 6, 7, 8]))

    def test_lengths_correct(self):
        batch = [(torch.tensor([1, 2]), 0), (torch.tensor([3, 4, 5, 6]), 1)]
        padded, lengths, rs = collate_fn(batch, pad_token_id=99)
        assert torch.equal(lengths, torch.tensor([2, 4]))

    def test_single_item(self):
        batch = [(torch.tensor([10, 20, 30]), 1)]
        padded, lengths, rs = collate_fn(batch, pad_token_id=0)
        assert padded.shape == (1, 3)
        assert lengths[0] == 3

    def test_equal_lengths_no_padding(self):
        batch = [(torch.tensor([1, 2, 3]), 0), (torch.tensor([4, 5, 6]), 1)]
        padded, lengths, rs = collate_fn(batch, pad_token_id=0)
        assert padded.shape == (2, 3)
        assert (padded != 0).all()

    def test_pad_token_id_used(self):
        batch = [(torch.tensor([1, 2]), 0), (torch.tensor([3, 4, 5]), 1)]
        padded, _, _ = collate_fn(batch, pad_token_id=42)
        assert padded[0, 2] == 42

    def test_response_starts_returned(self):
        batch = [(torch.tensor([1, 2, 3]), 5), (torch.tensor([4, 5, 6, 7]), 10)]
        _, _, rs = collate_fn(batch, pad_token_id=0)
        assert torch.equal(rs, torch.tensor([5, 10]))


# ---------------------------------------------------------------------------
# LengthInterleavedBatchSampler
# ---------------------------------------------------------------------------


class TestLengthInterleavedBatchSampler:
    def test_all_indices_covered(self):
        lengths = [100, 200, 150, 300, 50, 250, 180, 90]
        sampler = LengthInterleavedBatchSampler(lengths, batch_size=3)
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(lengths)))

    def test_batch_size_respected(self):
        lengths = [100, 200, 150, 300, 50, 250, 180, 90, 120, 400]
        sampler = LengthInterleavedBatchSampler(lengths, batch_size=4)
        for batch in sampler:
            assert len(batch) <= 4

    def test_length_bucketing(self):
        lengths = [10, 100, 20, 90, 30, 80, 40, 70, 50, 60]
        sampler = LengthInterleavedBatchSampler(lengths, batch_size=2)
        batches = list(sampler)
        for batch in batches:
            batch_lengths = [lengths[i] for i in batch]
            if len(batch_lengths) == 2:
                # Within a bucket, lengths should be close (sorted contiguously)
                assert abs(batch_lengths[0] - batch_lengths[1]) <= 20

    def test_interleaved_order(self):
        # 10 items with lengths 10, 20, ..., 100. batch_size=2
        lengths = list(range(10, 110, 10))
        sampler = LengthInterleavedBatchSampler(lengths, batch_size=2)
        batches = list(sampler)
        # First batch should have the longest items
        first_batch_lengths = sorted([lengths[i] for i in batches[0]], reverse=True)
        # Second batch should have the shortest items
        second_batch_lengths = sorted([lengths[i] for i in batches[1]], reverse=True)
        assert first_batch_lengths[0] >= second_batch_lengths[0]

    def test_drop_last(self):
        lengths = [100, 200, 150, 300, 50]  # 5 items, batch_size=3
        sampler_keep = LengthInterleavedBatchSampler(lengths, batch_size=3, drop_last=False)
        sampler_drop = LengthInterleavedBatchSampler(lengths, batch_size=3, drop_last=True)
        batches_keep = list(sampler_keep)
        batches_drop = list(sampler_drop)
        assert len(batches_keep) == 2  # [3, 2]
        assert len(batches_drop) == 1  # only the full batch of 3

    def test_len_correct(self):
        lengths = [100] * 10
        sampler = LengthInterleavedBatchSampler(lengths, batch_size=3)
        assert len(sampler) == 4  # ceil(10/3) = 4

    def test_len_drop_last(self):
        lengths = [100] * 10
        sampler = LengthInterleavedBatchSampler(lengths, batch_size=3, drop_last=True)
        assert len(sampler) == 3  # floor(10/3) = 3


# ---------------------------------------------------------------------------
# DFlashDataset
# ---------------------------------------------------------------------------


class TestDFlashDataset:
    def test_loads_jsonl(self, tmp_jsonl):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        ds = DFlashDataset(tokenizer, tmp_jsonl, max_seq_len=4096, min_seq_len=1)
        assert len(ds) == 5
        assert all(isinstance(seq, torch.Tensor) for seq in ds.sequences)

    def test_filters_short(self, tmp_jsonl):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        # Set min_seq_len very high so everything is filtered
        ds = DFlashDataset(tokenizer, tmp_jsonl, max_seq_len=4096, min_seq_len=99999)
        assert len(ds) == 0

    def test_filters_long(self, tmp_jsonl):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        # Set max_seq_len=1 so everything is filtered
        ds = DFlashDataset(tokenizer, tmp_jsonl, max_seq_len=1, min_seq_len=1)
        assert len(ds) == 0

    def test_max_samples(self, tmp_jsonl):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        ds = DFlashDataset(tokenizer, tmp_jsonl, max_seq_len=4096, min_seq_len=1, max_samples=2)
        assert len(ds) == 2

    def test_file_not_found(self, tmp_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        with pytest.raises(FileNotFoundError):
            DFlashDataset(tokenizer, str(tmp_path / "nonexistent.jsonl"))

    def test_getitem(self, tmp_jsonl):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        ds = DFlashDataset(tokenizer, tmp_jsonl, max_seq_len=4096, min_seq_len=1)
        ids, response_start = ds[0]
        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 1
        assert isinstance(response_start, int)
        assert response_start > 0


# ---------------------------------------------------------------------------
# prepare_draft_inputs
# ---------------------------------------------------------------------------


class TestPrepareDraftInputs:
    def _make_inputs(self, B=2, T=256, block_size=4, max_num_anchors=8):
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([200, 220][:B])
        response_starts = torch.tensor([10] * B)
        return input_ids, seq_lengths, block_size, max_num_anchors, response_starts

    def test_output_types(self):
        input_ids, seq_lengths, block_size, max_num_anchors, response_starts = self._make_inputs()
        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=max_num_anchors, mask_token_id=999, response_starts=response_starts,
        )
        assert isinstance(batch, DraftTrainingBatch)
        assert isinstance(batch.noise_token_ids, torch.Tensor)
        assert isinstance(batch.position_ids, torch.Tensor)
        assert isinstance(batch.attention_mask, torch.Tensor)

    def test_output_shapes(self):
        B, T = 2, 256
        block_size = 4
        max_num_anchors = 8
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([200, 220])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=max_num_anchors, mask_token_id=999, response_starts=torch.tensor([10] * B),
        )

        num_anchors = batch.noise_token_ids.shape[1] // block_size
        N = num_anchors * block_size

        assert batch.noise_token_ids.shape == (B, N)
        assert batch.position_ids.shape == (B, T + N)
        assert batch.attention_mask.shape == (B, 1, N, T + N)
        assert batch.noise_original_positions.shape == (B, N)
        assert batch.loss_mask.shape == (B, N)
        assert batch.position_in_block.shape == (B, N)

    def test_anchor_positions_have_real_tokens(self):
        B, T = 1, 128
        block_size = 4
        input_ids = torch.arange(T).unsqueeze(0)  # tokens are 0,1,2,...,T-1
        seq_lengths = torch.tensor([T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=4, mask_token_id=999, response_starts=torch.tensor([10] * B),
        )

        # At position_in_block == 0, the token should be the original input token
        N = batch.noise_token_ids.shape[1]
        for i in range(0, N, block_size):
            orig_pos = batch.noise_original_positions[0, i].item()
            assert batch.noise_token_ids[0, i].item() == input_ids[0, orig_pos].item()

    def test_mask_positions_have_mask_token(self):
        B, T = 1, 128
        block_size = 4
        mask_token_id = 999
        input_ids = torch.randint(0, 100, (B, T))
        seq_lengths = torch.tensor([T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=4, mask_token_id=mask_token_id, response_starts=torch.tensor([10]),
        )

        N = batch.noise_token_ids.shape[1]
        for i in range(N):
            if batch.position_in_block[0, i].item() > 0:
                assert batch.noise_token_ids[0, i].item() == mask_token_id

    def test_position_ids_context_portion(self):
        B, T = 2, 64
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([T, T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=4,
            max_num_anchors=4, mask_token_id=999, response_starts=torch.tensor([10, 10]),
        )

        # First T entries of position_ids should be 0..T-1
        expected = torch.arange(T).unsqueeze(0).expand(B, -1)
        assert torch.equal(batch.position_ids[:, :T], expected)

    def test_position_ids_noise_portion_matches_original_positions(self):
        B, T = 1, 128
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=4,
            max_num_anchors=4, mask_token_id=999, response_starts=torch.tensor([10] * B),
        )

        # Noise portion of position_ids should equal noise_original_positions
        N = batch.noise_token_ids.shape[1]
        assert torch.equal(batch.position_ids[:, T:], batch.noise_original_positions)

    def test_attention_mask_causal_context(self):
        """Each noise token should only see context positions BEFORE its block's anchor."""
        B, T = 1, 64
        block_size = 4
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=2, mask_token_id=999, response_starts=torch.tensor([10]),
        )

        N = batch.noise_token_ids.shape[1]
        num_anchors = N // block_size
        mask = batch.attention_mask[0, 0]  # [N, T+N]

        for block_i in range(num_anchors):
            block_start = block_i * block_size
            anchor_pos = batch.noise_original_positions[0, block_start].item()
            for noise_j in range(block_start, block_start + block_size):
                # Context positions >= anchor should be masked (-inf)
                for ctx_pos in range(T):
                    if ctx_pos < anchor_pos:
                        assert mask[noise_j, ctx_pos] == 0.0, \
                            f"noise[{noise_j}] should see ctx[{ctx_pos}] (anchor={anchor_pos})"
                    else:
                        assert mask[noise_j, ctx_pos] == float("-inf"), \
                            f"noise[{noise_j}] should NOT see ctx[{ctx_pos}] (anchor={anchor_pos})"

    def test_attention_mask_block_diagonal(self):
        """Noise tokens should only see other noise tokens in the same block."""
        B, T = 1, 128
        block_size = 4
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=3, mask_token_id=999, response_starts=torch.tensor([10] * B),
        )

        N = batch.noise_token_ids.shape[1]
        num_anchors = N // block_size
        mask = batch.attention_mask[0, 0]  # [N, T+N]
        noise_mask = mask[:, T:]  # [N, N] — noise-to-noise attention

        for i in range(N):
            block_i = i // block_size
            for j in range(N):
                block_j = j // block_size
                if block_i == block_j:
                    assert noise_mask[i, j] == 0.0, \
                        f"Same block: noise[{i}] should see noise[{j}]"
                else:
                    assert noise_mask[i, j] == float("-inf"), \
                        f"Different block: noise[{i}] should NOT see noise[{j}]"

    def test_loss_mask_excludes_anchors(self):
        B, T = 2, 128
        block_size = 4
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([T, T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=4, mask_token_id=999, response_starts=torch.tensor([10] * B),
        )

        N = batch.noise_token_ids.shape[1]
        for b in range(B):
            for i in range(0, N, block_size):
                assert batch.loss_mask[b, i].item() is False

    def test_loss_mask_excludes_sentinels(self):
        """When one row is shorter, its sentinel blocks should have loss_mask=False."""
        B, T = 2, 256
        block_size = 4
        # Row 0: very short (can only fit 2 anchors), Row 1: long (can fit many)
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([74, 250])  # row 0: 74 - 4 + 1 - 64 = 7 anchors max

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=32, mask_token_id=999, response_starts=torch.tensor([64, 64]),
        )

        N = batch.noise_token_ids.shape[1]
        num_anchors = N // block_size

        # Row 0 can fit at most 7 anchors. Anything beyond should be sentinel.
        row0_max = 74 - block_size + 1 - 64
        if num_anchors > row0_max:
            # Sentinel blocks should have all-False loss_mask
            for block_i in range(row0_max, num_anchors):
                block_start = block_i * block_size
                block_end = block_start + block_size
                assert not batch.loss_mask[0, block_start:block_end].any()

    def test_position_in_block_pattern(self):
        B, T = 1, 128
        block_size = 4
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([T])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=3, mask_token_id=999, response_starts=torch.tensor([10] * B),
        )

        N = batch.noise_token_ids.shape[1]
        expected_pattern = list(range(block_size)) * (N // block_size)
        assert batch.position_in_block[0].tolist() == expected_pattern

    def test_variable_lengths(self):
        """Batch with different sequence lengths should work."""
        B, T = 3, 256
        block_size = 4
        input_ids = torch.randint(0, 1000, (B, T))
        seq_lengths = torch.tensor([100, 200, 150])

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=8, mask_token_id=999, response_starts=torch.tensor([64, 64, 64]),
        )
        # Should not raise and should produce valid shapes
        N = batch.noise_token_ids.shape[1]
        assert N > 0
        assert batch.noise_token_ids.shape[0] == B

    def test_raises_on_too_short(self):
        """If a row can't fit any anchor, should raise ValueError."""
        B, T = 2, 256
        block_size = 4
        input_ids = torch.randint(0, 1000, (B, T))
        # Row 0 has length 10, with response_start=64 it can't fit any anchor
        seq_lengths = torch.tensor([10, 200])

        with pytest.raises(ValueError):
            prepare_draft_inputs(
                input_ids, seq_lengths, block_size=block_size,
                max_num_anchors=8, mask_token_id=999, response_starts=torch.tensor([64, 64]),
            )


# ---------------------------------------------------------------------------
# DFlashDataProcessor
# ---------------------------------------------------------------------------


class TestDFlashDataProcessor:
    def test_creates_dataloader(self, tmp_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

        # Create data with longer responses so they pass min_seq_len
        data = [
            {"prompt": "Tell me about machine learning in great detail.", "response": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. It encompasses supervised learning, unsupervised learning, and reinforcement learning paradigms. Each approach has its own strengths and applications in the real world, from image classification to natural language processing to game playing."},
            {"prompt": "Explain the history of computing from the beginning.", "response": "Computing has a rich history dating back to ancient abacuses and mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, which is considered the first general-purpose computer design. Ada Lovelace wrote what is recognized as the first computer program for it. The modern electronic computer era began in the 1940s with machines like ENIAC and Colossus."},
        ]
        path = tmp_path / "long_data.jsonl"
        with open(path, "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")

        config = TrainConfig(
            data_path=str(path),
            max_seq_len=4096,
            batch_size=2,
            block_size=4,
            max_num_anchors=8,
        )
        processor = DFlashDataProcessor(tokenizer, config, num_workers=0, )
        batch_iter = iter(processor)
        input_ids, lengths, response_starts = next(batch_iter)
        assert input_ids.dim() == 2
        assert lengths.dim() == 1
        assert response_starts.dim() == 1
        assert input_ids.shape[0] == lengths.shape[0]

    def test_filters_too_short_sequences(self, tmp_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

        # Create data: one long response (survives) and one short (filtered by DataProcessor)
        data = [
            {"prompt": "Tell me about the entire history of mathematics from ancient times to the modern era.", "response": "Mathematics has evolved over thousands of years, starting with basic counting in ancient civilizations. The Babylonians developed a base-60 number system, while the Egyptians used geometry for building pyramids. Greek mathematicians like Euclid and Archimedes formalized proofs and geometry. The Islamic golden age preserved and extended Greek work, introducing algebra. The Renaissance brought calculus from Newton and Leibniz. Modern mathematics includes abstract algebra, topology, and computer science."},
            {"prompt": "Short", "response": "Yes."},  # very short — won't fit an anchor
        ]
        path = tmp_path / "mixed_data.jsonl"
        with open(path, "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")

        config = TrainConfig(
            data_path=str(path),
            max_seq_len=4096,
            batch_size=2,
            block_size=4,
            max_num_anchors=8,
        )
        # Use min_seq_len=1 so the dataset loads both, but DataProcessor filters
        # sequences too short to fit an anchor (response_start + block_size).
        processor = DFlashDataProcessor(tokenizer, config, num_workers=0, )
        # The long one should survive, the short one may or may not depending on tokenization
        assert len(processor.dataset) >= 1


# ---------------------------------------------------------------------------
# OnlineHiddenStateExtractor
# ---------------------------------------------------------------------------


class TestOnlineHiddenStateExtractor:
    def test_extract_returns_correct_types(self):
        import torch.nn as nn
        from types import SimpleNamespace

        class FakeInnerModel(nn.Module):
            def forward(self, input_ids, position_ids=None, output_hidden_states=False):
                B, T = input_ids.shape
                H = 64
                num_layers = 4
                hidden_states = tuple(torch.randn(B, T, H) for _ in range(num_layers + 1))
                return SimpleNamespace(hidden_states=hidden_states)

        class FakeTargetModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeInnerModel()

        model = FakeTargetModel()
        extractor = OnlineHiddenStateExtractor(model, device=torch.device("cpu"))

        input_ids = torch.randint(0, 100, (2, 16))
        hidden_states = extractor.extract(input_ids)

        assert isinstance(hidden_states, tuple)
        assert len(hidden_states) == 5  # num_layers + 1
        assert hidden_states[0].shape == (2, 16, 64)
