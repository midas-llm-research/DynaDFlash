"""Tests for midas/train.py — checkpoint utilities and train loop smoke test."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import pytest

from midas.train import (
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    build_default_model,
)
from midas.config import TrainConfig, RouterConfig
from midas.models.model import DFlashDraftModel
from midas.models.model_context_router import DFlashDraftModelRouted
from tests.conftest import make_tiny_qwen3_config


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def _make_model_and_optimizer(self):
        config = make_tiny_qwen3_config()
        model = DFlashDraftModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
        return model, optimizer, scheduler

    def test_creates_files(self, tmp_path):
        model, optimizer, scheduler = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "ckpts")
        save_checkpoint(model, optimizer, scheduler, epoch=1, step=100, prev_batch_idx=5, loss=0.5, save_dir=save_dir)

        ckpt_dir = tmp_path / "ckpts" / "checkpoint_epoch1_step100"
        assert ckpt_dir.exists()
        assert (ckpt_dir / "model.pt").exists()
        assert (ckpt_dir / "optimizer.pt").exists()
        assert (ckpt_dir / "scheduler.pt").exists()
        assert (ckpt_dir / "metadata.json").exists()

    def test_metadata_contents(self, tmp_path):
        model, optimizer, scheduler = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "ckpts")
        save_checkpoint(model, optimizer, scheduler, epoch=2, step=200, prev_batch_idx=10, loss=0.3, save_dir=save_dir)

        ckpt_dir = tmp_path / "ckpts" / "checkpoint_epoch2_step200"
        with open(ckpt_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["epoch"] == 2
        assert meta["step"] == 200
        assert meta["prev_batch_idx"] == 10
        assert meta["loss"] == 0.3

    def test_latest_symlink(self, tmp_path):
        model, optimizer, scheduler = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "ckpts")
        save_checkpoint(model, optimizer, scheduler, epoch=1, step=100, prev_batch_idx=5, loss=0.5, save_dir=save_dir)

        latest = tmp_path / "ckpts" / "latest"
        assert latest.is_symlink()
        assert latest.resolve().name == "checkpoint_epoch1_step100"

    def test_latest_symlink_updates(self, tmp_path):
        model, optimizer, scheduler = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "ckpts")
        save_checkpoint(model, optimizer, scheduler, epoch=1, step=100, prev_batch_idx=5, loss=0.5, save_dir=save_dir)
        save_checkpoint(model, optimizer, scheduler, epoch=1, step=200, prev_batch_idx=10, loss=0.4, save_dir=save_dir)

        latest = tmp_path / "ckpts" / "latest"
        assert latest.resolve().name == "checkpoint_epoch1_step200"

    def test_best_symlink(self, tmp_path):
        model, optimizer, scheduler = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "ckpts")
        save_checkpoint(model, optimizer, scheduler, epoch=1, step=100, prev_batch_idx=5, loss=0.5, save_dir=save_dir, is_best=True)

        best = tmp_path / "ckpts" / "best"
        assert best.is_symlink()
        assert best.resolve().name == "checkpoint_epoch1_step100"

    def test_no_best_symlink_when_not_best(self, tmp_path):
        model, optimizer, scheduler = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "ckpts")
        save_checkpoint(model, optimizer, scheduler, epoch=1, step=100, prev_batch_idx=5, loss=0.5, save_dir=save_dir, is_best=False)

        best = tmp_path / "ckpts" / "best"
        assert not best.exists()


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def _make_model_and_optimizer(self):
        config = make_tiny_qwen3_config()
        model = DFlashDraftModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
        return model, optimizer, scheduler, config

    def test_returns_zeros_when_no_checkpoint(self, tmp_path):
        model, optimizer, scheduler, _ = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "empty_ckpts")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        epoch, step, batch_idx = load_checkpoint(model, optimizer, scheduler, save_dir)
        assert (epoch, step, batch_idx) == (0, 0, -1)

    def test_restores_state(self, tmp_path):
        model, optimizer, scheduler, config = self._make_model_and_optimizer()
        save_dir = str(tmp_path / "ckpts")

        # Modify model state
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(42.0)

        # Do an optimizer step to get non-zero optimizer state
        dummy_loss = sum(p.sum() for p in model.parameters())
        dummy_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Save
        save_checkpoint(model, optimizer, scheduler, epoch=3, step=500, prev_batch_idx=20, loss=0.2, save_dir=save_dir)

        # Create a fresh model and load
        model2 = DFlashDraftModel(config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        from transformers import get_cosine_schedule_with_warmup
        scheduler2 = get_cosine_schedule_with_warmup(optimizer2, num_warmup_steps=10, num_training_steps=100)

        epoch, step, batch_idx = load_checkpoint(model2, optimizer2, scheduler2, save_dir)
        assert epoch == 3
        assert step == 500
        assert batch_idx == 20

        # Model weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# cleanup_old_checkpoints
# ---------------------------------------------------------------------------


class TestCleanupOldCheckpoints:
    def test_keeps_last_n(self, tmp_path):
        save_dir = str(tmp_path / "ckpts")
        Path(save_dir).mkdir()

        # Create 5 fake checkpoint dirs with staggered mtimes
        for i in range(5):
            d = Path(save_dir) / f"checkpoint_epoch0_step{i * 100}"
            d.mkdir()
            (d / "model.pt").touch()
            time.sleep(0.01)  # ensure different mtimes

        cleanup_old_checkpoints(save_dir, keep_last_n=2)

        remaining = [d.name for d in Path(save_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
        assert len(remaining) == 2

    def test_preserves_best(self, tmp_path):
        save_dir = str(tmp_path / "ckpts")
        Path(save_dir).mkdir()

        # Create 5 fake checkpoint dirs
        for i in range(5):
            d = Path(save_dir) / f"checkpoint_epoch0_step{i * 100}"
            d.mkdir()
            (d / "model.pt").touch()
            time.sleep(0.01)

        # Mark the oldest one as "best"
        best_link = Path(save_dir) / "best"
        best_link.symlink_to("checkpoint_epoch0_step0")

        cleanup_old_checkpoints(save_dir, keep_last_n=2)

        # Best should still exist even though it's the oldest
        assert (Path(save_dir) / "checkpoint_epoch0_step0").exists()


# ---------------------------------------------------------------------------
# build_default_model
# ---------------------------------------------------------------------------


class TestBuildDefaultModel:
    def test_raises_without_router_config(self):
        config = TrainConfig(router=None)
        with pytest.raises(ValueError, match="router"):
            build_default_model(config, device=torch.device("cpu"), model_cls=DFlashDraftModelRouted)


# ---------------------------------------------------------------------------
# Train loop smoke test
# ---------------------------------------------------------------------------


class TestTrainSmoke:
    def test_one_step_loss_is_finite(self, tmp_path, tmp_jsonl):
        """
        Run a single forward+backward pass with a tiny model and mock target.
        Verifies the training pipeline doesn't crash and produces a finite loss.
        """
        from midas.data import prepare_draft_inputs, DraftTrainingBatch

        config = make_tiny_qwen3_config()
        model = DFlashDraftModel(config)
        model.train()

        B, T = 2, 64
        H = config.hidden_size
        V = config.vocab_size
        block_size = config.block_size
        num_target_layer_ids = len(model.target_layer_ids)

        # Simulate target model output
        input_ids = torch.randint(0, V, (B, T))
        seq_lengths = torch.tensor([T, T])

        # Fake hidden states from target
        num_target_layers = config.num_target_layers
        hidden_states = tuple(torch.randn(B, T, H) for _ in range(num_target_layers + 1))
        target_logits = torch.randn(B, T, V)

        # Extract target hidden
        target_hidden = model.extract_target_hidden(hidden_states)

        # Prepare draft inputs
        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=8, mask_token_id=0, min_anchor_pos=10,
        )

        # Embed noise tokens (simulate target embedding)
        embed = nn.Embedding(V, H)
        noise_embedding = embed(batch.noise_token_ids)

        # Forward pass
        draft_output = model(
            position_ids=batch.position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=batch.attention_mask,
        )

        # Compute loss
        lm_head = nn.Linear(H, V, bias=False)
        loss = model.compute_loss(
            draft_output, target_logits, batch, lm_head, gamma=7.0,
        )

        assert loss.isfinite()
        assert loss.item() > 0

        # Backward pass should work
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step_changes_weights(self, tmp_path):
        """Verify that an optimizer step actually updates model weights."""
        from midas.data import prepare_draft_inputs

        config = make_tiny_qwen3_config()
        model = DFlashDraftModel(config)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        B, T = 2, 64
        H = config.hidden_size
        V = config.vocab_size
        block_size = config.block_size

        # Store initial weights
        initial_fc_weight = model.fc.weight.clone()

        # Forward
        input_ids = torch.randint(0, V, (B, T))
        seq_lengths = torch.tensor([T, T])
        hidden_states = tuple(torch.randn(B, T, H) for _ in range(config.num_target_layers + 1))
        target_logits = torch.randn(B, T, V)
        target_hidden = model.extract_target_hidden(hidden_states)

        batch = prepare_draft_inputs(
            input_ids, seq_lengths, block_size=block_size,
            max_num_anchors=8, mask_token_id=0, min_anchor_pos=10,
        )

        embed = nn.Embedding(V, H)
        noise_embedding = embed(batch.noise_token_ids)

        draft_output = model(
            position_ids=batch.position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=batch.attention_mask,
        )

        lm_head = nn.Linear(H, V, bias=False)
        loss = model.compute_loss(draft_output, target_logits, batch, lm_head, gamma=7.0)
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.equal(model.fc.weight, initial_fc_weight)
