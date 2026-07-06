"""Tests for midas/models/model_block_init.py — DFlashDraftModelBlockInit."""

import pytest
import torch

from midas.models.model import DFlashDraftModel
from midas.models.model_block_init import (
    AnchorBlockInitializer,
    DFlashDraftModelBlockInit,
)
from tests.conftest import make_tiny_qwen3_config, MockLMHead


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config()


@pytest.fixture
def block_init_model(tiny_config):
    model = DFlashDraftModelBlockInit(tiny_config)
    model.eval()
    return model


@pytest.fixture
def base_model(tiny_config):
    model = DFlashDraftModel(tiny_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestBlockInitInit:
    def test_has_block_initializer(self, block_init_model):
        assert hasattr(block_init_model, "block_initializer")
        assert isinstance(block_init_model.block_initializer, AnchorBlockInitializer)

    def test_inherits_fc_and_hidden_norm(self, block_init_model):
        assert hasattr(block_init_model, "fc")
        assert hasattr(block_init_model, "hidden_norm")

    def test_anchor_proj_zero_init(self, block_init_model):
        w = block_init_model.block_initializer.anchor_proj.weight
        assert (w == 0).all()

    def test_block_pos_embed_zero_init(self, block_init_model):
        w = block_init_model.block_initializer.block_pos_embed.weight
        assert (w == 0).all()

    def test_param_count_reasonable(self, block_init_model, tiny_config):
        init_params = sum(
            p.numel() for p in block_init_model.block_initializer.parameters()
        )
        H = tiny_config.hidden_size
        block_size = tiny_config.block_size
        expected = H * H + block_size * H
        assert init_params == expected


# ---------------------------------------------------------------------------
# Zero-init equivalence: output should match base model at initialization
# ---------------------------------------------------------------------------


class TestZeroInitEquivalence:
    def _make_inputs(self, tiny_config, model):
        B, T = 2, 16
        N = tiny_config.block_size * 2  # 2 blocks
        H = tiny_config.hidden_size
        num_target_layers = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        block_size = tiny_config.block_size
        position_in_block = (
            torch.arange(block_size).repeat(N // block_size).unsqueeze(0).expand(B, -1)
        )
        noise_block_anchor = torch.stack([
            torch.tensor([5] * block_size + [10] * block_size),
            torch.tensor([5] * block_size + [10] * block_size),
        ])

        return {
            "noise_embedding": noise_embedding,
            "target_hidden": target_hidden,
            "position_ids": position_ids,
            "position_in_block": position_in_block,
            "noise_block_anchor": noise_block_anchor,
        }

    def test_zero_init_matches_base(self, block_init_model, base_model, tiny_config):
        """With zero-initialized weights, BlockInit model should produce
        identical output to the base DFlash model."""
        # Copy base model weights into block_init model (they share the same base params)
        base_state = base_model.state_dict()
        block_state = block_init_model.state_dict()
        for key, value in base_state.items():
            if key in block_state:
                block_state[key] = value
        block_init_model.load_state_dict(block_state)

        inputs = self._make_inputs(tiny_config, block_init_model)

        with torch.no_grad():
            base_out = base_model(
                position_ids=inputs["position_ids"],
                noise_embedding=inputs["noise_embedding"].clone(),
                target_hidden=inputs["target_hidden"].clone(),
            )
            block_init_out = block_init_model(
                position_ids=inputs["position_ids"],
                noise_embedding=inputs["noise_embedding"].clone(),
                target_hidden=inputs["target_hidden"].clone(),
                noise_block_anchor=inputs["noise_block_anchor"],
                position_in_block=inputs["position_in_block"],
            )

        assert torch.allclose(base_out, block_init_out, atol=1e-5), (
            f"Max diff: {(base_out - block_init_out).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestBlockInitForward:
    def _run_forward(self, model, tiny_config, pass_kwargs=True):
        B, T = 2, 16
        N = tiny_config.block_size * 2
        H = tiny_config.hidden_size
        num_target_layers = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        kwargs = {}
        if pass_kwargs:
            block_size = tiny_config.block_size
            kwargs["position_in_block"] = (
                torch.arange(block_size).repeat(N // block_size).unsqueeze(0).expand(B, -1)
            )
            kwargs["noise_block_anchor"] = torch.full((B, N), 5, dtype=torch.long)

        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            **kwargs,
        )
        return output, B, N, H

    def test_output_shape(self, block_init_model, tiny_config):
        output, B, N, H = self._run_forward(block_init_model, tiny_config)
        assert output.shape == (B, N, H)

    def test_no_nans(self, block_init_model, tiny_config):
        output, *_ = self._run_forward(block_init_model, tiny_config)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_inference_mode_no_kwargs(self, block_init_model, tiny_config):
        """Without noise_block_anchor/position_in_block, model defaults to
        inference behavior (anchor = last position)."""
        output, B, N, H = self._run_forward(
            block_init_model, tiny_config, pass_kwargs=False
        )
        assert output.shape == (B, N, H)
        assert not torch.isnan(output).any()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestBlockInitGradients:
    def test_gradient_flows_to_anchor_proj(self, block_init_model, tiny_config):
        block_init_model.train()
        B, T = 2, 16
        N = tiny_config.block_size * 2
        H = tiny_config.hidden_size
        num_target_layers = len(block_init_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        block_size = tiny_config.block_size
        position_in_block = (
            torch.arange(block_size).repeat(N // block_size).unsqueeze(0).expand(B, -1)
        )
        noise_block_anchor = torch.full((B, N), 5, dtype=torch.long)

        # Make anchor_proj non-zero so output differs from baseline
        with torch.no_grad():
            block_init_model.block_initializer.anchor_proj.weight.fill_(0.01)

        output = block_init_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            noise_block_anchor=noise_block_anchor,
            position_in_block=position_in_block,
        )
        loss = output.sum()
        loss.backward()

        assert block_init_model.block_initializer.anchor_proj.weight.grad is not None
        assert (block_init_model.block_initializer.anchor_proj.weight.grad != 0).any()

    def test_gradient_flows_to_pos_embed(self, block_init_model, tiny_config):
        block_init_model.train()
        B, T = 2, 16
        N = tiny_config.block_size * 2
        H = tiny_config.hidden_size
        num_target_layers = len(block_init_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        block_size = tiny_config.block_size
        position_in_block = (
            torch.arange(block_size).repeat(N // block_size).unsqueeze(0).expand(B, -1)
        )
        noise_block_anchor = torch.full((B, N), 5, dtype=torch.long)

        with torch.no_grad():
            block_init_model.block_initializer.block_pos_embed.weight.fill_(0.01)

        output = block_init_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            noise_block_anchor=noise_block_anchor,
            position_in_block=position_in_block,
        )
        loss = output.sum()
        loss.backward()

        assert block_init_model.block_initializer.block_pos_embed.weight.grad is not None
        assert (block_init_model.block_initializer.block_pos_embed.weight.grad != 0).any()


# ---------------------------------------------------------------------------
# AnchorBlockInitializer unit tests
# ---------------------------------------------------------------------------


class TestAnchorBlockInitializer:
    def test_zero_init_is_noop(self):
        H, block_size = 64, 4
        init = AnchorBlockInitializer(H, block_size)
        B, N = 2, 8  # 2 blocks
        noise = torch.randn(B, N, H)
        anchor_h = torch.randn(B, 2, H)
        pos_in_block = torch.arange(block_size).repeat(2).unsqueeze(0).expand(B, -1)

        out = init(noise, anchor_h, pos_in_block)
        assert torch.allclose(out, noise, atol=1e-7)

    def test_only_masks_modified(self):
        H, block_size = 64, 4
        init = AnchorBlockInitializer(H, block_size)
        with torch.no_grad():
            init.anchor_proj.weight.fill_(1.0)
            init.block_pos_embed.weight.fill_(1.0)

        B, N = 1, 4
        noise = torch.zeros(B, N, H)
        anchor_h = torch.ones(B, 1, H)
        pos_in_block = torch.tensor([[0, 1, 2, 3]])

        out = init(noise, anchor_h, pos_in_block)
        # Position 0 (anchor) should be unchanged (still zero)
        assert (out[0, 0, :] == 0).all()
        # Positions 1-3 (masks) should be non-zero
        assert (out[0, 1:, :] != 0).all()

    def test_different_positions_get_different_values(self):
        H, block_size = 64, 4
        init = AnchorBlockInitializer(H, block_size)
        with torch.no_grad():
            init.block_pos_embed.weight.normal_()

        B, N = 1, 4
        noise = torch.zeros(B, N, H)
        anchor_h = torch.zeros(B, 1, H)
        pos_in_block = torch.tensor([[0, 1, 2, 3]])

        out = init(noise, anchor_h, pos_in_block)
        # Positions 1, 2, 3 should differ (different position embeddings)
        assert not torch.allclose(out[0, 1], out[0, 2])
        assert not torch.allclose(out[0, 2], out[0, 3])

    def test_assertion_on_bad_N(self):
        H, block_size = 64, 4
        init = AnchorBlockInitializer(H, block_size)
        noise = torch.randn(1, 5, H)  # 5 is NOT divisible by 4
        anchor_h = torch.randn(1, 1, H)
        pos_in_block = torch.arange(5).unsqueeze(0)

        with pytest.raises(AssertionError):
            init(noise, anchor_h, pos_in_block)
