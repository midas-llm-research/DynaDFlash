"""Tests for midas/models/model_dflash_plus.py — DFlashDraftModelPlus."""

import pytest
import torch

from midas.config import RouterConfig
from midas.models.model_context_router import DFlashDraftModelRouted
from midas.models.model_dflash_plus import DFlashDraftModelPlus
from midas.models.model_token_mix import Qwen3DFlashTokenMixDecoderLayer
from tests.conftest import make_tiny_qwen3_config


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config()


@pytest.fixture
def router_config(tiny_config):
    # num_layers must match the number of extracted target layers, which the
    # base derives as one per draft layer.
    return RouterConfig(
        hidden_size=tiny_config.hidden_size,
        num_layers=tiny_config.num_hidden_layers,
        rank=16,
    )


@pytest.fixture
def model(tiny_config, router_config):
    return DFlashDraftModelPlus(tiny_config, router_config).eval()


@pytest.fixture
def fwd(tiny_config):
    """Forward kwargs shaped like a training batch (multiple concatenated blocks)."""
    B, T = 2, 16
    N = tiny_config.block_size * 2
    H = tiny_config.hidden_size
    n_layers = tiny_config.num_hidden_layers
    return dict(
        position_ids=torch.arange(T + N).unsqueeze(0).expand(B, -1),
        noise_embedding=torch.randn(B, N, H),
        target_hidden=torch.randn(B, T, n_layers * H),
    )


class TestPlusInit:
    def test_all_layers_are_token_mix(self, model):
        assert all(isinstance(l, Qwen3DFlashTokenMixDecoderLayer) for l in model.layers)

    def test_has_context_router_not_fc(self, model):
        assert hasattr(model, "context_router")
        # the router replaces DFlash's static fusion
        assert not hasattr(model, "fc")
        assert not hasattr(model, "hidden_norm")

    def test_token_mixers_zero_at_init(self, model):
        for layer in model.layers:
            assert torch.equal(
                layer.token_mixer.weight, torch.zeros_like(layer.token_mixer.weight)
            )

    def test_mixer_shape_and_param_delta(self, model, tiny_config, router_config):
        bs = tiny_config.block_size
        for layer in model.layers:
            assert layer.token_mixer.weight.shape == (bs, bs)
        routed = DFlashDraftModelRouted(tiny_config, router_config)
        delta = sum(p.numel() for p in model.parameters()) - sum(
            p.numel() for p in routed.parameters()
        )
        assert delta == bs * bs * tiny_config.num_hidden_layers

    def test_state_dict_is_routed_plus_mixers(self, model, tiny_config, router_config):
        routed = DFlashDraftModelRouted(tiny_config, router_config)
        extra = set(model.state_dict()) - set(routed.state_dict())
        assert set(routed.state_dict()) - set(model.state_dict()) == set()
        assert extra and all("token_mixer" in k for k in extra)


class TestZeroInitEquivalence:
    """With mixers zeroed, output must match the ROUTED model (not the base) —
    the ContextRouter is not a no-op, only the token mixer is."""

    def test_matches_routed_model(self, tiny_config, router_config, fwd):
        routed = DFlashDraftModelRouted(tiny_config, router_config).eval()
        plus = DFlashDraftModelPlus(tiny_config, router_config).eval()
        sd = routed.state_dict()
        plus.load_state_dict({**plus.state_dict(), **sd})
        with torch.no_grad():
            a = routed(**fwd)
            b = plus(**fwd)
        assert torch.allclose(a, b, atol=1e-5)


class TestPlusForward:
    def test_output_shape_and_finite(self, model, fwd):
        out = model(**fwd)
        noise = fwd["noise_embedding"]
        assert out.shape == noise.shape
        assert torch.isfinite(out).all()

    def test_single_block_inference_shape(self, model, tiny_config):
        """N == block_size, the shape dflash_generate produces."""
        B, H = 1, tiny_config.hidden_size
        N, T = tiny_config.block_size, 6
        n_layers = tiny_config.num_hidden_layers
        out = model(
            position_ids=torch.arange(T + N).unsqueeze(0),
            noise_embedding=torch.randn(B, N, H),
            target_hidden=torch.randn(B, T, n_layers * H),
            attention_mask=torch.zeros(B, 1, N, T + N),
        )
        assert out.shape == (B, N, H)
        assert torch.isfinite(out).all()

    def test_non_multiple_block_size_asserts(self, model, tiny_config):
        """Silent skip would diverge train vs inference — must raise instead."""
        B, H = 1, tiny_config.hidden_size
        N, T = tiny_config.block_size + 1, 6
        n_layers = tiny_config.num_hidden_layers
        with pytest.raises(AssertionError):
            model(
                position_ids=torch.arange(T + N).unsqueeze(0),
                noise_embedding=torch.randn(B, N, H),
                target_hidden=torch.randn(B, T, n_layers * H),
                attention_mask=torch.zeros(B, 1, N, T + N),
            )


class TestPlusGradients:
    def test_gradients_reach_both_mechanisms(self, model, fwd):
        out = model(**fwd)
        out.sum().backward()
        for layer in model.layers:
            g = layer.token_mixer.weight.grad
            assert g is not None and (g != 0).any()
        for name in ("W_q", "W_k"):
            g = getattr(model.context_router, name).weight.grad
            assert g is not None and (g != 0).any()
