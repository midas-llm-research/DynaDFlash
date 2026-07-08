"""Tests for midas/models/model_token_mix.py — DFlashDraftModelTokenMix."""

import pytest
import torch

from midas.models.model import DFlashDraftModel
from midas.models.model_token_mix import (
    DFlashDraftModelTokenMix,
    Qwen3DFlashTokenMixDecoderLayer,
)
from tests.conftest import make_tiny_qwen3_config


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config()


@pytest.fixture
def token_mix_model(tiny_config):
    model = DFlashDraftModelTokenMix(tiny_config)
    model.eval()
    return model


@pytest.fixture
def base_model(tiny_config):
    model = DFlashDraftModel(tiny_config)
    model.eval()
    return model


class TestTokenMixInit:
    def test_layers_have_token_mixer(self, token_mix_model):
        for layer in token_mix_model.layers:
            assert isinstance(layer, Qwen3DFlashTokenMixDecoderLayer)
            assert hasattr(layer, "token_mixer")

    def test_token_mixer_zero_init(self, token_mix_model):
        for layer in token_mix_model.layers:
            assert (layer.token_mixer.weight == 0).all()

    def test_token_mixer_shape(self, token_mix_model, tiny_config):
        for layer in token_mix_model.layers:
            bs = tiny_config.block_size
            assert layer.token_mixer.weight.shape == (bs, bs)

    def test_inherits_fusion(self, token_mix_model):
        assert hasattr(token_mix_model, "fc")
        assert hasattr(token_mix_model, "hidden_norm")

    def test_total_mixer_params(self, token_mix_model, tiny_config):
        bs = tiny_config.block_size
        n_layers = tiny_config.num_hidden_layers
        mixer_params = sum(
            layer.token_mixer.weight.numel() for layer in token_mix_model.layers
        )
        assert mixer_params == bs * bs * n_layers


class TestZeroInitEquivalence:
    def _make_inputs(self, tiny_config, model):
        B, T = 2, 16
        N = tiny_config.block_size * 2
        H = tiny_config.hidden_size
        num_target_layers = len(model.target_layer_ids)
        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        return noise_embedding, target_hidden, position_ids

    def test_zero_init_matches_base(self, token_mix_model, base_model, tiny_config):
        base_state = base_model.state_dict()
        mix_state = token_mix_model.state_dict()
        for key, value in base_state.items():
            if key in mix_state:
                mix_state[key] = value
        token_mix_model.load_state_dict(mix_state)

        noise, target_h, pos_ids = self._make_inputs(tiny_config, token_mix_model)
        with torch.no_grad():
            base_out = base_model(
                position_ids=pos_ids,
                noise_embedding=noise.clone(),
                target_hidden=target_h.clone(),
            )
            mix_out = token_mix_model(
                position_ids=pos_ids,
                noise_embedding=noise.clone(),
                target_hidden=target_h.clone(),
            )
        assert torch.allclose(base_out, mix_out, atol=1e-5), (
            f"Max diff: {(base_out - mix_out).abs().max().item()}"
        )


class TestTokenMixForward:
    def _run_forward(self, model, tiny_config):
        B, T = 2, 16
        N = tiny_config.block_size * 2
        H = tiny_config.hidden_size
        num_target_layers = len(model.target_layer_ids)
        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        return output, B, N, H

    def test_output_shape(self, token_mix_model, tiny_config):
        output, B, N, H = self._run_forward(token_mix_model, tiny_config)
        assert output.shape == (B, N, H)

    def test_no_nans(self, token_mix_model, tiny_config):
        output, *_ = self._run_forward(token_mix_model, tiny_config)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_single_block_inference(self, token_mix_model, tiny_config):
        """Simulates inference: N = block_size (single block)."""
        B = 1
        N = tiny_config.block_size
        T = 8
        H = tiny_config.hidden_size
        num_target_layers = len(token_mix_model.target_layer_ids)
        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0)
        output = token_mix_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        assert output.shape == (B, N, H)
        assert not torch.isnan(output).any()


class TestTokenMixGradients:
    def test_gradient_flows_to_mixer(self, token_mix_model, tiny_config):
        token_mix_model.train()
        # Make mixers non-zero so they contribute
        for layer in token_mix_model.layers:
            with torch.no_grad():
                layer.token_mixer.weight.fill_(0.01)

        B, T = 2, 16
        N = tiny_config.block_size * 2
        H = tiny_config.hidden_size
        num_target_layers = len(token_mix_model.target_layer_ids)
        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        output = token_mix_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        loss = output.sum()
        loss.backward()

        for layer in token_mix_model.layers:
            assert layer.token_mixer.weight.grad is not None
            assert (layer.token_mixer.weight.grad != 0).any()

    def test_gradient_flows_at_zero_init(self, token_mix_model, tiny_config):
        """Even at zero init, gradients should flow (grad doesn't depend on weight value)."""
        token_mix_model.train()
        B, T = 2, 16
        N = tiny_config.block_size * 2
        H = tiny_config.hidden_size
        num_target_layers = len(token_mix_model.target_layer_ids)
        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layers * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        output = token_mix_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        loss = output.sum()
        loss.backward()

        for layer in token_mix_model.layers:
            assert layer.token_mixer.weight.grad is not None
            assert (layer.token_mixer.weight.grad != 0).any()
