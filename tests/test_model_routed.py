"""Tests for midas/models/model_context_router.py — DFlashDraftModelRouted."""

import torch
import pytest

from midas.config import RouterConfig
from midas.models.model_context_router import DFlashDraftModelRouted, ContextRouter
from midas.models.model import DFlashDraftModel
from tests.conftest import make_tiny_qwen3_config, MockLMHead


# ---------------------------------------------------------------------------
# DFlashDraftModelRouted initialization
# ---------------------------------------------------------------------------


class TestRoutedModelInit:
    def test_has_context_router(self, routed_model):
        assert hasattr(routed_model, "context_router")
        assert isinstance(routed_model.context_router, ContextRouter)

    def test_no_fc_or_hidden_norm(self, routed_model):
        assert not hasattr(routed_model, "fc")
        assert not hasattr(routed_model, "hidden_norm")

    def test_last_routing_weights_initially_none(self, routed_model):
        assert routed_model.last_routing_weights is None

    def test_router_config_matches(self, routed_model, tiny_router_config):
        router = routed_model.context_router
        assert router.num_layers == tiny_router_config.num_layers
        assert router.hidden_size == tiny_router_config.hidden_size
        assert router.rank == tiny_router_config.rank


# ---------------------------------------------------------------------------
# DFlashDraftModelRouted forward
# ---------------------------------------------------------------------------


class TestRoutedModelForward:
    def _run_forward(self, routed_model, tiny_config):
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(routed_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        output = routed_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        return output, B, N, T, H

    def test_output_shape(self, routed_model, tiny_config):
        output, B, N, T, H = self._run_forward(routed_model, tiny_config)
        assert output.shape == (B, N, H)

    def test_stores_routing_weights(self, routed_model, tiny_config):
        self._run_forward(routed_model, tiny_config)
        assert routed_model.last_routing_weights is not None

    def test_routing_weights_shape(self, routed_model, tiny_config, tiny_router_config):
        _, B, N, T, H = self._run_forward(routed_model, tiny_config)
        rw = routed_model.last_routing_weights
        assert rw.shape == (B, T, tiny_router_config.num_layers)

    def test_routing_weights_sum_to_one(self, routed_model, tiny_config):
        self._run_forward(routed_model, tiny_config)
        rw = routed_model.last_routing_weights
        sums = rw.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_routing_weights_non_negative(self, routed_model, tiny_config):
        self._run_forward(routed_model, tiny_config)
        rw = routed_model.last_routing_weights
        assert (rw >= 0).all()



class TestRoutedModelComputeLoss:
    def test_loss_matches_base(self, tiny_config, tiny_router_config):
        """Routed model loss should equal base CE loss (no entropy reg)."""
        routed = DFlashDraftModelRouted(tiny_config, router_config=tiny_router_config)
        routed.train()

        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        V = tiny_config.vocab_size
        num_target_layer_ids = len(routed.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        draft_output = routed(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        input_ids = torch.randint(0, V, (B, T))
        lm_head = MockLMHead(H, V)

        block_size = tiny_config.block_size
        num_anchors = N // block_size
        position_in_block = torch.tensor(list(range(block_size)) * num_anchors).unsqueeze(0).expand(B, -1)
        loss_mask = position_in_block > 0
        noise_original_positions = torch.clamp(
            torch.arange(N).unsqueeze(0).expand(B, -1) + 2, max=T - 1
        )

        from midas.data import DraftTrainingBatch
        batch = DraftTrainingBatch(
            noise_token_ids=torch.zeros(B, N, dtype=torch.long),
            position_ids=position_ids,
            attention_mask=torch.zeros(B, 1, N, T + N),
            noise_original_positions=noise_original_positions,
            loss_mask=loss_mask,
            position_in_block=position_in_block,
        )

        from midas.models.model import compute_position_weighted_ce_loss
        expected = compute_position_weighted_ce_loss(draft_output, input_ids, batch, lm_head, gamma=7.0)
        actual = routed.compute_loss(draft_output, input_ids, batch, lm_head, gamma=7.0)
        assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# Routed model log_metrics
# ---------------------------------------------------------------------------


class TestRoutedModelLogMetrics:
    def test_includes_router_fields(self, routed_model, tiny_config, tiny_router_config):
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(routed_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        routed_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        metrics = routed_model.log_metrics(loss=0.5, lr=1e-4, epoch=0, step=10)
        assert "routing_weight_layer_0" in metrics
        assert "gamma" in metrics
        assert "tau" in metrics
        for i in range(tiny_router_config.num_layers):
            assert f"routing_weight_layer_{i}" in metrics

    def test_includes_base_fields(self, routed_model, tiny_config):
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(routed_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        routed_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        metrics = routed_model.log_metrics(loss=0.5, lr=1e-4, epoch=0, step=10)
        assert metrics["loss"] == 0.5
        assert metrics["lr"] == 1e-4


# ---------------------------------------------------------------------------
# Gradient flow through router
# ---------------------------------------------------------------------------


class TestRoutedModelGradients:
    def test_gradient_flows_to_router(self, tiny_config, tiny_router_config):
        model = DFlashDraftModelRouted(tiny_config, router_config=tiny_router_config)
        model.train()

        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        loss = output.sum()
        loss.backward()

        router = model.context_router
        for name, param in router.named_parameters():
            assert param.grad is not None, f"No gradient for router.{name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for router.{name}"
