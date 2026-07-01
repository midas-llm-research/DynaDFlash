"""Tests for midas/models/model_split_query.py — DFlashDraftModelSplitQuery."""

import pytest
import torch

from midas.models.model_split_query import (
    DFlashDraftModelSplitQuery,
    Qwen3DFlashSplitQueryAttention,
)
from midas.models.model import DFlashDraftModel
from tests.conftest import make_tiny_qwen3_config, MockLMHead


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config()


@pytest.fixture
def split_query_model(tiny_config):
    model = DFlashDraftModelSplitQuery(tiny_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestSplitQueryInit:
    def test_layers_have_split_query_attention(self, split_query_model):
        for layer in split_query_model.layers:
            assert isinstance(layer.self_attn, Qwen3DFlashSplitQueryAttention)

    def test_has_separate_q_projections(self, split_query_model):
        attn = split_query_model.layers[0].self_attn
        assert hasattr(attn, "q_proj_target")
        assert hasattr(attn, "q_proj_draft")
        assert not hasattr(attn, "q_proj")

    def test_q_projections_same_shape(self, split_query_model, tiny_config):
        attn = split_query_model.layers[0].self_attn
        expected_out = tiny_config.num_attention_heads * attn.head_dim
        assert attn.q_proj_target.weight.shape == (expected_out, tiny_config.hidden_size)
        assert attn.q_proj_draft.weight.shape == (expected_out, tiny_config.hidden_size)

    def test_has_standard_kv_projections(self, split_query_model):
        attn = split_query_model.layers[0].self_attn
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "o_proj")

    def test_inherits_target_fusion(self, split_query_model):
        assert hasattr(split_query_model, "fc")
        assert hasattr(split_query_model, "hidden_norm")


# ---------------------------------------------------------------------------
# Forward pass — shape test
# ---------------------------------------------------------------------------


class TestSplitQueryForward:
    def _run_forward(self, model, tiny_config):
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
        return output, B, N, T, H

    def test_output_shape(self, split_query_model, tiny_config):
        output, B, N, T, H = self._run_forward(split_query_model, tiny_config)
        assert output.shape == (B, N, H)

    def test_no_nans(self, split_query_model, tiny_config):
        output, *_ = self._run_forward(split_query_model, tiny_config)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_with_attention_mask(self, tiny_config):
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.eval()
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        attention_mask = torch.zeros(B, 1, N, T + N)

        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
        )
        assert output.shape == (B, N, H)
        assert not torch.isnan(output).any()


# ---------------------------------------------------------------------------
# Joint softmax test
# ---------------------------------------------------------------------------


class TestJointSoftmax:
    def test_attention_weights_sum_to_one(self, tiny_config):
        """Verify A_t.sum(-1) + A_d.sum(-1) ≈ 1 for every query token/head."""
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.eval()

        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        attention_mask = torch.zeros(B, 1, N, T + N)

        # Run attention directly to get weights
        fused_target = model._apply_target_hidden_fusion(target_hidden)
        position_embeddings = model.rotary_emb(noise_embedding, position_ids)
        hidden = model.layers[0].input_layernorm(noise_embedding)

        _, attn_weights = model.layers[0].self_attn(
            hidden_states=hidden,
            target_hidden=fused_target,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        # attn_weights is the full softmax output: [B, num_heads, q_len, total_kv_len]
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestSplitQueryGradients:
    def test_gradients_flow_to_both_q_projections(self, tiny_config):
        model = DFlashDraftModelSplitQuery(tiny_config)
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

        for i, layer in enumerate(model.layers):
            attn = layer.self_attn
            assert attn.q_proj_target.weight.grad is not None, \
                f"No gradient for layer {i} q_proj_target"
            assert attn.q_proj_target.weight.grad.abs().sum() > 0, \
                f"Zero gradient for layer {i} q_proj_target"
            assert attn.q_proj_draft.weight.grad is not None, \
                f"No gradient for layer {i} q_proj_draft"
            assert attn.q_proj_draft.weight.grad.abs().sum() > 0, \
                f"Zero gradient for layer {i} q_proj_draft"


# ---------------------------------------------------------------------------
# Backward compatibility — base model still works
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_base_model_unchanged(self, tiny_config):
        """DFlashDraftModel still uses the original single q_proj path."""
        model = DFlashDraftModel(tiny_config)
        model.eval()
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
        assert output.shape == (B, N, H)
        assert hasattr(model.layers[0].self_attn, "q_proj")


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


class TestSplitQueryLoss:
    def test_loss_computes(self, tiny_config):
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.train()

        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        V = tiny_config.vocab_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        draft_output = model(
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

        loss = model.compute_loss(draft_output, input_ids, batch, lm_head, gamma=7.0)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Log metrics
# ---------------------------------------------------------------------------


class TestSplitQueryLogMetrics:
    def test_includes_q_norm_metrics(self, split_query_model, tiny_config):
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(split_query_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        split_query_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        metrics = split_query_model.log_metrics(loss=0.5, lr=1e-4, epoch=0, step=10)
        assert "q_norm_target_L0" in metrics
        assert "q_norm_draft_L0" in metrics
        assert "q_norm_ratio_L0" in metrics
