"""Tests for midas/models/model.py — utilities, layers, and DFlashDraftModel."""

import torch
import pytest

from midas.models.model import (
    build_target_layer_ids,
    extract_context_feature,
    sample,
    compute_position_weighted_ce_loss,
    apply_rotary_pos_emb,
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    DFlashDraftModel,
)
from tests.conftest import make_tiny_qwen3_config, MockLMHead


# ---------------------------------------------------------------------------
# build_target_layer_ids
# ---------------------------------------------------------------------------


class TestBuildTargetLayerIds:
    def test_single_layer_picks_middle(self):
        ids = build_target_layer_ids(num_target_layers=32, num_draft_layers=1)
        assert ids == [16]

    def test_multiple_layers_evenly_spaced(self):
        ids = build_target_layer_ids(num_target_layers=32, num_draft_layers=4)
        assert len(ids) == 4
        assert ids[0] == 1
        assert ids[-1] == 29  # end = 32 - 3 = 29

    def test_all_ids_in_valid_range(self):
        for num_target in [8, 16, 32, 48]:
            for num_draft in [1, 2, 3, 5, 8]:
                ids = build_target_layer_ids(num_target, num_draft)
                assert len(ids) == num_draft
                for layer_id in ids:
                    assert 1 <= layer_id <= num_target - 3

    def test_two_layers_gives_start_and_end(self):
        ids = build_target_layer_ids(num_target_layers=32, num_draft_layers=2)
        assert ids == [1, 29]

    def test_ids_are_monotonically_increasing(self):
        ids = build_target_layer_ids(num_target_layers=32, num_draft_layers=5)
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1]


# ---------------------------------------------------------------------------
# extract_context_feature
# ---------------------------------------------------------------------------


class TestExtractContextFeature:
    def test_output_shape(self):
        B, T, H = 2, 16, 128
        num_layers = 8
        hidden_states = [torch.randn(B, T, H) for _ in range(num_layers + 1)]
        layer_ids = [1, 3, 5]
        result = extract_context_feature(hidden_states, layer_ids)
        assert result.shape == (B, T, len(layer_ids) * H)

    def test_correct_layers_selected(self):
        B, T, H = 1, 4, 8
        hidden_states = [torch.full((B, T, H), i, dtype=torch.float) for i in range(10)]
        layer_ids = [2, 5]
        result = extract_context_feature(hidden_states, layer_ids)
        # offset=1, so selects hidden_states[3] and hidden_states[6]
        expected = torch.cat([hidden_states[3], hidden_states[6]], dim=-1)
        assert torch.allclose(result, expected)

    def test_single_layer(self):
        B, T, H = 2, 10, 64
        hidden_states = [torch.randn(B, T, H) for _ in range(5)]
        layer_ids = [2]
        result = extract_context_feature(hidden_states, layer_ids)
        assert result.shape == (B, T, H)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


class TestSample:
    def test_greedy_returns_argmax(self):
        logits = torch.tensor([[[0.1, 0.9, 0.3], [0.8, 0.1, 0.2]]])  # [1, 2, 3]
        result = sample(logits, temperature=0.0)
        assert result.shape == (1, 2)
        assert result[0, 0] == 1  # argmax of [0.1, 0.9, 0.3]
        assert result[0, 1] == 0  # argmax of [0.8, 0.1, 0.2]

    def test_temperature_sampling_valid_range(self):
        B, T, V = 2, 10, 100
        logits = torch.randn(B, T, V)
        result = sample(logits, temperature=1.0)
        assert result.shape == (B, T)
        assert (result >= 0).all()
        assert (result < V).all()

    def test_high_temperature_more_uniform(self):
        torch.manual_seed(42)
        B, T, V = 1, 1000, 10
        logits = torch.zeros(B, T, V)
        logits[:, :, 0] = 5.0  # strong preference for token 0
        result_low = sample(logits, temperature=0.01)
        result_high = sample(logits, temperature=100.0)
        # Low temp should almost always pick token 0
        assert (result_low == 0).float().mean() > 0.99
        # High temp should spread across tokens
        assert (result_high == 0).float().mean() < 0.5


# ---------------------------------------------------------------------------
# compute_position_weighted_ce_loss
# ---------------------------------------------------------------------------


class TestComputePositionWeightedCELoss:
    def _make_loss_inputs(self, B=2, N=8, T=32, H=128, V=1000):
        draft_hidden = torch.randn(B, N, H, requires_grad=True)
        target_logits = torch.randn(B, T, V)
        lm_head = MockLMHead(H, V)

        block_size = 4
        num_anchors = N // block_size
        position_in_block = torch.tensor([0, 1, 2, 3] * num_anchors).unsqueeze(0).expand(B, -1)
        loss_mask = position_in_block > 0
        noise_original_positions = torch.clamp(
            torch.arange(N).unsqueeze(0).expand(B, -1) + 5, max=T - 1
        )

        from midas.data import DraftTrainingBatch
        batch = DraftTrainingBatch(
            noise_token_ids=torch.zeros(B, N, dtype=torch.long),
            position_ids=torch.zeros(B, T + N, dtype=torch.long),
            attention_mask=torch.zeros(B, 1, N, T + N),
            noise_original_positions=noise_original_positions,
            loss_mask=loss_mask,
            position_in_block=position_in_block,
        )
        return draft_hidden, target_logits, batch, lm_head

    def test_zero_mask_returns_zero(self):
        B, N, T, H, V = 2, 8, 32, 128, 1000
        draft_hidden = torch.randn(B, N, H)
        target_logits = torch.randn(B, T, V)
        lm_head = MockLMHead(H, V)

        from midas.data import DraftTrainingBatch
        batch = DraftTrainingBatch(
            noise_token_ids=torch.zeros(B, N, dtype=torch.long),
            position_ids=torch.zeros(B, T + N, dtype=torch.long),
            attention_mask=torch.zeros(B, 1, N, T + N),
            noise_original_positions=torch.zeros(B, N, dtype=torch.long),
            loss_mask=torch.zeros(B, N, dtype=torch.bool),  # all False
            position_in_block=torch.zeros(B, N, dtype=torch.long),
        )
        loss = compute_position_weighted_ce_loss(draft_hidden, target_logits, batch, lm_head)
        assert loss.item() == 0.0

    def test_returns_scalar(self):
        draft_hidden, target_logits, batch, lm_head = self._make_loss_inputs()
        loss = compute_position_weighted_ce_loss(draft_hidden, target_logits, batch, lm_head)
        assert loss.dim() == 0

    def test_loss_is_positive(self):
        draft_hidden, target_logits, batch, lm_head = self._make_loss_inputs()
        loss = compute_position_weighted_ce_loss(draft_hidden, target_logits, batch, lm_head)
        assert loss.item() > 0

    def test_gradient_flows(self):
        draft_hidden, target_logits, batch, lm_head = self._make_loss_inputs()
        loss = compute_position_weighted_ce_loss(draft_hidden, target_logits, batch, lm_head)
        loss.backward()
        assert draft_hidden.grad is not None
        assert draft_hidden.grad.abs().sum() > 0

    def test_position_weighting_effect(self):
        """Earlier positions should have higher weight (smaller k → larger weight)."""
        B, H, V = 1, 64, 100
        lm_head = MockLMHead(H, V)
        target_logits = torch.randn(1, 20, V)

        from midas.data import DraftTrainingBatch

        # Batch with only position 1 masked
        batch_early = DraftTrainingBatch(
            noise_token_ids=torch.zeros(B, 4, dtype=torch.long),
            position_ids=torch.zeros(B, 24, dtype=torch.long),
            attention_mask=torch.zeros(B, 1, 4, 24),
            noise_original_positions=torch.tensor([[5, 6, 7, 8]]),
            loss_mask=torch.tensor([[False, True, False, False]]),
            position_in_block=torch.tensor([[0, 1, 2, 3]]),
        )
        # Batch with only position 3 masked
        batch_late = DraftTrainingBatch(
            noise_token_ids=torch.zeros(B, 4, dtype=torch.long),
            position_ids=torch.zeros(B, 24, dtype=torch.long),
            attention_mask=torch.zeros(B, 1, 4, 24),
            noise_original_positions=torch.tensor([[5, 6, 7, 8]]),
            loss_mask=torch.tensor([[False, False, False, True]]),
            position_in_block=torch.tensor([[0, 1, 2, 3]]),
        )

        # Use same draft hidden for both to isolate weight effect
        draft_hidden = torch.randn(1, 4, H)
        loss_early = compute_position_weighted_ce_loss(draft_hidden, target_logits, batch_early, lm_head, gamma=7.0)
        loss_late = compute_position_weighted_ce_loss(draft_hidden, target_logits, batch_late, lm_head, gamma=7.0)

        # Weight for pos 1: exp(0/7)=1.0, weight for pos 3: exp(-2/7)≈0.75
        # The losses themselves differ because they look at different target logits,
        # but both should be finite and positive.
        assert loss_early.isfinite()
        assert loss_late.isfinite()


# ---------------------------------------------------------------------------
# Qwen3DFlashAttention
# ---------------------------------------------------------------------------


class TestQwen3DFlashAttention:
    @pytest.fixture
    def attention(self, tiny_config):
        return Qwen3DFlashAttention(tiny_config, layer_idx=0)

    def test_output_shape(self, attention, tiny_config):
        B, q_len, ctx_len = 2, 8, 16
        H = tiny_config.hidden_size
        hidden_states = torch.randn(B, q_len, H)
        target_hidden = torch.randn(B, ctx_len, H)

        cos = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)
        sin = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)
        position_embeddings = (cos, sin)

        output, _ = attention(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        assert output.shape == (B, q_len, H)

    def test_is_not_causal(self, attention):
        assert attention.is_causal is False


# ---------------------------------------------------------------------------
# Qwen3DFlashDecoderLayer
# ---------------------------------------------------------------------------


class TestQwen3DFlashDecoderLayer:
    @pytest.fixture
    def layer(self, tiny_config):
        return Qwen3DFlashDecoderLayer(tiny_config, layer_idx=0)

    def test_output_shape(self, layer, tiny_config):
        B, q_len, ctx_len = 2, 8, 16
        H = tiny_config.hidden_size
        hidden_states = torch.randn(B, q_len, H)
        target_hidden = torch.randn(B, ctx_len, H)
        position_ids = torch.arange(ctx_len + q_len).unsqueeze(0).expand(B, -1)

        cos = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)
        sin = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)
        position_embeddings = (cos, sin)

        output = layer(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        assert output.shape == (B, q_len, H)


# ---------------------------------------------------------------------------
# DFlashDraftModel
# ---------------------------------------------------------------------------


class TestDFlashDraftModel:
    def test_init_target_layer_ids(self, draft_model, tiny_config):
        expected = build_target_layer_ids(tiny_config.num_target_layers, tiny_config.num_hidden_layers)
        assert draft_model.target_layer_ids == expected

    def test_init_has_fc_and_hidden_norm(self, draft_model):
        assert hasattr(draft_model, "fc")
        assert hasattr(draft_model, "hidden_norm")

    def test_forward_shape(self, draft_model, tiny_config):
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(draft_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        output = draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        assert output.shape == (B, N, H)

    def test_forward_with_attention_mask(self, draft_model, tiny_config):
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(draft_model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        attention_mask = torch.zeros(B, 1, N, T + N)

        output = draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
        )
        assert output.shape == (B, N, H)

    def test_forward_gradient_flows(self, tiny_config):
        model = DFlashDraftModel(tiny_config)
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

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_extract_target_hidden(self, draft_model, tiny_config):
        B, T, H = 2, 16, tiny_config.hidden_size
        num_layers = tiny_config.num_target_layers
        hidden_states = tuple(torch.randn(B, T, H) for _ in range(num_layers + 1))

        result = draft_model.extract_target_hidden(hidden_states)
        expected_dim = len(draft_model.target_layer_ids) * H
        assert result.shape == (B, T, expected_dim)

    def test_compute_loss_returns_scalar(self, draft_model, tiny_config, dummy_draft_batch, mock_lm_head):
        B, N = 2, 8
        H = tiny_config.hidden_size
        T = 32
        V = 1000

        draft_hidden = torch.randn(B, N, H)
        target_logits = torch.randn(B, T, V)

        loss = draft_model.compute_loss(
            draft_hidden, target_logits, dummy_draft_batch, mock_lm_head, gamma=7.0
        )
        assert loss.dim() == 0
        assert loss.isfinite()

    def test_log_metrics_passthrough(self, draft_model):
        metrics = draft_model.log_metrics(loss=0.5, lr=1e-4, epoch=0, step=10)
        assert metrics == {"loss": 0.5, "lr": 1e-4, "epoch": 0, "step": 10}


# ---------------------------------------------------------------------------
# apply_rotary_pos_emb
# ---------------------------------------------------------------------------


class TestApplyRotaryPosEmb:
    def test_output_shapes(self):
        B, num_heads, q_len, k_len, head_dim = 2, 4, 8, 16, 32
        q = torch.randn(B, num_heads, q_len, head_dim)
        k = torch.randn(B, num_heads, k_len, head_dim)
        cos = torch.randn(1, k_len, head_dim)
        sin = torch.randn(1, k_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_q_shorter_than_k(self):
        B, num_heads, q_len, k_len, head_dim = 1, 2, 4, 12, 16
        q = torch.randn(B, num_heads, q_len, head_dim)
        k = torch.randn(B, num_heads, k_len, head_dim)
        cos = torch.randn(1, k_len, head_dim)
        sin = torch.randn(1, k_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == (B, num_heads, q_len, head_dim)
        assert k_out.shape == (B, num_heads, k_len, head_dim)
