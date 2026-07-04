"""Tests for midas/models/model_split_query.py — DFlashDraftModelSplitQuery."""

import pytest
import torch
from transformers import DynamicCache

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

    def test_has_separate_q_norms(self, split_query_model):
        attn = split_query_model.layers[0].self_attn
        assert hasattr(attn, "q_norm_target")
        assert hasattr(attn, "q_norm_draft")
        assert attn.q_norm_target is not attn.q_norm_draft

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

    def test_no_split_modules_attribute(self):
        assert "Qwen3DFlashSplitQueryDecoderLayer" in DFlashDraftModelSplitQuery._no_split_modules


# ---------------------------------------------------------------------------
# Forward pass — shape and finiteness
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

    def test_with_real_attention_mask(self, tiny_config):
        """Test with a mask that has actual -inf entries (block-diagonal pattern)."""
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.eval()
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        # Build a mask: block-diagonal for noise, causal for context
        block_size = tiny_config.block_size
        num_blocks = N // block_size
        attention_mask = torch.full((B, 1, N, T + N), float("-inf"))
        # Context: each noise token can see all context positions
        attention_mask[..., :T] = 0.0
        # Noise: block-diagonal visibility
        for blk in range(num_blocks):
            s = blk * block_size
            e = s + block_size
            attention_mask[:, :, s:e, T + s:T + e] = 0.0

        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
        )
        assert output.shape == (B, N, H)
        assert not torch.isnan(output).any()

    def test_produces_different_output_from_base(self, tiny_config):
        """Verify the split-query model is NOT identical to base (different architecture)."""
        torch.manual_seed(42)
        base = DFlashDraftModel(tiny_config)
        base.eval()
        torch.manual_seed(42)
        split = DFlashDraftModelSplitQuery(tiny_config)
        split.eval()

        B, N, T = 1, 4, 8
        H = tiny_config.hidden_size
        num_target_layer_ids = len(base.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        out_base = base(position_ids=position_ids, noise_embedding=noise_embedding, target_hidden=target_hidden)
        out_split = split(position_ids=position_ids, noise_embedding=noise_embedding, target_hidden=target_hidden)
        # They share fc/hidden_norm/rotary weights from the same seed, but attention
        # layers are entirely different architectures, so outputs should differ.
        assert not torch.allclose(out_base, out_split, atol=1e-3)


# ---------------------------------------------------------------------------
# Gated mixing tests — verify gate mechanism works correctly
# ---------------------------------------------------------------------------


class TestGatedMixing:
    def test_gate_exists_and_correct_shape(self, tiny_config):
        model = DFlashDraftModelSplitQuery(tiny_config)
        attn = model.layers[0].self_attn
        assert hasattr(attn, "gate_proj")
        assert attn.gate_proj.weight.shape == (tiny_config.num_attention_heads, tiny_config.hidden_size)
        assert attn.gate_proj.bias.shape == (tiny_config.num_attention_heads,)

    def test_gate_zero_initialized(self, tiny_config):
        """Gate should start at sigmoid(0) = 0.5 (equal mix of target and draft)."""
        model = DFlashDraftModelSplitQuery(tiny_config)
        attn = model.layers[0].self_attn
        assert torch.allclose(attn.gate_proj.weight, torch.zeros_like(attn.gate_proj.weight))
        assert torch.allclose(attn.gate_proj.bias, torch.zeros_like(attn.gate_proj.bias))

    def test_forward_returns_none_attn_weights(self, tiny_config):
        """With SDPA backend, attention weights are not returned."""
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.eval()

        B, N, T = 1, 4, 8
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        attention_mask = torch.zeros(B, 1, N, T + N)

        fused_target = model._apply_target_hidden_fusion(target_hidden)
        position_embeddings = model.rotary_emb(noise_embedding, position_ids)
        hidden = model.layers[0].input_layernorm(noise_embedding)

        _, attn_weights = model.layers[0].self_attn(
            hidden_states=hidden,
            target_hidden=fused_target,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        assert attn_weights is None

    def test_gate_affects_output(self, tiny_config):
        """Modifying gate bias should change the output."""
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.eval()

        B, N, T = 1, 4, 8
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        out_neutral = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        ).clone()

        # Bias gate heavily toward target attention
        with torch.no_grad():
            model.layers[0].self_attn.gate_proj.bias.fill_(10.0)

        out_target_biased = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        assert not torch.allclose(out_neutral, out_target_biased, atol=1e-4)


# ---------------------------------------------------------------------------
# Gradient flow — comprehensive check
# ---------------------------------------------------------------------------


class TestSplitQueryGradients:
    def test_gradients_flow_to_all_parameters(self, tiny_config):
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

        # Check ALL trainable parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


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
        assert not isinstance(model.layers[0].self_attn, Qwen3DFlashSplitQueryAttention)


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
# Inference with DynamicCache
# ---------------------------------------------------------------------------


class TestSplitQueryInference:
    def test_forward_with_cache(self, tiny_config):
        """Verify the model works with DynamicCache (inference mode)."""
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.eval()

        B, N, T = 1, 4, 8
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(B, T, num_target_layer_ids * H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)
        cache = DynamicCache()

        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=cache,
            use_cache=True,
        )
        assert output.shape == (B, N, H)
        assert not torch.isnan(output).any()
        # Cache should have entries now
        assert cache.get_seq_length() == T + N

    def test_cache_crop_and_continue(self, tiny_config):
        """Simulate the inference loop: prefill → crop → next block."""
        model = DFlashDraftModelSplitQuery(tiny_config)
        model.eval()

        B, block_size = 1, tiny_config.block_size
        T = 12
        H = tiny_config.hidden_size
        num_target_layer_ids = len(model.target_layer_ids)

        # Step 1: "Prefill" — full context + first draft block
        noise_embedding_1 = torch.randn(B, block_size, H)
        target_hidden_1 = torch.randn(B, T, num_target_layer_ids * H)
        position_ids_1 = torch.arange(T + block_size).unsqueeze(0).expand(B, -1)
        cache = DynamicCache()

        output_1 = model(
            position_ids=position_ids_1,
            noise_embedding=noise_embedding_1,
            target_hidden=target_hidden_1,
            past_key_values=cache,
            use_cache=True,
        )
        assert output_1.shape == (B, block_size, H)
        assert cache.get_seq_length() == T + block_size

        # Simulate acceptance of 2 tokens: crop to T + 2
        acceptance = 2
        start = T + acceptance
        cache.crop(start)
        assert cache.get_seq_length() == start

        # Step 2: Next block — new target hidden (for accepted tokens) + new draft block
        new_ctx_len = acceptance + 1  # accepted tokens produce new target context
        noise_embedding_2 = torch.randn(B, block_size, H)
        target_hidden_2 = torch.randn(B, new_ctx_len, num_target_layer_ids * H)
        position_ids_2 = torch.arange(start, start + new_ctx_len + block_size).unsqueeze(0).expand(B, -1)

        output_2 = model(
            position_ids=position_ids_2,
            noise_embedding=noise_embedding_2,
            target_hidden=target_hidden_2,
            past_key_values=cache,
            use_cache=True,
        )
        assert output_2.shape == (B, block_size, H)
        assert not torch.isnan(output_2).any()
        assert cache.get_seq_length() == start + new_ctx_len + block_size


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
        assert "gate_mean_L0" in metrics
        assert metrics["q_norm_target_L0"] > 0
        assert metrics["q_norm_draft_L0"] > 0
        assert abs(metrics["gate_mean_L0"] - 0.5) < 0.01  # zero-init → sigmoid(0) = 0.5
