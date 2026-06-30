"""Correctness tests for DFlare"""

import math

import pytest
import torch
import torch.nn as nn

from midas.config import DFlareConfig
from midas.data import prepare_draft_inputs
from midas.models.model_dflare import (
    AdaptiveLayerFusion,
    DFlareDraftModel,
    Qwen3DFlareAttention,
    build_dflare_target_layer_ids,
    compute_progressive_gamma,
)


def _set_linear_identity(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        rows = min(linear.weight.shape)
        linear.weight[:rows, :rows] = torch.eye(rows, dtype=linear.weight.dtype)
        if linear.bias is not None:
            linear.bias.zero_()


class TestDFlareTargetLayerSelection:
    @pytest.mark.parametrize(
        "num_target_layers,num_fused,expected",
        [
            (8, 3, [1, 3, 5]),
            (12, 5, [1, 3, 5, 7, 9]),
            (32, 9, [1, 4, 8, 12, 15, 18, 22, 26, 29]),
        ],
    )
    def test_uniformly_selects_second_to_third_last_layers(
        self, num_target_layers, num_fused, expected
    ):
        assert build_dflare_target_layer_ids(num_target_layers, num_fused) == expected

    def test_model_selects_target_layers_from_original_depth(
        self, tiny_config, tiny_dflare_config
    ):
        tiny_config.num_hidden_layers = 2
        tiny_config.num_target_layers = 8

        model = DFlareDraftModel(tiny_config, dflare_config=tiny_dflare_config)

        assert model.config.num_hidden_layers == tiny_dflare_config.num_draft_layers
        assert model.target_layer_ids == [1, 3, 5]


class TestAdaptiveLayerFusion:
    def test_computes_softmax_weighted_sum_per_draft_layer_then_normalizes(self):
        cfg = DFlareConfig(hidden_size=4, num_draft_layers=2, num_target_layers=3)
        fusion = AdaptiveLayerFusion(cfg)
        layer_hiddens = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],
                [[[4.0, 3.0, 2.0, 1.0]]],
                [[[1.0, 3.0, 5.0, 7.0]]],
            ]
        )
        with torch.no_grad():
            fusion.fusion_logits.copy_(
                torch.tensor(
                    [
                        [0.0, math.log(2.0), math.log(3.0)],
                        [math.log(3.0), math.log(2.0), 0.0],
                    ]
                )
            )

        actual = fusion(layer_hiddens)
        weights = torch.softmax(fusion.fusion_logits, dim=-1)
        expected_raw = torch.einsum("dt,tblh->dblh", weights, layer_hiddens)
        expected = torch.stack(
            [fusion.output_norms[i](expected_raw[i]) for i in range(cfg.num_draft_layers)],
            dim=0,
        )

        assert torch.allclose(actual, expected, atol=1e-6)
        assert not torch.allclose(actual[0], actual[1])

    def test_rejects_wrong_number_of_target_layers(self):
        fusion = AdaptiveLayerFusion(
            DFlareConfig(hidden_size=4, num_draft_layers=2, num_target_layers=3)
        )

        with pytest.raises(ValueError, match="Expected 3 target layers"):
            fusion(torch.zeros(2, 1, 1, 4))


class TestDFlareHiddenExtraction:
    def test_extracts_exact_selected_target_layers_not_concatenated(self, dflare_model, tiny_config):
        B, T, H = 1, 2, tiny_config.hidden_size
        hidden_states = tuple(
            torch.full((B, T, H), fill_value=float(i))
            for i in range(tiny_config.num_target_layers + 1)
        )

        selected = dflare_model.extract_target_hidden(hidden_states)

        expected = torch.stack(
            [hidden_states[layer_id + 1] for layer_id in dflare_model.target_layer_ids],
            dim=0,
        )
        assert selected.shape == expected.shape
        assert torch.equal(selected, expected)
        assert selected.dim() == 4


class TestDFlareHeterogeneousKV:
    def test_disables_sliding_window_for_full_target_context(self, tiny_config):
        tiny_config.sliding_window = 2

        attn = Qwen3DFlareAttention(tiny_config, layer_idx=0)

        assert attn.sliding_window is None

    def test_attention_mask_matches_concatenated_target_and_block_keys(
        self, tiny_config
    ):
        attn = Qwen3DFlareAttention(tiny_config, layer_idx=0)
        captured = {}

        def fake_attn(_module, q, k, v, attention_mask, **_kwargs):
            captured["q_shape"] = q.shape
            captured["k_shape"] = k.shape
            captured["mask_shape"] = attention_mask.shape
            return torch.zeros(q.shape[0], q.shape[2], tiny_config.hidden_size), None

        attn.config._attn_implementation = "eager"
        import midas.models.model_dflare as model_dflare

        original = model_dflare.eager_attention_forward
        model_dflare.eager_attention_forward = fake_attn
        try:
            B, q_len, ctx_len, H = 2, 4, 7, tiny_config.hidden_size
            hidden_states = torch.randn(B, q_len, H)
            target_hidden = torch.randn(B, ctx_len, H)
            cos = torch.ones(B, ctx_len + q_len, tiny_config.head_dim)
            sin = torch.zeros(B, ctx_len + q_len, tiny_config.head_dim)
            attention_mask = torch.zeros(B, 1, q_len, ctx_len + q_len)
            attn(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask,
            )
        finally:
            model_dflare.eager_attention_forward = original

        assert captured["q_shape"][-2] == q_len
        assert captured["k_shape"][-2] == ctx_len + q_len
        assert captured["mask_shape"] == (B, 1, q_len, ctx_len + q_len)

    def test_rope_sees_full_key_positions_and_block_query_suffix(
        self, tiny_config, monkeypatch
    ):
        attn = Qwen3DFlareAttention(tiny_config, layer_idx=0)
        captured = {}

        def fake_rope(q, k, cos, sin):
            captured["q_len"] = q.size(-2)
            captured["k_len"] = k.size(-2)
            captured["cos_len"] = cos.size(-2)
            captured["sin_len"] = sin.size(-2)
            return q, k

        def fake_attn(_module, q, k, v, attention_mask, **_kwargs):
            return torch.zeros(q.shape[0], q.shape[2], tiny_config.hidden_size), None

        import midas.models.model_dflare as model_dflare

        monkeypatch.setattr(model_dflare, "apply_rotary_pos_emb", fake_rope)
        monkeypatch.setattr(model_dflare, "eager_attention_forward", fake_attn)

        B, q_len, ctx_len, H = 1, 3, 8, tiny_config.hidden_size
        hidden_states = torch.randn(B, q_len, H)
        target_hidden = torch.randn(B, ctx_len, H)
        cos = torch.ones(B, ctx_len + q_len, tiny_config.head_dim)
        sin = torch.zeros(B, ctx_len + q_len, tiny_config.head_dim)
        attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=(cos, sin),
            attention_mask=torch.zeros(B, 1, q_len, ctx_len + q_len),
        )

        assert captured == {
            "q_len": q_len,
            "k_len": ctx_len + q_len,
            "cos_len": ctx_len + q_len,
            "sin_len": ctx_len + q_len,
        }

    def test_target_context_and_block_tokens_use_separate_kv_projection_paths(self, tiny_config):
        tiny_config.num_attention_heads = 1
        tiny_config.num_key_value_heads = 1
        tiny_config.head_dim = tiny_config.hidden_size
        attn = Qwen3DFlareAttention(tiny_config, layer_idx=0)

        _set_linear_identity(attn.q_proj)
        _set_linear_identity(attn.k_proj)
        _set_linear_identity(attn.v_proj)
        _set_linear_identity(attn.target_k_proj)
        _set_linear_identity(attn.target_v_proj)
        _set_linear_identity(attn.o_proj)

        captured = {}

        def fake_attn(_module, q, k, v, attention_mask, **_kwargs):
            captured["k"] = k.detach().clone()
            captured["v"] = v.detach().clone()
            return torch.zeros(q.shape[0], q.shape[2], tiny_config.hidden_size), None

        attn.config._attn_implementation = "eager"
        import midas.models.model_dflare as model_dflare

        original = model_dflare.eager_attention_forward
        model_dflare.eager_attention_forward = fake_attn
        try:
            hidden_states = torch.randn(1, 2, tiny_config.hidden_size)
            target_hidden = torch.randn(1, 3, tiny_config.hidden_size)
            cos = torch.ones(1, 5, tiny_config.head_dim)
            sin = torch.zeros(1, 5, tiny_config.head_dim)
            attn(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                attention_mask=None,
            )
        finally:
            model_dflare.eager_attention_forward = original

        projected_target_k = attn.k_norm(
            attn.target_k_proj(target_hidden).view(1, 3, 1, tiny_config.head_dim)
        )
        projected_block_k = attn.k_norm(
            attn.k_proj(hidden_states).view(1, 2, 1, tiny_config.head_dim)
        )
        expected_k = torch.cat([projected_target_k, projected_block_k], dim=1).transpose(1, 2)
        expected_v = torch.cat(
            [
                attn.target_v_proj(target_hidden).view(1, 3, 1, tiny_config.head_dim),
                attn.v_proj(hidden_states).view(1, 2, 1, tiny_config.head_dim),
            ],
            dim=1,
        ).transpose(1, 2)

        assert torch.allclose(captured["k"], expected_k, atol=1e-6)
        assert torch.allclose(captured["v"], expected_v, atol=1e-6)


class TestDFlareForwardBehavior:
    def test_forward_uses_different_fused_context_for_each_draft_layer(
        self, tiny_config, tiny_dflare_config
    ):
        model = DFlareDraftModel(tiny_config, dflare_config=tiny_dflare_config)
        seen_contexts = []

        for layer in model.layers:
            def capture_layer(*, target_hidden, hidden_states, **_kwargs):
                seen_contexts.append(target_hidden.detach().clone())
                return hidden_states

            layer.forward = capture_layer

        with torch.no_grad():
            model.adaptive_layer_fusion.fusion_logits[0].copy_(
                torch.tensor([20.0, -20.0, -20.0])
            )
            model.adaptive_layer_fusion.fusion_logits[1].copy_(
                torch.tensor([-20.0, -20.0, 20.0])
            )

        B, N, T, H = 1, 2, 3, tiny_config.hidden_size
        target_hidden = torch.stack(
            [
                torch.arange(H, dtype=torch.float).view(1, 1, H).expand(B, T, H),
                torch.arange(H, 0, -1, dtype=torch.float).view(1, 1, H).expand(B, T, H),
                torch.remainder(torch.arange(H, dtype=torch.float), 7.0)
                .view(1, 1, H)
                .expand(B, T, H),
            ],
            dim=0,
        )
        model(
            position_ids=torch.arange(T + N).unsqueeze(0),
            noise_embedding=torch.zeros(B, N, H),
            target_hidden=target_hidden,
        )

        assert len(seen_contexts) == tiny_dflare_config.num_draft_layers
        assert not torch.allclose(seen_contexts[0], seen_contexts[1])


class TestDFlareTrainingBatchContract:
    def test_real_batch_contains_anchor_plus_masked_positions_and_sparse_mask(self):
        B, T, block_size = 1, 64, 4
        mask_token_id = 999
        input_ids = torch.arange(T).unsqueeze(0)
        batch = prepare_draft_inputs(
            input_ids,
            torch.tensor([T]),
            block_size=block_size,
            max_num_anchors=3,
            mask_token_id=mask_token_id,
            response_starts=torch.tensor([10]),
        )
        N = batch.noise_token_ids.shape[1]
        assert batch.attention_mask.shape == (B, 1, N, T + N)
        assert torch.equal(batch.position_ids[:, :T], torch.arange(T).unsqueeze(0))
        assert torch.equal(batch.position_ids[:, T:], batch.noise_original_positions)

        mask = batch.attention_mask[0, 0]
        for block_start in range(0, N, block_size):
            anchor_pos = batch.noise_original_positions[0, block_start].item()
            assert batch.noise_token_ids[0, block_start].item() == input_ids[
                0, anchor_pos
            ].item()
            assert torch.equal(
                batch.noise_token_ids[0, block_start + 1 : block_start + block_size],
                torch.full((block_size - 1,), mask_token_id),
            )

            block_rows = slice(block_start, block_start + block_size)
            assert torch.all(mask[block_rows, :anchor_pos] == 0)
            assert torch.all(mask[block_rows, anchor_pos:T] == float("-inf"))
            assert torch.all(mask[block_rows, T + block_start : T + block_start + block_size] == 0)

            before = mask[block_rows, T:block_start + T]
            after = mask[block_rows, T + block_start + block_size:]
            assert not before.numel() or torch.all(before == float("-inf"))
            assert not after.numel() or torch.all(after == float("-inf"))


class TestProgressivePositionWeights:
    def test_gamma_increases_linearly_by_epoch(self):
        cfg = DFlareConfig(gamma_initial=4.5, gamma_increment_per_epoch=1.0)
        assert [compute_progressive_gamma(cfg, epoch) for epoch in range(4)] == [
            4.5,
            5.5,
            6.5,
            7.5,
        ]
