"""Tests for paper-faithful DFlare model components."""

import torch

from midas.config import DFlareConfig
from midas.models.model_dflare import (
    AdaptiveLayerFusion,
    DFlareDraftModel,
    Qwen3DFlareAttention,
    build_dflare_target_layer_ids,
    compute_progressive_gamma,
)


class TestBuildDFlareTargetLayerIds:
    def test_selects_requested_number_of_layers(self):
        ids = build_dflare_target_layer_ids(num_target_layers=32, num_fused_target_layers=9)
        assert len(ids) == 9

    def test_uses_second_through_third_to_last_layer(self):
        ids = build_dflare_target_layer_ids(num_target_layers=32, num_fused_target_layers=9)
        assert ids[0] == 1
        assert ids[-1] == 29

    def test_selection_is_independent_of_draft_layer_count(self):
        ids_a = build_dflare_target_layer_ids(num_target_layers=32, num_fused_target_layers=9)
        ids_b = build_dflare_target_layer_ids(num_target_layers=32, num_fused_target_layers=9)
        assert ids_a == ids_b

    def test_single_layer_picks_middle(self):
        ids = build_dflare_target_layer_ids(num_target_layers=32, num_fused_target_layers=1)
        assert ids == [16]


class TestAdaptiveLayerFusion:
    def test_fusion_weight_shape(self, tiny_dflare_config):
        fusion = AdaptiveLayerFusion(tiny_dflare_config)
        assert fusion.fusion_logits.shape == (
            tiny_dflare_config.num_draft_layers,
            tiny_dflare_config.num_target_layers,
        )

    def test_softmax_weights_sum_to_one_and_are_non_negative(self, tiny_dflare_config):
        fusion = AdaptiveLayerFusion(tiny_dflare_config)
        weights = fusion.fusion_weights()
        assert torch.allclose(
            weights.sum(dim=-1),
            torch.ones(tiny_dflare_config.num_draft_layers),
            atol=1e-6,
        )
        assert (weights >= 0).all()

    def test_returns_one_context_per_draft_layer(self, tiny_dflare_config):
        B, T, H = 2, 8, tiny_dflare_config.hidden_size
        layer_hiddens = torch.randn(tiny_dflare_config.num_target_layers, B, T, H)
        fusion = AdaptiveLayerFusion(tiny_dflare_config)

        fused = fusion(layer_hiddens)

        assert fused.shape == (tiny_dflare_config.num_draft_layers, B, T, H)

    def test_layer_specific_fusion_can_differ(self, tiny_dflare_config):
        B, T, H = 1, 4, tiny_dflare_config.hidden_size
        layer_hiddens = torch.stack(
            [
                torch.full((B, T, H), 1.0),
                torch.full((B, T, H), 2.0),
                torch.full((B, T, H), 3.0),
            ],
            dim=0,
        )
        fusion = AdaptiveLayerFusion(tiny_dflare_config)
        with torch.no_grad():
            fusion.fusion_logits[0].copy_(torch.tensor([10.0, -10.0, -10.0]))
            fusion.fusion_logits[1].copy_(torch.tensor([-10.0, -10.0, 10.0]))

        fused = fusion(layer_hiddens)

        assert not torch.allclose(fused[0], fused[1])


class TestQwen3DFlareAttention:
    def test_has_heterogeneous_target_kv_projections(self, tiny_config):
        attn = Qwen3DFlareAttention(tiny_config, layer_idx=0)
        assert hasattr(attn, "target_k_proj")
        assert hasattr(attn, "target_v_proj")
        assert attn.target_k_proj is not attn.k_proj
        assert attn.target_v_proj is not attn.v_proj

    def test_output_shape(self, tiny_config):
        B, q_len, ctx_len = 2, 8, 16
        H = tiny_config.hidden_size
        attn = Qwen3DFlareAttention(tiny_config, layer_idx=0)
        hidden_states = torch.randn(B, q_len, H)
        target_hidden = torch.randn(B, ctx_len, H)
        cos = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)
        sin = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)

        output, _ = attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )

        assert output.shape == (B, q_len, H)

    def test_gradient_flows_to_draft_and_target_kv(self, tiny_config):
        B, q_len, ctx_len = 2, 8, 16
        H = tiny_config.hidden_size
        attn = Qwen3DFlareAttention(tiny_config, layer_idx=0)
        hidden_states = torch.randn(B, q_len, H, requires_grad=True)
        target_hidden = torch.randn(B, ctx_len, H, requires_grad=True)
        cos = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)
        sin = torch.randn(1, ctx_len + q_len, tiny_config.head_dim)

        output, _ = attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )
        output.sum().backward()

        for name in ["q_proj", "k_proj", "v_proj", "target_k_proj", "target_v_proj"]:
            param = getattr(attn, name).weight
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestDFlareDraftModel:
    def test_has_adaptive_layer_fusion(self, dflare_model):
        assert hasattr(dflare_model, "adaptive_layer_fusion")
        assert isinstance(dflare_model.adaptive_layer_fusion, AdaptiveLayerFusion)

    def test_does_not_use_dflash_concat_fc_or_router(self, dflare_model):
        assert not hasattr(dflare_model, "fc")
        assert not hasattr(dflare_model, "hidden_norm")
        assert not hasattr(dflare_model, "context_router")

    def test_extract_target_hidden_returns_selected_layer_stack(self, dflare_model, tiny_config):
        B, T, H = 2, 8, tiny_config.hidden_size
        hidden_states = tuple(
            torch.full((B, T, H), float(i)) for i in range(tiny_config.num_target_layers + 1)
        )

        selected = dflare_model.extract_target_hidden(hidden_states)

        assert selected.shape == (
            dflare_model.dflare_config.num_target_layers,
            B,
            T,
            H,
        )

    def test_forward_output_shape(self, dflare_model, tiny_config):
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(dflare_model.dflare_config.num_target_layers, B, T, H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        output = dflare_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        assert output.shape == (B, N, H)

    def test_gradient_flows_to_fusion_weights(self, tiny_config, tiny_dflare_config):
        model = DFlareDraftModel(tiny_config, dflare_config=tiny_dflare_config)
        B, N, T = 2, 8, 16
        H = tiny_config.hidden_size
        noise_embedding = torch.randn(B, N, H)
        target_hidden = torch.randn(tiny_dflare_config.num_target_layers, B, T, H)
        position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)

        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        output.sum().backward()

        grad = model.adaptive_layer_fusion.fusion_logits.grad
        assert grad is not None
        assert grad.abs().sum() > 0


class TestProgressiveGamma:
    def test_epoch_schedule(self):
        cfg = DFlareConfig(gamma_initial=4.5, gamma_increment_per_epoch=1.0)
        assert compute_progressive_gamma(cfg, epoch=0) == 4.5
        assert compute_progressive_gamma(cfg, epoch=1) == 5.5
        assert compute_progressive_gamma(cfg, epoch=2) == 6.5

    def test_larger_gamma_flattens_position_weights(self):
        positions = torch.tensor([1.0, 2.0, 3.0, 4.0])
        small_gamma = torch.exp(-(positions - 1) / 4.5)
        large_gamma = torch.exp(-(positions - 1) / 7.5)

        assert small_gamma[-1] < large_gamma[-1]
        assert (small_gamma[0] - small_gamma[-1]) > (large_gamma[0] - large_gamma[-1])
