"""Tests for midas/models/model_skip.py — layer-skip (early-exit) draft model.

Heavy on explicit shape assertions: every annotated tensor shape in model_skip.py
is double-checked here so the annotations can't silently drift.
"""

import torch
import pytest
from transformers import DynamicCache

from midas.config import SkipConfig, BlockAggregationType
from midas.data import DraftTrainingBatch
from midas.latency import LatencyModel
from midas.models.model_skip import DFlashDraftModelSkip, BlockAggregator
from tests.conftest import make_tiny_qwen3_config, MockLMHead


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_skip_model(agg=BlockAggregationType.ATTENTION, num_hidden_layers=2, **skip_kwargs):
    cfg = make_tiny_qwen3_config(num_hidden_layers=num_hidden_layers)
    model = DFlashDraftModelSkip(cfg, SkipConfig(aggregation=agg, **skip_kwargs))
    return cfg, model


def make_train_inputs(cfg, model, B, N, T):
    """Inputs matching the training forward signature (context length T, N noise tokens)."""
    H = cfg.hidden_size
    n_ids = len(model.target_layer_ids)
    noise = torch.randn(B, N, H)                      # [B, N, H]
    target_hidden = torch.randn(B, T, n_ids * H)      # [B, T, n_ids*H]
    position_ids = torch.arange(T + N).unsqueeze(0).expand(B, -1)  # [B, T+N]
    return noise, target_hidden, position_ids


# ---------------------------------------------------------------------------
# BlockAggregator — EXP_DECAY mode (fixed, depth-independent weights)
# ---------------------------------------------------------------------------


class TestBlockAggregatorExpDecay:
    def _agg(self, gamma=7.0):
        return BlockAggregator(SkipConfig(aggregation=BlockAggregationType.EXP_DECAY, gamma=gamma),
                               hidden_size=16, num_layers=3)

    def test_weights_shape_single_block(self):
        agg = self._agg()
        block = torch.randn(2, 4, 16)                 # [B, bs, H]
        w = agg.weights(block, query=torch.randn(2, 16), layer_idx=0)
        assert w.shape == (4,)                         # depth/leading-independent -> [bs]

    def test_weights_shape_batched_blocks(self):
        agg = self._agg()
        block = torch.randn(2, 5, 4, 16)              # [B, A, bs, H]
        w = agg.weights(block, query=torch.randn(2, 5, 16), layer_idx=2)
        assert w.shape == (4,)                         # still [bs]

    def test_weights_sum_to_one_and_monotonic(self):
        agg = self._agg(gamma=7.0)
        block = torch.randn(1, 6, 16)                  # bs=6
        w = agg.weights(block, query=torch.randn(1, 16), layer_idx=0)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
        assert w[0] == 0.0                              # anchor (non-drafted) excluded
        # earlier DRAFTED positions weighted strictly more (decay from position 1)
        assert all(w[i] > w[i + 1] for i in range(1, len(w) - 1))

    def test_weights_ignore_query_and_layer(self):
        agg = self._agg()
        block = torch.randn(1, 4, 16)
        w0 = agg.weights(block, query=torch.randn(1, 16), layer_idx=0)
        w1 = agg.weights(block, query=torch.randn(1, 16) * 9, layer_idx=2)
        assert torch.allclose(w0, w1)                  # exp_decay depends on neither

    def test_aggregate_shapes(self):
        agg = self._agg()
        # single block: [B,bs,H] x [bs] -> [B,H]
        out = agg.aggregate(torch.randn(2, 4, 16), torch.ones(4) / 4)
        assert out.shape == (2, 16)
        # batched: [B,A,bs,H] x [bs] -> [B,A,H]
        out = agg.aggregate(torch.randn(2, 5, 4, 16), torch.ones(4) / 4)
        assert out.shape == (2, 5, 16)

    def test_aggregate_uniform_equals_mean(self):
        agg = self._agg()
        block = torch.randn(3, 4, 16)
        out = agg.aggregate(block, torch.ones(4) / 4)
        assert torch.allclose(out, block.mean(dim=-2), atol=1e-6)

    def test_aggregate_matches_manual_weighted_sum(self):
        agg = self._agg()
        block = torch.randn(2, 4, 8)                   # [B, bs, H]
        w = torch.tensor([0.4, 0.3, 0.2, 0.1])         # [bs]
        out = agg.aggregate(block, w)
        manual = (block * w[None, :, None]).sum(dim=1)  # [B, H]
        assert out.shape == (2, 8)
        assert torch.allclose(out, manual, atol=1e-6)

    def test_weights_exact_values(self):
        # gamma=2, bs=3: anchor excluded; drafts weighted exp(-(k-1)/2) for k=1,2.
        agg = self._agg(gamma=2.0)
        block = torch.zeros(1, 3, 16)                  # content irrelevant for exp_decay
        w = agg.weights(block, query=torch.zeros(1, 16), layer_idx=0)
        # raw = [0, exp(0), exp(-0.5)] = [0, 1, 0.60653]; normalized over the drafts
        raw = torch.tensor([0.0, 1.0, 0.60653])
        expected = raw / raw.sum()
        assert torch.allclose(w, expected, atol=1e-5)
        assert torch.allclose(w, torch.tensor([0.0, 0.6225, 0.3775]), atol=1e-3)

    def test_aggregate_exact_numeric(self):
        agg = self._agg()
        block = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [B=1, bs=2, H=2]
        w = torch.tensor([0.25, 0.75])                    # [bs]
        out = agg.aggregate(block, w)
        # 0.25*[1,2] + 0.75*[3,4] = [2.5, 3.5]
        assert torch.allclose(out, torch.tensor([[2.5, 3.5]]), atol=1e-6)


# ---------------------------------------------------------------------------
# BlockAggregator — ATTENTION mode (per-layer, target-hidden query)
# ---------------------------------------------------------------------------


class TestBlockAggregatorAttention:
    def _agg(self, num_layers=3, rank=8):
        return BlockAggregator(SkipConfig(aggregation=BlockAggregationType.ATTENTION, rank=rank),
                               hidden_size=16, num_layers=num_layers)

    def test_has_per_layer_projections(self):
        agg = self._agg(num_layers=3)
        assert len(agg.W_q) == 3 and len(agg.W_k) == 3
        # distinct module instances
        assert agg.W_q[0] is not agg.W_q[1]

    def test_weights_shape_single_block(self):
        agg = self._agg()
        block = torch.randn(2, 4, 16)                  # [B, bs, H]
        query = torch.randn(2, 16)                     # [B, H]
        w = agg.weights(block, query, layer_idx=0)
        assert w.shape == (2, 4)                        # [B, bs]
        assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5)

    def test_weights_shape_batched_blocks(self):
        agg = self._agg()
        block = torch.randn(2, 5, 4, 16)               # [B, A, bs, H]
        query = torch.randn(2, 5, 16)                  # [B, A, H]
        w = agg.weights(block, query, layer_idx=1)
        assert w.shape == (2, 5, 4)                     # [B, A, bs]
        assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-5)

    def test_weights_depend_on_query(self):
        torch.manual_seed(0)
        agg = self._agg()
        block = torch.randn(1, 4, 16)
        w_a = agg.weights(block, torch.randn(1, 16), layer_idx=0)
        w_b = agg.weights(block, torch.randn(1, 16), layer_idx=0)
        assert not torch.allclose(w_a, w_b)            # query is actually used

    def test_weights_depend_on_layer_idx(self):
        torch.manual_seed(0)
        agg = self._agg()
        block = torch.randn(1, 4, 16)
        query = torch.randn(1, 16)
        w0 = agg.weights(block, query, layer_idx=0)
        w1 = agg.weights(block, query, layer_idx=1)
        assert not torch.allclose(w0, w1)              # per-layer projections differ

    def test_weights_exact_with_identity_projections(self):
        # With W_q = W_k = I (H=r=2), scores = q·k / sqrt(2); verify the exact softmax.
        agg = BlockAggregator(
            SkipConfig(aggregation=BlockAggregationType.ATTENTION, rank=2),
            hidden_size=2, num_layers=1,
        )
        with torch.no_grad():
            agg.W_q[0].weight.copy_(torch.eye(2))      # Linear weight is [out=r, in=H]
            agg.W_k[0].weight.copy_(torch.eye(2))
        query = torch.tensor([[1.0, 0.0]])             # [B=1, H=2]
        block = torch.tensor([[[1.0, 0.0],            # token 0 aligns with query
                               [0.0, 1.0]]])           # [B=1, bs=2, H=2]
        w = agg.weights(block, query, layer_idx=0)
        # q=[1,0], k=[[1,0],[0,1]] -> raw scores [1,0] -> /sqrt(2) -> softmax
        expected = torch.softmax(torch.tensor([[1.0, 0.0]]) / (2 ** 0.5), dim=-1)
        assert w.shape == (1, 2)
        assert torch.allclose(w, expected, atol=1e-6)
        assert w[0, 0] > w[0, 1]                        # more weight on the aligned token


# ---------------------------------------------------------------------------
# _greedy_accept_lengths — pure helper, hand-built tensors
# ---------------------------------------------------------------------------


class TestGreedyAcceptLengths:
    def test_known_acceptance_per_layer(self):
        # B=1, bs=4, A=1, N=4. anchor at index 0 (drafted=False).
        bs = 4
        drafted = torch.tensor([[False, True, True, True]])     # [B, N]
        gold = torch.tensor([[0, 1, 2, 3]])                      # [B, N]
        preds = torch.tensor([
            [[9, 1, 2, 9]],   # L0: pos1,2 correct, pos3 wrong -> run 2 -> accept 3
            [[0, 1, 2, 3]],   # L1: all correct -> run 3 -> accept 4
            [[0, 9, 2, 3]],   # L2: pos1 wrong -> run 0 -> accept 1
        ])                                                       # [L, B, N]
        accept = DFlashDraftModelSkip._greedy_accept_lengths(preds, gold, drafted, bs)
        assert accept.shape == (3, 1, 1)                         # [L, B, A]
        assert accept.squeeze(-1).squeeze(-1).tolist() == [3.0, 4.0, 1.0]

    def test_middle_miss_truncates_run(self):
        bs = 4
        drafted = torch.tensor([[False, True, True, True]])
        gold = torch.tensor([[0, 1, 2, 3]])
        # pos1 correct, pos2 wrong, pos3 correct -> cumprod stops at pos2 -> run 1 -> accept 2
        preds = torch.tensor([[[0, 1, 9, 3]]])                   # [1, 1, 4]
        accept = DFlashDraftModelSkip._greedy_accept_lengths(preds, gold, drafted, bs)
        assert accept.squeeze().item() == 2.0

    def test_anchor_prediction_ignored(self):
        bs = 4
        drafted = torch.tensor([[False, True, True, True]])
        gold = torch.tensor([[5, 1, 2, 3]])
        # anchor (index 0) prediction is wrong but masked out; drafted all correct -> accept 4
        preds = torch.tensor([[[9, 1, 2, 3]]])
        accept = DFlashDraftModelSkip._greedy_accept_lengths(preds, gold, drafted, bs)
        assert accept.squeeze().item() == 4.0

    def test_multiple_blocks(self):
        # N=8, bs=4 -> A=2 contiguous blocks
        bs = 4
        drafted = torch.tensor([[False, True, True, True, False, True, True, True]])  # [1, 8]
        gold = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        preds = torch.tensor([[[0, 1, 2, 3, 0, 9, 2, 3]]])       # block0 all correct; block1 pos1 wrong
        accept = DFlashDraftModelSkip._greedy_accept_lengths(preds, gold, drafted, bs)
        assert accept.shape == (1, 1, 2)                         # [L, B, A]
        assert accept[0, 0].tolist() == [4.0, 1.0]


# ---------------------------------------------------------------------------
# _exit_labels — pure helper, controlled LatencyModel
# ---------------------------------------------------------------------------


class TestExitLabels:
    @staticmethod
    def _linear_depth_latency():
        # draft_latency(depth, ctx) = depth ; verify_cost = 0
        return LatencyModel(d0=0.0, d_depth=1.0, d_ctx=0.0, d_dc=0.0, v0=0.0, v_ctx=0.0)

    @staticmethod
    def _constant_latency():
        # draft_latency = 1 ; verify_cost = 0  (latency identical across depths)
        return LatencyModel(d0=1.0, d_depth=0.0, d_ctx=0.0, d_dc=0.0, v0=0.0, v_ctx=0.0)

    def test_nonpositive_latency_is_clamped_no_nan(self):
        # Unconstrained fits can extrapolate to lat<=0; labels must stay finite/binary.
        accept = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
        ctx = torch.tensor([[100.0]])
        # draft_latency = -depth (negative), verify = 0  -> lat<=0 before the clamp
        lm = LatencyModel(d0=0.0, d_depth=-1.0, d_ctx=0.0, d_dc=0.0, v0=0.0, v_ctx=0.0)
        labels, lstar = DFlashDraftModelSkip._exit_labels(accept, ctx, lm)
        assert labels.shape == (3, 1, 1)
        assert torch.isfinite(labels).all()
        assert ((labels == 0) | (labels == 1)).all()
        assert torch.isfinite(lstar.float()).all()

    def test_deeper_is_better_picks_deepest(self):
        accept = torch.tensor([1.0, 2.0, 4.0]).view(3, 1, 1)     # [L, B, A], grows with depth
        ctx = torch.tensor([[5.0]])                               # [B, A]
        labels, lstar = DFlashDraftModelSkip._exit_labels(accept, ctx, self._linear_depth_latency())
        # score = accept/depth = [1, 1, 1.333] -> argmax = 2
        assert labels.shape == (3, 1, 1) and lstar.shape == (1, 1)
        assert lstar.item() == 2
        assert labels.squeeze().tolist() == [0.0, 0.0, 1.0]      # 0 before l*, 1 from l*

    def test_flat_accept_increasing_latency_exits_first(self):
        accept = torch.tensor([2.0, 2.0, 2.0]).view(3, 1, 1)
        ctx = torch.tensor([[5.0]])
        labels, lstar = DFlashDraftModelSkip._exit_labels(accept, ctx, self._linear_depth_latency())
        # score = 2/[1,2,3] decreasing -> shallowest best
        assert lstar.item() == 0
        assert labels.squeeze().tolist() == [1.0, 1.0, 1.0]      # exit immediately

    def test_ties_favor_shallower(self):
        accept = torch.tensor([2.0, 2.0]).view(2, 1, 1)
        ctx = torch.tensor([[5.0]])
        labels, lstar = DFlashDraftModelSkip._exit_labels(accept, ctx, self._constant_latency())
        # equal scores -> argmax returns first index
        assert lstar.item() == 0
        assert labels.squeeze().tolist() == [1.0, 1.0]

    def test_per_block_optimal_depth(self):
        # B=1, A=2: block0 deepens, block1 flat
        accept = torch.tensor([[[1.0, 3.0]], [[4.0, 3.0]]])       # [L=2, B=1, A=2]
        ctx = torch.tensor([[5.0, 5.0]])
        labels, lstar = DFlashDraftModelSkip._exit_labels(accept, ctx, self._linear_depth_latency())
        # block0: score=[1, 2] -> l*=1 ; block1: score=[3, 1.5] -> l*=0
        assert lstar.shape == (1, 2)
        assert lstar[0].tolist() == [1, 0]
        assert labels.shape == (2, 1, 2)
        assert labels[:, 0, 0].tolist() == [0.0, 1.0]            # block0: keep, then exit
        assert labels[:, 0, 1].tolist() == [1.0, 1.0]            # block1: exit immediately


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class TestSkipConstruction:
    def test_gates_count_and_init(self):
        cfg, model = make_skip_model(num_hidden_layers=2)
        assert len(model.gates) == cfg.num_hidden_layers
        for g in model.gates:
            assert g.weight.shape == (1, cfg.hidden_size)        # Linear(H, 1)
            assert torch.all(g.weight == 0)                       # zero weight ...
            assert torch.allclose(g.bias, torch.tensor(-10.0))    # ... and never-exit bias

    def test_attention_aggregator_has_projections(self):
        _, model = make_skip_model(BlockAggregationType.ATTENTION, num_hidden_layers=3)
        assert len(model.aggregator.W_q) == 3 and len(model.aggregator.W_k) == 3

    def test_exp_decay_aggregator_has_no_projections(self):
        _, model = make_skip_model(BlockAggregationType.EXP_DECAY)
        assert not hasattr(model.aggregator, "W_q")

    def test_defaults(self):
        _, model = make_skip_model(exit_threshold=0.5)
        assert model.enable_early_exit is True
        assert model.exit_threshold == 0.5
        assert isinstance(model.latency_model, LatencyModel)


# ---------------------------------------------------------------------------
# Training forward
# ---------------------------------------------------------------------------


class TestSkipTrainingForward:
    @pytest.mark.parametrize("agg", [BlockAggregationType.EXP_DECAY, BlockAggregationType.ATTENTION])
    def test_shapes_and_stash(self, agg):
        cfg, model = make_skip_model(agg, num_hidden_layers=2)
        model.train()
        B, N, T = 2, 8, 32
        noise, target_hidden, position_ids = make_train_inputs(cfg, model, B, N, T)
        out = model(position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden)
        H = cfg.hidden_size
        assert out.shape == (B, N, H)
        assert len(model._layer_hiddens) == cfg.num_hidden_layers
        for h in model._layer_hiddens:
            assert h.shape == (B, N, H)
        assert model._fused.shape == (B, T, H)                    # fused target hidden stashed

    def test_training_runs_all_layers_despite_gate(self):
        cfg, model = make_skip_model(num_hidden_layers=2)
        with torch.no_grad():
            model.gates[0].bias.fill_(50.0)  # would force exit at inference
        model.train()                         # but training must run every layer
        noise, target_hidden, position_ids = make_train_inputs(cfg, model, 1, 8, 16)
        model(position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden)
        assert len(model._layer_hiddens) == cfg.num_hidden_layers


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


class TestSkipComputeLoss:
    def _run(self, agg, dummy_draft_batch, gate_loss_weight=1.0):
        torch.manual_seed(0)
        cfg, model = make_skip_model(agg, num_hidden_layers=2, gate_loss_weight=gate_loss_weight)
        model.train()
        B, N = dummy_draft_batch.noise_token_ids.shape
        T, H = 32, cfg.hidden_size
        noise, target_hidden, position_ids = make_train_inputs(cfg, model, B, N, T)
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        lm_head = MockLMHead(H, cfg.vocab_size)
        out = model(position_ids=position_ids, noise_embedding=noise,
                    target_hidden=target_hidden, attention_mask=dummy_draft_batch.attention_mask)
        loss = model.compute_loss(out, input_ids, dummy_draft_batch, lm_head, gamma=7.0)
        return model, loss, lm_head

    @pytest.mark.parametrize("agg", [BlockAggregationType.EXP_DECAY, BlockAggregationType.ATTENTION])
    def test_scalar_positive_and_metrics(self, agg, dummy_draft_batch):
        model, loss, _ = self._run(agg, dummy_draft_batch)
        assert loss.dim() == 0 and loss.item() > 0
        for key in ("ce_loss", "gate_loss", "mean_optimal_depth"):
            assert key in model._last_metrics
        assert model._last_metrics["mean_optimal_depth"] == model._last_metrics["mean_optimal_depth"]  # not NaN

    def test_grad_flows_to_gates_and_backbone(self, dummy_draft_batch):
        model, loss, lm_head = self._run(BlockAggregationType.ATTENTION, dummy_draft_batch)
        loss.backward()
        # gate BCE -> gates + per-layer W_q
        assert sum(p.grad.abs().sum().item() for p in model.gates.parameters() if p.grad is not None) > 0
        assert all(wq.weight.grad is not None for wq in model.aggregator.W_q)
        # multi-layer CE -> backbone + lm_head
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layers.parameters())
        assert lm_head.linear.weight.grad is not None

    def test_gate_loss_weight_zero_is_pure_ce(self, dummy_draft_batch):
        model, loss, _ = self._run(BlockAggregationType.ATTENTION, dummy_draft_batch, gate_loss_weight=0.0)
        # with weight 0, returned loss == multi-layer CE only
        assert abs(loss.item() - model._last_metrics["ce_loss"]) < 1e-5


# ---------------------------------------------------------------------------
# Inference early exit + KV backfill
# ---------------------------------------------------------------------------


class TestSkipInferenceEarlyExit:
    def _infer(self, model, cfg, ctx, gate_setup=None):
        H, bs = cfg.hidden_size, model.block_size
        n_ids = len(model.target_layer_ids)
        if gate_setup is not None:
            with torch.no_grad():
                gate_setup(model)
        model.eval()
        cache = DynamicCache()
        out = model(
            position_ids=torch.arange(ctx + bs).unsqueeze(0),
            noise_embedding=torch.randn(1, bs, H),
            target_hidden=torch.randn(1, ctx, n_ids * H),
            past_key_values=cache,
            use_cache=True,
        )
        return out, cache

    def test_exit_at_first_layer_backfills_all(self):
        cfg, model = make_skip_model(num_hidden_layers=4)
        ctx, bs = 6, model.block_size
        out, cache = self._infer(model, cfg, ctx, lambda m: m.gates[0].bias.fill_(50.0))
        assert out.shape == (1, bs, cfg.hidden_size)
        assert model.last_exit_depth == 1
        assert len(cache.layers) == cfg.num_hidden_layers
        # layer 0 ran (cached ctx + noise); layers 1..3 backfilled (ctx only)
        assert cache.layers[0].keys.shape[2] == ctx + bs
        for i in range(1, cfg.num_hidden_layers):
            assert cache.layers[i].keys.shape[2] == ctx

    @pytest.mark.parametrize("exit_layer", [0, 1, 2])
    def test_cache_lengths_equal_after_dflash_crop(self, exit_layer):
        # Integration invariant: dflash_generate runs the draft forward then
        # crop(get_seq_length() - block_size) each block. After an early exit (which
        # backfills deep layers' context), that crop must leave EVERY layer's cache
        # the same length, or future deeper blocks read a misaligned cache.
        cfg, model = make_skip_model(num_hidden_layers=4)
        ctx, bs = 6, model.block_size
        out, cache = self._infer(model, cfg, ctx, lambda m: m.gates[exit_layer].bias.fill_(50.0))
        assert model.last_exit_depth == exit_layer + 1
        # mirror dflash_generate's per-block crop (model.py)
        cache.crop(cache.get_seq_length() - bs)
        lengths = [layer.keys.shape[2] for layer in cache.layers]
        assert len(set(lengths)) == 1, f"per-layer cache lengths diverged: {lengths}"
        assert lengths[0] == ctx                      # this block's context retained, noise dropped

    def test_exit_at_middle_layer(self):
        cfg, model = make_skip_model(num_hidden_layers=4)
        ctx, bs = 6, model.block_size

        def setup(m):
            m.gates[1].bias.fill_(50.0)  # exit right after layer 1
        _, cache = self._infer(model, cfg, ctx, setup)
        assert model.last_exit_depth == 2
        # layers 0,1 ran (ctx+bs); layers 2,3 backfilled (ctx)
        assert cache.layers[0].keys.shape[2] == ctx + bs
        assert cache.layers[1].keys.shape[2] == ctx + bs
        assert cache.layers[2].keys.shape[2] == ctx
        assert cache.layers[3].keys.shape[2] == ctx

    def test_no_exit_runs_all_layers(self):
        cfg, model = make_skip_model(num_hidden_layers=3)  # default gate bias -10 => never exit
        _, cache = self._infer(model, cfg, ctx=6)
        assert model.last_exit_depth == cfg.num_hidden_layers
        # every layer ran the full forward (ctx + noise), none backfilled
        for layer in cache.layers:
            assert layer.keys.shape[2] == 6 + model.block_size

    def test_threshold_blocks_exit(self):
        cfg, model = make_skip_model(num_hidden_layers=3, exit_threshold=100.0)
        # gate logit 50 < threshold 100 => should NOT exit
        self._infer(model, cfg, ctx=6, gate_setup=lambda m: m.gates[0].bias.fill_(50.0))
        assert model.last_exit_depth == cfg.num_hidden_layers

    def test_disable_early_exit_runs_dense(self):
        cfg, model = make_skip_model(num_hidden_layers=3)
        with torch.no_grad():
            model.gates[0].bias.fill_(50.0)  # would exit, but...
        model.enable_early_exit = False       # ...forced dense
        model.eval()
        H, bs, ctx = cfg.hidden_size, model.block_size, 6
        n_ids = len(model.target_layer_ids)
        out = model(
            position_ids=torch.arange(ctx + bs).unsqueeze(0),
            noise_embedding=torch.randn(1, bs, H),
            target_hidden=torch.randn(1, ctx, n_ids * H),
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        assert out.shape == (1, bs, H)
        assert len(model._layer_hiddens) == cfg.num_hidden_layers  # all-layers path taken


# ---------------------------------------------------------------------------
# Numeric reference checks: decent-sized inputs vs independent (loop) oracles
# ---------------------------------------------------------------------------


def ref_exp_decay_weights(bs, gamma):
    """Reference: anchor (index 0) excluded; drafts weighted exp(-(k-1)/gamma) -> [bs]."""
    w = torch.exp(-(torch.arange(bs, dtype=torch.float32) - 1) / gamma)
    w[0] = 0.0
    return w / w.sum()


def ref_attention_weights(block, query, Wq, Wk, r):
    """Reference attention pool weights from the mathematical definition."""
    q = query @ Wq.t()                              # [*, r]
    k = block @ Wk.t()                              # [*, bs, r]
    scores = (q.unsqueeze(-2) * k).sum(-1) / (r ** 0.5)  # [*, bs]
    return torch.softmax(scores, dim=-1)


def ref_aggregate(block, w):
    """Reference weighted sum over the block axis, via an explicit loop."""
    bs, H = block.shape[-2], block.shape[-1]
    out = torch.zeros(*block.shape[:-2], H, dtype=block.dtype)
    for s in range(bs):
        ws = w[s] if w.dim() == 1 else w[..., s].unsqueeze(-1)
        out = out + block[..., s, :] * ws
    return out


def ref_greedy_accept(preds, gold, drafted, bs):
    """Reference acceptance length via explicit per-(layer,block) loops."""
    L, B, N = preds.shape
    A = N // bs
    out = torch.zeros(L, B, A)
    for l in range(L):
        for b in range(B):
            for a in range(A):
                run = 0
                for k in range(1, bs):              # drafted positions 1..bs-1
                    idx = a * bs + k
                    if bool(drafted[b, idx]) and int(preds[l, b, idx]) == int(gold[b, idx]):
                        run += 1
                    else:
                        break
                out[l, b, a] = run + 1               # +1 for the anchor
    return out


def ref_exit_labels(accept, ctx, lm):
    """Reference exit labels via explicit per-block argmax over the speedup score."""
    L, B, A = accept.shape
    labels = torch.zeros(L, B, A)
    lstar = torch.zeros(B, A, dtype=torch.long)
    for b in range(B):
        for a in range(A):
            c = float(ctx[b, a])
            scores = [float(accept[l, b, a]) / (lm.draft_latency(l + 1, c) + lm.verify_cost(c))
                      for l in range(L)]
            best = int(torch.tensor(scores).argmax())
            lstar[b, a] = best
            for l in range(L):
                labels[l, b, a] = 1.0 if l >= best else 0.0
    return labels, lstar


class TestNumericReference:
    def test_exp_decay_weights_vs_reference(self):
        agg = BlockAggregator(SkipConfig(aggregation=BlockAggregationType.EXP_DECAY, gamma=3.0),
                              hidden_size=32, num_layers=4)
        block = torch.randn(3, 4, 8, 32)               # [B, A, bs=8, H]
        w = agg.weights(block, query=torch.randn(3, 4, 32), layer_idx=2)
        assert w.shape == (8,)
        assert torch.allclose(w, ref_exp_decay_weights(8, 3.0), atol=1e-6)

    def test_attention_weights_vs_reference(self):
        torch.manual_seed(1)
        B, A, bs, H, r = 3, 4, 6, 8, 4
        agg = BlockAggregator(SkipConfig(aggregation=BlockAggregationType.ATTENTION, rank=r),
                              hidden_size=H, num_layers=3)
        block = torch.randn(B, A, bs, H)               # [B, A, bs, H]
        query = torch.randn(B, A, H)                   # [B, A, H]
        layer = 2
        w = agg.weights(block, query, layer_idx=layer)
        expected = ref_attention_weights(block, query, agg.W_q[layer].weight, agg.W_k[layer].weight, r)
        assert w.shape == (B, A, bs)
        assert torch.allclose(w.sum(-1), torch.ones(B, A), atol=1e-5)
        assert torch.allclose(w, expected, atol=1e-5)

    def test_aggregate_vs_reference_batched(self):
        torch.manual_seed(2)
        B, A, bs, H = 3, 4, 6, 8
        agg = BlockAggregator(SkipConfig(aggregation=BlockAggregationType.EXP_DECAY),
                              hidden_size=H, num_layers=2)
        block = torch.randn(B, A, bs, H)
        # per-block weights [B, A, bs] that sum to 1
        w = torch.softmax(torch.randn(B, A, bs), dim=-1)
        out = agg.aggregate(block, w)
        assert out.shape == (B, A, H)
        assert torch.allclose(out, ref_aggregate(block, w), atol=1e-5)

    def test_greedy_accept_vs_reference(self):
        torch.manual_seed(3)
        L, B, A, bs, vocab = 5, 3, 4, 6, 4             # small vocab -> frequent matches
        N = A * bs
        preds = torch.randint(0, vocab, (L, B, N))
        gold = torch.randint(0, vocab, (B, N))
        # standard drafted pattern: False at each block's anchor (index 0), True elsewhere
        pib = torch.arange(N) % bs
        drafted = (pib > 0).unsqueeze(0).expand(B, -1).contiguous()
        accept = DFlashDraftModelSkip._greedy_accept_lengths(preds, gold, drafted, bs)
        assert accept.shape == (L, B, A)
        assert torch.equal(accept, ref_greedy_accept(preds, gold, drafted, bs))

    def test_exit_labels_vs_reference(self):
        torch.manual_seed(4)
        L, B, A = 5, 3, 4
        accept = torch.randint(1, 10, (L, B, A)).float()
        ctx = torch.randint(10, 500, (B, A)).float()
        lm = LatencyModel(d0=0.1, d_depth=0.05, d_ctx=1e-3, d_dc=2e-5, v0=1.0, v_ctx=2e-3)
        labels, lstar = DFlashDraftModelSkip._exit_labels(accept, ctx, lm)
        ref_labels, ref_lstar = ref_exit_labels(accept, ctx, lm)
        assert labels.shape == (L, B, A) and lstar.shape == (B, A)
        assert torch.equal(lstar, ref_lstar)
        assert torch.equal(labels, ref_labels)


# ---------------------------------------------------------------------------
# Sentinel (padding) block handling
# ---------------------------------------------------------------------------


class TestSentinelHandling:
    """Sentinel blocks share position_in_block with real blocks; only loss_mask
    distinguishes them. The gate loss must drop them via loss_mask, not drafted."""

    def _setup(self, num_hidden_layers=2):
        torch.manual_seed(0)
        cfg, model = make_skip_model(BlockAggregationType.ATTENTION, num_hidden_layers=num_hidden_layers)
        model.train()
        B, bs, A, T = 1, model.block_size, 2, 32
        N = A * bs
        noise, target_hidden, position_ids = make_train_inputs(cfg, model, B, N, T)
        out = model(position_ids=position_ids, noise_embedding=noise, target_hidden=target_hidden)
        lm_head = MockLMHead(cfg.hidden_size, cfg.vocab_size)
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        pib = (torch.arange(N) % bs).unsqueeze(0)        # [1, N] shared by all blocks
        return cfg, model, out, lm_head, input_ids, pib, B, bs, A, N, T, position_ids

    def _batch(self, position_ids, pib, loss_mask, nop, B, N, T):
        return DraftTrainingBatch(
            noise_token_ids=torch.zeros(B, N, dtype=torch.long),
            position_ids=position_ids,
            attention_mask=torch.zeros(B, 1, N, T + N),
            noise_original_positions=nop,
            loss_mask=loss_mask,
            position_in_block=pib,
        )

    def test_all_sentinel_batch_gives_zero_loss(self):
        cfg, model, out, lm_head, input_ids, pib, B, bs, A, N, T, position_ids = self._setup()
        nop = (torch.arange(N) % bs + 10).unsqueeze(0)   # valid positions < T
        batch = self._batch(position_ids, pib, torch.zeros(B, N, dtype=torch.bool), nop, B, N, T)
        loss = model.compute_loss(out, input_ids, batch, lm_head, gamma=7.0)
        assert loss.item() == 0.0                         # no real positions -> CE 0 and gate 0

    def test_sentinel_content_does_not_affect_loss(self):
        # block 0 real, block 1 sentinel. Changing the sentinel block's anchor/gold
        # must NOT change the loss (it's masked out). With the bug it would.
        cfg, model, out, lm_head, input_ids, pib, B, bs, A, N, T, position_ids = self._setup()
        loss_mask = torch.tensor([[False, True, True, True, False, False, False, False]])

        def loss_with_sentinel_anchor(a1):
            nop = torch.tensor([[10, 11, 12, 13, a1, a1 + 1, a1 + 2, a1 + 3]])
            batch = self._batch(position_ids, pib, loss_mask, nop, B, N, T)
            return model.compute_loss(out, input_ids, batch, lm_head, gamma=7.0)

        loss_a = loss_with_sentinel_anchor(10)
        loss_b = loss_with_sentinel_anchor(20)            # different sentinel content
        assert torch.allclose(loss_a, loss_b)

    def test_real_block_change_does_affect_loss(self):
        # Sanity counterpart: changing the REAL block's gold DOES change the loss,
        # so the invariance above isn't trivially true for all changes.
        cfg, model, out, lm_head, input_ids, pib, B, bs, A, N, T, position_ids = self._setup()
        loss_mask = torch.tensor([[False, True, True, True, False, False, False, False]])

        def loss_with_real_anchor(a0):
            nop = torch.tensor([[a0, a0 + 1, a0 + 2, a0 + 3, 10, 11, 12, 13]])
            batch = self._batch(position_ids, pib, loss_mask, nop, B, N, T)
            return model.compute_loss(out, input_ids, batch, lm_head, gamma=7.0)

        assert not torch.allclose(loss_with_real_anchor(5), loss_with_real_anchor(20))


# ---------------------------------------------------------------------------
# End-to-end: skip draft driven by dflash_generate (sparse DISABLED)
# ---------------------------------------------------------------------------


class TestSkipEndToEndDflashGenerate:
    """Runs the real dflash_generate loop with a skip draft + a tiny target on CPU
    (return_stats=False avoids CUDA timing). Validates the integration glue —
    early-exit backfill + per-block crop across multiple blocks — with sparse OFF."""

    def _tiny_target(self, num_hidden_layers):
        from transformers import Qwen3ForCausalLM
        cfg = make_tiny_qwen3_config(num_hidden_layers=num_hidden_layers)
        return Qwen3ForCausalLM(cfg).eval()

    def _run(self, gate_setup=None):
        from midas.models.model import dflash_generate
        torch.manual_seed(0)
        draft_cfg = make_tiny_qwen3_config(num_hidden_layers=2)   # num_target_layers=8
        draft = DFlashDraftModelSkip(draft_cfg, SkipConfig())
        target = self._tiny_target(num_hidden_layers=draft_cfg.num_target_layers)
        if gate_setup is not None:
            with torch.no_grad():
                gate_setup(draft)
        draft.eval()
        input_ids = torch.randint(0, draft_cfg.vocab_size, (1, 6))
        out = dflash_generate(
            draft, target, input_ids,
            max_new_tokens=8, stop_token_ids=[draft_cfg.vocab_size - 1],
            temperature=0.0, mask_token_id=0,           # sparse_attention defaults None -> dense
        )
        return input_ids, out

    def test_runs_dense_no_exit(self):
        # default gate bias -10 => never exits; full-depth draft through the real loop
        input_ids, out = self._run()
        assert out.dim() == 2 and out.shape[0] == 1
        assert out.shape[1] >= input_ids.shape[1]

    def test_runs_with_early_exit_across_blocks(self):
        # force exit after layer 0 every block -> exercises cache_context_kv backfill
        # AND the next block reading the backfilled deep-layer cache, repeatedly.
        input_ids, out = self._run(lambda d: d.gates[0].bias.fill_(50.0))
        assert out.shape[0] == 1
        assert out.shape[1] >= input_ids.shape[1]
