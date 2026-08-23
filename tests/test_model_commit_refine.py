"""Tests for the commit-and-refine drafter.

Two are load-bearing. `test_committed_slots_emit_stage_one` pins the property the whole
design rests on — the model conditions on exactly the token it emits, which is what makes
conditioning errors free under prefix acceptance. `test_identity_at_init` pins that stage 2
starts as plain extra depth, so the arm is a clean comparison against DFlash.
"""

import pytest
import torch
from torch import nn

from midas.config import CommitRefineConfig
from midas.models.model import DFlashDraftModel
from midas.models.model_commit_refine import CommitRefineDraftModel
from tests.conftest import MockLMHead, make_tiny_qwen3_config

pytestmark = pytest.mark.model

COMMIT = 1


@pytest.fixture
def shared(tiny_config):
    """Stand-ins for the target's frozen head and embedding table."""
    return MockLMHead(tiny_config.hidden_size, tiny_config.vocab_size), \
        nn.Embedding(tiny_config.vocab_size, tiny_config.hidden_size)


@pytest.fixture
def cr_config(tiny_config):
    return CommitRefineConfig(commit=COMMIT, refine_layers=2, refine_intermediate=64)


@pytest.fixture
def model(tiny_config, cr_config, shared):
    m = CommitRefineDraftModel(tiny_config, commit_refine_config=cr_config).eval()
    m.bind_shared(*shared)
    return m


def _inputs(config, B=2, T=12):
    n = config.block_size * 2
    return {
        "position_ids": torch.cat(
            [torch.arange(T).unsqueeze(0).expand(B, -1),
             torch.arange(T, T + n).unsqueeze(0).expand(B, -1)], dim=1),
        "noise_embedding": torch.randn(B, n, config.hidden_size),
        "target_hidden": torch.randn(B, T, config.hidden_size * config.num_hidden_layers),
    }


class TestCommitConsistency:
    def test_committed_slots_emit_stage_one(self, tiny_config, cr_config, shared, model):
        """Committed slots must emit stage 1 — the token stage 2 conditioned on.

        If they emitted stage 2 instead, the model could condition on one token and emit
        another, and conditioning errors would stop being free under prefix acceptance.
        """
        stage1 = DFlashDraftModel(tiny_config).eval()
        stage1.load_state_dict(
            {k: v for k, v in model.state_dict().items() if k in stage1.state_dict()})
        args = _inputs(tiny_config)
        with torch.no_grad():
            out, ref = model(**args), stage1(**args)

        offsets = torch.arange(out.shape[1]) % tiny_config.block_size
        committed = offsets <= COMMIT
        torch.testing.assert_close(out[:, committed], ref[:, committed])

    def test_refined_slots_differ_once_the_gate_opens(self, tiny_config, model):
        args = _inputs(tiny_config)
        with torch.no_grad():
            before = model(**args)
            nn.init.normal_(model.commit_gate, std=0.5)
            after = model(**args)
        offsets = torch.arange(before.shape[1]) % tiny_config.block_size
        refined = offsets > COMMIT
        assert not torch.allclose(before[:, refined], after[:, refined])
        # committed slots are upstream of the gate and must be untouched
        torch.testing.assert_close(before[:, ~refined], after[:, ~refined])


class TestInit:
    def test_identity_at_init(self, tiny_config, cr_config, shared):
        """Zero gate => stage 2 is plain extra depth, not conditioning."""
        m = CommitRefineDraftModel(tiny_config, commit_refine_config=cr_config)
        assert m.commit_gate.abs().sum() == 0

    def test_stage_two_gets_its_own_kv_slots(self, tiny_config, model):
        """Refine layers must not collide with the backbone's cache entries."""
        idx = [l.self_attn.layer_idx for l in model.layers] + \
              [l.self_attn.layer_idx for l in model.refine_layers]
        assert idx == list(range(len(idx)))

    def test_gate_is_logged(self, model):
        """A flat-zero gate is the only way to tell 'conditioning did nothing' apart from
        'stage 2 helped as extra depth', without a separate control arm."""
        assert model.log_metrics(loss=1.0)["commit_gate_mean"] == 0.0
        nn.init.normal_(model.commit_gate, std=0.1)
        assert model.log_metrics(loss=1.0)["commit_gate_mean"] > 0

    def test_rejects_a_commit_that_leaves_nothing_to_refine(self, tiny_config):
        with pytest.raises(ValueError, match="at least one slot"):
            CommitRefineDraftModel(
                tiny_config, CommitRefineConfig(commit=tiny_config.block_size - 1))


class TestPlumbing:
    def test_forward_requires_bind_shared(self, tiny_config, cr_config):
        m = CommitRefineDraftModel(tiny_config, commit_refine_config=cr_config).eval()
        with pytest.raises(ValueError, match="bind_shared"):
            m(**_inputs(tiny_config))

    def test_shared_modules_stay_out_of_the_state_dict(self, model):
        assert not any("_shared" in k or "lm_head" in k for k in model.state_dict())

    def test_rejects_a_ragged_block(self, tiny_config, model):
        args = _inputs(tiny_config)
        args["noise_embedding"] = args["noise_embedding"][:, :-1]
        args["position_ids"] = args["position_ids"][:, :-1]
        with pytest.raises(ValueError, match="multiple of block_size"):
            model(**args)

    def test_gradients_reach_stage_two(self, tiny_config, cr_config, shared):
        m = CommitRefineDraftModel(tiny_config, commit_refine_config=cr_config)
        m.bind_shared(*shared)
        nn.init.normal_(m.commit_gate, std=0.1)
        m(**_inputs(tiny_config)).sum().backward()
        assert m.commit_gate.grad.abs().sum() > 0
        assert m.refine_layers[0].mlp.down_proj.weight.grad.abs().sum() > 0


class TestInjectionScale:
    """The commit is only conditioning if it is big enough to be read.

    The first trained run put a raw token embedding (RMS ~0.02) onto a residual of RMS ~40.
    That is a 0.009% perturbation, and it showed: removing the commit entirely changed the
    token at 0.1% of refined slots while stage 2 changed 41% of them. The gate would have
    needed to reach ~80 to matter and got to 0.28 in 8794 steps. So assert the invariant
    that fixes it — the injection lands on the residual's scale, whatever the embedding's.
    """

    @staticmethod
    def _injection_and_residual(model, hidden_rms, embed_scale):
        with torch.no_grad():                       # a realistic scale mismatch
            model._shared[1].weight.mul_(embed_scale)
        n = model.block_size * 2
        hidden = torch.randn(2, n, model.config.hidden_size) * hidden_rms
        offsets = model._block_offsets(n, hidden.device)
        inj = model._injection(hidden, model.norm(hidden), offsets)
        committed = ((offsets >= 1) & (offsets <= COMMIT))
        rms = lambda t: t.pow(2).mean().sqrt().item()
        return rms(inj[:, committed]), rms(hidden), inj, committed

    def test_injection_matches_the_residual_scale(self, model):
        inj, res, _, _ = self._injection_and_residual(model, hidden_rms=40.0, embed_scale=1e-3)
        assert 0.5 < inj / res < 2.0, (
            f"injection RMS {inj:.4g} vs residual {res:.4g}; a gate of O(0.1) has to be an "
            f"O(10%) perturbation, not O(0.01%)")

    def test_unscaled_mode_reproduces_the_original_bug(self, tiny_config, shared):
        """Guards the compatibility path: it must stay faithful to what v1 trained."""
        m = CommitRefineDraftModel(
            tiny_config,
            commit_refine_config=CommitRefineConfig(commit=COMMIT, refine_layers=2,
                                                    refine_intermediate=64,
                                                    scaled_injection=False)).eval()
        m.bind_shared(*shared)
        inj, res, _, _ = self._injection_and_residual(m, hidden_rms=40.0, embed_scale=1e-3)
        assert inj / res < 1e-3
        assert "commit_norm.weight" not in m.state_dict()

    def test_uncommitted_slots_stay_exactly_zero(self, model):
        """RMSNorm maps zero to zero, so the rescale must not leak into unwritten slots."""
        _, _, inj, committed = self._injection_and_residual(model, 40.0, 1e-3)
        assert inj[:, ~committed].abs().max() == 0

    def test_zero_gate_is_still_an_exact_no_op(self, tiny_config, cr_config, shared):
        """Stage 2 must still start as plain extra depth, as the depth-only control did."""
        m = CommitRefineDraftModel(tiny_config, commit_refine_config=cr_config).eval()
        m.bind_shared(*shared)
        assert m.commit_gate.abs().sum() == 0
        assert "commit_norm.weight" in m.state_dict()


class TestConditioningIsReal:
    """The hypothesis, stated as a test: stage 2 conditions on WHICH token was committed.

    v1 satisfied every check above — committed slots emitted stage 1, the gate opened, the
    output moved — and its conditioning was still inert, because a raw embedding on a much
    larger residual is a rounding error. So these run a counterfactual on the committed
    token under a realistic scale mismatch, which is what production actually had.
    """

    @staticmethod
    def _forward_committing(model, args, token_id):
        """Run the model with `token_id` force-committed at every committed slot."""
        real = model._committed_embedding

        def fake(stage1, offsets):
            out = torch.zeros_like(stage1)
            slots = ((offsets >= 1) & (offsets <= COMMIT)).nonzero(as_tuple=True)[0]
            ids = torch.full((stage1.shape[0], len(slots)), token_id, dtype=torch.long)
            out[:, slots] = model._shared[1](ids).to(stage1.dtype)
            return out

        model._committed_embedding = fake
        try:
            with torch.no_grad():
                return model(**args)
        finally:
            model._committed_embedding = real

    @classmethod
    def _counterfactual(cls, model, tiny_config):
        with torch.no_grad():                  # the production scale mismatch: emb << residual
            model._shared[1].weight.mul_(1e-3)
        nn.init.normal_(model.commit_gate, std=0.5)
        args = _inputs(tiny_config)
        a = cls._forward_committing(model, args, 3)
        b = cls._forward_committing(model, args, 17)

        refined = (torch.arange(a.shape[1]) % tiny_config.block_size) > COMMIT
        # Committed slots emit stage 1, which is upstream of the commit — so they must be
        # bit-identical, proving the measured change is the commit pathway and nothing else.
        torch.testing.assert_close(a[:, ~refined], b[:, ~refined])
        return ((a[:, refined] - b[:, refined]).norm() / a[:, refined].norm()).item()

    @staticmethod
    def _model(tiny_config, scaled):
        # Fresh head and embedding per model: `_counterfactual` rescales the table in place,
        # so a reused fixture would hand the second model an already-shrunk one.
        m = CommitRefineDraftModel(
            tiny_config,
            commit_refine_config=CommitRefineConfig(commit=COMMIT, refine_layers=2,
                                                    refine_intermediate=64,
                                                    scaled_injection=scaled)).eval()
        m.bind_shared(MockLMHead(tiny_config.hidden_size, tiny_config.vocab_size),
                      nn.Embedding(tiny_config.vocab_size, tiny_config.hidden_size))
        return m

    def test_committed_token_identity_reaches_the_refined_slots(self, tiny_config):
        """Scaled vs unscaled, same seed, same inputs — the only difference is the fix.

        Compared as a ratio, not an absolute threshold: these refine layers are randomly
        initialised and so near-identity that even a large perturbation at the committed
        slot propagates weakly. The ratio is what isolates the mechanism.
        """
        rel = {}
        for scaled in (True, False):
            torch.manual_seed(0)
            m = self._model(tiny_config, scaled)
            torch.manual_seed(1)
            rel[scaled] = self._counterfactual(m, tiny_config)

        assert rel[True] > 100 * rel[False], (
            f"committed-token identity moved the refined slots by {rel[True]:.4%} with the "
            f"scaled injection vs {rel[False]:.4%} without — the commit is not reaching "
            f"stage 2 (v1 shipped at the second number and its conditioning was inert)")
        assert rel[False] < 1e-4          # pins what the depth-only control actually does
