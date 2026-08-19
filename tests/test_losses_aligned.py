"""Tests for midas/losses_aligned.py — acceptance-aligned auxiliary losses."""

import pytest
import torch
import torch.nn as nn

from midas.config import AlignedLossConfig
from midas.losses_aligned import (
    compute_aligned_loss,
    compute_eal_loss,
    top1_gap,
    compute_feature_reg_loss,
    compute_margin_loss,
    target_argmax_chunked,
)
from midas.models.model import compute_position_weighted_ce_loss
from tests.conftest import MockLMHead


class TestMarginLoss:
    def test_zero_when_gap_exceeds_margin(self):
        # y*=0 leads by 5.0 everywhere, margin 1.0 -> no loss, no gradient
        logits = torch.tensor([[10.0, 5.0, 1.0], [8.0, 3.0, 0.0]])
        labels = torch.tensor([0, 0])
        assert torch.equal(
            compute_margin_loss(logits, labels, margin=1.0), torch.zeros(2)
        )

    def test_positive_and_exact_when_gap_below_margin(self):
        # gap = 10.0 - 9.5 = 0.5 ; margin 2.0 -> loss 1.5
        logits = torch.tensor([[10.0, 9.5, 1.0]])
        labels = torch.tensor([0])
        got = compute_margin_loss(logits, labels, margin=2.0)
        assert torch.allclose(got, torch.tensor([1.5]), atol=1e-6)

    def test_negative_gap_when_label_is_not_top1(self):
        # y*=1 scores 9.5, runner-up (the actual top-1) is 10.0 -> gap -0.5
        logits = torch.tensor([[10.0, 9.5, 1.0]])
        labels = torch.tensor([1])
        got = compute_margin_loss(logits, labels, margin=1.0)
        assert torch.allclose(got, torch.tensor([1.5]), atol=1e-6)

    def test_runner_up_selection_both_branches(self):
        """Runner-up is 2nd entry when y* is top-1, else the 1st."""
        logits = torch.tensor([[5.0, 3.0, 1.0], [5.0, 3.0, 1.0]])
        labels = torch.tensor([0, 1])  # top-1 / not top-1
        got = compute_margin_loss(logits, labels, margin=0.0)
        # row0 gap 5-3=2 -> 0 ; row1 gap 3-5=-2 -> 2
        assert torch.allclose(got, torch.tensor([0.0, 2.0]), atol=1e-6)

    def test_identical_in_logit_and_logprob_space(self):
        """log_softmax subtracts a per-row constant, so the gap is invariant."""
        torch.manual_seed(0)
        logits = torch.randn(16, 50) * 3
        labels = torch.randint(0, 50, (16,))
        logprobs = torch.log_softmax(logits, dim=-1)
        assert torch.allclose(
            compute_margin_loss(logits, labels, 1.0),
            compute_margin_loss(logprobs, labels, 1.0),
            atol=1e-5,
        )

    def test_gradient_pushes_label_up_and_runner_down(self):
        logits = torch.tensor([[1.0, 2.0, 0.0]], requires_grad=True)
        compute_margin_loss(logits, torch.tensor([0]), margin=1.0).sum().backward()
        assert logits.grad[0, 0] < 0  # descending -> raises the label logit
        assert logits.grad[0, 1] > 0  # descending -> lowers the runner-up
        assert logits.grad[0, 2] == 0  # untouched third token


class TestFeatureRegLoss:
    @pytest.mark.parametrize("metric", ["smooth_l1", "cosine"])
    def test_zero_for_identical_vectors(self, metric):
        h = torch.randn(4, 8)
        assert torch.allclose(
            compute_feature_reg_loss(h, h, metric), torch.zeros(4), atol=1e-6
        )

    @pytest.mark.parametrize("metric", ["smooth_l1", "cosine"])
    def test_positive_for_different_vectors(self, metric):
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        assert (compute_feature_reg_loss(a, b, metric) > 0).all()

    def test_cosine_is_one_for_orthogonal(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        assert torch.allclose(
            compute_feature_reg_loss(a, b, "cosine"), torch.tensor([1.0]), atol=1e-6
        )

    def test_cosine_invariant_to_positive_rescaling(self):
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        assert torch.allclose(
            compute_feature_reg_loss(a, b, "cosine"),
            compute_feature_reg_loss(a * 7.5, b, "cosine"),
            atol=1e-5,
        )

    def test_default_metric_is_cosine(self):
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        assert torch.equal(
            compute_feature_reg_loss(a, b), compute_feature_reg_loss(a, b, "cosine")
        )

    def test_both_metrics_have_comparable_gradient_scale(self):
        """Measured: smooth_l1's mean over H divides its gradient by H, so it is
        NOT stronger than cosine. This is why lambda_reg has to be large."""
        torch.manual_seed(0)
        t = torch.randn(32, 256)
        grads = {}
        for metric in ("smooth_l1", "cosine"):
            d = torch.randn(32, 256, requires_grad=True)
            compute_feature_reg_loss(d, t, metric).mean().backward()
            grads[metric] = d.grad.norm().item()
        assert 0.1 < grads["smooth_l1"] / grads["cosine"] < 10.0, grads

    def test_rejects_unknown_metric(self):
        with pytest.raises(ValueError, match="unknown reg metric"):
            compute_feature_reg_loss(torch.randn(2, 4), torch.randn(2, 4), "l2")


class TestChunkedArgmax:
    def test_matches_unchunked(self):
        torch.manual_seed(0)
        lm_head = MockLMHead(16, 97)
        h = torch.randn(37, 16)
        expected = lm_head(h).argmax(dim=-1)
        for chunk in (1, 5, 37, 1000):
            assert torch.equal(target_argmax_chunked(lm_head, h, chunk), expected)

    def test_empty_input(self):
        out = target_argmax_chunked(MockLMHead(16, 97), torch.zeros(0, 16), 8)
        assert out.shape == (0,) and out.dtype == torch.long

    def test_only_calls_lm_head(self):
        """Must never touch lm_head.weight — nn.Identity is passed in existing tests."""
        h = torch.randn(6, 5)
        assert torch.equal(target_argmax_chunked(nn.Identity(), h, 4), h.argmax(dim=-1))


class TestAlignedLossEquivalence:
    def test_zero_weights_bit_identical_to_baseline_ce(
        self, dummy_draft_batch, mock_lm_head
    ):
        torch.manual_seed(0)
        draft_hidden = torch.randn(2, 8, 128)
        input_ids = torch.randint(0, 1000, (2, 32))
        base = compute_position_weighted_ce_loss(
            draft_hidden, input_ids, dummy_draft_batch, mock_lm_head, gamma=7.0
        )
        aligned = compute_aligned_loss(
            draft_hidden, input_ids, dummy_draft_batch, mock_lm_head, gamma=7.0
        )
        assert torch.equal(base, aligned), f"{base.item()} != {aligned.item()}"

    def test_zero_weights_needs_no_target_hidden(self, dummy_draft_batch, mock_lm_head):
        """The lambda=0 path must not require the new argument at all."""
        compute_aligned_loss(
            torch.randn(2, 8, 128),
            torch.randint(0, 1000, (2, 32)),
            dummy_draft_batch,
            mock_lm_head,
            gamma=7.0,
            target_final_hidden=None,
        )

    def test_raises_if_target_hidden_missing_when_enabled(
        self, dummy_draft_batch, mock_lm_head
    ):
        with pytest.raises(ValueError, match="target_final_hidden"):
            compute_aligned_loss(
                torch.randn(2, 8, 128),
                torch.randint(0, 1000, (2, 32)),
                dummy_draft_batch,
                mock_lm_head,
                gamma=7.0,
                lambda_margin=0.5,
            )

    def test_empty_loss_mask_returns_zero(self, dummy_draft_batch, mock_lm_head):
        dummy_draft_batch.loss_mask = torch.zeros_like(dummy_draft_batch.loss_mask)
        loss = compute_aligned_loss(
            torch.randn(2, 8, 128),
            torch.randint(0, 1000, (2, 32)),
            dummy_draft_batch,
            mock_lm_head,
            gamma=7.0,
            lambda_margin=0.5,
        )
        assert loss.item() == 0.0


class TestAlignedLossTerms:
    def _inputs(self):
        torch.manual_seed(0)
        return (
            torch.randn(2, 8, 128, requires_grad=True),
            torch.randint(0, 1000, (2, 32)),
            torch.randn(2, 32, 128),
        )

    def test_each_term_increases_loss_and_flows_gradient(
        self, dummy_draft_batch, mock_lm_head
    ):
        draft_hidden, input_ids, tgt = self._inputs()
        kw = dict(gamma=7.0, target_final_hidden=tgt)
        base = compute_aligned_loss(
            draft_hidden, input_ids, dummy_draft_batch, mock_lm_head, **kw
        )
        for term in ({"lambda_margin": 0.5}, {"lambda_reg": 0.1}):
            draft_hidden.grad = None
            loss = compute_aligned_loss(
                draft_hidden, input_ids, dummy_draft_batch, mock_lm_head, **kw, **term
            )
            assert loss.item() > base.item(), term
            loss.backward()
            assert draft_hidden.grad is not None and (draft_hidden.grad != 0).any(), term

    def test_metrics_are_populated(self, dummy_draft_batch, mock_lm_head):
        draft_hidden, input_ids, tgt = self._inputs()
        metrics = {}
        compute_aligned_loss(
            draft_hidden, input_ids, dummy_draft_batch, mock_lm_head,
            gamma=7.0, target_final_hidden=tgt,
            lambda_margin=0.5, lambda_reg=0.1, metrics=metrics,
        )
        for key in ("loss_ce", "loss_margin", "loss_featreg",
                    "frac_margin_active", "sdd_argmax_agreement"):
            assert key in metrics and isinstance(metrics[key], float), key

    def test_relabel_ce_changes_the_loss(self, dummy_draft_batch, mock_lm_head):
        draft_hidden, input_ids, tgt = self._inputs()
        kw = dict(gamma=7.0, target_final_hidden=tgt)
        a = compute_aligned_loss(
            draft_hidden, input_ids, dummy_draft_batch, mock_lm_head, **kw, relabel_ce=False
        )
        b = compute_aligned_loss(
            draft_hidden, input_ids, dummy_draft_batch, mock_lm_head, **kw, relabel_ce=True
        )
        assert not torch.allclose(a, b)


class TestAlignedModel:
    def test_state_dict_identical_to_base(self, tiny_config):
        from midas.models.model import DFlashDraftModel
        from midas.models.model_aligned import DFlashDraftModelAligned

        base = DFlashDraftModel(tiny_config)
        aligned = DFlashDraftModelAligned(tiny_config, AlignedLossConfig(lambda_margin=0.5))
        assert set(base.state_dict()) == set(aligned.state_dict())
        assert sum(p.numel() for p in base.parameters()) == sum(
            p.numel() for p in aligned.parameters()
        )

    def test_compute_loss_runs_and_logs(self, tiny_config, dummy_draft_batch, mock_lm_head):
        from midas.models.model_aligned import DFlashDraftModelAligned

        model = DFlashDraftModelAligned(
            tiny_config, AlignedLossConfig(lambda_margin=0.5, lambda_reg=0.1)
        )
        loss = model.compute_loss(
            torch.randn(2, 8, 128),
            torch.randint(0, 1000, (2, 32)),
            dummy_draft_batch,
            mock_lm_head,
            gamma=7.0,
            target_final_hidden=torch.randn(2, 32, 128),
        )
        assert torch.isfinite(loss)
        assert "loss_margin" in model.log_metrics(loss=loss.item())


class TestEALLoss:
    """tau_hat = sum_k prod_{j<=k} sigmoid(gap_j / T), normalized to [-1, 0]."""

    def _batch(self, B=2, blocks=2, bs=4):
        """A DraftTrainingBatch-shaped stub with `blocks` real blocks per row."""
        from midas.data import DraftTrainingBatch
        N = blocks * bs
        pib = torch.arange(bs).repeat(blocks).unsqueeze(0).expand(B, -1)
        return DraftTrainingBatch(
            noise_token_ids=torch.zeros(B, N, dtype=torch.long),
            position_ids=torch.zeros(B, N, dtype=torch.long),
            attention_mask=torch.zeros(B, 1, N, N),
            noise_original_positions=torch.zeros(B, N, dtype=torch.long),
            loss_mask=(pib > 0),
            position_in_block=pib,
        )

    def _idx(self, batch):
        m = batch.loss_mask.nonzero(as_tuple=False)
        return m[:, 0], m[:, 1]

    def test_perfect_gaps_give_maximal_tau(self):
        """Huge positive gaps -> every position accepted -> L = -1."""
        b = self._batch(); bi, ni = self._idx(b)
        loss = compute_eal_loss(torch.full((bi.numel(),), 50.0), b, bi, ni)
        assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-5), loss

    def test_hopeless_gaps_give_zero_tau(self):
        b = self._batch(); bi, ni = self._idx(b)
        loss = compute_eal_loss(torch.full((bi.numel(),), -50.0), b, bi, ni)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5), loss

    def test_matches_explicit_cumprod(self):
        torch.manual_seed(0)
        b = self._batch(B=1, blocks=1, bs=5); bi, ni = self._idx(b)
        gaps = torch.randn(bi.numel())
        got = compute_eal_loss(gaps, b, bi, ni, temperature=1.0)
        sig = torch.sigmoid(gaps)
        want = -(sig.cumprod(0).sum()) / 1 / (5 - 1)
        assert torch.allclose(got, want, atol=1e-6), (got, want)

    def test_early_position_has_larger_gradient_than_late(self):
        """The whole point: dtau/dgap is survival-weighted, so position 1 leads."""
        b = self._batch(B=1, blocks=1, bs=5); bi, ni = self._idx(b)
        g = torch.zeros(bi.numel(), requires_grad=True)   # all gaps 0 -> sigma=0.5
        compute_eal_loss(g, b, bi, ni).backward()
        mag = g.grad.abs()
        assert (mag[:-1] > mag[1:]).all(), mag

    def test_doomed_prefix_kills_downstream_gradient(self):
        """If position 1 is hopeless the suffix is worth nothing and gets no gradient."""
        b = self._batch(B=1, blocks=1, bs=5); bi, ni = self._idx(b)
        g = torch.tensor([-50.0, 0.0, 0.0, 0.0], requires_grad=True)
        compute_eal_loss(g, b, bi, ni).backward()
        assert g.grad[1:].abs().max() < 1e-6, g.grad

    def test_sentinel_blocks_excluded(self):
        """Blocks gated out of loss_mask must not contribute."""
        b = self._batch(B=1, blocks=2, bs=4)
        b.loss_mask = b.loss_mask.clone()
        b.loss_mask[0, 4:] = False                      # kill the 2nd block
        bi, ni = self._idx(b)
        loss = compute_eal_loss(torch.full((bi.numel(),), 50.0), b, bi, ni)
        assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-5), loss

    def test_temperature_sharpens(self):
        b = self._batch(B=1, blocks=1, bs=5); bi, ni = self._idx(b)
        gaps = torch.full((bi.numel(),), 1.0)
        hot = compute_eal_loss(gaps, b, bi, ni, temperature=5.0)
        cold = compute_eal_loss(gaps, b, bi, ni, temperature=0.2)
        assert cold < hot, (cold, hot)   # lower T -> sigmoid closer to 1 -> larger tau

    def test_top1_gap_shared_with_margin(self):
        logits = torch.tensor([[5.0, 3.0, 1.0]])
        labels = torch.tensor([0])
        assert torch.allclose(top1_gap(logits, labels), torch.tensor([2.0]))
        assert torch.allclose(
            compute_margin_loss(logits, labels, 3.0),
            torch.relu(3.0 - top1_gap(logits, labels)),
        )

    def test_end_to_end_through_aligned_loss(self, dummy_draft_batch, mock_lm_head):
        dh = torch.randn(2, 8, 128, requires_grad=True)
        ids = torch.randint(0, 1000, (2, 32))
        kw = dict(gamma=7.0, target_final_hidden=torch.randn(2, 32, 128))
        base = compute_aligned_loss(dh, ids, dummy_draft_batch, mock_lm_head, **kw)
        m = {}
        loss = compute_aligned_loss(dh, ids, dummy_draft_batch, mock_lm_head,
                                    **kw, lambda_eal=1.0, metrics=m)
        assert torch.isfinite(loss) and loss.item() != base.item()
        assert "loss_eal" in m and "tau_hat" in m
        loss.backward()
        assert dh.grad is not None and (dh.grad != 0).any()
