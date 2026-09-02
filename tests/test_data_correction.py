"""Tests for midas/data_correction.py — round-2 batch construction."""

import pytest
import torch

from midas.config import CorrectionConfig, DFlash2Config
from midas.data import prepare_draft_inputs
from midas.data_correction import (
    build_round2_attention_mask,
    prepare_correction_batch,
    run_round1,
)
from midas.models.model_correction import (
    CorrectionDraftModel,
    build_correction_block,
    positions_to_redraft,
)
from midas.models.model_dflash2 import DFlash2DraftModel
from tests.conftest import MockLMHead, make_tiny_qwen3_config

BLOCK = 4
HIDDEN = 128
VOCAB = 1000
MASK_ID = 999


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config(num_hidden_layers=3, block_size=BLOCK, vocab_size=VOCAB)


@pytest.fixture
def d2_config():
    return DFlash2Config(selector_rank=16, selector_top_k=4, conv_group_size=16)


@pytest.fixture
def parent(tiny_config, d2_config):
    return DFlash2DraftModel(tiny_config, d2_config).eval()


@pytest.fixture
def correction(tiny_config, d2_config):
    return CorrectionDraftModel(
        tiny_config, CorrectionConfig(dflash2=d2_config, min_remaining=1)
    ).eval()


@pytest.fixture
def embed():
    return torch.nn.Embedding(VOCAB, HIDDEN)


@pytest.fixture
def lm_head():
    return MockLMHead(HIDDEN, VOCAB)


def make_round1(parent, seq_lengths, response_starts, max_num_anchors=4, T=32):
    torch.manual_seed(0)
    B = len(seq_lengths)
    input_ids = torch.randint(0, VOCAB - 1, (B, T))
    batch = prepare_draft_inputs(
        input_ids,
        torch.tensor(seq_lengths),
        block_size=BLOCK,
        max_num_anchors=max_num_anchors,
        mask_token_id=MASK_ID,
        response_starts=torch.tensor(response_starts),
        dtype=torch.float32,
    )
    target_hidden = torch.randn(B, T, len(parent.target_layer_ids) * HIDDEN)
    return input_ids, batch, target_hidden


class TestRunRound1:
    def test_propose_is_called_per_block(self, parent, embed, lm_head):
        """The selector walk must never chain across a block boundary."""
        _, batch, target_hidden = make_round1(parent, [32, 32], [8, 8])
        seen = []
        original = parent.propose

        def spy(hidden, anchor_ids, *args, **kwargs):
            seen.append((tuple(hidden.shape), tuple(anchor_ids.shape)))
            return original(hidden, anchor_ids, *args, **kwargs)

        parent.propose = spy
        try:
            run_round1(parent, batch, target_hidden, embed, lm_head, block_size=BLOCK)
        finally:
            parent.propose = original

        num_blocks = batch.noise_token_ids.shape[1] // BLOCK
        # one call (chunk_blocks=32 covers all 2*4 blocks), each block contributing
        # block_size-1 query positions and one anchor
        assert seen == [((2 * num_blocks, BLOCK - 1, HIDDEN), (2 * num_blocks,))]

    def test_chunking_does_not_change_the_result(self, parent, embed, lm_head):
        _, batch, target_hidden = make_round1(parent, [32, 32], [8, 8])
        args = (parent, batch, target_hidden, embed, lm_head)
        whole = run_round1(*args, block_size=BLOCK, chunk_blocks=1000)
        chunked = run_round1(*args, block_size=BLOCK, chunk_blocks=1)
        assert torch.equal(whole[1], chunked[1])
        assert torch.allclose(whole[0], chunked[0])

    def test_anchor_slot_carries_the_known_token(self, parent, embed, lm_head):
        _, batch, target_hidden = make_round1(parent, [32, 32], [8, 8])
        _, proposed = run_round1(parent, batch, target_hidden, embed, lm_head, block_size=BLOCK)
        anchors = batch.noise_token_ids[:, ::BLOCK]
        assert torch.equal(proposed[:, ::BLOCK], anchors)

    def test_returns_the_prenorm_residual(self, parent, embed, lm_head):
        """Not the normed output — that is what makes the correction layer act as
        the parent's next layer."""
        _, batch, target_hidden = make_round1(parent, [32, 32], [8, 8])
        prenorm, _ = run_round1(parent, batch, target_hidden, embed, lm_head, block_size=BLOCK)
        normed = parent(
            position_ids=batch.position_ids,
            attention_mask=batch.attention_mask,
            noise_embedding=embed(batch.noise_token_ids),
            target_hidden=target_hidden,
        )
        assert not torch.allclose(prenorm, normed)
        assert torch.allclose(parent.norm(prenorm), normed, atol=1e-5)


class TestRound2AttentionMask:
    def _block(self, acceptance, anchor=8, block_size=8, T=32):
        n = block_size
        return build_correction_block(
            torch.zeros(1, n, 4),
            torch.arange(n).view(1, n),
            torch.tensor([[acceptance]]),
            torch.tensor([[anchor]]),
            block_size=block_size,
            mask_token_id=MASK_ID,
        ), T

    def test_ctx_opens_up_to_the_confirmed_prefix(self):
        """Round 2 sees target hidden for s..s+a — the tokens verification just
        confirmed — and stops there, because the hidden state at s+a+1 was
        computed from the rejected token."""
        blk, T = self._block(acceptance=2, anchor=8)
        mask = build_round2_attention_mask(blk, T, dtype=torch.float32)
        ctx = mask[0, 0, 1, :T]  # a recycled slot's view of the context
        assert (ctx[: 8 + 3] == 0).all(), "confirmed prefix s..s+a is hidden"
        assert torch.isinf(ctx[8 + 3 :]).all(), "saw past the corrected token"

    def test_every_slot_in_a_block_shares_the_cutoff(self):
        blk, T = self._block(acceptance=2, anchor=8)
        mask = build_round2_attention_mask(blk, T, dtype=torch.float32)
        ctx = mask[0, 0, :, :T]
        assert (ctx == ctx[0]).all()

    def test_pads_are_invisible_as_keys(self):
        blk, T = self._block(acceptance=2, block_size=8)
        mask = build_round2_attention_mask(blk, T, dtype=torch.float32)
        noise = mask[0, 0, :, T:]
        pad = blk.is_pad[0]
        assert torch.isinf(noise[:, pad]).all(), "a valid slot can read a pad slot"
        assert (noise[1, ~pad] == 0).all()

    def test_no_all_masked_query_row(self):
        """An all -inf row would NaN the softmax. Pad queries keep the anchor."""
        for a in range(0, 7):
            blk, T = self._block(acceptance=a, block_size=8)
            mask = build_round2_attention_mask(blk, T, dtype=torch.float32)
            assert (~torch.isinf(mask[0, 0])).any(dim=-1).all(), f"a={a}"

    def test_blocks_cannot_see_each_other(self):
        blk = build_correction_block(
            torch.zeros(1, 8, 4),
            torch.arange(8).view(1, 8),
            torch.tensor([[1, 1]]),
            torch.tensor([[0, 4]]),
            block_size=4,
            mask_token_id=MASK_ID,
        )
        mask = build_round2_attention_mask(blk, 32, dtype=torch.float32)
        noise = mask[0, 0, :, 32:]
        assert torch.isinf(noise[:4, 4:]).all()
        assert torch.isinf(noise[4:, :4]).all()


class TestPrepareCorrectionBatch:
    def _prepare(self, parent, embed, lm_head, seq_lengths=(32, 32),
                 response_starts=(8, 8), max_num_anchors=4, **kwargs):
        input_ids, batch, target_hidden = make_round1(
            parent, list(seq_lengths), list(response_starts), max_num_anchors
        )
        kwargs.setdefault("below_trigger_keep_prob", 0.0)
        kwargs.setdefault("min_remaining", 1)
        step = prepare_correction_batch(
            parent, input_ids, batch, target_hidden, embed, lm_head,
            block_size=BLOCK, mask_token_id=MASK_ID, dtype=torch.float32, **kwargs,
        )
        return input_ids, batch, target_hidden, step

    def test_shapes_line_up_with_round_one(self, parent, embed, lm_head):
        _, batch, _, step = self._prepare(parent, embed, lm_head)
        B, N = batch.noise_token_ids.shape
        assert step.batch.noise_token_ids.shape == (B, N)
        assert step.recycled_hidden.shape == (B, N, HIDDEN)
        assert step.is_recycled.shape == (B, N)
        assert step.batch.attention_mask.shape == (B, 1, N, 32 + N)

    def test_supervises_only_recycled_slots(self, parent, embed, lm_head):
        _, _, _, step = self._prepare(parent, embed, lm_head)
        assert not (step.batch.loss_mask & ~step.is_recycled).any()

    def test_below_trigger_blocks_are_gated_out(self, parent, embed, lm_head):
        """min_remaining above what any block can offer => nothing supervised."""
        _, _, _, step = self._prepare(parent, embed, lm_head, min_remaining=BLOCK)
        assert step.batch.loss_mask.sum() == 0
        assert step.metrics["trigger_rate"] == 0.0

    def test_round1_sentinel_blocks_are_gated_out(self, parent, embed, lm_head):
        """A short row is padded with duplicate anchors; those must not be trained
        on, or the same block gets supervised repeatedly."""
        _, batch, _, step = self._prepare(
            parent, embed, lm_head, seq_lengths=(32, 20), response_starts=(8, 8),
            max_num_anchors=12,
        )
        B, N = batch.noise_token_ids.shape
        num_blocks = N // BLOCK
        round1_active = batch.loss_mask.view(B, num_blocks, BLOCK)[:, :, 1:].any(-1)
        assert not round1_active.all(), "no sentinels in this fixture — test is vacuous"
        dead = ~round1_active.repeat_interleave(BLOCK, dim=1)
        assert not (step.batch.loss_mask & dead).any()

    def test_keep_prob_one_admits_below_trigger_blocks(self, parent, embed, lm_head):
        _, _, _, gated = self._prepare(parent, embed, lm_head, min_remaining=BLOCK)
        torch.manual_seed(0)
        _, _, _, kept = self._prepare(
            parent, embed, lm_head, min_remaining=BLOCK, below_trigger_keep_prob=1.0
        )
        assert gated.batch.loss_mask.sum() == 0
        assert kept.batch.loss_mask.sum() > 0

    def test_anchor_is_the_verifier_token_at_the_rejection_point(self, parent, embed, lm_head):
        input_ids, batch, _, step = self._prepare(parent, embed, lm_head)
        # round-2 anchor position s+a+1 -> its token must be ground truth there
        anchors = step.batch.noise_original_positions[:, ::BLOCK]
        expected = input_ids.gather(1, anchors)
        assert torch.equal(step.batch.noise_token_ids[:, ::BLOCK], expected)

    def test_positions_stay_inside_the_round1_block(self, parent, embed, lm_head):
        _, batch, _, step = self._prepare(parent, embed, lm_head)
        B, N = batch.noise_token_ids.shape
        num_blocks = N // BLOCK
        r1 = batch.noise_original_positions.view(B, num_blocks, BLOCK)
        r2 = step.batch.noise_original_positions.view(B, num_blocks, BLOCK)
        assert (r2 <= r1[:, :, -1:]).all()
        assert (r2 >= r1[:, :, :1]).all()

    def test_metrics_are_reported(self, parent, embed, lm_head):
        _, _, _, step = self._prepare(parent, embed, lm_head)
        assert set(step.metrics) == {
            "round1_acceptance", "trigger_rate", "blocks_kept",
            "supervised_positions", "supervised_width",
        }
        assert 0.0 <= step.metrics["round1_acceptance"] <= BLOCK - 1

    def test_feeds_the_correction_model_and_its_loss(
        self, parent, correction, embed, lm_head
    ):
        """End to end: round-2 batch -> forward -> loss -> gradient."""
        input_ids, _, target_hidden, step = self._prepare(parent, embed, lm_head)
        out = correction(
            position_ids=step.batch.position_ids,
            attention_mask=step.batch.attention_mask,
            noise_embedding=embed(step.batch.noise_token_ids),
            target_hidden=target_hidden,
            recycled_hidden=step.recycled_hidden,
            is_recycled=step.is_recycled,
        )
        assert out.shape == step.recycled_hidden.shape
        assert torch.isfinite(out).all(), "masking produced NaNs"
        loss = correction.compute_loss(
            out, input_ids, step.batch, lm_head, embed_tokens=embed
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert correction.input_mixer.recycled_proj.weight.grad is not None

    def test_parent_receives_no_gradient(self, parent, embed, lm_head):
        input_ids, _, target_hidden, step = self._prepare(parent, embed, lm_head)
        assert not step.recycled_hidden.requires_grad
