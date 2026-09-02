"""Tests for midas/models/model_correction.py — the depth-1 correction drafter."""

import pytest
import torch

from midas.config import CorrectionConfig, DFlash2Config
from midas.models.model_correction import (
    CorrectionDraftModel,
    RecycledInputMixer,
    build_correction_block,
    positions_to_redraft,
    should_correct,
    simulate_acceptance,
)
from midas.models.model_dflash2 import DFlash2DraftModel
from tests.conftest import MockLMHead, make_tiny_qwen3_config

BLOCK = 4
HIDDEN = 128


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config(num_hidden_layers=3, block_size=BLOCK)


@pytest.fixture
def correction_config():
    return CorrectionConfig(
        dflash2=DFlash2Config(selector_rank=16, selector_top_k=4, conv_group_size=16),
        min_remaining=1,
    )


@pytest.fixture
def model(tiny_config, correction_config):
    return CorrectionDraftModel(tiny_config, correction_config).eval()


class TestTriggerPolicy:
    def test_remaining_count_excludes_anchor_and_bonus(self):
        # block 16, accept 5 -> s..s+5 confirmed, bonus at s+6 becomes the anchor,
        # leaving s+7..s+15 = 9 slots to re-draft.
        assert positions_to_redraft(5, 16) == 9

    def test_fires_when_enough_left(self):
        assert should_correct(5, 16, min_remaining=7)

    def test_silent_when_acceptance_was_high(self):
        assert not should_correct(10, 16, min_remaining=7)

    def test_never_twice_in_a_row(self):
        assert not should_correct(0, 16, min_remaining=7, previous_was_correction=True)


class TestBuildCorrectionBlock:
    def _build(self, acceptance, block_size=8, num_blocks=1, B=1):
        n = num_blocks * block_size
        # hidden[.., j, 0] = j so we can assert which round-1 slot landed where.
        prenorm = torch.zeros(B, n, 4)
        prenorm[..., 0] = torch.arange(n).float()
        true_ids = torch.arange(100, 100 + n).view(1, n).expand(B, n).contiguous()
        return build_correction_block(
            prenorm,
            true_ids,
            torch.full((B, num_blocks), acceptance, dtype=torch.long),
            torch.zeros(B, num_blocks, dtype=torch.long),
            block_size=block_size,
            mask_token_id=999,
        )

    def test_slot_layout(self):
        # block 8, accept 2 -> anchor at s+3, redraft s+4..s+7 (4 slots), 3 pads.
        blk = self._build(acceptance=2, block_size=8)
        assert blk.is_recycled[0].tolist() == [False, True, True, True, True, False, False, False]
        assert blk.valid_mask.sum().item() == positions_to_redraft(2, 8)

    def test_anchor_is_the_corrected_token(self):
        blk = self._build(acceptance=2, block_size=8)
        # true token at round-1 slot a+1 = 3 -> id 103.
        assert blk.noise_token_ids[0, 0].item() == 103
        assert blk.noise_token_ids[0, 1:].unique().tolist() == [999]

    def test_recycled_slots_read_past_the_rejection_point(self):
        blk = self._build(acceptance=2, block_size=8)
        # slot i (i>=1) must carry round-1 slot i+a+1, i.e. 4,5,6,7 — never slot 3,
        # whose residual predicted the rejected token and is replaced by the anchor.
        got = blk.recycled_hidden[0, 1:5, 0].long().tolist()
        assert got == [4, 5, 6, 7]

    def test_positions_never_run_past_the_original_block(self):
        for a in range(0, 7):
            blk = self._build(acceptance=a, block_size=8)
            valid = blk.noise_original_positions[0][blk.valid_mask[0]]
            assert valid.numel() == max(0, positions_to_redraft(a, 8))
            if valid.numel():
                assert valid.min().item() == a + 2
                assert valid.max().item() == 7

    def test_position_in_block_measures_from_the_new_anchor(self):
        blk = self._build(acceptance=3, block_size=8)
        assert blk.position_in_block[0].tolist() == list(range(8))

    def test_per_block_acceptance_is_independent(self):
        n = 2 * 8
        prenorm = torch.zeros(1, n, 4)
        prenorm[..., 0] = torch.arange(n).float()
        blk = build_correction_block(
            prenorm,
            torch.arange(n).view(1, n),
            torch.tensor([[1, 4]]),
            torch.tensor([[0, 8]]),
            block_size=8,
            mask_token_id=0,
        )
        assert blk.valid_mask[0, :8].sum().item() == positions_to_redraft(1, 8)
        assert blk.valid_mask[0, 8:].sum().item() == positions_to_redraft(4, 8)
        # second block's recycled slots read from the second block only (>= 8)
        assert blk.recycled_hidden[0, 8:, 0][blk.valid_mask[0, 8:]].min().item() >= 8

    def test_rejects_ragged_length(self):
        with pytest.raises(ValueError, match="not a multiple of block_size"):
            build_correction_block(
                torch.zeros(1, 7, 4),
                torch.zeros(1, 7, dtype=torch.long),
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 1, dtype=torch.long),
                block_size=4,
                mask_token_id=0,
            )


class TestInputMixer:
    def test_recycled_passes_through_at_init(self):
        mixer = RecycledInputMixer(HIDDEN, 1e-6)
        recycled = torch.randn(2, BLOCK, HIDDEN)
        embed = torch.randn(2, BLOCK, HIDDEN)
        is_recycled = torch.tensor([[False, True, True, True]] * 2)
        out = mixer(embed, recycled, is_recycled)
        assert torch.allclose(out[is_recycled], recycled[is_recycled], atol=1e-5)
        assert torch.allclose(out[~is_recycled], embed[~is_recycled], atol=1e-5)

    def test_normalize_recycled_changes_residual_scale(self):
        """The ablation knob actually does something (and is off by default)."""
        recycled = torch.randn(1, BLOCK, HIDDEN) * 50.0
        embed = torch.zeros(1, BLOCK, HIDDEN)
        is_recycled = torch.ones(1, BLOCK, dtype=torch.bool)
        plain = RecycledInputMixer(HIDDEN, 1e-6, normalize_recycled=False)
        normed = RecycledInputMixer(HIDDEN, 1e-6, normalize_recycled=True)
        assert plain(embed, recycled, is_recycled).abs().mean() > 10.0
        assert normed(embed, recycled, is_recycled).abs().mean() < 2.0

    def test_masks_only_mode(self):
        mixer = RecycledInputMixer(HIDDEN, 1e-6)
        embed = torch.randn(1, BLOCK, HIDDEN)
        assert torch.allclose(mixer(embed, None, None), embed, atol=1e-5)

    def test_requires_is_recycled(self):
        mixer = RecycledInputMixer(HIDDEN, 1e-6)
        with pytest.raises(ValueError, match="is_recycled"):
            mixer(torch.randn(1, BLOCK, HIDDEN), torch.randn(1, BLOCK, HIDDEN), None)


class TestModel:
    def test_depth_is_forced_to_one(self, model):
        assert len(model.layers) == 1
        assert model.config.num_hidden_layers == 1

    def test_caller_config_is_not_mutated(self, tiny_config, correction_config):
        CorrectionDraftModel(tiny_config, correction_config)
        assert tiny_config.num_hidden_layers == 3

    def test_inherits_the_parent_target_layer_ids(self, tiny_config, correction_config):
        parent = DFlash2DraftModel(tiny_config, correction_config.dflash2)
        model = CorrectionDraftModel(tiny_config, correction_config)
        assert model.target_layer_ids == parent.target_layer_ids
        assert model.fc.weight.shape == parent.fc.weight.shape

    def test_forward_shapes(self, model, tiny_config):
        B, T, N = 2, 10, BLOCK
        target_hidden = torch.randn(B, T, len(model.target_layer_ids) * HIDDEN)
        out = model(
            position_ids=torch.arange(T + N).unsqueeze(0).expand(B, -1),
            attention_mask=torch.zeros(B, 1, N, T + N),
            noise_embedding=torch.randn(B, N, HIDDEN),
            recycled_hidden=torch.randn(B, N, HIDDEN),
            is_recycled=torch.tensor([[False, True, True, True]] * B),
            target_hidden=target_hidden,
        )
        assert out.shape == (B, N, HIDDEN)

    def test_return_prenorm_gives_both(self, model):
        B, T, N = 1, 10, BLOCK
        kwargs = dict(
            position_ids=torch.arange(T + N).unsqueeze(0),
            attention_mask=torch.zeros(B, 1, N, T + N),
            noise_embedding=torch.randn(B, N, HIDDEN),
            target_hidden=torch.randn(B, T, len(model.target_layer_ids) * HIDDEN),
        )
        normed, prenorm = model(**kwargs, return_prenorm=True)
        assert normed.shape == prenorm.shape == (B, N, HIDDEN)
        assert not torch.allclose(normed, prenorm)

    def test_rejects_ragged_noise_length(self, model):
        with pytest.raises(ValueError, match="not a multiple of block_size"):
            model(
                position_ids=torch.arange(10 + 3).unsqueeze(0),
                noise_embedding=torch.randn(1, 3, HIDDEN),
                target_hidden=torch.randn(1, 10, len(model.target_layer_ids) * HIDDEN),
            )

    def test_warm_start_makes_it_the_parents_next_layer(self, tiny_config, correction_config):
        parent = DFlash2DraftModel(tiny_config, correction_config.dflash2).eval()
        model = CorrectionDraftModel(tiny_config, correction_config).eval()
        copied, _ = model.warm_start_from_parent(parent)
        assert copied > 0
        last = parent.layers[-1].state_dict()
        for key, value in model.layers[0].state_dict().items():
            assert torch.equal(value, last[key]), key
        # the mixer is not in the parent, so it must still be identity
        assert torch.allclose(
            model.input_mixer.recycled_proj.weight, torch.eye(HIDDEN), atol=1e-6
        )

    def test_is_exactly_the_parents_next_layer(self, tiny_config, correction_config):
        """The design claim: identity mixer + warm start from the parent's last
        layer means an all-recycled correction pass is bit-for-bit the layer that
        would have come next in the parent's own stack."""
        parent = DFlash2DraftModel(tiny_config, correction_config.dflash2).eval()
        model = CorrectionDraftModel(tiny_config, correction_config).eval()
        model.warm_start_from_parent(parent)

        B, T, N = 1, 12, BLOCK
        target_hidden = torch.randn(B, T, len(parent.target_layer_ids) * HIDDEN)
        position_ids = torch.arange(T + N).unsqueeze(0)
        attention_mask = torch.zeros(B, 1, N, T + N)
        layer_kwargs = dict(attention_mask=attention_mask, position_ids=position_ids)

        with torch.no_grad():
            fused = parent._apply_target_hidden_fusion(target_hidden)
            hidden = torch.randn(B, N, HIDDEN)
            pos_emb = parent.rotary_emb(hidden, position_ids)
            for layer in parent.layers[:-1]:
                hidden = layer(
                    hidden_states=hidden, target_hidden=fused,
                    position_embeddings=pos_emb, **layer_kwargs,
                )
            expected = parent.norm(
                parent.layers[-1](
                    hidden_states=hidden, target_hidden=fused,
                    position_embeddings=pos_emb, **layer_kwargs,
                )
            )
            got = model(
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise_embedding=torch.zeros_like(hidden),
                recycled_hidden=hidden,
                is_recycled=torch.ones(B, N, dtype=torch.bool),
                target_hidden=target_hidden,
            )
        assert torch.allclose(got, expected, atol=1e-5)

    def test_round1_to_round2_pipeline(self, tiny_config, correction_config):
        """Round 1 (DFlash 2, return_prenorm) -> build_correction_block -> round 2."""
        parent = DFlash2DraftModel(tiny_config, correction_config.dflash2).eval()
        model = CorrectionDraftModel(tiny_config, correction_config).eval()
        model.warm_start_from_parent(parent)
        embed = torch.nn.Embedding(1000, HIDDEN)
        lm_head = MockLMHead(HIDDEN, 1000)

        B, T, N = 1, 12, BLOCK
        anchor_pos = 8
        target_hidden = torch.randn(B, T, len(parent.target_layer_ids) * HIDDEN)
        position_ids = torch.cat(
            [torch.arange(T), torch.arange(anchor_pos, anchor_pos + N)]
        ).unsqueeze(0)
        attention_mask = torch.zeros(B, 1, N, T + N)

        with torch.no_grad():
            _, prenorm = parent(
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise_embedding=torch.randn(B, N, HIDDEN),
                target_hidden=target_hidden,
                return_prenorm=True,
            )
            accepted = 0
            blk = build_correction_block(
                prenorm,
                torch.randint(0, 1000, (B, N)),
                torch.tensor([[accepted]]),
                torch.tensor([[anchor_pos]]),
                block_size=BLOCK,
                mask_token_id=999,
            )
            round2_position_ids = torch.cat(
                [torch.arange(T), blk.noise_original_positions[0]]
            ).unsqueeze(0)
            out = model(
                position_ids=round2_position_ids,
                attention_mask=attention_mask,
                noise_embedding=embed(blk.noise_token_ids),
                recycled_hidden=blk.recycled_hidden,
                is_recycled=blk.is_recycled,
                target_hidden=target_hidden,
            )
            tokens = model.propose(out, blk.noise_token_ids[:, 0], lm_head, embed)

        assert blk.valid_mask.sum().item() == positions_to_redraft(accepted, BLOCK)
        assert out.shape == (B, N, HIDDEN)
        assert tokens.shape == (B, N)

    def test_uses_dflash2s_loss_unchanged(self, model):
        """No correction-specific weighting: DFlash's flat exp(-(k-1)/gamma) is
        the whole story, since every supervised slot in the narrow variant is a
        recycled one."""
        assert type(model).compute_loss is DFlash2DraftModel.compute_loss

    def test_propose_and_loss_are_inherited(self, model, dummy_draft_batch):
        lm_head = MockLMHead(HIDDEN, 1000)
        embed = torch.nn.Embedding(1000, HIDDEN)
        draft_hidden = torch.randn(2, 8, HIDDEN)
        tokens = model.propose(draft_hidden, torch.zeros(2, dtype=torch.long), lm_head, embed)
        assert tokens.shape == (2, 8)
        loss = model.compute_loss(
            draft_hidden,
            torch.randint(0, 1000, (2, 32)),
            dummy_draft_batch,
            lm_head,
            embed_tokens=embed,
        )
        assert loss.requires_grad and torch.isfinite(loss)


class TestSimulateAcceptance:
    def _sim(self, drafted, committed, block_size=8):
        return simulate_acceptance(
            torch.tensor([drafted]), torch.tensor([committed]), block_size
        )

    def test_matches_the_eval_time_rule(self):
        """Same cumprod-of-prefix rule dflash_generate uses."""
        # slot:        0  1  2  3  4  5  6  7   (slot 0 is the anchor, skipped)
        drafted = [9, 1, 2, 3, 9, 5, 6, 7]
        committed = [0, 1, 2, 3, 4, 5, 6, 7]
        assert self._sim(drafted, committed).item() == 3

    def test_anchor_slot_is_ignored(self):
        assert self._sim([0, 1, 2, 3, 4, 5, 6, 7], [9, 1, 2, 3, 4, 5, 6, 7]).item() == 7

    def test_a_later_match_does_not_resurrect_a_rejected_prefix(self):
        assert self._sim([0, 9, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]).item() == 0

    def test_per_block(self):
        drafted = torch.tensor([[0, 1, 9, 3, 0, 1, 2, 3]])
        committed = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        assert simulate_acceptance(drafted, committed, 4).tolist() == [[1, 3]]

    def test_feeds_the_block_builder(self):
        """The two halves compose: simulated a -> suffix that gets trained."""
        drafted = torch.tensor([[0, 1, 2, 9, 9, 9, 9, 9]])
        committed = torch.arange(8).view(1, 8)
        a = simulate_acceptance(drafted, committed, 8)
        assert a.item() == 2
        blk = build_correction_block(
            torch.zeros(1, 8, 4), committed, a, torch.zeros(1, 1, dtype=torch.long),
            block_size=8, mask_token_id=0,
        )
        assert blk.valid_mask.sum().item() == positions_to_redraft(2, 8)

    def test_rejects_ragged_length(self):
        with pytest.raises(ValueError, match="not a multiple of block_size"):
            simulate_acceptance(torch.zeros(1, 7, dtype=torch.long),
                                torch.zeros(1, 7, dtype=torch.long), 4)
