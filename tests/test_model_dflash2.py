"""Tests for midas/models/model_dflash2.py — DFlash 2."""

import pytest
import torch
import torch.nn as nn

from midas.config import DFlash2Config
from midas.models.model import DFlashDraftModel
from midas.models.model_dflash2 import (
    BlockLocalDynamicConv,
    CandidateSelector,
    DFlash2DraftModel,
    Qwen3DFlash2DecoderLayer,
)
from tests.conftest import MockLMHead, make_tiny_qwen3_config


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config()


@pytest.fixture
def d2_config():
    return DFlash2Config(selector_rank=16, selector_top_k=4, conv_group_size=16)


@pytest.fixture
def model(tiny_config, d2_config):
    return DFlash2DraftModel(tiny_config, d2_config).eval()


class TestBlockLocalConv:
    def _conv(self, hidden=32, block_size=4):
        return BlockLocalDynamicConv(hidden, kernel_size=2, group_size=16, block_size=block_size)

    def test_identity_at_init(self):
        conv = self._conv()
        x = torch.randn(2, 8, 32)
        out, dyn = conv.prepare(x)
        assert torch.allclose(out, x, atol=1e-6)
        assert torch.allclose(conv.finish(x, dyn), x, atol=1e-6)

    def test_tap_never_crosses_a_block_boundary(self):
        """The whole reason this differs from the reference implementation."""
        conv = self._conv(block_size=4)
        conv.base_kernel.data[0, 1] = 1.0          # switch on the predecessor tap
        x = torch.zeros(1, 8, 32)
        x[0, 3] = 1.0                              # last position of block 0
        out, _ = conv.prepare(x)
        assert out[0, 4].abs().max() == 0.0, "block 1 position 0 read block 0's tail"
        x2 = torch.zeros(1, 8, 32)
        x2[0, 4] = 1.0                             # first position of block 1
        out2, _ = conv.prepare(x2)
        assert out2[0, 5].abs().max() > 0.0, "within-block predecessor tap is dead"

    def test_rejects_ragged_length(self):
        conv = self._conv(block_size=4)
        with pytest.raises(ValueError, match="not a multiple of block_size"):
            conv.prepare(torch.randn(1, 7, 32))

    def test_param_count_formula(self):
        """hidden^2/4 per module for kernel_size=2, group_size=16."""
        hidden = 64
        conv = BlockLocalDynamicConv(hidden, 2, 16, 4)
        assert conv.kernel_projection.weight.numel() == hidden * hidden // 4


class TestCandidateSelector:
    def test_walk_only_returns_tokens_from_the_candidate_lists(self):
        torch.manual_seed(0)
        sel = CandidateSelector(hidden_size=8, rank=4, top_k=3)
        embed = nn.Embedding(50, 8)
        hidden, logits = torch.randn(2, 5, 8), torch.randn(2, 5, 50)
        path = sel.walk(hidden, logits, torch.tensor([1, 2]), embed)
        _, candidates = torch.topk(logits, 3, dim=-1)
        assert path.shape == (2, 5)
        assert (path.unsqueeze(-1) == candidates).any(-1).all()

    def test_reduces_to_argmax_when_bilinear_term_is_zero(self):
        sel = CandidateSelector(hidden_size=8, rank=4, top_k=3)
        for proj in (sel.predecessor_proj, sel.successor_proj, sel.hidden_projection):
            nn.init.zeros_(proj.weight)
        logits = torch.randn(2, 5, 50)
        path = sel.walk(torch.randn(2, 5, 8), logits, torch.tensor([1, 2]), nn.Embedding(50, 8))
        assert torch.equal(path, logits.argmax(-1))

    def test_predecessor_changes_the_choice(self):
        """Otherwise the selector is not doing anything the argmax wasn't."""
        torch.manual_seed(0)
        sel = CandidateSelector(hidden_size=8, rank=4, top_k=8)
        # scale the bilinear branch up so it can outvote near-tied unary logits
        for proj in (sel.predecessor_proj, sel.successor_proj, sel.hidden_projection):
            proj.weight.data.mul_(5.0)
        embed = nn.Embedding(50, 8)
        hidden = torch.randn(1, 4, 8)
        logits = torch.randn(1, 4, 50) * 0.01          # nearly tied candidates
        paths = {tuple(sel.walk(hidden, logits, torch.tensor([p]), embed)[0].tolist())
                 for p in range(50)}
        assert len(paths) > 1, "predecessor never changed the selected path"


class TestDFlash2Model:
    def test_layers_and_selector_present(self, model, tiny_config):
        assert all(isinstance(l, Qwen3DFlash2DecoderLayer) for l in model.layers)
        assert all(l.attention_conv is not None and l.mlp_conv is not None for l in model.layers)
        assert hasattr(model, "candidate_selector")

    def test_state_dict_is_base_plus_conv_and_selector(self, tiny_config, d2_config):
        base, d2 = DFlashDraftModel(tiny_config), DFlash2DraftModel(tiny_config, d2_config)
        assert set(base.state_dict()) - set(d2.state_dict()) == set()
        extra = set(d2.state_dict()) - set(base.state_dict())
        assert extra and all(
            ("conv" in k) or k.startswith("candidate_selector.") for k in extra
        )

    def test_identity_conv_reproduces_base_forward(self, tiny_config, d2_config):
        """Convs are no-ops at init, so DFlash 2 starts as exactly DFlash."""
        base = DFlashDraftModel(tiny_config).eval()
        d2 = DFlash2DraftModel(tiny_config, d2_config).eval()
        d2.load_state_dict({**d2.state_dict(), **base.state_dict()})
        B, T, H = 2, 16, tiny_config.hidden_size
        N = tiny_config.block_size * 2
        args = dict(
            position_ids=torch.arange(T + N).unsqueeze(0).expand(B, -1),
            noise_embedding=torch.randn(B, N, H),
            target_hidden=torch.randn(B, T, len(base.target_layer_ids) * H),
        )
        with torch.no_grad():
            assert torch.allclose(base(**args), d2(**args), atol=1e-5)

    def test_propose_requires_embeddings(self, model):
        with pytest.raises(ValueError, match="embedding table"):
            model.propose(torch.randn(1, 4, 128), torch.tensor([1]), MockLMHead(128, 1000))

    def test_propose_returns_valid_tokens(self, model, tiny_config):
        lm_head, embed = MockLMHead(128, 1000), nn.Embedding(1000, 128)
        out = model.propose(torch.randn(1, tiny_config.block_size, 128), torch.tensor([7]), lm_head, embed)
        assert out.shape == (1, tiny_config.block_size)
        assert (out >= 0).all() and (out < 1000).all()

    def test_compute_loss_and_gradients(self, model, dummy_draft_batch, d2_config):
        # vocab == top_k so recall@k is 1.0 and the selector term is always active
        # (from a random drafter, recall@k is ~k/V and the term would be skipped)
        vocab = d2_config.selector_top_k
        lm_head, embed = MockLMHead(128, vocab), nn.Embedding(vocab, 128)
        # run a real forward so the conv parameters are actually on the graph
        draft_hidden = model(
            position_ids=torch.arange(40).unsqueeze(0).expand(2, -1),
            noise_embedding=torch.randn(2, 8, 128),
            target_hidden=torch.randn(2, 32, len(model.target_layer_ids) * 128),
        )
        loss = model.compute_loss(
            draft_hidden,
            torch.randint(0, vocab, (2, 32)),
            dummy_draft_batch,
            lm_head,
            gamma=7.0,
            embed_tokens=embed,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert (model.candidate_selector.hidden_projection.weight.grad != 0).any()
        assert (model.layers[0].attention_conv.kernel_projection.weight.grad != 0).any()
        for key in ("loss_ce", "loss_selector", "recall_at_k", "recall_at_1"):
            assert key in model.log_metrics(loss=loss.item()), key

    def test_selector_loss_needs_embeddings(self, model, dummy_draft_batch, mock_lm_head):
        with pytest.raises(ValueError, match="embed_tokens"):
            model.compute_loss(
                torch.randn(2, 8, 128), torch.randint(0, 1000, (2, 32)),
                dummy_draft_batch, mock_lm_head, gamma=7.0,
            )

    def test_lambda_zero_matches_baseline_ce(self, tiny_config, dummy_draft_batch, mock_lm_head):
        from midas.models.model import compute_position_weighted_ce_loss

        m = DFlash2DraftModel(tiny_config, DFlash2Config(lambda_selector=0.0))
        dh, ids = torch.randn(2, 8, 128), torch.randint(0, 1000, (2, 32))
        assert torch.equal(
            compute_position_weighted_ce_loss(dh, ids, dummy_draft_batch, mock_lm_head, gamma=7.0),
            m.compute_loss(dh, ids, dummy_draft_batch, mock_lm_head, gamma=7.0),
        )
