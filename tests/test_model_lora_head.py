"""Tests for the LM-head LoRA adapter variant.

`test_identity_at_init` is the load-bearing one: it pins that this model is
*exactly* DFlash at step 0, which is what makes it a matched arm against the
baseline rather than a differently-initialized model.
"""

import pytest
import torch

from midas.models.model import DFlashDraftModel
from midas.models.model_lora_head import DFlashDraftModelLoRAHead, LMHeadAdapter
from tests.conftest import make_tiny_qwen3_config

pytestmark = pytest.mark.model

RANK = 64


@pytest.fixture
def lora_model(tiny_config):
    return DFlashDraftModelLoRAHead(tiny_config, rank=RANK).eval()


def _inputs(config, B=2, T=16):
    """One forward's worth of inputs: T context positions, two blocks of noise."""
    n = config.block_size * 2
    # extract_target_hidden concatenates one target layer per draft layer.
    ctx_width = config.hidden_size * config.num_hidden_layers
    return {
        "position_ids": torch.cat(
            [torch.arange(T).unsqueeze(0).expand(B, -1), torch.arange(T, T + n).unsqueeze(0).expand(B, -1)],
            dim=1,
        ),
        "noise_embedding": torch.randn(B, n, config.hidden_size),
        "target_hidden": torch.randn(B, T, ctx_width),
    }


class TestAdapter:
    def test_identity_at_init(self):
        adapter = LMHeadAdapter(128, RANK)
        h = torch.randn(3, 7, 128)
        torch.testing.assert_close(adapter(h), h)

    def test_down_is_not_zero(self):
        """Zeroing both matrices would leave the pair with no gradient path."""
        assert LMHeadAdapter(128, RANK).down.weight.abs().sum() > 0

    def test_applies_once_trained(self):
        adapter = LMHeadAdapter(128, RANK)
        torch.nn.init.normal_(adapter.up.weight, std=0.1)
        h = torch.randn(3, 7, 128)
        assert not torch.allclose(adapter(h), h)

    def test_parameter_count(self):
        adapter = LMHeadAdapter(2560, 64)
        assert sum(p.numel() for p in adapter.parameters()) == 327_680


class TestModel:
    def test_is_exactly_dflash_at_init(self, tiny_config):
        """Same seed, same weights: the two models must agree bit for bit."""
        torch.manual_seed(0)
        base = DFlashDraftModel(tiny_config).eval()
        torch.manual_seed(0)
        lora = DFlashDraftModelLoRAHead(tiny_config, rank=RANK).eval()
        lora.load_state_dict(base.state_dict(), strict=False)

        args = _inputs(tiny_config)
        with torch.no_grad():
            torch.testing.assert_close(lora(**args), base(**args))

    def test_adapter_survives_post_init(self, lora_model):
        """post_init() runs inside super().__init__, before the adapter exists."""
        assert lora_model.lm_head_adapter.up.weight.abs().sum() == 0
        assert lora_model.lm_head_adapter.down.weight.abs().sum() > 0

    def test_adapter_changes_output_once_trained(self, tiny_config, lora_model):
        args = _inputs(tiny_config)
        with torch.no_grad():
            before = lora_model(**args)
            torch.nn.init.normal_(lora_model.lm_head_adapter.up.weight, std=0.1)
            after = lora_model(**args)
        assert not torch.allclose(before, after)

    def test_only_addition_is_the_adapter(self, tiny_config):
        base = set(DFlashDraftModel(tiny_config).state_dict())
        lora = set(DFlashDraftModelLoRAHead(tiny_config, rank=RANK).state_dict())
        assert lora - base == {"lm_head_adapter.down.weight", "lm_head_adapter.up.weight"}
        assert base - lora == set()

    def test_gradients_reach_both_matrices(self, tiny_config, lora_model, mock_lm_head):
        lora_model.train()
        out = lora_model(**_inputs(tiny_config))
        mock_lm_head(out).sum().backward()
        assert lora_model.lm_head_adapter.up.weight.grad.abs().sum() > 0
        # `down` only receives gradient once `up` is non-zero, as standard for LoRA.
        torch.nn.init.normal_(lora_model.lm_head_adapter.up.weight, std=0.1)
        lora_model.zero_grad()
        mock_lm_head(lora_model(**_inputs(tiny_config))).sum().backward()
        assert lora_model.lm_head_adapter.down.weight.grad.abs().sum() > 0

    def test_state_dict_round_trip(self, tiny_config):
        trained = DFlashDraftModelLoRAHead(tiny_config, rank=RANK)
        torch.nn.init.normal_(trained.lm_head_adapter.up.weight, std=0.1)
        fresh = DFlashDraftModelLoRAHead(tiny_config, rank=RANK)
        fresh.load_state_dict(trained.state_dict())
        torch.testing.assert_close(
            fresh.lm_head_adapter.up.weight, trained.lm_head_adapter.up.weight
        )
