"""Shared fixtures for the midas test suite."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from midas.config import DFlareConfig, RouterConfig
from midas.data import DraftTrainingBatch


def make_tiny_qwen3_config(
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=512,
    vocab_size=1000,
    head_dim=32,
    num_target_layers=8,
    block_size=4,
) -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
        max_position_embeddings=2048,
    )
    config.dflash_config = {}
    config.block_size = block_size
    config.num_target_layers = num_target_layers
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def tiny_config():
    return make_tiny_qwen3_config()


@pytest.fixture
def tiny_router_config():
    return RouterConfig(hidden_size=128, num_layers=2, rank=32, gamma_init=0.01, temperature_init=1.0)


@pytest.fixture
def draft_model(tiny_config):
    from midas.models.model import DFlashDraftModel
    model = DFlashDraftModel(tiny_config)
    model.eval()
    return model


@pytest.fixture
def tiny_router_config_nofc():
    return RouterConfig(hidden_size=128, num_layers=2, rank=32, gamma_init=0.01, temperature_init=1.0, use_fc=False)


@pytest.fixture
def tiny_dflare_config():
    return DFlareConfig(
        hidden_size=128,
        num_draft_layers=2,
        num_target_layers=3,
        gamma_initial=4.5,
        gamma_increment_per_epoch=1.0,
    )


@pytest.fixture
def dflare_model(tiny_config, tiny_dflare_config):
    from midas.models.model_dflare import DFlareDraftModel

    model = DFlareDraftModel(tiny_config, dflare_config=tiny_dflare_config)
    model.eval()
    return model


@pytest.fixture
def routed_model(tiny_config, tiny_router_config):
    from midas.models.model_context_router import DFlashDraftModelRouted
    model = DFlashDraftModelRouted(tiny_config, router_config=tiny_router_config)
    model.eval()
    return model


@pytest.fixture
def routed_model_nofc(tiny_config, tiny_router_config_nofc):
    from midas.models.model_context_router import DFlashDraftModelRouted
    model = DFlashDraftModelRouted(tiny_config, router_config=tiny_router_config_nofc)
    model.eval()
    return model


@pytest.fixture
def dummy_draft_batch():
    """A minimal DraftTrainingBatch for testing loss computation."""
    B, N = 2, 8  # 2 anchors * block_size 4
    T = 32
    block_size = 4
    num_anchors = 2

    noise_token_ids = torch.randint(0, 1000, (B, N))
    position_ids = torch.cat([
        torch.arange(T).unsqueeze(0).expand(B, -1),
        torch.arange(T, T + N).unsqueeze(0).expand(B, -1),
    ], dim=1)

    position_in_block = torch.tensor([0, 1, 2, 3] * num_anchors).unsqueeze(0).expand(B, -1)
    loss_mask = position_in_block > 0

    noise_original_positions = torch.stack([
        torch.arange(10, 10 + N),
        torch.arange(10, 10 + N),
    ])

    attention_mask = torch.zeros(B, 1, N, T + N)

    noise_block_anchor = torch.stack([
        torch.tensor([10, 10, 10, 10, 14, 14, 14, 14]),
        torch.tensor([10, 10, 10, 10, 14, 14, 14, 14]),
    ])

    return DraftTrainingBatch(
        noise_token_ids=noise_token_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        noise_original_positions=noise_original_positions,
        loss_mask=loss_mask,
        position_in_block=position_in_block,
        noise_block_anchor=noise_block_anchor,
    )


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Create a temporary JSONL file with sample data."""
    data = [
        {"prompt": "What is 2+2?", "response": "The answer is 4."},
        {"prompt": "Explain gravity.", "response": "Gravity is a force that attracts objects with mass toward each other."},
        {"prompt": "Hello!", "response": "Hi there! How can I help you today?"},
        {"prompt": "Write a short poem.", "response": "Roses are red, violets are blue, code is my art, and bugs are my glue."},
        {"prompt": "What is Python?", "response": "Python is a high-level programming language known for its readability and versatility."},
    ]
    path = tmp_path / "test_data.jsonl"
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    return str(path)


class MockLMHead(nn.Module):
    """A simple linear layer mimicking a language model head."""
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def mock_lm_head():
    return MockLMHead(128, 1000)
