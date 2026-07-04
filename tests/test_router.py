import pytest
pytestmark = pytest.mark.router

"""Basic shape and gradient tests for the ContextRouter."""

import torch
import pytest

from midas.config import RouterConfig
from midas.models import ContextRouter


@pytest.fixture
def router():
    config = RouterConfig(hidden_size=256, num_layers=5, rank=64)
    return ContextRouter(config)


@pytest.fixture
def dummy_input():
    B, T, H, N = 2, 10, 256, 5
    return [torch.randn(B, T, H) for _ in range(N)]


def test_output_shape(router, dummy_input):
    context, alpha = router(dummy_input)
    B, T, H = 2, 10, 256
    assert context.shape == (B, T, H)
    assert alpha.shape == (B, T, 5)


def test_routing_weights_sum_to_one(router, dummy_input):
    _, alpha = router(dummy_input)
    sums = alpha.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_routing_weights_non_negative(router, dummy_input):
    _, alpha = router(dummy_input)
    assert (alpha >= 0).all()


def test_gradient_flows_through_gamma(router, dummy_input):
    context, _ = router(dummy_input)
    loss = context.sum()
    loss.backward()
    assert router.log_gamma.grad is not None
    assert router.log_gamma.grad.abs() > 0


def test_gradient_flows_through_all_params(router, dummy_input):
    context, _ = router(dummy_input)
    loss = context.sum()
    loss.backward()

    for name, param in router.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_zero_gamma_gives_uniform_routing():
    config = RouterConfig(hidden_size=256, num_layers=5, rank=64, gamma_init=1e-10)
    router = ContextRouter(config)

    B, T, H, N = 2, 10, 256, 5
    inputs = [torch.randn(B, T, H) for _ in range(N)]

    _, alpha = router(inputs)

    # With gamma ≈ 0, routing should be near-uniform (from prior)
    expected = 1.0 / N
    assert torch.allclose(alpha, torch.full_like(alpha, expected), atol=0.05)


def test_different_inputs_give_different_routing():
    config = RouterConfig(hidden_size=256, num_layers=5, rank=64, gamma_init=1.0)
    router = ContextRouter(config)

    B, T, H, N = 2, 10, 256, 5
    inputs_a = [torch.randn(B, T, H) for _ in range(N)]
    inputs_b = [torch.randn(B, T, H) * 5 for _ in range(N)]

    _, alpha_a = router(inputs_a)
    _, alpha_b = router(inputs_b)

    # Different inputs should produce different routing
    assert not torch.allclose(alpha_a, alpha_b, atol=1e-3)


# ---------------------------------------------------------------------------
# ContextRouter with use_fc=False
# ---------------------------------------------------------------------------


@pytest.fixture
def router_nofc():
    config = RouterConfig(hidden_size=256, num_layers=5, rank=64, use_fc=False)
    return ContextRouter(config)


def test_nofc_has_no_fc_layer(router_nofc):
    assert not hasattr(router_nofc, "fc")


def test_nofc_output_shape(router_nofc, dummy_input):
    context, alpha = router_nofc(dummy_input)
    B, T, H = 2, 10, 256
    assert context.shape == (B, T, H)
    assert alpha.shape == (B, T, 5)


def test_nofc_routing_weights_sum_to_one(router_nofc, dummy_input):
    _, alpha = router_nofc(dummy_input)
    sums = alpha.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_nofc_gradient_flows_through_all_params(router_nofc, dummy_input):
    context, _ = router_nofc(dummy_input)
    loss = context.sum()
    loss.backward()

    for name, param in router_nofc.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
