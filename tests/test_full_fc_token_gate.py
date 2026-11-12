import torch
from torch import nn

from dynamic_dflash.core.full_fc_token_gate import FullFCTokenGateProjector
from dynamic_dflash.core.model import RouterConfig
from dynamic_dflash.core.scratch_dflash import NativeFCProjector


class Draft(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.target_layer_ids = (1, 9, 17, 25, 33)
        self.fc = nn.Linear(5 * hidden_size, hidden_size, bias=False)
        self.hidden_norm = nn.Identity()


def test_initial_parity_and_gate_gradients():
    torch.manual_seed(12012)
    size = 8
    config = RouterConfig(hidden_size=size)
    draft = Draft(size)
    native = NativeFCProjector(draft, config)
    gated = FullFCTokenGateProjector(draft, config)
    states = [torch.randn(2, 3, size) for _ in range(config.num_target_layers)]
    native_output = native(states)
    gated_output, gains = gated(states, return_weights=True)
    torch.testing.assert_close(gains, torch.ones_like(gains), rtol=0, atol=0)
    torch.testing.assert_close(gated_output, native_output, rtol=0, atol=0)
    assert draft.fc.weight.shape == (size, 5 * size)
    gated_output.square().sum().backward()
    assert gated.gate.weight.grad is not None
    assert gated.gate.weight.grad.abs().sum().item() > 0
    assert draft.fc.weight.grad is not None
    assert draft.fc.weight.grad.abs().sum().item() > 0
