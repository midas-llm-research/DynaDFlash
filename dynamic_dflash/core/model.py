from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import torch
from torch import Tensor

@dataclass(frozen=True)
class RouterConfig:
    hidden_size: int = 4096
    num_target_layers: int = 36
    selected_layers: tuple[int, ...] = (1, 9, 17, 25, 33)
    num_heads: int = 4
    head_dim: int = 32
    rms_epsilon: float = 1e-06

    @property
    def rank(self) -> int:
        return self.num_heads * self.head_dim

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.num_target_layers <= 0:
            raise ValueError('invalid target geometry')
        if not self.selected_layers or len(set(self.selected_layers)) != len(self.selected_layers):
            raise ValueError('selected layers must be unique and nonempty')
        if min(self.selected_layers) < 0 or max(self.selected_layers) >= self.num_target_layers:
            raise ValueError('selected layer is out of range')
        if self.rank <= 0:
            raise ValueError('router rank must be positive')

def _rms_no_weight(x: Tensor, epsilon: float) -> Tensor:
    scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon)
    return x * scale.to(x.dtype)

def extract_decoder_states(output_hidden_states: Sequence[Tensor], config: RouterConfig) -> tuple[Tensor, ...]:
    if len(output_hidden_states) != config.num_target_layers + 1:
        raise ValueError('target returned unexpected hidden-state tuple length')
    return tuple((output_hidden_states[index + 1] for index in range(config.num_target_layers)))
