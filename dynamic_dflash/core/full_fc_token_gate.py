from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from dynamic_dflash.core.forward import forward_fused
from dynamic_dflash.core.model import RouterConfig, _rms_no_weight
from dynamic_dflash.core.scratch_dflash import NativeFCProjector

class FullFCTokenGateProjector(NativeFCProjector):

    def __init__(self, draft: nn.Module, config: RouterConfig):
        super().__init__(draft, config)
        self.gate = nn.Linear(config.hidden_size, len(config.selected_layers), bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, states: Sequence[Tensor], *, return_weights: bool=False) -> Tensor | tuple[Tensor, Tensor]:
        if len(states) != self.config.num_target_layers:
            raise ValueError('wrong target-state count')
        selected = [states[i] for i in self.config.selected_layers]
        shape = selected[0].shape
        if len(shape) != 3 or shape[-1] != self.config.hidden_size or any((state.shape != shape for state in selected)):
            raise ValueError('target states have inconsistent geometry')
        query = _rms_no_weight(selected[-1], self.config.rms_epsilon).float()
        gains = len(selected) * torch.softmax(self.gate(query), dim=-1)
        scaled = torch.cat([state.float() * gains[..., i:i + 1] for i, state in enumerate(selected)], dim=-1)
        fused = self.draft.hidden_norm(self.draft.fc(scaled))
        return (fused, gains) if return_weights else fused

class FullFCTokenGateDFlash(nn.Module):
    forward_fused = forward_fused

    def __init__(self, draft: nn.Module, config: RouterConfig | None=None, *, activation_checkpointing: bool=True):
        super().__init__()
        config = config or RouterConfig()
        if len(draft.layers) != 5:
            raise ValueError('expected five DFlash draft blocks')
        self.draft = draft
        self.draft.requires_grad_(False)
        self.projector = FullFCTokenGateProjector(draft, config)
        self.draft.fc.requires_grad_(True)
        self.draft.hidden_norm.requires_grad_(True)
        self.draft.norm.requires_grad_(True)
        for layer in self.draft.layers:
            layer.requires_grad_(True)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, states: Sequence[Tensor], noise_embedding: Tensor, position_ids: Tensor, attention_mask: Tensor | None=None) -> Tensor:
        return self.forward_fused(self.projector(states), noise_embedding, position_ids, attention_mask)
