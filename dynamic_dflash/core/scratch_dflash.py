from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import Tensor, nn
from dynamic_dflash.core.forward import forward_fused
from dynamic_dflash.core.model import RouterConfig

class NativeFCProjector(nn.Module):

    def __init__(self, draft: nn.Module, config: RouterConfig):
        super().__init__()
        config.validate()
        if tuple((int(i) for i in draft.target_layer_ids)) != config.selected_layers or not isinstance(draft.fc, nn.Linear) or draft.fc.bias is not None or (draft.fc.weight.shape != (config.hidden_size, len(config.selected_layers) * config.hidden_size)):
            raise ValueError('native DFlash context projection geometry differs')
        self.config = config
        object.__setattr__(self, 'draft', draft)

    def forward(self, states: Sequence[Tensor], *, return_weights: bool=False) -> Tensor | tuple[Tensor, None]:
        if len(states) != self.config.num_target_layers:
            raise ValueError('wrong target-state count')
        selected = [states[i] for i in self.config.selected_layers]
        shape = selected[0].shape
        if len(shape) != 3 or shape[-1] != self.config.hidden_size or any((state.shape != shape for state in selected)):
            raise ValueError('target states have inconsistent geometry')
        fused = self.draft.hidden_norm(self.draft.fc(torch.cat(selected, dim=-1).float()))
        return (fused, None) if return_weights else fused

class ScratchDFlash(nn.Module):
    forward_fused = forward_fused

    def __init__(self, draft: nn.Module, config: RouterConfig | None=None, *, activation_checkpointing: bool=True):
        super().__init__()
        config = config or RouterConfig()
        if len(draft.layers) != 5:
            raise ValueError('expected five native DFlash draft blocks')
        self.draft = draft
        self.draft.requires_grad_(False)
        self.projector = NativeFCProjector(draft, config)
        self.draft.fc.requires_grad_(True)
        self.draft.hidden_norm.requires_grad_(True)
        self.draft.norm.requires_grad_(True)
        for layer in self.draft.layers:
            layer.requires_grad_(True)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, states: Sequence[Tensor], noise_embedding: Tensor, position_ids: Tensor, attention_mask: Tensor | None=None) -> Tensor:
        return self.forward_fused(self.projector(states), noise_embedding, position_ids, attention_mask)
