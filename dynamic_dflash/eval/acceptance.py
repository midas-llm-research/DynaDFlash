from __future__ import annotations
from typing import Any
import torch
import torch.nn.functional as F
from dynamic_dflash.core.blocks import BLOCK_SIZE
from dynamic_dflash.core.model import extract_decoder_states
from dynamic_dflash.training.common import _autocast

@torch.no_grad()
def score_context(target: Any, model: Any, input_ids: list[int], prefix_length: int, eos_token_id: int) -> dict[str, Any]:
    if not 0 < prefix_length < len(input_ids) - 1:
        raise ValueError('context has no valid future target token')
    device = next(target.parameters()).device
    prompt = torch.tensor(input_ids[:prefix_length], device=device, dtype=torch.long)[None]
    with _autocast(device):
        prefill = target(input_ids=prompt, output_hidden_states=True, use_cache=False, logits_to_keep=1, return_dict=True)
        states = extract_decoder_states(prefill.hidden_states, model.projector.config)
        if hasattr(model.projector, 'gains'):
            context = model.projector(states)
            weights = model.projector.gains(states)
        else:
            context, weights = model.projector(states, return_weights=True)
        anchor = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        route = weights[0, -1].float().cpu().tolist() if weights is not None else None
        if int(anchor.item()) == eos_token_id:
            return {'accepted_length': 1, 'accepted_drafts': 0, 'anchor_is_end': True, 'draft_target_greedy_nll': None, 'last_context_route': route}
        mask_ids = torch.full((1, BLOCK_SIZE), model.draft.mask_token_id, device=device, dtype=torch.long)
        mask_ids[:, 0] = anchor[:, 0]
        noise = target.model.embed_tokens(mask_ids)
        positions = torch.arange(prefix_length + BLOCK_SIZE, device=device)[None]
        hidden = model.forward_fused(context, noise, positions)
        draft_logits = target.lm_head(hidden[:, 1:]).float()
        proposals = draft_logits.argmax(dim=-1)
        block = torch.cat([anchor, proposals], dim=1)
        verified = target(input_ids=torch.cat([prompt, block], dim=1), use_cache=False, logits_to_keep=BLOCK_SIZE, return_dict=True)
        posterior = verified.logits[:, :-1].argmax(dim=-1)
    matches = (proposals == posterior).int()
    accepted = int(matches.cumprod(dim=1).sum().item())
    count = accepted + 1
    end_positions = (block[0, :count] == eos_token_id).nonzero(as_tuple=False)
    if end_positions.numel():
        count = int(end_positions[0, 0].item()) + 1
        accepted = count - 1
    nll = -F.log_softmax(draft_logits, dim=-1).gather(-1, posterior.unsqueeze(-1)).mean()
    return {'accepted_length': count, 'accepted_drafts': accepted, 'anchor_is_end': False, 'draft_target_greedy_nll': float(nll.item()), 'last_context_route': route}
