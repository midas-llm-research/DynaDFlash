from __future__ import annotations
import contextlib
import hashlib
import json
import math
import mmap
import os
import random
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
from dynamic_dflash.core.blocks import BLOCK_SIZE, anchor_positions, build_blocks, weighted_block_loss
from dynamic_dflash.core.model import extract_decoder_states
PINNED_FILE_SHA256 = {'target_config': 'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30', 'draft_config': '9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102', 'draft_modeling': 'd2b61a1fa5d469830c2285b552d75773274943fae5ac66dc2713beddea338528'}

def verify_model_files(target_path: Path, draft_path: Path) -> None:
    files = {'target_config': target_path / 'config.json', 'draft_config': draft_path / 'config.json', 'draft_modeling': draft_path / 'modeling_dflash.py'}
    for key, path in files.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != PINNED_FILE_SHA256[key]:
            raise RuntimeError(f'pinned model file mismatch: {path}')

class IndexedJsonl:

    def __init__(self, path: Path):
        self.path = path
        self.file_descriptor = os.open(path, os.O_RDONLY)
        self.index_descriptor = os.open(path.with_suffix('.idx'), os.O_RDONLY)
        self.size = os.stat(self.index_descriptor).st_size // 8
        if os.stat(self.index_descriptor).st_size % 8:
            raise ValueError('corrupt offset index')
        self.offsets = mmap.mmap(self.index_descriptor, 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        start = struct.unpack_from('<Q', self.offsets, index * 8)[0]
        if index + 1 < self.size:
            stop = struct.unpack_from('<Q', self.offsets, (index + 1) * 8)[0]
            line = os.pread(self.file_descriptor, stop - start, start)
        else:
            chunks = []
            offset = start
            while True:
                chunk = os.pread(self.file_descriptor, 4096, offset)
                if not chunk:
                    break
                chunks.append(chunk)
                offset += len(chunk)
                if b'\n' in chunk:
                    break
            line = b''.join(chunks).split(b'\n', 1)[0]
        return json.loads(line)

    def close(self) -> None:
        self.offsets.close()
        os.close(self.index_descriptor)
        os.close(self.file_descriptor)

def stratified_validation_indices(validation: IndexedJsonl, source_counts: dict[str, int], limit: int) -> list[int]:
    limit = min(limit, len(validation))
    if limit < 1:
        raise ValueError('validation selection must be nonempty')
    source_counts = {source: count for source, count in source_counts.items() if count > 0}
    total = sum(source_counts.values())
    quotas = {source: count * limit // total for source, count in source_counts.items()}
    remainder = limit - sum(quotas.values())
    for source in sorted(source_counts, key=lambda s: (-(source_counts[s] * limit % total), s))[:remainder]:
        quotas[source] += 1
    by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index in range(len(validation)):
        row = validation[index]
        by_source[row['source']].append((row['prompt_hash'], index))
    if set(by_source) != set(source_counts):
        raise RuntimeError('validation sources differ from frozen manifest')
    selected = []
    for source in sorted(source_counts):
        entries = sorted(by_source[source])
        if len(entries) != source_counts[source] or quotas[source] > len(entries):
            raise RuntimeError('validation source count mismatch')
        selected.extend((index for _, index in entries[:quotas[source]]))
    if len(selected) != limit:
        raise AssertionError('stratified validation count mismatch')
    return selected

def _ordered_indices(count: int, epoch: int, rank: int, world: int, seed: str) -> list[int]:
    indices = list(range(count))
    key = hashlib.sha256(f'{seed}\x00{epoch}'.encode()).digest()
    random.Random(int.from_bytes(key[:8], 'big')).shuffle(indices)
    usable = count - count % (world * 4)
    return indices[:usable][rank::world]

def _autocast(device: torch.device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()

def scheduler_for(optimizer: torch.optim.Optimizer, total_updates: int):
    warmup = max(1, round(total_updates * 0.04))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_updates - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

def _validation_context_points(record: dict[str, Any]) -> list[tuple[str, int]]:
    prompt = int(record['prompt_length'])
    length = len(record['input_ids'])
    boundary = int(record.get('think_boundary', -1))
    if record.get('mode') == 'non_thinking':
        if boundary != prompt:
            raise ValueError('non-thinking record has a reasoning span')
        return list(dict.fromkeys((('answer', position) for position in (prompt, (prompt + length - 1) // 2) if 0 < position < length - 1)))
    points = [('reasoning', prompt)]
    if boundary > prompt + 2:
        points.append(('reasoning', (prompt + boundary) // 2))
    if prompt < boundary < length - 1:
        points.append(('answer', boundary))
        if length - boundary > 2:
            points.append(('answer', (boundary + length - 1) // 2))
    return list(dict.fromkeys(((phase, position) for phase, position in points if 0 < position < length - 1)))

@torch.no_grad()
def heldout_context_acceptance(target: Any, model: Any, validation: IndexedJsonl, device: torch.device, *, limit: int=64, indices: list[int] | None=None) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    accepted: dict[str, list[int]] = defaultdict(list)
    try:
        for index in indices if indices is not None else range(min(limit, len(validation))):
            record = validation[index]
            for phase, prefix_length in _validation_context_points(record):
                prompt = torch.tensor(record['input_ids'][:prefix_length], device=device, dtype=torch.long)[None]
                with _autocast(device):
                    prefill = target(input_ids=prompt, output_hidden_states=True, use_cache=False, logits_to_keep=1, return_dict=True)
                    states = extract_decoder_states(prefill.hidden_states, model.projector.config)
                    context = model.projector(states)
                    anchor = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
                    if int(anchor.item()) == 151645:
                        accepted[phase].append(1)
                        continue
                    mask_ids = torch.full((1, BLOCK_SIZE), model.draft.mask_token_id, device=device, dtype=torch.long)
                    mask_ids[:, 0] = anchor[:, 0]
                    noise = target.model.embed_tokens(mask_ids)
                    positions = torch.arange(prompt.shape[1] + BLOCK_SIZE, device=device)[None]
                    hidden = model.forward_fused(context, noise, positions)
                    proposals = target.lm_head(hidden[:, 1:]).argmax(dim=-1)
                    block = torch.cat([anchor, proposals], dim=1)
                    verified = target(input_ids=torch.cat([prompt, block], dim=1), use_cache=False, logits_to_keep=BLOCK_SIZE, return_dict=True)
                posterior = verified.logits[:, :-1].argmax(dim=-1)
                matches = (proposals == posterior).int()
                count = 1 + int(matches.cumprod(dim=1).sum().item())
                eos_positions = (block[0, :count] == 151645).nonzero(as_tuple=False)
                if eos_positions.numel():
                    count = int(eos_positions[0, 0].item()) + 1
                accepted[phase].append(count)
    finally:
        model.train(was_training)
    values = [value for phase_values in accepted.values() for value in phase_values]
    if not values:
        raise RuntimeError('no valid held-out speculative contexts')
    return {'mean_accepted_length': sum(values) / len(values), 'by_phase': {phase: sum(values) / len(values) for phase, values in accepted.items() if values}, 'context_counts': {phase: len(values) for phase, values in accepted.items()}}

@torch.no_grad()
def heldout_weighted_loss(target: Any, model: Any, validation: IndexedJsonl, device: torch.device, *, limit: int=8, indices: list[int] | None=None) -> float:
    was_training = model.training
    model.eval()
    numerator_total = 0.0
    denominator_total = 0.0
    try:
        for index in indices if indices is not None else range(min(limit, len(validation))):
            record = validation[index]
            ids = torch.tensor(record['input_ids'], device=device, dtype=torch.long)
            anchors = anchor_positions(int(record['prompt_length']), ids.numel(), record['prompt_hash'], 0, max_anchors=8)
            batch = build_blocks(ids, anchors, model.draft.mask_token_id)
            with _autocast(device):
                output = target.model(input_ids=ids[None], output_hidden_states=True, use_cache=False, return_dict=True)
                states = extract_decoder_states(output.hidden_states, model.projector.config)
                context = model.projector(states)
                noise = target.model.embed_tokens(batch.token_ids)
                hidden = model.forward_fused(context, noise, batch.position_ids, batch.attention_mask)
                logits = target.lm_head(hidden)
                numerator, denominator = weighted_block_loss(logits, batch)
            numerator_total += float(numerator.item())
            denominator_total += float(denominator.item())
    finally:
        model.train(was_training)
    return numerator_total / denominator_total

def _rng_snapshot() -> dict[str, Any]:
    return {'python': random.getstate(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def _rng_restore(snapshot: dict[str, Any]) -> None:
    random.setstate(snapshot['python'])
    torch.set_rng_state(snapshot['torch'])
    if snapshot['cuda'] is not None:
        torch.cuda.set_rng_state(snapshot['cuda'])
