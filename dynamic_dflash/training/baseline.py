from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import torch
import torch.distributed as dist
from dynamic_dflash.core.data import DATASET_REVISION
from dynamic_dflash.core.scratch_dflash import ScratchDFlash
from dynamic_dflash.training.loss import PAPER_BLOCK_SIZE, PAPER_GAMMA, one_sequence_loss
from dynamic_dflash.core.model import RouterConfig
from dynamic_dflash.core.pins import DRAFT_METADATA_SHA256, TARGET_METADATA_SHA256
from dynamic_dflash.training.common import IndexedJsonl, _ordered_indices, _rng_restore, _rng_snapshot, heldout_context_acceptance, heldout_weighted_loss, scheduler_for, stratified_validation_indices, verify_model_files
STAGE = 'scratch_native_dflash_paper_lr_oneepoch_branch_18000_nothink'
INITIALIZATION = 'random_from_pinned_dflash_config_seed_12012'
PUBLISHED_FC_SHA256 = '6746c787462f9dfbe0915d058f06a19140004e98d2d210bc29fbfac0ee602a1b'
BATCH_SIZE = 32
WORLD_SIZE = 2
EPOCH_UPDATES = 24750
SCHEDULE_UPDATES = 24750
SCHEDULE_WARMUP = 990
SCHEDULE_TRANSITION = 18000
AUTHORIZED_STOP = EPOCH_UPDATES
LEGACY_STOP = SCHEDULE_TRANSITION
LEGACY_SOURCE_SHA256 = '43f5e8312ea9870119f1ad424fc8902a6ac802eee9d6ffc3082819061ee34cc8'
SAVE_INTERVAL = 250

def schedule_multiplier(step: int, mode: str='original') -> float:
    if mode != 'original':
        raise ValueError('only the original one-epoch schedule is allowed')
    if step < SCHEDULE_WARMUP:
        return (step + 1) / SCHEDULE_WARMUP
    progress = min(1.0, (step - SCHEDULE_WARMUP) / (EPOCH_UPDATES - SCHEDULE_WARMUP))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def scheduler_for_continuation(optimizer, mode: str):
    if mode != 'original':
        raise ValueError('schedule must remain original')
    return scheduler_for(optimizer, EPOCH_UPDATES)

def code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    names = ('core/blocks.py', 'core/data.py', 'core/nonthinking_data.py', 'core/model.py', 'core/forward.py', 'core/scratch_dflash.py', 'core/full_fc_token_gate.py', 'training/common.py', 'training/loss.py', 'training/baseline.py', 'training/dynadflash.py')
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()

def verify_manifest(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != manifest.get('manifest_sha256') or manifest.get('dataset_revision') != DATASET_REVISION or manifest.get('mode') != 'non_thinking' or (manifest.get('template_mode') != 'enable_thinking=False') or ((manifest.get('pool_size'), manifest.get('train_size'), manifest.get('validation_size')) != (800000, 792000, 8000)):
        raise RuntimeError('requires the pinned non-thinking 792K/8K split')
    return manifest

def initial_probes(model: ScratchDFlash) -> dict[str, torch.Tensor]:
    named = dict(model.named_parameters())
    names = [f'draft.layers.{i}.{part}.weight' for i in range(5) for part in ('self_attn.k_proj', 'self_attn.v_proj', 'mlp.down_proj')]
    return {name: named[name].detach().flatten()[:4096].cpu().clone() for name in names}

def load_models(target_path: Path, draft_path: Path, device: torch.device):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    verify_model_files(target_path, draft_path)
    target = AutoModelForCausalLM.from_pretrained(target_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').eval().to(device)
    target.requires_grad_(False)
    draft_config = AutoConfig.from_pretrained(draft_path, local_files_only=True, trust_remote_code=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12012)
        draft = AutoModel.from_config(draft_config, trust_remote_code=True, attn_implementation='sdpa').float().to(device)
    cfg = RouterConfig(hidden_size=int(draft.config.hidden_size))
    if draft.__class__.__name__ != 'DFlashDraftModel' or len(draft.layers) != 5 or int(target.config.num_hidden_layers) != cfg.num_target_layers or (int(draft.block_size) != 16) or (tuple(draft.target_layer_ids) != cfg.selected_layers) or (draft.fc.weight.shape != (cfg.hidden_size, 5 * cfg.hidden_size)) or (draft.fc.bias is not None):
        raise RuntimeError('native DFlash target/draft geometry differs')
    random_fc_hash = hashlib.sha256(draft.fc.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if random_fc_hash == PUBLISHED_FC_SHA256:
        raise RuntimeError('draft FC was not randomly initialized')
    model = ScratchDFlash(draft, cfg).to(device)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if 'draft.fc.weight' not in trainable or len(trainable) != 58 or any((not name.startswith(('draft.fc.', 'draft.layers.', 'draft.norm.', 'draft.hidden_norm.')) for name in trainable)) or any((not any((name.startswith(f'draft.layers.{i}.') for name in trainable)) for i in range(5))) or any((p.requires_grad for p in target.parameters())):
        raise RuntimeError('native full-draft trainable set differs')
    return (target, model, random_fc_hash)

def metadata(manifest: dict, random_fc_hash: str, args: argparse.Namespace) -> dict:
    return {'stage': STAGE, 'initialization': INITIALIZATION, 'source_draft_metadata_sha256': DRAFT_METADATA_SHA256, 'source_target_metadata_sha256': TARGET_METADATA_SHA256, 'random_initial_fc_sha256': random_fc_hash, 'manifest_sha256': manifest['manifest_sha256'], 'training_code_sha256': code_sha256(), 'world_size': WORLD_SIZE, 'router_config': {'selected_layers': [1, 9, 17, 25, 33], 'projection': 'native_random_fc_20480_to_4096'}, 'run_config': {'total_updates': EPOCH_UPDATES, 'authorized_stop': AUTHORIZED_STOP, 'global_batch': BATCH_SIZE, 'per_gpu_batch': BATCH_SIZE // WORLD_SIZE, 'anchor_limit': args.anchor_limit, 'anchor_chunk': args.anchor_chunk, 'save_interval': SAVE_INTERVAL, 'validation_limit': args.validation_limit, 'draft_learning_rate': args.draft_learning_rate, 'dense_learning_rate': args.dense_learning_rate, 'objective': 'target_response_weighted_masked_block_v1', 'block_size': PAPER_BLOCK_SIZE, 'loss_decay_gamma': PAPER_GAMMA, 'activation_checkpointing': True, 'gradient_reduction': 'sum_of_global_batch_normalized_rank_gradients', 'sample_order': 'global_one_epoch_permutation_contiguous_batch_shards'}}

def _reduce_gradients(model: ScratchDFlash) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f'missing gradient: {name}')
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

def gradient_report(model: ScratchDFlash) -> dict:

    def norm(parameters) -> float:
        values = [p.grad.float().square().sum() for p in parameters if p.grad is not None]
        return float(torch.stack(values).sum().sqrt()) if values else 0.0
    return {'draft_layer_gradient_norms': [norm(layer.parameters()) for layer in model.draft.layers], 'context_kv_gradient_norms': [norm((layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight)) for layer in model.draft.layers], 'fc_gradient_norm': norm(model.draft.fc.parameters())}

def _assert_rank_parity(model: ScratchDFlash) -> None:
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().contiguous().cpu().numpy().tobytes())
    hashes = [None] * WORLD_SIZE
    dist.all_gather_object(hashes, digest.hexdigest())
    if len(set(hashes)) != 1:
        raise RuntimeError(f'draft replicas diverged: {hashes}')

def save(path: Path, state: dict, model: ScratchDFlash, optimizer, scheduler, rank: int) -> None:
    rng = [None] * WORLD_SIZE
    dist.all_gather_object(rng, _rng_snapshot())
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.pt.tmp')
        torch.save({**state, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng_by_rank': rng}, temporary)
        os.replace(temporary, path)
    dist.barrier()

def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE or int(os.environ.get('WORLD_SIZE', '0')) != WORLD_SIZE:
        raise RuntimeError('requires torchrun on exactly two GPUs')
    if args.anchor_limit != 64 or not 1 <= args.anchor_chunk <= 16 or args.save_interval != SAVE_INTERVAL or (args.validate_interval != 1000) or (args.draft_learning_rate != 0.0006) or (args.dense_learning_rate != 0.0006):
        raise ValueError('training settings differ from the pinned run')
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')
    torch.manual_seed(12012)
    random.seed(12012 + rank)
    torch.set_float32_matmul_precision('high')
    manifest = verify_manifest(args.data_dir)
    pins = json.loads(args.pins_report.read_text())
    if pins.get('target_metadata_sha256') != TARGET_METADATA_SHA256 or pins.get('draft_metadata_sha256') != DRAFT_METADATA_SHA256 or pins.get('target_shards_verified') != 5:
        raise RuntimeError('published model integrity report failed')
    training = IndexedJsonl(args.data_dir / 'train.jsonl')
    validation = IndexedJsonl(args.data_dir / 'validation.jsonl')
    try:
        if len(training) != 792000 or len(validation) != 8000:
            raise RuntimeError('pinned prepared data row counts differ')
        target, model, random_fc_hash = load_models(args.target_path, args.draft_path, device)
        expected = metadata(manifest, random_fc_hash, args)
        state = {**expected, 'global_step': 0, 'epoch': 0, 'recent_losses': [], 'recent_update_seconds': [], 'last_validation': None, 'local_offset': 0, 'best_acceptance': -1.0, 'initial_probes': initial_probes(model), 'initial_fc_probe': model.draft.fc.weight.detach().flatten()[:4096].cpu().clone(), 'last_metrics': {}}
        draft_params = [p for name, p in model.named_parameters() if name.startswith(('draft.layers.', 'draft.norm.', 'draft.hidden_norm.'))]
        optimizer = torch.optim.AdamW([{'params': draft_params, 'lr': args.draft_learning_rate}, {'params': [model.draft.fc.weight], 'lr': args.dense_learning_rate}], weight_decay=0.01, betas=(0.9, 0.95), eps=1e-08)
        scheduler = scheduler_for_continuation(optimizer, args.schedule_mode)
        latest = args.output_dir / STAGE / 'latest.pt'
        if args.resume:
            previous = torch.load(args.resume, map_location='cpu', weights_only=False, mmap=True)
            for key, value in expected.items():
                if key in ('training_code_sha256', 'run_config', 'stage'):
                    continue
                if previous.get(key) != value:
                    raise RuntimeError(f'resume lineage differs: {key}')
            previous_step = previous.get('global_step', -1)
            previous_run = dict(previous.get('run_config', {}))
            expected_run = dict(expected['run_config'])
            previous_source = previous.get('training_code_sha256')
            legacy_run = dict(expected_run)
            legacy_run['authorized_stop'] = LEGACY_STOP
            is_legacy = previous_step == LEGACY_STOP
            if not (is_legacy and previous.get('stage') == 'scratch_native_dflash_paper_lr_2h100_nothink' and (previous_run == legacy_run) and (previous_source == LEGACY_SOURCE_SHA256) or (isinstance(previous_step, int) and LEGACY_STOP < previous_step < AUTHORIZED_STOP and (previous_step % SAVE_INTERVAL == 0) and (previous.get('stage') == STAGE) and (previous_run == expected_run) and (previous_source == code_sha256()))):
                raise RuntimeError('resume source or original optimizer configuration differs')
            if previous.get('local_offset') != previous.get('global_step', -1) * BATCH_SIZE or len(previous.get('rng_by_rank', [])) != WORLD_SIZE or previous.get('global_step', -1) > AUTHORIZED_STOP:
                raise RuntimeError('resume cursor, RNG or stop boundary corrupt')
            model.load_state_dict(previous['model'], strict=True)
            optimizer.load_state_dict(previous['optimizer'])
            scheduler.load_state_dict(previous['scheduler'])
            expected_rates = [float(base) * schedule_multiplier(previous_step) for base in scheduler.base_lrs]
            if len(expected_rates) != len(optimizer.param_groups) or scheduler.last_epoch != previous_step or any((abs(float(group['lr']) - rate) > 1e-10 for group, rate in zip(optimizer.param_groups, expected_rates))) or any((abs(float(saved) - rate) > 1e-10 for saved, rate in zip(scheduler._last_lr, expected_rates))):
                raise RuntimeError('original one-epoch LR was not preserved')
            _rng_restore(previous['rng_by_rank'][rank])
            for key in ('global_step', 'epoch', 'local_offset', 'best_acceptance', 'initial_probes', 'initial_fc_probe', 'last_metrics', 'recent_losses', 'recent_update_seconds', 'last_validation'):
                state[key] = previous[key]
            del previous
        elif latest.exists():
            raise RuntimeError('scratch DFlash checkpoint exists; resume explicitly')
        if args.stop_after_update is None or not (state['global_step'] < args.stop_after_update <= AUTHORIZED_STOP and (args.stop_after_update - state['global_step'] <= 1000 or args.stop_after_update == AUTHORIZED_STOP)):
            raise ValueError('trainer segment exceeds authorized stop')
        _assert_rank_parity(model)
        validation_indices = stratified_validation_indices(validation, manifest['validation_source_counts'], args.validation_limit)
        loss_indices = stratified_validation_indices(validation, manifest['validation_source_counts'], min(8, args.validation_limit))
        order = _ordered_indices(len(training), 0, 0, 1, manifest['seed'])
        model.train()
        if not args.resume and rank == 0:
            initial = heldout_context_acceptance(target, model, validation, device, indices=validation_indices)
            state['best_acceptance'] = initial['mean_accepted_length']
            state['last_validation'] = {'step': 0, 'acceptance': initial, 'weighted_loss': None, 'loss_anchor_limit': 8}
            print(json.dumps({'step': 0, 'validation': initial}), flush=True)
        dist.barrier()
        offset = state['local_offset']
        local_batch = BATCH_SIZE // WORLD_SIZE
        while state['global_step'] < args.stop_after_update:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            first = offset + rank * local_batch
            local_losses = [one_sequence_loss(target, model, training[index], device, anchor_limit=args.anchor_limit, anchor_chunk=args.anchor_chunk, accumulation=BATCH_SIZE) for index in order[first:first + local_batch]]
            if len(local_losses) != local_batch:
                raise RuntimeError('incomplete global batch')
            _reduce_gradients(model)
            gradients = gradient_report(model)
            if any((not math.isfinite(value) or value <= 0 for value in [*gradients['draft_layer_gradient_norms'], *gradients['context_kv_gradient_norms'], gradients['fc_gradient_norm']])):
                raise RuntimeError('non-finite or zero FC/draft gradient')
            global_loss = torch.tensor(sum(local_losses), device=device)
            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            global_loss = float(global_loss) / BATCH_SIZE
            gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError('non-finite full-draft gradient norm')
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            offset += BATCH_SIZE
            state['local_offset'] = offset
            state['global_step'] += 1
            step = state['global_step']
            state['recent_losses'] = (state['recent_losses'] + [global_loss])[-SAVE_INTERVAL:]
            state['recent_update_seconds'] = (state['recent_update_seconds'] + [elapsed])[-SAVE_INTERVAL:]
            state['last_metrics'] = {'step': step, 'prefix_loss': global_loss, 'recent_loss_mean': sum(state['recent_losses']) / len(state['recent_losses']), 'recent_update_seconds_mean': sum(state['recent_update_seconds']) / len(state['recent_update_seconds']), 'recent_window_updates': len(state['recent_losses']), 'gradient_norm': float(gradient_norm), **gradients, 'update_seconds': elapsed, 'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2 ** 30}
            if rank == 0 and (step % args.log_interval == 0 or step == args.stop_after_update):
                print(json.dumps({'stage': STAGE, **state['last_metrics']}), flush=True)
            if step % args.validate_interval == 0 or step == args.stop_after_update:
                dist.barrier()
                if rank == 0:
                    acceptance = heldout_context_acceptance(target, model, validation, device, indices=validation_indices)
                    heldout_loss = heldout_weighted_loss(target, model, validation, device, indices=loss_indices)
                    score = acceptance['mean_accepted_length']
                    state['last_validation'] = {'step': step, 'acceptance': acceptance, 'weighted_loss': heldout_loss, 'loss_anchor_limit': 8}
                    print(json.dumps({'step': step, 'validation': acceptance, 'validation_weighted_loss': heldout_loss}), flush=True)
                    state['best_acceptance'] = max(state['best_acceptance'], score)
                dist.barrier()
            if step % SAVE_INTERVAL == 0 or step == args.stop_after_update:
                _assert_rank_parity(model)
                save(latest, state, model, optimizer, scheduler, rank)
                if rank == 0:
                    if step % 1000 == 0 or step == AUTHORIZED_STOP:
                        milestone_dir = latest.parent / 'milestones'
                        milestone_dir.mkdir(parents=True, exist_ok=True)
                        milestone = milestone_dir / f'step{step}.pt'
                        if milestone.exists():
                            raise RuntimeError('immutable scratch milestone already exists')
                        temporary = milestone.with_suffix('.pt.tmp')
                        shutil.copyfile(latest, temporary)
                        os.replace(temporary, milestone)
                    print(json.dumps({'stage': STAGE, 'checkpoint_saved': step, 'checkpoint_bytes': latest.stat().st_size}), flush=True)
    finally:
        training.close()
        validation.close()
        if dist.is_initialized():
            dist.destroy_process_group()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--target-path', type=Path, required=True)
    parser.add_argument('--draft-path', type=Path, required=True)
    parser.add_argument('--pins-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-update', type=int, required=True)
    parser.add_argument('--anchor-limit', type=int, default=64)
    parser.add_argument('--anchor-chunk', type=int, default=8)
    parser.add_argument('--validation-limit', type=int, default=16)
    parser.add_argument('--validate-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=SAVE_INTERVAL)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--draft-learning-rate', type=float, default=0.0006)
    parser.add_argument('--dense-learning-rate', type=float, default=0.0006)
    parser.add_argument('--schedule-mode', choices=('original',), required=True)
    train(parser.parse_args())
if __name__ == '__main__':
    main()
