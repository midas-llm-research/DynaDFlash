from __future__ import annotations
import argparse
import hashlib
import json
import re
import sqlite3
import struct
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
DATASET_ID = 'inference-optimization/Qwen3-8B-Regenerated-Collection'
DATASET_REVISION = '65d219d6b40bb27c45afe16665147a1d3fa21069'
TARGET_ID = 'Qwen/Qwen3-8B'
TARGET_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
MAX_TOKENS = 3072
POOL_SIZE = 800000
VALIDATION_SIZE = 8000
SEED = 'dflash-dynamic-router-20261012-v1'
QUOTAS = {'magpie_output.jsonl': 300000, 'ultrachat_output.jsonl': 207864, 'metamathqa_train_Qwen3-8B.jsonl': 192136, 'evol_codealpaca_train_Qwen3-8B.jsonl': 100000}
SHARDS = ('autoif_train_Qwen3-8B.jsonl', 'evol_codealpaca_train_Qwen3-8B.jsonl', 'lmsys_arena_train_Qwen3-8B.jsonl', 'magpie_output.jsonl', 'metamathqa_train_Qwen3-8B.jsonl', 'nemotron_chat_Qwen3-8B.jsonl', 'nemotron_math_Qwen3-8B.jsonl', 'nemotron_stem_Qwen3-8B.jsonl', 'orca_math_train_Qwen3-8B.jsonl', 'tulu3_Qwen3-8B.jsonl', 'ultrachat_output.jsonl', 'ultrafeedback_train_sft_Qwen3-8B.jsonl', 'ultrainteract_train_Qwen3-8B.jsonl')

def canonical_text(value: str) -> str:
    value = unicodedata.normalize('NFKC', value).replace('\r\n', '\n').replace('\r', '\n')
    return '\n'.join((line.rstrip() for line in value.split('\n'))).strip()

def digest(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()

def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def sample_key(shard: str, prompt_hash: str) -> str:
    return digest(f'{SEED}\x00{shard}\x00{prompt_hash}')

def split_key(prompt_hash: str) -> str:
    return digest(f'{SEED}\x00validation\x00{prompt_hash}')

def normalize_record(row: dict[str, Any]) -> tuple[list[dict[str, str]], str, str]:
    metadata = row.get('metadata') or {}
    if not isinstance(metadata, dict):
        raise ValueError('invalid_metadata')
    if metadata.get('error') not in (None, '', False):
        raise ValueError('generation_error')
    if metadata.get('finish_reason') != 'stop':
        raise ValueError('not_stopped')
    turns = row.get('conversations')
    if not isinstance(turns, list) or len(turns) < 2 or len(turns) % 2:
        raise ValueError('invalid_turn_count')
    messages: list[dict[str, str]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError('invalid_turn')
        expected = 'human' if index % 2 == 0 else 'gpt'
        if turn.get('from') != expected or not isinstance(turn.get('value'), str):
            raise ValueError('invalid_turn_order')
        text = turn['value']
        if not text.strip():
            raise ValueError('empty_turn')
        if index < len(turns) - 1:
            messages.append({'role': 'user' if expected == 'human' else 'assistant', 'content': text})
    response = turns[-1]['value']
    if not response.startswith('<think>'):
        raise ValueError('thinking_trace_not_at_response_start')
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        raise ValueError('missing_or_multiple_thinking_tags')
    if response.index('<think>') > response.index('</think>'):
        raise ValueError('reversed_thinking_tags')
    if not response.split('<think>', 1)[1].split('</think>', 1)[0].strip():
        raise ValueError('empty_thinking_trace')
    if not response.split('</think>', 1)[1].strip():
        raise ValueError('empty_final_answer')
    if '<|im_end|>' in response:
        raise ValueError('embedded_end_token')
    last_user = next((message['content'] for message in reversed(messages) if message['role'] == 'user'))
    return (messages, response, digest(canonical_text(last_user)))

def prompt_hash(messages: list[dict[str, str]]) -> str:
    normalized = [{'role': m['role'], 'content': canonical_text(m['content'])} for m in messages]
    return digest(json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(',', ':')))

def load_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    hashes: set[str] = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row if isinstance(row, str) else row.get('prompt')
            if not isinstance(text, str):
                raise ValueError('benchmark exclusion lines must contain a prompt string')
            hashes.add(digest(canonical_text(text)))
    return hashes

def _create_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS candidates (prompt_hash TEXT PRIMARY KEY, last_user_hash TEXT NOT NULL, source TEXT NOT NULL, sample_key TEXT NOT NULL, record_hash TEXT NOT NULL, record_id TEXT NOT NULL, messages_json TEXT NOT NULL, response TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS source_order ON candidates(source, sample_key)')
    db.execute('CREATE INDEX IF NOT EXISTS global_order ON candidates(sample_key)')
    return db

def index_shards(raw_dir: Path, work_dir: Path, *, exclusions: set[str] | None=None, shards: Iterable[str]=SHARDS, max_lines_per_shard: int | None=None) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    db = _create_db(work_dir / 'candidates.sqlite')
    exclusions = exclusions or set()
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    raw_hashes: dict[str, str] = {}
    priority = {name: i for i, name in enumerate(QUOTAS)}
    for shard in shards:
        path = raw_dir / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        hasher = hashlib.sha256()
        with path.open('rb') as handle:
            for line_number, raw_line in enumerate(handle, 1):
                hasher.update(raw_line)
                if max_lines_per_shard is not None and line_number > max_lines_per_shard:
                    break
                stats[shard]['lines'] += 1
                try:
                    row = json.loads(raw_line)
                    if not isinstance(row, dict):
                        raise ValueError('non_object')
                    messages, response, last_user_hash = normalize_record(row)
                    if last_user_hash in exclusions:
                        raise ValueError('benchmark_overlap')
                    p_hash = prompt_hash(messages)
                    r_hash = digest(json.dumps([messages, response], ensure_ascii=False))
                    existing = db.execute('SELECT source,record_hash FROM candidates WHERE prompt_hash=?', (p_hash,)).fetchone()
                    new_rank = (priority.get(shard, len(priority)), shard, r_hash)
                    if existing is not None:
                        old_rank = (priority.get(existing[0], len(priority)), existing[0], existing[1])
                        if new_rank >= old_rank:
                            stats[shard]['duplicate_dropped'] += 1
                            continue
                        stats[existing[0]]['duplicate_replaced'] += 1
                    db.execute('INSERT OR REPLACE INTO candidates VALUES(?,?,?,?,?,?,?,?)', (p_hash, last_user_hash, shard, sample_key(shard, p_hash), r_hash, str(row.get('id', line_number)), json.dumps(messages, ensure_ascii=False, separators=(',', ':')), response))
                    stats[shard]['indexed'] += 1
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    stats[shard][f'excluded_{error}'] += 1
                if line_number % 10000 == 0:
                    db.commit()
        db.commit()
        raw_hashes[shard] = hasher.hexdigest() if max_lines_per_shard is None else 'INCOMPLETE_SMOKE_HASH'
    count = db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
    db.close()
    report = {'dataset_id': DATASET_ID, 'revision': DATASET_REVISION, 'raw_sha256': raw_hashes, 'candidate_count': count, 'source_stats': {source: dict(values) for source, values in stats.items()}, 'smoke_only': max_lines_per_shard is not None}
    (work_dir / 'index_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report

def _strip_template_overlap(prefix: str, response: str) -> str:
    for opener in ('<think>\n', '<think>'):
        if prefix.endswith(opener) and response.startswith(opener):
            return response[len(opener):]
    return response

def tokenize_record(tokenizer: Any, messages: list[dict[str, str]], response: str, *, max_tokens: int=MAX_TOKENS) -> tuple[list[int], int, int]:
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    completion = _strip_template_overlap(prefix, response)
    if (prefix + completion).split('<|im_start|>assistant')[-1].count('<think>') != 1:
        raise ValueError('duplicated_or_missing_think_opener')
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    joint_ids = tokenizer(prefix + completion, add_special_tokens=False).input_ids
    if joint_ids != prefix_ids + completion_ids:
        raise ValueError('token_boundary_mismatch')
    end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if not isinstance(end_id, int) or end_id < 0 or end_id == tokenizer.unk_token_id:
        raise ValueError('missing_end_token')
    if len(prefix_ids) >= max_tokens:
        raise ValueError('prompt_over_length')
    full_ids = prefix_ids + completion_ids + [end_id]
    ids = full_ids[:max_tokens]
    if len(ids) - len(prefix_ids) < 2:
        raise ValueError('too_few_supervised_tokens')
    think_end_id = tokenizer.convert_tokens_to_ids('</think>')
    boundary = next((i + 1 for i in range(len(prefix_ids), len(ids)) if ids[i] == think_end_id), -1)
    if boundary < 0 and len(full_ids) <= max_tokens:
        raise ValueError('missing_thinking_boundary_after_truncation')
    return (ids, len(prefix_ids), boundary)

def _write_indexed_line(handle: Any, index_handle: Any, record: dict[str, Any]) -> None:
    index_handle.write(struct.pack('<Q', handle.tell()))
    handle.write((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode())

def prepare_pool(work_dir: Path, tokenizer: Any, *, pool_size: int=POOL_SIZE, validation_size: int=VALIDATION_SIZE) -> dict[str, Any]:
    if validation_size <= 0 or validation_size >= pool_size:
        raise ValueError('invalid validation size')
    db = sqlite3.connect(work_dir / 'candidates.sqlite')
    selected: set[str] = set()
    records: list[tuple[str, str]] = []
    exclusions = Counter()
    source_counts = Counter()
    selected_path = work_dir / 'selected_unsplit.jsonl'
    with selected_path.open('wb') as selected_out:

        def consume(rows: Iterable[tuple]) -> None:
            for p_hash, source, record_id, messages_json, response in rows:
                if len(selected) >= pool_size or p_hash in selected:
                    continue
                try:
                    ids, prompt_length, think_boundary = tokenize_record(tokenizer, json.loads(messages_json), response)
                except ValueError as error:
                    exclusions[str(error)] += 1
                    continue
                selected.add(p_hash)
                source_counts[source] += 1
                records.append((source, split_key(p_hash)))
                payload = {'source': source, 'record_id': record_id, 'prompt_hash': p_hash, 'input_ids': ids, 'prompt_length': prompt_length, 'think_boundary': think_boundary}
                selected_out.write((json.dumps(payload, separators=(',', ':')) + '\n').encode())
        scale = pool_size / POOL_SIZE
        scaled_quotas = {source: int(quota * scale) for source, quota in QUOTAS.items()}
        for source in QUOTAS:
            if sum(scaled_quotas.values()) >= pool_size:
                break
            scaled_quotas[source] += 1
        for source, quota in scaled_quotas.items():
            if len(selected) >= pool_size:
                break
            rows = db.execute('SELECT prompt_hash,source,record_id,messages_json,response FROM candidates WHERE source=? ORDER BY sample_key', (source,))

            def quota_rows() -> Iterable[tuple]:
                for row in rows:
                    if source_counts[source] >= quota:
                        break
                    yield row
            consume(quota_rows())
        if len(selected) < pool_size:
            consume(db.execute('SELECT prompt_hash,source,record_id,messages_json,response FROM candidates ORDER BY sample_key'))
    db.close()
    if len(selected) != pool_size:
        raise RuntimeError(f'only {len(selected)} tokenized records; need {pool_size}')
    per_source_val = {s: n * validation_size // pool_size for s, n in source_counts.items()}
    remainder = validation_size - sum(per_source_val.values())
    order = sorted(source_counts, key=lambda s: (-(source_counts[s] * validation_size % pool_size), s))
    for source in order[:remainder]:
        per_source_val[source] += 1
    by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, (source, key) in enumerate(records):
        by_source[source].append((key, index))
    validation_indices = {index for source, entries in by_source.items() for _, index in sorted(entries)[:per_source_val[source]]}
    if len(validation_indices) != validation_size:
        raise AssertionError('validation split size mismatch')
    handles = {split: ((work_dir / f'{split}.jsonl').open('wb'), (work_dir / f'{split}.idx').open('wb')) for split in ('train', 'validation')}
    try:
        with selected_path.open('rb') as source_file:
            for index, line in enumerate(source_file):
                split = 'validation' if index in validation_indices else 'train'
                output, index_output = handles[split]
                index_output.write(struct.pack('<Q', output.tell()))
                output.write(line)
    finally:
        for output, index_output in handles.values():
            output.close()
            index_output.close()
    manifest = {'dataset_id': DATASET_ID, 'dataset_revision': DATASET_REVISION, 'target_id': TARGET_ID, 'target_revision': TARGET_REVISION, 'pool_size': pool_size, 'train_size': pool_size - validation_size, 'validation_size': validation_size, 'source_counts': dict(source_counts), 'validation_source_counts': per_source_val, 'tokenization_exclusions': dict(exclusions), 'max_tokens': MAX_TOKENS, 'seed': SEED, 'index_report_sha256': digest((work_dir / 'index_report.json').read_bytes()), 'selected_sha256': file_digest(selected_path), 'train_sha256': file_digest(work_dir / 'train.jsonl'), 'validation_sha256': file_digest(work_dir / 'validation.jsonl')}
    manifest['manifest_sha256'] = digest(json.dumps(manifest, sort_keys=True))
    (work_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest

def download_raw(raw_dir: Path) -> dict[str, Any]:
    import os
    import urllib.request
    from huggingface_hub import constants, get_hf_file_metadata, hf_hub_download, hf_hub_url
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    constants.HF_HUB_DISABLE_XET = True
    raw_dir.mkdir(parents=True, exist_ok=True)
    verified = {}
    skipped = {}
    for shard in SHARDS:
        url = hf_hub_url(DATASET_ID, shard, repo_type='dataset', revision=DATASET_REVISION)
        metadata = get_hf_file_metadata(url)
        expected = metadata.etag.strip('"')
        if metadata.commit_hash != DATASET_REVISION or not re.fullmatch('[0-9a-f]{64}', expected):
            raise RuntimeError(f'pinned LFS metadata unavailable or changed: {shard}')
        with urllib.request.urlopen(urllib.request.Request(metadata.location, method='HEAD'), timeout=60) as response:
            served_size = int(response.headers.get('Content-Length', '-1'))
        if served_size != metadata.size:
            if shard in QUOTAS:
                raise RuntimeError(f'main quota shard is corrupt at Hub CDN: {shard}')
            skipped[shard] = {'reason': 'pinned_lfs_vs_cdn_size_mismatch', 'expected_bytes': metadata.size, 'served_bytes': served_size, 'expected_sha256': expected}
            continue
        destination = raw_dir / shard
        if not (destination.is_file() and destination.stat().st_size == metadata.size and (file_digest(destination) == expected)):
            path = hf_hub_download(repo_id=DATASET_ID, repo_type='dataset', revision=DATASET_REVISION, filename=shard, local_dir=raw_dir, force_download=destination.is_file())
            if not Path(path).samefile(destination):
                raise RuntimeError(f'download escaped isolated raw directory: {path}')
        actual_size = destination.stat().st_size
        actual_sha = file_digest(destination)
        if actual_size != metadata.size or actual_sha != expected:
            raise RuntimeError(f'pinned raw shard checksum mismatch: {shard}, size {actual_size}/{metadata.size}, sha {actual_sha}/{expected}')
        verified[shard] = {'sha256': actual_sha, 'bytes': actual_size, 'commit': metadata.commit_hash}
    report = {'dataset_id': DATASET_ID, 'revision': DATASET_REVISION, 'disable_xet': True, 'shards': verified, 'skipped_shards': skipped}
    (raw_dir / 'download_verification.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('download', 'index', 'prepare'))
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--benchmark-prompts', type=Path)
    parser.add_argument('--max-lines-per-shard', type=int)
    parser.add_argument('--pool-size', type=int, default=POOL_SIZE)
    parser.add_argument('--validation-size', type=int, default=VALIDATION_SIZE)
    args = parser.parse_args()
    if args.command == 'download':
        download_raw(args.raw_dir)
    elif args.command == 'index':
        index_shards(args.raw_dir, args.work_dir, exclusions=load_exclusions(args.benchmark_prompts), max_lines_per_shard=args.max_lines_per_shard)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TARGET_ID, revision=TARGET_REVISION)
        prepare_pool(args.work_dir, tokenizer, pool_size=args.pool_size, validation_size=args.validation_size)
if __name__ == '__main__':
    main()
