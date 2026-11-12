from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import struct
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from dynamic_dflash.core.data import MAX_TOKENS, digest, file_digest
NONTHINK_TEMPLATE_SUFFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'

def final_answer(text: str) -> str:
    if not text.startswith('<think>') or text.count('<think>') != 1 or text.count('</think>') != 1 or ('<|im_end|>' in text):
        raise ValueError('malformed_recorded_thinking_response')
    answer = text.split('</think>', 1)[1].lstrip('\r\n')
    if not answer.strip() or '<think>' in answer or '</think>' in answer:
        raise ValueError('empty_or_nested_final_answer')
    return answer

def nonthinking_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for message in messages:
        role, content = (message['role'], message['content'])
        if role == 'assistant':
            if content.startswith('<think>'):
                content = final_answer(content)
            elif '<think>' in content or '</think>' in content:
                raise ValueError('malformed_prior_assistant_trace')
        result.append({'role': role, 'content': content})
    return result

def tokenize_nonthinking(tokenizer: Any, messages: list[dict[str, str]], recorded_response: str, *, max_tokens: int=MAX_TOKENS) -> tuple[list[int], int, int]:
    prompt = tokenizer.apply_chat_template(nonthinking_messages(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not prompt.endswith(NONTHINK_TEMPLATE_SUFFIX):
        raise ValueError('pinned_nonthinking_template_changed')
    answer = final_answer(recorded_response)
    prefix_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if not isinstance(end_id, int) or end_id < 0 or end_id == tokenizer.unk_token_id:
        raise ValueError('missing_end_token')
    prompt_trim = max(0, len(prefix_ids) - (max_tokens - 2))
    if prompt_trim > 16:
        raise ValueError('prompt_over_length_by_more_than_16_tokens')
    if prompt_trim:
        prefix_ids = prefix_ids[prompt_trim:]
    if not answer_ids:
        raise ValueError('empty_tokenized_final_answer')
    ids = (prefix_ids + answer_ids + [end_id])[:max_tokens]
    if len(ids) - len(prefix_ids) < 2:
        raise ValueError('too_few_supervised_tokens')
    decoded = tokenizer.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if unicodedata.normalize('NFC', decoded) != unicodedata.normalize('NFC', answer):
        raise ValueError('answer_token_roundtrip_mismatch_beyond_nfc')
    return (ids, len(prefix_ids), prompt_trim)

def _write_progress(path: Path, progress: dict) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)

def _complete_manifest(output_dir: Path, source_manifest: dict) -> dict:
    path = output_dir / 'manifest.json'
    manifest = json.loads(path.read_text())
    old_hash = manifest.pop('manifest_sha256')
    if digest(json.dumps(manifest, sort_keys=True)) != old_hash or manifest['source_manifest_sha256'] != source_manifest['manifest_sha256']:
        raise RuntimeError('completed non-thinking manifest is corrupt or mismatched')
    for name in ('train', 'validation'):
        if file_digest(output_dir / f'{name}.jsonl') != manifest[f'{name}_sha256']:
            raise RuntimeError(f'completed non-thinking {name} split hash differs')
    for key in ('seed', 'target_id', 'target_revision'):
        if key in manifest and manifest[key] != source_manifest[key]:
            raise RuntimeError(f'completed non-thinking {key} differs')
        manifest[key] = source_manifest[key]
    if 'prompt_truncation' not in manifest:
        summary = {'records': 0, 'max_left_trim': 0}
        for name in ('train', 'validation'):
            with (output_dir / f'{name}.jsonl').open() as handle:
                for line in handle:
                    trim = json.loads(line).get('prompt_left_truncated_tokens', 0)
                    summary['records'] += bool(trim)
                    summary['max_left_trim'] = max(summary['max_left_trim'], trim)
        manifest['prompt_truncation'] = summary
    manifest['manifest_sha256'] = digest(json.dumps(manifest, sort_keys=True))
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)
    return manifest

def prepare_nonthinking(source_dir: Path, output_dir: Path, tokenizer: Any, *, limit_per_split: int | None=None, checkpoint_every: int=10000) -> dict:
    if checkpoint_every < 1 or (limit_per_split is not None and limit_per_split < 1):
        raise ValueError('invalid conversion geometry')
    source_manifest = json.loads((source_dir / 'manifest.json').read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / 'conversion_progress.json'
    signature = {'source_manifest_sha256': source_manifest['manifest_sha256'], 'mode': 'non_thinking', 'max_tokens': MAX_TOKENS, 'limit_per_split': limit_per_split}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if {key: progress[key] for key in signature} != signature:
            raise RuntimeError('non-thinking conversion cursor differs from frozen source')
    else:
        if any(((output_dir / name).exists() for name in ('train.jsonl', 'train.idx', 'validation.jsonl', 'validation.idx'))):
            raise RuntimeError('untracked non-thinking split exists')
        progress = {**signature, 'splits': {name: {'rows': 0, 'jsonl_bytes': 0, 'idx_bytes': 0} for name in ('train', 'validation')}, 'counts': {}, 'complete': False}
        _write_progress(progress_path, progress)
    if progress['complete']:
        return _complete_manifest(output_dir, source_manifest)
    db = sqlite3.connect(f"file:{source_dir / 'candidates.sqlite'}?mode=ro", uri=True)
    try:
        for split in ('train', 'validation'):
            expected = source_manifest[f'{split}_size']
            if limit_per_split is not None:
                expected = min(expected, limit_per_split)
            state = progress['splits'][split]
            output_path = output_dir / f'{split}.jsonl'
            index_path = output_dir / f'{split}.idx'
            with (source_dir / f'{split}.jsonl').open('rb') as source, output_path.open('r+b' if output_path.exists() else 'w+b') as output, index_path.open('r+b' if index_path.exists() else 'w+b') as index:
                output.truncate(state['jsonl_bytes'])
                index.truncate(state['idx_bytes'])
                output.seek(state['jsonl_bytes'])
                index.seek(state['idx_bytes'])
                for row_number, raw in enumerate(source):
                    if row_number < state['rows']:
                        continue
                    if row_number >= expected:
                        break
                    old = json.loads(raw)
                    candidate = db.execute('SELECT source,record_id,messages_json,response FROM candidates WHERE prompt_hash=?', (old['prompt_hash'],)).fetchone()
                    if candidate is None or candidate[0] != old['source'] or candidate[1] != old['record_id']:
                        raise RuntimeError(f'frozen candidate mismatch at {split} row {row_number}')
                    try:
                        ids, prompt_length, prompt_trim = tokenize_nonthinking(tokenizer, json.loads(candidate[2]), candidate[3])
                    except ValueError as error:
                        raise RuntimeError(f'non-thinking tokenization failed at {split} row {row_number}: {error}') from error
                    new = {'source': old['source'], 'record_id': old['record_id'], 'prompt_hash': old['prompt_hash'], 'input_ids': ids, 'prompt_length': prompt_length, 'think_boundary': prompt_length, 'mode': 'non_thinking'}
                    if prompt_trim:
                        new['prompt_left_truncated_tokens'] = prompt_trim
                    index.write(struct.pack('<Q', output.tell()))
                    output.write((json.dumps(new, separators=(',', ':')) + '\n').encode())
                    state['rows'] = row_number + 1
                    if state['rows'] % checkpoint_every == 0:
                        output.flush()
                        index.flush()
                        os.fsync(output.fileno())
                        os.fsync(index.fileno())
                        state['jsonl_bytes'] = output.tell()
                        state['idx_bytes'] = index.tell()
                        _write_progress(progress_path, progress)
                if state['rows'] != expected:
                    raise RuntimeError(f"{split} conversion ended at {state['rows']}/{expected}")
                output.flush()
                index.flush()
                os.fsync(output.fileno())
                os.fsync(index.fileno())
                state['jsonl_bytes'] = output.tell()
                state['idx_bytes'] = index.tell()
                _write_progress(progress_path, progress)
    finally:
        db.close()
    counts = {}
    prompt_truncation = {'records': 0, 'max_left_trim': 0}
    for split in ('train', 'validation'):
        counter = Counter()
        with (output_dir / f'{split}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                counter[row['source']] += 1
                trim = row.get('prompt_left_truncated_tokens', 0)
                prompt_truncation['records'] += bool(trim)
                prompt_truncation['max_left_trim'] = max(prompt_truncation['max_left_trim'], trim)
        counts[split] = dict(counter)
    train_size = progress['splits']['train']['rows']
    validation_size = progress['splits']['validation']['rows']
    if limit_per_split is None and (train_size, validation_size) != (source_manifest['train_size'], source_manifest['validation_size']):
        raise RuntimeError('non-thinking pool changed the original split size')
    manifest = {'mode': 'non_thinking', 'dataset_id': source_manifest['dataset_id'], 'dataset_revision': source_manifest['dataset_revision'], 'seed': source_manifest['seed'], 'target_id': source_manifest['target_id'], 'target_revision': source_manifest['target_revision'], 'source_manifest_sha256': source_manifest['manifest_sha256'], 'pool_size': train_size + validation_size, 'train_size': train_size, 'validation_size': validation_size, 'source_counts': dict(Counter(counts['train']) + Counter(counts['validation'])), 'validation_source_counts': counts['validation'], 'max_tokens': MAX_TOKENS, 'template_mode': 'enable_thinking=False', 'prompt_truncation': prompt_truncation, 'answer_extraction': 'recorded text after closing </think>; leading CR/LF removed', 'train_sha256': file_digest(output_dir / 'train.jsonl'), 'validation_sha256': file_digest(output_dir / 'validation.jsonl')}
    manifest['manifest_sha256'] = digest(json.dumps(manifest, sort_keys=True))
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    progress['complete'] = True
    _write_progress(progress_path, progress)
    return manifest
