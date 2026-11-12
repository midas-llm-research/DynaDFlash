from __future__ import annotations
import hashlib
import json
from pathlib import Path
TARGET_METADATA_SHA256 = '1743244349047a37ff7f8793bf97a91c8488e4d191dcfa6dd84c4ad10f8c9575'
DRAFT_METADATA_SHA256 = 'f7aad255d176a97bcabe375a7a4f46477848537fc8efaba72978deb9f28daf3d'
TARGET_CONFIG_SHA256 = 'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30'
DRAFT_CONFIG_SHA256 = '9834d608c9ca53d5548b415471ae9e8ebc9aab6cedfc2a7af95b6bd097373102'

def file_sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            state.update(chunk)
    return state.hexdigest()

def verify(base_root: Path, pins_root: Path) -> dict:
    target_meta_path = pins_root / 'target_download.json'
    draft_meta_path = pins_root / 'draft_download.json'
    if file_sha256(target_meta_path) != TARGET_METADATA_SHA256:
        raise RuntimeError('packaged target pin manifest changed')
    if file_sha256(draft_meta_path) != DRAFT_METADATA_SHA256:
        raise RuntimeError('packaged drafter pin manifest changed')
    target_meta = json.loads(target_meta_path.read_text())
    draft_meta = json.loads(draft_meta_path.read_text())
    if (target_meta['repository'], target_meta['revision']) != ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'):
        raise RuntimeError('target repository or revision changed')
    if (draft_meta['repository'], draft_meta['revision']) != ('z-lab/Qwen3-8B-DFlash-b16', '9b41424b7109f9c5413454f481b09a82b85333f4'):
        raise RuntimeError('drafter repository or revision changed')
    target = base_root / 'target_8b'
    draft = base_root / 'draft_8b'
    if file_sha256(target / 'config.json') != TARGET_CONFIG_SHA256:
        raise RuntimeError('target config changed')
    if file_sha256(draft / 'config.json') != DRAFT_CONFIG_SHA256:
        raise RuntimeError('drafter config changed')
    for filename, record in target_meta['weights'].items():
        path = target / filename
        if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
            raise RuntimeError(f'target weight shard changed: {filename}')
    for filename, record in draft_meta['files'].items():
        path = draft / filename
        if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
            raise RuntimeError(f'drafter file changed: {filename}')
    for filename, key in (('tokenizer.json', 'tokenizer_sha256'), ('tokenizer_config.json', 'tokenizer_config_sha256')):
        if file_sha256(target / filename) != target_meta[key]:
            raise RuntimeError(f'target {filename} changed')
    report = {'target_metadata_sha256': TARGET_METADATA_SHA256, 'draft_metadata_sha256': DRAFT_METADATA_SHA256, 'target_shards_verified': len(target_meta['weights']), 'draft_files_verified': len(draft_meta['files']), 'target_config_sha256': TARGET_CONFIG_SHA256, 'draft_config_sha256': DRAFT_CONFIG_SHA256}
    return report
