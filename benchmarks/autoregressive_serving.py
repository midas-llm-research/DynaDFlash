from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = Path('/workspace/frozen_prompts.json')
PROMPTS_SHA256 = 'da3674d908e456344924233bb78c907659b9491a86c34b74318627eeaae17d9d'
TARGET = '/base_models/models/target_8b'
BASE = modal.Volume.from_name('dflash-query-routing-20261007', environment_name='main', create_if_missing=False)
IMAGE = modal.Image.from_registry('lmsysorg/sglang:v0.5.18').add_local_file(
    ROOT / 'benchmarks/frozen_metamath_prompts.json', remote_path=str(PROMPTS)
)
app = modal.App('dynadflash-autoregressive-serving-20261112')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def request(prompt: dict, length: int) -> dict:
    payload = {'text': prompt['prompt'], 'stream': True,
               'sampling_params': {'temperature': 0.0, 'max_new_tokens': length,
                                   'ignore_eos': True}}
    req = urllib.request.Request('http://127.0.0.1:30000/generate',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    start = time.perf_counter()
    first = None
    response_text = ''
    meta = {}
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data:'):
                continue
            body = line[5:].strip()
            if body == '[DONE]':
                break
            event = json.loads(body)
            fragment = event.get('text', '')
            if fragment:
                first = first or time.perf_counter()
                response_text = fragment if fragment.startswith(response_text) else response_text + fragment
            meta = event.get('meta_info', meta)
    elapsed = time.perf_counter() - start
    if meta.get('completion_tokens') != length or meta.get('prompt_tokens') != prompt['prompt_tokens']:
        raise RuntimeError('server output count or prompt token parity mismatch')
    return {'index': prompt['index'], 'prompt_hash': prompt['prompt_hash'],
            'prompt_tokens': prompt['prompt_tokens'], 'completion_tokens': length,
            'text': response_text, 'meta_info': meta,
            'latency_seconds': elapsed,
            'ttft_seconds': (first - start) if first else None}


def batch(rows: list[dict], concurrency: int, length: int) -> dict:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        observations = list(pool.map(lambda row: request(row, length), rows))
    elapsed = time.perf_counter() - start
    return {'concurrency': concurrency, 'elapsed_seconds': elapsed,
            'output_tokens': sum(row['completion_tokens'] for row in observations),
            'observations': observations}


def await_server(process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('server exited: ' + '\n'.join(log_path.read_text(errors='replace').splitlines()[-40:]))
        try:
            with urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionResetError):
            time.sleep(2)
    raise TimeoutError('server did not reach health endpoint')


@app.function(image=IMAGE, volumes={'/base_models': BASE}, gpu='A100', cpu=8,
              memory=65536, timeout=7200, max_containers=1)
def benchmark(phase: str) -> dict:
    if phase not in ('smoke', 'full'):
        raise ValueError('phase must be smoke or full')
    raw = PROMPTS.read_bytes()
    if sha(raw) != PROMPTS_SHA256:
        raise RuntimeError('frozen MetaMathQA prompts changed')
    selected = json.loads(raw)
    if len(selected) != 24 or len({row['prompt_hash'] for row in selected}) != 24:
        raise RuntimeError('frozen prompt identities changed')
    length = 16 if phase == 'smoke' else 64
    cycles = 1 if phase == 'smoke' else 3
    concurrencies = (1,) if phase == 'smoke' else (1, 8)
    command = [sys.executable, '-m', 'sglang.launch_server',
               '--model-path', TARGET,
               '--host', '127.0.0.1', '--port', '30000', '--tp-size', '1',
               '--dtype', 'bfloat16', '--mem-fraction-static', '0.75',
               '--chunked-prefill-size', '1024', '--disable-radix-cache',
               '--random-seed', '24750', '--cuda-graph-backend-prefill', 'disabled',
               '--trust-remote-code']
    spec = {'phase': phase, 'arm': 'autoregressive', 'target_model_path': TARGET,
            'prompts_sha256': PROMPTS_SHA256, 'prompt_count': 24,
            'sglang': '0.5.18', 'gpu': 'A100', 'dtype': 'bfloat16',
            'output_tokens': length, 'cycles': cycles,
            'concurrencies': list(concurrencies), 'server_command': command,
            'benchmark_code_sha256': sha(Path(__file__).read_bytes())}
    results = {'spec': spec, 'cycles': []}
    for cycle in range(cycles):
        log = Path(f'/tmp/autoregressive_{phase}_{cycle}.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                await_server(process, log)
                request(selected[0], 16)
                measured = []
                for concurrency in concurrencies:
                    batch(selected, concurrency, length)
                    measured.append(batch(selected, concurrency, length))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
        log_bytes = log.read_bytes()
        results['cycles'].append({'cycle': cycle, 'arm': 'autoregressive',
                                  'measured': measured, 'server_log_sha256': sha(log_bytes),
                                  'server_log_tail': log_bytes.decode(errors='replace').splitlines()[-30:]})
    summary = {}
    for concurrency in concurrencies:
        trials = [next(row for row in cycle['measured'] if row['concurrency'] == concurrency)
                  for cycle in results['cycles']]
        observations = [row for trial in trials for row in trial['observations']]
        output = sum(trial['output_tokens'] for trial in trials)
        elapsed = sum(trial['elapsed_seconds'] for trial in trials)
        summary[str(concurrency)] = {
            'output_tokens': output, 'elapsed_seconds': elapsed,
            'output_tokens_per_second': output / elapsed,
            'cycle_rates': [trial['output_tokens'] / trial['elapsed_seconds'] for trial in trials],
            'latency_median_seconds': statistics.median(row['latency_seconds'] for row in observations),
            'request_count': len(observations)}
    results['summary'] = summary
    return results


@app.local_entrypoint()
def main(phase: str = 'smoke') -> None:
    result = benchmark.remote(phase)
    output = ROOT / 'benchmarks/results' / f'autoregressive_{phase}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'artifact': str(output), 'spec': result['spec'],
                      'summary': result['summary']}, indent=2, sort_keys=True))
