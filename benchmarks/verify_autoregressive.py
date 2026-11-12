from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / 'frozen_metamath_prompts.json'
RESULT = ROOT / 'results/autoregressive_full.json'
EXPECTED_PROMPTS_SHA = 'da3674d908e456344924233bb78c907659b9491a86c34b74318627eeaae17d9d'


def main() -> None:
    if hashlib.sha256(PROMPTS.read_bytes()).hexdigest() != EXPECTED_PROMPTS_SHA:
        raise RuntimeError('prompt fixture hash differs')
    prompts = json.loads(PROMPTS.read_text())
    result = json.loads(RESULT.read_text())
    spec = result['spec']
    if (spec['arm'] != 'autoregressive' or spec['phase'] != 'full' or
            spec['prompts_sha256'] != EXPECTED_PROMPTS_SHA or spec['prompt_count'] != 24 or
            spec['gpu'] != 'A100' or spec['sglang'] != '0.5.18' or
            spec['dtype'] != 'bfloat16' or spec['output_tokens'] != 64 or
            spec['cycles'] != 3 or spec['concurrencies'] != [1, 8] or
            '--speculative-algorithm' in spec['server_command']):
        raise RuntimeError('benchmark protocol differs')
    if len(prompts) != 24 or len({row['prompt_hash'] for row in prompts}) != 24:
        raise RuntimeError('prompt set differs')
    if len(result['cycles']) != 3:
        raise RuntimeError('cycle count differs')
    report = {}
    for concurrency in (1, 8):
        trials = []
        for cycle_number, cycle in enumerate(result['cycles']):
            if cycle['cycle'] != cycle_number or cycle['arm'] != 'autoregressive':
                raise RuntimeError('cycle identity differs')
            trial = next(row for row in cycle['measured'] if row['concurrency'] == concurrency)
            if len(trial['observations']) != 24 or trial['output_tokens'] != 1536:
                raise RuntimeError('trial shape differs')
            for observation, prompt in zip(trial['observations'], prompts):
                meta = observation['meta_info']
                if (observation['index'] != prompt['index'] or
                        observation['prompt_hash'] != prompt['prompt_hash'] or
                        observation['prompt_tokens'] != prompt['prompt_tokens'] or
                        observation['completion_tokens'] != 64 or
                        meta['prompt_tokens'] != prompt['prompt_tokens'] or
                        meta['completion_tokens'] != 64 or
                        meta['finish_reason']['type'] != 'length'):
                    raise RuntimeError('request identity or output count differs')
            trials.append(trial)
        output = sum(trial['output_tokens'] for trial in trials)
        elapsed = sum(trial['elapsed_seconds'] for trial in trials)
        rate = output / elapsed
        saved = result['summary'][str(concurrency)]
        if (saved['output_tokens'] != output or saved['request_count'] != 72 or
                abs(saved['elapsed_seconds'] - elapsed) > 1e-6 or
                abs(saved['output_tokens_per_second'] - rate) > 1e-6):
            raise RuntimeError('throughput summary differs')
        report[str(concurrency)] = {'output_tokens': output,
                                    'elapsed_seconds': elapsed,
                                    'output_tokens_per_second': rate,
                                    'cycle_rates': [trial['output_tokens'] / trial['elapsed_seconds']
                                                    for trial in trials]}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
