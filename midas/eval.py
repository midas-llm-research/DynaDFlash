"""
Throughput evaluation for any DFlash draft model on the transformers backend.

Runs baseline (block_size=1, no spec decoding) and DFlash speculative decoding
(block_size=N) on standard benchmarks, then reports throughput, speedup, mean
acceptance length, and acceptance histogram.

Usage (baseline DFlash, z-lab checkpoint):
    python -m midas.eval --target Qwen/Qwen3-4B \\
        --draft z-lab/Qwen3-4B-DFlash-b16 --dataset gsm8k

Usage (routed variant with a trained checkpoint):
    python -m midas.eval --target Qwen/Qwen3-4B \\
        --draft z-lab/Qwen3-4B-DFlash-b16 --routed \\
        --router-ckpt /checkpoints/midas-router/best/model.pt --dataset gsm8k

To evaluate a custom DFlashDraftModel subclass, call `evaluate()` directly
with your already-constructed draft model.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from midas.models import DFlashDraftModel, DFlashDraftModelRouted, dflash_generate

random.seed(42)

CACHE_DIR = Path(__file__).parent.parent / "cache"


DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    out_path = CACHE_DIR / f"{name}.jsonl"
    logger.info(f"Downloading {name} -> {out_path}")
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])
    with open(out_path, "w") as f:
        for row in dataset:
            if cfg.get("multi_turn"):
                turns = cfg["format"](row)
            else:
                turns = [cfg["format"](row)]
            f.write(json.dumps({"turns": turns}) + "\n")
    return out_path


def load_benchmark(name: str) -> list[dict]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    path = CACHE_DIR / f"{name}.jsonl"
    if not path.exists():
        _prepare_dataset(name)
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def evaluate(
    target: torch.nn.Module,
    draft: DFlashDraftModel,
    tokenizer,
    dataset_name: str,
    block_size: Optional[int] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    max_samples: Optional[int] = None,
    enable_thinking: bool = False,
) -> tuple[list[dict], int]:
    """
    Run baseline (block_size=1) and speculative (block_size=N) decoding on every
    sample in `dataset_name`. Returns `(responses, block_size)` where each
    response is `{1: stats_baseline, N: stats_spec}` from `dflash_generate`.
    """
    block_size = block_size or draft.block_size
    dataset = load_benchmark(dataset_name)
    if max_samples is not None and len(dataset) > max_samples:
        random.shuffle(dataset)
        dataset = dataset[:max_samples]

    responses: list[dict] = []
    for instance in tqdm(dataset, desc=f"eval/{dataset_name}"):
        messages: list[dict] = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            sample_stats: dict = {}
            for bs in (1, block_size):
                sample_stats[bs] = dflash_generate(
                    draft,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=temperature,
                    block_size=bs,
                    return_stats=True,
                )

            # Append assistant turn so multi-turn datasets see realistic context
            spec = sample_stats[block_size]
            generated_ids = spec.output_ids[0, spec.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(sample_stats)

    return responses, block_size


def report(responses: list[dict], block_size: int) -> dict:
    """Print + return a summary of throughput, speedup, and acceptance stats."""
    baseline_tpot = float(np.mean([r[1].time_per_output_token for r in responses]))
    dflash_tpot = float(np.mean([r[block_size].time_per_output_token for r in responses]))
    mean_accept = float(np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses]))

    accept_all = list(chain.from_iterable(r[block_size].acceptance_lengths for r in responses))
    histogram = [accept_all.count(b) / len(accept_all) for b in range(block_size + 1)]

    summary = {
        "baseline_tps": 1.0 / baseline_tpot,
        "dflash_tps": 1.0 / dflash_tpot,
        "speedup": baseline_tpot / dflash_tpot,
        "mean_acceptance_length": mean_accept,
        "acceptance_histogram": histogram,
        "num_samples": len(responses),
        "block_size": block_size,
    }

    print()
    print("=" * 50)
    print(f"Samples evaluated:      {summary['num_samples']}")
    print(f"Block size:             {block_size}")
    print(f"Baseline throughput:    {summary['baseline_tps']:,.2f} tok/s")
    print(f"DFlash throughput:      {summary['dflash_tps']:,.2f} tok/s")
    print(f"Decoding speedup:       {summary['speedup']:.2f}x")
    print(f"Mean acceptance length: {mean_accept:.2f}")
    print(f"Acceptance histogram:   {[f'{x * 100:.1f}%' for x in histogram]}")
    print("=" * 50)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _get_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash_attn not installed — using sdpa. Install for better speedup: "
            "pip install flash-attn --no-build-isolation"
        )
        return "sdpa"


def _load_draft(
    draft_path: str,
    routed: bool,
    router_ckpt: Optional[str],
    config_path: Optional[str],
    device: torch.device,
    attn_impl: str,
) -> DFlashDraftModel:
    if routed:
        from midas.config import RouterConfig, TrainConfig

        if config_path:
            router_config = TrainConfig.from_yaml(config_path).router
        else:
            router_config = RouterConfig()
        draft = DFlashDraftModelRouted.from_pretrained_dflash(
            draft_path,
            router_config=router_config,
            device=device,
            dtype=torch.bfloat16,
        )
        if router_ckpt:
            logger.info(f"Loading trained router weights from {router_ckpt}")
            state = torch.load(router_ckpt, weights_only=True, map_location=device)
            draft.load_state_dict(state)
        return draft

    return (
        DFlashDraftModel.from_pretrained(
            draft_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DFlash throughput eval (transformers backend)"
    )
    parser.add_argument("--target", required=True, help="Target model name/path")
    parser.add_argument("--draft", required=True, help="Draft model checkpoint (DFlash format)")
    parser.add_argument("--dataset", default="gsm8k", choices=list(DATASETS))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=None,
                        help="Override draft.block_size (default uses model config)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--routed", action="store_true",
                        help="Use DFlashDraftModelRouted instead of base DFlashDraftModel")
    parser.add_argument("--router-ckpt", type=str, default=None,
                        help="Path to trained router state_dict (model.pt). Implies --routed.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config providing router params (only used with --routed)")
    args = parser.parse_args()

    if args.router_ckpt and not args.routed:
        args.routed = True

    if args.enable_thinking and any(
        s in args.target.lower() for s in ["qwen3-4b", "qwen3-8b"]
    ):
        raise ValueError(
            "DFlash drafts for Qwen3-4B/8B were not trained with thinking traces — "
            "--enable-thinking will produce poor speedups."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = _get_attn_impl()

    logger.info(f"Loading target: {args.target}")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    logger.info(f"Loading draft: {args.draft} (routed={args.routed})")
    draft = _load_draft(
        args.draft, args.routed, args.router_ckpt, args.config, device, attn_impl
    )
    draft.eval()

    responses, block_size = evaluate(
        target,
        draft,
        tokenizer,
        dataset_name=args.dataset,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_samples=args.max_samples,
        enable_thinking=args.enable_thinking,
    )
    report(responses, block_size)


if __name__ == "__main__":
    main()
