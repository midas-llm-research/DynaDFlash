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

Usage (DFlare variant with a trained checkpoint):
    python -m midas.eval --target Qwen/Qwen3-4B \\
        --draft z-lab/Qwen3-4B-DFlash-b16 --variant dflare \\
        --dflare-ckpt /checkpoints/dflare-qwen3-4b/best/model.pt \\
        --config configs/dflare_4b.yaml --dataset gsm8k

To evaluate a custom DFlashDraftModel subclass, call `evaluate()` directly
with your already-constructed draft model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from midas.models import (
    DFlashDraftModel,
    DFlashDraftModelRouted,
    DFlareDraftModel,
    dflash_generate,
)

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
    "mt_bench": {
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
# Prompt loading
# ---------------------------------------------------------------------------


def load_eval_prompts(dataset_name: str, n: int = 50, seed: int = 42) -> list[str]:
    """Load n prompts from a benchmark with a fixed seed for reproducibility."""
    import random as _rng

    dataset = load_benchmark(dataset_name)
    rng = _rng.Random(seed)
    samples = rng.sample(dataset, min(n, len(dataset)))

    prompts = []
    for item in samples:
        turns = item["turns"]
        prompts.append(turns[0])  # first turn only (handles multi-turn like mt_bench)
    return prompts


# ---------------------------------------------------------------------------
# Throughput evaluation
# ---------------------------------------------------------------------------


def measure_ar_baseline(
    target: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 2048,
    num_warmup: int = 3,
) -> dict:
    """Measure autoregressive baseline throughput. Returns {tps, tokens, time_s}."""
    import time

    device = next(target.parameters()).device
    use_cuda = device.type == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    logger.info("Warmup (AR)...")
    wids = tokenizer("Solve: 2+2=", return_tensors="pt").input_ids.to(device)
    for _ in range(num_warmup):
        with torch.no_grad():
            target.generate(wids, max_new_tokens=32, do_sample=False)

    logger.info(f"Measuring AR baseline ({len(prompts)} prompts, max_new_tokens={max_new_tokens})...")
    ar_times, ar_tokens = [], []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        sync()
        t1 = time.perf_counter()
        ar_tokens.append(out.shape[1] - input_ids.shape[1])
        ar_times.append(t1 - t0)
        if (i + 1) % 10 == 0:
            logger.info(f"  AR: {i+1}/{len(prompts)}")

    ar_tps = sum(ar_tokens) / sum(ar_times)
    logger.info(f"Autoregressive: {ar_tps:.1f} tok/s ({sum(ar_tokens)} tokens, {sum(ar_times):.1f}s)")
    if use_cuda:
        torch.cuda.empty_cache()
    return {"tps": ar_tps, "tokens": sum(ar_tokens), "time_s": sum(ar_times)}


def evaluate(
    draft: DFlashDraftModel,
    target: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    num_warmup: int = 3,
    ar_baseline: Optional[dict] = None,
) -> dict:
    """
    Evaluate speculative decoding throughput against autoregressive baseline.

    If `ar_baseline` is provided (from `measure_ar_baseline`), skips AR
    measurement and reuses those numbers. Otherwise measures AR first.

    Returns metrics dict with throughput, acceptance length, and speedup.
    """
    import time

    device = next(target.parameters()).device
    use_cuda = device.type == "cuda"
    eos = [tokenizer.eos_token_id]

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    # Phase 0: AR baseline — measure, reuse, or skip.
    if ar_baseline is None:
        ar_tps = None
    else:
        ar_tps = ar_baseline["tps"]

    # Phase 1: Warmup speculative decoding.
    logger.info("Warmup (speculative)...")
    wids = tokenizer("Solve: 2+2=", return_tensors="pt").input_ids.to(device)
    for _ in range(num_warmup):
        with torch.no_grad():
            dflash_generate(draft, target, wids, max_new_tokens=32,
                          stop_token_ids=eos, temperature=0.0)
    if use_cuda:
        torch.cuda.empty_cache()

    # Phase 2: Speculative decoding — same prompts, same max_new_tokens.
    logger.info(f"Measuring speculative decoding ({len(prompts)} prompts)...")
    spec_results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = dflash_generate(
                draft, target, input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=eos,
                temperature=temperature, return_stats=True,
            )
        sync()
        t1 = time.perf_counter()
        spec_results.append({
            "tokens": stats.num_output_tokens,
            "time": t1 - t0,
            "acceptance": stats.acceptance_lengths,
        })
        if (i + 1) % 10 == 0:
            logger.info(f"  Spec: {i+1}/{len(prompts)}")

    # Phase 3: Compute metrics.
    spec_total_tokens = sum(r["tokens"] for r in spec_results)
    spec_total_time = sum(r["time"] for r in spec_results)
    spec_tps = spec_total_tokens / spec_total_time
    all_acc = [a for r in spec_results for a in r["acceptance"]]
    mean_acc = float(np.mean(all_acc))
    speedup = (spec_tps / ar_tps) if ar_tps else None

    logger.info(f"Speculative: {spec_tps:.1f} tok/s ({spec_total_tokens} tokens, {spec_total_time:.1f}s)")
    logger.info(f"Acceptance length (τ): {mean_acc:.2f}")
    if speedup is not None:
        logger.info(f"Speedup over AR: {speedup:.2f}x")

    per_prompt = [
        {"tokens": r["tokens"], "time_s": round(r["time"], 3),
         "tps": round(r["tokens"] / r["time"], 1) if r["time"] > 0 else 0,
         "acceptance_lengths": r["acceptance"]}
        for r in spec_results
    ]

    result = {
        "tokens_per_second": round(spec_tps, 1),
        "mean_acceptance_length": round(mean_acc, 2),
        "speedup_over_ar": round(speedup, 2) if speedup is not None else None,
        "autoregressive_tps": round(ar_tps, 1) if ar_tps is not None else None,
        "total_tokens": spec_total_tokens,
        "total_time_s": round(spec_total_time, 1),
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "per_prompt": per_prompt,
        "all_acceptance_lengths": all_acc,
    }
    if ar_baseline is not None:
        result["ar_total_tokens"] = ar_baseline["tokens"]
        result["ar_total_time_s"] = round(ar_baseline["time_s"], 1)
    return result


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
    variant: str,
    checkpoint: Optional[str],
    config_path: Optional[str],
    device: torch.device,
    attn_impl: str,
) -> DFlashDraftModel:
    if variant == "router":
        from midas.config import RouterConfig, TrainConfig

        if config_path:
            router_config = TrainConfig.from_yaml(config_path).router
            if router_config is None:
                router_config = RouterConfig()
        else:
            router_config = RouterConfig()
        draft = DFlashDraftModelRouted.from_pretrained_dflash(
            draft_path,
            router_config=router_config,
            device=device,
            dtype=torch.bfloat16,
        )
        if checkpoint:
            logger.info(f"Loading trained router weights from {checkpoint}")
            state = torch.load(checkpoint, weights_only=True, map_location=device)
            draft.load_state_dict(state)
        return draft

    if variant == "dflare":
        from midas.config import DFlareConfig, TrainConfig

        if config_path:
            dflare_config = TrainConfig.from_yaml(config_path).dflare
            if dflare_config is None:
                dflare_config = DFlareConfig()
        else:
            dflare_config = DFlareConfig()
        draft = DFlareDraftModel.from_pretrained_dflash(
            draft_path,
            dflare_config=dflare_config,
            device=device,
            dtype=torch.bfloat16,
        )
        if checkpoint:
            logger.info(f"Loading trained DFlare weights from {checkpoint}")
            state = torch.load(checkpoint, weights_only=True, map_location=device)
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
    parser.add_argument("--dataset", default="math500", choices=list(DATASETS))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--variant", choices=["base", "router", "dflare"], default="base",
                        help="Draft variant to load")
    parser.add_argument("--routed", action="store_true",
                        help="Backward-compatible alias for --variant router")
    parser.add_argument("--dflare", action="store_true",
                        help="Backward-compatible alias for --variant dflare")
    parser.add_argument("--router-ckpt", type=str, default=None,
                        help="Path to trained router state_dict (model.pt). Implies --routed.")
    parser.add_argument("--dflare-ckpt", type=str, default=None,
                        help="Path to trained DFlare state_dict (model.pt). Implies --variant dflare.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config providing variant params (router or dflare)")
    args = parser.parse_args()

    if (args.routed or args.router_ckpt) and (args.dflare or args.dflare_ckpt):
        parser.error("Cannot combine --routed/--router-ckpt with --dflare/--dflare-ckpt")
    if args.routed or args.router_ckpt:
        args.variant = "router"
    if args.dflare or args.dflare_ckpt:
        args.variant = "dflare"
    checkpoint = (
        args.router_ckpt if args.variant == "router"
        else args.dflare_ckpt if args.variant == "dflare"
        else None
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = _get_attn_impl()

    logger.info(f"Loading target: {args.target}")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target, attn_implementation=attn_impl, torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    logger.info(f"Loading draft: {args.draft} (variant={args.variant})")
    draft = _load_draft(
        args.draft, args.variant, checkpoint, args.config, device, attn_impl
    )
    draft.eval()

    # Load prompts
    dataset = load_benchmark(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset[:args.max_samples]
    prompts = [d["turns"][0] for d in dataset]

    results = evaluate(
        draft, target, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
