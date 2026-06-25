"""
Head-to-head evaluation: Baseline DFlash vs Router DFlash.
Both trained identically from scratch on same SDD data.
Reports acceptance length (τ), tokens/sec, speedup over autoregressive.

Uses MATH-500 (50 problems) matching DFlash paper eval.

Usage:
    modal run scripts/modal_eval_compare.py
"""

import modal
from pathlib import Path

app = modal.App("midas-eval-compare")

checkpoint_volume = modal.Volume.from_name("midas-checkpoints")
model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)

PROJECT_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.57.0",
        "accelerate",
        "datasets",
        "numpy",
        "tqdm",
        "loguru",
        "huggingface_hub",
    )
    .add_local_dir(str(PROJECT_ROOT / "midas"), "/root/midas", copy=True)
    .add_local_dir(str(PROJECT_ROOT / "baseline"), "/root/baseline", copy=True)
)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
)
def evaluate():
    import json
    import sys
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from loguru import logger

    sys.path.insert(0, "/root")
    from baseline.model import DFlashDraftModel
    from baseline.model import dflash_generate as baseline_generate
    from midas.config import RouterConfig
    from midas.model import DFlashDraftModelRouted
    from midas.model import dflash_generate as router_generate

    device = torch.device("cuda")

    # === Load target model ===
    logger.info("Loading target: Qwen3-4B")
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Load baseline model ===
    logger.info("Loading baseline DFlash (trained)")
    baseline_draft = DFlashDraftModel.from_config_random(
        "/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16,
    )
    baseline_ckpt = Path("/checkpoints/midas-baseline/checkpoint_step9276")
    sd = torch.load(baseline_ckpt / "model.pt", weights_only=True, map_location=device)
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    baseline_draft.load_state_dict(sd)
    baseline_draft.eval()
    with open(baseline_ckpt / "metadata.json") as f:
        baseline_meta = json.load(f)
    logger.info(f"Baseline: step={baseline_meta['step']}, loss={baseline_meta['loss']:.4f}")

    # === Load router model ===
    logger.info("Loading router DFlash (trained)")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)
    router_draft = DFlashDraftModelRouted.from_pretrained_dflash(
        "/models/Qwen3-4B-DFlash-b16", router_config=router_config, device=device, dtype=torch.bfloat16,
    )
    router_ckpt = Path("/checkpoints/midas-router-v2/checkpoint_step9276")
    rsd = torch.load(router_ckpt / "model.pt", weights_only=True, map_location=device)
    rsd = {k: v.to(torch.bfloat16) for k, v in rsd.items()}
    router_draft.load_state_dict(rsd)
    router_draft.eval()
    with open(router_ckpt / "metadata.json") as f:
        router_meta = json.load(f)
    logger.info(f"Router: step={router_meta['step']}, loss={router_meta['loss']:.4f}")

    # === Load MATH-500 (50 problems) ===
    logger.info("Loading MATH-500")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [s["problem"] for s in list(math500)[:50]]
    logger.info(f"Evaluating on {len(prompts)} MATH-500 problems (greedy, max_new_tokens=2048)")

    # === Measure autoregressive baseline (target only, no drafting) ===
    logger.info("Measuring autoregressive baseline (no spec decode)...")
    ar_times = []
    ar_tokens = []
    for prompt in prompts[:10]:  # 10 samples for AR baseline
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target.generate(input_ids, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ar_tokens.append(out.shape[1] - input_ids.shape[1])
        ar_times.append(t1 - t0)
    ar_tps = sum(ar_tokens) / sum(ar_times)
    logger.info(f"Autoregressive: {ar_tps:.1f} tok/s")

    # === Warmup both draft models ===
    logger.info("Warmup...")
    wids = tokenizer("Solve: 1+1", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        baseline_generate(baseline_draft, target, wids, max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)
        router_generate(router_draft, target, wids, max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)

    # === Evaluate baseline ===
    logger.info("Evaluating baseline DFlash...")
    baseline_results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = baseline_generate(
                baseline_draft, target, input_ids,
                max_new_tokens=2048, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                return_stats=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline_results.append({
            "tokens": stats.num_output_tokens,
            "time": t1 - t0,
            "acceptance": stats.acceptance_lengths,
        })
        if (i + 1) % 10 == 0:
            logger.info(f"  baseline: {i+1}/{len(prompts)}")

    # === Evaluate router ===
    logger.info("Evaluating router DFlash...")
    router_results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = router_generate(
                router_draft, target, input_ids,
                max_new_tokens=2048, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                return_stats=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        router_results.append({
            "tokens": stats.num_output_tokens,
            "time": t1 - t0,
            "acceptance": stats.acceptance_lengths,
        })
        if (i + 1) % 10 == 0:
            logger.info(f"  router: {i+1}/{len(prompts)}")

    # === Compute metrics ===
    baseline_total_tokens = sum(r["tokens"] for r in baseline_results)
    baseline_total_time = sum(r["time"] for r in baseline_results)
    baseline_tps = baseline_total_tokens / baseline_total_time
    baseline_all_acc = [a for r in baseline_results for a in r["acceptance"]]
    baseline_mean_acc = float(np.mean(baseline_all_acc))
    baseline_speedup = baseline_tps / ar_tps

    router_total_tokens = sum(r["tokens"] for r in router_results)
    router_total_time = sum(r["time"] for r in router_results)
    router_tps = router_total_tokens / router_total_time
    router_all_acc = [a for r in router_results for a in r["acceptance"]]
    router_mean_acc = float(np.mean(router_all_acc))
    router_speedup = router_tps / ar_tps

    results = {
        "autoregressive_tps": round(ar_tps, 1),
        "baseline": {
            "tokens_per_second": round(baseline_tps, 1),
            "mean_acceptance_length": round(baseline_mean_acc, 2),
            "speedup_over_ar": round(baseline_speedup, 2),
            "total_tokens": baseline_total_tokens,
            "total_time_s": round(baseline_total_time, 1),
            "training_loss": baseline_meta["loss"],
            "fusion_params": "32.8M (concat+FC)",
        },
        "router": {
            "tokens_per_second": round(router_tps, 1),
            "mean_acceptance_length": round(router_mean_acc, 2),
            "speedup_over_ar": round(router_speedup, 2),
            "total_tokens": router_total_tokens,
            "total_time_s": round(router_total_time, 1),
            "training_loss": router_meta["loss"],
            "fusion_params": "7.2M (router+FC)",
        },
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("  HEAD-TO-HEAD: BASELINE DFlash vs ROUTER DFlash")
    logger.info("  (Both trained from scratch, same data, same steps)")
    logger.info("=" * 70)
    logger.info(f"  {'Metric':<30} {'Baseline (FC)':<20} {'Router (Ours)':<20}")
    logger.info(f"  {'-'*70}")
    logger.info(f"  {'Fusion params':<30} {'32.8M':<20} {'7.2M':<20}")
    logger.info(f"  {'Training loss':<30} {baseline_meta['loss']:<20.4f} {router_meta['loss']:<20.4f}")
    logger.info(f"  {'Acceptance length (τ)':<30} {baseline_mean_acc:<20.2f} {router_mean_acc:<20.2f}")
    logger.info(f"  {'Tokens/sec':<30} {baseline_tps:<20.1f} {router_tps:<20.1f}")
    logger.info(f"  {'Speedup over AR':<30} {baseline_speedup:<20.2f}x {router_speedup:<20.2f}x")
    logger.info(f"  {'-'*70}")
    logger.info(f"  {'Autoregressive baseline':<30} {ar_tps:.1f} tok/s")
    logger.info("=" * 70)

    with open("/checkpoints/eval_head_to_head.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


@app.local_entrypoint()
def main():
    results = evaluate.remote()
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: BASELINE DFlash vs ROUTER DFlash")
    print("=" * 70)
    b = results["baseline"]
    r = results["router"]
    print(f"  {'Metric':<30} {'Baseline (FC)':<20} {'Router (Ours)':<20}")
    print(f"  {'-'*70}")
    print(f"  {'Fusion params':<30} {'32.8M':<20} {'7.2M':<20}")
    print(f"  {'Training loss':<30} {b['training_loss']:<20.4f} {r['training_loss']:<20.4f}")
    print(f"  {'Acceptance length (τ)':<30} {b['mean_acceptance_length']:<20.2f} {r['mean_acceptance_length']:<20.2f}")
    print(f"  {'Tokens/sec':<30} {b['tokens_per_second']:<20.1f} {r['tokens_per_second']:<20.1f}")
    print(f"  {'Speedup over AR':<30} {b['speedup_over_ar']:<20.2f}x {r['speedup_over_ar']:<20.2f}x")
    print(f"  {'-'*70}")
    print(f"  {'Autoregressive baseline':<30} {results['autoregressive_tps']} tok/s")
    print("=" * 70)
