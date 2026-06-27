"""
Head-to-head evaluation: Baseline vs Router vs Router-NoFC.
All trained identically from scratch on same SDD data.
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
    .add_local_dir(str(PROJECT_ROOT / "configs"), "/root/configs", copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
)
def evaluate(
    baseline_ckpt: str = "/checkpoints/midas-baseline/checkpoint_step9276",
    router_ckpt: str = "/checkpoints/midas-router-v2/checkpoint_step9276",
    router_nofc_ckpt: str = "/checkpoints/midas-router-nofc-v3/best",
):
    import json
    import sys
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from loguru import logger

    sys.path.insert(0, "/root")
    from midas.config import RouterConfig
    from midas.models.model import DFlashDraftModel
    from midas.models.model import dflash_generate as baseline_generate
    from midas.models.model_context_router import DFlashDraftModelRouted
    from midas.models.model_context_router_nofc import DFlashDraftModelRoutedNoFC

    device = torch.device("cuda")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)

    # === Load target model ===
    logger.info("Loading target: Qwen3-4B")
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Load trained models ===
    def load_trained(cls, ckpt_path, **kwargs):
        model = cls.from_config_random(
            "/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, **kwargs
        )
        ckpt = Path(ckpt_path)
        if ckpt.is_symlink():
            ckpt = ckpt.resolve()
        sd = torch.load(ckpt / "model.pt", weights_only=True, map_location=device)
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        with open(ckpt / "metadata.json") as f:
            meta = json.load(f)
        logger.info(f"Loaded {cls.__name__} from {ckpt_path} (step={meta['step']}, loss={meta['loss']:.4f})")
        return model, meta

    logger.info("Loading baseline DFlash (trained)")
    baseline, baseline_meta = load_trained(DFlashDraftModel, baseline_ckpt)

    logger.info("Loading router DFlash (trained)")
    router, router_meta = load_trained(DFlashDraftModelRouted, router_ckpt, router_config=router_config)

    logger.info("Loading router-nofc DFlash (trained)")
    router_nofc, router_nofc_meta = load_trained(DFlashDraftModelRoutedNoFC, router_nofc_ckpt, router_config=router_config)

    # === Load MATH-500 (50 problems) ===
    logger.info("Loading MATH-500")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [s["problem"] for s in list(math500)[:50]]
    logger.info(f"Evaluating on {len(prompts)} MATH-500 problems (greedy, max_new_tokens=2048)")

    # === Measure autoregressive baseline (target only, no drafting) ===
    logger.info("Measuring autoregressive baseline (no spec decode)...")
    ar_times = []
    ar_tokens = []
    for prompt in prompts[:10]:
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

    # === Warmup all models ===
    logger.info("Warmup...")
    wids = tokenizer("Solve: 1+1", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        baseline_generate(baseline, target, wids, max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)
        baseline_generate(router, target, wids, max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)
        baseline_generate(router_nofc, target, wids, max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)

    # === Evaluate each model ===
    def eval_model(model, name):
        logger.info(f"Evaluating {name}...")
        results = []
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                stats = baseline_generate(
                    model, target, input_ids,
                    max_new_tokens=2048, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                    return_stats=True,
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results.append({
                "tokens": stats.num_output_tokens,
                "time": t1 - t0,
                "acceptance": stats.acceptance_lengths,
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  {name}: {i+1}/{len(prompts)}")
        return results

    baseline_results = eval_model(baseline, "baseline")
    router_results = eval_model(router, "router")
    router_nofc_results = eval_model(router_nofc, "router-nofc")

    # === Compute metrics ===
    def compute_metrics(results, meta):
        total_tokens = sum(r["tokens"] for r in results)
        total_time = sum(r["time"] for r in results)
        tps = total_tokens / total_time
        all_acc = [a for r in results for a in r["acceptance"]]
        return {
            "tokens_per_second": round(tps, 1),
            "mean_acceptance_length": round(float(np.mean(all_acc)), 2),
            "speedup_over_ar": round(tps / ar_tps, 2),
            "total_tokens": total_tokens,
            "total_time_s": round(total_time, 1),
            "training_loss": meta["loss"],
            "training_step": meta["step"],
        }

    results = {
        "autoregressive_tps": round(ar_tps, 1),
        "baseline": compute_metrics(baseline_results, baseline_meta),
        "router": compute_metrics(router_results, router_meta),
        "router_nofc": compute_metrics(router_nofc_results, router_nofc_meta),
    }

    # === Print results ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("  HEAD-TO-HEAD: BASELINE vs ROUTER vs ROUTER-NOFC")
    logger.info("  (All trained from scratch, same data, same steps)")
    logger.info("=" * 80)
    logger.info(f"  {'Metric':<30} {'Baseline (FC)':<20} {'Router':<20} {'Router-NoFC':<20}")
    logger.info(f"  {'-'*80}")
    logger.info(f"  {'Training loss':<30} {results['baseline']['training_loss']:<20.4f} {results['router']['training_loss']:<20.4f} {results['router_nofc']['training_loss']:<20.4f}")
    logger.info(f"  {'Acceptance length (τ)':<30} {results['baseline']['mean_acceptance_length']:<20.2f} {results['router']['mean_acceptance_length']:<20.2f} {results['router_nofc']['mean_acceptance_length']:<20.2f}")
    logger.info(f"  {'Tokens/sec':<30} {results['baseline']['tokens_per_second']:<20.1f} {results['router']['tokens_per_second']:<20.1f} {results['router_nofc']['tokens_per_second']:<20.1f}")
    logger.info(f"  {'Speedup over AR':<30} {results['baseline']['speedup_over_ar']:<20.2f}x {results['router']['speedup_over_ar']:<20.2f}x {results['router_nofc']['speedup_over_ar']:<20.2f}x")
    logger.info(f"  {'-'*80}")
    logger.info(f"  {'Autoregressive baseline':<30} {ar_tps:.1f} tok/s")
    logger.info("=" * 80)

    with open("/checkpoints/eval_compare_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


@app.local_entrypoint()
def main():
    results = evaluate.remote()
    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD: BASELINE vs ROUTER vs ROUTER-NOFC")
    print("=" * 80)
    b = results["baseline"]
    r = results["router"]
    n = results["router_nofc"]
    print(f"  {'Metric':<30} {'Baseline (FC)':<20} {'Router':<20} {'Router-NoFC':<20}")
    print(f"  {'-'*80}")
    print(f"  {'Training loss':<30} {b['training_loss']:<20.4f} {r['training_loss']:<20.4f} {n['training_loss']:<20.4f}")
    print(f"  {'Acceptance length (τ)':<30} {b['mean_acceptance_length']:<20.2f} {r['mean_acceptance_length']:<20.2f} {n['mean_acceptance_length']:<20.2f}")
    print(f"  {'Tokens/sec':<30} {b['tokens_per_second']:<20.1f} {r['tokens_per_second']:<20.1f} {n['tokens_per_second']:<20.1f}")
    print(f"  {'Speedup over AR':<30} {b['speedup_over_ar']:<20.2f}x {r['speedup_over_ar']:<20.2f}x {n['speedup_over_ar']:<20.2f}x")
    print(f"  {'-'*80}")
    print(f"  {'Autoregressive baseline':<30} {results['autoregressive_tps']} tok/s")
    print("=" * 80)
