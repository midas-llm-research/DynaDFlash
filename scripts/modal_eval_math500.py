"""
Evaluation on MATH-500: z-lab DFlash vs Our Baseline.
Reports acceptance length, throughput, latency per block, total time.

Usage:
    modal run scripts/modal_eval_math500.py
"""

import modal
from pathlib import Path

app = modal.App("midas-eval-math500")

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
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from loguru import logger

    sys.path.insert(0, "/root")
    from baseline.model import DFlashDraftModel
    from baseline.model import dflash_generate as baseline_generate

    device = torch.device("cuda")

    # === Load target ===
    logger.info("Loading target: Qwen3-4B")
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Load z-lab published DFlash ===
    logger.info("Loading z-lab DFlash (published)")
    zlab_draft = AutoModel.from_pretrained(
        "/models/Qwen3-4B-DFlash-b16", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()

    # === Load our baseline ===
    logger.info("Loading our baseline DFlash (SDD-trained)")
    our_draft = DFlashDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16)
    ckpt = Path("/checkpoints/midas-baseline/best")
    if ckpt.is_symlink():
        ckpt = ckpt.resolve()
    sd = torch.load(ckpt / "model.pt", weights_only=True, map_location=device)
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    our_draft.load_state_dict(sd)
    our_draft.eval()
    logger.info(f"Loaded baseline from {ckpt}")

    # === Load MATH-500 ===
    logger.info("Loading MATH-500 dataset")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [s["problem"] for s in list(math500)[:50]]
    logger.info(f"Evaluating on {len(prompts)} MATH-500 problems")

    # === Warmup ===
    logger.info("Warmup...")
    wids = tokenizer("Solve: 2+2", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        zlab_draft.spec_generate(target=target, input_ids=wids, max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)
        baseline_generate(our_draft, target, wids, max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)

    # === Evaluate both models ===
    def run_eval(name, generate_fn, prompts):
        results = []
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            stats = generate_fn(input_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            results.append(stats | {"wall_time": t1 - t0})

            if (i + 1) % 100 == 0:
                logger.info(f"  {name}: {i+1}/{len(prompts)} done")

        return results

    # z-lab eval
    logger.info("Evaluating z-lab DFlash on MATH-500...")
    zlab_results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = zlab_draft.spec_generate(
                target=target, input_ids=input_ids,
                max_new_tokens=2048, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # spec_generate returns output_ids tensor (no stats)
        num_tokens = stats.shape[1] - input_ids.shape[1]
        zlab_results.append({"tokens": num_tokens, "wall_time": t1 - t0})
        if (i + 1) % 100 == 0:
            logger.info(f"  z-lab: {i+1}/{len(prompts)} done")

    # Our baseline eval (with detailed stats)
    logger.info("Evaluating our baseline on MATH-500...")
    our_results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = baseline_generate(
                our_draft, target, input_ids,
                max_new_tokens=2048, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                return_stats=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        our_results.append({
            "tokens": stats.num_output_tokens,
            "wall_time": t1 - t0,
            "acceptance_lengths": stats.acceptance_lengths,
            "time_to_first_token": stats.time_to_first_token,
            "time_per_output_token": stats.time_per_output_token,
        })
        if (i + 1) % 100 == 0:
            logger.info(f"  ours: {i+1}/{len(prompts)} done")

    # Get z-lab acceptance by running through our generate function (same 50 prompts)
    logger.info("Getting z-lab acceptance lengths via dflash_generate...")
    zlab_acceptance_results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            stats = baseline_generate(
                zlab_draft, target, input_ids,
                max_new_tokens=2048, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                return_stats=True,
            )
        zlab_acceptance_results.append(stats.acceptance_lengths)

    # === Compute Metrics ===
    # z-lab metrics
    zlab_total_tokens = sum(r["tokens"] for r in zlab_results)
    zlab_total_time = sum(r["wall_time"] for r in zlab_results)
    zlab_tps = zlab_total_tokens / zlab_total_time
    zlab_avg_latency = zlab_total_time / len(zlab_results)
    zlab_all_acceptance = [a for r in zlab_acceptance_results for a in r]
    zlab_mean_acceptance = float(np.mean(zlab_all_acceptance)) if zlab_all_acceptance else 0

    # Our metrics
    our_total_tokens = sum(r["tokens"] for r in our_results)
    our_total_time = sum(r["wall_time"] for r in our_results)
    our_tps = our_total_tokens / our_total_time
    our_avg_latency = our_total_time / len(our_results)
    our_all_acceptance = [a for r in our_results for a in r["acceptance_lengths"]]
    our_mean_acceptance = float(np.mean(our_all_acceptance))
    our_ttft = float(np.mean([r["time_to_first_token"] for r in our_results if r["time_to_first_token"]]))
    our_tpot = float(np.mean([r["time_per_output_token"] for r in our_results]))

    # Number of draft iterations (blocks)
    our_total_blocks = sum(len(r["acceptance_lengths"]) for r in our_results)
    our_block_latency = our_total_time / our_total_blocks if our_total_blocks > 0 else 0
    zlab_total_blocks_est = zlab_total_tokens / zlab_mean_acceptance if zlab_mean_acceptance > 0 else 1
    zlab_block_latency = zlab_total_time / zlab_total_blocks_est

    results = {
        "dataset": "MATH-500",
        "num_prompts": len(prompts),
        "zlab_dflash": {
            "tokens_per_second": round(zlab_tps, 1),
            "total_tokens": zlab_total_tokens,
            "total_time_s": round(zlab_total_time, 1),
            "avg_latency_per_prompt_s": round(zlab_avg_latency, 3),
            "mean_acceptance_length": round(zlab_mean_acceptance, 2),
            "est_block_latency_ms": round(zlab_block_latency * 1000, 1),
        },
        "our_baseline": {
            "tokens_per_second": round(our_tps, 1),
            "total_tokens": our_total_tokens,
            "total_time_s": round(our_total_time, 1),
            "avg_latency_per_prompt_s": round(our_avg_latency, 3),
            "mean_acceptance_length": round(our_mean_acceptance, 2),
            "time_to_first_token_ms": round(our_ttft * 1000, 1),
            "time_per_output_token_ms": round(our_tpot * 1000, 2),
            "block_latency_ms": round(our_block_latency * 1000, 1),
            "total_blocks": our_total_blocks,
        },
    }

    logger.info("=" * 70)
    logger.info("MATH-500 EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<35} {'z-lab DFlash':<20} {'Our Baseline':<20}")
    logger.info("-" * 70)
    logger.info(f"{'Tokens/second':<35} {zlab_tps:<20.1f} {our_tps:<20.1f}")
    logger.info(f"{'Mean acceptance length':<35} {zlab_mean_acceptance:<20.2f} {our_mean_acceptance:<20.2f}")
    logger.info(f"{'Avg latency/prompt (s)':<35} {zlab_avg_latency:<20.3f} {our_avg_latency:<20.3f}")
    logger.info(f"{'Block latency (ms)':<35} {zlab_block_latency*1000:<20.1f} {our_block_latency*1000:<20.1f}")
    logger.info(f"{'Total tokens generated':<35} {zlab_total_tokens:<20} {our_total_tokens:<20}")
    logger.info("=" * 70)

    with open("/checkpoints/midas-baseline/eval_math500.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


@app.local_entrypoint()
def main():
    results = evaluate.remote()
    print("\n" + "=" * 70)
    print("  MATH-500: Z-LAB DFLASH vs OUR BASELINE")
    print("=" * 70)
    z = results["zlab_dflash"]
    o = results["our_baseline"]
    print(f"  {'Metric':<35} {'z-lab':<15} {'Ours':<15}")
    print(f"  {'-'*65}")
    print(f"  {'Tokens/sec':<35} {z['tokens_per_second']:<15} {o['tokens_per_second']:<15}")
    print(f"  {'Acceptance length':<35} {z['mean_acceptance_length']:<15} {o['mean_acceptance_length']:<15}")
    print(f"  {'Avg latency/prompt (s)':<35} {z['avg_latency_per_prompt_s']:<15} {o['avg_latency_per_prompt_s']:<15}")
    print(f"  {'Block latency (ms)':<35} {z['est_block_latency_ms']:<15} {o['block_latency_ms']:<15}")
    print("=" * 70)
