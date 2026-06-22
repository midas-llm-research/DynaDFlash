"""
Evaluation: compare our trained baseline DFlash vs z-lab's published DFlash checkpoint.
Measures acceptance length and tokens/second on same prompts.

Usage:
    modal run scripts/modal_eval_baseline.py
"""

import modal
from pathlib import Path

app = modal.App("midas-eval-baseline")

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
    from baseline.model import DFlashDraftModel, dflash_generate

    device = torch.device("cuda")

    # === Load target model ===
    logger.info("Loading target: Qwen3-4B")
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B",
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Load z-lab's published DFlash draft (reference) ===
    logger.info("Loading z-lab DFlash draft (published checkpoint)")
    zlab_draft = AutoModel.from_pretrained(
        "/models/Qwen3-4B-DFlash-b16",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    # === Load our trained baseline ===
    logger.info("Loading our baseline DFlash (trained on SDD)")
    our_draft = DFlashDraftModel.from_config_random(
        "/models/Qwen3-4B-DFlash-b16",
        device=device,
        dtype=torch.bfloat16,
    )
    # Load trained weights from best checkpoint
    ckpt_path = Path("/checkpoints/midas-baseline/best")
    if ckpt_path.is_symlink():
        ckpt_path = ckpt_path.resolve()
    state_dict = torch.load(ckpt_path / "model.pt", weights_only=True, map_location=device)
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    our_draft.load_state_dict(state_dict)
    our_draft.eval()
    logger.info(f"Loaded baseline checkpoint from {ckpt_path}")

    # === Prepare eval prompts ===
    logger.info("Loading eval prompts...")
    prompts = []

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    for s in list(gsm8k)[:25]:
        prompts.append(("gsm8k", s["question"]))

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    for s in list(humaneval)[:25]:
        prompts.append(("humaneval", s["prompt"]))

    logger.info(f"Evaluating on {len(prompts)} prompts")

    # === Warmup ===
    logger.info("Warmup...")
    warmup_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        zlab_draft.spec_generate(
            target=target, input_ids=warmup_ids,
            max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
        )
        dflash_generate(
            our_draft, target, warmup_ids,
            max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
        )

    # === Evaluate z-lab model ===
    logger.info("Evaluating z-lab DFlash...")
    zlab_results = []
    for i, (domain, prompt_text) in enumerate(prompts):
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = zlab_draft.spec_generate(
                target=target, input_ids=input_ids,
                max_new_tokens=256, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tokens = output.shape[1] - input_ids.shape[1]
        zlab_results.append({"domain": domain, "tokens": tokens, "time": t1 - t0})

    # === Evaluate our baseline ===
    logger.info("Evaluating our baseline DFlash...")
    our_results = []
    for i, (domain, prompt_text) in enumerate(prompts):
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = dflash_generate(
                our_draft, target, input_ids,
                max_new_tokens=256, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
                return_stats=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        our_results.append({
            "domain": domain,
            "tokens": stats.num_output_tokens,
            "time": t1 - t0,
            "acceptance": stats.acceptance_lengths,
        })

    # === Compute metrics ===
    zlab_total_tokens = sum(r["tokens"] for r in zlab_results)
    zlab_total_time = sum(r["time"] for r in zlab_results)
    zlab_tps = zlab_total_tokens / zlab_total_time

    our_total_tokens = sum(r["tokens"] for r in our_results)
    our_total_time = sum(r["time"] for r in our_results)
    our_tps = our_total_tokens / our_total_time

    all_acceptances = []
    for r in our_results:
        all_acceptances.extend(r["acceptance"])

    results = {
        "zlab_dflash": {
            "total_tokens": zlab_total_tokens,
            "total_time_s": round(zlab_total_time, 2),
            "tokens_per_second": round(zlab_tps, 1),
        },
        "our_baseline": {
            "total_tokens": our_total_tokens,
            "total_time_s": round(our_total_time, 2),
            "tokens_per_second": round(our_tps, 1),
            "mean_acceptance": round(float(np.mean(all_acceptances)), 2),
            "median_acceptance": round(float(np.median(all_acceptances)), 2),
        },
        "comparison": {
            "tps_ratio": round(our_tps / zlab_tps, 4) if zlab_tps > 0 else 0,
        },
    }

    logger.info("=" * 60)
    logger.info("RESULTS: Our Baseline vs z-lab Published DFlash")
    logger.info("=" * 60)
    logger.info(f"z-lab DFlash:   {zlab_tps:.1f} tok/s ({zlab_total_tokens} tokens, {zlab_total_time:.1f}s)")
    logger.info(f"Our Baseline:   {our_tps:.1f} tok/s ({our_total_tokens} tokens, {our_total_time:.1f}s)")
    logger.info(f"Our Acceptance: {results['our_baseline']['mean_acceptance']:.2f} tokens/block")
    logger.info(f"TPS Ratio:      {results['comparison']['tps_ratio']:.4f} (>1 = ours faster)")
    logger.info("=" * 60)

    # Save
    output_path = Path("/checkpoints/midas-baseline/eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


@app.local_entrypoint()
def main():
    results = evaluate.remote()
    print("\n" + "=" * 60)
    print("  OUR BASELINE vs Z-LAB PUBLISHED DFLASH")
    print("=" * 60)
    print(f"  z-lab DFlash:  {results['zlab_dflash']['tokens_per_second']} tok/s")
    print(f"  Our Baseline:  {results['our_baseline']['tokens_per_second']} tok/s")
    print(f"  Acceptance:    {results['our_baseline']['mean_acceptance']} tokens/block")
    print(f"  TPS Ratio:     {results['comparison']['tps_ratio']} (>1 = ours faster)")
    print("=" * 60)
