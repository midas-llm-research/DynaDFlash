"""
Evaluation: compare routed DFlash vs original DFlash.
Measures tokens/second (throughput) on same prompts with same target.

Usage:
    modal run scripts/modal_eval.py
"""

import modal
from pathlib import Path

app = modal.App("midas-eval")

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
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=7200,
)
def evaluate(checkpoint_path: str = "/checkpoints/midas-router/best"):
    import json
    import sys
    import time

    import numpy as np
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from loguru import logger

    sys.path.insert(0, "/root")
    from midas.config import RouterConfig
    from midas.model import DFlashDraftModelRouted, dflash_generate

    device = torch.device("cuda")

    # ============================================================
    # Load models
    # ============================================================
    logger.info("Loading target: Qwen3-4B")
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B",
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    logger.info("Loading original DFlash draft")
    original_draft = AutoModel.from_pretrained(
        "/models/Qwen3-4B-DFlash-b16",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    logger.info(f"Loading routed draft from: {checkpoint_path}")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)
    routed_draft = DFlashDraftModelRouted.from_pretrained_dflash(
        "/models/Qwen3-4B-DFlash-b16",
        router_config=router_config,
        device=device,
        dtype=torch.bfloat16,
    )
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_symlink():
        ckpt_path = ckpt_path.resolve()
    state_dict = torch.load(ckpt_path / "model.pt", weights_only=True, map_location=device)
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    routed_draft.load_state_dict(state_dict)
    routed_draft.eval()

    # ============================================================
    # Prepare prompts
    # ============================================================
    logger.info("Loading eval prompts...")
    prompts = []

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    for s in list(gsm8k)[:25]:
        prompts.append(("gsm8k", s["question"]))

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    for s in list(humaneval)[:25]:
        prompts.append(("humaneval", s["prompt"]))

    logger.info(f"Evaluating on {len(prompts)} prompts")

    # ============================================================
    # Warmup (first call has compilation overhead)
    # ============================================================
    logger.info("Warmup...")
    warmup_ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        original_draft.spec_generate(
            target=target, input_ids=warmup_ids,
            max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
        )
        dflash_generate(
            routed_draft, target, warmup_ids,
            max_new_tokens=32, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
        )

    # ============================================================
    # Evaluate both models on same prompts
    # ============================================================
    orig_results = []
    routed_results = []

    for i, (domain, prompt_text) in enumerate(prompts):
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # --- Original DFlash ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            orig_out = original_draft.spec_generate(
                target=target,
                input_ids=input_ids,
                max_new_tokens=256,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        orig_tokens = orig_out.shape[1] - input_ids.shape[1]
        orig_results.append({"domain": domain, "tokens": orig_tokens, "time": t1 - t0})

        # --- Routed DFlash ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            routed_stats = dflash_generate(
                routed_draft, target, input_ids,
                max_new_tokens=256,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                return_stats=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        routed_tokens = routed_stats.num_output_tokens
        routed_results.append({
            "domain": domain,
            "tokens": routed_tokens,
            "time": t1 - t0,
            "acceptance": routed_stats.acceptance_lengths,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(prompts)} done")

    # ============================================================
    # Compute metrics
    # ============================================================
    orig_total_tokens = sum(r["tokens"] for r in orig_results)
    orig_total_time = sum(r["time"] for r in orig_results)
    orig_tps = orig_total_tokens / orig_total_time

    routed_total_tokens = sum(r["tokens"] for r in routed_results)
    routed_total_time = sum(r["time"] for r in routed_results)
    routed_tps = routed_total_tokens / routed_total_time

    all_acceptances = []
    for r in routed_results:
        all_acceptances.extend(r["acceptance"])

    results = {
        "original_dflash": {
            "total_tokens": orig_total_tokens,
            "total_time_s": round(orig_total_time, 2),
            "tokens_per_second": round(orig_tps, 1),
        },
        "routed_dflash": {
            "total_tokens": routed_total_tokens,
            "total_time_s": round(routed_total_time, 2),
            "tokens_per_second": round(routed_tps, 1),
            "mean_acceptance_length": round(float(np.mean(all_acceptances)), 2),
            "median_acceptance_length": round(float(np.median(all_acceptances)), 2),
        },
        "comparison": {
            "tps_ratio": round(routed_tps / orig_tps, 4) if orig_tps > 0 else 0,
            "time_ratio": round(routed_total_time / orig_total_time, 4) if orig_total_time > 0 else 0,
        },
    }

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Original DFlash:  {orig_tps:.1f} tok/s ({orig_total_tokens} tokens, {orig_total_time:.1f}s)")
    logger.info(f"Routed DFlash:    {routed_tps:.1f} tok/s ({routed_total_tokens} tokens, {routed_total_time:.1f}s)")
    logger.info(f"Acceptance:       {results['routed_dflash']['mean_acceptance_length']:.2f} tokens/block")
    logger.info(f"TPS ratio:        {results['comparison']['tps_ratio']:.4f} (>1 = routed is faster)")
    logger.info("=" * 60)

    # Save
    output_path = Path("/checkpoints/midas-router/eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


@app.local_entrypoint()
def main():
    results = evaluate.remote()
    print("\n" + "=" * 60)
    print("  ROUTED DFLASH vs ORIGINAL DFLASH")
    print("=" * 60)
    print(f"  Original: {results['original_dflash']['tokens_per_second']} tok/s")
    print(f"  Routed:   {results['routed_dflash']['tokens_per_second']} tok/s")
    print(f"  Acceptance: {results['routed_dflash']['mean_acceptance_length']} tokens/block")
    print(f"  TPS ratio: {results['comparison']['tps_ratio']} (>1 = routed faster)")
    print("=" * 60)
