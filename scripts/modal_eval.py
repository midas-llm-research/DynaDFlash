"""
Evaluate a single DFlash variant on MATH-500 (50 problems).
Each invocation gets its own fresh GPU — run separately per variant
to eliminate warmup/caching bias between models.

Usage:
    modal run scripts/modal_eval.py --variant baseline
    modal run scripts/modal_eval.py --variant router
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


CHECKPOINTS = {
    "baseline": "/checkpoints/midas-baseline/checkpoint_step9276",
    "router": "/checkpoints/midas-router-v2/checkpoint_step9276",
    "router-nofc": "/checkpoints/midas-router-nofc-v3/best",
}


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
def evaluate(variant: str = "baseline"):
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
    from midas.models.model import DFlashDraftModel, dflash_generate
    from midas.models.model_context_router import DFlashDraftModelRouted
    from midas.models.model_context_router_nofc import DFlashDraftModelRoutedNoFC

    device = torch.device("cuda")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)

    ckpt_path = CHECKPOINTS[variant]
    logger.info(f"Evaluating variant={variant}, checkpoint={ckpt_path}")

    # === Load target model ===
    logger.info("Loading target: Qwen3-4B")
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Load draft model ===
    if variant == "baseline":
        model = DFlashDraftModel.from_config_random(
            "/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16,
        )
    elif variant == "router":
        model = DFlashDraftModelRouted.from_config_random(
            "/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16,
            router_config=router_config,
        )
    elif variant == "router-nofc":
        model = DFlashDraftModelRoutedNoFC.from_config_random(
            "/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16,
            router_config=router_config,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'baseline', 'router', or 'router-nofc'.")

    ckpt = Path(ckpt_path)
    if ckpt.is_symlink():
        ckpt = ckpt.resolve()
    sd = torch.load(ckpt / "model.pt", weights_only=True, map_location=device)
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    with open(ckpt / "metadata.json") as f:
        meta = json.load(f)
    logger.info(f"Loaded {variant}: step={meta['step']}, loss={meta['loss']:.4f}")

    # === Load MATH-500 (50 problems) ===
    logger.info("Loading MATH-500")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [s["problem"] for s in list(math500)[:50]]
    logger.info(f"Evaluating on {len(prompts)} MATH-500 problems (greedy, max_new_tokens=2048)")

    # === Measure autoregressive baseline (10 prompts) ===
    logger.info("Measuring autoregressive baseline...")
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

    # === Warmup (3 runs) ===
    logger.info("Warmup...")
    wids = tokenizer("Solve: 1+1=", return_tensors="pt").input_ids.to(device)
    for _ in range(3):
        with torch.no_grad():
            dflash_generate(model, target, wids, max_new_tokens=64,
                          stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)

    # === Evaluate on all 50 prompts ===
    logger.info(f"Evaluating {variant}...")
    results = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = dflash_generate(
                model, target, input_ids,
                max_new_tokens=2048, stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0, return_stats=True,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        results.append({
            "tokens": stats.num_output_tokens,
            "time": t1 - t0,
            "acceptance": stats.acceptance_lengths,
        })
        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(prompts)}")

    # === Compute metrics ===
    total_tokens = sum(r["tokens"] for r in results)
    total_time = sum(r["time"] for r in results)
    tps = total_tokens / total_time
    all_acc = [a for r in results for a in r["acceptance"]]
    mean_acc = float(np.mean(all_acc))
    speedup = tps / ar_tps

    output = {
        "variant": variant,
        "checkpoint": ckpt_path,
        "training_step": meta["step"],
        "training_loss": meta["loss"],
        "autoregressive_tps": round(ar_tps, 1),
        "tokens_per_second": round(tps, 1),
        "mean_acceptance_length": round(mean_acc, 2),
        "speedup_over_ar": round(speedup, 2),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {variant.upper()} RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Tokens/sec:           {tps:.1f}")
    logger.info(f"  Acceptance length:    {mean_acc:.2f}")
    logger.info(f"  Speedup over AR:      {speedup:.2f}x")
    logger.info(f"  AR baseline:          {ar_tps:.1f} tok/s")
    logger.info("=" * 60)

    out_path = f"/checkpoints/eval_{variant}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    checkpoint_volume.commit()
    return output


@app.local_entrypoint()
def main(variant: str = "baseline"):
    results = evaluate.remote(variant=variant)
    print(f"\n{'=' * 60}")
    print(f"  {variant.upper()} RESULTS")
    print(f"{'=' * 60}")
    print(f"  Tokens/sec:           {results['tokens_per_second']}")
    print(f"  Acceptance length:    {results['mean_acceptance_length']}")
    print(f"  Speedup over AR:      {results['speedup_over_ar']}x")
    print(f"  AR baseline:          {results['autoregressive_tps']} tok/s")
    print(f"{'=' * 60}")
