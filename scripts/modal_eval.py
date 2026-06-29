"""
Evaluate a single DFlash variant on MATH-500 (50 problems).
Each invocation gets its own fresh GPU — run separately per variant
to eliminate warmup/caching bias between models.

Usage:
    modal run scripts/modal_eval.py --variant baseline
    modal run scripts/modal_eval.py --variant router
    modal run scripts/modal_eval.py --variant router-nofc
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
def run_eval(
    variant: str = "baseline",
    checkpoint: str = "",
    benchmarks: str = "math500,humaneval,mt_bench",
    seed: int = 42,
    n_prompts: int = 50,
):
    import json
    import sys
    from datetime import datetime, timezone

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from loguru import logger

    sys.path.insert(0, "/root")
    from midas.config import RouterConfig
    from midas.eval import evaluate, load_eval_prompts
    from midas.models.model import DFlashDraftModel
    from midas.models.model_context_router import DFlashDraftModelRouted

    device = torch.device("cuda")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)
    ckpt_path = checkpoint or CHECKPOINTS[variant]
    benchmark_list = [b.strip() for b in benchmarks.split(",")]
    logger.info(f"Evaluating variant={variant}, benchmarks={benchmark_list}, seed={seed}")

    # Load target
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # Load draft
    if variant == "baseline":
        model = DFlashDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16)
    elif variant == "router":
        model = DFlashDraftModelRouted.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, router_config=router_config)
    elif variant == "router-nofc":
        nofc_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0, use_fc=False)
        model = DFlashDraftModelRouted.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, router_config=nofc_config)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ckpt = Path(ckpt_path)
    if ckpt.is_symlink():
        ckpt = ckpt.resolve()
    sd = torch.load(ckpt / "model.pt", weights_only=True, map_location=device)
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    with open(ckpt / "metadata.json") as f:
        meta = json.load(f)
    logger.info(f"Loaded: step={meta['step']}, loss={meta['loss']:.4f}")

    # Evaluate each benchmark
    benchmark_results = {}
    for bench in benchmark_list:
        logger.info(f"\n{'='*60}\n  Benchmark: {bench}\n{'='*60}")
        prompts = load_eval_prompts(bench, n=n_prompts, seed=seed)
        benchmark_results[bench] = evaluate(model, target, tokenizer, prompts)

    # Aggregate across benchmarks
    total_tokens = sum(r["total_tokens"] for r in benchmark_results.values())
    total_time = sum(r["total_time_s"] for r in benchmark_results.values())
    all_acc = [a for r in benchmark_results.values() for a in r["all_acceptance_lengths"]]

    import numpy as np
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "checkpoint": ckpt_path,
        "training_step": meta["step"],
        "training_loss": meta["loss"],
        "seed": seed,
        "n_prompts_per_benchmark": n_prompts,
        "benchmarks": benchmark_list,
        "aggregate": {
            "tokens_per_second": round(total_tokens / total_time, 1),
            "mean_acceptance_length": round(float(np.mean(all_acc)), 2),
            "speedup_over_ar": round(np.mean([r["speedup_over_ar"] for r in benchmark_results.values()]), 2),
        },
        "results": benchmark_results,
    }

    # Save
    out_path = f"/checkpoints/eval_{variant}_full.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    checkpoint_volume.commit()
    logger.info(f"Saved to {out_path}")
    return output


@app.local_entrypoint()
def main(variant: str = "baseline", checkpoint: str = "", benchmarks: str = "math500,humaneval,mt_bench", seed: int = 42, n_prompts: int = 50):
    output = run_eval.remote(variant=variant, checkpoint=checkpoint, benchmarks=benchmarks, seed=seed, n_prompts=n_prompts)
    agg = output["aggregate"]
    print(f"\n{'=' * 60}")
    print(f"  {variant.upper()} — AGGREGATE RESULTS")
    print(f"  Benchmarks: {output['benchmarks']}")
    print(f"{'=' * 60}")
    print(f"  Tokens/sec:           {agg['tokens_per_second']}")
    print(f"  Acceptance length:    {agg['mean_acceptance_length']}")
    print(f"  Speedup over AR:      {agg['speedup_over_ar']}x")
    print(f"{'=' * 60}")
    for bench, res in output["results"].items():
        print(f"  {bench:<15} τ={res['mean_acceptance_length']:.2f}  {res['tokens_per_second']} tok/s  {res['speedup_over_ar']}x")
    print(f"{'=' * 60}")
