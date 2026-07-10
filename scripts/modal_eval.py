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
    "baseline": "/checkpoints/midas-baseline/latest",
    "router": "/checkpoints/midas-router-v2/latest",
    "router-nofc": "/checkpoints/midas-router-nofc-v3/latest",
    "split-query": "/checkpoints/midas-split-query-v1/latest",
    "dflare": "/checkpoints/dflare-qwen3-4b/latest",
    "token-mix": "/checkpoints/midas-token-mix-v1/latest",
    "causal-block": "/checkpoints/midas-causal-block-v1/latest",
    "dflare-deep": "/checkpoints/dflare-deep-qwen3-4b/latest",
    "dflare-plus": "/checkpoints/dflare-plus-qwen3-4b/latest",
}


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=43200,
)
def run_eval(
    variants: str = "baseline",
    checkpoint: str = "",
    benchmarks: str = "math500,humaneval,mt_bench",
    seed: int = 42,
    n_prompts: int = 50,
    measure_ar: bool = False,
):
    import json
    import sys
    from datetime import datetime, timezone

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from loguru import logger

    sys.path.insert(0, "/root")
    from midas.config import DFlareConfig, RouterConfig
    from midas.eval import evaluate, load_eval_prompts, measure_ar_baseline
    from midas.models.model import DFlashDraftModel
    from midas.models.model_context_router import DFlashDraftModelRouted
    from midas.models.model_split_query import DFlashDraftModelSplitQuery
    from midas.models.model_causal_block import DFlashDraftModelCausalBlock
    from midas.models.model_token_mix import DFlashDraftModelTokenMix
    from midas.models.model_dflare import DFlareDraftModel
    from midas.models.model_dflare_deep import DFlareDeepDraftModel
    from midas.models.model_dflare_plus import DFlarePlusDraftModel

    device = torch.device("cuda")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)
    variant_list = [v.strip() for v in variants.split(",")]
    benchmark_list = [b.strip() for b in benchmarks.split(",")]
    logger.info(f"Evaluating variants={variant_list}, benchmarks={benchmark_list}, seed={seed}")

    # Load target (shared across all variants)
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # Load prompts per benchmark (shared across all variants)
    bench_prompts = {b: load_eval_prompts(b, n=n_prompts, seed=seed) for b in benchmark_list}

    # Measure AR baseline once per benchmark (optional — adds ~1hr per benchmark)
    ar_baselines = {}
    if measure_ar:
        for bench, prompts in bench_prompts.items():
            logger.info(f"\n{'='*60}\n  AR Baseline: {bench}\n{'='*60}")
            ar_baselines[bench] = measure_ar_baseline(target, tokenizer, prompts)

    # Helper to load a draft model
    def load_draft(variant_name):
        ckpt_path = checkpoint or CHECKPOINTS[variant_name]
        if variant_name == "baseline":
            model = DFlashDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16)
        elif variant_name == "router":
            model = DFlashDraftModelRouted.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, router_config=router_config)
        elif variant_name == "router-nofc":
            nofc_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0, use_fc=False)
            model = DFlashDraftModelRouted.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, router_config=nofc_config)
        elif variant_name == "split-query":
            model = DFlashDraftModelSplitQuery.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16)
        elif variant_name == "dflare":
            dflare_config = DFlareConfig(hidden_size=2560, num_draft_layers=7, num_target_layers=9)
            model = DFlareDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, dflare_config=dflare_config)
        elif variant_name == "token-mix":
            model = DFlashDraftModelTokenMix.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16)
        elif variant_name == "causal-block":
            model = DFlashDraftModelCausalBlock.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16)
        elif variant_name == "dflare-deep":
            from midas.config import DFlarePlusConfig
            dflare_deep_config = DFlarePlusConfig(hidden_size=2560, num_draft_layers=9, num_target_layers=9, intermediate_size=4864)
            model = DFlareDeepDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, dflare_config=dflare_deep_config)
        elif variant_name == "dflare-plus":
            from midas.config import DFlarePlusConfig
            dflare_plus_config = DFlarePlusConfig(hidden_size=2560, num_draft_layers=9, num_target_layers=9, intermediate_size=4864)
            model = DFlarePlusDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, dflare_config=dflare_plus_config)
        else:
            raise ValueError(f"Unknown variant: {variant_name}")
        ckpt = Path(ckpt_path)
        if ckpt.is_symlink():
            ckpt = ckpt.resolve()
        sd = torch.load(ckpt / "model.pt", weights_only=True, map_location=device)
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        with open(ckpt / "metadata.json") as f:
            meta = json.load(f)
        logger.info(f"Loaded {variant_name}: step={meta['step']}, loss={meta['loss']:.4f}")
        return model, meta, ckpt_path

    # Evaluate each variant
    all_variant_results = {}
    for variant in variant_list:
        logger.info(f"\n{'='*60}\n  Variant: {variant}\n{'='*60}")
        model, meta, ckpt_path = load_draft(variant)

        benchmark_results = {}
        for bench, prompts in bench_prompts.items():
            logger.info(f"  Benchmark: {bench}")
            benchmark_results[bench] = evaluate(
                model, target, tokenizer, prompts, ar_baseline=ar_baselines.get(bench)
            )

        total_tokens = sum(r["total_tokens"] for r in benchmark_results.values())
        total_time = sum(r["total_time_s"] for r in benchmark_results.values())
        all_acc = [a for r in benchmark_results.values() for a in r["all_acceptance_lengths"]]

        speedups = [r["speedup_over_ar"] for r in benchmark_results.values() if r["speedup_over_ar"] is not None]
        all_variant_results[variant] = {
            "variant": variant,
            "checkpoint": ckpt_path,
            "training_step": meta["step"],
            "training_loss": meta["loss"],
            "aggregate": {
                "tokens_per_second": round(total_tokens / total_time, 1),
                "mean_acceptance_length": round(float(np.mean(all_acc)), 2),
                "speedup_over_ar": round(np.mean(speedups), 2) if speedups else None,
            },
            "results": benchmark_results,
        }

        # Free model memory before loading next
        del model
        torch.cuda.empty_cache()

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_prompts_per_benchmark": n_prompts,
        "benchmarks": benchmark_list,
        "variants": all_variant_results,
    }

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"/checkpoints/eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    checkpoint_volume.commit()
    logger.info(f"Saved to {out_path}")
    return output


@app.local_entrypoint()
def main(variants: str = "baseline,router,split-query,dflare", checkpoint: str = "", benchmarks: str = "math500,humaneval,mt_bench", seed: int = 42, n_prompts: int = 50, measure_ar: bool = False):
    output = run_eval.remote(variants=variants, checkpoint=checkpoint, benchmarks=benchmarks, seed=seed, n_prompts=n_prompts, measure_ar=measure_ar)
    print(f"\n{'=' * 70}")
    print(f"  EVAL RESULTS — Benchmarks: {output['benchmarks']}, Seed: {output['seed']}")
    print(f"{'=' * 70}")
    print(f"  {'Variant':<15} {'τ':<8} {'tok/s':<10} {'Speedup':<10}")
    print(f"  {'-'*50}")
    for name, data in output["variants"].items():
        agg = data["aggregate"]
        speedup_str = f"{agg['speedup_over_ar']}x" if agg['speedup_over_ar'] else "—"
        print(f"  {name:<15} {agg['mean_acceptance_length']:<8} {agg['tokens_per_second']:<10} {speedup_str:<10}")
    print(f"{'=' * 70}")
