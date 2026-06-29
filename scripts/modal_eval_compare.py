"""
Head-to-head evaluation: Baseline vs Router vs Router-NoFC.
All trained identically from scratch on same SDD data.
Reports acceptance length (τ), tokens/sec, speedup over autoregressive.

Uses MATH-500 (first N problems) matching DFlash paper eval.

Select which methods to run with boolean flags (default: all three):
    modal run scripts/modal_eval_compare.py                       # all methods
    modal run scripts/modal_eval_compare.py --baseline            # baseline only
    modal run scripts/modal_eval_compare.py --baseline --router   # baseline + router

Sample selection is deterministic (first N MATH-500 problems), so every method
is scored on the identical set regardless of which subset of methods you run.
"""

import modal
from pathlib import Path

app = modal.App("midas-eval-compare")

checkpoint_volume = modal.Volume.from_name("midas-checkpoints")
model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)

PROJECT_ROOT = Path(__file__).parent.parent

# Method registry: cli-name -> (display label, default checkpoint).
# The model class is resolved lazily inside `evaluate` so an unselected method
# whose model module is missing (e.g. router-no-fc) never breaks the run.
METHODS = {
    "baseline": "Baseline (FC)",
    "router": "Router",
    "router-no-fc": "Router-NoFC",
}

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
    methods: list[str],
    baseline_ckpt: str = "/checkpoints/midas-baseline/checkpoint_step9276",
    router_ckpt: str = "/checkpoints/midas-router-v2/checkpoint_step9276",
    router_nofc_ckpt: str = "/checkpoints/midas-router-nofc-v3/best",
    max_samples: int = 50,
    max_new_tokens: int = 2048,
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
    from midas.models.model import dflash_generate
    from midas.models.model_context_router import DFlashDraftModelRouted

    logger.info(f"Methods to evaluate: {methods}")
    device = torch.device("cuda")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)

    # === Load target model ===
    logger.info("Loading target: Qwen3-4B")
    target = (
        AutoModelForCausalLM.from_pretrained(
            "/models/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Trained-model loader ===
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

    # Per-method builders. router-no-fc imports its model lazily so a missing
    # module only errors when that method is actually selected.
    def build(name):
        if name == "baseline":
            return load_trained(DFlashDraftModel, baseline_ckpt)
        if name == "router":
            return load_trained(DFlashDraftModelRouted, router_ckpt, router_config=router_config)
        if name == "router-no-fc":
            from midas.models.model_context_router_nofc import DFlashDraftModelRoutedNoFC
            return load_trained(DFlashDraftModelRoutedNoFC, router_nofc_ckpt, router_config=router_config)
        raise ValueError(f"Unknown method '{name}'")

    logger.info("Loading selected models...")
    models = {}
    metas = {}
    for name in methods:
        logger.info(f"Loading {name}...")
        models[name], metas[name] = build(name)

    # === Load MATH-500 (first max_samples problems, deterministic) ===
    logger.info("Loading MATH-500")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [s["problem"] for s in list(math500)[:max_samples]]
    logger.info(f"Evaluating on {len(prompts)} MATH-500 problems (greedy, max_new_tokens={max_new_tokens})")

    # === Measure autoregressive baseline (target only, no drafting) ===
    logger.info("Measuring autoregressive baseline (no spec decode)...")
    ar_times, ar_tokens = [], []
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

    # === Warmup selected models ===
    logger.info("Warmup...")
    wids = tokenizer("Solve: 1+1", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        for name in methods:
            dflash_generate(models[name], target, wids, max_new_tokens=32,
                            stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)

    # === Evaluate each selected model ===
    def eval_model(model, name):
        logger.info(f"Evaluating {name}...")
        results = []
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                stats = dflash_generate(
                    model, target, input_ids,
                    max_new_tokens=max_new_tokens, stop_token_ids=[tokenizer.eos_token_id],
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
                logger.info(f"  {name}: {i+1}/{len(prompts)}")
        return results

    per_method_results = {name: eval_model(models[name], name) for name in methods}

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

    results = {"autoregressive_tps": round(ar_tps, 1)}
    for name in methods:
        results[name] = compute_metrics(per_method_results[name], metas[name])

    # === Print results (only selected methods) ===
    _print_table(results, methods, ar_tps, log=logger.info)

    with open("/checkpoints/eval_compare.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


def _print_table(results, methods, ar_tps, log=print):
    """Render the comparison table for whichever methods were run."""
    labels = [METHODS[m] for m in methods]
    col = 20
    log("")
    log("=" * (32 + col * len(methods)))
    log("  HEAD-TO-HEAD" + " vs ".join([""] + labels))
    log("  (All trained from scratch, same data, same steps)")
    log("=" * (32 + col * len(methods)))

    def row(title, fmt):
        cells = "".join(f"{fmt(results[m]):<{col}}" for m in methods)
        log(f"  {title:<30} {cells}")

    log(f"  {'Metric':<30} " + "".join(f"{lbl:<{col}}" for lbl in labels))
    log(f"  {'-' * (28 + col * len(methods))}")
    row("Training loss", lambda r: f"{r['training_loss']:.4f}")
    row("Acceptance length (τ)", lambda r: f"{r['mean_acceptance_length']:.2f}")
    row("Tokens/sec", lambda r: f"{r['tokens_per_second']:.1f}")
    row("Speedup over AR", lambda r: f"{r['speedup_over_ar']:.2f}x")
    log(f"  {'-' * (28 + col * len(methods))}")
    log(f"  {'Autoregressive baseline':<30} {ar_tps:.1f} tok/s")
    log("=" * (32 + col * len(methods)))


@app.local_entrypoint()
def main(
    baseline: bool = False,
    router: bool = False,
    router_no_fc: bool = False,
    max_samples: int = 50,
    max_new_tokens: int = 2048,
):
    # Build the selected method list; no flags => run all three.
    selected = []
    if baseline:
        selected.append("baseline")
    if router:
        selected.append("router")
    if router_no_fc:
        selected.append("router-no-fc")
    if not selected:
        selected = list(METHODS)

    results = evaluate.remote(
        methods=selected,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
    )

    _print_table(results, selected, results["autoregressive_tps"])
