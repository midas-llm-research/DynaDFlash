"""
Throughput eval on Modal: compare a baseline DFlash draft against the routed
variant on a chosen benchmark. Both runs go through `midas.eval.evaluate`, so
the loop, prompting, and stats logic stays in one place.

Usage:
    modal run scripts/modal_eval.py
    modal run scripts/modal_eval.py --dataset math500 --max-samples 100
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
def evaluate_models(
    routed_ckpt: str = "/checkpoints/midas-router/best/model.pt",
    baseline_ckpt: str = "/checkpoints/midas-baseline/best/model.pt",
    dataset: str = "gsm8k",
    max_samples: int = 50,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
):
    import sys

    import torch
    from loguru import logger
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root")
    from midas.config import RouterConfig
    from midas.eval import evaluate, report
    from midas.models import DFlashDraftModel, DFlashDraftModelRouted

    device = torch.device("cuda")

    logger.info("Loading target: Qwen3-4B")
    target = (
        AutoModelForCausalLM.from_pretrained(
            "/models/Qwen3-4B",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # ---- Baseline DFlash (z-lab pretrained, optionally overlaid with our trained baseline) ----
    logger.info("Loading baseline DFlash (z-lab pretrained)")
    baseline = (
        DFlashDraftModel.from_pretrained(
            "/models/Qwen3-4B-DFlash-b16",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
    )
    baseline_ckpt_path = Path(baseline_ckpt)
    if baseline_ckpt_path.is_symlink():
        baseline_ckpt_path = baseline_ckpt_path.resolve()
    if baseline_ckpt_path.exists():
        logger.info(f"Overlaying trained baseline weights: {baseline_ckpt_path}")
        state = torch.load(baseline_ckpt_path, weights_only=True, map_location=device)
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}
        baseline.load_state_dict(state)
    baseline.eval()

    logger.info("=" * 50)
    logger.info("Evaluating BASELINE")
    logger.info("=" * 50)
    responses_b, bs = evaluate(
        target, baseline, tokenizer,
        dataset_name=dataset,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_samples=max_samples,
    )
    baseline_summary = report(responses_b, bs)
    del baseline
    torch.cuda.empty_cache()

    # ---- Routed DFlash ----
    logger.info("Loading routed DFlash")
    router_config = RouterConfig(
        hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0
    )
    routed = DFlashDraftModelRouted.from_pretrained_dflash(
        "/models/Qwen3-4B-DFlash-b16",
        router_config=router_config,
        device=device,
        dtype=torch.bfloat16,
    )
    routed_ckpt_path = Path(routed_ckpt)
    if routed_ckpt_path.is_symlink():
        routed_ckpt_path = routed_ckpt_path.resolve()
    if routed_ckpt_path.exists():
        logger.info(f"Loading trained router weights: {routed_ckpt_path}")
        state = torch.load(routed_ckpt_path, weights_only=True, map_location=device)
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}
        routed.load_state_dict(state)
    routed.eval()

    logger.info("=" * 50)
    logger.info("Evaluating ROUTED")
    logger.info("=" * 50)
    responses_r, bs = evaluate(
        target, routed, tokenizer,
        dataset_name=dataset,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_samples=max_samples,
    )
    routed_summary = report(responses_r, bs)

    # ---- Side-by-side ----
    print()
    print("=" * 60)
    print("  BASELINE vs ROUTED")
    print("=" * 60)
    print(f"  Baseline throughput: {baseline_summary['dflash_tps']:,.2f} tok/s")
    print(f"  Routed   throughput: {routed_summary['dflash_tps']:,.2f} tok/s")
    print(f"  Baseline acceptance: {baseline_summary['mean_acceptance_length']:.2f}")
    print(f"  Routed   acceptance: {routed_summary['mean_acceptance_length']:.2f}")
    if baseline_summary["dflash_tps"] > 0:
        ratio = routed_summary["dflash_tps"] / baseline_summary["dflash_tps"]
        print(f"  Routed/baseline TPS ratio: {ratio:.3f}x  (>1 = routed faster)")
    print("=" * 60)

    return {"baseline": baseline_summary, "routed": routed_summary}


@app.local_entrypoint()
def main(
    dataset: str = "gsm8k",
    max_samples: int = 50,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
):
    evaluate_models.remote(
        dataset=dataset,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    print("\nEval complete. See logs above for the side-by-side summary.")
