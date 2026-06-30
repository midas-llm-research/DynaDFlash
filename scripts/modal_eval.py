"""
Evaluate a single DFlash variant on MATH-500 (50 problems).
Each invocation gets its own fresh GPU — run separately per variant
to eliminate warmup/caching bias between models.

Usage:
    modal run scripts/modal_eval.py --variant baseline
    modal run scripts/modal_eval.py --variant router
    modal run scripts/modal_eval.py --variant router-nofc
    modal run scripts/modal_eval.py --variant dflare
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
    "dflare": "/checkpoints/dflare-qwen3-4b/best",
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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from loguru import logger

    sys.path.insert(0, "/root")
    from midas.config import DFlareConfig, RouterConfig
    from midas.eval import evaluate
    from midas.models.model import DFlashDraftModel
    from midas.models.model_context_router import DFlashDraftModelRouted
    from midas.models.model_dflare import DFlareDraftModel

    device = torch.device("cuda")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)
    dflare_config = DFlareConfig(hidden_size=2560, num_draft_layers=7, num_target_layers=9)
    ckpt_path = CHECKPOINTS[variant]
    logger.info(f"Evaluating variant={variant}, checkpoint={ckpt_path}")

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
    elif variant == "dflare":
        model = DFlareDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16, dflare_config=dflare_config)
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

    # Load prompts
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = [s["problem"] for s in list(math500)[:50]]

    # Evaluate
    results = evaluate(model, target, tokenizer, prompts)
    results.update({"variant": variant, "checkpoint": ckpt_path, "training_step": meta["step"], "training_loss": meta["loss"]})

    # Save
    with open(f"/checkpoints/eval_{variant}.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


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
