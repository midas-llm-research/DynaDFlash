"""
3-way evaluation: z-lab DFlash vs Our Baseline vs Our Router.
Same prompts, same target model.

Usage:
    modal run scripts/modal_eval_all.py
"""

import modal
from pathlib import Path

app = modal.App("midas-eval-all")

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
    from midas.config import RouterConfig
    from midas.model import DFlashDraftModelRouted
    from midas.model import dflash_generate as router_generate

    device = torch.device("cuda")

    # === Load target ===
    logger.info("Loading target: Qwen3-4B")
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Load z-lab published ===
    logger.info("Loading z-lab DFlash")
    zlab_draft = AutoModel.from_pretrained(
        "/models/Qwen3-4B-DFlash-b16", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()

    # === Load our baseline ===
    logger.info("Loading our baseline DFlash")
    baseline_draft = DFlashDraftModel.from_config_random("/models/Qwen3-4B-DFlash-b16", device=device, dtype=torch.bfloat16)
    ckpt = Path("/checkpoints/midas-baseline/best")
    if ckpt.is_symlink():
        ckpt = ckpt.resolve()
    sd = torch.load(ckpt / "model.pt", weights_only=True, map_location=device)
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    baseline_draft.load_state_dict(sd)
    baseline_draft.eval()

    # === Load our router ===
    logger.info("Loading our router DFlash")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)
    router_draft = DFlashDraftModelRouted.from_pretrained_dflash(
        "/models/Qwen3-4B-DFlash-b16", router_config=router_config, device=device, dtype=torch.bfloat16,
    )
    rckpt = Path("/checkpoints/midas-router/best")
    if rckpt.is_symlink():
        rckpt = rckpt.resolve()
    rsd = torch.load(rckpt / "model.pt", weights_only=True, map_location=device)
    rsd = {k: v.to(torch.bfloat16) for k, v in rsd.items()}
    router_draft.load_state_dict(rsd)
    router_draft.eval()

    # === Prompts ===
    logger.info("Loading eval prompts...")
    prompts = []
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    for s in list(gsm8k)[:25]:
        prompts.append(("gsm8k", s["question"]))
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    for s in list(humaneval)[:25]:
        prompts.append(("humaneval", s["prompt"]))
    logger.info(f"{len(prompts)} prompts")

    # === Warmup ===
    logger.info("Warmup...")
    wids = tokenizer("Hello", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        zlab_draft.spec_generate(target=target, input_ids=wids, max_new_tokens=16, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)
        baseline_generate(baseline_draft, target, wids, max_new_tokens=16, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)
        router_generate(router_draft, target, wids, max_new_tokens=16, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)

    # === Eval function ===
    def eval_model(name, generate_fn, prompts):
        results = []
        for domain, prompt_text in prompts:
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                stats = generate_fn(input_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            results.append(stats | {"time": t1 - t0, "domain": domain})
        return results

    # === Run evals ===
    logger.info("Evaluating z-lab...")
    zlab_res = []
    for domain, prompt_text in prompts:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = zlab_draft.spec_generate(target=target, input_ids=input_ids, max_new_tokens=256, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0)
        torch.cuda.synchronize()
        zlab_res.append({"tokens": out.shape[1] - input_ids.shape[1], "time": time.perf_counter() - t0})

    logger.info("Evaluating baseline...")
    base_res = []
    for domain, prompt_text in prompts:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = baseline_generate(baseline_draft, target, input_ids, max_new_tokens=256, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0, return_stats=True)
        torch.cuda.synchronize()
        base_res.append({"tokens": stats.num_output_tokens, "time": time.perf_counter() - t0, "acceptance": stats.acceptance_lengths})

    logger.info("Evaluating router...")
    router_res = []
    for domain, prompt_text in prompts:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            stats = router_generate(router_draft, target, input_ids, max_new_tokens=256, stop_token_ids=[tokenizer.eos_token_id], temperature=0.0, return_stats=True)
        torch.cuda.synchronize()
        router_res.append({"tokens": stats.num_output_tokens, "time": time.perf_counter() - t0, "acceptance": stats.acceptance_lengths})

    # === Results ===
    zlab_tps = sum(r["tokens"] for r in zlab_res) / sum(r["time"] for r in zlab_res)
    base_tps = sum(r["tokens"] for r in base_res) / sum(r["time"] for r in base_res)
    router_tps = sum(r["tokens"] for r in router_res) / sum(r["time"] for r in router_res)

    base_acc = np.mean([a for r in base_res for a in r["acceptance"]])
    router_acc = np.mean([a for r in router_res for a in r["acceptance"]])

    results = {
        "zlab": {"tps": round(zlab_tps, 1)},
        "baseline": {"tps": round(base_tps, 1), "acceptance": round(float(base_acc), 2)},
        "router": {"tps": round(router_tps, 1), "acceptance": round(float(router_acc), 2)},
    }

    logger.info("=" * 60)
    logger.info("3-WAY COMPARISON")
    logger.info("=" * 60)
    logger.info(f"z-lab DFlash (published): {zlab_tps:.1f} tok/s")
    logger.info(f"Our Baseline (SDD):       {base_tps:.1f} tok/s, acceptance={base_acc:.2f}")
    logger.info(f"Our Router:               {router_tps:.1f} tok/s, acceptance={router_acc:.2f}")
    logger.info("=" * 60)

    with open("/checkpoints/midas-baseline/eval_3way.json", "w") as f:
        json.dump(results, f, indent=2)
    checkpoint_volume.commit()
    return results


@app.local_entrypoint()
def main():
    results = evaluate.remote()
    print("\n" + "=" * 60)
    print("  3-WAY COMPARISON")
    print("=" * 60)
    print(f"  z-lab DFlash (published): {results['zlab']['tps']} tok/s")
    print(f"  Our Baseline (SDD):       {results['baseline']['tps']} tok/s, acceptance={results['baseline']['acceptance']}")
    print(f"  Our Router:               {results['router']['tps']} tok/s, acceptance={results['router']['acceptance']}")
    print("=" * 60)
