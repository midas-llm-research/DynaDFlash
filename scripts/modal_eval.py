"""
Evaluation script: compare routed DFlash vs original DFlash checkpoint.

Usage:
    modal run scripts/modal_eval.py
"""

import modal

app = modal.App("midas-eval")

checkpoint_volume = modal.Volume.from_name("midas-checkpoints")
model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)

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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def evaluate(checkpoint_path: str = "/checkpoints/midas-router/best"):
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    sys.path.insert(0, "/root")
    from midas.config import RouterConfig
    from midas.model import DFlashDraftModelRouted, dflash_generate

    device = torch.device("cuda")

    # Load target model
    target = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    # Load our routed draft model
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128)
    draft = DFlashDraftModelRouted.from_pretrained_dflash(
        "z-lab/Qwen3-4B-DFlash-b16",
        router_config=router_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Load trained router weights
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_symlink():
        ckpt_path = ckpt_path.resolve()
    state_dict = torch.load(ckpt_path / "model.pt", weights_only=True)
    draft.load_state_dict(state_dict)
    draft.eval()

    # Evaluation datasets
    eval_sets = {
        "gsm8k": load_dataset("openai/gsm8k", "main", split="test"),
        "humaneval": load_dataset("openai/openai_humaneval", split="test"),
    }

    results = {}
    for name, dataset in eval_sets.items():
        acceptance_lengths = []
        total_tokens = 0
        total_time = 0

        samples = list(dataset)[:100]  # Eval on first 100 samples

        for sample in samples:
            if name == "gsm8k":
                prompt = sample["question"]
            elif name == "humaneval":
                prompt = sample["prompt"]
            else:
                prompt = sample.get("text", sample.get("prompt", ""))

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            stats = dflash_generate(
                draft,
                target,
                input_ids,
                max_new_tokens=256,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0,
                return_stats=True,
            )

            acceptance_lengths.extend(stats.acceptance_lengths)
            total_tokens += stats.num_output_tokens
            total_time += stats.time_per_output_token * stats.num_output_tokens

        results[name] = {
            "mean_acceptance_length": float(np.mean(acceptance_lengths)),
            "median_acceptance_length": float(np.median(acceptance_lengths)),
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "num_samples": len(samples),
        }
        print(f"{name}: acceptance={results[name]['mean_acceptance_length']:.2f}, "
              f"tps={results[name]['tokens_per_second']:.1f}")

    # Save results
    output_path = Path("/checkpoints/midas-router/eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    checkpoint_volume.commit()
    return results


@app.local_entrypoint()
def main(checkpoint: str = "/checkpoints/midas-router/best"):
    results = evaluate.remote(checkpoint_path=checkpoint)
    print("\n=== Final Results ===")
    for name, metrics in results.items():
        print(f"{name}: acceptance={metrics['mean_acceptance_length']:.2f}")
