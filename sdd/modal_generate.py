"""
Generate 800K SDD samples using 2x H100 in parallel.

Each GPU generates 400K samples from different Nemotron splits.
Estimated: ~2.5 hours wall-clock, ~$31 total.

Usage:
    modal run sdd/modal_generate.py
"""

import modal
from pathlib import Path

app = modal.App("midas-sdd-generate")

model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("midas-checkpoints", create_if_missing=True)

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
    .add_local_dir(str(PROJECT_ROOT / "sdd"), "/root/sdd", copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=28800,  # 8h max
)
def run_shard(
    shard_id: int,
    nemotron_splits: dict,
    codealpaca_count: int = 10_000,
):
    import sys
    sys.path.insert(0, "/root")
    from sdd.generate import generate_shard

    output_path = f"/checkpoints/sdd/shard_{shard_id}.jsonl"

    result = generate_shard(
        target_model_path="/models/Qwen3-4B",
        tokenizer_path="/models/Qwen3-4B",
        nemotron_splits=nemotron_splits,
        codealpaca_count=codealpaca_count,
        output_path=output_path,
        batch_size=64,
        max_new_tokens=2048,
        max_prompt_tokens=1024,
        temperature=0.6,
        commit_fn=checkpoint_volume.commit,
    )

    checkpoint_volume.commit()
    return result


@app.local_entrypoint()
def main():
    # Shard 0: math(195K) + code(175K, all available) + CodeAlpaca(10K) = 380K
    # Shard 1: stem(195K) + chat(215K) + CodeAlpaca(10K) = 420K
    # Total: 800K
    print("Spawning 2 SDD generation shards on H100s...")
    print("Shard 0: math(195K) + code(175K) + CodeAlpaca(10K) = 380K")
    print("Shard 1: stem(195K) + chat(215K) + CodeAlpaca(10K) = 420K")
    print("Config: batch=64, temp=0.6, max_new_tokens=2048")
    print()

    shard_0 = run_shard.spawn(0, {"math": 195_000, "code": 175_000}, 10_000)
    shard_1 = run_shard.spawn(1, {"stem": 195_000, "chat": 215_000}, 10_000)

    print(f"Shard 0 spawned: {shard_0.object_id}")
    print(f"Shard 1 spawned: {shard_1.object_id}")
    print()
    print("Monitor: https://modal.com/apps/midas-llm-research/main/")
    print("Output: /checkpoints/sdd/shard_0.jsonl + shard_1.jsonl")
    print("Estimated: ~2.5 hours, ~$31 total")
