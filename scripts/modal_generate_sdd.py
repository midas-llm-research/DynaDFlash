"""
Generate 100K SDD samples on 1x H100.
Target model (Qwen3-4B) generates responses at temperature=0.7
(Qwen3 non-thinking recipe: top_p=0.8, top_k=20, min_p=0).

Usage:
    modal run scripts/modal_generate_sdd.py
"""

import modal
from pathlib import Path

app = modal.App("midas-sdd-generate")

model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("midas-checkpoints", create_if_missing=True)

PROJECT_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm==0.10.0",
        "transformers>=4.57.0",
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
    timeout=28800,
)
def generate(fresh: bool = False):
    import sys
    sys.path.insert(0, "/root")
    from sdd.generate_sdd import generate_shard

    # 100K samples: 20K per Nemotron split + 20K CodeAlpaca
    result = generate_shard(
        target_model_path="/models/Qwen3-4B",
        nemotron_splits={"math": 20_000, "code": 20_000, "stem": 20_000, "chat": 20_000},
        codealpaca_count=20_000,
        output_path="/checkpoints/sdd/sdd_100k.jsonl",
        max_seq_len=3072,
        temperature=0.7,
        fresh=fresh,
        commit_fn=checkpoint_volume.commit,
    )

    checkpoint_volume.commit()
    return result


@app.local_entrypoint()
def main(fresh: bool = True):
    call = generate.spawn(fresh=fresh)
    print(f"SDD generation spawned: {call.object_id}")
    print("Config: 100K samples, temp=0.7, top_p=0.8, top_k=20, max_seq_len=3072")
    print("Splits: 20K each from math/code/stem/chat + 20K CodeAlpaca")
    print("Output: /checkpoints/sdd/sdd_100k.jsonl")
    print("Monitor: https://modal.com/apps/midas-llm-research/main/")
