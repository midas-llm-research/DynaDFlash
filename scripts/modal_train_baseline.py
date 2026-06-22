"""
Train baseline DFlash (z-lab architecture, random init) using midas.train.

The only thing this script does beyond the routed run is construct a random-init
`DFlashDraftModel` from z-lab's config (instead of loading the routed variant).
The training loop, loss, checkpointing, and logging all come from
`midas.train.train`.

Usage:
    modal run scripts/modal_train_baseline.py
"""

import modal
from pathlib import Path

app = modal.App("midas-baseline-train")

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
        "wandb",
        "pyyaml",
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
    timeout=86400,
)
def train(config_path: str = "/root/configs/base_4b.yaml", fresh: bool = True):
    import shutil
    import sys

    import torch
    from loguru import logger
    from transformers import AutoConfig

    sys.path.insert(0, "/root")
    from midas.config import TrainConfig
    from midas.models import DFlashDraftModel
    from midas.train import train as run_training

    device = torch.device("cuda")

    config = TrainConfig.from_yaml(config_path)
    config.target_model = "/models/Qwen3-4B"
    config.draft_checkpoint = "/models/Qwen3-4B-DFlash-b16"
    config.checkpoint.save_dir = "/checkpoints/midas-baseline"

    if fresh:
        ckpt_dir = Path(config.checkpoint.save_dir)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info(f"Cleared {ckpt_dir}")

    # Random-init baseline DFlash using z-lab's architecture config.
    logger.info("Building random-init baseline DFlash from z-lab config")
    arch_config = AutoConfig.from_pretrained(
        config.draft_checkpoint, trust_remote_code=True
    )
    arch_config._attn_implementation = "sdpa"
    draft = DFlashDraftModel(arch_config).to(device=device, dtype=torch.bfloat16)

    run_training(config, model=draft)
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(config: str = "configs/base_4b.yaml", fresh: bool = True):
    call = train.spawn(config_path="/root/configs/base_4b.yaml", fresh=fresh)
    print(f"Baseline training spawned: {call.object_id}")
    print(f"Config: {config}, fresh={fresh}")
    print("Monitor: https://modal.com/apps/midas-llm-research/main/")
