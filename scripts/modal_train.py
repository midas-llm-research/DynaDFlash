"""
Modal cloud training script for DFlash context router.

Usage:
    modal run scripts/generate_data.py                # Step 1: generate target responses
    modal run scripts/modal_train.py                  # Step 2: train router
    modal run scripts/modal_train.py --download       # (one-time) download models
"""

import modal
from pathlib import Path

app = modal.App("midas-dflash-router")

# Persistent volumes
model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("midas-checkpoints", create_if_missing=True)

# Project root for mounting local code
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
    gpu="A100-40GB",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    timeout=86400,
)
def train(config_path: str = "/root/configs/base_4b.yaml", fresh: bool = False):
    import sys
    import shutil
    from pathlib import Path
    sys.path.insert(0, "/root")

    from midas.config import TrainConfig
    from midas.train import train as run_training

    config = TrainConfig.from_yaml(config_path)

    # Use volume paths
    config.target_model = "/models/Qwen3-4B"
    config.draft_checkpoint = "/models/Qwen3-4B-DFlash-b16"
    config.checkpoint.save_dir = "/checkpoints/midas-router"

    # Clear old checkpoints for a fresh start
    if fresh:
        ckpt_dir = Path(config.checkpoint.save_dir)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f"Cleared old checkpoints at {ckpt_dir}")

    run_training(config)

    checkpoint_volume.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def download_models():
    """Pre-download target and draft models to the shared volume."""
    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen3-4B", local_dir="/models/Qwen3-4B")
    print("Downloaded Qwen3-4B")

    snapshot_download("z-lab/Qwen3-4B-DFlash-b16", local_dir="/models/Qwen3-4B-DFlash-b16")
    print("Downloaded Qwen3-4B-DFlash-b16")

    model_volume.commit()


@app.local_entrypoint()
def main(config: str = "configs/base_4b.yaml", download: bool = False):
    if download:
        download_models.remote()
    else:
        call = train.spawn(config_path="/root/configs/base_4b.yaml", fresh=True)
        # NOTE: set fresh=False for future runs to resume from checkpoint
        print(f"Training spawned. Function call ID: {call.object_id}")
        print("Monitor at: https://modal.com/apps/midas-llm-research/main/")
        print("Checkpoints saved to midas-checkpoints volume.")
