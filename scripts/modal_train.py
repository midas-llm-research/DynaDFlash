"""
Modal cloud training script for DFlash context router.

Usage:
    modal run scripts/modal_train.py
    modal run scripts/modal_train.py --config configs/base_4b.yaml
"""

import modal

app = modal.App("midas-dflash-router")

# Persistent volumes for model weights and checkpoints
model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("midas-checkpoints", create_if_missing=True)

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
        "hf_transfer",
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
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=86400,
)
def train(config_path: str = "configs/base_4b.yaml"):
    import sys
    sys.path.insert(0, "/root")

    from midas.config import TrainConfig
    from midas.train import train as run_training

    config = TrainConfig.from_yaml(config_path)
    config.checkpoint.save_dir = "/checkpoints/midas-router"
    run_training(config)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_models():
    """Pre-download target and draft models to the shared volume."""
    from huggingface_hub import snapshot_download

    # Target model
    snapshot_download(
        "Qwen/Qwen3-4B",
        local_dir="/models/Qwen3-4B",
    )
    print("Downloaded Qwen3-4B")

    # Pre-trained DFlash draft (for weight initialization)
    snapshot_download(
        "z-lab/Qwen3-4B-DFlash-b16",
        local_dir="/models/Qwen3-4B-DFlash-b16",
    )
    print("Downloaded Qwen3-4B-DFlash-b16")

    model_volume.commit()


@app.local_entrypoint()
def main(config: str = "configs/base_4b.yaml", download: bool = False):
    if download:
        download_models.remote()
    else:
        train.remote(config_path=config)
