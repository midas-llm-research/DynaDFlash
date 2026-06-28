"""
Modal cloud training script for DFlash draft models.

Supports both the routed variant (MIDAS context router) and the base variant
(z-lab's original fc+hidden_norm fusion). Select via --variant flag.

Usage:
    modal run scripts/modal_train.py --variant router         # Train router variant
    modal run scripts/modal_train.py --variant base           # Train base variant
    modal run scripts/modal_train.py --variant router --config configs/base_4b.yaml
    modal run scripts/modal_train.py --download               # (one-time) download models
"""

import modal
from pathlib import Path

app = modal.App("midas-dflash-train")

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
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    timeout=86400,
)
def train(
    config_path: str = "/root/configs/base_4b.yaml",
    variant: str = "router",
    fresh: bool = True,
):
    import sys
    import shutil

    import torch
    from loguru import logger

    sys.path.insert(0, "/root")

    from midas.config import TrainConfig
    from midas.models import DFlashDraftModel, DFlashDraftModelRouted
    from midas.train import train as run_training

    config = TrainConfig.from_yaml(config_path)

    # Override paths to use Modal volumes
    config.target_model = "/models/Qwen3-4B"
    config.draft_checkpoint = "/models/Qwen3-4B-DFlash-b16"

    if variant == "router":
        model_cls = DFlashDraftModelRouted
        config.checkpoint.save_dir = "/checkpoints/midas-router-v3"
        logger.info("Training ROUTED variant (MIDAS context router)")
    elif variant == "router-nofc":
        model_cls = DFlashDraftModelRouted
        if config.router is None:
            from midas.config import RouterConfig
            config.router = RouterConfig()
        config.router.use_fc = False
        config.checkpoint.save_dir = "/checkpoints/midas-router-nofc-v3"
        logger.info("Training ROUTED-NOFC variant (router without FC mixing)")
    elif variant == "base":
        model_cls = DFlashDraftModel
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-base-v3"
        logger.info("Training BASE variant (z-lab fc+hidden_norm)")
    else:
        raise ValueError(f"Unknown variant '{variant}'. Use 'router', 'router-nofc', or 'base'.")

    if fresh:
        ckpt_dir = Path(config.checkpoint.save_dir)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info(f"Cleared old checkpoints at {ckpt_dir}")

    run_training(config, model_cls=model_cls, random_init=True)
    checkpoint_volume.commit()


@app.function(
    image=image,
    gpu="H100",
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
def main(
    variant: str = "router",
    config: str = "configs/base_4b.yaml",
    download: bool = False,
    fresh: bool = True,
):
    if download:
        download_models.remote()
        return

    # Map local config path to container path
    config_name = Path(config).name
    container_config = f"/root/configs/{config_name}"

    train.remote(config_path=container_config, variant=variant, fresh=fresh)
