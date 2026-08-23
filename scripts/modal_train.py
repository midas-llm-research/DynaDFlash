"""
Modal cloud training script for DFlash-family draft models.

Supports router, router-nofc, DFlare, and base variants. Select via --variant.

Usage:
    modal run scripts/modal_train.py --variant router         # Train router variant
    modal run scripts/modal_train.py --variant dflare         # Train DFlare variant
    modal run scripts/modal_train.py --variant base           # Train base variant
    modal run scripts/modal_train.py --variant router --config configs/base_4b.yaml
    modal run scripts/modal_train.py --variant dflare --config configs/dflare_4b.yaml
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
    epochs: int = 0,
):
    import sys
    import shutil

    import torch
    from loguru import logger

    sys.path.insert(0, "/root")

    from midas.config import TrainConfig
    from midas.models import DFlashDraftModel, DFlashDraftModelLoRAHead, DFlashDraftModelRouted, DFlashDraftModelSplitQuery, DFlareDraftModel
    from midas.train import train as run_training

    config = TrainConfig.from_yaml(config_path)
    if epochs > 0:
        config.num_epochs = epochs

    # Override paths to use Modal volumes
    config.target_model = "/models/Qwen3-4B"
    config.draft_checkpoint = "/models/Qwen3-4B-DFlash-b16"

    if variant == "router":
        model_cls = DFlashDraftModelRouted
        config.checkpoint.save_dir = "/checkpoints/midas-router-v4"
        logger.info("Training ROUTED variant (MIDAS context router)")
    elif variant == "dflare":
        model_cls = DFlareDraftModel
        if config.dflare is None:
            from midas.config import DFlareConfig
            config.dflare = DFlareConfig()
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/dflare-qwen3-4b"
        logger.info("Training DFLARE variant (adaptive fusion + heterogeneous KV)")
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
        config.checkpoint.save_dir = "/checkpoints/midas-base-v4"
        logger.info("Training BASE variant (z-lab fc+hidden_norm)")
    elif variant == "lora-head":
        # Matched arm for `base`: same config file, same random init, same 3
        # epochs. The adapter is identity at step 0, so the two start as the
        # same function and differ only by the adapter's 327,680 parameters.
        model_cls = DFlashDraftModelLoRAHead
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-lora-head-v1"
        config.logging.wandb_project = "midas-lora-head"
        logger.info("Training LORA-HEAD variant (low-rank adapter before the LM head)")
    elif variant == "lora-head-continued":
        # 6-epoch arm. Takes the 3-epoch lora-head weights and runs the SAME
        # fresh 3-epoch schedule base-continued ran (new optimizer, new cosine),
        # so midas-lora-head-continued-v1 and midas-base-continued-v1 are
        # comparable at ~6 epochs of total training.
        model_cls = DFlashDraftModelLoRAHead
        config.router = None
        config.warm_start = "/checkpoints/midas-lora-head-v1/latest"
        config.checkpoint.save_dir = "/checkpoints/midas-lora-head-continued-v1"
        config.logging.wandb_project = "midas-lora-head-continued"
        logger.info("Training LORA-HEAD-CONTINUED (3-epoch warm start, 3 more epochs)")
    elif variant == "split-query":
        model_cls = DFlashDraftModelSplitQuery
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-split-query-v1"
        logger.info("Training SPLIT-QUERY variant (separate q_proj for target/draft KV)")
    else:
        raise ValueError(f"Unknown variant '{variant}'. Use 'router', 'router-nofc', 'base', 'lora-head', 'lora-head-continued', 'split-query', or 'dflare'.")

    # `fresh` wipes save_dir; a warm start living inside it would be deleted first.
    if config.warm_start and Path(config.warm_start).resolve().parent == Path(config.checkpoint.save_dir):
        raise ValueError(
            f"save_dir {config.checkpoint.save_dir} contains the warm-start checkpoint."
        )

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
    epochs: int = 0,
    spawn: bool = False,
):
    if download:
        download_models.remote()
        return
    if variant == "dflare" and config == "configs/base_4b.yaml":
        config = "configs/dflare_4b.yaml"

    # Map local config path to container path
    config_name = Path(config).name
    container_config = f"/root/configs/{config_name}"

    if spawn:
        # A .remote() call is cancelled when the local caller disconnects, which
        # silently kills a multi-hour run. spawn() hands off instead.
        call = train.spawn(config_path=container_config, variant=variant, fresh=fresh, epochs=epochs)
        print(f"Spawned {variant} training — call id: {call.object_id}")
        print(f"Follow with: modal app logs {app.app_id}")
        return
    train.remote(config_path=container_config, variant=variant, fresh=fresh, epochs=epochs)
