"""
Modal cloud training script for DFlash-family draft models.

Supports router, router-nofc, DFlare, DFlash 2, correction, and base variants.
Select via --variant.

Always pass `--detach`, before the script path:

    modal run --detach scripts/modal_train.py --variant router

The entrypoint spawns the job and exits, so the run is not tied to this terminal.
Without `--detach` the ephemeral app is torn down as soon as the entrypoint
returns, which cancels the spawned job within seconds. The two go together.

Usage:
    modal run --detach scripts/modal_train.py --variant router      # Train router variant
    modal run --detach scripts/modal_train.py --variant dflare      # Train DFlare variant
    modal run --detach scripts/modal_train.py --variant base        # Train base variant
    modal run --detach scripts/modal_train.py --variant router --config configs/base_4b.yaml
    modal run --detach scripts/modal_train.py --variant correction  # Train correction variant
    modal run --detach scripts/modal_train.py --variant correction --no-fresh      # RESUME
    modal run --detach scripts/modal_train.py --variant correction --no-fresh --max-steps 520
    modal run scripts/modal_train.py --download                     # (one-time) download models

`--fresh` defaults to True and DELETES the variant's checkpoint directory before
starting. Pass `--no-fresh` to resume from the last checkpoint. (It is a click
boolean flag -- `--fresh false` is a parse error, not a way to disable it.)

`--max-steps` caps the ABSOLUTE global_step (midas/train.py:516), and global_step
is restored on resume -- it is not "run N more steps". Resuming a step-500
checkpoint with `--max-steps 20` stops after a single step, because 500 >= 20,
and still writes a checkpoint on the way out. For 20 more steps from step 500,
pass `--max-steps 520`.
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
    warm_start: str = "",
    max_steps: int = 0,
):
    import sys
    import shutil

    import torch
    from loguru import logger

    sys.path.insert(0, "/root")

    from midas.config import TrainConfig
    from midas.models import CorrectionDraftModel, DFlash2DraftModel, DFlashDraftModel, DFlashDraftModelRouted, DFlashDraftModelSplitQuery, DFlareDraftModel
    from midas.train import train as run_training

    config = TrainConfig.from_yaml(config_path)
    if epochs > 0:
        config.num_epochs = epochs
    if max_steps > 0:
        config.max_steps = max_steps
        logger.info(f"Capped at {max_steps} optimizer steps")

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
    elif variant == "dflash2":
        model_cls = DFlash2DraftModel
        if config.dflash2 is None:
            from midas.config import DFlash2Config
            config.dflash2 = DFlash2Config()
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-dflash2-v1"
        config.logging.wandb_project = "midas-dflash2"
        # Warm start from our own trained DFlash. The convolutions are
        # identity-initialized, so step 0 IS the baseline; and the selector's
        # K-way loss only fires where the true token is already in the drafter's
        # top-K, which from random init is ~k/vocab -- it would get no gradient.
        config.warm_start = "/checkpoints/midas-baseline/latest"
        logger.info("Training DFLASH2 variant (path selector + block-local dynamic conv)")
    elif variant == "correction":
        model_cls = CorrectionDraftModel
        if config.correction is None:
            from midas.config import CorrectionConfig
            config.correction = CorrectionConfig()
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-correction-v1"
        config.logging.wandb_project = "midas-correction"
        # Round 1 must be the TRAINED DFlash 2 — the recycled residuals are its
        # output, so a base-checkpoint round 1 trains this model on residuals
        # deployment will never produce.
        if not config.correction.parent_checkpoint:
            config.correction.parent_checkpoint = "/checkpoints/midas-dflash2-v1/best"
        logger.info("Training CORRECTION variant (depth-1 refiner over recycled residuals)")
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
    elif variant == "split-query":
        model_cls = DFlashDraftModelSplitQuery
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-split-query-v1"
        logger.info("Training SPLIT-QUERY variant (separate q_proj for target/draft KV)")
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Use 'router', 'router-nofc', 'base', "
            f"'split-query', 'dflash2', 'dflare', or 'correction'."
        )

    if warm_start:
        config.warm_start = None if warm_start.lower() == "none" else warm_start
    if config.warm_start:
        logger.info(f"Warm starting from {config.warm_start}")

    if fresh:
        ckpt_dir = Path(config.checkpoint.save_dir)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info(f"Cleared old checkpoints at {ckpt_dir}")

    # Commit after EVERY checkpoint, not just at the end: without this the
    # checkpoints a long run writes are only in this container's view of the
    # volume, so a crash, preemption or timeout throws away all of them.
    run_training(
        config,
        model_cls=model_cls,
        random_init=True,
        on_checkpoint=checkpoint_volume.commit,
    )
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
    warm_start: str = "",
    max_steps: int = 0,
):
    if download:
        # spawn().get() rather than remote(): same blocking behaviour, but the
        # download is an ASYNC invocation so it is not cancelled if this
        # terminal drops mid-download (see the note on train.spawn below).
        download_models.spawn().get()
        return
    defaults = {
        "dflare": "configs/dflare_4b.yaml",
        "dflash2": "configs/dflash2_4b.yaml",
        "correction": "configs/correction_4b.yaml",
    }
    if config == "configs/base_4b.yaml" and variant in defaults:
        config = defaults[variant]

    # Map local config path to container path
    config_name = Path(config).name
    container_config = f"/root/configs/{config_name}"

    # spawn(), not remote(): .remote() is a SYNC invocation bound to this local
    # client, so the server cancels it ~2.5 min after the client disconnects --
    # even under `modal run --detach`, which only keeps the *app* alive. A 24h
    # training run dies when the laptop sleeps. spawn() is an ASYNC invocation:
    # it hands off and returns immediately, and nothing is tied to this process.
    call = train.spawn(
        config_path=container_config, variant=variant, fresh=fresh,
        epochs=epochs, warm_start=warm_start, max_steps=max_steps,
    )
    print(f"Spawned {variant} training - call id: {call.object_id}")
    print(f"Follow with: modal app logs {app.app_id}")
