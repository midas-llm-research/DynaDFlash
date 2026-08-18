"""
Modal training pipeline for the acceptance-aligned loss experiments.

Deliberately SEPARATE from scripts/modal_train.py so the established pipeline and
its checkpoints stay untouched. The model architecture is plain DFlash — only the
training objective differs (see midas/losses_aligned.py), so checkpoints produced
here load into DFlashDraftModel and evaluate through the existing eval pipeline
with no changes.

Usage:
    modal run --detach scripts/modal_train_aligned.py --variant margin
    modal run --detach scripts/modal_train_aligned.py --variant featreg
"""

import modal
from pathlib import Path

app = modal.App("midas-aligned-train")

model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("midas-checkpoints", create_if_missing=True)

PROJECT_ROOT = Path(__file__).parent.parent

# variant -> (config yaml, checkpoint dir, wandb project).
# Checkpoint dirs are set here and never read from the YAML: `fresh=True` rmtree's
# this path, so a mismatched --config must not be able to redirect the wipe at
# another experiment's checkpoints.
VARIANTS = {
    "margin": ("aligned_margin_4b.yaml", "/checkpoints/midas-aligned-margin-v1", "midas-aligned-margin"),
    "featreg": ("aligned_featreg_4b.yaml", "/checkpoints/midas-aligned-featreg-v1", "midas-aligned-featreg"),
}

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
    volumes={"/models": model_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=86400,
)
def train(
    variant: str = "margin",
    config_path: str = "",
    fresh: bool = True,
    epochs: int = 0,
):
    import sys
    import shutil

    from loguru import logger

    sys.path.insert(0, "/root")

    from midas.config import TrainConfig
    from midas.models import DFlashDraftModelAligned
    from midas.train import train as run_training

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Valid: {sorted(VARIANTS)}")
    default_yaml, save_dir, wandb_project = VARIANTS[variant]
    config_path = config_path or f"/root/configs/{default_yaml}"

    config = TrainConfig.from_yaml(config_path)
    if epochs > 0:
        config.num_epochs = epochs

    if config.aligned_loss is None:
        raise ValueError(
            f"{config_path} has no `aligned_loss:` block — that would train a plain "
            f"DFlash baseline under a new name. Use configs/{default_yaml}."
        )
    if config.aligned_loss.lambda_margin == 0.0 and config.aligned_loss.lambda_reg == 0.0:
        raise ValueError(
            "Both aligned-loss weights are 0, which is exactly the existing baseline "
            "objective. Refusing to spend a GPU run on a duplicate of midas-baseline."
        )

    config.target_model = "/models/Qwen3-4B"
    config.draft_checkpoint = "/models/Qwen3-4B-DFlash-b16"
    config.router = None
    config.checkpoint.save_dir = save_dir
    config.logging.wandb_project = wandb_project

    logger.info(
        f"Training ALIGNED-{variant.upper()} | {config.aligned_loss} | -> {save_dir}"
    )

    if fresh:
        ckpt_dir = Path(save_dir)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            logger.info(f"Cleared old checkpoints at {ckpt_dir}")

    run_training(config, model_cls=DFlashDraftModelAligned, random_init=True)
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(
    variant: str = "margin",
    config: str = "",
    fresh: bool = True,
    epochs: int = 0,
):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Valid: {sorted(VARIANTS)}")
    container_config = f"/root/configs/{Path(config).name}" if config else ""

    # spawn(), not remote(): under `modal run --detach` a .remote() call is cancelled
    # the moment the local caller disconnects, silently killing long training runs.
    call = train.spawn(
        variant=variant, config_path=container_config, fresh=fresh, epochs=epochs
    )
    print(f"Spawned aligned-{variant} training — call id: {call.object_id}")
    print(f"Follow with: modal app logs {app.app_id}")
