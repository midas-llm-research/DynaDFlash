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

# Single source of truth for which YAML each variant needs. A variant whose
# architecture is described by a method block (dflare / dflare_plus) MUST get
# the matching YAML — training against the wrong one silently builds a
# different architecture, and the mismatch only surfaces as a load_state_dict
# shape error at eval time, after the GPU spend.
VARIANT_CONFIGS = {
    "base": "base_4b.yaml",
    "router": "base_4b.yaml",
    "router-nofc": "base_4b.yaml",
    "split-query": "base_4b.yaml",
    "token-mix": "base_4b.yaml",
    "token-mix-continued": "base_4b.yaml",
    "causal-block": "base_4b.yaml",
    "dflare": "dflare_4b.yaml",
    "dflare-deep": "dflare_deep_4b.yaml",
    "dflare-plus": "dflare_plus_4b.yaml",
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


def _require_dflare_plus(config, variant: str, config_path: str) -> None:
    """Fail before the GPU spend if the YAML lacks the arch block this variant needs.

    Silently falling back to `DFlarePlusConfig()` gives intermediate_size=None
    (MLP stays 9728), which trains a different architecture than eval expects.
    """
    if config.dflare_plus is None:
        raise ValueError(
            f"Variant '{variant}' requires a `dflare_plus:` block, but "
            f"{config_path} has none. Use configs/{VARIANT_CONFIGS[variant]} "
            f"(or drop the explicit --config so the default is picked)."
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
    from midas.models import DFlashDraftModel, DFlashDraftModelCausalBlock, DFlashDraftModelRouted, DFlashDraftModelSplitQuery, DFlashDraftModelTokenMix, DFlareDeepDraftModel, DFlareDraftModel, DFlarePlusDraftModel
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
            raise ValueError(
                f"Variant 'dflare' requires a `dflare:` block, but {config_path} "
                f"has none. Use configs/{VARIANT_CONFIGS['dflare']}."
            )
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
    elif variant == "split-query":
        model_cls = DFlashDraftModelSplitQuery
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-split-query-v1"
        logger.info("Training SPLIT-QUERY variant (separate q_proj for target/draft KV)")
    elif variant == "token-mix-continued":
        # 6-epoch arm, mirroring exactly how midas-base-continued-v1 was made: take the
        # 3-epoch weights and run the SAME fresh 3-epoch schedule (new optimizer, new
        # cosine), so this and the 6-epoch baseline are comparable at ~6 epochs total.
        model_cls = DFlashDraftModelTokenMix
        config.router = None
        config.warm_start = "/checkpoints/midas-token-mix-v1/latest"
        config.checkpoint.save_dir = "/checkpoints/midas-token-mix-continued-v1"
        config.logging.wandb_project = "midas-token-mix-continued"
        logger.info("Training TOKEN-MIX-CONTINUED (3-epoch warm start, 3 more epochs)")
    elif variant == "token-mix":
        model_cls = DFlashDraftModelTokenMix
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-token-mix-v1"
        config.logging.wandb_project = "midas-token-mix"
        logger.info("Training TOKEN-MIX variant (per-layer position mixing)")
    elif variant == "causal-block":
        model_cls = DFlashDraftModelCausalBlock
        config.router = None
        config.checkpoint.save_dir = "/checkpoints/midas-causal-block-v1"
        config.logging.wandb_project = "midas-causal-block"
        logger.info("Training CAUSAL-BLOCK variant (causal attention within blocks)")
    elif variant == "dflare-deep":
        model_cls = DFlareDeepDraftModel
        _require_dflare_plus(config, variant, config_path)
        config.router = None
        config.dflare = None
        # Set explicitly, not from the YAML: `fresh=True` rmtree's this path, so
        # a mismatched --config must never be able to redirect it at another
        # variant's checkpoints.
        config.checkpoint.save_dir = "/checkpoints/dflare-deep-qwen3-4b"
        config.logging.wandb_project = "dflare-deep"
        logger.info("Training DFLARE-DEEP variant (DFlare + shrunk MLP + more layers, no causal/mixing)")
    elif variant == "dflare-plus":
        model_cls = DFlarePlusDraftModel
        _require_dflare_plus(config, variant, config_path)
        config.router = None
        config.dflare = None
        config.checkpoint.save_dir = "/checkpoints/dflare-plus-qwen3-4b"
        config.logging.wandb_project = "dflare-plus"
        logger.info("Training DFLARE-PLUS variant (DFlare + token mixing + shrunk MLP; no causal mask)")
    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Valid variants: {sorted(VARIANT_CONFIGS)}"
        )

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
    config: str = "",
    download: bool = False,
    fresh: bool = True,
    epochs: int = 0,
):
    if download:
        download_models.remote()
        return
    if variant not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Valid variants: {sorted(VARIANT_CONFIGS)}"
        )
    # Default to the variant's own config. Passing --config explicitly overrides
    # it, but train() then fail-fasts if the YAML lacks the required arch block.
    config = config or f"configs/{VARIANT_CONFIGS[variant]}"

    # Map local config path to container path
    config_name = Path(config).name
    container_config = f"/root/configs/{config_name}"

    # spawn(), not remote(): with `modal run --detach`, a .remote() call is
    # cancelled as soon as the local caller disconnects, which silently kills
    # long training runs mid-flight. spawn() hands off and returns immediately.
    call = train.spawn(
        config_path=container_config, variant=variant, fresh=fresh, epochs=epochs
    )
    print(f"Spawned {variant} training — call id: {call.object_id}")
    print(f"Follow with: modal app logs {app.app_id}")
