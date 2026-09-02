from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class RouterConfig:
    hidden_size: int = 2560
    num_layers: int = 5
    rank: int = 128
    gamma_init: float = 1.0
    temperature_init: float = 1.0
    use_fc: bool = True


@dataclass
class DFlareConfig:
    hidden_size: int = 2560
    num_draft_layers: int = 7
    num_target_layers: int = 9
    gamma_initial: float = 4.5
    gamma_increment_per_epoch: float = 1.0


@dataclass
class DFlash2Config:
    """DFlash 2: path selector + block-local dynamic convolution.

    selector_rank/top_k follow the released checkpoint (256 / 16). Selector cost is
    3 * hidden * rank = 1.97M at hidden=2560 (matching the blog's "+2.0M"); the two
    convs per layer cost hidden^2/4 each, i.e. 16.4M over 5 layers (their "16.5M, 3%").
    """
    selector_rank: int = 256
    selector_top_k: int = 16
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    lambda_selector: float = 1.0


@dataclass
class CorrectionConfig:
    """Depth-1 correction model that re-drafts a rejected suffix from its own
    recycled hidden states (see papers/hidden_state_recycling.md).

    `target_layer_ids=None` means "inherit the parent drafter's layers", which is
    the default for a reason: `dflash_generate` extracts `target_hidden` once per
    iteration, so sharing the layer set lets one tensor feed both models instead
    of forcing a second extraction and a second fusion. It also makes the
    parent's `fc` shape-compatible, which is what `warm_start_from_parent` needs.
    """
    num_layers: int = 1
    target_layer_ids: Optional[list[int]] = None
    # Trigger policy: run a correction round only when at least this many
    # positions of the block are left to re-draft, and never twice in a row.
    min_remaining: int = 7
    # RMSNorm the recycled residual before mixing. OFF by default: a recycled
    # position then passes through untouched, so a parent-warm-started layer is
    # exactly "the 6th drafter layer" at init. Normalizing shrinks the residual
    # relative to the sublayer contribution (layer-5 residual RMS >> 1), which
    # makes the correction edit far more aggressive. Exposed as an ablation.
    normalize_recycled: bool = False
    # The correction layer keeps DFlash 2's conv + selector; depth 1 is exactly
    # the regime the conv was introduced to rescue.
    dflash2: DFlash2Config = field(default_factory=DFlash2Config)


@dataclass
class CheckpointConfig:
    save_every_steps: int = 2000
    save_dir: str = "/checkpoints/midas-router"
    keep_last_n: int = 3


@dataclass
class LoggingConfig:
    wandb_project: str = "midas-dflash-router"
    log_every_steps: int = 50


@dataclass
class TrainConfig:
    target_model: str = "Qwen/Qwen3-4B"
    draft_checkpoint: str = "z-lab/Qwen3-4B-DFlash-b16"
    data_path: str = "/checkpoints/sdd/sdd_100k.jsonl"

    lr: float = 6e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 3072
    block_size: int = 16
    max_num_anchors: int = 512
    warmup_ratio: float = 0.04
    grad_clip: float = 1.0
    loss_gamma: float = 7.0
    # Directory holding a trained model.pt to overlay onto a freshly built
    # model (weights only — optimizer and LR schedule start fresh, so this
    # is a NEW run on top of that checkpoint, not a resume).
    warm_start: Optional[str] = None

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Method-specific sub-configs. None by default — the base DFlash model has
    # no method-specific config. To train a variant that needs one, set the
    # corresponding field (e.g. `router=RouterConfig(...)`) before calling
    # `train()`, or include a top-level block of the same name in your yaml.
    # Add new method configs as new `Optional[...] = None` fields below.
    router: Optional[RouterConfig] = None
    dflare: Optional[DFlareConfig] = None
    dflash2: Optional[DFlash2Config] = None
    correction: Optional[CorrectionConfig] = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        ckpt_cfg = CheckpointConfig(**raw.pop("checkpointing", {}))
        log_cfg = LoggingConfig(**raw.pop("logging", {}))
        train_kwargs = raw.pop("training", {})

        # Method-specific sub-configs are only built if their yaml block exists.
        # Mirror this branch when adding a new method config above.
        method_kwargs: dict = {}
        router_raw = raw.pop("router", None)
        if router_raw is not None:
            method_kwargs["router"] = RouterConfig(**router_raw)
        dflare_raw = raw.pop("dflare", None)
        if dflare_raw is not None:
            method_kwargs["dflare"] = DFlareConfig(**dflare_raw)
        dflash2_raw = raw.pop("dflash2", None)
        if dflash2_raw is not None:
            method_kwargs["dflash2"] = DFlash2Config(**dflash2_raw)
        correction_raw = raw.pop("correction", None)
        if correction_raw is not None:
            nested = correction_raw.pop("dflash2", None)
            correction = CorrectionConfig(**correction_raw)
            if nested is not None:
                correction.dflash2 = DFlash2Config(**nested)
            method_kwargs["correction"] = correction

        known_top_keys = {"target_model", "draft_checkpoint", "data_path"}
        unknown = set(raw.keys()) - known_top_keys
        if unknown:
            raise ValueError(f"Unknown keys in config YAML: {unknown}")

        known_top_keys = {"target_model", "draft_checkpoint", "data_path"}
        unknown = set(raw.keys()) - known_top_keys
        if unknown:
            raise ValueError(f"Unknown keys in config YAML: {unknown}")

        return cls(
            target_model=raw.get("target_model", cls.target_model),
            draft_checkpoint=raw.get("draft_checkpoint", cls.draft_checkpoint),
            data_path=raw.get("data_path", cls.data_path),
            checkpoint=ckpt_cfg,
            logging=log_cfg,
            **train_kwargs,
            **method_kwargs,
        )
