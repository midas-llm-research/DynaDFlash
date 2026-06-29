from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import yaml


class SparseAttentionType(str, Enum):
    SLIDING_WINDOW = "sliding_window"


@dataclass
class SparseAttentionConfig:
    type: SparseAttentionType = SparseAttentionType.SLIDING_WINDOW
    sink_tokens: int = 4
    window: int = 512


class BlockAggregationType(str, Enum):
    """How a draft block's per-token hidden states are collapsed to one vector
    before the layer-skip gate scores it."""
    EXP_DECAY = "exp_decay"   # fixed weights ~ exp(-(pos-1)/gamma) over drafted tokens (anchor excluded)
    ATTENTION = "attention"   # per-layer attention pool (query = most-recent context hidden)


@dataclass
class SkipConfig:
    """Layer-skip (early-exit) for the draft, used by DFlashDraftModelSkip.

    After each decoder layer, the block hidden is aggregated to one vector and a
    per-layer linear gate decides whether to exit. Training adds an LM head at
    every depth (multi-layer CE) plus a BCE gate loss whose labels come from a
    latency-vs-acceptance speedup score (see midas.latency.LatencyModel).
    """
    aggregation: BlockAggregationType = BlockAggregationType.ATTENTION
    gamma: float = 7.0            # exp_decay: position weight ~ exp(-pos / gamma)
    rank: int = 128               # attention: q/k projection rank
    gate_loss_weight: float = 1.0  # weight of gate BCE relative to multi-layer CE
    exit_threshold: float = 0.0   # inference: exit when gate logit > threshold
    latency_path: Optional[str] = None  # JSON of fitted latency coefficients (None -> defaults)


@dataclass
class RouterConfig:
    hidden_size: int = 2560
    num_layers: int = 5
    rank: int = 128
    gamma_init: float = 0.01
    temperature_init: float = 1.0
    use_fc: bool = True


@dataclass
class CheckpointConfig:
    save_every_steps: int = 2000
    save_dir: str = "/checkpoints/midas-router"
    keep_last_n: int = 3


@dataclass
class LoggingConfig:
    wandb_project: str = "midas-dflash-router"
    log_every_steps: int = 50
    eval_every_steps: int = 1000


@dataclass
class TrainConfig:
    target_model: str = "Qwen/Qwen3-4B"
    draft_checkpoint: str = "z-lab/Qwen3-4B-DFlash-b16"
    data_path: str = "/checkpoints/sdd/sdd_100k.jsonl"

    lr: float = 6e-4
    weight_decay: float = 0.01
    num_epochs: int = 2  # remember to increase to 8ish for later runs
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 3072
    block_size: int = 16
    max_num_anchors: int = 512
    warmup_ratio: float = 0.04
    grad_clip: float = 1.0
    loss_gamma: float = 7.0

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Method-specific sub-configs. None by default — the base DFlash model has
    # no method-specific config. To train a variant that needs one, set the
    # corresponding field (e.g. `router=RouterConfig(...)`) before calling
    # `train()`, or include a top-level block of the same name in your yaml.
    # Add new method configs as new `Optional[...] = None` fields below.
    router: Optional[RouterConfig] = None
    skip: Optional[SkipConfig] = None

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
        skip_raw = raw.pop("skip", None)
        if skip_raw is not None:
            method_kwargs["skip"] = SkipConfig(**skip_raw)

        return cls(
            target_model=raw.get("target_model", cls.target_model),
            draft_checkpoint=raw.get("draft_checkpoint", cls.draft_checkpoint),
            data_path=raw.get("data_path", cls.data_path),
            checkpoint=ckpt_cfg,
            logging=log_cfg,
            **train_kwargs,
            **method_kwargs,
        )
