"""
Training loop for any DFlash draft model.

Works with any subclass of `DFlashDraftModel`. The loop uses only the public
model interface:
  - `model.extract_target_hidden(...)`  — picks layers + format
  - `model(...)`                         — forward pass
  - `model.compute_loss(...)`            — full loss (subclasses may add aux)

Subclass-specific kwargs (e.g. `entropy_reg_weight` for the routed variant)
are passed through `compute_loss`; the base class swallows them via `**_`.

Pipeline pieces all live elsewhere:
  - data: `DFlashDataProcessor`, `OnlineHiddenStateExtractor`, `prepare_draft_inputs`
  - loss: `compute_position_weighted_ce_loss` (called inside `model.compute_loss`)
  - block construction: `prepare_draft_inputs`
"""

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from midas.config import CorrectionConfig, DFlareConfig, TrainConfig
from midas.data import (
    DFlashDataProcessor,
    OnlineHiddenStateExtractor,
    prepare_draft_inputs,
)
from midas.data_correction import prepare_correction_batch
from midas.models import (
    CorrectionDraftModel,
    DFlash2DraftModel,
    DFlashDraftModel,
    DFlashDraftModelRouted,
    DFlashDraftModelSplitQuery,
    DFlareDraftModel,
    compute_progressive_gamma,
)


def _router_constructor_kwargs(config: TrainConfig) -> dict:
    if config.router is None:
        raise ValueError(
            "DFlashDraftModelRouted needs a `router` config. Add a `router:` "
            "block to your yaml or set `config.router = RouterConfig(...)` "
            "before calling train()."
        )
    return {"router_config": config.router}


def _dflare_constructor_kwargs(config: TrainConfig) -> dict:
    if config.dflare is None:
        from midas.config import DFlareConfig
        config.dflare = DFlareConfig()
    return {"dflare_config": config.dflare}


def _dflash2_constructor_kwargs(config: TrainConfig) -> dict:
    if config.dflash2 is None:
        from midas.config import DFlash2Config
        config.dflash2 = DFlash2Config()
    return {"dflash2_config": config.dflash2}


def _correction_constructor_kwargs(config: TrainConfig) -> dict:
    if config.correction is None:
        config.correction = CorrectionConfig()
    return {"correction_config": config.correction}


_MODEL_CONSTRUCTOR_KWARGS = {
    CorrectionDraftModel: _correction_constructor_kwargs,
    DFlashDraftModel: lambda config: {},
    DFlash2DraftModel: _dflash2_constructor_kwargs,
    DFlashDraftModelRouted: _router_constructor_kwargs,
    DFlashDraftModelSplitQuery: lambda config: {},
    DFlareDraftModel: _dflare_constructor_kwargs,
}


def save_checkpoint(
    model: DFlashDraftModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    prev_batch_idx: int,
    loss: float,
    save_dir: str,
    is_best: bool = False,
):
    ckpt_dir = Path(save_dir) / f"checkpoint_epoch{epoch}_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "epoch": epoch,
                "step": step,
                "prev_batch_idx": prev_batch_idx,
                "loss": loss,
            },
            f,
        )

    latest = Path(save_dir) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_dir.name)

    if is_best:
        best = Path(save_dir) / "best"
        if best.exists() or best.is_symlink():
            best.unlink()
        best.symlink_to(ckpt_dir.name)

    logger.info(f"Saved checkpoint to {ckpt_dir} (best={is_best})")


def load_checkpoint(
    model: DFlashDraftModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    save_dir: str,
) -> tuple[int, int, int]:
    latest = Path(save_dir) / "latest"
    if not latest.exists():
        return 0, 0, -1

    ckpt_dir = latest.resolve()
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", weights_only=True))
    optimizer.load_state_dict(
        torch.load(ckpt_dir / "optimizer.pt", weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(ckpt_dir / "scheduler.pt", weights_only=True)
    )

    with open(ckpt_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info(
        f"Resumed from {ckpt_dir} (epoch={metadata['epoch']}, step={metadata['step']})"
    )
    return metadata["epoch"], metadata["step"], metadata.get("prev_batch_idx", -1)


def cleanup_old_checkpoints(save_dir: str, keep_last_n: int):
    save_path = Path(save_dir)
    checkpoints = sorted(
        [
            d
            for d in save_path.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint_")
        ],
        key=lambda d: d.stat().st_mtime,
    )

    best_link = save_path / "best"
    best_name = best_link.resolve().name if best_link.exists() else None

    for ckpt in checkpoints[:-keep_last_n]:
        if ckpt.name != best_name:
            shutil.rmtree(ckpt)
            logger.info(f"Removed old checkpoint: {ckpt.name}")


def load_warm_start(model: DFlashDraftModel, checkpoint_dir: str) -> None:
    """Overlay a trained checkpoint's weights onto a possibly-extended model.

    Modules the checkpoint does not contain (e.g. DFlash 2's convolutions and
    selector) keep their constructor initialization — for identity-initialized
    convs that means the model starts as *exactly* the loaded checkpoint.

    Fails loudly on any key the model does not accept. A silent partial load
    would look like a warm start while actually training from near-random, so
    an unexpected key is treated as "wrong checkpoint for this variant".
    """
    path = Path(checkpoint_dir)
    if path.is_symlink():
        path = path.resolve()
    state = torch.load(path / "model.pt", weights_only=True, map_location="cpu")
    dtype = next(model.parameters()).dtype
    state = {k: v.to(dtype) for k, v in state.items()}

    # strict=False tolerates the NEW modules being absent; shape mismatches still raise.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(
            f"Warm-start checkpoint {path} has {len(unexpected)} keys this model does "
            f"not accept (e.g. {sorted(unexpected)[:3]}). Wrong variant for this checkpoint?"
        )
    loaded = len(state)
    total = len(model.state_dict())
    logger.info(
        f"Warm-started {loaded}/{total} tensors from {path}. "
        f"Newly initialized: {sorted(missing)[:4]}{' ...' if len(missing) > 4 else ''} "
        f"({len(missing)} tensors)"
    )
    if loaded == 0:
        raise ValueError(f"Warm-start checkpoint {path} matched no parameters at all.")


def build_default_model(
    config: TrainConfig,
    device: torch.device,
    model_cls: type[DFlashDraftModel] = DFlashDraftModelRouted,
    random_init: bool = False,
) -> DFlashDraftModel:
    """
    Build a draft model from `config.draft_checkpoint`. Pass `model_cls` to
    pick which DFlashDraftModel subclass to use. If `random_init=True`, all
    weights start random (for fair from-scratch comparisons).
    """
    if model_cls not in _MODEL_CONSTRUCTOR_KWARGS:
        raise ValueError(
            f"Unsupported model class: {model_cls.__name__}. "
            f"Supported: {[c.__name__ for c in _MODEL_CONSTRUCTOR_KWARGS]}"
        )
    constructor_kwargs = _MODEL_CONSTRUCTOR_KWARGS[model_cls](config)
    factory = model_cls.from_config_random if random_init else model_cls.from_pretrained_dflash
    model = factory(
        config.draft_checkpoint,
        device=device,
        dtype=torch.float32,
        **constructor_kwargs,
    )
    if config.warm_start:
        load_warm_start(model, config.warm_start)
    return model


def train(
    config: TrainConfig,
    model: Optional[DFlashDraftModel] = None,
    model_cls: type[DFlashDraftModel] = DFlashDraftModelRouted,
    random_init: bool = False,
    on_checkpoint: Optional[Callable[[], None]] = None,
):
    """Main training loop. `model` can be any DFlashDraftModel subclass.

    `on_checkpoint` runs right after every `save_checkpoint`. On Modal this is
    `Volume.commit` — without it the checkpoint files exist only in the
    container's view of the volume until the function returns, so a crash,
    preemption or timeout loses every checkpoint the run wrote.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import wandb

        wandb.init(project=config.logging.wandb_project, config=vars(config))
        use_wandb = True
    except Exception:
        use_wandb = False
        logger.warning("WandB not available, logging to stdout only")

    logger.info(f"Loading target model: {config.target_model}")
    target_model = AutoModelForCausalLM.from_pretrained(
        config.target_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(config.target_model)

    if model is None:
        logger.info(
            f"Building {model_cls.__name__} from {config.draft_checkpoint}"
        )
        model = build_default_model(config, device, model_cls=model_cls, random_init=random_init)
    elif isinstance(model, DFlareDraftModel) and config.dflare is None:
        config.dflare = DFlareConfig()
    elif isinstance(model, CorrectionDraftModel) and config.correction is None:
        config.correction = model.correction_config
    model.train()

    # A correction run needs a SECOND, frozen model: the round-1 drafter whose
    # residuals are being recycled. bf16 because it is inference-only — only the
    # trainable model needs fp32 for the optimizer.
    parent_drafter = None
    if isinstance(model, CorrectionDraftModel):
        parent_drafter = DFlash2DraftModel.from_pretrained_dflash(
            config.draft_checkpoint,
            device=device,
            dtype=torch.bfloat16,
            dflash2_config=config.correction.dflash2,
        )
        if config.correction.parent_checkpoint:
            load_warm_start(parent_drafter, config.correction.parent_checkpoint)
        else:
            logger.warning(
                "correction.parent_checkpoint is unset, so round 1 runs the base "
                "checkpoint with a randomly initialized selector and identity convs. "
                "The recycled residuals will not be the ones deployment produces."
            )
        parent_drafter.eval()
        for param in parent_drafter.parameters():
            param.requires_grad = False
        # One `extract_target_hidden` call per step feeds BOTH models, so they
        # have to agree on which target layers they consume. They do by default
        # (CorrectionConfig inherits the parent's ids); catch an explicit
        # override here rather than at the first forward, where it surfaces as a
        # shape mismatch inside the fusion.
        if model.target_layer_ids != parent_drafter.target_layer_ids:
            raise ValueError(
                f"correction target_layer_ids {model.target_layer_ids} != parent's "
                f"{parent_drafter.target_layer_ids}; the train loop extracts target "
                f"hidden once and hands the same tensor to both rounds."
            )
        # An explicit warm_start (resuming a correction run) wins over seeding
        # from the parent's last layer.
        if not config.warm_start:
            model.warm_start_from_parent(parent_drafter)

    extractor = OnlineHiddenStateExtractor(target_model, device)
    processor = DFlashDataProcessor(tokenizer, config)

    total_steps = (
        len(processor) * config.num_epochs // config.gradient_accumulation_steps
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    start_epoch, start_step, start_batch_idx = load_checkpoint(
        model, optimizer, scheduler, config.checkpoint.save_dir
    )

    best_loss = float("inf")
    global_step = start_step
    stop_early = False
    # Per-micro-batch scalars, averaged at the optimizer step. Logging only the
    # last micro-batch is not just noisy but BIASED: the length-interleaved
    # sampler yields [longest, shortest, 2nd-longest, ...], so with an even
    # gradient_accumulation_steps the last one always comes from the short end,
    # where `prepare_draft_inputs` fits far fewer anchors and the position-
    # weighted mean is taken over very few supervised positions.
    micro_losses: list[float] = []
    micro_metrics: list[dict] = []

    logger.info(
        f"Starting training: {config.num_epochs} epochs, "
        f"{total_steps} total steps, {len(processor.dataset)} samples"
    )

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (input_ids, seq_lengths, response_starts) in enumerate(processor):
            if epoch == start_epoch and batch_idx <= start_batch_idx:
                continue

            input_ids = input_ids.to(device)
            seq_lengths = seq_lengths.to(device)
            response_starts = response_starts.to(device)

            # Frozen target forward → per-layer hidden states.
            hidden_states = extractor.extract(input_ids)
            target_hidden = model.extract_target_hidden(hidden_states)

            # Anchor sampling from response portion + block construction.
            batch = prepare_draft_inputs(
                input_ids,
                seq_lengths,
                block_size=config.block_size,
                max_num_anchors=config.max_num_anchors,
                mask_token_id=model.mask_token_id or 0,
                response_starts=response_starts,
            )

            # Correction runs replace the round-1 batch with the round-2 one: the
            # frozen drafter proposes, acceptance is simulated against the target's
            # committed tokens, and the rejected suffix becomes this step's block.
            correction_kwargs: dict = {}
            correction_metrics: dict = {}
            if parent_drafter is not None:
                step = prepare_correction_batch(
                    parent_drafter,
                    input_ids,
                    batch,
                    target_hidden,
                    target_model.model.embed_tokens,
                    target_model.lm_head,
                    block_size=config.block_size,
                    mask_token_id=model.mask_token_id or 0,
                    min_remaining=config.correction.min_remaining,
                    below_trigger_keep_prob=config.correction.below_trigger_keep_prob,
                    chunk_blocks=config.correction.round1_chunk_blocks,
                )
                batch = step.batch
                correction_kwargs = {
                    "recycled_hidden": step.recycled_hidden,
                    "is_recycled": step.is_recycled,
                }
                correction_metrics = step.metrics

            # Draft forward + loss in bf16 autocast (fast), optimizer in fp32 (precise).
            noise_embedding = target_model.model.embed_tokens(batch.noise_token_ids)
            autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
            with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                draft_output = model(
                    position_ids=batch.position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    attention_mask=batch.attention_mask,
                    **correction_kwargs,
                )
                raw_model = model.module if hasattr(model, "module") else model
                loss_gamma = (
                    compute_progressive_gamma(config.dflare, epoch)
                    if isinstance(raw_model, DFlareDraftModel) and config.dflare is not None
                    else config.loss_gamma
                )
                loss = model.compute_loss(
                    draft_output,
                    input_ids,
                    batch,
                    target_model.lm_head,
                    gamma=loss_gamma,
                    # Only the DFlash 2 selector reads this; every other
                    # compute_loss absorbs it via **_.
                    embed_tokens=target_model.model.embed_tokens,
                )

            micro_losses.append(loss.item())
            micro_metrics.append(
                {**correction_metrics, **getattr(raw_model, "_last_loss_metrics", {})}
            )

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = sum(micro_losses) / len(micro_losses)
                window_metrics = {
                    k: sum(m[k] for m in micro_metrics if k in m)
                    / max(1, sum(k in m for m in micro_metrics))
                    for k in {k for m in micro_metrics for k in m}
                }
                micro_losses.clear()
                micro_metrics.clear()
                epoch_loss += step_loss
                epoch_steps += 1

                if global_step % config.logging.log_every_steps == 0:
                    log_dict = model.log_metrics(
                        loss=step_loss,
                        lr=scheduler.get_last_lr()[0],
                        epoch=epoch,
                        step=global_step,
                    )
                    # After, not as kwargs: DFlash 2's log_metrics appends its own
                    # `_last_loss_metrics` last, which would clobber the averages
                    # with the final micro-batch's values.
                    log_dict.update(window_metrics)

                    if use_wandb:
                        wandb.log(log_dict, step=global_step)
                    else:
                        logger.info(
                            f"step={global_step} loss={step_loss:.4f} "
                            + " ".join(
                                f"{k}={v:.4f}"
                                for k, v in log_dict.items()
                                if k not in {"epoch", "step"} and isinstance(v, float)
                            )
                        )

                if global_step % config.checkpoint.save_every_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, global_step, batch_idx, avg_loss,
                        config.checkpoint.save_dir, is_best,
                    )
                    cleanup_old_checkpoints(
                        config.checkpoint.save_dir, config.checkpoint.keep_last_n
                    )
                    if on_checkpoint is not None:
                        on_checkpoint()

                if config.max_steps is not None and global_step >= config.max_steps:
                    logger.info(
                        f"Reached max_steps={config.max_steps}; saving and stopping."
                    )
                    stop_early = True
                    break

        avg_loss = epoch_loss / max(epoch_steps, 1)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1, global_step, -1, avg_loss,
            config.checkpoint.save_dir, is_best,
        )
        cleanup_old_checkpoints(
            config.checkpoint.save_dir, config.checkpoint.keep_last_n
        )
        if on_checkpoint is not None:
            on_checkpoint()
        logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")
        if stop_early:
            break

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    if use_wandb:
        wandb.finish()
