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
from typing import Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from midas.config import TrainConfig
from midas.data import (
    DFlashDataProcessor,
    OnlineHiddenStateExtractor,
    prepare_draft_inputs,
)
from midas.models import DFlashDraftModel, DFlashDraftModelRouted, DFlashDraftModelSkip


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
    extra_kwargs: dict = {}
    if model_cls is DFlashDraftModelSkip:
        if config.skip is None:
            raise ValueError(
                "DFlashDraftModelSkip needs a `skip` config. Add a `skip:` block "
                "to your yaml or set `config.skip = SkipConfig(...)` before train()."
            )
        extra_kwargs["skip_config"] = config.skip
    elif model_cls is not DFlashDraftModel:
        if config.router is None:
            raise ValueError(
                f"{model_cls.__name__} needs a `router` config. Add a "
                "`router:` block to your yaml or set `config.router = "
                "RouterConfig(...)` before calling train()."
            )
        extra_kwargs["router_config"] = config.router

    factory = model_cls.from_config_random if random_init else model_cls.from_pretrained_dflash
    return factory(
        config.draft_checkpoint,
        device=device,
        dtype=torch.float32,
        **extra_kwargs,
    )


def train(
    config: TrainConfig,
    model: Optional[DFlashDraftModel] = None,
    model_cls: type[DFlashDraftModel] = DFlashDraftModelRouted,
    random_init: bool = False,
):
    """Main training loop. `model` can be any DFlashDraftModel subclass."""
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
    model.train()

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

            # Draft forward + loss in bf16 autocast (fast), optimizer in fp32 (precise).
            noise_embedding = target_model.model.embed_tokens(batch.noise_token_ids)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                draft_output = model(
                    position_ids=batch.position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    attention_mask=batch.attention_mask,
                )
                loss = model.compute_loss(
                    draft_output,
                    input_ids,
                    batch,
                    target_model.lm_head,
                    gamma=config.loss_gamma,
                )

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = loss.item() * config.gradient_accumulation_steps
                epoch_loss += step_loss
                epoch_steps += 1

                if global_step % config.logging.log_every_steps == 0:
                    log_dict = model.log_metrics(
                        loss=step_loss,
                        lr=scheduler.get_last_lr()[0],
                        epoch=epoch,
                        step=global_step,
                    )

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
        logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    if use_wandb:
        wandb.finish()
