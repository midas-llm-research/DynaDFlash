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
from midas.models import DFlashDraftModel, DFlashDraftModelRouted


def save_checkpoint(
    model: DFlashDraftModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
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
        json.dump({"epoch": epoch, "step": step, "loss": loss}, f)

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
) -> tuple[int, int]:
    latest = Path(save_dir) / "latest"
    if not latest.exists():
        return 0, 0

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
    return metadata["epoch"], metadata["step"]


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
) -> DFlashDraftModel:
    """
    Build a draft model from `config.draft_checkpoint`. Pass `model_cls` to
    pick which DFlashDraftModel subclass to use. Method-specific configs (e.g.
    `config.router` for the routed variant) are pulled here and forwarded as
    constructor kwargs.
    """
    extra_kwargs: dict = {}
    if model_cls is DFlashDraftModelRouted:
        if config.router is None:
            raise ValueError(
                "DFlashDraftModelRouted needs a `router` config. Add a "
                "`router:` block to your yaml, set `config.router = "
                "RouterConfig(...)` before calling train(), or pass "
                "`model_cls=DFlashDraftModel` to train the base variant."
            )
        extra_kwargs["router_config"] = config.router
    return model_cls.from_pretrained_dflash(
        config.draft_checkpoint,
        device=device,
        dtype=torch.bfloat16,
        **extra_kwargs,
    )


def train(
    config: TrainConfig,
    model: Optional[DFlashDraftModel] = None,
    model_cls: type[DFlashDraftModel] = DFlashDraftModelRouted,
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
        model = build_default_model(config, device, model_cls=model_cls)
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

    start_epoch, start_step = load_checkpoint(
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

        for batch_idx, (input_ids, seq_lengths) in enumerate(processor):
            if epoch == start_epoch and batch_idx < start_step % len(processor):
                continue

            input_ids = input_ids.to(device)
            seq_lengths = seq_lengths.to(device)

            # Frozen target forward → per-layer hidden states + logits.
            hidden_states, target_logits = extractor.extract(input_ids)
            target_hidden = model.extract_target_hidden(hidden_states)

            # Anchor sampling + block construction. seq_lengths bounds anchors
            # away from the right-padded region per row.
            batch = prepare_draft_inputs(
                input_ids,
                seq_lengths,
                block_size=config.block_size,
                num_anchors=config.num_anchors,
                mask_token_id=model.mask_token_id or 0,
            )

            # Draft forward.
            noise_embedding = target_model.model.embed_tokens(batch.noise_token_ids)
            draft_output = model(
                position_ids=batch.position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                attention_mask=batch.attention_mask,
            )

            # Loss (subclass-specific terms live inside model.compute_loss).
            loss = model.compute_loss(
                draft_output,
                target_logits,
                batch,
                target_model.lm_head,
                gamma=config.loss_gamma,
                entropy_reg_weight=config.entropy_reg_weight,
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
                        epoch, global_step, avg_loss,
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
            epoch + 1, global_step, avg_loss,
            config.checkpoint.save_dir, is_best,
        )
        cleanup_old_checkpoints(
            config.checkpoint.save_dir, config.checkpoint.keep_last_n
        )
        logger.info(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    if use_wandb:
        wandb.finish()
