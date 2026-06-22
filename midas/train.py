"""
Training loop for any DFlash draft model.

Works with any subclass of `DFlashDraftModel` — the loop only uses the
abstract interface (`extract_target_hidden`, `forward`, `routing_aux_loss`),
so adding a new method = new file in `midas/models/` + swap the constructor.

Implements:
- Position-weighted KL loss (DFlash paper Eq. 4)
- Online hidden state extraction from frozen target
- Optional auxiliary loss from the draft model (e.g. router entropy reg)
- Checkpointing with full resume capability
- WandB logging
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from midas.config import TrainConfig
from midas.data import (
    DFlashDataset,
    OnlineHiddenStateExtractor,
    build_block_attention_mask,
    collate_fn,
    create_masked_blocks,
)
from midas.models import DFlashDraftModel, DFlashDraftModelRouted


def compute_position_weighted_kl_loss(
    draft_hidden: torch.Tensor,
    target_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    position_in_block: torch.Tensor,
    lm_head: nn.Module,
    gamma: float = 7.0,
) -> torch.Tensor:
    """
    Position-weighted KL divergence between target and draft, computed only at
    masked positions to avoid OOM on full [B, T, V] logit tensors.

    The target is causal (logits[t] predicts t+1); the draft is diffusion
    (hidden[t] predicts the token AT t). So the target's prediction for the
    masked position t comes from target_logits[t-1]. Masked positions are
    always >= 1, so t-1 >= 0.
    """
    masked_indices = loss_mask.nonzero(as_tuple=False)
    if masked_indices.shape[0] == 0:
        return torch.tensor(0.0, device=draft_hidden.device, requires_grad=True)

    batch_idx = masked_indices[:, 0]
    seq_idx = masked_indices[:, 1]

    draft_hidden_masked = draft_hidden[batch_idx, seq_idx]
    target_logits_masked = target_logits[batch_idx, seq_idx - 1]
    positions_masked = position_in_block[batch_idx, seq_idx]

    draft_logits_masked = lm_head(draft_hidden_masked)

    weights = torch.exp(-((positions_masked.float() - 1) / gamma))

    target_log_probs = F.log_softmax(target_logits_masked, dim=-1)
    draft_log_probs = F.log_softmax(draft_logits_masked, dim=-1)

    kl = F.kl_div(
        draft_log_probs, target_log_probs.exp(), reduction="none"
    ).sum(dim=-1)
    return (kl * weights).sum() / weights.sum().clamp(min=1)


def build_training_position_ids(
    seq_len: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    """
    Position ids for training: concatenate noise positions twice, since
    attention keys are [k_ctx, k_noise] both covering the same logical
    sequence positions [0..T-1]. Length 2T.
    """
    pos = torch.arange(seq_len, device=device)
    return torch.cat([pos, pos]).unsqueeze(0).expand(batch_size, -1)


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
    config: TrainConfig, device: torch.device
) -> DFlashDraftModel:
    """
    Default draft model factory. Swap this (or pass `model` to `train`) to
    train a different DFlash variant.
    """
    return DFlashDraftModelRouted.from_pretrained_dflash(
        config.draft_checkpoint,
        router_config=config.router,
        device=device,
        dtype=torch.bfloat16,
    )


def train(
    config: TrainConfig,
    model: Optional[DFlashDraftModel] = None,
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
        logger.info(f"Building draft model from: {config.draft_checkpoint}")
        model = build_default_model(config, device)
    model.train()

    extractor = OnlineHiddenStateExtractor(target_model, device)

    logger.info("Loading training data...")
    dataset = DFlashDataset(tokenizer, max_seq_len=config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id or 0),
        num_workers=4,
        pin_memory=True,
    )

    total_steps = (
        len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
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
        f"{total_steps} total steps, {len(dataset)} samples"
    )

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, input_ids in enumerate(dataloader):
            if epoch == start_epoch and batch_idx < start_step % len(dataloader):
                continue

            input_ids = input_ids.to(device)

            hidden_states, target_logits = extractor.extract(input_ids)
            target_hidden = model.extract_target_hidden(hidden_states)

            masked_ids, loss_mask, position_in_block, block_ids, block_anchors = (
                create_masked_blocks(
                    input_ids,
                    block_size=config.block_size,
                    num_anchors=config.num_anchors,
                    mask_token_id=model.mask_token_id or 0,
                )
            )

            ctx_len = input_ids.shape[1]
            attn_mask = build_block_attention_mask(
                block_ids, block_anchors, ctx_len, dtype=torch.bfloat16
            )

            noise_embedding = target_model.model.embed_tokens(masked_ids)
            position_ids = build_training_position_ids(
                masked_ids.shape[1], masked_ids.shape[0], device
            )

            draft_output = model(
                position_ids=position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                attention_mask=attn_mask,
            )

            loss = compute_position_weighted_kl_loss(
                draft_output,
                target_logits,
                loss_mask,
                position_in_block,
                target_model.lm_head,
                config.loss_gamma,
            )

            aux = model.routing_aux_loss()
            if aux is not None and config.entropy_reg_weight > 0:
                loss = loss - config.entropy_reg_weight * aux

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
                    log_dict = {
                        "loss": step_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step,
                    }
                    if aux is not None:
                        log_dict["aux_loss"] = aux.item()
                    last_rw = getattr(model, "last_routing_weights", None)
                    if last_rw is not None:
                        mean_weights = last_rw.mean(dim=(0, 1)).tolist()
                        for i, w in enumerate(mean_weights):
                            log_dict[f"routing_weight_layer_{i}"] = w
                        if hasattr(model, "context_router"):
                            log_dict["gamma"] = (
                                model.context_router.log_gamma.exp().item()
                            )
                            log_dict["tau"] = (
                                model.context_router.log_tau.exp().item()
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
