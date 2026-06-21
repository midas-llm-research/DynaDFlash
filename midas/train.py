"""
Training loop for DFlash with context router.

Implements:
- Position-weighted KL loss (DFlash paper Eq. 4)
- Online hidden state extraction from frozen target
- Checkpointing with full resume capability
- WandB logging of loss, routing entropy, gamma, etc.
"""

import json
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from loguru import logger

from midas.config import TrainConfig
from midas.data import (
    DFlashDataset,
    OnlineHiddenStateExtractor,
    build_block_attention_mask,
    collate_fn,
    create_masked_blocks,
)
from midas.model import DFlashDraftModelRouted, build_target_layer_ids
from midas.config import RouterConfig


def compute_position_weighted_kl_loss(
    draft_hidden: torch.Tensor,
    target_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    position_in_block: torch.Tensor,
    lm_head: nn.Module,
    gamma: float = 7.0,
) -> torch.Tensor:
    """
    Position-weighted KL divergence loss, computed only at masked positions
    to avoid OOM from full [B, T, V] logit tensors.

    Args:
        draft_hidden: [B, T, H] draft model hidden states (before lm_head)
        target_logits: [B, T, V] target model logits (full sequence)
        loss_mask: [B, T] binary mask for valid loss positions
        position_in_block: [B, T] actual position within each block (1-indexed)
        lm_head: the target model's lm_head (shared with draft)
        gamma: decay rate (higher = slower decay)
    """
    # Gather only masked positions to save memory
    masked_indices = loss_mask.nonzero(as_tuple=False)  # [num_masked, 2]

    if masked_indices.shape[0] == 0:
        return torch.tensor(0.0, device=draft_hidden.device, requires_grad=True)

    # Extract hidden states and target logits at masked positions only
    batch_idx = masked_indices[:, 0]
    seq_idx = masked_indices[:, 1]

    draft_hidden_masked = draft_hidden[batch_idx, seq_idx]  # [num_masked, H]
    # CRITICAL: Target is causal LM (logits[t] predicts t+1), draft is diffusion
    # (hidden[t] predicts token AT t). So target's prediction for position t
    # comes from target_logits[t-1].
    # Masked positions are always >= 1 (after anchors), so t-1 >= 0.
    target_logits_masked = target_logits[batch_idx, seq_idx - 1]  # [num_masked, V]
    positions_masked = position_in_block[batch_idx, seq_idx]  # [num_masked]

    # Compute draft logits only at masked positions (much smaller than full [B,T,V])
    draft_logits_masked = lm_head(draft_hidden_masked)  # [num_masked, V]

    # Position weights
    weights = torch.exp(-((positions_masked.float() - 1) / gamma))  # [num_masked]

    # KL(target || draft) — memory efficient on [num_masked, V]
    target_log_probs = F.log_softmax(target_logits_masked, dim=-1)
    draft_log_probs = F.log_softmax(draft_logits_masked, dim=-1)

    # Use F.kl_div for numerical stability (input=draft_log_probs, target=target_probs)
    kl = F.kl_div(draft_log_probs, target_log_probs.exp(), reduction="none").sum(dim=-1)  # [num_masked]

    # Weighted average
    loss = (kl * weights).sum() / weights.sum().clamp(min=1)
    return loss


def compute_routing_entropy(routing_weights: torch.Tensor) -> torch.Tensor:
    """Mean entropy of routing distribution (for logging and regularization)."""
    entropy = -(routing_weights * routing_weights.clamp(min=1e-8).log()).sum(dim=-1)
    return entropy.mean()


def save_checkpoint(
    model: DFlashDraftModelRouted,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    save_dir: str,
    is_best: bool = False,
):
    """Save full training state for resumption."""
    ckpt_dir = Path(save_dir) / f"checkpoint_epoch{epoch}_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    metadata = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Update latest symlink
    latest = Path(save_dir) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_dir.name)

    # Update best symlink
    if is_best:
        best = Path(save_dir) / "best"
        if best.exists() or best.is_symlink():
            best.unlink()
        best.symlink_to(ckpt_dir.name)

    logger.info(f"Saved checkpoint to {ckpt_dir} (best={is_best})")


def load_checkpoint(
    model: DFlashDraftModelRouted,
    optimizer: torch.optim.Optimizer,
    scheduler,
    save_dir: str,
) -> tuple[int, int]:
    """Load latest checkpoint for resumption. Returns (epoch, step)."""
    latest = Path(save_dir) / "latest"
    if not latest.exists():
        return 0, 0

    ckpt_dir = latest.resolve()
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", weights_only=True))
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", weights_only=True))
    scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", weights_only=True))

    with open(ckpt_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info(f"Resumed from {ckpt_dir} (epoch={metadata['epoch']}, step={metadata['step']})")
    return metadata["epoch"], metadata["step"]


def cleanup_old_checkpoints(save_dir: str, keep_last_n: int):
    """Remove old checkpoints, keeping only the N most recent + best."""
    save_path = Path(save_dir)
    checkpoints = sorted(
        [d for d in save_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: d.stat().st_mtime,
    )

    # Keep best checkpoint
    best_link = save_path / "best"
    best_name = best_link.resolve().name if best_link.exists() else None

    for ckpt in checkpoints[:-keep_last_n]:
        if ckpt.name != best_name:
            shutil.rmtree(ckpt)
            logger.info(f"Removed old checkpoint: {ckpt.name}")


def train(config: TrainConfig):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    try:
        import wandb
        wandb.init(project=config.logging.wandb_project, config=vars(config))
        use_wandb = True
    except Exception:
        use_wandb = False
        logger.warning("WandB not available, logging to stdout only")

    # Load target model (frozen, bf16, flash attention)
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

    # Load draft model from pre-trained DFlash checkpoint
    # Model kept in fp32 (master weights) — autocast handles bf16 forward pass.
    # This ensures routing scalars (log_gamma, log_tau, log_prior) get precise updates.
    logger.info(f"Loading draft model from: {config.draft_checkpoint}")
    router_config = config.router
    model = DFlashDraftModelRouted.from_pretrained_dflash(
        config.draft_checkpoint,
        router_config=router_config,
        device=device,
        dtype=torch.float32,
    )
    model.train()

    # Hidden state extractor
    target_layer_ids = model.target_layer_ids
    extractor = OnlineHiddenStateExtractor(target_model, target_layer_ids, device)

    # Dataset and dataloader (loads from Nemotron — same source as DFlash paper)
    logger.info("Loading training data from Nemotron-Post-Training-Dataset-v2...")
    dataset = DFlashDataset(tokenizer, max_seq_len=config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id or 0),
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer and scheduler
    total_steps = (
        len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # Resume from checkpoint if available
    start_epoch, start_step = load_checkpoint(
        model, optimizer, scheduler, config.checkpoint.save_dir
    )

    # Training state
    best_loss = float("inf")
    global_step = start_step

    logger.info(
        f"Starting training: {config.num_epochs} epochs, "
        f"{total_steps} total steps, "
        f"{len(dataset)} samples"
    )

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, input_ids in enumerate(dataloader):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_step % len(dataloader):
                continue

            input_ids = input_ids.to(device)

            # Extract target hidden states (no grad, bf16)
            layer_hiddens, target_logits = extractor.extract(input_ids)

            # Create masked blocks for block-diffusion training
            masked_ids, loss_mask, position_in_block, block_ids, block_anchors = create_masked_blocks(
                input_ids,
                block_size=config.block_size,
                num_anchors=config.num_anchors,
                mask_token_id=model.mask_token_id or 0,
            )

            # Build attention mask with causal consistency + inter-block isolation
            ctx_len = input_ids.shape[1]
            attn_mask = build_block_attention_mask(block_ids, block_anchors, ctx_len, dtype=torch.bfloat16)

            # Draft model forward + loss in bf16 autocast (fast), but optimizer in fp32 (precise)
            noise_embedding = target_model.model.embed_tokens(masked_ids)
            position_ids = torch.arange(
                masked_ids.shape[1], device=device
            ).unsqueeze(0).expand(masked_ids.shape[0], -1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                draft_output, routing_weights = model(
                    position_ids=position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=layer_hiddens,
                    attention_mask=attn_mask,
                    return_routing_weights=True,
                )

                # Position-weighted KL loss (only computed at masked positions to save memory)
                loss = compute_position_weighted_kl_loss(
                    draft_output,
                    target_logits,
                    loss_mask,
                    position_in_block,
                    target_model.lm_head,
                    config.loss_gamma,
                )

                # Optional entropy regularization
                if config.entropy_reg_weight > 0:
                    entropy = compute_routing_entropy(routing_weights)
                    loss = loss - config.entropy_reg_weight * entropy

            # Gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                step_loss = loss.item() * config.gradient_accumulation_steps
                epoch_loss += step_loss
                epoch_steps += 1

                if global_step % config.logging.log_every_steps == 0:
                    gamma_val = model.context_router.log_gamma.exp().item()
                    tau_val = model.context_router.log_tau.exp().item()
                    entropy_val = compute_routing_entropy(routing_weights).item()
                    mean_weights = routing_weights.mean(dim=(0, 1)).tolist()

                    log_dict = {
                        "loss": step_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "gamma": gamma_val,
                        "tau": tau_val,
                        "routing_entropy": entropy_val,
                        "epoch": epoch,
                        "step": global_step,
                    }
                    for i, w in enumerate(mean_weights):
                        log_dict[f"routing_weight_layer_{i}"] = w

                    if use_wandb:
                        wandb.log(log_dict, step=global_step)
                    else:
                        logger.info(
                            f"step={global_step} loss={step_loss:.4f} "
                            f"γ={gamma_val:.4f} τ={tau_val:.4f} "
                            f"entropy={entropy_val:.4f}"
                        )

                # Checkpointing
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

        # End of epoch checkpoint
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
