"""
Train baseline DFlash (original FC architecture) from random init.
Same training setup as the router model for fair comparison.

Usage:
    modal run scripts/modal_train_baseline.py
"""

import modal
from pathlib import Path

app = modal.App("midas-baseline-train")

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
        "pyyaml",
        "numpy",
        "tqdm",
        "loguru",
        "huggingface_hub",
    )
    .add_local_dir(str(PROJECT_ROOT / "midas"), "/root/midas", copy=True)
    .add_local_dir(str(PROJECT_ROOT / "baseline"), "/root/baseline", copy=True)
    .add_local_dir(str(PROJECT_ROOT / "configs"), "/root/configs", copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/models": model_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=86400,
)
def train(fresh: bool = True):
    import json
    import shutil
    import sys

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from loguru import logger

    sys.path.insert(0, "/root")
    from midas.data import (
        SDDDataset, OnlineHiddenStateExtractor,
        build_block_attention_mask, sdd_collate_fn, create_masked_blocks,
    )
    from baseline.model import DFlashDraftModel

    device = torch.device("cuda")
    save_dir = "/checkpoints/midas-baseline"

    # === Config ===
    LR = 6e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 1
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
    MAX_SEQ_LEN = 3072
    BLOCK_SIZE = 16
    NUM_ANCHORS = 128
    WARMUP_RATIO = 0.04
    GRAD_CLIP = 1.0
    LOSS_GAMMA = 7.0
    TARGET_TEMPERATURE = 1.0
    SAVE_EVERY = 500
    LOG_EVERY = 50
    SDD_PATH = "/checkpoints/sdd/sdd_100k.jsonl"
    KL_CHUNK_SIZE = 1024

    # Clear old checkpoints if fresh
    if fresh:
        ckpt_dir = Path(save_dir)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f"Cleared {ckpt_dir}")

    # === Load target model (frozen, bf16) ===
    logger.info("Loading target: Qwen3-4B")
    target_model = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B",
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()
    for p in target_model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained("/models/Qwen3-4B")

    # === Create baseline DFlash with RANDOM weights (fp32 master) ===
    logger.info("Creating baseline DFlash with random init")
    model = DFlashDraftModel.from_config_random(
        "/models/Qwen3-4B-DFlash-b16",
        device=device,
        dtype=torch.float32,
    )
    model.train()

    # === Data (from pre-generated SDD) ===
    logger.info(f"Loading SDD data from {SDD_PATH}...")
    extractor = OnlineHiddenStateExtractor(target_model, model.target_layer_ids, device)
    dataset = SDDDataset(tokenizer, sdd_path=SDD_PATH, max_seq_len=MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: sdd_collate_fn(batch, tokenizer.pad_token_id or 0),
        num_workers=4, pin_memory=True,
    )

    # === Optimizer ===
    total_steps = len(dataloader) * NUM_EPOCHS // GRAD_ACCUM
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps,
    )

    # === Resume ===
    start_step = 0
    if not fresh:
        latest = Path(save_dir) / "latest"
        if latest.exists():
            ckpt = latest.resolve()
            model.load_state_dict(torch.load(ckpt / "model.pt", weights_only=True))
            optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", weights_only=True))
            scheduler.load_state_dict(torch.load(ckpt / "scheduler.pt", weights_only=True))
            with open(ckpt / "metadata.json") as f:
                start_step = json.load(f)["step"]
            logger.info(f"Resumed from step {start_step}")

    logger.info(f"Training: {total_steps} steps, {len(dataset)} samples, batch={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")

    # === Training Loop ===
    global_step = start_step
    epoch_loss = 0.0
    epoch_steps = 0
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (input_ids, response_starts) in enumerate(dataloader):
            if batch_idx < start_step * GRAD_ACCUM:
                continue

            input_ids = input_ids.to(device)
            response_starts = response_starts.to(device)

            # Target forward (frozen, no grad)
            layer_hiddens, target_logits = extractor.extract(input_ids)

            # Masked blocks (anchors only in response portion) + attention mask
            masked_ids, loss_mask, position_in_block, block_ids, block_anchors = create_masked_blocks(
                input_ids, BLOCK_SIZE, NUM_ANCHORS, model.mask_token_id or 0,
                response_starts=response_starts,
            )
            ctx_len = input_ids.shape[1]
            attn_mask = build_block_attention_mask(block_ids, block_anchors, ctx_len, dtype=torch.bfloat16)

            # Draft forward (autocast bf16)
            noise_embedding = target_model.model.embed_tokens(masked_ids)
            position_ids = torch.arange(masked_ids.shape[1], device=device).unsqueeze(0).expand(input_ids.shape[0], -1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                draft_output = model(
                    position_ids=position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=layer_hiddens,
                    attention_mask=attn_mask,
                )

                # === Chunked KL loss (prevents OOM) ===
                masked_indices = loss_mask.nonzero(as_tuple=False)
                if masked_indices.shape[0] > 0:
                    batch_idx_m = masked_indices[:, 0]
                    seq_idx = masked_indices[:, 1]

                    draft_hidden_masked = draft_output[batch_idx_m, seq_idx]
                    target_logits_masked = target_logits[batch_idx_m, seq_idx - 1]
                    positions_masked = position_in_block[batch_idx_m, seq_idx]
                    weights = torch.exp(-((positions_masked.float() - 1) / LOSS_GAMMA))

                    num_masked = masked_indices.shape[0]
                    total_loss = torch.tensor(0.0, device=device)
                    total_weight = torch.tensor(0.0, device=device)

                    for chunk_start in range(0, num_masked, KL_CHUNK_SIZE):
                        chunk_end = min(chunk_start + KL_CHUNK_SIZE, num_masked)
                        chunk_draft_h = draft_hidden_masked[chunk_start:chunk_end]
                        chunk_target_l = target_logits_masked[chunk_start:chunk_end]
                        chunk_w = weights[chunk_start:chunk_end]

                        chunk_draft_logits = target_model.lm_head(chunk_draft_h)

                        t_log = F.log_softmax(chunk_target_l / TARGET_TEMPERATURE, dim=-1)
                        d_log = F.log_softmax(chunk_draft_logits / TARGET_TEMPERATURE, dim=-1)
                        kl = F.kl_div(d_log, t_log.exp(), reduction="none").sum(dim=-1)

                        total_loss = total_loss + (kl * chunk_w).sum()
                        total_weight = total_weight + chunk_w.sum()

                    loss = total_loss / total_weight.clamp(min=1)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Backward + step
            (loss / GRAD_ACCUM).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = loss.item()
                epoch_loss += step_loss
                epoch_steps += 1

                if global_step % LOG_EVERY == 0:
                    logger.info(f"step={global_step} loss={step_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f}")

                if global_step % SAVE_EVERY == 0:
                    avg_loss = epoch_loss / epoch_steps
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss

                    ckpt_dir = Path(save_dir) / f"checkpoint_step{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), ckpt_dir / "model.pt")
                    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
                    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
                    with open(ckpt_dir / "metadata.json", "w") as f:
                        json.dump({"step": global_step, "loss": avg_loss}, f)
                    for link_name in ["latest"] + (["best"] if is_best else []):
                        link = Path(save_dir) / link_name
                        if link.exists() or link.is_symlink():
                            link.unlink()
                        link.symlink_to(ckpt_dir.name)
                    checkpoint_volume.commit()
                    logger.info(f"Saved step={global_step} loss={avg_loss:.4f} best={is_best}")

    # Final checkpoint
    avg_loss = epoch_loss / max(epoch_steps, 1)
    is_best = avg_loss < best_loss
    ckpt_dir = Path(save_dir) / f"checkpoint_step{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump({"step": global_step, "loss": avg_loss}, f)
    for link_name in ["latest"] + (["best"] if is_best else []):
        link = Path(save_dir) / link_name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(ckpt_dir.name)
    checkpoint_volume.commit()
    logger.info(f"Training complete. Final loss: {avg_loss:.4f}, best: {best_loss:.4f}")


@app.local_entrypoint()
def main():
    call = train.spawn(fresh=True)
    print(f"Baseline training spawned: {call.object_id}")
    print("Monitor: https://modal.com/apps/midas-llm-research/main/")
