"""
Train DFlash with dynamic router (our architecture) from random init.
Identical setup to baseline for fair comparison — only fusion differs.

Usage:
    modal run scripts/modal_train_router.py
"""

import modal
from pathlib import Path

app = modal.App("midas-router-train")

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
        build_concat_block_attention_mask, sdd_collate_fn, extract_blocks,
    )
    from midas.config import RouterConfig
    from midas.model import DFlashDraftModelRouted

    device = torch.device("cuda")
    save_dir = "/checkpoints/midas-router-v2"

    # === Config (matching DFlash paper exactly) ===
    LR = 6e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
    MAX_SEQ_LEN = 3072          # DFlash paper: 3072
    BLOCK_SIZE = 16             # DFlash paper: block size 16
    NUM_ANCHORS = 128
    WARMUP_RATIO = 0.04         # DFlash paper: 4% warmup
    GRAD_CLIP = 1.0             # DFlash paper: gradient clip 1.0
    LOSS_GAMMA = 7.0            # DFlash paper: gamma=7 for block_size=16
    SAVE_EVERY = 500
    LOG_EVERY = 50
    SDD_PATH = "/checkpoints/sdd/sdd_100k.jsonl"

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

    # === Create Router DFlash with RANDOM weights (fp32 master) ===
    logger.info("Creating Router DFlash with random init")
    router_config = RouterConfig(hidden_size=2560, num_layers=5, rank=128, gamma_init=1.0)
    model = DFlashDraftModelRouted.from_pretrained_dflash(
        "/models/Qwen3-4B-DFlash-b16",
        router_config=router_config,
        device=device,
        dtype=torch.float32,
    )
    # Re-randomize ALL weights (decoder layers too) for fair comparison
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.xavier_normal_(param)
        elif param.dim() == 1 and "norm" in name:
            torch.nn.init.ones_(param)
        else:
            torch.nn.init.zeros_(param)
    # Re-init router specifically
    model.context_router._init_weights()
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

            # Target forward (frozen, no grad) — need hidden states for context injection
            layer_hiddens, _ = extractor.extract(input_ids)

            # Extract independent blocks (no overlap waste)
            block_token_ids, block_targets, block_pos_ids, anchor_pos, pos_in_block, block_keep_mask = extract_blocks(
                input_ids, BLOCK_SIZE, NUM_ANCHORS, model.mask_token_id or 0,
                response_starts=response_starts,
            )

            # Attention mask for concatenated blocks
            ctx_len = input_ids.shape[1]
            attn_mask = build_concat_block_attention_mask(
                anchor_pos, NUM_ANCHORS, BLOCK_SIZE, ctx_len, dtype=torch.bfloat16
            )

            # Draft forward on concatenated blocks (autocast bf16)
            noise_embedding = target_model.model.embed_tokens(block_token_ids)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                draft_output = model(
                    position_ids=block_pos_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=layer_hiddens,
                    attention_mask=attn_mask,
                    use_cache=False,
                )

                # CE loss on masked positions (positions 1..block_size-1 in valid blocks)
                keep_expanded = block_keep_mask.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).reshape(input_ids.shape[0], -1)
                loss_positions = (pos_in_block > 0) & keep_expanded  # [B, N*block_size]
                if loss_positions.any():
                    draft_hidden_masked = draft_output[loss_positions]
                    target_tokens = block_targets[loss_positions]
                    positions = pos_in_block[loss_positions]
                    weights = torch.exp(-((positions.float() - 1) / LOSS_GAMMA))

                    # Chunked CE
                    num_loss = draft_hidden_masked.shape[0]
                    CE_CHUNK = 4096
                    total_loss = torch.tensor(0.0, device=device)
                    total_weight = torch.tensor(0.0, device=device)

                    for cs in range(0, num_loss, CE_CHUNK):
                        ce_end = min(cs + CE_CHUNK, num_loss)
                        chunk_logits = target_model.lm_head(draft_hidden_masked[cs:ce_end])
                        chunk_ce = F.cross_entropy(chunk_logits, target_tokens[cs:ce_end], reduction="none")
                        chunk_w = weights[cs:ce_end]
                        total_loss = total_loss + (chunk_ce * chunk_w).sum()
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
