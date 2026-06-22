"""
Core SDD (Self-Distillation Data) generation logic.

Generates responses from the target model at temperature=0.6,
matching the DFlash/DFlare training recipe exactly.
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(nemotron_splits: dict[str, int], codealpaca_count: int) -> list[dict]:
    """
    Load prompts from Nemotron V2 splits + CodeAlpaca.

    Args:
        nemotron_splits: dict mapping split name to number of samples to take
                         e.g. {"math": 195000, "code": 175000}
        codealpaca_count: number of CodeAlpaca samples
    """
    prompts = []

    for split_name, target_count in nemotron_splits.items():
        ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=split_name, streaming=True)
        count = 0
        for sample in ds:
            if count >= target_count:
                break
            messages = sample.get("messages", [])
            user_msg = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if user_msg and len(user_msg.strip()) > 20:
                prompts.append({"prompt": user_msg.strip(), "source": "nemotron", "split": split_name})
                count += 1
        logger.info(f"Loaded {count} prompts from Nemotron/{split_name} (target: {target_count})")

    if codealpaca_count > 0:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        count = 0
        for sample in ds:
            if count >= codealpaca_count:
                break
            instruction = sample.get("instruction", "")
            if instruction and len(instruction.strip()) > 20:
                prompts.append({"prompt": instruction.strip(), "source": "codealpaca", "split": "code"})
                count += 1
        logger.info(f"Loaded {count} prompts from CodeAlpaca")

    logger.info(f"Total prompts: {len(prompts)}")
    return prompts


def generate_shard(
    target_model_path: str,
    tokenizer_path: str,
    nemotron_splits: dict[str, int],
    codealpaca_count: int,
    output_path: str,
    batch_size: int = 64,
    max_new_tokens: int = 2048,
    max_prompt_tokens: int = 1024,
    temperature: float = 0.6,
    commit_fn=None,
):
    """Generate SDD for one shard (one GPU)."""
    device = torch.device("cuda")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading target model from {target_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        target_model_path, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load prompts
    prompts = load_prompts(nemotron_splits, codealpaca_count)

    # Resume: count existing output
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        logger.info(f"Resuming from {existing} existing samples")

    prompts_remaining = prompts[existing:]
    logger.info(f"Generating {len(prompts_remaining)} responses (batch={batch_size}, temp={temperature})")

    generated = existing
    total_response_tokens = 0

    with open(output_path, "a") as f:
        for i in tqdm(range(0, len(prompts_remaining), batch_size), desc="Generating"):
            batch = prompts_remaining[i : i + batch_size]
            batch_prompts = [p["prompt"] for p in batch]

            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j, (prompt_info, output_ids) in enumerate(zip(batch, outputs)):
                prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
                response_ids = output_ids[prompt_len:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)

                if len(response.strip()) > 5:
                    record = {
                        "prompt": prompt_info["prompt"],
                        "response": response,
                        "source": prompt_info["source"],
                        "split": prompt_info["split"],
                    }
                    f.write(json.dumps(record) + "\n")
                    generated += 1
                    total_response_tokens += len(response_ids)

            # Periodic commit
            if commit_fn and (i // batch_size) % 50 == 0 and i > 0:
                f.flush()
                commit_fn()
                avg_len = total_response_tokens / max(generated - existing, 1)
                logger.info(f"Committed. {generated} total, avg response: {avg_len:.0f} tokens")

    if commit_fn:
        commit_fn()

    avg_len = total_response_tokens / max(generated - existing, 1)
    logger.info(f"Shard complete: {generated} samples, avg response {avg_len:.0f} tokens")
    return {"total": generated, "avg_response_tokens": avg_len}
