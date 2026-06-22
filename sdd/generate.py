"""
Core SDD (Self-Distillation Data) generation logic.

Generates responses from the target model using Qwen3's recommended
non-thinking sampling params (temp=0.7, top_p=0.8, top_k=20, min_p=0).
Uses vLLM for fast batched inference.
"""

import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger


def load_prompts(nemotron_splits: dict[str, int], codealpaca_count: int) -> list[dict]:
    """
    Load prompts from Nemotron V2 splits + CodeAlpaca.

    Args:
        nemotron_splits: dict mapping split name to number of samples to take
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
        if count < target_count:
            logger.warning(f"Only got {count}/{target_count} from Nemotron/{split_name}")

    if codealpaca_count > 0:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        count = 0
        for sample in ds:
            if count >= codealpaca_count:
                break
            instruction = sample.get("instruction", "")
            inp = sample.get("input", "")
            if inp and inp.strip():
                prompt = f"{instruction}\n{inp}"
            else:
                prompt = instruction
            if prompt and len(prompt.strip()) > 20:
                prompts.append({"prompt": prompt.strip(), "source": "codealpaca", "split": "code"})
                count += 1
        logger.info(f"Loaded {count} prompts from CodeAlpaca (target: {codealpaca_count})")

    logger.info(f"Total prompts loaded: {len(prompts)}")
    return prompts


def generate_shard(
    target_model_path: str,
    nemotron_splits: dict[str, int],
    codealpaca_count: int,
    output_path: str,
    max_seq_len: int = 3072,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    fresh: bool = False,
    commit_fn=None,
):
    """Generate SDD using vLLM for fast batched inference."""
    from vllm import LLM, SamplingParams

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fresh:
        prompts_cache = output_path.with_suffix(".prompts.json")
        if output_path.exists():
            output_path.unlink()
            logger.info(f"Deleted old output: {output_path}")
        if prompts_cache.exists():
            prompts_cache.unlink()
            logger.info(f"Deleted old prompts cache: {prompts_cache}")

    # Load prompts (with caching for resume)
    prompts_cache = output_path.with_suffix(".prompts.json")
    if prompts_cache.exists():
        logger.info(f"Loading cached prompt list from {prompts_cache}")
        with open(prompts_cache) as f:
            prompts = json.load(f)
    else:
        prompts = load_prompts(nemotron_splits, codealpaca_count)
        with open(prompts_cache, "w") as f:
            json.dump(prompts, f)
        logger.info(f"Saved {len(prompts)} prompts to {prompts_cache}")
        if commit_fn:
            commit_fn()

    # Resume: count existing output lines
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        logger.info(f"Resuming from {existing} existing samples")

    prompts_remaining = prompts[existing:]
    if not prompts_remaining:
        logger.info("All prompts already generated. Done.")
        return {"total": existing, "avg_response_tokens": 0}

    # Load vLLM engine — max_model_len enforces total (prompt + response) ≤ max_seq_len
    logger.info(f"Loading vLLM engine: {target_model_path} (max_seq_len={max_seq_len})")
    from transformers import AutoTokenizer as HFTokenizer
    tokenizer = HFTokenizer.from_pretrained(target_model_path)

    llm = LLM(
        model=target_model_path,
        dtype="bfloat16",
        max_model_len=max_seq_len,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_seq_len,  # vLLM auto-caps at max_model_len - prompt_len
    )

    # Generate in chunks (vLLM handles batching internally)
    chunk_size = 5000
    generated = existing
    total_response_tokens = 0

    logger.info(f"Generating {len(prompts_remaining)} responses with vLLM (temp={temperature}, thinking=off)")

    def format_prompt(text: str) -> str:
        """Apply chat template with thinking mode disabled."""
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    with open(output_path, "a") as f:
        for chunk_start in range(0, len(prompts_remaining), chunk_size):
            chunk = prompts_remaining[chunk_start : chunk_start + chunk_size]

            # Format prompts and filter out any that exceed max_model_len
            formatted = []
            chunk_filtered = []
            for p in chunk:
                text = format_prompt(p["prompt"])
                token_count = len(tokenizer.encode(text))
                if token_count < max_seq_len - 256:
                    formatted.append(text)
                    chunk_filtered.append(p)

            if not formatted:
                continue

            outputs = llm.generate(formatted, sampling_params)

            for prompt_info, output in zip(chunk_filtered, outputs):
                response = output.outputs[0].text
                response_tokens = len(output.outputs[0].token_ids)

                if len(response.strip()) >= 25:
                    record = {
                        "prompt": prompt_info["prompt"],
                        "response": response,
                        "source": prompt_info["source"],
                        "split": prompt_info["split"],
                    }
                    f.write(json.dumps(record) + "\n")
                    generated += 1
                    total_response_tokens += response_tokens

            f.flush()
            if commit_fn:
                commit_fn()
            avg_len = total_response_tokens / max(generated - existing, 1)
            logger.info(f"Chunk done. {generated} total samples, avg response: {avg_len:.0f} tokens")

    if commit_fn:
        commit_fn()

    avg_len = total_response_tokens / max(generated - existing, 1)
    logger.info(f"Done: {generated} total samples, avg response {avg_len:.0f} tokens")
    return {"total": generated, "avg_response_tokens": avg_len}
