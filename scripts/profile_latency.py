"""
Offline latency profiling for the layer-skip speedup objective (run on the same
GPU you train/eval on). Sweeps draft depth x context length and target verify x
context length at batch=1 (the real inference shape), fits the linear
`midas.latency.LatencyModel`, and writes its coefficients to JSON.

    modal run scripts/profile_latency.py --out /checkpoints/latency_h100.json

Then point training at it:  SkipConfig(latency_path="/checkpoints/latency_h100.json").

The absolute units don't matter (only the draft/verify ratio drives the exit
labels), but profiling on the deployment GPU keeps that ratio realistic.
"""

import modal
from pathlib import Path

app = modal.App("midas-profile-latency")

checkpoint_volume = modal.Volume.from_name("midas-checkpoints")
model_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
PROJECT_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2.0", "transformers>=4.57.0", "accelerate", "numpy", "loguru")
    .add_local_dir(str(PROJECT_ROOT / "midas"), "/root/midas", copy=True)
    .env({"PYTHONPATH": "/root"})
)


@app.function(image=image, gpu="H100",
              volumes={"/models": model_volume, "/checkpoints": checkpoint_volume},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def profile(out: str = "/checkpoints/latency.json", repeats: int = 30, num_layers: int = None):
    import sys
    import torch
    from loguru import logger
    from transformers import AutoModelForCausalLM, DynamicCache

    sys.path.insert(0, "/root")
    from midas.latency import fit_latency_model
    from midas.models.model import DFlashDraftModel

    device = torch.device("cuda")
    dtype = torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(
        "/models/Qwen3-4B", dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    # num_layers overrides the draft depth (everything else from the checkpoint config),
    # so the latency fit covers the draft architecture you'll actually train/deploy.
    overrides = {"num_hidden_layers": num_layers} if num_layers is not None else None
    draft = DFlashDraftModel.from_config_random(
        "/models/Qwen3-4B-DFlash-b16", device=device, dtype=dtype, config_overrides=overrides,
    ).eval()
    logger.info(f"Profiling draft with {len(draft.layers)} layers (num_layers override={num_layers})")

    H = draft.config.hidden_size
    block_size = draft.block_size
    n_layers = len(draft.layers)
    CTXS = [64, 128, 256, 512, 1024, 2048]
    DEPTHS = list(range(1, n_layers + 1))

    def _time(fn) -> float:
        for _ in range(5):  # warmup
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(repeats):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / repeats  # ms

    @torch.inference_mode()
    def draft_step(depth: int, ctx: int) -> float:
        # Legend: H=hidden_size, bs=block_size, P=ctx+bs (the position span).
        fused = torch.randn(1, ctx, H, device=device, dtype=dtype)            # [1, ctx, H] context
        noise = torch.randn(1, block_size, H, device=device, dtype=dtype)     # [1, bs, H] block
        pos_ids = torch.arange(ctx + block_size, device=device).unsqueeze(0)  # [1, P]
        pos_emb = draft.rotary_emb(noise, pos_ids)                            # cos,sin: [1, P, head_dim]

        def run():
            cache = DynamicCache()
            hidden = noise                                                    # [1, bs, H]
            for layer in draft.layers[:depth]:
                hidden = layer(hidden_states=hidden, target_hidden=fused,     # [1, bs, H] -> [1, bs, H]
                               attention_mask=None, position_ids=pos_ids,
                               past_key_value=cache, use_cache=True,
                               position_embeddings=pos_emb)
        return _time(run)

    @torch.inference_mode()
    def verify_step(ctx: int) -> float:
        # Prefill the target cache to `ctx`, then time one block_size verification step.
        prefill = torch.randint(0, 1000, (1, ctx), device=device)            # [1, ctx] tokens
        block = torch.randint(0, 1000, (1, block_size), device=device)       # [1, bs] tokens

        def run():
            cache = DynamicCache()
            target(prefill, past_key_values=cache, use_cache=True)           # builds [.., ctx, ..] cache
            target(block, past_key_values=cache, use_cache=True)             # verifies bs over ctx context
        # Subtract the prefill cost so we isolate the per-block verify cost.
        full = _time(run)

        def prefill_only():
            cache = DynamicCache()
            target(prefill, past_key_values=cache, use_cache=True)
        return max(full - _time(prefill_only), 0.0)

    # ---- collect samples: (depth, ctx, ms) and (ctx, ms) ----
    draft_samples = []
    for ctx in CTXS:
        for depth in DEPTHS:
            t = draft_step(depth, ctx)
            draft_samples.append((depth, ctx, t))
            logger.info(f"draft depth={depth} ctx={ctx}: {t:.4f} ms")
    verify_samples = []
    for ctx in CTXS:
        t = verify_step(ctx)
        verify_samples.append((ctx, t))
        logger.info(f"verify ctx={ctx}: {t:.4f} ms")

    # ---- least-squares fit (pure helper, also unit-tested) ----
    model = fit_latency_model(draft_samples, verify_samples)
    model.save(out)
    checkpoint_volume.commit()
    logger.info(f"Saved fitted LatencyModel -> {out}: {model}")
    return out


@app.local_entrypoint()
def main(out: str = "/checkpoints/latency.json", repeats: int = 30, num_layers: int = None):
    print(f"Fitted latency model written to: "
          f"{profile.remote(out=out, repeats=repeats, num_layers=num_layers)}")
