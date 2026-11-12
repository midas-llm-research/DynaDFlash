from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
STUDY = Path("/study/dflash_dynamic_router_20261012")
BASE = Path("/base/models")
STUDY_VOLUME = modal.Volume.from_name("dflash-dynamic-router-20261012", create_if_missing=False)
BASE_VOLUME = modal.Volume.from_name("dflash-query-routing-20261007", create_if_missing=False)
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1", "transformers==5.14.1", "safetensors==0.8.0",
        "accelerate==1.15.0", "huggingface_hub==1.33.0", "numpy==2.5.4"
    )
    .add_local_dir(ROOT / "dynamic_dflash", remote_path="/workspace/dynamic_dflash")
)
app = modal.App("dynadflash")
TRAINERS = {
    "baseline": ("dynamic_dflash.training.baseline", "scratch_native_dflash_paper_lr_oneepoch_branch_18000_nothink"),
    "dynadflash": ("dynamic_dflash.training.dynadflash", "scratch_full_fc_token_gate_oneepoch_0_24750_nothink"),
}


@app.function(
    image=IMAGE,
    volumes={"/study": STUDY_VOLUME, "/base": BASE_VOLUME},
    gpu="H100:2",
    cpu=16,
    memory=131072,
    timeout=86400,
    retries=modal.Retries(max_retries=2, initial_delay=30),
    max_containers=1,
)
def train(arm: str, run_name: str) -> dict:
    if arm not in TRAINERS or not run_name.isidentifier():
        raise ValueError("invalid arm or run name")
    module, stage = TRAINERS[arm]
    output_dir = STUDY / "runs" / run_name
    latest = output_dir / stage / "latest.pt"
    resume = ["--resume", str(latest)] if latest.exists() else []
    command = [
        "torchrun", "--standalone", "--nproc_per_node=2", "-m", module,
        "--data-dir", str(STUDY / "data/prepared_nothink"),
        "--target-path", str(BASE / "target_8b"),
        "--draft-path", str(BASE / "draft_8b"),
        "--pins-report", str(STUDY / "pins_verified.json"),
        "--output-dir", str(output_dir),
        "--stop-after-update", "24750",
        "--save-interval", "250",
        "--validate-interval", "1000",
        "--anchor-limit", "64",
        "--anchor-chunk", "8",
        "--draft-learning-rate", "0.0006",
        "--dense-learning-rate", "0.0006",
        "--schedule-mode", "original",
        *resume,
    ]
    env = {**os.environ, "PYTHONPATH": "/workspace"}
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if '"checkpoint_saved"' in line:
                record = json.loads(line)
                if record.get("checkpoint_saved") is not None:
                    STUDY_VOLUME.commit()
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if not latest.exists():
        raise RuntimeError("final checkpoint missing")
    return {"arm": arm, "run_name": run_name, "checkpoint": str(latest)}


@app.function(
    image=IMAGE,
    volumes={"/study": STUDY_VOLUME},
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=7200,
)
def evaluate(output_name: str, limit: int, subset: int) -> dict:
    if not output_name.isidentifier() or limit < 1 or subset < 0:
        raise ValueError("invalid evaluation request")
    output = STUDY / "evaluations" / output_name / "paired.jsonl"
    if output.exists():
        raise RuntimeError("evaluation output already exists")
    command = [
        sys.executable, "-m", "dynamic_dflash.eval.paired",
        "--data-dir", str(STUDY / "data/prepared_nothink"),
        "--output", str(output),
        "--limit", str(limit),
        "--subset", str(subset),
    ]
    subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": "/workspace"})
    STUDY_VOLUME.commit()
    return json.loads(output.with_suffix(".summary.json").read_text())


@app.local_entrypoint()
def main(action: str, arm: str = "dynadflash", run_name: str = "", output_name: str = "", limit: int = 64, subset: int = 1):
    if action == "train":
        call = train.spawn(arm, run_name)
        print(json.dumps({"function_call_id": call.object_id}))
    elif action == "eval":
        print(json.dumps(evaluate.remote(output_name, limit, subset), indent=2))
    else:
        raise ValueError("unknown action")
