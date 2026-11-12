from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from dynamic_dflash.core.full_fc_token_gate import FullFCTokenGateDFlash
from dynamic_dflash.core.model import RouterConfig
from dynamic_dflash.core.scratch_dflash import ScratchDFlash
from dynamic_dflash.eval.acceptance import score_context
from dynamic_dflash.eval.domains import domain_for_source
from dynamic_dflash.eval.split import stratified_subset_indices
from dynamic_dflash.training.common import IndexedJsonl, _validation_context_points


REVISIONS = {
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "midas-llm-research/qwen3-8b-dflash-baseline": "782441988763135ed30451bda07d2171114f103c",
    "midas-llm-research/qwen3-8b-dynadflash": "c413051be0e3d3db982310c4cb4257cc6aa7f3bd",
}


def local_path(value: str) -> Path:
    path = Path(value)
    return path if path.exists() else Path(snapshot_download(value, revision=REVISIONS.get(value)))


def load_drafter(value: str, dynamic: bool, device: torch.device):
    path = local_path(value)
    draft = AutoModel.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    config = RouterConfig(hidden_size=int(draft.config.hidden_size))
    model = FullFCTokenGateDFlash(draft, config) if dynamic else ScratchDFlash(draft, config)
    if dynamic:
        state = load_file(str(path / "router.safetensors"), device="cpu")
        if set(state) != {"gate.weight"}:
            raise RuntimeError("unexpected DynaDFlash router weights")
        model.projector.gate.load_state_dict({"weight": state["gate.weight"]}, strict=True)
    return model.eval().requires_grad_(False)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires a CUDA GPU")
    device = torch.device("cuda:0")
    target_path = local_path(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        target_path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).eval().to(device)
    target.requires_grad_(False)
    baseline = load_drafter(args.baseline, False, device)
    dynamic = load_drafter(args.dynamic, True, device)
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    validation = IndexedJsonl(args.data_dir / "validation.jsonl")
    rows = []
    try:
        indices = stratified_subset_indices(
            validation, manifest["validation_source_counts"], args.limit, args.subset
        )
        for index in indices:
            record = validation[index]
            for phase, prefix_length in _validation_context_points(record):
                baseline_score = score_context(
                    target, baseline, record["input_ids"], prefix_length, eos
                )
                dynamic_score = score_context(
                    target, dynamic, record["input_ids"], prefix_length, eos
                )
                rows.append({
                    "index": index,
                    "prompt_hash": record["prompt_hash"],
                    "domain": domain_for_source(record["source"]),
                    "phase": phase,
                    "prefix_length": prefix_length,
                    "baseline_accepted_length": baseline_score["accepted_length"],
                    "dynadflash_accepted_length": dynamic_score["accepted_length"],
                    "paired_delta": dynamic_score["accepted_length"] - baseline_score["accepted_length"],
                })
    finally:
        validation.close()
    if len(rows) != 2 * len(indices):
        raise RuntimeError("expected two contexts per prompt")
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_hash"]].append(row["paired_delta"])
    if len(by_prompt) != len(indices) or any(len(values) != 2 for values in by_prompt.values()):
        raise RuntimeError("paired prompt structure differs")
    prompt_deltas = [mean(values) for values in by_prompt.values()]
    rng = random.Random(12012)
    bootstrap = sorted(mean(rng.choices(prompt_deltas, k=len(prompt_deltas))) for _ in range(5000))
    groups = {domain: [row for row in rows if row["domain"] == domain]
              for domain in ("math", "code", "chat")}
    groups["overall"] = rows
    summary = {
        "baseline": args.baseline,
        "dynamic": args.dynamic,
        "target": args.target,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "subset": args.subset,
        "prompt_count": len(indices),
        "context_count": len(rows),
        "by_domain": {
            domain: {
                "context_count": len(items),
                "baseline_mean_accepted_length": mean([row["baseline_accepted_length"] for row in items]),
                "dynadflash_mean_accepted_length": mean([row["dynadflash_accepted_length"] for row in items]),
                "paired_delta": mean([row["paired_delta"] for row in items]),
            }
            for domain, items in groups.items() if items
        },
        "prompt_bootstrap_95_percent": [bootstrap[124], bootstrap[4874]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen3-8B")
    parser.add_argument("--baseline", default="midas-llm-research/qwen3-8b-dflash-baseline")
    parser.add_argument("--dynamic", default="midas-llm-research/qwen3-8b-dynadflash")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--subset", type=int, default=1)
    print(json.dumps(evaluate(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
