"""
Cheap closed-form latency model for the layer-skip speedup objective.

Measuring per-block draft/verify latency during (batched) training is infeasible,
so we measure it ONCE offline at batch=1 (the real inference shape) over a sweep
of (draft depth, context length), fit a linear regression, and evaluate that
closed form during training to score exit depths.

    draft_latency(depth, ctx) = d0 + d_depth*depth + d_ctx*ctx + d_dc*depth*ctx
    verify_cost(ctx)          = v0 + v_ctx*ctx

`depth` = number of draft decoder layers actually run; `ctx` = context length the
draft/target attends over. The interaction term captures attention growing with
the KV it reads (per-layer). Both methods accept python floats or torch tensors
and broadcast, so the training-time scoring is fully vectorized.

Fit the coefficients with `scripts/profile_latency.py`, which writes the JSON this
loads. Units are arbitrary (whatever the timer used) — only the *ratio* between
draft and verify cost matters for the exit labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LatencyModel:
    # draft_latency coefficients
    d0: float = 0.05
    d_depth: float = 0.05
    d_ctx: float = 0.0
    d_dc: float = 1.0e-4
    # verify_cost coefficients
    v0: float = 1.0
    v_ctx: float = 1.0e-3
    fitted: bool = False  # False => placeholder defaults; run profile_latency.py

    def draft_latency(self, depth, ctx):
        """depth, ctx: float or broadcastable torch tensors -> latency (same shape)."""
        return self.d0 + self.d_depth * depth + self.d_ctx * ctx + self.d_dc * depth * ctx

    def verify_cost(self, ctx):
        return self.v0 + self.v_ctx * ctx

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | None) -> "LatencyModel":
        if path is None:
            return cls()
        with open(path) as f:
            return cls(**json.load(f))


def fit_latency_model(draft_samples, verify_samples) -> "LatencyModel":
    """Least-squares fit a LatencyModel from measured timings (pure / GPU-free).

    draft_samples:  iterable of (depth, ctx, latency) -> fits draft_latency
    verify_samples: iterable of (ctx, latency)        -> fits verify_cost

    Design matrices: draft rows are [1, depth, ctx, depth*ctx]; verify rows are
    [1, ctx]. Returns a LatencyModel with fitted=True.
    """
    import numpy as np

    dX = np.array([[1.0, d, c, d * c] for (d, c, _t) in draft_samples], dtype=float)  # [Nd, 4]
    dy = np.array([t for (_d, _c, t) in draft_samples], dtype=float)                  # [Nd]
    vX = np.array([[1.0, c] for (c, _t) in verify_samples], dtype=float)              # [Nv, 2]
    vy = np.array([t for (_c, t) in verify_samples], dtype=float)                     # [Nv]
    dcoef, *_ = np.linalg.lstsq(dX, dy, rcond=None)   # [4] = [d0, d_depth, d_ctx, d_dc]
    vcoef, *_ = np.linalg.lstsq(vX, vy, rcond=None)   # [2] = [v0, v_ctx]
    return LatencyModel(
        d0=float(dcoef[0]), d_depth=float(dcoef[1]), d_ctx=float(dcoef[2]), d_dc=float(dcoef[3]),
        v0=float(vcoef[0]), v_ctx=float(vcoef[1]), fitted=True,
    )
