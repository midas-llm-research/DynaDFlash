"""Tests for the latency model and the offline-profiling fit (midas/latency.py).

The Modal profiling entrypoint (scripts/profile_latency.py) needs a GPU, but its
math lives in the pure, GPU-free `fit_latency_model`, which is what's tested here
along with LatencyModel's formula / serialization.
"""

import json

import torch

from midas.latency import LatencyModel, fit_latency_model


# ---------------------------------------------------------------------------
# LatencyModel formulas
# ---------------------------------------------------------------------------


class TestLatencyModelFormula:
    def test_draft_latency_scalar(self):
        m = LatencyModel(d0=0.5, d_depth=0.2, d_ctx=1e-3, d_dc=5e-5)
        # 0.5 + 0.2*3 + 1e-3*100 + 5e-5*3*100 = 0.5 + 0.6 + 0.1 + 0.015 = 1.215
        assert abs(m.draft_latency(3, 100) - 1.215) < 1e-9

    def test_verify_cost_scalar(self):
        m = LatencyModel(v0=2.0, v_ctx=3e-3)
        assert abs(m.verify_cost(500) - (2.0 + 1.5)) < 1e-9

    def test_broadcasts_over_tensors(self):
        m = LatencyModel(d0=0.0, d_depth=1.0, d_ctx=0.0, d_dc=0.0)
        depth = torch.arange(1, 5).view(4, 1, 1).float()      # [4,1,1]
        ctx = torch.tensor([[10.0, 20.0]])                    # [1,2]
        out = m.draft_latency(depth, ctx.view(1, 1, 2))       # [4,1,1] broadcast [1,1,2] -> [4,1,2]
        assert out.shape == (4, 1, 2)
        # draft_latency = depth here -> each row equals its depth, independent of ctx
        assert torch.allclose(out[:, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert torch.allclose(out[:, 0, 0], out[:, 0, 1])

    def test_default_is_unfitted(self):
        assert LatencyModel().fitted is False


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestLatencyModelIO:
    def test_from_json_none_returns_defaults(self):
        m = LatencyModel.from_json(None)
        assert m == LatencyModel()  # dataclass equality
        assert m.fitted is False

    def test_save_and_reload_roundtrip(self, tmp_path):
        m = LatencyModel(d0=1.5, d_depth=0.3, d_ctx=2e-3, d_dc=7e-5, v0=4.0, v_ctx=5e-3, fitted=True)
        path = str(tmp_path / "lat.json")
        m.save(path)
        # on-disk JSON is the plain coefficient dict
        with open(path) as f:
            raw = json.load(f)
        assert raw["d_depth"] == 0.3 and raw["fitted"] is True
        assert LatencyModel.from_json(path) == m


# ---------------------------------------------------------------------------
# fit_latency_model — recovers coefficients from measured samples
# ---------------------------------------------------------------------------


class TestFitLatencyModel:
    def _samples(self, true: LatencyModel, depths, ctxs):
        draft = [(d, c, true.draft_latency(d, c)) for d in depths for c in ctxs]
        verify = [(c, true.verify_cost(c)) for c in ctxs]
        return draft, verify

    def test_recovers_exact_linear_coefficients(self):
        true = LatencyModel(d0=0.5, d_depth=0.2, d_ctx=1e-3, d_dc=5e-5, v0=2.0, v_ctx=3e-3)
        draft, verify = self._samples(true, depths=range(1, 9), ctxs=[64, 128, 256, 512, 1024, 2048])
        fit = fit_latency_model(draft, verify)
        assert fit.fitted is True
        for attr in ("d0", "d_depth", "d_ctx", "d_dc", "v0", "v_ctx"):
            assert abs(getattr(fit, attr) - getattr(true, attr)) < 1e-6, attr

    def test_fit_predictions_match_truth(self):
        true = LatencyModel(d0=1.0, d_depth=0.4, d_ctx=2e-3, d_dc=1e-4, v0=3.0, v_ctx=4e-3)
        draft, verify = self._samples(true, depths=range(1, 7), ctxs=[100, 500, 2000])
        fit = fit_latency_model(draft, verify)
        # predictions agree at held-out-ish points
        assert abs(fit.draft_latency(5, 1500) - true.draft_latency(5, 1500)) < 1e-4
        assert abs(fit.verify_cost(1500) - true.verify_cost(1500)) < 1e-4

    def test_least_squares_on_noisy_samples(self):
        torch.manual_seed(0)
        true = LatencyModel(d0=0.5, d_depth=0.2, d_ctx=1e-3, d_dc=5e-5, v0=2.0, v_ctx=3e-3)
        depths, ctxs = list(range(1, 9)), [64, 128, 256, 512, 1024, 2048]
        draft, verify = self._samples(true, depths, ctxs)
        # add small zero-mean noise; least squares should stay close to truth
        draft = [(d, c, t + float(torch.randn(1)) * 1e-3) for (d, c, t) in draft]
        fit = fit_latency_model(draft, verify)
        assert abs(fit.d_depth - true.d_depth) < 0.05
        assert abs(fit.v_ctx - true.v_ctx) < 1e-3
