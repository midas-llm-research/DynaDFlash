"""
DFlash draft model with learned per-layer EARLY EXIT (layer skip).

Idea
----
The draft runs its decoder layers top to bottom. After each layer we collapse the
block's per-token hidden states into one vector and a per-layer linear gate scores
"is this depth already good enough to stop?". At inference we exit at the first
gate that fires, saving the deeper layers' block compute; the skipped layers still
get their context K/V backfilled (a cheap projection of `target_hidden`) so future,
deeper blocks attend over a complete cache.

Training (warm-started from a trained baseline draft)
-----------------------------------------------------
1. Multi-layer CE: decode EVERY layer's block hidden through the shared lm_head and
   apply the position-weighted CE, so every depth is lm-head-decodable (the thing
   that makes exiting early actually produce usable tokens).
2. Gate labels from a latency-vs-acceptance speedup score. Per block, per layer l:
     accept(l)  = greedy acceptance length (leading prefix where argmax==gold)
     score(l)   = accept(l) / (draft_latency(l, ctx) + verify_cost(ctx))
   l* = argmax_l score(l); label layers < l* as 0 (keep going) and >= l* as 1 (exit).
   The gates are trained with BCE against these labels (gate input detached, so the
   gate/aggregator learn on fixed features without disturbing the backbone).

Block aggregation has two modes (SkipConfig.aggregation):
  - EXP_DECAY:  fixed weights ~ exp(-(pos-1)/gamma) over the DRAFTED positions
    (the anchor at position 0 is excluded), earlier drafted tokens weighted more.
  - ATTENTION:  per-layer attention pool — query = the block's most-recent context
    hidden, keys = W_k(block tokens); weights are recomputed at each layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from midas.config import BlockAggregationType, SkipConfig
from midas.latency import LatencyModel
from midas.models.model import DFlashDraftModel, compute_position_weighted_ce_loss


# ---------------------------------------------------------------------------
# Block aggregation
# ---------------------------------------------------------------------------


class BlockAggregator(nn.Module):
    """Collapse a block hidden `[..., block_size, H]` to `[..., H]` via a weighting
    over the block positions. Works for a single block (inference) or a batch of
    blocks `[B, num_anchors, block_size, H]` (training) — it only touches the last
    two dims.

    The weighting is recomputed PER LAYER from that layer's hidden (block-token
    importance can shift with depth). For ATTENTION mode each layer has its own
    `W_q`/`W_k`; EXP_DECAY is depth-independent (fixed weights)."""

    def __init__(self, skip_config: SkipConfig, hidden_size: int, num_layers: int):
        super().__init__()
        self.mode = BlockAggregationType(skip_config.aggregation)
        self.gamma = skip_config.gamma
        if self.mode is BlockAggregationType.ATTENTION:
            self.rank = skip_config.rank
            # Per-layer projections (cheap: H*r each, ~0.3% of a decoder layer's MACs).
            self.W_q = nn.ModuleList(
                [nn.Linear(hidden_size, self.rank, bias=False) for _ in range(num_layers)]
            )
            self.W_k = nn.ModuleList(
                [nn.Linear(hidden_size, self.rank, bias=False) for _ in range(num_layers)]
            )

    def weights(self, block: torch.Tensor, query: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Aggregation weights `[..., block_size]` from layer `layer_idx`'s block hidden.

        Legend: `*`=leading dims (B inference, or B,A in training), bs=block_size,
        H=hidden_size, r=rank.
          block: [*, bs, H]   this layer's per-block-token hidden states (the keys)
          query: [*, H]       this block's most-recent target hidden state (the query)
        EXP_DECAY ignores `query` (fixed, depth-independent weights)."""
        bs = block.shape[-2]
        if self.mode is BlockAggregationType.EXP_DECAY:
            # Weight DRAFTED positions only (exclude the anchor at index 0, which is
            # the non-drafted seed token); earlier drafts weighted more: w_k ~ exp(-(k-1)/gamma).
            k = torch.arange(bs, device=block.device, dtype=torch.float32)    # [bs] = 0..bs-1
            w = torch.exp(-(k - 1) / self.gamma)                              # [bs]
            w[0] = 0.0                                                        # drop the anchor
            return (w / w.sum().clamp(min=1e-9)).to(block.dtype)             # normalize over drafts -> [bs]
        # attention pool: query = block's most-recent target hidden, keys = block tokens
        q = self.W_q[layer_idx](query)                        # [*,H] -> [*,r]
        k = self.W_k[layer_idx](block)                        # [*,bs,H] -> [*,bs,r]
        scores = (q.unsqueeze(-2) * k).sum(-1) / (self.rank ** 0.5)  # [*,1,r]*[*,bs,r] -> sum over r -> [*,bs]
        return torch.softmax(scores, dim=-1)                  # [*,bs]

    @staticmethod
    def aggregate(block_hidden: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted sum over the block_size axis.
        block_hidden: [*, bs, H]; weights: [bs] or [*, bs] -> [*, H]."""
        # weights.unsqueeze(-1): [bs]->[bs,1] or [*,bs]->[*,bs,1], broadcasts over H;
        # product [*,bs,H], sum over bs (dim -2) -> [*,H]
        return (block_hidden * weights.unsqueeze(-1)).sum(-2)


# ---------------------------------------------------------------------------
# Early-exit draft model
# ---------------------------------------------------------------------------


class DFlashDraftModelSkip(DFlashDraftModel):
    """DFlash draft with per-layer early-exit gates. Keeps the standard fc+hidden
    fusion; adds an aggregator + per-layer gates and overrides forward/compute_loss."""

    def __init__(self, config, skip_config: SkipConfig):
        super().__init__(config)
        self.skip_config = skip_config
        n_layers = config.num_hidden_layers
        self.aggregator = BlockAggregator(skip_config, config.hidden_size, n_layers)
        self.gates = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for _ in range(n_layers)]
        )
        # Init gates to "never exit" so a warm-started model behaves exactly like the
        # dense baseline until the gates learn otherwise (preserves acceptance).
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, -10.0)

        self.latency_model = LatencyModel.from_json(skip_config.latency_path)
        if not self.latency_model.fitted:
            logger.warning(
                "DFlashDraftModelSkip: latency model is UNFITTED (placeholder "
                "coefficients) — gate exit labels will be computed from fictitious "
                "latency. Run scripts/profile_latency.py and set "
                "SkipConfig(latency_path=...) before trusting trained gates."
            )
        self.enable_early_exit = True          # set False to force dense (e.g. baseline eval)
        self.exit_threshold = skip_config.exit_threshold

        # Stashed by forward for compute_loss / logging (mirrors the router pattern).
        self._layer_hiddens: list[torch.Tensor] | None = None
        self._fused: torch.Tensor | None = None  # fused target hidden, for the gate query
        self.last_exit_depth: int | None = None
        self._last_metrics: dict = {}

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        position_ids,
        attention_mask=None,
        noise_embedding=None,
        target_hidden=None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ):
        # Training (and forced-dense eval) run every layer; inference early-exits.
        if self.training or not self.enable_early_exit:
            return self._forward_all_layers(
                position_ids, attention_mask, noise_embedding,
                target_hidden, past_key_values, use_cache, **kwargs,
            )
        return self._forward_early_exit(
            position_ids, attention_mask, noise_embedding,
            target_hidden, past_key_values, use_cache, **kwargs,
        )

    def _run_layer(self, layer, hidden_states, fused, attention_mask, position_ids,
                   past_key_values, use_cache, pos_emb, **kwargs):
        return layer(
            hidden_states=hidden_states,
            target_hidden=fused,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            position_embeddings=pos_emb,
            **kwargs,
        )

    def _forward_all_layers(self, position_ids, attention_mask, noise_embedding,
                            target_hidden, past_key_values, use_cache, **kwargs):
        # Legend: B=batch, N=block tokens (q_len), H=hidden_size, T=context len,
        # Nfeat=len(target_layer_ids)*H. noise_embedding: [B, N, H].
        hidden_states = noise_embedding                              # [B, N, H]
        fused = self._apply_target_hidden_fusion(target_hidden)      # [B, T, Nfeat] -> [B, T, H]
        pos_emb = self.rotary_emb(hidden_states, position_ids)       # cos,sin: [B, T+N, head_dim]
        layer_hiddens = []
        for layer in self.layers:
            hidden_states = self._run_layer(                         # [B, N, H] -> [B, N, H]
                layer, hidden_states, fused, attention_mask, position_ids,
                past_key_values, use_cache, pos_emb, **kwargs,
            )
            layer_hiddens.append(hidden_states)                      # collect L x [B, N, H]
        self._layer_hiddens = layer_hiddens  # kept for compute_loss (graph retained)
        self._fused = fused                  # [B, T, H] — gate query source in compute_loss
        return self.norm(hidden_states)                             # [B, N, H]

    def _forward_early_exit(self, position_ids, attention_mask, noise_embedding,
                            target_hidden, past_key_values, use_cache, **kwargs):
        # Inference: B is typically 1, N = block_size. Same legend as above.
        hidden_states = noise_embedding                              # [B, N, H]
        fused = self._apply_target_hidden_fusion(target_hidden)      # [B, T, Nfeat] -> [B, T, H]
        # query = the block's most-recent target hidden (last context position).
        query = fused[:, -1, :]                                      # [B, T, H] -> [B, H]
        pos_emb = self.rotary_emb(hidden_states, position_ids)       # cos,sin: [B, T+N, head_dim]
        n_layers = len(self.layers)
        for l, layer in enumerate(self.layers):
            hidden_states = self._run_layer(                         # [B, N, H] -> [B, N, H]
                layer, hidden_states, fused, attention_mask, position_ids,
                past_key_values, use_cache, pos_emb, **kwargs,
            )
            if l < n_layers - 1:
                weights = self.aggregator.weights(hidden_states, query, l)  # per-layer: [B,N,H],[B,H] -> [N] or [B, N]
                agg = self.aggregator.aggregate(hidden_states, weights)  # [B,N,H] -> [B, H]
                logit = self.gates[l](agg).squeeze(-1)                   # [B,H] -> [B,1] -> [B]
                if bool((logit > self.exit_threshold).all()):
                    # Backfill deeper layers' context K/V so future blocks stay correct.
                    for bl in range(l + 1, n_layers):
                        self.layers[bl].self_attn.cache_context_kv(
                            fused, pos_emb, past_key_values,
                        )
                    self.last_exit_depth = l + 1
                    return self.norm(hidden_states)
        self.last_exit_depth = n_layers
        return self.norm(hidden_states)

    # ------------------------------------------------------------------ loss
    def compute_loss(self, draft_output, input_ids, batch, lm_head, *, gamma: float = 7.0, **_):
        assert self._layer_hiddens is not None, "forward() must run before compute_loss()"
        layer_hiddens = self._layer_hiddens
        # Norm each layer's block hidden ONCE; reused by both the CE term and the
        # gate-label argmax (avoids 2L RMSNorm passes per step).
        normed = [self.norm(h) for h in layer_hiddens]           # L x [B, N, H]

        # 1) Multi-layer CE: every depth must be lm-head-decodable.
        ce = draft_output.new_zeros(())
        for nh in normed:
            ce = ce + compute_position_weighted_ce_loss(
                nh, input_ids, batch, lm_head, gamma=gamma,
            )
        ce = ce / len(normed)

        # 2) Gate BCE against latency-aware exit labels.
        gate = self._compute_gate_loss(layer_hiddens, normed, input_ids, batch, lm_head)

        self._last_metrics["ce_loss"] = ce.item()
        self._last_metrics["gate_loss"] = gate.item()
        return ce + self.skip_config.gate_loss_weight * gate

    @staticmethod
    def _greedy_accept_lengths(preds, gold, drafted, bs):
        """Greedy acceptance length per block, per layer (pure tensor op).

        preds:   [L, B, N]  per-layer argmax token ids
        gold:    [B, N]     ground-truth token at each noise position
        drafted: [B, N]     bool, True at drafted (mask) positions, False at the anchor
        Returns accept: [L, B, A] (A = N // bs) = 1 (anchor) + leading correct run."""
        L, B, N = preds.shape
        A = N // bs
        correct = (preds == gold.unsqueeze(0)) & drafted.unsqueeze(0)  # [L,B,N] & [1,B,N] -> [L, B, N]
        correct = correct.view(L, B, A, bs)                            # [L, B, N] -> [L, B, A, bs]
        # drop the anchor (index 0); cumprod over drafted positions zeroes everything
        # after the first miss; sum -> leading-correct run length.
        run = correct[..., 1:].cumprod(dim=-1).sum(dim=-1)             # [L,B,A,bs-1] -> [L, B, A]
        return run.float() + 1.0                                       # +1 for the anchor -> [L, B, A]

    @staticmethod
    def _exit_labels(accept, ctx, latency_model):
        """Speedup-optimal exit labels from acceptance lengths (pure tensor op).

        accept: [L, B, A]   acceptance length at each depth
        ctx:    [B, A]       per-block context length (for the latency model)
        Returns (labels [L, B, A] in {0,1}, lstar [B, A]).
        score(l) = accept(l) / (draft_latency(l, ctx) + verify_cost(ctx)); l* = argmax_l;
        label = 0 for layers < l*, 1 for layers >= l* (prefix of 0s, suffix of 1s)."""
        L, B, A = accept.shape
        depths = torch.arange(1, L + 1, device=accept.device, dtype=torch.float32).view(L, 1, 1)  # [L,1,1]
        lat = (
            latency_model.draft_latency(depths, ctx.view(1, B, A))     # [L,1,1],[1,B,A] -> [L, B, A]
            + latency_model.verify_cost(ctx).view(1, B, A)             # [1, B, A]
        ).clamp(min=1e-6)                                              # [L, B, A]; guard against
        # non-positive latency: least-squares fits are unconstrained and can extrapolate
        # to lat<=0 outside the profiled (depth, ctx) range, which would flip/NaN the score.
        score = accept / lat                                           # [L, B, A]
        lstar = score.argmax(dim=0)                                    # argmax over L -> [B, A]
        layer_idx = torch.arange(L, device=accept.device).view(L, 1, 1)  # [L,1,1]
        labels = (layer_idx >= lstar.view(1, B, A)).float()           # [L,1,1] >= [1,B,A] -> [L, B, A]
        return labels, lstar

    def _compute_gate_loss(self, layer_hiddens, normed, input_ids, batch, lm_head) -> torch.Tensor:
        # Legend: B=batch, N=noise seq len, H=hidden_size, L=num layers, bs=block_size,
        # A=num_anchors (=N/bs), M=supervised positions, V=vocab.
        # layer_hiddens / normed: lists of L tensors [B, N, H] (raw / RMSNorm'd).
        B, N, H = layer_hiddens[0].shape
        L = len(layer_hiddens)
        bs = self.block_size
        A = N // bs  # num_anchors (blocks are contiguous, fixed size)

        # input_ids: [B, T]; noise_original_positions: [B, N] -> gather along T -> [B, N]
        gold = input_ids.gather(1, batch.noise_original_positions)   # [B, N]
        drafted = batch.position_in_block > 0                         # [B, N] bool (anchor=False)
        anchor_pos = batch.noise_original_positions.view(B, A, bs)[:, :, 0]  # [B,N]->[B,A,bs]->[B, A]

        # ---- labels (no grad): per-layer, per-block greedy acceptance length ----
        with torch.no_grad():
            # Run lm_head ONLY on supervised (loss_mask) positions — anchors and
            # sentinel blocks are masked out of the loss anyway, so projecting them
            # to full vocab is wasted. preds defaults to -1 (never equals gold), so
            # non-supervised positions count as misses (harmless: they're masked).
            idx = batch.loss_mask.nonzero(as_tuple=False)            # [M, 2] -> (b, n)
            preds = torch.full((L, B, N), -1, dtype=torch.long, device=gold.device)  # [L, B, N]
            if idx.numel():
                b_sel, n_sel = idx[:, 0], idx[:, 1]                  # [M], [M]
                for l, nh in enumerate(normed):
                    preds[l, b_sel, n_sel] = lm_head(nh[b_sel, n_sel]).argmax(-1)  # [M,H]->[M,V]->[M]
            accept = self._greedy_accept_lengths(preds, gold, drafted, bs)       # [L, B, A]
            ctx = anchor_pos.float()                                  # [B, A] context length per block
            labels, lstar = self._exit_labels(accept, ctx, self.latency_model)   # [L,B,A], [B,A]
            # Sentinel (padding) blocks share position_in_block with real blocks, so
            # `drafted` can't distinguish them — loss_mask is the real/sentinel flag
            # (False everywhere in a sentinel block). Use it to drop sentinels.
            block_real = batch.loss_mask.view(B, A, bs).any(dim=2)   # [B,A,bs] -> [B, A] (real blocks only)

        # ---- gate logits (grad flows to gates + aggregator, not the backbone) ----
        # query = each block's most-recent context hidden = fused at position anchor-1
        # (matches the inference query fused[:, -1, :], the last accepted token).
        fused = self._fused.detach()                                 # [B, T, H]
        prev_pos = (anchor_pos - 1).clamp(min=0)                     # [B, A] most-recent context pos
        query = fused.gather(1, prev_pos.unsqueeze(-1).expand(B, A, H))  # gather over T -> [B, A, H]
        gate_logits = []
        for l, h in enumerate(layer_hiddens):
            blocks = h.detach().view(B, A, bs, H)                    # [B,N,H] -> [B, A, bs, H]
            weights = self.aggregator.weights(blocks, query, l)      # per-layer: [B,A,bs,H],[B,A,H] -> [bs] or [B, A, bs]
            agg = self.aggregator.aggregate(blocks, weights)         # [B,A,bs,H] -> [B, A, H]
            gate_logits.append(self.gates[l](agg).squeeze(-1))       # [B,A,H] -> [B,A,1] -> [B, A]
        gate_logits = torch.stack(gate_logits, dim=0).float()        # L x [B,A] -> [L, B, A]

        bce = F.binary_cross_entropy_with_logits(gate_logits, labels, reduction="none")  # [L, B, A]
        mask = block_real.unsqueeze(0).expand_as(bce)                # [B,A] -> [1,B,A] -> [L, B, A]
        self._last_metrics["mean_optimal_depth"] = (
            (lstar.float() + 1)[block_real].mean().item() if block_real.any() else float("nan")
        )
        return (bce * mask).sum() / mask.sum().clamp(min=1)          # masked mean -> scalar

    # ------------------------------------------------------------------ logging
    def log_metrics(self, **base) -> dict:
        metrics = super().log_metrics(**base)
        metrics.update(self._last_metrics)
        return metrics
