"""
DFlash 2 — DFlash plus a path selector and a block-local dynamic convolution.

Reproduces https://inco.ai/blog/dflash2/ (2026-08-18) on top of our DFlash base.
See papers/dflash2_breakdown.md for the full analysis. Two independent additions,
neither of which touches the one-pass parallel design:

1. CandidateSelector — the drafter's top-1 is right 85% of the time but the right
   token is in its top-16 99.5% of the time, so acceptance is limited by SELECTION,
   not prediction (their oracle: tau 4.27 -> 6.79). A low-rank bilinear score over
   adjacent candidates picks a coherent path. The sequential work is O(K=16) per
   position instead of O(vocab) as in DSpark-style correction heads.

2. BlockLocalDynamicConv — within-block attention mass decays down the layers
   (30% at layer 1 to 8% at layer 5), so late positions lose their dependencies.
   A two-tap dynamic depthwise conv around every attention and MLP sublayer gives
   each position its predecessor directly, recovering most of what 3x the depth buys.

Two deliberate departures from the released reference (dflash/model.py), both
documented at their site: block-local convolution (their code is inference-only,
where it cannot matter; ours trains on concatenated blocks, where it does), and
codebooks as projections of the frozen embedding table rather than free
vocab-sized tables.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
)
from typing_extensions import Unpack

from midas.config import DFlash2Config
from midas.models.model import DFlashDraftModel, Qwen3DFlashAttention

if TYPE_CHECKING:
    from midas.data import DraftTrainingBatch


def _convolve(x, dynamic, base, group_size):
    """Depthwise causal conv over the LENGTH axis of x. Returns [b, l, h].

    kernel[tap] = base[tap] + dynamic[tap], with one dynamic coefficient shared
    per `group_size` channels. Mirrors `_grouped_dynamic_convolve` upstream.
    """
    b, l, h = x.shape
    groups = h // group_size
    blocks = x.view(b, l, groups, group_size)
    dyn = dynamic.reshape(b, l, base.shape[0], groups, 1)
    out = torch.zeros_like(blocks)
    for tap in range(base.shape[0]):
        shifted = blocks if tap == 0 else F.pad(blocks[:, :-tap], (0, 0, 0, 0, tap, 0))
        out = out + base[tap].view(1, 1, groups, group_size).to(x.dtype) * shifted
        out = torch.addcmul(out, dyn[:, :, tap], shifted)
    return out.view(b, l, h)


class BlockLocalDynamicConv(nn.Module):
    """Two-tap dynamic depthwise conv, applied WITHIN each block.

        Conv(x)_t = k_{t,0} * x_t + k_{t,1} * x_{t-1}

    One module serves both the pre- and post-sublayer convolution: `prepare`
    returns the convolved input plus the dynamic kernel that `finish` applies
    afterwards, so a single projection produces both (as upstream does).

    BLOCK LOCALITY: the reference shifts along the raw sequence, which is correct
    for inference (one block) but WRONG for our training, where `prepare_draft_inputs`
    concatenates `num_anchors` blocks — position 0 of block i+1 would read the last
    position of block i, leaking across the boundary that the attention mask exists
    to prevent. Reshaping to [B*num_blocks, block_size, H] before shifting makes the
    tap block-local by construction, so each block's first position (the anchor)
    has no predecessor and every later position reads within its own block.

    Identity-initialized (current tap 1, predecessor tap 0, zero dynamic
    correction) so the module is an exact no-op at init.
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int, block_size: int):
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(f"hidden_size {hidden_size} not divisible by group_size {group_size}")
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.block_size = block_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=False)
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0, :] = 1.0          # current position passes through
        nn.init.zeros_(self.kernel_projection.weight)  # no dynamic correction yet

    def _convolve_blocks(self, hidden, dynamic, base):
        # NB: not `_apply` -- that name belongs to nn.Module and is used by
        # .to()/.float(); overriding it silently breaks device/dtype moves.
        b, n, h = hidden.shape
        if n % self.block_size:
            raise ValueError(
                f"sequence length {n} is not a multiple of block_size {self.block_size}; "
                f"the convolution tap would cross a block boundary"
            )
        blocks = b * (n // self.block_size)
        out = _convolve(
            hidden.reshape(blocks, self.block_size, h),
            dynamic.reshape(blocks, self.block_size, *dynamic.shape[2:]),
            base,
            self.group_size,
        )
        return out.reshape(b, n, h)

    def prepare(self, hidden):
        """-> (pre-sublayer conv output, dynamic kernel for `finish`)"""
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return self._convolve_blocks(hidden, dynamic[..., 0, :, :], self.base_kernel[0]), dynamic[..., 1, :, :]

    def finish(self, hidden, dynamic):
        return self._convolve_blocks(hidden, dynamic, self.base_kernel[1])


class CandidateSelector(nn.Module):
    """Low-rank bilinear scorer over adjacent candidates.

        S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>

    U_t(b) is the drafter's own logit for candidate b (free). A and B embed the
    predecessor and the candidate; H gates which parts of the match count.

    A and B are rank-r PROJECTIONS OF THE FROZEN EMBEDDING TABLE rather than free
    vocab-sized codebooks. The released code ships `nn.Embedding(vocab, rank)`
    tables, but that costs 2*V*r (77.8M at V=151936, r=256) whereas the blog quotes
    +2.0M for the selector -- which is exactly 3*hidden*rank (3*2560*256 = 1.97M).
    Projections reconcile the two and are what a shipped table would be folded from.
    """

    def __init__(self, hidden_size: int, rank: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.predecessor_proj = nn.Linear(hidden_size, rank, bias=False)   # A
        self.successor_proj = nn.Linear(hidden_size, rank, bias=False)     # B
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)  # H

    def score(self, gate, predecessor_ids, unary, candidate_ids, embed_tokens):
        """[B, r], [B], [B, K], [B, K] -> [B, K] scores."""
        a = self.predecessor_proj(embed_tokens(predecessor_ids))          # [B, r]
        b = self.successor_proj(embed_tokens(candidate_ids))              # [B, K, r]
        return unary + torch.einsum("br,bkr->bk", a * gate, b)

    def walk(self, hidden, logits, anchor_ids, embed_tokens):
        """Greedy left-to-right walk. -> [B, L] chosen tokens.

        Fully parallel except this walk, which does one embedding lookup and a
        rank-r dot product against K candidates per position -- no LM head, no
        backbone pass.
        """
        unary, candidates = torch.topk(logits, self.top_k, dim=-1)
        gate = self.hidden_projection(hidden)
        predecessor, path = anchor_ids, []
        for t in range(hidden.shape[1]):
            scores = self.score(
                gate[:, t], predecessor, unary[:, t], candidates[:, t], embed_tokens
            )
            chosen = candidates[:, t].gather(-1, scores.argmax(-1, keepdim=True))[:, 0]
            predecessor, _ = chosen, path.append(chosen)
        return torch.stack(path, dim=1)


class Qwen3DFlash2DecoderLayer(GradientCheckpointingLayer):
    """DFlash decoder layer with a conv before and after each sublayer."""

    def __init__(self, config: Qwen3Config, layer_idx: int, dflash2_config: DFlash2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        make_conv = lambda: BlockLocalDynamicConv(  # noqa: E731
            config.hidden_size,
            dflash2_config.conv_kernel_size,
            dflash2_config.conv_group_size,
            config.block_size,
        )
        self.attention_conv = make_conv()
        self.mlp_conv = make_conv()

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return residual + hidden_states


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash + path selector + block-local dynamic convolution."""

    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    def __init__(self, config, dflash2_config: Optional[DFlash2Config] = None) -> None:
        super().__init__(config)
        self.dflash2_config = dflash2_config or DFlash2Config()
        self.layers = nn.ModuleList(
            [
                Qwen3DFlash2DecoderLayer(config, i, self.dflash2_config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.candidate_selector = CandidateSelector(
            config.hidden_size,
            self.dflash2_config.selector_rank,
            self.dflash2_config.selector_top_k,
        )
        self.post_init()
        # post_init re-randomizes the freshly built layers; restore the convs to
        # an exact no-op so the model starts as plain DFlash.
        for layer in self.layers:
            layer.attention_conv.reset_to_identity()
            layer.mlp_conv.reset_to_identity()
        self._last_loss_metrics: dict = {}

    def propose(self, draft_hidden, anchor_ids, lm_head, embed_tokens=None):
        if embed_tokens is None:
            raise ValueError("DFlash2DraftModel.propose needs the shared embedding table.")
        return self.candidate_selector.walk(
            draft_hidden, lm_head(draft_hidden), anchor_ids, embed_tokens
        )

    def compute_loss(
        self,
        draft_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        batch: "DraftTrainingBatch",
        lm_head: nn.Module,
        *,
        gamma: float = 7.0,
        embed_tokens: Optional[nn.Module] = None,
        **_,
    ) -> torch.Tensor:
        """Position-weighted CE plus a K-way selector loss.

        Written out rather than delegating to `compute_position_weighted_ce_loss`
        so the [M, vocab] draft logits are materialized once and shared by both
        terms -- at M=7680, V=151936 a second copy costs ~2.3 GB.

        The selector is trained teacher-forced: at a masked position whose original
        index is p, the predecessor is the ground-truth token at p-1 (which is the
        anchor for the block's first mask), and the label is the ground-truth token's
        slot in the drafter's own top-K list. Positions whose ground truth is not in
        the top-K are excluded -- the selector cannot pick what is not there.
        """
        loss_mask = batch.loss_mask
        masked = loss_mask.nonzero(as_tuple=False)
        if masked.shape[0] == 0:
            return torch.tensor(0.0, device=draft_hidden.device, requires_grad=True)
        b_idx, n_idx = masked[:, 0], masked[:, 1]

        hidden = draft_hidden[b_idx, n_idx]
        logits = lm_head(hidden)
        orig = batch.noise_original_positions[b_idx, n_idx]
        targets = input_ids[b_idx, orig]

        weights = torch.exp(-(batch.position_in_block[b_idx, n_idx].float() - 1) / gamma)
        weight_sum = weights.sum().clamp(min=1)
        ce = F.cross_entropy(logits, targets, reduction="none")
        total = (ce * weights).sum() / weight_sum
        metrics = {"loss_ce": total.detach().item()}

        lam = self.dflash2_config.lambda_selector
        if lam > 0.0:
            if embed_tokens is None:
                raise ValueError(
                    "DFlash2 selector training needs `embed_tokens`; pass the target's "
                    "embedding table through compute_loss."
                )
            sel = self.candidate_selector
            unary, candidates = torch.topk(logits, sel.top_k, dim=-1)
            scores = sel.score(
                sel.hidden_projection(hidden),
                input_ids[b_idx, orig - 1],          # teacher-forced predecessor
                unary,
                candidates,
                embed_tokens,
            )
            hit = candidates == targets.unsqueeze(1)
            in_topk = hit.any(dim=1)
            if in_topk.any():
                label = hit.float().argmax(dim=1)
                sel_ce = F.cross_entropy(scores, label, reduction="none")
                w = weights * in_topk
                sel_loss = (sel_ce * w).sum() / w.sum().clamp(min=1)
                total = total + lam * sel_loss
                metrics["loss_selector"] = sel_loss.detach().item()
            metrics["recall_at_k"] = in_topk.float().mean().item()
            metrics["recall_at_1"] = (logits.argmax(-1) == targets).float().mean().item()

        self._last_loss_metrics = metrics
        return total

    def log_metrics(self, **base) -> dict:
        return {**super().log_metrics(**base), **self._last_loss_metrics}
