"""
Acceptance-aligned auxiliary losses for DFlash-family draft models.

Cross-entropy asks "how much probability mass is on the right token?". Acceptance
at temperature 0 asks "is the right token ranked first?". Those are different
questions, and empirically in this project they are decoupled: the variant with
the lowest CE had flat acceptance, and the variant with the best acceptance had
worse CE than the baseline.

Two terms target the gap. Both are training-only — no parameters, no change to
inference or to the acceptance rule.

1. `compute_margin_loss` — a hinge on the exact quantity acceptance depends on.
2. `compute_feature_reg_loss` — dense supervision on the draft's output geometry.

Both are labelled by the TARGET (its argmax / its final hidden state), not by the
token stored in the training corpus: that corpus was sampled at temperature 0.7,
so its token is frequently not the target's argmax, and a margin pointed at it
would train the drafter to rank the sample above the argmax.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def target_argmax_chunked(
    lm_head: nn.Module,
    hidden: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """argmax_v lm_head(hidden)[v], computed in chunks.

    Materializing [M, vocab] for M=7680, vocab=151936 costs ~2.3 GB in bf16 on top
    of what the CE term already allocates. Chunking keeps the transient at
    chunk_size x vocab and never holds the full logit matrix.

    Args:
        lm_head: the target's frozen head. Only ever CALLED — never introspected,
            so `nn.Identity()` and test doubles work.
        hidden: [M, H]
    Returns:
        [M] long
    """
    if hidden.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=hidden.device)
    out = []
    for start in range(0, hidden.shape[0], chunk_size):
        out.append(lm_head(hidden[start : start + chunk_size]).argmax(dim=-1))
    return torch.cat(out, dim=0)


def top1_gap(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """g = z[y*] - max_{v != y*} z[v]. Returns [M], float32.

    g > 0 is exactly the temperature-0 acceptance condition. No softmax is
    needed: log_softmax subtracts the same per-row logsumexp from both terms, so
    the gap in log-prob space equals the gap in logit space. topk(2) gets the
    runner-up without materializing a second [M, V] tensor.
    """
    z_true = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    top2 = logits.topk(2, dim=-1)
    already_top1 = top2.indices[:, 0] == labels
    z_runner = torch.where(already_top1, top2.values[:, 1], top2.values[:, 0])
    return (z_true - z_runner).float()


def compute_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Hinge on the top-1 agreement gap. Returns [M], unreduced.

        g = z[y*] - max_{v != y*} z[v]
        loss = relu(margin - g)

    g > 0 is exactly the condition for the token to be accepted at temperature 0,
    so this optimizes the acceptance event itself with a buffer of `margin`.

    Zero gradient once g > margin (the position is already safely accepted), and
    constant-magnitude gradient otherwise — pushing y* up and the single most
    dangerous competitor down.

    No softmax is needed: log_softmax subtracts the same per-row logsumexp from
    both terms, so the gap in log-prob space equals the gap in logit space.
    `topk(2)` gets the runner-up without materializing a second [M, V] tensor.
    """
    return F.relu(margin - top1_gap(logits, labels))


def compute_feature_reg_loss(
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """Distance between draft and target final hidden states. Returns [M].

    `cosine` (default) vs `smooth_l1`: MEASURED to have essentially the same
    gradient scale w.r.t. the draft hidden (0.011 vs 0.009 on unit-scale vectors),
    because smooth_l1's mean over the hidden dim divides its gradient by H.
    Cosine is the default because argmax(W h) is invariant to positive rescaling
    of h -- magnitude cannot affect acceptance -- and because being bounded in
    [0, 2] makes its weight easier to reason about.

    NOTE ON SCALE: both metrics give a gradient ~O(1/sqrt(H)) per position,
    roughly 50x weaker than cross-entropy's. lambda_reg must therefore be LARGE
    (order 10-100, not EAGLE's 0.1) for this term to influence training at all;
    EAGLE's 0.1 is calibrated for un-normalized features with far larger errors.
    Pick it from a gradient-share sweep, never by analogy.
    """
    d, t = draft_hidden.float(), target_hidden.float()
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(d, t, dim=-1)
    if metric == "smooth_l1":
        return F.smooth_l1_loss(d, t, reduction="none").mean(dim=-1)
    raise ValueError(f"unknown reg metric {metric!r}; use 'smooth_l1' or 'cosine'")


def compute_eal_loss(
    gaps_flat: torch.Tensor,
    batch,
    batch_idx: torch.Tensor,
    noise_idx: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    r"""Differentiable expected accepted length. Returns a scalar in [-1, 0].

        tau_hat = sum_k prod_{j<=k} sigmoid(gap_j / T)
        L       = -tau_hat / (block_size - 1)

    Why this rather than a position-weighted per-token loss: the true objective is
    tau = sum_k prod_{j<=k} a_j, whose derivative w.r.t. accuracy at position j is

        (prod_{i<j} a_i) * (1 + sum_{k>j} prod_{j<i<=k} a_i)
         \____survival___/   \______downstream value______/

    DFlash's exp(-(k-1)/gamma) is a STATIC guess at that quantity, but it is
    dynamic and data-dependent. The cumprod here produces it exactly, for free:
    doomed suffixes get no gradient, and positions after an already-safe prefix
    get more -- the opposite of what a fixed exponential decay does.

    sigmoid(gap/T) is the temperature-0 acceptance surrogate (is y* ranked first),
    not q(y*) -- the latter is the temperature-1 quantity and is what makes a
    plain cumprod-over-probabilities loss mismatched with greedy evaluation.

    Computed in log space (logsigmoid + cumsum) because a 15-term cumprod of
    values ~0.8 underflows toward 1e-2 and back-props poorly in bf16.
    """
    B, N = batch.loss_mask.shape
    block_size = int(batch.position_in_block.max().item()) + 1
    num_blocks = N // block_size

    # Scatter the per-position gaps back onto the [B, N] block grid.
    gap_grid = torch.zeros(B, N, dtype=gaps_flat.dtype, device=gaps_flat.device)
    gap_grid = gap_grid.index_put((batch_idx, noise_idx), gaps_flat)
    gap_grid = gap_grid.view(B, num_blocks, block_size)[..., 1:]  # drop the anchor

    # A block is real (not a padding sentinel) iff its k=1 slot is in the loss mask.
    block_active = batch.loss_mask.view(B, num_blocks, block_size)[..., 1]

    log_surv = F.logsigmoid(gap_grid / temperature).cumsum(dim=-1)
    tau_hat = log_surv.exp().sum(dim=-1)                      # [B, num_blocks]

    denom = block_active.sum().clamp(min=1)
    return -(tau_hat * block_active).sum() / denom / (block_size - 1)


def compute_aligned_loss(
    draft_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    batch,
    lm_head: nn.Module,
    *,
    gamma: float = 7.0,
    target_final_hidden: Optional[torch.Tensor] = None,
    lambda_margin: float = 0.0,
    lambda_reg: float = 0.0,
    lambda_eal: float = 0.0,
    margin: float = 1.0,
    eal_temperature: float = 1.0,
    relabel_ce: bool = False,
    reg_metric: str = "cosine",
    argmax_chunk: int = 1024,
    metrics: Optional[dict] = None,
) -> torch.Tensor:
    """Position-weighted CE plus the two acceptance-aligned terms.

        L = L_ce + lambda_margin * L_margin + lambda_reg * L_reg

    Every term uses the same exp(-(k-1)/gamma) position weighting and the same
    weight-sum normalization as the baseline loss.

    ALIGNMENT (the part that is easy to get wrong): the draft slot for a masked
    position is supervised on input_ids[b, p] where p = noise_original_positions.
    The target's hidden at position p predicts the token at p+1. So the target
    state that predicts the SAME token the draft slot predicts lives at p-1.
    Both target-derived signals are therefore gathered at `p - 1`.

    Callers must pass `target_final_hidden` = the target's post-final-norm hidden
    states [B, T, H] (i.e. `output.hidden_states[-1]`, the tensor lm_head consumes)
    whenever either lambda is non-zero or relabel_ce is set.
    """
    loss_mask = batch.loss_mask
    masked_indices = loss_mask.nonzero(as_tuple=False)
    if masked_indices.shape[0] == 0:
        return torch.tensor(0.0, device=draft_hidden.device, requires_grad=True)

    batch_idx = masked_indices[:, 0]
    noise_idx = masked_indices[:, 1]

    draft_hidden_masked = draft_hidden[batch_idx, noise_idx]
    draft_logits_masked = lm_head(draft_hidden_masked)

    orig_positions = batch.noise_original_positions[batch_idx, noise_idx]
    sdd_tokens = input_ids[batch_idx, orig_positions]

    pos_in_block = batch.position_in_block[batch_idx, noise_idx].float()
    weights = torch.exp(-(pos_in_block - 1) / gamma)
    weight_sum = weights.sum().clamp(min=1)

    needs_target = (
        lambda_margin > 0.0 or lambda_reg > 0.0 or lambda_eal > 0.0 or relabel_ce
    )
    target_argmax = None
    target_hidden_masked = None
    if needs_target:
        if target_final_hidden is None:
            raise ValueError(
                "compute_aligned_loss needs `target_final_hidden` (the target's "
                "post-final-norm hidden states) when any aligned term is enabled."
            )
        # The state that predicts the token at p lives at p-1.
        prev_positions = orig_positions - 1
        if bool((prev_positions < 0).any()):
            raise ValueError(
                "orig_position - 1 went negative; anchors must start at or after "
                "position 1 of the sequence."
            )
        target_hidden_masked = target_final_hidden[batch_idx, prev_positions]
        if lambda_margin > 0.0 or lambda_eal > 0.0 or relabel_ce:
            target_argmax = target_argmax_chunked(
                lm_head, target_hidden_masked, chunk_size=argmax_chunk
            )

    ce_labels = target_argmax if relabel_ce else sdd_tokens
    ce = F.cross_entropy(draft_logits_masked, ce_labels, reduction="none")
    total = (ce * weights).sum() / weight_sum

    if metrics is not None:
        metrics["loss_ce"] = total.detach().item()
        if target_argmax is not None:
            metrics["sdd_argmax_agreement"] = (
                (target_argmax == sdd_tokens).float().mean().item()
            )

    if lambda_margin > 0.0:
        margin_per_pos = compute_margin_loss(draft_logits_masked, target_argmax, margin)
        margin_loss = (margin_per_pos * weights).sum() / weight_sum
        total = total + lambda_margin * margin_loss
        if metrics is not None:
            metrics["loss_margin"] = margin_loss.detach().item()
            metrics["frac_margin_active"] = (margin_per_pos > 0).float().mean().item()

    if lambda_eal > 0.0:
        gaps = top1_gap(draft_logits_masked, target_argmax)
        eal_loss = compute_eal_loss(
            gaps, batch, batch_idx, noise_idx, temperature=eal_temperature
        )
        total = total + lambda_eal * eal_loss
        if metrics is not None:
            metrics["loss_eal"] = eal_loss.detach().item()
            metrics["tau_hat"] = -eal_loss.detach().item() * (
                int(batch.position_in_block.max().item())
            )

    if lambda_reg > 0.0:
        reg_per_pos = compute_feature_reg_loss(
            draft_hidden_masked, target_hidden_masked, metric=reg_metric
        )
        reg_loss = (reg_per_pos * weights).sum() / weight_sum
        total = total + lambda_reg * reg_loss
        if metrics is not None:
            metrics["loss_featreg"] = reg_loss.detach().item()

    return total
