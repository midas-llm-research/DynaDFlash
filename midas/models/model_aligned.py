"""
DFlash draft model trained with acceptance-aligned auxiliary losses.

Architecturally IDENTICAL to DFlashDraftModel — same layers, same forward, same
state dict, zero extra parameters. The only difference is `compute_loss`, which
adds a top-1 margin term and/or a feature-regression term on top of the usual
position-weighted cross-entropy.

Because the state dict is unchanged, a checkpoint trained here loads into the
plain DFlashDraftModel and evaluates through the existing pipeline untouched.
"""

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from midas.config import AlignedLossConfig
from midas.losses_aligned import compute_aligned_loss
from midas.models.model import DFlashDraftModel

if TYPE_CHECKING:
    from midas.data import DraftTrainingBatch


class AlignedLossMixin:
    """Swaps compute_loss for the acceptance-aligned objective.

    Mixed in before a DFlashDraftModel subclass so the same objective can later be
    applied to any variant (dflare-deep, token-mix, ...) without duplication.
    """

    def _init_aligned_loss(self, aligned_loss_config: Optional[AlignedLossConfig]) -> None:
        self.aligned_loss_config = aligned_loss_config or AlignedLossConfig()
        self._last_loss_metrics: dict = {}

    def compute_loss(
        self,
        draft_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        batch: "DraftTrainingBatch",
        lm_head: nn.Module,
        *,
        gamma: float = 7.0,
        target_final_hidden: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        cfg = self.aligned_loss_config
        metrics: dict = {}
        loss = compute_aligned_loss(
            draft_hidden,
            input_ids,
            batch,
            lm_head,
            gamma=gamma,
            target_final_hidden=target_final_hidden,
            lambda_margin=cfg.lambda_margin,
            lambda_reg=cfg.lambda_reg,
            margin=cfg.margin,
            relabel_ce=cfg.relabel_ce,
            argmax_chunk=cfg.argmax_chunk,
            metrics=metrics,
        )
        self._last_loss_metrics = metrics
        return loss

    def log_metrics(self, **base) -> dict:
        metrics = super().log_metrics(**base)
        metrics.update(self._last_loss_metrics)
        return metrics


class DFlashDraftModelAligned(AlignedLossMixin, DFlashDraftModel):
    """DFlash base + acceptance-aligned training objective."""

    def __init__(self, config, aligned_loss_config: Optional[AlignedLossConfig] = None):
        super().__init__(config)
        self._init_aligned_loss(aligned_loss_config)
