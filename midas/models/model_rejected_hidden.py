"""Rejected-State Warm-Start for DFlash.

Reuses information from rejected speculative tokens by projecting both the
target and draft hidden states at rejected positions into a warm-start
embedding for the next draft block.

Architecture (per rejected position i):
    t_i = RMSNorm(h_i^T)                          target hidden state
    d_i = RMSNorm(h_i^D)                          draft hidden state
    u_i = [t_i; d_i; e_Δ(i - r)]                  concat with distance embed
    r_i = W @ u_i + b                             project to hidden dim
    z_i^(0) = e_mask + e_warm + r_i               warm-started embedding

W and b are zero-initialized so training starts from baseline DFlash behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from midas.config import RejectedHiddenConfig
from midas.models.model import DFlashDraftModel, compute_position_weighted_ce_loss

if TYPE_CHECKING:
    from midas.data import DraftTrainingBatch


class RejectedStateProjector(nn.Module):
    """Projects rejected target+draft hidden states into warm-start embeddings."""

    def __init__(self, hidden_size: int, config: RejectedHiddenConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.distance_embed_dim = config.distance_embed_dim
        self.max_distance = config.max_distance

        self.target_norm = Qwen3RMSNorm(hidden_size, eps=1e-6)
        self.draft_norm = Qwen3RMSNorm(hidden_size, eps=1e-6)
        self.distance_embed = nn.Embedding(config.max_distance, config.distance_embed_dim)

        input_dim = 2 * hidden_size + config.distance_embed_dim
        self.proj = nn.Linear(input_dim, hidden_size, bias=True)
        self.warm_embed = nn.Parameter(torch.zeros(hidden_size))

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        target_hidden: torch.Tensor,
        draft_hidden: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_hidden: [M, H] target hidden states at rejected positions
            draft_hidden: [M, H] draft hidden states at rejected positions
            distances: [M] integer distances from first rejection (clamped to max_distance-1)

        Returns:
            [M, H] projected warm-start representations r_i
        """
        t = self.target_norm(target_hidden)
        d = self.draft_norm(draft_hidden)
        distances = distances.clamp(max=self.max_distance - 1)
        e_delta = self.distance_embed(distances)
        u = torch.cat([t, d, e_delta], dim=-1)
        return self.proj(u)


class DFlashDraftModelRejectedHidden(DFlashDraftModel):
    """DFlash draft model with rejected-state warm-start.

    Same decoder architecture as DFlashDraftModel. Only addition is the
    RejectedStateProjector module used to build warm-start embeddings from
    previous block rejections.
    """

    def __init__(self, config, rejected_hidden_config: RejectedHiddenConfig):
        super().__init__(config)
        self.rejected_hidden_config = rejected_hidden_config
        self.rejected_state_proj = RejectedStateProjector(
            config.hidden_size, rejected_hidden_config
        )

    def prepare_warm_start_embedding(
        self,
        noise_embedding: torch.Tensor,
        draft_hidden: torch.Tensor,
        target_at_noise: torch.Tensor,
        batch: "DraftTrainingBatch",
        block_size: int,
        lm_head: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build warm-started noise embeddings by simulating rejection.

        Simulates sequential verification (cumprod) to find the first rejection
        per block, then warm-starts all subsequent positions in that block.

        Args:
            noise_embedding: [B, N, H] standard mask-token embeddings
            draft_hidden: [B, N, H] detached draft output from first pass
            target_at_noise: [B, N, H] target final-layer hidden at noise positions
            batch: DraftTrainingBatch with position_in_block, noise_original_positions, loss_mask
            block_size: int
            lm_head: target model's LM head (for computing draft predictions)
            input_ids: [B, T] original token ids

        Returns:
            [B, N, H] embedding tensor with warm-starts at rejected positions
        """
        B, N, H = noise_embedding.shape
        device = noise_embedding.device
        num_anchors = N // block_size

        draft_logits = lm_head(draft_hidden)
        draft_preds = draft_logits.argmax(dim=-1)
        target_tokens = input_ids.gather(1, batch.noise_original_positions)

        is_correct_at_mask = (draft_preds == target_tokens) | (
            batch.position_in_block == 0
        )
        is_correct_blocks = is_correct_at_mask.view(B, num_anchors, block_size)

        # cumprod over mask positions to simulate sequential verification
        mask_correct = is_correct_blocks[:, :, 1:]
        accepted_prefix = mask_correct.cumprod(dim=-1)

        # Number of accepted mask positions per block → first rejection position
        n_accepted = accepted_prefix.sum(dim=-1).long()
        first_rejection = n_accepted + 1  # [B, A] in position_in_block terms

        # Warm-start positions: strictly AFTER first rejection within each block
        pos_in_block_blocks = batch.position_in_block.view(B, num_anchors, block_size)
        first_rej_expanded = first_rejection.unsqueeze(-1)
        is_warm_start = pos_in_block_blocks > first_rej_expanded

        # Distance from first rejection for each position
        distances_blocks = (pos_in_block_blocks - first_rej_expanded).clamp(min=0)

        is_warm_start_flat = is_warm_start.view(B, N)
        distances_flat = distances_blocks.view(B, N)

        warm_indices = is_warm_start_flat.nonzero(as_tuple=False)
        if warm_indices.shape[0] == 0:
            return noise_embedding

        batch_idx = warm_indices[:, 0]
        noise_idx = warm_indices[:, 1]

        target_h = target_at_noise[batch_idx, noise_idx]
        draft_h = draft_hidden[batch_idx, noise_idx]
        dists = distances_flat[batch_idx, noise_idx]

        r = self.rejected_state_proj(target_h, draft_h, dists)

        result = noise_embedding.clone()
        result[batch_idx, noise_idx] = (
            noise_embedding[batch_idx, noise_idx]
            + self.rejected_state_proj.warm_embed
            + r
        )
        return result

    def compute_loss(
        self,
        draft_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        batch: "DraftTrainingBatch",
        lm_head: nn.Module,
        *,
        gamma: float = 7.0,
        **_,
    ) -> torch.Tensor:
        return compute_position_weighted_ce_loss(
            draft_hidden, input_ids, batch, lm_head, gamma=gamma
        )

    def log_metrics(self, **base) -> dict:
        metrics = dict(base)
        with torch.no_grad():
            proj_norm = self.rejected_state_proj.proj.weight.norm().item()
            warm_norm = self.rejected_state_proj.warm_embed.norm().item()
        metrics["proj_weight_norm"] = proj_norm
        metrics["warm_embed_norm"] = warm_norm
        return metrics

    @classmethod
    def from_pretrained_dflash(
        cls,
        dflash_checkpoint: str,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        **constructor_kwargs,
    ) -> "DFlashDraftModelRejectedHidden":
        from transformers import AutoModel

        original = AutoModel.from_pretrained(
            dflash_checkpoint,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        config = original.config
        config._attn_implementation = "sdpa"

        model = cls(config, **constructor_kwargs)
        model_state = model.state_dict()
        original_state = original.state_dict()
        loaded, skipped = 0, []
        for key, value in original_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
            else:
                skipped.append(key)
        model.load_state_dict(model_state)

        if device is None:
            device = next(original.parameters()).device
        model = model.to(device=device, dtype=dtype)
        del original

        from loguru import logger

        logger.info(
            f"Loaded {loaded}/{len(original_state)} params from {dflash_checkpoint} "
            f"into {cls.__name__}. Skipped: {skipped or 'none'}. "
            f"RejectedStateProjector is zero-initialized. Model on {device}, {dtype}."
        )
        return model

    @classmethod
    def from_config_random(
        cls,
        dflash_checkpoint: str,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        **constructor_kwargs,
    ) -> "DFlashDraftModelRejectedHidden":
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            dflash_checkpoint, trust_remote_code=True
        )
        config._attn_implementation = "sdpa"
        model = cls(config, **constructor_kwargs)
        if device is not None:
            model = model.to(device=device, dtype=dtype)
        from loguru import logger

        logger.info(
            f"Created {cls.__name__} with random weights. "
            f"{sum(p.numel() for p in model.parameters()):,} params on {device}, {dtype}."
        )
        return model
