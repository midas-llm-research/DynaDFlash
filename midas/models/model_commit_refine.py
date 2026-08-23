"""DFlash with a commit-and-refine second stage.

Measured in `papers/ideas-8.md`: the drafter's entropy grows ~0.22 nats per horizon while
the target's, given the true prefix, is flat at ~0.28 — and 0.22/horizon is the accumulated
entropy of the tokens in between. The drafter computes a correct MARGINAL; prefix acceptance
consumes a CONDITIONAL. That is why six attempts at better context, one at better output
geometry, and three at better re-ranking all returned ~0: each improves how well the marginal
is computed, and the marginal is the wrong object.

Under prefix acceptance, conditioning errors are free — if slot k-1 is wrong it is rejected,
so slot k never mattered. But that only holds if the model CONDITIONS ON THE TOKEN IT EMITS.
A mid-backbone readout that later gets revised forfeits it.

So the commit has to come after the backbone, not inside it:

    stage 1   the existing layers, unchanged. Slots 1..commit emit from here.
    commit    their argmax through the frozen LM head, detached.
    stage 2   narrow extra layers over the block, with the committed embeddings injected
              and the target context re-read. Slots commit+1.. emit from here.

Committed slots emit exactly what stage 2 conditions on, so the free-error property holds.

The logit lens says why an intermediate commit will not do. Per-horizon agreement reading out
from layer 2 / 3 / 4(final) is 0.698 / 0.857 / 0.885 at horizon 1 and 0.615 / 0.742 / 0.802 at
horizon 2, so committing mid-backbone costs 6-8 points exactly where leverage is highest.

`commit_gate` is zero-initialised, so stage 2 starts as plain extra depth and the conditioning
has to earn what it changes.
"""

import copy
from typing import Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from midas.config import CommitRefineConfig
from midas.models.model import DFlashDraftModel, Qwen3DFlashDecoderLayer


class CommitRefineDraftModel(DFlashDraftModel):
    """DFlash plus a conditioned second stage. Stage 1 is byte-identical to DFlash."""

    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config, commit_refine_config: Optional[CommitRefineConfig] = None) -> None:
        super().__init__(config)
        cfg = commit_refine_config or CommitRefineConfig()
        if not 1 <= cfg.commit < config.block_size - 1:
            raise ValueError(
                f"commit={cfg.commit} must leave at least one slot to refine "
                f"(block_size={config.block_size}, offset 0 is the anchor)."
            )
        self.commit = cfg.commit

        # Narrow: stage 2 applies a correction, it does not re-derive the block.
        refine_config = copy.deepcopy(config)
        refine_config.intermediate_size = cfg.refine_intermediate
        # The attention module indexes `layer_types[layer_idx]`, and stage 2's indices run
        # past the backbone's, so the list has to be extended to match.
        types = list(getattr(config, "layer_types", None)
                     or ["full_attention"] * config.num_hidden_layers)
        refine_config.layer_types = types + [types[-1]] * cfg.refine_layers
        # Layer indices continue past the backbone so the two stages get separate KV slots.
        self.refine_layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(refine_config, config.num_hidden_layers + i)
                for i in range(cfg.refine_layers)
            ]
        )
        self.refine_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # None reproduces the pre-fix unscaled injection; see `_injection`. Absent from the
        # state dict in that mode, so checkpoints from either era load strictly.
        self.commit_norm = (Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                            if cfg.scaled_injection else None)
        self.commit_gate = nn.Parameter(torch.zeros(config.hidden_size))
        self.post_init()
        nn.init.zeros_(self.commit_gate)   # post_init re-randomizes; re-close the gate
        self._shared: list = []

    def bind_shared(self, lm_head: nn.Module, embed_tokens: nn.Module) -> None:
        """Give stage 2 the frozen head and embedding it commits through.

        Held in a plain list so `nn.Module` does not register them — they belong to the
        target and must stay out of this model's state dict.
        """
        self._shared = [lm_head, embed_tokens]

    def _block_offsets(self, n: int, device) -> torch.Tensor:
        if n % self.block_size:
            raise ValueError(
                f"sequence length {n} is not a multiple of block_size {self.block_size}; "
                f"committed slots would be misaligned. Training concatenates whole blocks "
                f"and generation passes exactly one, so this holds in both."
            )
        return torch.arange(n, device=device) % self.block_size

    def _injection(self, hidden: torch.Tensor, stage1: torch.Tensor,
                   offsets: torch.Tensor) -> torch.Tensor:
        """The committed tokens, put on the residual stream's scale.

        Scale is the whole point. A raw token embedding has RMS ~0.02 while the residual
        here has RMS ~40, so adding it directly is a 0.009% perturbation — measured, and it
        left the conditioning inert (removing the commit changed 0.1% of refined tokens).
        No reachable gate value fixes that: it would need to be ~80 and it reaches ~0.3.

        Normalising the embedding and rescaling to the residual makes a gate of O(0.1) an
        O(10%) perturbation. RMSNorm maps zero to zero, so uncommitted slots stay untouched
        and a zero gate is still an exact no-op.
        """
        committed = self._committed_embedding(stage1, offsets)
        if self.commit_norm is None:
            return committed                     # pre-fix behaviour; see CommitRefineConfig
        scale = hidden.detach().pow(2).mean(-1, keepdim=True).sqrt()
        return self.commit_norm(committed) * scale

    @torch.no_grad()
    def _committed_embedding(self, stage1: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Embeddings of the tokens stage 1 commits to; zero everywhere else.

        Detached on purpose. Stage 1 is trained to be right, not to make stage 2's job
        easier, and this matches inference exactly, where the committed token is simply
        stage 1's argmax.

        The argmax here must stay in step with `propose`, which is what actually emits the
        token at generation time. `propose` defaults to temperature 0, so the two agree
        unconditionally today — but a `propose` that sampled, or DFlash 2's path search,
        would emit a token different from the one committed here and silently forfeit the
        free-error property this design rests on.
        """
        lm_head, embed_tokens = self._shared
        slots = ((offsets >= 1) & (offsets <= self.commit)).nonzero(as_tuple=True)[0]
        out = torch.zeros_like(stage1)
        out[:, slots] = embed_tokens(lm_head(stage1[:, slots]).argmax(-1)).to(stage1.dtype)
        return out

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if not self._shared:
            raise ValueError(
                "CommitRefineDraftModel.forward needs the target's head and embedding. "
                "Call model.bind_shared(target.lm_head, target.model.embed_tokens) first."
            )
        hidden = noise_embedding
        layer_kwargs = dict(
            target_hidden=self._apply_target_hidden_fusion(target_hidden),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            position_embeddings=self.rotary_emb(hidden, position_ids),
            **kwargs,
        )
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, **layer_kwargs)
        stage1 = self.norm(hidden)

        offsets = self._block_offsets(hidden.shape[1], hidden.device)
        hidden = hidden + self.commit_gate * self._injection(hidden, stage1, offsets)
        for layer in self.refine_layers:
            hidden = layer(hidden_states=hidden, **layer_kwargs)
        stage2 = self.refine_norm(hidden)

        # Committed slots emit stage 1 — exactly what stage 2 conditioned on.
        return torch.where((offsets <= self.commit).view(1, -1, 1), stage1, stage2)

    def log_metrics(self, **base) -> dict:
        """How far the commit gate has opened.

        Stage 2 is plain extra depth until this moves, so a flat zero says the conditioning
        never engaged and whatever tau changed is depth — which a result could not otherwise
        be distinguished from without training a separate depth-only control.
        """
        gate = self.commit_gate.detach().float().abs()
        return {**super().log_metrics(**base),
                "commit_gate_mean": gate.mean().item(),
                "commit_gate_max": gate.max().item()}
