"""
Correction drafter — a depth-1 DFlash 2 that re-drafts a rejected block suffix
from the *hidden states* the rejected positions already produced.

See papers/hidden_state_recycling.md. In one paragraph: when the target rejects a
block suffix, DFlash throws away the drafter's hidden states for those positions
and re-drafts them from mask embeddings. Those states were computed against the
correct target prefix; the only thing verification revealed is "the token at the
rejection point is X, not Y". So a single layer, fed the retained residuals plus
the corrected token as its anchor, should be able to finish the block — at 1/5
the drafter's cost.

Indexing (matching `dflash_generate` in model.py, where a block is anchor + 15
drafted tokens and `acceptance_length` counts accepted *drafted* tokens):

    round 1   slot j -> position s + j,  j = 0 .. block_size-1
              accept a  ->  s..s+a confirmed, target's bonus token at s+a+1
    round 2   slot 0            -> anchor, the bonus token at s+a+1 (true embedding)
              slot i, i >= 1    -> position s+a+1+i, fed round 1's residual from
                                   slot i+a+1
              valid slots       -> block_size - 1 - a  (1 anchor + block_size-2-a redrafts)
              slots past that   -> tail padding

The round-2 block is always materialized at full `block_size` width with the
padding at the *tail*, never shrunk. Two reasons: `BlockLocalDynamicConv` requires
the sequence length to be a multiple of `block_size`, and its taps read backwards
(t-1), so tail padding provably cannot leak into a valid position. Visibility of
the pad slots is the caller's job via `attention_mask`; the loss is gated by
`valid_mask`. The pad slots must NOT be sent to the target for verification.

This module is the round-2 model only. Round 1 stays the trained DFlash 2
drafter, which must be called with `return_prenorm=True` so its raw residual is
available to recycle.
"""

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from midas.config import CorrectionConfig
from midas.models.model import build_target_layer_ids
from midas.models.model_dflash2 import DFlash2DraftModel


# ---------------------------------------------------------------------------
# Trigger policy
# ---------------------------------------------------------------------------


def positions_to_redraft(acceptance_length: int, block_size: int) -> int:
    """How many slots a correction round would actually predict.

    `acceptance_length` is the raw loop variable from `dflash_generate` — accepted
    *drafted* tokens, excluding the target's bonus token. (Note the loop logs
    `acceptance_length + 1`; that is not what this takes.) One slot is consumed by
    the bonus token becoming the anchor, so `block_size - 2 - a` remain.
    """
    return block_size - 2 - acceptance_length


def should_correct(
    acceptance_length: int,
    block_size: int,
    *,
    min_remaining: int = 7,
    previous_was_correction: bool = False,
) -> bool:
    """Trigger for a correction round.

    Never twice in a row: round-2 residuals feeding a round 3 would make the
    input distribution recursive (the model is trained on round-1 residuals
    only), and it would let a low-acceptance round feed back on itself.

    `min_remaining` gates on having enough slots left to be worth a target
    verification pass — the correction round buys its own verify, so the win
    condition is that it accepts at least as many tokens as a fresh full block
    would have.
    """
    if previous_was_correction:
        return False
    return positions_to_redraft(acceptance_length, block_size) >= min_remaining


def simulate_acceptance(
    proposed_tokens: torch.Tensor,
    verifier_tokens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Which suffix the target would have rejected. [B, N], [B, N] -> [B, num_blocks].

    Reproduces the eval-time rule from `dflash_generate`:
        `(block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(1).sum(1)`
    i.e. the longest prefix of drafted slots matching what the verifier committed.
    Slot 0 (the anchor) is skipped in both tensors. This is exact for training even
    though the target is teacher-forced on ground truth rather than on the draft:
    within the accepted prefix the two conditionings are identical, by definition
    of acceptance, and the first mismatch is where the cumprod stops.

    On `verifier_tokens` and temperature. The SDD corpus was generated at
    temperature 0.7 (`sdd/generate_sdd.py`), so **the ground-truth token IS a T=0.7
    draw from the target** at that position. Passing ground truth therefore
    simulates a T=0.7 verifier for free, and keeps the round-2 anchor and every
    label on the one path we have labels for. Passing `sample(target_logits, 0.7)`
    instead is statistically equivalent for the distribution of `a`, but a fresh
    draw can disagree with the corpus token, which puts the block off-path after
    the rejection point. Passing `argmax` gives the greedy rule `midas/eval.py`
    defaults to (`--temperature 0.0`); worth logging as a diagnostic, since it is
    systematically more permissive than T=0.7 and so shifts the width distribution
    the correction model trains on.
    """
    B, N = proposed_tokens.shape
    if N % block_size:
        raise ValueError(f"token length {N} is not a multiple of block_size {block_size}")
    num_blocks = N // block_size
    drafted = proposed_tokens.view(B, num_blocks, block_size)[:, :, 1:]
    committed = verifier_tokens.view(B, num_blocks, block_size)[:, :, 1:]
    return (drafted == committed).long().cumprod(dim=-1).sum(dim=-1)


# ---------------------------------------------------------------------------
# Round-2 block construction
# ---------------------------------------------------------------------------


@dataclass
class CorrectionBlock:
    """Block-local inputs for one correction round, shapes [B, N] / [B, N, H]
    with N = num_blocks * block_size.

    - `noise_token_ids`: anchor slots hold the corrected token, everything else
      holds `mask_token_id` (recycled slots take the residual branch, so their id
      is unused; pads are inert).
    - `recycled_hidden`: round-1 residuals gathered into their round-2 slots.
      Garbage at non-recycled slots — `is_recycled` selects.
    - `is_recycled`: True exactly on slots fed by a recycled residual.
    - `noise_original_positions`: original sequence position per slot, clamped on
      pads to the last valid position (same sentinel trick `prepare_draft_inputs`
      uses) so nothing indexes out of range.
    - `position_in_block`: slot index, i.e. distance from the *corrected* anchor.
      This is what the `exp(-(k-1)/gamma)` loss weighting should see; measuring
      from the round-1 anchor would weight the wrong thing.
    - `valid_mask`: slots the model is asked to predict (excludes anchor and pads).
      Use as `loss_mask` in training and to size the round-2 verify at inference.

    Attention mask is deliberately NOT built here: training needs the
    concatenated-blocks ctx mask from `prepare_draft_inputs`, inference needs a
    KV-cache-relative one. Whatever the caller builds must (a) let each slot see
    target hidden at positions < its anchor, i.e. <= s+a, (b) allow within-block
    visibility only, and (c) hide the pad slots.
    """

    noise_token_ids: torch.Tensor
    recycled_hidden: torch.Tensor
    is_recycled: torch.Tensor
    noise_original_positions: torch.Tensor
    position_in_block: torch.Tensor
    valid_mask: torch.Tensor
    block_size: int

    @property
    def is_pad(self) -> torch.Tensor:
        """Tail slots this block does not reach. A slot is either the anchor
        (position 0), recycled, or padding — so padding is what is left over.
        Nothing may attend to these, and the target must never verify them.
        """
        return (self.position_in_block > 0) & ~self.is_recycled

    @property
    def anchor_positions(self) -> torch.Tensor:
        """[B, num_blocks] the round-2 anchor position s+a+1 per block, which is
        also the ctx cutoff: a slot may see target hidden strictly before it.
        """
        return self.by_block(self.noise_original_positions)[:, :, 0]

    def by_block(self, flat: torch.Tensor) -> torch.Tensor:
        """[B, N, ...] -> [B, num_blocks, block_size, ...]."""
        b, n = flat.shape[:2]
        return flat.view(b, n // self.block_size, self.block_size, *flat.shape[2:])


def build_correction_block(
    round1_prenorm: torch.Tensor,
    true_token_ids: torch.Tensor,
    acceptance: torch.Tensor,
    block_anchors: torch.Tensor,
    *,
    block_size: int,
    mask_token_id: int,
) -> CorrectionBlock:
    """Map round-1 state onto round-2 slots. Batched over blocks; inference is
    just the B=1, num_blocks=1 case.

    Args:
        round1_prenorm: [B, N, H] pre-norm residuals from round 1 (i.e. the
            drafter called with `return_prenorm=True`), N = num_blocks*block_size.
        true_token_ids: [B, N] the correct token at each round-1 slot. Training
            reads these from ground truth; inference only needs index a+1 of each
            block, which the target's posterior provides.
        acceptance: [B, num_blocks] accepted drafted tokens per block (the raw
            `acceptance_length`, not the +1 logged value).
        block_anchors: [B, num_blocks] original position s of each round-1 block.

    Callers must gate on `should_correct`; `acceptance` is clamped here only to
    keep gathers in range, not to make a degenerate block meaningful.
    """
    B, N, H = round1_prenorm.shape
    if N % block_size:
        raise ValueError(f"round1_prenorm length {N} is not a multiple of block_size {block_size}")
    num_blocks = N // block_size
    device = round1_prenorm.device

    a = acceptance.to(device=device, dtype=torch.long).clamp(0, block_size - 2).unsqueeze(-1)
    slot = torch.arange(block_size, device=device).view(1, 1, -1)

    # 1 anchor + (block_size - 2 - a) recycled slots.
    n_valid = (block_size - 1 - a).clamp(min=1)
    is_recycled = (slot >= 1) & (slot < n_valid)
    # slot i is fed by round-1 slot i + a + 1 (so slot 1 <- a+2, the first
    # position past the corrected token). Clamped for pads.
    src = (slot + a + 1).clamp(max=block_size - 1)

    blocks = round1_prenorm.view(B, num_blocks, block_size, H)
    recycled_hidden = blocks.gather(2, src.unsqueeze(-1).expand(B, num_blocks, block_size, H))

    true_blocks = true_token_ids.view(B, num_blocks, block_size)
    anchor_ids = true_blocks.gather(2, (a + 1).clamp(max=block_size - 1))  # [B, nb, 1]
    noise_token_ids = torch.full(
        (B, num_blocks, block_size), mask_token_id, dtype=true_token_ids.dtype, device=device
    )
    noise_token_ids[:, :, :1] = anchor_ids

    # Pads reuse the last valid position so downstream gathers stay in range.
    positions = block_anchors.to(device).unsqueeze(-1) + a + 1 + slot.minimum(n_valid - 1)

    return CorrectionBlock(
        noise_token_ids=noise_token_ids.view(B, N),
        recycled_hidden=recycled_hidden.view(B, N, H),
        is_recycled=is_recycled.expand(B, num_blocks, block_size).reshape(B, N),
        noise_original_positions=positions.view(B, N),
        position_in_block=slot.expand(B, num_blocks, block_size).reshape(B, N),
        valid_mask=is_recycled.expand(B, num_blocks, block_size).reshape(B, N),
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Input mixing
# ---------------------------------------------------------------------------


class RecycledInputMixer(nn.Module):
    """Builds the layer input from two sources: recycled residuals and token
    embeddings.

        h_t = (recycled_proj(recycled_t) if recycled else fresh_proj(embed_t))
              + source_tag(is_recycled_t)

    Both projections are identity-initialized and the tag is zero-initialized, so
    at init a recycled slot passes its residual through untouched — a
    parent-warm-started correction layer is then exactly "the 6th drafter layer",
    which is the whole point of warm-starting it.

    Why a projection on the *fresh* branch too: the anchor slot carries a raw
    token embedding, which is in-distribution for the drafter's layer 1 but not
    for its layer 5. `fresh_proj` gives training a way to lift embeddings into
    residual space without touching the recycled path.

    Why the tag: the two branches live at very different scales (a layer-5
    residual has RMS >> an embedding), and while every consumer downstream of the
    residual stream is per-position RMS-normed — `input_layernorm` for the
    sublayers, `self.norm` for the LM head — so cross-position scale mostly
    washes out, the layer still needs to *know* which slots are stale.
    """

    def __init__(self, hidden_size: int, rms_norm_eps: float, normalize_recycled: bool = False):
        super().__init__()
        self.recycled_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fresh_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.recycled_norm = (
            Qwen3RMSNorm(hidden_size, eps=rms_norm_eps) if normalize_recycled else None
        )
        self.source_tag = nn.Embedding(2, hidden_size)
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        with torch.no_grad():
            nn.init.eye_(self.recycled_proj.weight)
            nn.init.eye_(self.fresh_proj.weight)
            nn.init.zeros_(self.source_tag.weight)

    def forward(
        self,
        noise_embedding: torch.Tensor,
        recycled_hidden: Optional[torch.Tensor],
        is_recycled: Optional[torch.Tensor],
    ) -> torch.Tensor:
        fresh = self.fresh_proj(noise_embedding)
        if recycled_hidden is None:
            # Masks-only mode: the control arm, and also what the first block of a
            # generation would see. Tag every slot as fresh.
            zeros = torch.zeros(
                noise_embedding.shape[:-1], dtype=torch.long, device=noise_embedding.device
            )
            return fresh + self.source_tag(zeros).to(fresh.dtype)
        if is_recycled is None:
            raise ValueError("`is_recycled` is required whenever `recycled_hidden` is given")
        if recycled_hidden.shape != noise_embedding.shape:
            raise ValueError(
                f"recycled_hidden {tuple(recycled_hidden.shape)} must match "
                f"noise_embedding {tuple(noise_embedding.shape)}"
            )
        recycled = recycled_hidden.to(noise_embedding.dtype)
        if self.recycled_norm is not None:
            recycled = self.recycled_norm(recycled)
        recycled = self.recycled_proj(recycled)
        gate = is_recycled.unsqueeze(-1)
        return torch.where(gate, recycled, fresh) + self.source_tag(is_recycled.long()).to(fresh.dtype)


# ---------------------------------------------------------------------------
# Correction model
# ---------------------------------------------------------------------------


def _patch_config_for_correction(config, correction_config: CorrectionConfig):
    """Shallow depth without mutating the caller's config.

    `target_layer_ids` defaults to the *parent's* set rather than
    `build_target_layer_ids(n, 1)`. `dflash_generate` extracts `target_hidden`
    once per iteration, so sharing the layer set lets a single tensor feed both
    models; it also keeps `fc` shape-compatible for `warm_start_from_parent`. The
    fc is 5*H*H = 32M params at H=2560, which is not worth optimizing away.
    """
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    parent_ids = dflash_config.get("target_layer_ids") or build_target_layer_ids(
        config.num_target_layers, config.num_hidden_layers
    )
    target_layer_ids = correction_config.target_layer_ids or parent_ids

    patched = copy.deepcopy(config)
    patched.num_hidden_layers = correction_config.num_layers
    layer_types = getattr(patched, "layer_types", None)
    if layer_types is not None and len(layer_types) > correction_config.num_layers:
        patched.layer_types = layer_types[: correction_config.num_layers]
    dflash_config["target_layer_ids"] = list(target_layer_ids)
    patched.dflash_config = dflash_config
    return patched


class CorrectionDraftModel(DFlash2DraftModel):
    """Depth-1 DFlash 2 whose noise slots are fed recycled residuals.

    Inherits `propose` (the candidate-selector walk, which starts from the
    corrected anchor — exactly the predecessor chain round 2 needs), `compute_loss`
    (position-weighted CE + selector CE), and `log_metrics` from DFlash 2
    unchanged. Only `__init__` and `forward` differ: depth is forced to 1 and the
    layer input comes from `RecycledInputMixer` instead of straight from
    `noise_embedding`.

    Keeps `BlockLocalDynamicConv`: depth 1 is precisely the regime it was
    introduced to rescue, so it should matter more here than at depth 5.
    """

    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    def __init__(self, config, correction_config: Optional[CorrectionConfig] = None) -> None:
        self.correction_config = correction_config or CorrectionConfig()
        super().__init__(
            _patch_config_for_correction(config, self.correction_config),
            self.correction_config.dflash2,
        )
        # Built after DFlash2's post_init(), so it is not re-randomized by it —
        # but an explicit reset keeps the identity guarantee if init_weights() is
        # ever re-run, matching how the convs are handled upstream.
        self.input_mixer = RecycledInputMixer(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            normalize_recycled=self.correction_config.normalize_recycled,
        )
        self.input_mixer.reset_to_identity()

    # -- inference/training entry point -------------------------------------

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        recycled_hidden: Optional[torch.Tensor] = None,
        is_recycled: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        return_prenorm: bool = False,
        **kwargs,
    ):
        """Same contract as `DFlashDraftModel.forward` plus the two recycling
        arguments. `recycled_hidden=None` runs the masks-only control arm.
        """
        if noise_embedding is None:
            raise ValueError("CorrectionDraftModel.forward needs `noise_embedding` for the anchor slot")
        if noise_embedding.shape[1] % self.block_size:
            raise ValueError(
                f"noise length {noise_embedding.shape[1]} is not a multiple of block_size "
                f"{self.block_size}; round-2 blocks must be tail-padded to full width, "
                f"not shrunk (BlockLocalDynamicConv reshapes by block)"
            )

        hidden_states = self.input_mixer(noise_embedding, recycled_hidden, is_recycled)
        target_hidden = self._apply_target_hidden_fusion(target_hidden)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if return_prenorm:
            return self.norm(hidden_states), hidden_states
        return self.norm(hidden_states)

    # -- warm start ----------------------------------------------------------

    @torch.no_grad()
    def warm_start_from_parent(self, parent: nn.Module) -> tuple[int, list[str]]:
        """Copy the parent drafter's *last* layers plus its shared modules.

        `from_pretrained_dflash` matches keys exactly, which would give this model
        the parent's layer 0. The last layer is what we want: the correction model
        is meant to act as the layer that would have come next, and with the
        identity-initialized mixer that makes it a literal continuation of the
        parent's residual stream at init.

        Returns (params copied, keys skipped).
        """
        parent_layers = parent.config.num_hidden_layers
        own_layers = self.config.num_hidden_layers
        if own_layers > parent_layers:
            raise ValueError(
                f"correction model has {own_layers} layers, parent only has {parent_layers}"
            )
        offset = parent_layers - own_layers

        own = self.state_dict()
        copied, skipped = 0, []
        for key, value in parent.state_dict().items():
            if key.startswith("layers."):
                idx = int(key.split(".")[1])
                if idx < offset:
                    continue  # earlier parent layers have no destination
                dest = f"layers.{idx - offset}." + key.split(".", 2)[2]
            else:
                dest = key
            if dest in own and own[dest].shape == value.shape:
                own[dest] = value
                copied += 1
            else:
                skipped.append(key)
        self.load_state_dict(own)

        from loguru import logger

        logger.info(
            f"warm_start_from_parent: copied {copied} tensors from "
            f"{type(parent).__name__} layers[{offset}:] into {type(self).__name__}. "
            f"Skipped: {skipped or 'none'}. input_mixer left at identity."
        )
        return copied, skipped
