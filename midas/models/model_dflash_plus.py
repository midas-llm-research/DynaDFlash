"""
DFlash draft model combining the two mechanisms that target distinct bottlenecks:

1. ContextRouter target-hidden fusion (input side) — replaces DFlash's static
   concat+FC with a per-token attention-style mix over the selected target
   layers, so each token gets its own view of the target hierarchy.
2. Per-layer token mixing (within-block side) — a learned
   block_size x block_size matrix applied after each MLP, giving direct
   cross-position communication that does not depend on attention query
   quality.

The two are orthogonal: the router changes *what context each token sees*, the
mixer changes *how block positions talk to each other*. Neither touches the
other's code path, so this class is just DFlashDraftModelRouted with the
decoder layers swapped for the token-mix variant.

Everything else — attention, target-layer extraction, loss, generation — is
inherited unchanged from DFlashDraftModel.
"""

import torch.nn as nn

from midas.config import RouterConfig
from midas.models.model_context_router import DFlashDraftModelRouted
from midas.models.model_token_mix import Qwen3DFlashTokenMixDecoderLayer


class DFlashDraftModelPlus(DFlashDraftModelRouted):
    """DFlash + ContextRouter fusion + per-layer token mixing."""

    _no_split_modules = ["Qwen3DFlashTokenMixDecoderLayer"]

    def __init__(self, config, router_config: RouterConfig) -> None:
        # Builds the ContextRouter through the inherited
        # _build_target_hidden_fusion hook, then the standard DFlash layers.
        super().__init__(config, router_config)
        # Swap in the token-mixing decoder layers.
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashTokenMixDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()
        # post_init re-randomizes the freshly built layers, including the
        # mixers — re-zero them last so they start as an exact no-op.
        for layer in self.layers:
            nn.init.zeros_(layer.token_mixer.weight)
