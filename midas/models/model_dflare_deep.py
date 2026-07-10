"""
DFlare with shrunk MLP and more layers — ablation to test depth vs width tradeoff.

Identical to DFlare except:
- intermediate_size configurable (default: halved from 9728 to 4864)
- num_draft_layers configurable (default: 9 instead of 7)

No causal mask, no token mixing. Isolates the MLP/depth change.
"""

import copy
from typing import Optional

from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from midas.models.model_dflare import DFlareDraftModel


class DFlareDeepDraftModel(DFlareDraftModel):
    """DFlare with configurable MLP width. For depth/width ablation."""

    def __init__(self, config: Qwen3Config, dflare_config):
        if hasattr(dflare_config, "intermediate_size") and dflare_config.intermediate_size is not None:
            config = copy.deepcopy(config)
            config.intermediate_size = dflare_config.intermediate_size
        super().__init__(config, dflare_config)
