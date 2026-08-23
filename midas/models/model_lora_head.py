"""DFlash with a low-rank residual adapter before the frozen LM head.

The drafter's output is consumed by the target's LM head, which is frozen and was
trained to read representations produced by 36 layers. The drafter has five, and
its final RMSNorm can only rescale — it cannot rotate or mix dimensions — so the
draft states have to land in the head's geometry unaided.

The symptom is a ranking failure, not a knowledge failure: at the first drafted
position the true token is in the drafter's top-16 93.1% of the time but ranked
first only 62.7%. The information is present and the head mis-orders it. A
low-rank residual adapter gives the drafter the rotation the norm cannot:

    h  <-  h + W_up(W_down(h))

`W_up` is zero-initialised, so the model is exactly DFlash at step 0 and the
adapter has to earn whatever it changes. 327,680 parameters at hidden 2560,
rank 64 (0.06% of the drafter), and one small matmul on the block.
"""

import torch
from torch import nn

from midas.models.model import DFlashDraftModel


class LMHeadAdapter(nn.Module):
    """Rank-`r` residual adapter. An exact identity at initialization.

    `down` keeps its default random init while `up` is zeroed: the product is
    zero either way, but zeroing both would leave the pair with no gradient.
    """

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.up(self.down(hidden))


class DFlashDraftModelLoRAHead(DFlashDraftModel):
    """DFlash plus an `LMHeadAdapter` on its output. Identical in every other way.

    `forward` is the only override. Both consumers of the drafter's output — the
    training loss and `dflash_generate` — apply the frozen LM head to whatever
    `forward` returns, so adapting the return value covers both paths and neither
    caller changes.
    """

    def __init__(self, config, rank: int = 64) -> None:
        super().__init__(config)
        # Built after super().__init__ has run post_init(), which would otherwise
        # re-randomize `up` and destroy the identity initialization.
        self.lm_head_adapter = LMHeadAdapter(config.hidden_size, rank)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.lm_head_adapter(super().forward(*args, **kwargs))
