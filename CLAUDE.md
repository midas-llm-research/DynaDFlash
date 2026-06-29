# midas — repo conventions

## Tensor shape annotations (required)

Whenever you write tensor operations, **explicitly annotate the shapes of the
tensor inputs and outputs** in inline comments. This is a hard requirement, not a
suggestion — it keeps the spec-decoding / draft-model code (where block, context,
head, and KV-cache dims are easy to confuse) readable and reviewable.

Rules:
- Annotate every line that changes a tensor's shape (proj, view/reshape, transpose,
  cat, slice, unsqueeze, matmul, reduction, broadcast, etc.) with the resulting shape.
- For a non-trivial block, start with a short **shape legend** defining the symbols
  used (e.g. `B`=batch, `H`=hidden_size, `D`=head_dim), then use those symbols.
- Show the transformation, not just the final shape, when a line does several steps
  (e.g. `[B,C,H] -> [B,C,Hkv*D] -> [B,C,Hkv,D]`).
- Call out broadcasts explicitly (which dim broadcasts against which).

Example (the style to follow):

```python
# Legend: B=batch, C=ctx_len, H=hidden_size, Hkv=num_key_value_heads, D=head_dim.
k = self.k_proj(x).view(B, C, -1, D)        # [B,C,H] -> [B,C,Hkv*D] -> [B,C,Hkv,D]
k = self.k_norm(k).transpose(1, 2)          # norm over D; [B,C,Hkv,D] -> [B,Hkv,C,D]
cos_ctx = cos[:, :C].unsqueeze(1)           # [B,P,D] -> [B,C,D] -> [B,1,C,D]
k = k * cos_ctx                             # [B,Hkv,C,D] * [B,1,C,D] (broadcast over Hkv) -> [B,Hkv,C,D]
```

## Refactor complicated tensor math into named helpers

When a tensor/math computation is intricate enough that its *intent* isn't obvious
from the operations alone, **pull it out into a separate, well-named helper function**
whenever you judge that doing so meaningfully aids understanding. Prefer this over
leaving a dense block inline. Use your judgment — a one-liner doesn't need extracting,
but a multi-step reduction/broadcast/grouping with real semantics usually does.

A good extraction:
- has a name that states *what* it computes (`_greedy_accept_lengths`, not `_step2`),
- is **pure** where possible (inputs → output, no hidden state), so it's directly
  unit-testable with hand-built tensors and an independent reference,
- carries a docstring giving the shape signature (`accept: [L,B,A]; ctx: [B,A] -> ...`),
- still follows the shape-annotation rules above inside its body.

Rule of thumb: if you'd want to write a comment like "this block does X", that's a
signal to make it a function called `x(...)` instead. Pure tensor helpers are also the
unit-test boundary — extract so the math can be tested in isolation (see
`midas/models/model_skip.py`'s `_greedy_accept_lengths` / `_exit_labels` and their
reference-oracle tests).

## Loss masking — always use `loss_mask`

`DraftTrainingBatch.loss_mask` (`[B, N]`) is the single source of truth for what
gets supervised. It is `True` only at the masked positions of REAL blocks, and
`False` at every anchor and at *every* position of a SENTINEL (padding) block.

**Every loss computation MUST mask with `loss_mask`.** This is a hard rule.
- Never decide what to supervise from `position_in_block` — sentinel blocks share
  `position_in_block` with real blocks, so it does NOT distinguish them; using it
  silently trains on padding.
- For a block-level loss, derive the real-block mask *from `loss_mask`*, e.g.
  `block_real = loss_mask.view(B, num_anchors, block_size).any(dim=2)` — not from
  `position_in_block`.

This holds for any batch shape: each row may have a different number of sentinel
blocks, and each batch a different `num_anchors` (`N = num_anchors * block_size`
varies batch to batch). `loss_mask` already encodes all of that per position, so
masking with it is automatically correct in every case.
