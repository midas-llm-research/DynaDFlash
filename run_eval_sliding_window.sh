#!/usr/bin/env bash
#
# Eval a DFlash draft with SPARSE attention: keep the first SINK_TOKENS context
# tokens (attention sinks) + the most recent SLIDING_WINDOW tokens, evict the rest.
# Override via env var, e.g.:  SLIDING_WINDOW=256 ./run_eval_sliding_window.sh
# Extra modal flags pass straight through.
#
# NOTE: eviction only fires when a generation exceeds SINK_TOKENS + SLIDING_WINDOW,
# so the generations must run long enough or nothing changes vs the dense baseline.
#
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="${VARIANT:-baseline}"
SLIDING_WINDOW="${SLIDING_WINDOW:-512}"
SINK_TOKENS="${SINK_TOKENS:-4}"

# Prefer the project venv's modal if present, else whatever is on PATH.
MODAL="modal"
[ -x ".venv/bin/modal" ] && MODAL=".venv/bin/modal"

echo "Running SPARSE eval: variant=$VARIANT sink=$SINK_TOKENS window=$SLIDING_WINDOW"
exec "$MODAL" run scripts/modal_eval.py \
    --variant "$VARIANT" \
    --sliding-window "$SLIDING_WINDOW" \
    --sink-tokens "$SINK_TOKENS" \
    "$@"
