#!/usr/bin/env bash
#
# Eval a DFlash draft with DENSE (full-context) attention — the baseline.
# Override the variant via env var, e.g.:  VARIANT=router ./run_eval_baseline.sh
# Extra modal flags pass straight through, e.g.:  ./run_eval_baseline.sh --variant router
#
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="${VARIANT:-baseline}"

# Prefer the project venv's modal if present, else whatever is on PATH.
MODAL="modal"
[ -x ".venv/bin/modal" ] && MODAL=".venv/bin/modal"

echo "Running DENSE eval: variant=$VARIANT"
exec "$MODAL" run scripts/modal_eval.py \
    --variant "$VARIANT" \
    "$@"
