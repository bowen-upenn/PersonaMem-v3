#!/usr/bin/env bash
# Sequential per-persona eval wrapper.
#
# Usage:
#   scripts/run_eval.sh PERSONA_ID [--mode MODE] [--limit N] [--resume] [--dry_run]
#
# Cross-persona parallelism happens at the xargs layer (see run_eval_all.sh);
# this script runs exactly one persona end-to-end.
set -euo pipefail

PERSONA_ID="${1:?usage: run_eval.sh PERSONA_ID [--mode MODE] [--limit N] [--resume] [--dry_run]}"
shift || true

RUN_DIR="${PM3_RUN_DIR:-benchmark/${PERSONA_ID}/runs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

exec python -m evaluation.run_eval \
    --user_id "$PERSONA_ID" \
    --run_dir "$RUN_DIR" \
    "$@"
