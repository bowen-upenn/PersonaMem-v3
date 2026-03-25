#!/bin/bash
# Convenience wrapper for the persona pipeline (API mode only).
# For Claude Code mode, use Claude Code directly — see README.md.
#
# ============================================================================
# Examples
# ============================================================================
#
# --- Azure OpenAI API (default, uses AZURE_OPENAI_* vars from .env) ---------
#
#   # All users, 4 parallel threads (default)
#   bash scripts/run_persona_pipeline.sh data/test_interactions.csv
#
#   # All users, sequential (one at a time)
#   bash scripts/run_persona_pipeline.sh data/test_interactions.csv --max_workers 1
#
#   # Single user
#   bash scripts/run_persona_pipeline.sh data/test_interactions.csv --user_id 2124791 --verbose
#
# --- OpenAI API (set OPENAI_API_KEY in .env, leave Azure vars unset) --------
#
#   # Uses gpt-5-chat by default; override with --model
#   bash scripts/run_persona_pipeline.sh data/test_interactions.csv --model gpt-4o
#
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

INPUT_CSV="${1:-data/test_interactions.csv}"
shift 1 2>/dev/null || true

cd "$REPO_ROOT"
python scripts/run_persona_pipeline.py --input_csv "$INPUT_CSV" "$@"
