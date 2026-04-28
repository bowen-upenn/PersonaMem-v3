"""Per-instance audit helpers for benchmark build-time guardrails (Phase D).

Each builder in `evaluation/build_benchmark.py` (and the e-task builders)
wraps every `bucket.append(instance)` with `_audit_instance(...)`. If the
instance fails any structural check, it's regenerated up to N=3 times with
a bumped seed before being dropped. Build-level counters (n_built,
n_passed_first_try, n_regenerated, n_dropped_after_max_attempts) are
tracked per task and persisted to `benchmark/{uid}/build_audit.json`.

The structural checks are deterministic and cheap (no LLM):

  - length normalization: max_len/min_len of candidate captions ≤ 4×
  - canonical keys: every candidate exposes the same fields (no
    distinguishing-by-presence)
  - format-action uniqueness: held-out's content_type must NOT be
    unique among the pool (mix reels + images + threads)
  - question-token overlap: no single candidate has > 3 unique
    non-stopword tokens overlapping the question text that no other
    candidate shares

A separate optional `_blind_baseline_check(...)` helper wraps a small LLM
call; turning it on costs ~$0.001 per instance and is gated by
`prepare_eval_data.py --audit_blind` (off by default for fast iteration).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

# Re-use the stopword set + tokenizer from the verifiers module so the
# audit's "common-token overlap" notion matches what the eval scorer uses.
from evaluation.tasks.agentic_verifiers import _tokens, _STOPWORDS  # noqa: F401


CANONICAL_SLATE_KEYS = frozenset(["idx", "app", "title", "caption", "hashtags", "content_type"])


def _candidate_text(c: dict) -> str:
    return f"{c.get('title', '')} {c.get('caption', '')}".strip()


def structural_audit(inst: dict, task_type: str) -> dict:
    """Deterministic structural checks. Returns:
        {"pass": bool, "checks": [(name, "pass"|"fail (...)"), ...]}
    """
    checks: list[tuple[str, str]] = []

    # Only ranking-style instances have a `slate` to audit deeply. Other
    # task families (chatbot, agentic, e6) run a tighter check set.
    slate = inst.get("slate") or []

    if slate:
        # 1. Canonical-keys uniformity — drops format-by-presence leaks
        bad_keys: list[int] = []
        for i, c in enumerate(slate):
            extra = set(c.keys()) - CANONICAL_SLATE_KEYS
            if extra:
                bad_keys.append(i)
        if bad_keys:
            checks.append(("canonical_keys", f"fail (extra keys at {bad_keys[:3]})"))
        else:
            checks.append(("canonical_keys", "pass"))

        # 2. Length normalization — max/min caption length ≤ 4×
        lens = [len(_candidate_text(c)) for c in slate]
        non_empty = [n for n in lens if n > 0]
        if non_empty and max(non_empty) > 4 * max(1, min(non_empty)):
            checks.append(("length_normalization",
                           f"fail (max={max(non_empty)} min={min(non_empty)} ratio={max(non_empty)/min(non_empty):.1f})"))
        else:
            checks.append(("length_normalization", "pass"))

        # 3. Format-action uniqueness — held_out's content_type must NOT
        # be the only one of its kind. (A pool of all `text` is fine; what
        # we forbid is held_out=`reel` while all distractors=`text`.)
        held_idx = inst.get("held_out_idx")
        if isinstance(held_idx, int) and 0 <= held_idx < len(slate):
            held_ct = slate[held_idx].get("content_type")
            other_cts = Counter(c.get("content_type") for i, c in enumerate(slate) if i != held_idx)
            if held_ct and held_ct not in other_cts:
                checks.append(("content_type_not_unique",
                               f"fail (held_out_ct={held_ct!r}, distractor_cts={dict(other_cts)})"))
            else:
                checks.append(("content_type_not_unique", "pass"))

    # 4. Question-token overlap (any task with a question) — single candidate
    # must NOT exclusively share too many non-stopword tokens with the prompt.
    question = (
        inst.get("user_query")
        or inst.get("triggering_user_query")
        or inst.get("directive_user_message")
        or inst.get("query_text")
        or ""
    )
    if question and slate:
        q_tokens = _tokens(question)
        if q_tokens:
            cand_tokens_list = [_tokens(_candidate_text(c)) for c in slate]
            unique_overlap = []
            for i, cts in enumerate(cand_tokens_list):
                shared = q_tokens & cts
                # tokens this candidate shares with question that no OTHER candidate shares
                exclusive = {t for t in shared if not any(t in cand_tokens_list[j] for j in range(len(slate)) if j != i)}
                unique_overlap.append(len(exclusive))
            if unique_overlap and max(unique_overlap) > 3:
                worst = unique_overlap.index(max(unique_overlap))
                checks.append(("question_token_overlap",
                               f"fail (cand idx={worst} has {max(unique_overlap)} unique overlap tokens with question)"))
            else:
                checks.append(("question_token_overlap", "pass"))

    passed = all(s == "pass" for _, s in checks)
    return {"pass": passed, "checks": checks}


def audit_instance(
    inst: dict,
    task_type: str,
    rebuild_fn: Callable[[dict, int], dict] | None = None,
    max_attempts: int = 3,
    blind_baseline: Callable | None = None,
) -> tuple[dict | None, dict]:
    """Audit-and-regenerate. Returns (instance_or_None, audit_report).

    instance: the candidate instance dict
    rebuild_fn: closure that takes (current_inst, seed_bump) and returns a
                regenerated instance (different distractor sample, same target).
                If None, no regeneration happens — single-shot audit.
    max_attempts: total tries before dropping (1 = no retry).
    blind_baseline: optional callable taking the instance and returning
                {"correct": bool, "details": ...} — when present and `correct`
                is True, the instance is regenerated as if the structural
                check failed.
    """
    last_report: dict = {}
    current = inst
    for attempt in range(max_attempts):
        s = structural_audit(current, task_type)
        b: dict = {"skipped": True}
        if blind_baseline is not None:
            try:
                b = blind_baseline(current)
            except Exception as exc:
                b = {"error": f"{type(exc).__name__}: {exc}"}
        last_report = {"attempt": attempt + 1, "structural": s, "blind": b}
        struct_ok = bool(s.get("pass"))
        blind_ok = not bool(b.get("correct"))
        if struct_ok and blind_ok:
            return current, last_report
        if rebuild_fn is None:
            return None, last_report
        try:
            current = rebuild_fn(inst, attempt + 1)
        except Exception as exc:
            last_report["rebuild_error"] = f"{type(exc).__name__}: {exc}"
            return None, last_report
    return None, last_report


# ---------------------------------------------------------------------------
# Build-level reporter — accumulates per-task counters, dumps to JSON.
# ---------------------------------------------------------------------------

class BuildAuditReporter:
    """Accumulate per-task audit outcomes; flush to benchmark/{uid}/build_audit.json."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._stats: dict[str, dict] = {}
        self._examples: dict[str, list[dict]] = {}

    def record(self, task_type: str, audit_report: dict, kept: bool) -> None:
        s = self._stats.setdefault(task_type, {
            "n_built": 0, "n_passed_first_try": 0,
            "n_regenerated": 0, "n_dropped_after_max_attempts": 0,
        })
        s["n_built"] += 1
        attempts = audit_report.get("attempt") or 1
        if not kept:
            s["n_dropped_after_max_attempts"] += 1
            ex = self._examples.setdefault(task_type, [])
            if len(ex) < 5:
                ex.append({"reason": "exhausted_attempts", **audit_report})
        elif attempts == 1:
            s["n_passed_first_try"] += 1
        else:
            s["n_regenerated"] += 1

    def write(self, out_dir):
        from pathlib import Path
        import json
        out = Path(out_dir) / f"build_audit.json"
        report = {"user_id": self.user_id, "by_task": self._stats, "drop_examples": self._examples}
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return out
