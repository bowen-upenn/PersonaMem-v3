"""Shared plumbing for eval tasks: test-item discovery, GT slice construction,
long-context serialization, token counting, agent-loop runner.

Layered on top of `backend_query.BackendQuery`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from evaluation.backend_query import APPS, BackendQuery, materialize_snapshot


# --- Data classes ----------------------------------------------------------

@dataclass
class TestItem:
    """A single held-out test preference plus everything needed to evaluate it."""

    user_id: str
    app: str                       # app where the parent event lives
    source_object_id: str
    source_timestamp: int          # T_test
    formatted_timestamp: str
    source_interaction_type: str
    source_hashtags: list[str]
    content: dict                  # parent event's content
    interaction_format: dict
    preference: dict               # the test preference itself (un-stripped — harness-side only)
    over_personalization_irrelevant: list[dict] = field(default_factory=list)
    conversation: list[dict] | None = None  # only for chatbot
    conversation_type: str | None = None

    @property
    def polarity(self) -> str:
        return "positive" if "positive" in self.source_interaction_type else "negative"

    @property
    def persona_item(self) -> str:
        return self.preference.get("persona_item", "")

    @property
    def category(self) -> str:
        return self.preference.get("category", "")


# --- Test-item discovery ---------------------------------------------------

# Per-app selection cap when we have to pick test moments ourselves
# (R8 data — no `split: "test"` labels in the backend anymore).
#
# The legacy pipeline produced a floor of 10 test items per user total
# (see old `build_test_split` docstring). Ten per social app + up to 15
# chatbot moments gives ~45 total for a rich user — within the old range
# but not requiring any LLM call to select.
_PER_APP_SELECTION_CAP = 10
_CHATBOT_SELECTION_CAP = 15
# Same floor the removed pipeline step used:
# `init >= 0.75 AND cross_ref > canonical_xref_threshold(mix)`. We
# approximate the mix-dependent bar with a conservative single floor of
# 20.0 (the XREF_THRESHOLD_EXPLICIT floor) since splitting by mix means
# reading atomic-level counts we no longer persist. For the purposes of
# test-item selection this is strictly MORE selective than the old gate,
# not less — we only lose a few borderline implicit-heavy canonicals.
_MIN_INIT_FOR_TEST = 0.75
_MIN_XREF_FOR_TEST = 20.0
# Stratified Jaccard buckets for distractor selection — replaces the previous
# single "Jaccard <= 0.15" filter. The old design picked seven *trivially*
# topically-disjoint distractors which the model could reject by surface-level
# keyword match; F1 trivially saturated. The new design mixes:
#   - trivial: J <= 0.15 (clearly off-topic)
#   - medium:  0.15 < J <= 0.40 (loosely related)
#   - hard:    0.40 < J <= 0.70 (same topic family, contextually irrelevant)
# All three buckets are still genuinely irrelevant to the test event's specific
# preference — the held-out positive remains the unique correct answer — but
# the model can no longer rely on hashtag overlap alone.
_DISTRACTOR_J_TRIVIAL_MAX: float = 0.15
_DISTRACTOR_J_MEDIUM_MAX: float = 0.40
_DISTRACTOR_J_HARD_MAX: float = 0.70
_DISTRACTOR_QUOTA_TRIVIAL: int = 2
_DISTRACTOR_QUOTA_MEDIUM: int = 3
_DISTRACTOR_QUOTA_HARD: int = 2
_DISTRACTOR_POOL_SIZE: int = (
    _DISTRACTOR_QUOTA_TRIVIAL + _DISTRACTOR_QUOTA_MEDIUM + _DISTRACTOR_QUOTA_HARD
)  # = 7 distractors → 1 held-out + 7 = 8-item pool


def _hashtag_jaccard_norm(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {h.lstrip("#").lower() for h in (a or []) if h}
    sb = {h.lstrip("#").lower() for h in (b or []) if h}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _select_test_items_from_timeline(
    raw_events_by_app: dict[str, list[dict]],
    user_id: str,
) -> list[TestItem]:
    """R8 selector: pick test items when the backend no longer emits
    `split: "test"`. Deterministic, LLM-free.

    Per social app: pick up to `_PER_APP_SELECTION_CAP` positive preferences
    by (confidence_cross_referenced DESC, source_timestamp DESC), de-duped
    by persona_item (one TestItem per canonical, newest occurrence wins).

    Chatbot: pick up to `_CHATBOT_SELECTION_CAP` positive chatbot events
    whose first user turn is a standalone query (the test moment is the
    chatbot turn itself, not a held-out preference per se).

    `over_personalization_irrelevant` is computed on the fly via hashtag
    Jaccard against the full pool of high-confidence preferences (lowest
    overlap wins, up to `_DISTRACTOR_POOL_SIZE`).
    """
    # Step 1: flatten all positive-preference occurrences, annotated with their event.
    pool: list[dict] = []
    for app, events in raw_events_by_app.items():
        for e in events:
            it = e.get("source_interaction_type", "")
            if "positive" not in it:
                continue
            for pref in e.get("preferences", []):
                init = float(pref.get("confidence_score_init") or 0.0)
                xref = float(pref.get("confidence_cross_referenced") or 0.0)
                if init < _MIN_INIT_FOR_TEST or xref < _MIN_XREF_FOR_TEST:
                    continue
                pool.append({
                    "app": app,
                    "event": e,
                    "pref": pref,
                    "persona_item": pref.get("persona_item") or "",
                    "hashtags": list(e.get("source_hashtags") or []),
                    "xref": xref,
                    "ts": int(e.get("source_timestamp") or 0),
                })

    # Step 2: per social app, pick top-N by xref/ts, de-duped by persona_item.
    picked: list[dict] = []
    for app in ("instagram", "facebook", "threads"):
        app_items = [p for p in pool if p["app"] == app]
        # Latest occurrence per canonical
        by_canonical: dict[str, dict] = {}
        for p in app_items:
            key = p["persona_item"]
            if not key:
                continue
            prev = by_canonical.get(key)
            if prev is None or p["ts"] > prev["ts"]:
                by_canonical[key] = p
        ranked = sorted(
            by_canonical.values(),
            key=lambda p: (p["xref"], p["ts"]),
            reverse=True,
        )[:_PER_APP_SELECTION_CAP]
        picked.extend(ranked)

    # Step 3: chatbot picks — pick events whose first user turn looks like
    # a standalone query (ignore events that are pure continuations).
    chatbot_items = [p for p in pool if p["app"] == "chatbot"]
    chatbot_by_event: dict[str, dict] = {}
    for p in chatbot_items:
        eid = str(p["event"].get("source_object_id") or "")
        if not eid:
            continue
        prev = chatbot_by_event.get(eid)
        if prev is None or p["xref"] > prev["xref"]:
            chatbot_by_event[eid] = p
    chatbot_ranked = sorted(
        chatbot_by_event.values(),
        key=lambda p: (p["xref"], p["ts"]),
        reverse=True,
    )[:_CHATBOT_SELECTION_CAP]
    picked.extend(chatbot_ranked)

    # Step 4: build distractor shortlists on the fly. The distractor pool
    # is "all high-confidence preferences OTHER than this test's canonical,
    # with hashtag Jaccard <= _DISTRACTOR_MAX_JACCARD against this test's
    # event hashtags". Keep up to _DISTRACTOR_POOL_SIZE per test item.
    distractor_pool = [
        {
            "persona_item": p["persona_item"],
            "category": p["pref"].get("category", ""),
            "source_hashtags": p["hashtags"],
        }
        for p in pool if p["persona_item"]
    ]
    # De-dup by persona_item
    seen: set[str] = set()
    unique_distractor_pool = []
    for d in distractor_pool:
        if d["persona_item"] in seen:
            continue
        seen.add(d["persona_item"])
        unique_distractor_pool.append(d)

    # Step 5: emit TestItems
    out: list[TestItem] = []
    for p in picked:
        e = p["event"]
        app = p["app"]
        pref = p["pref"]
        test_hashtags = p["hashtags"]
        # Stratified distractor pick: quotas across trivial / medium / hard
        # Jaccard buckets so the agent can't win the over-personalization
        # rejection task by hashtag overlap alone. Within each bucket items
        # are sorted by Jaccard so the most representative items are picked
        # first; ties broken by persona_item lexically for determinism.
        ranked_distractors = sorted(
            (
                (d, _hashtag_jaccard_norm(d["source_hashtags"], test_hashtags))
                for d in unique_distractor_pool
                if d["persona_item"] != p["persona_item"]
            ),
            key=lambda pair: (pair[1], pair[0]["persona_item"] or ""),
        )
        bucket_trivial: list[dict] = []
        bucket_medium: list[dict] = []
        bucket_hard: list[dict] = []
        for d, j in ranked_distractors:
            if j <= _DISTRACTOR_J_TRIVIAL_MAX:
                bucket_trivial.append(d)
            elif j <= _DISTRACTOR_J_MEDIUM_MAX:
                bucket_medium.append(d)
            elif j <= _DISTRACTOR_J_HARD_MAX:
                bucket_hard.append(d)
            # j > _DISTRACTOR_J_HARD_MAX is dropped — too on-topic to be
            # genuinely "irrelevant".

        chosen: list[dict] = []
        chosen.extend(bucket_trivial[:_DISTRACTOR_QUOTA_TRIVIAL])
        chosen.extend(bucket_medium[:_DISTRACTOR_QUOTA_MEDIUM])
        chosen.extend(bucket_hard[:_DISTRACTOR_QUOTA_HARD])
        # Backfill from any remaining bucket if a quota was short — keeps the
        # pool size stable for users with sparse hashtag overlap.
        leftover = (
            bucket_trivial[_DISTRACTOR_QUOTA_TRIVIAL:]
            + bucket_medium[_DISTRACTOR_QUOTA_MEDIUM:]
            + bucket_hard[_DISTRACTOR_QUOTA_HARD:]
        )
        for d in leftover:
            if len(chosen) >= _DISTRACTOR_POOL_SIZE:
                break
            chosen.append(d)
        chosen = chosen[:_DISTRACTOR_POOL_SIZE]
        out.append(TestItem(
            user_id=user_id,
            app=app,
            source_object_id=str(e.get("source_object_id", "")),
            source_timestamp=int(e.get("source_timestamp", 0)),
            formatted_timestamp=e.get("formatted_timestamp", ""),
            source_interaction_type=e.get("source_interaction_type", ""),
            source_hashtags=list(e.get("source_hashtags", [])),
            content=e.get("content", {}),
            interaction_format=e.get("interaction_format", {}),
            preference=pref,
            over_personalization_irrelevant=chosen,
            conversation=e.get("conversation") if app == "chatbot" else None,
            conversation_type=e.get("conversation_type") if app == "chatbot" else None,
        ))
    out.sort(key=lambda t: t.source_timestamp)
    return out


def load_test_items(
    backend_dir: str | Path,
    user_id: str,
    apps: Iterable[str] = APPS,
) -> list[TestItem]:
    """Emit one TestItem per test moment for this user.

    Two selection paths:

    1. **Legacy (pre-R8 backends):** if any preference in the raw events
       carries `split: "test"`, use exactly that filter. Preserves
       reproducibility of benchmarks built against pre-R8 data.

    2. **R8 backends:** data-gen no longer emits `split` or
       `over_personalization_irrelevant`. Fall back to the deterministic
       selector (`_select_test_items_from_timeline`): per-app top-N high-
       confidence preferences (init >= 0.75, xref >= 20.0), time-newest
       first, de-duped by canonical. Distractors are computed on the fly
       via hashtag Jaccard.

    We read the raw JSON (not the stripped BackendQuery view) because the
    harness itself needs the un-stripped preference + hashtags for
    distractor pairing and ground-truth building.
    """
    base = Path(backend_dir) / user_id
    raw_by_app: dict[str, list[dict]] = {}
    for app in apps:
        path = base / f"{app}.json"
        if not path.exists():
            raw_by_app[app] = []
            continue
        with path.open() as f:
            raw_by_app[app] = json.load(f)

    # Legacy path: any `split: "test"` preference present?
    legacy_items: list[TestItem] = []
    has_legacy_split = False
    for app, events in raw_by_app.items():
        for e in events:
            for pref in e.get("preferences", []):
                if pref.get("split") == "test":
                    has_legacy_split = True
                    legacy_items.append(TestItem(
                        user_id=user_id,
                        app=app,
                        source_object_id=str(e.get("source_object_id", "")),
                        source_timestamp=int(e.get("source_timestamp", 0)),
                        formatted_timestamp=e.get("formatted_timestamp", ""),
                        source_interaction_type=e.get("source_interaction_type", ""),
                        source_hashtags=list(e.get("source_hashtags", [])),
                        content=e.get("content", {}),
                        interaction_format=e.get("interaction_format", {}),
                        preference=pref,
                        over_personalization_irrelevant=list(
                            pref.get("over_personalization_irrelevant") or []
                        ),
                        conversation=e.get("conversation") if app == "chatbot" else None,
                        conversation_type=e.get("conversation_type") if app == "chatbot" else None,
                    ))
    if has_legacy_split:
        legacy_items.sort(key=lambda t: t.source_timestamp)
        return legacy_items

    # R8 path: pick test moments from the full timeline
    return _select_test_items_from_timeline(raw_by_app, user_id)


# --- Same-day ground-truth slice (Task B) ----------------------------------

DAY_SECONDS = 24 * 60 * 60


def build_gt_slice(
    bq: BackendQuery,
    test_item: TestItem,
    window_seconds: int = DAY_SECONDS,
) -> dict:
    """Construct the TARGET / AVOID asymmetric slice for a test moment.

    TARGET = held-out positive + other positive preferences across all apps in
             [T_test - window, T_test + window]
    AVOID  = all negative preferences across all apps in the same window

    Contemporaneous (±window) lookup uses the raw backend, bypassing
    BackendQuery's time mask, because this slice is ground truth for scoring —
    not input to the agent.
    """
    user_id = test_item.user_id
    t = test_item.source_timestamp
    lo, hi = t - window_seconds, t + window_seconds
    # Load raw events directly; we need contemporaneous (post-T) items too.
    base = Path(bq.base) / user_id
    target: list[dict] = []
    avoid: list[dict] = []
    for app in APPS:
        path = base / f"{app}.json"
        if not path.exists():
            continue
        with path.open() as f:
            events = json.load(f)
        for e in events:
            ts = e.get("source_timestamp", 0)
            if ts < lo or ts > hi:
                continue
            it = e.get("source_interaction_type", "")
            for p in e.get("preferences", []):
                item = {
                    "persona_item": p.get("persona_item"),
                    "category": p.get("category"),
                    "source_hashtags": e.get("source_hashtags", []),
                    "polarity": "positive" if "positive" in it else "negative" if "negative" in it else "other",
                    "source_app": app,
                    "source_timestamp": ts,
                    "source_interaction_type": it,
                }
                if item["polarity"] == "positive":
                    target.append(item)
                elif item["polarity"] == "negative":
                    avoid.append(item)

    # Make sure the held-out positive is first in TARGET and uniquely identifiable.
    held_out = {
        "persona_item": test_item.preference.get("persona_item"),
        "category": test_item.preference.get("category"),
        "source_hashtags": test_item.source_hashtags,
        "polarity": test_item.polarity,
        "source_app": test_item.app,
        "source_timestamp": test_item.source_timestamp,
        "source_interaction_type": test_item.source_interaction_type,
        "is_held_out": True,
    }
    # De-dup TARGET by persona_item, keeping the held-out version first.
    seen = {held_out["persona_item"]}
    target_dedup = [held_out]
    for item in target:
        if item["persona_item"] not in seen:
            target_dedup.append(item)
            seen.add(item["persona_item"])

    return {
        "t_test": t,
        "window_seconds": window_seconds,
        "target": target_dedup,
        "avoid": avoid,
    }


# --- Long-context serializer (Modes 1b & 2) --------------------------------

def _compact_event(e: dict, strip_preferences: bool = False) -> dict:
    """Trim an event down to what's useful in a long-context prompt."""
    out = {
        "t": e.get("source_timestamp"),
        "when": e.get("formatted_timestamp"),
        "action": e.get("interaction_format", {}).get("action_label"),
        "user_message": e.get("interaction_format", {}).get("user_message"),
        "type": e.get("source_interaction_type"),
        "hashtags": e.get("source_hashtags", []),
        "content_type": e.get("content_type"),
    }
    content = e.get("content") or {}
    if "title" in content:
        out["title"] = content.get("title")
    if "caption" in content:
        out["caption"] = content.get("caption")
    if content.get("overall_description"):
        out["desc"] = content["overall_description"]
    if e.get("conversation"):
        out["conversation"] = e["conversation"]
    if not strip_preferences and e.get("preferences"):
        out["inferred_prefs"] = [
            {"persona_item": p.get("persona_item"), "category": p.get("category")}
            for p in e.get("preferences", [])
        ]
    return {k: v for k, v in out.items() if v not in (None, "", [])}


# --- Token counting --------------------------------------------------------

_TIKTOKEN_ENC = None


def count_tokens(text: str, model: str | None = None) -> int:
    """Best-effort token count. Tries tiktoken; falls back to chars / 4."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken  # type: ignore
            try:
                _TIKTOKEN_ENC = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
            except (KeyError, ValueError):
                _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _TIKTOKEN_ENC = False  # sentinel: library not available
    if _TIKTOKEN_ENC is False:
        return max(1, len(text) // 4)
    return len(_TIKTOKEN_ENC.encode(text))


# --- Token-cost helpers (Phase B) ------------------------------------------

# Canonical keys every task runner can spread into its `metrics` dict for
# uniform cost reporting. cache_read_tokens + cost_usd are subagent-specific
# (Claude Code reports them); for llm_longctx mode they default to 0.
_TOKEN_KEYS: tuple[str, ...] = ("input_tokens", "output_tokens", "cache_read_tokens", "cost_usd")


def token_metrics_from_subagent(stats: dict) -> dict:
    """Pull canonical token keys from a Claude Code subagent stats dict."""
    return {k: stats.get(k) or 0 for k in _TOKEN_KEYS}


def token_metrics_from_text(prompt: str, response: str, model: str | None = None) -> dict:
    """Compute prompt + response token counts for non-subagent (e.g., llm_longctx) modes."""
    return {
        "input_tokens": count_tokens(prompt, model),
        "output_tokens": count_tokens(response, model),
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
    }


def merge_token_metrics(metrics: dict, *, prompt: str, response: str, stats: dict, model: str | None = None) -> dict:
    """Spread token cost into `metrics`. Pulls from subagent stats when present
    (Claude Code mode), else counts locally (llm_longctx). In-place + returns.
    """
    if any(stats.get(k) for k in _TOKEN_KEYS):
        toks = token_metrics_from_subagent(stats)
    else:
        toks = token_metrics_from_text(prompt, response, model)
    metrics.update(toks)
    return metrics


def serialize_history_for_context(
    bq: BackendQuery,
    user_id: str,
    since_timestamp: int,
    model: str | None = None,
    budget_tokens: int | None = None,
) -> tuple[str, dict]:
    """Build the big concatenated history prompt for Modes 1b and 2.

    Returns (text, stats). Annotates each app block with its event count and
    rolling token totals. If the budget is exceeded, reservoir-samples per app
    (warning logged in stats).
    """
    sections: list[str] = []
    per_app_stats: dict[str, dict] = {}
    running_tokens = 0
    truncated = False

    # NOTE: profile preface is DELIBERATELY NOT prepended here. The eval-side
    # firewall (Phase G) hides profile.json from the agent so personalization
    # must be inferred from the event timeline alone — no demographic / app
    # personas / hidden-persona scaffolding that would shortcut the test.

    for app in APPS:
        events = bq.get_events(user_id=user_id, app=app, since_timestamp=since_timestamp)
        compacts = [_compact_event(e) for e in events]
        body = "\n".join(json.dumps(c, ensure_ascii=False) for c in compacts)
        app_tokens = count_tokens(body, model)

        if budget_tokens is not None and running_tokens + app_tokens > budget_tokens:
            # Reservoir-sample: keep a stratified subset that fits.
            remaining = max(0, budget_tokens - running_tokens - 256)
            if remaining > 0 and compacts:
                keep: list[dict] = []
                step = max(1, len(compacts) // max(1, remaining // max(1, app_tokens // max(1, len(compacts)))))
                for i in range(0, len(compacts), step):
                    keep.append(compacts[i])
                body = "\n".join(json.dumps(c, ensure_ascii=False) for c in keep)
                app_tokens = count_tokens(body, model)
            else:
                body = ""
                app_tokens = 0
            truncated = True

        header = f"\n# App: {app} — {len(compacts)} events, ~{app_tokens} tokens (running total before: {running_tokens})\n"
        sections.append(header)
        if body:
            sections.append(body)
        running_tokens += app_tokens
        sections.append(f"\n[After {app}: running total {running_tokens}]\n")
        per_app_stats[app] = {"events": len(compacts), "tokens": app_tokens}

    text = "\n".join(sections)
    stats = {
        "total_tokens": running_tokens,
        "per_app": per_app_stats,
        "budget_tokens": budget_tokens,
        "truncated": truncated,
    }
    return text, stats


# --- Mode dispatch helper (shared across task drivers) ---------------------

def dispatch_agent_run(
    mode: str,
    prompt: str,
    *,
    bq: BackendQuery,
    user_id: str,
    t: int,
    claude_model: str,
    llm_client,
    run_dir: Path | None = None,
    enabled_mcp_apps: tuple[str, ...] = ("instagram", "facebook", "threads", "chatbot"),
) -> tuple[str, int, dict]:
    """Single point of truth for how each inference mode gets an agent response.

    Returns `(text, tool_call_count, subagent_stats_dict)`.

    - `agent_tools`: Claude Code subagent with filesystem Read on a snapshot.
    - `mcp_agent`:   Claude Code subagent with MCP tools (writes to overlay).
    - `llm_longctx`: single QueryLLM call (non-Claude provider baseline).
    """
    from evaluation.claude_subagent import run_subagent

    if mode == "agent_tools":
        snap = materialize_snapshot(bq, user_id, t)
        # Hard timeout on the Claude Code subprocess. Without this, a
        # hung `claude -p` (deadlocked SDK, frozen stream-json output, or
        # any other never-returns case) blocks the worker forever and
        # the parent's `as_completed` never resolves the row's future.
        # Same value used by mcp_agent mode.
        sub = run_subagent(
            prompt=prompt, snapshot_dir=snap, model=claude_model,
            timeout_seconds=600,
        )
        return sub.text, sub.turns, _pack_stats(sub, include_denials=True)

    if mode == "mcp_agent":
        from evaluation.mcp_config_builder import build_mcp_config, mcp_allowed_tools, write_mcp_config
        # Snapshot still used as cwd (Claude Code needs a scope dir even with MCP only),
        # but filesystem tools are denied so it's inert.
        snap = materialize_snapshot(bq, user_id, t)
        run_base = Path(run_dir) if run_dir else Path("benchmark") / user_id / "runs" / "_tmp" / str(t)
        run_base.mkdir(parents=True, exist_ok=True)
        # Honor PM3_OVERLAY_PATH when the sequential harness is driving —
        # one overlay per persona-run so writes accumulate across queries.
        env_overlay = os.environ.get("PM3_OVERLAY_PATH")
        overlay_path = Path(env_overlay) if env_overlay else run_base / f"writes_{t}.jsonl"
        cfg_path = run_base / f"mcp_config_{t}.json"
        cfg = build_mcp_config(
            user_id=user_id, t_test=t,
            overlay_path=overlay_path,
            backend_dir=str(bq.base),
            enabled_apps=enabled_mcp_apps,
        )
        write_mcp_config(cfg_path, cfg)
        mcp_patterns = mcp_allowed_tools(enabled_mcp_apps)
        sub = run_subagent(
            prompt=prompt, snapshot_dir=snap, model=claude_model,
            allowed_tools=(),  # no filesystem tools in MCP mode
            mcp_config_path=cfg_path,
            mcp_tool_patterns=tuple(mcp_patterns),
            timeout_seconds=600,
        )
        # Return the overlay path in stats so the grader can read writes.jsonl.
        stats = _pack_stats(sub, include_denials=True)
        stats["overlay_path"] = str(overlay_path)
        stats["mcp_config_path"] = str(cfg_path)
        return sub.text, sub.turns, stats

    # llm_longctx — non-Claude baseline via QueryLLM.
    if llm_client is None:
        return "", 0, {"error": "llm_longctx mode requires a QueryLLM client but none was passed"}
    response = llm_client.query_llm(prompt) or ""
    # Phase B: count tokens locally so the per-task metrics_json gets
    # input/output token counts even in llm_longctx mode (Claude Code modes
    # already populate these via _pack_stats).
    stats = {
        "input_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(response),
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
    }
    return response, 0, stats


def _pack_stats(sub, include_denials: bool = False) -> dict:
    out = {
        "duration_ms": sub.duration_ms,
        "cost_usd": sub.cost_usd,
        "input_tokens": sub.input_tokens,
        "output_tokens": sub.output_tokens,
        "cache_read_tokens": sub.cache_read_tokens,
    }
    if include_denials:
        out["permission_denials"] = len(sub.permission_denials)
    return out


# --- Snapshot cache (shared time masking) -----------------------------------

class SnapshotCache:
    """Per-test-moment view cache. Modes 1b and 2 reuse the same materialized
    concatenated text across tasks for the same (user_id, T_test, model) key.

    LRU-bounded so a long sequential run (~175 queries × distinct t_test)
    doesn't balloon memory — each cached entry can be several MB.
    """

    MAX_ENTRIES = 8

    def __init__(self, max_entries: int | None = None):
        from collections import OrderedDict
        self._store: OrderedDict[tuple, tuple[str, dict]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries if max_entries is not None else self.MAX_ENTRIES

    def get_or_build(self, bq: BackendQuery, user_id: str, t_test: int, model: str | None, budget: int | None) -> tuple[str, dict]:
        key = (user_id, t_test, model, budget)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        text, stats = serialize_history_for_context(bq, user_id, t_test, model=model, budget_tokens=budget)
        with self._lock:
            self._store[key] = (text, stats)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)
        return text, stats


# --- Focused judge evidence (optional judge layer) -------------------------

def build_judge_evidence(
    bq: BackendQuery,
    test_item: TestItem,
    agent_output: str,
    top_k_prefs: int = 8,
    local_window_days: int = 7,
    max_events: int = 30,
) -> dict:
    """Build the focused evidence slice passed to the LLM judge.

    Centered on (a) the same-day ground-truth slice, (b) top-K relevant
    profile preferences, (c) a local 7-day engagement window filtered to
    hashtag/category overlap with the agent output or retrieved prefs, and
    (d) minimal user context.
    """
    gt = build_gt_slice(bq, test_item)
    profile = bq.get_full_profile(test_item.user_id)
    flat_prefs = profile.get("preferences", []) or []

    # Step 24 of the persona pipeline emits each entry in
    # `profile.preferences` as a timestamp-prefixed STRING (format:
    # "YYYY-MM-DD HH:MM : <persona_item>"), not a dict. Coerce to a
    # uniform dict shape so downstream scoring works regardless of
    # whether a future builder reverts to dicts.
    def _coerce_pref(p) -> dict:
        if isinstance(p, dict):
            return p
        if isinstance(p, str):
            # Strip the optional "<ts> : " prefix.
            text = p
            if " : " in text:
                _, _, text = text.partition(" : ")
            return {"persona_item": text.strip(), "category": "", "source_hashtags": []}
        return {"persona_item": "", "category": "", "source_hashtags": []}

    flat_prefs = [_coerce_pref(p) for p in flat_prefs]

    out_lower = (agent_output or "").lower()

    def rel_score(p: dict) -> float:
        s = 0.0
        item = (p.get("persona_item") or "").lower()
        cat = (p.get("category") or "").lower()
        if item and any(tok in out_lower for tok in item.split() if len(tok) > 3):
            s += 2.0
        if cat and cat in out_lower:
            s += 1.5
        for h in p.get("source_hashtags", []) or []:
            if h.lower().lstrip("#") in out_lower:
                s += 1.0
        return s

    ranked = sorted(flat_prefs, key=rel_score, reverse=True)
    top_prefs = [
        {k: p.get(k) for k in ("persona_item", "category", "source_hashtags") if k in p}
        for p in ranked[:top_k_prefs]
    ]

    since = test_item.source_timestamp - local_window_days * DAY_SECONDS
    local_events = bq.get_events(
        user_id=test_item.user_id,
        app=list(APPS),
        since_timestamp=test_item.source_timestamp,
    )
    filtered = []
    tokens = set(out_lower.split())
    for e in local_events:
        if e.get("source_timestamp", 0) < since:
            continue
        if any(h.lower().lstrip("#") in out_lower for h in e.get("source_hashtags", [])):
            filtered.append(_compact_event(e, strip_preferences=True))
            continue
        cats = {p.get("category", "").lower() for p in e.get("preferences", []) or []}
        if cats & tokens:
            filtered.append(_compact_event(e, strip_preferences=True))
    filtered = filtered[-max_events:]

    return {
        "same_day_slice": gt,
        "top_relevant_preferences": top_prefs,
        "local_engagement_window": filtered,
        "user_context": {
            "name": profile.get("name"),
            "career": profile.get("career"),
            "app_personas": {
                a: {
                    # New schema: delta_summary. Legacy fallback: style_description.
                    "delta_summary": (profile.get("app_personas", {}) or {}).get(a, {}).get("delta_summary")
                                     or (profile.get("app_personas", {}) or {}).get(a, {}).get("style_description"),
                }
                for a in ("Instagram", "Facebook", "Threads", "Chatbot")
            },
        },
    }
