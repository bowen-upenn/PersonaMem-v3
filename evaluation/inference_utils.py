"""Shared plumbing for eval tasks: test-item discovery, GT slice construction,
long-context serialization, token counting, agent-loop runner.

Layered on top of `backend_query.BackendQuery`.
"""

from __future__ import annotations

import json
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

def load_test_items(
    backend_dir: str | Path,
    user_id: str,
    apps: Iterable[str] = APPS,
) -> list[TestItem]:
    """Walk app JSONs and emit one TestItem per preference with split=="test".

    We read the raw JSON (not the stripped BackendQuery view) because the
    harness itself needs the split label + irrelevant distractors.
    """
    base = Path(backend_dir) / user_id
    out: list[TestItem] = []
    for app in apps:
        path = base / f"{app}.json"
        if not path.exists():
            continue
        with path.open() as f:
            events = json.load(f)
        for e in events:
            for pref in e.get("preferences", []):
                if pref.get("split") != "test":
                    continue
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
                    over_personalization_irrelevant=list(pref.get("over_personalization_irrelevant", [])),
                    conversation=e.get("conversation") if app == "chatbot" else None,
                    conversation_type=e.get("conversation_type") if app == "chatbot" else None,
                ))
    out.sort(key=lambda t: t.source_timestamp)
    return out


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

    # Profile preface (safe slice only).
    profile = bq.get_profile_summary(user_id)
    preface = "# User profile\n" + json.dumps(profile, ensure_ascii=False, indent=2)
    preface_tokens = count_tokens(preface, model)
    sections.append(preface + f"\n\n[Preface tokens: {preface_tokens}; running total: {preface_tokens}]\n")
    running_tokens += preface_tokens

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
    - `agent_longctx`: Claude Code subagent, no tools, history in prompt.
    - `llm_longctx`: single QueryLLM call (non-Claude provider baseline).
    """
    from evaluation.claude_subagent import run_subagent

    if mode == "agent_tools":
        snap = materialize_snapshot(bq, user_id, t)
        sub = run_subagent(prompt=prompt, snapshot_dir=snap, model=claude_model)
        return sub.text, sub.turns, _pack_stats(sub, include_denials=True)

    if mode == "agent_longctx":
        snap = materialize_snapshot(bq, user_id, t)
        sub = run_subagent(prompt=prompt, snapshot_dir=snap, model=claude_model, allowed_tools=())
        return sub.text, 0, _pack_stats(sub)

    if mode == "mcp_agent":
        from evaluation.mcp_config_builder import build_mcp_config, mcp_allowed_tools, write_mcp_config
        # Snapshot still used as cwd (Claude Code needs a scope dir even with MCP only),
        # but filesystem tools are denied so it's inert.
        snap = materialize_snapshot(bq, user_id, t)
        run_base = Path(run_dir) if run_dir else Path("benchmark") / user_id / "runs" / "_tmp" / str(t)
        run_base.mkdir(parents=True, exist_ok=True)
        overlay_path = run_base / f"writes_{t}.jsonl"
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
    return (llm_client.query_llm(prompt) or ""), 0, {}


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
    """

    def __init__(self):
        self._store: dict[tuple, tuple[str, dict]] = {}
        self._lock = threading.Lock()

    def get_or_build(self, bq: BackendQuery, user_id: str, t_test: int, model: str | None, budget: int | None) -> tuple[str, dict]:
        key = (user_id, t_test, model, budget)
        with self._lock:
            if key in self._store:
                return self._store[key]
        text, stats = serialize_history_for_context(bq, user_id, t_test, model=model, budget_tokens=budget)
        with self._lock:
            self._store[key] = (text, stats)
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
                a: {"style_description": (profile.get("app_personas", {}) or {}).get(a, {}).get("style_description")}
                for a in ("Instagram", "Facebook", "Threads", "Chatbot")
            },
        },
    }
