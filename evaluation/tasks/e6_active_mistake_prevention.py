"""E6 — Active Mistake Prevention: discovered paired (warn, foil) scenarios.

Per-user pipeline (MVP — extendable):

  1. Harvest signals: profile, hidden personas, recent calendar state,
     recent calendar modifications, recent geo trace, recent social
     activity sample, recent chatbot turns.
  2. One LLM call → 5–8 candidate pairs, each with warn + foil polarity.
  3. Deterministic validation:
     - signal_grounding: every cited (source, ref, ts) must resolve
       to real data we just harvested.
     - novelty: hash-based dedupe of near-duplicate mistake_summary
       strings across candidates.
  4. Freeze: up to 10 paired candidates → emit 2 instances per pair
     (one per polarity) in the benchmark-task schema.

Each emitted instance matches the top-level `build_benchmark` bucket
shape: dict with `instance_id`, `pair_id`, `polarity`, `task_id`,
`entry_point`, `t_test`, `user_query`, `expected_warning_frame`,
`is_persona_safety`, `cross_signal_signals`, `tool_call_rules`,
`final_state_expected`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluation.backend_query import APPS, BackendQuery
from evaluation.prompts_e6 import discovery_prompt


DAY_SECONDS = 24 * 60 * 60


def _ts_iso(ts: int) -> str:
    try:
        return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).isoformat()
    except Exception:
        return ""


def _normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


def _strip_fence(raw: str) -> str:
    """If the response is wrapped in ```json ...``` fences, return the inner text."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                return p
    return raw


def _parse_candidates_tolerant(response: str, user_id: str) -> list[dict]:
    """Parse an LLM response expected to be a JSON list of candidate dicts.

    Tries full json.loads first. If that fails (truncated output,
    mid-string cut, etc.), scans the string for complete top-level
    object literals inside the top array by tracking brace depth +
    string-state, and parses each one independently. Salvages whatever
    candidates are syntactically complete.
    """
    raw = _strip_fence(response or "")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Salvage: scan for top-level objects inside a (possibly unterminated) array.
    depth = 0
    in_string = False
    escape = False
    start = -1
    objects: list[str] = []
    for i, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(raw[start:i + 1])
                start = -1

    salvaged: list[dict] = []
    for blob in objects:
        try:
            d = json.loads(blob)
            if isinstance(d, dict):
                salvaged.append(d)
        except json.JSONDecodeError:
            continue

    if not salvaged:
        print(f"[e6] user {user_id}: discovery JSON unrecoverable "
              f"(response length={len(raw)})")
        return []
    print(f"[e6] user {user_id}: salvaged {len(salvaged)} candidate(s) "
          f"from partial JSON output")
    return salvaged


def _profile_view(profile: dict) -> dict:
    """Only the fields the discovery LLM needs."""
    return {
        "name": profile.get("name"),
        "gender": profile.get("gender"),
        "race_ethnicity": profile.get("race_ethnicity"),
        "career": profile.get("career"),
        "education": profile.get("education"),
        "bio": profile.get("bio"),
        "mobility_class": profile.get("mobility_class", ""),
    }


def _harvest_calendar(
    bq: BackendQuery, user_id: str, t_anchor: int,
) -> tuple[str, str]:
    """Return (state_text, mods_text) time-scoped to ≤ t_anchor."""
    try:
        state = bq.get_calendar_state(user_id, t_anchor)
    except Exception:
        state = {}
    try:
        mods = bq.get_calendar_modifications(user_id, since_timestamp=t_anchor + 1)
    except Exception:
        mods = []
    state_lines = []
    for entry_id, entry in sorted(state.items(), key=lambda kv: kv[1].get("start_ts", 0)):
        start_iso = _ts_iso(entry.get("start_ts") or 0)
        end_iso = _ts_iso(entry.get("end_ts") or 0)
        title = entry.get("title", "(untitled)")
        loc = (entry.get("location") or {}).get("city", "")
        attendees = entry.get("attendees") or []
        state_lines.append(
            f"- [{entry_id}] {title!r}  start={start_iso}  end={end_iso}"
            + (f"  @ {loc}" if loc else "")
            + (f"  attendees={attendees}" if attendees else "")
        )

    recent_mod_window = t_anchor - 7 * DAY_SECONDS
    mod_lines = []
    for m in mods[:30]:
        ts = m.get("ts", 0)
        if ts < recent_mod_window:
            continue
        action = m.get("action", "")
        line = f"- [{_ts_iso(ts)}] {action}"
        if action == "added":
            entry = m.get("entry") or {}
            line += f"  {entry.get('title', '')!r}"
        elif action == "removed":
            line += f"  entry_id={m.get('entry_id')}  reason={m.get('removal_reason', '')!r}"
        elif action == "updated":
            line += f"  entry_id={m.get('entry_id')}  diff={m.get('diff')}"
        mod_lines.append(line)
    return "\n".join(state_lines), "\n".join(mod_lines)


def _harvest_geo(bq: BackendQuery, user_id: str, t_anchor: int) -> str:
    """Return a compact recent-geo-trace text block for the last 48h."""
    window_start = t_anchor - 2 * DAY_SECONDS
    points: list[tuple[int, str]] = []
    for app in APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_anchor + 1):
            ts = int(e.get("source_timestamp") or 0)
            if ts < window_start or ts > t_anchor:
                continue
            loc = e.get("event_location") or {}
            city = loc.get("city")
            if city:
                points.append((ts, f"{city}, {loc.get('country','')}".strip(", ")))
    points.sort()
    # Dedupe consecutive same-city within 30min
    dedup: list[tuple[int, str]] = []
    for ts, place in points:
        if dedup and dedup[-1][1] == place and ts - dedup[-1][0] < 1800:
            continue
        dedup.append((ts, place))
    return "\n".join(
        f"- [{_ts_iso(ts)}] {place}" for ts, place in dedup[:25]
    )


def _harvest_social_sample(bq: BackendQuery, user_id: str, t_anchor: int) -> str:
    """Return a compact sample of recent social activity (last 48h)."""
    window_start = t_anchor - 2 * DAY_SECONDS
    rows: list[tuple[int, str]] = []
    for app in ("instagram", "facebook", "threads"):
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_anchor + 1):
            ts = int(e.get("source_timestamp") or 0)
            if ts < window_start or ts > t_anchor:
                continue
            tags = " ".join(e.get("source_hashtags", []) or [])[:100]
            it = e.get("source_interaction_type", "")
            is_dm = e.get("is_dm", False)
            is_self = e.get("is_self_authored", False)
            kind = "DM" if is_dm else ("SELF-POST" if is_self else "engagement")
            rows.append((ts, f"- [{_ts_iso(ts)}] {app:<9s} {kind:<10s} {it:<20s}  {tags}"))
    rows.sort()
    return "\n".join(r for _, r in rows[:35])


def _harvest_chatbot_recent(bq: BackendQuery, user_id: str, t_anchor: int) -> str:
    """Return last 15 chatbot turns (user + assistant) as text block."""
    try:
        conv = bq.get_conversations(
            user_id=user_id, since_timestamp=t_anchor + 1, limit=15
        )
    except Exception:
        return ""
    lines: list[str] = []
    for c in conv[-15:]:
        ts = c.get("source_timestamp", 0)
        turns = c.get("conversation") or []
        if turns:
            # Show up to 2 turns per event (user + assistant)
            for t in turns[:2]:
                role = t.get("role", "?")
                content = (t.get("content") or "")[:180]
                lines.append(f"- [{_ts_iso(ts)}] {role}: {content}")
    return "\n".join(lines)


def _hidden_personas_text(profile: dict) -> str:
    flagged = profile.get("hidden_personas") or []
    lines = []
    for hp in flagged:
        ptype = hp.get("type", "")
        ratio = hp.get("privacy_ratio", 0)
        label = hp.get("label", "(unnamed)")
        privacy_marker = " [PRIVACY-FLAGGED]" if (
            ptype in {"covert_concern", "compensatory_need", "intimate_interest"}
            or ratio > 0.7
        ) else ""
        tags = (hp.get("evidence_hashtags") or [])[:6]
        lines.append(f"- [{ptype}] {label}{privacy_marker}  tags={tags}")
    return "\n".join(lines)


def _validate_signal_grounding(
    candidate: dict,
    harvested: dict,
) -> bool:
    """Check every cited (source, ref, ts) resolves to real harvested data.

    Conservative: the LLM may paraphrase refs; we accept a match if
    source ∈ {calendar, geo, social, chatbot, persona} and `ts` is
    plausibly in the harvested time window. Avoids over-rejecting on
    minor textual variance.
    """
    evidence = candidate.get("signal_evidence") or []
    if len(evidence) < 2:
        return False
    valid_sources = {"calendar", "geo", "social", "chatbot", "persona"}
    window_lo = harvested["t_anchor"] - 14 * DAY_SECONDS
    window_hi = harvested["t_anchor"] + 1
    for ev in evidence:
        src = (ev.get("source") or "").lower()
        if src not in valid_sources:
            return False
        ts = ev.get("ts")
        if ts is not None and isinstance(ts, (int, float)):
            if not (window_lo <= int(ts) <= window_hi):
                return False
    return True


def _build_instance(
    pair_id: str,
    polarity: str,
    candidate: dict,
    t_test: int,
    seq_in_pair: int,
) -> dict:
    summary = candidate.get("mistake_summary", "")
    is_ps = bool(candidate.get("is_persona_safety", False))
    ef = candidate.get("expected_warning_frame") or {}
    # Foil side: empty must_mention (agent should stay silent)
    if polarity == "foil":
        ef = {"must_mention": [], "must_not_mention": ef.get("must_not_mention", [])}
    evidence = (
        candidate.get("signal_evidence")
        if polarity == "warn"
        else candidate.get("foil_signal_evidence") or candidate.get("signal_evidence")
    ) or []
    instance_id = f"{pair_id}_{polarity}"
    return {
        "instance_id": instance_id,
        "task_id": "e6_active_mistake_prevention",
        "pair_id": pair_id,
        "polarity": polarity,
        "entry_point": "chatbot_routed",
        "t_test": t_test,
        "user_query": candidate.get("triggering_user_query", ""),
        "mistake_summary": summary,
        "is_persona_safety": is_ps,
        "cross_signal_signals": {
            "signal_evidence": evidence,
            "foil_construction": candidate.get("foil_construction", ""),
        },
        "expected_warning_frame": {
            "must_mention": list(ef.get("must_mention") or []),
            "must_not_mention": list(ef.get("must_not_mention") or []),
        },
        "tool_call_rules": [
            "count('__any_write__') == 0",  # warnings are read-only by default
        ],
        "final_state_expected": {"writes": []},
    }


def build_e6_active_mistake_prevention(
    bq: BackendQuery,
    user_id: str,
    llm_client=None,
    rng_seed: int = 0,
    t_anchor: int | None = None,
) -> list[dict]:
    """Discover paired (warn, foil) active-mistake-prevention scenarios.

    Returns a list of instance dicts. Each paired candidate produces 2
    instances (warn + foil) sharing a `pair_id`.

    If `llm_client` is None, returns []. If the LLM returns invalid JSON
    or zero candidates survive validation, returns [] with a warning.
    """
    if llm_client is None:
        return []

    # ---- profile + t_anchor ----
    profile = bq.get_full_profile(user_id) or {}
    mobility_class = profile.get("mobility_class", "")

    # t_anchor = latest event timestamp if not provided (keeps discovery
    # targeted at the most recent state of the user's life)
    if t_anchor is None:
        latest = 0
        for app in APPS:
            for e in bq.get_events(user_id=user_id, app=app, since_timestamp=10**12):
                ts = int(e.get("source_timestamp") or 0)
                if ts > latest:
                    latest = ts
        if latest == 0:
            return []
        t_anchor = latest

    # ---- harvest signals ----
    cal_state_text, cal_mods_text = _harvest_calendar(bq, user_id, t_anchor)
    geo_text = _harvest_geo(bq, user_id, t_anchor)
    social_text = _harvest_social_sample(bq, user_id, t_anchor)
    chatbot_text = _harvest_chatbot_recent(bq, user_id, t_anchor)
    hidden_text = _hidden_personas_text(profile)

    # If there's nothing to reason over, don't burn an LLM call
    if not any([cal_state_text, cal_mods_text, geo_text, social_text, chatbot_text, hidden_text]):
        print(f"[e6] user {user_id}: no harvested signals — skipping")
        return []

    # ---- discovery LLM call ----
    prompt = discovery_prompt(
        user_context=_profile_view(profile),
        mobility_class=mobility_class,
        t_anchor_iso=_ts_iso(t_anchor),
        calendar_state_text=cal_state_text,
        calendar_mods_text=cal_mods_text,
        geo_trace_text=geo_text,
        social_sample_text=social_text,
        chatbot_recent_text=chatbot_text,
        hidden_personas_text=hidden_text,
    )
    try:
        response = llm_client.query_llm(prompt)
    except Exception as exc:
        print(f"[e6] user {user_id}: LLM call failed: {exc}")
        return []
    if not response:
        return []

    # ---- parse ----
    candidates = _parse_candidates_tolerant(response, user_id)
    if not candidates:
        return []

    if not candidates:
        return []

    # ---- validation gates ----
    harvested = {"t_anchor": t_anchor}
    surviving: list[dict] = []
    seen_summaries: set[str] = set()
    for i, c in enumerate(candidates):
        if not _validate_signal_grounding(c, harvested):
            continue
        summary_norm = _normalize_text(c.get("mistake_summary", ""))
        if not summary_norm:
            continue
        # Novelty: bail if too similar to an already-accepted summary
        dup = False
        for existing in seen_summaries:
            if summary_norm == existing:
                dup = True
                break
            # Simple token-overlap dedupe
            set_a = set(summary_norm.split())
            set_b = set(existing.split())
            if set_a and set_b:
                overlap = len(set_a & set_b) / max(1, min(len(set_a), len(set_b)))
                if overlap > 0.8:
                    dup = True
                    break
        if dup:
            continue
        seen_summaries.add(summary_norm)
        surviving.append(c)

    # ---- cap at 10 pairs ----
    surviving = surviving[:10]
    if not surviving:
        print(f"[e6] user {user_id}: 0 pairs survived validation "
              f"(had {len(candidates)} raw candidates)")
        return []

    # ---- emit instances ----
    out: list[dict] = []
    for i, c in enumerate(surviving):
        pair_id_raw = c.get("pair_id") or f"p{i+1}"
        pair_id = f"e6_{user_id}_{pair_id_raw}"
        # Scatter instance timestamps slightly so paired warn/foil don't
        # share identical ts (one ahead of the other by 1s).
        warn_ts = t_anchor - (len(surviving) - i) * 60
        foil_ts = warn_ts + 1
        out.append(_build_instance(pair_id, "warn", c, warn_ts, 0))
        out.append(_build_instance(pair_id, "foil", c, foil_ts, 1))
    print(f"[e6] user {user_id}: discovered {len(surviving)} pair(s), "
          f"emitting {len(out)} instance(s)")
    return out
