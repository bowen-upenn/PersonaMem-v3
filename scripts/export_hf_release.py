#!/usr/bin/env python3
"""Export the 20 eval personas as the PersonaMem-v3 HuggingFace release.

Stages a complete HF dataset repo under --out (default release/hf/):

    README.md               dataset card (v2-style) + YAML viewer configs
    column_descriptions.md  standalone per-column documentation
    samples/                7 preview CSVs (HF-viewer surface; SAMPLES only)
    backend/{uid}/          COMPLETE data, verbatim codebase-ready JSONs
                            (profile, 5 app JSONs, calendar, test, persona.html)

The preview CSVs center on the 3 most interesting personas (--trio, default
3/282/835) and put the persona.html HF link in column #2 of every table.
Full data lives only in backend/ — a download drops in at --backend_dir and
evaluation/run_eval.py runs unmodified.

Usage:
    python scripts/export_hf_release.py                # stage + validate + smoke
    python scripts/export_hf_release.py --upload       # ...then push to HF
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

csv.field_size_limit(sys.maxsize)

USERS = ["1", "2", "3", "5", "6", "8", "9", "10", "13", "14",
         "26", "105", "115", "209", "229", "282", "461", "655", "760", "835"]
TRIO = ["3", "282", "835"]
APPS = ["instagram", "facebook", "threads", "chatbot", "ai_studio"]
PER_PERSONA_FILES = ["profile.json", "instagram.json", "facebook.json", "threads.json",
                     "chatbot.json", "ai_studio.json", "calendar.json", "test.json",
                     "persona.html"]
REPO_ID = "bowen-upenn/PersonaMem-v3"
HF_BLOB = f"https://huggingface.co/datasets/{REPO_ID}/blob/main"
HF_RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

# ---------------------------------------------------------------- history ----

HISTORY_COLUMNS = [
    "persona_id", "persona_html", "event_id", "timestamp", "datetime", "app",
    "event_summary",
    "interaction_type", "action", "action_label", "user_message",
    "content_type", "title", "caption", "media_description", "audio_transcript",
    "hashtags", "conversation_json",
    "author_id", "recipient_id", "relationship", "is_self_authored", "is_dm",
    "is_ad", "is_trending", "location",
    "preferences_summary", "n_preferences", "preferences_json",
    "extras_json", "source_file",
]

# event top-level keys consumed into named columns (everything else -> extras)
_EVENT_CONSUMED = {
    "source_object_id", "source_timestamp", "formatted_timestamp",
    "source_hashtags", "source_interaction_type", "interaction_format",
    "content_type", "content", "conversation", "messages",
    "author_id", "recipient_id", "relationship", "is_self_authored", "is_dm",
    "is_ad", "is_trending", "event_location", "preferences",
}
# content.* keys consumed into named columns (everything else -> extras.content_extra)
_CONTENT_CONSUMED = {"title", "caption", "overall_description", "audio_transcript"}


def _jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _location_str(loc: dict) -> str:
    if not isinstance(loc, dict):
        return ""
    parts = [loc.get(k) for k in ("city", "region", "country")]
    return ", ".join(p for p in parts if p)


def _iso_utc(ts) -> str:
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return ""


def _event_summary(e: dict, fmt: dict, content: dict, conversation, caption) -> str:
    """One human-readable sentence describing the row — the PR column."""
    label = fmt.get("action_label") or fmt.get("action") or "Engaged"
    tags = [str(t).lstrip("#") for t in (e.get("source_hashtags") or [])][:3]
    tag_str = " ".join(f"#{t}" for t in tags)

    prefix = "[Sponsored] " if e.get("is_ad") else ("[Trending] " if e.get("is_trending") else "")
    if e.get("thread_id") or (e.get("is_dm") and e.get("messages") is not None):
        n = len(e.get("messages") or [])
        kind = "group DM" if e.get("is_group_dm") else "DM thread"
        who = ", ".join(p for p in (e.get("participants") or []) if p != "self")[:60]
        return f"{prefix}{kind} with {who or 'a friend'} ({n} messages)"
    if conversation is not None and not e.get("is_dm"):
        first_user = next((t.get("content", "") for t in conversation
                           if isinstance(t, dict) and t.get("role") == "user"), "")
        kind = e.get("conversation_type") or "conversation"
        opener = (first_user[:80] + "…") if len(first_user) > 80 else first_user
        return f"{prefix}{kind.replace('_', ' ').capitalize()} session — “{opener}”" if opener \
            else f"{prefix}{kind.replace('_', ' ').capitalize()} session"
    obj = content.get("title") or ""
    if obj:
        obj = f"“{obj[:70]}”"
    elif caption:
        c = caption.strip().replace("\n", " ")
        obj = f"“{(c[:70] + '…') if len(c) > 70 else c}”"
    elif tag_str:
        obj = f"about {tag_str}"
    out = label
    if obj.startswith("“"):
        out = f"{label}: {obj}"
        if tag_str:
            out += f" ({tag_str})"
    elif obj:  # "about #tags"
        out = f"{label} — {obj}"
    return prefix + out


def _prefs_summary(prefs: list) -> str:
    items = [p.get("persona_item", "") for p in prefs if p.get("persona_item")]
    seen, uniq = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            uniq.append(it)
    head = "; ".join(uniq[:3])
    more = len(uniq) - 3
    return head + (f"; (+{more} more)" if more > 0 else "")


def flatten_event(uid: str, app: str, e: dict) -> OrderedDict:
    fmt = e.get("interaction_format") or {}
    content = e.get("content") or {}
    caption = content.get("caption")
    if not caption and content.get("text"):
        caption = content.get("text")  # text posts; raw `text` kept in extras
    conversation = e.get("conversation")
    if conversation is None and e.get("messages") is not None:
        conversation = e.get("messages")  # DM threads
    hashtags = e.get("source_hashtags") or []

    extras: dict = {}
    for k, v in e.items():
        if k not in _EVENT_CONSUMED:
            extras[k] = v
    content_extra = {k: v for k, v in content.items() if k not in _CONTENT_CONSUMED}
    if content.get("text") is not None:
        content_extra["text"] = content["text"]  # keep raw text lossless
    if content_extra:
        extras["content_extra"] = content_extra
    if e.get("event_location"):
        extras["event_location"] = e["event_location"]
    if e.get("formatted_timestamp"):
        extras["formatted_timestamp"] = e["formatted_timestamp"]  # datetime col is re-rendered ISO
    if any((not isinstance(t, str)) or (" " in t) for t in hashtags):
        extras["source_hashtags_raw"] = hashtags

    prefs = e.get("preferences") or []
    row = OrderedDict()
    row["persona_id"] = uid
    row["persona_html"] = f"{HF_RESOLVE}/backend/{uid}/persona.html?download=true"
    row["event_id"] = _cell(e.get("source_object_id"))
    row["timestamp"] = _cell(e.get("source_timestamp"))
    row["datetime"] = _iso_utc(e.get("source_timestamp"))
    row["app"] = fmt.get("app") or app.replace("ai_studio", "AI_Studio").capitalize()
    row["event_summary"] = _event_summary(e, fmt, content, conversation, caption)
    row["interaction_type"] = _cell(e.get("source_interaction_type"))
    row["action"] = _cell(fmt.get("action"))
    row["action_label"] = _cell(fmt.get("action_label"))
    row["user_message"] = _cell(fmt.get("user_message"))
    row["content_type"] = _cell(e.get("content_type"))
    row["title"] = _cell(content.get("title"))
    row["caption"] = _cell(caption)
    row["media_description"] = _cell(content.get("overall_description"))
    row["audio_transcript"] = _cell(content.get("audio_transcript"))
    row["hashtags"] = " ".join(f"#{str(t).lstrip('#')}" for t in hashtags)
    row["conversation_json"] = _jdump(conversation) if conversation is not None else ""
    row["author_id"] = _cell(e.get("author_id"))
    row["recipient_id"] = _cell(e.get("recipient_id"))
    row["relationship"] = _cell(e.get("relationship"))
    # Emit literal booleans even when the source omits the key (Chatbot /
    # AI-Studio events) so every split types these columns identically.
    row["is_self_authored"] = _cell(bool(e.get("is_self_authored")))
    row["is_dm"] = _cell(bool(e.get("is_dm")))
    row["is_ad"] = _cell(e.get("is_ad", False))
    row["is_trending"] = _cell(e.get("is_trending", False))
    row["location"] = _location_str(e.get("event_location") or {})
    row["preferences_summary"] = _prefs_summary(prefs)
    row["n_preferences"] = str(len(prefs))
    row["preferences_json"] = _jdump(prefs) if prefs else ""
    row["extras_json"] = _jdump(extras) if extras else ""
    row["source_file"] = f"backend/{uid}/{app}.json"
    return row


def event_coverage_check(uid: str, app: str, e: dict, row: OrderedDict) -> list[str]:
    """Assert every piece of the event is present in the row (named column or
    extras). Returns a list of problems (empty = fully covered)."""
    problems = []
    extras = json.loads(row["extras_json"]) if row["extras_json"] else {}
    # 1. every top-level key consumed or in extras
    for k in e:
        if k not in _EVENT_CONSUMED and k not in extras:
            problems.append(f"{uid}/{app}: top-level key {k!r} lost")
    # 2. content keys consumed or in extras.content_extra
    content = e.get("content") or {}
    ce = extras.get("content_extra", {})
    for k in content:
        if k not in _CONTENT_CONSUMED and k not in ce:
            problems.append(f"{uid}/{app}: content key {k!r} lost")
    # 3. mapped scalar equality
    checks = [
        ("event_id", e.get("source_object_id")),
        ("timestamp", e.get("source_timestamp")),
        ("interaction_type", e.get("source_interaction_type")),
        ("action", (e.get("interaction_format") or {}).get("action")),
        ("title", content.get("title")),
        ("media_description", content.get("overall_description")),
    ]
    for col, want in checks:
        if want is not None and row[col] != str(want):
            problems.append(f"{uid}/{app}: column {col!r} mismatch")
    # 4. structured payloads survive
    if (e.get("preferences") or []) and json.loads(row["preferences_json"]) != e["preferences"]:
        problems.append(f"{uid}/{app}: preferences drifted")
    if e.get("conversation") is not None and json.loads(row["conversation_json"]) != e["conversation"]:
        problems.append(f"{uid}/{app}: conversation drifted")
    if e.get("messages") is not None and json.loads(row["conversation_json"]) != e["messages"]:
        problems.append(f"{uid}/{app}: DM messages drifted")
    if e.get("event_location") and extras.get("event_location") != e["event_location"]:
        problems.append(f"{uid}/{app}: event_location drifted")
    if e.get("formatted_timestamp") and extras.get("formatted_timestamp") != e["formatted_timestamp"]:
        problems.append(f"{uid}/{app}: formatted_timestamp lost")
    # 5. hashtags reversible
    got = [t.lstrip("#") for t in row["hashtags"].split()] if row["hashtags"] else []
    want_tags = [str(t).lstrip("#") for t in (e.get("source_hashtags") or [])]
    if got != want_tags and "source_hashtags_raw" not in extras:
        problems.append(f"{uid}/{app}: hashtags not reversible")
    return problems


# ---------------------------------------------------------------- queries ----

QUERY_COLUMNS = [
    "persona_id", "persona_html", "query_id", "task_family", "task_type",
    "what_this_tests", "query_kind", "expected_behavior", "timestamp", "datetime",
    "user_query", "prior_conversation",
    "groundtruth_preference", "groundtruth_preference_obj", "distractor_preferences",
    "example_response", "inferior_response", "reference_example",
    "rubric_tags", "tool_call", "internal_checks", "instance_full", "source_file",
]

_QA_INTERNAL = ["example_response_self_check", "example_response_voice_evidence",
                "inferior_response_voice_evidence", "voice_evidence_smoke_check",
                "voice_evidence_smoke_check_after_regen"]
_QUERY_CONSUMED = {
    "query_id", "task_family", "task_type", "query_kind", "expected_behavior",
    "ts", "ts_iso", "user_query", "prior_conversation", "groundtruth_preference",
    "groundtruth_preference_obj", "distractor_preferences", "example_response",
    "inferior_response", "reference_example", "rubric_tags", "tool_call",
    "instance_full", *_QA_INTERNAL,
}


def flatten_query(uid: str, r: dict) -> OrderedDict:
    row = OrderedDict()
    row["persona_id"] = uid
    row["persona_html"] = f"{HF_RESOLVE}/backend/{uid}/persona.html?download=true"
    row["query_id"] = _cell(r.get("query_id"))
    row["task_family"] = _cell(r.get("task_family"))
    row["task_type"] = _cell(r.get("task_type"))
    row["what_this_tests"] = TASK_DESCRIPTIONS.get(r.get("task_type") or "", "")
    row["query_kind"] = _cell(r.get("query_kind"))
    row["expected_behavior"] = _cell(r.get("expected_behavior"))
    row["timestamp"] = _cell(r.get("ts"))
    row["datetime"] = _cell(r.get("ts_iso"))
    row["user_query"] = _cell(r.get("user_query"))
    # top-level prior_conversation is null on every row; the real mid-session
    # turns live inside instance_full — surface them so the column is useful.
    pc = r.get("prior_conversation") or (r.get("instance_full") or {}).get("prior_conversation")
    row["prior_conversation"] = _jdump(pc) if pc else ""
    for col, key in [
                     ("groundtruth_preference_obj", "groundtruth_preference_obj"),
                     ("distractor_preferences", "distractor_preferences"),
                     ("reference_example", "reference_example"),
                     ("rubric_tags", "rubric_tags"),
                     ("tool_call", "tool_call")]:
        v = r.get(key)
        row[col] = _jdump(v) if v not in (None, [], {}) else ""
    def _text_or_json(v):
        if v is None:
            return ""
        return v if isinstance(v, str) else _jdump(v)  # ranking payloads are dicts
    row["groundtruth_preference"] = _text_or_json(r.get("groundtruth_preference"))
    row["example_response"] = _text_or_json(r.get("example_response"))
    row["inferior_response"] = _text_or_json(r.get("inferior_response"))
    internal = {k: r.get(k) for k in _QA_INTERNAL if r.get(k) is not None}
    row["internal_checks"] = _jdump(internal) if internal else ""
    row["instance_full"] = _jdump(r.get("instance_full")) if r.get("instance_full") is not None else ""
    row["source_file"] = f"backend/{uid}/test.json"
    # column order defined by QUERY_COLUMNS
    return OrderedDict((c, row[c]) for c in QUERY_COLUMNS)


def query_coverage_check(uid: str, r: dict, row: OrderedDict) -> list[str]:
    problems = [f"{uid}: query key {k!r} lost" for k in r if k not in _QUERY_CONSUMED]
    if r.get("instance_full") is not None and json.loads(row["instance_full"]) != r["instance_full"]:
        problems.append(f"{uid}/{r.get('query_id')}: instance_full drifted")
    for col, key in [("query_id", "query_id"), ("task_type", "task_type"),
                     ("user_query", "user_query")]:
        want = r.get(key)
        if want is not None and row[col] != str(want):
            problems.append(f"{uid}/{r.get('query_id')}: column {col!r} mismatch")
    for col in ("example_response", "inferior_response", "groundtruth_preference"):
        want = r.get(col)
        if want is None:
            continue
        got = row[col] if isinstance(want, str) else json.loads(row[col])
        if got != want:
            problems.append(f"{uid}/{r.get('query_id')}: column {col!r} drifted")
    return problems


# ---------------------------------------------------------------- sampling ---

def _special_class(e: dict) -> str | None:
    fmt = e.get("interaction_format") or {}
    act = fmt.get("action") or ""
    if e.get("is_ad"):
        return "ad"
    if e.get("is_trending"):
        return "trending"
    if e.get("is_group_dm"):
        return "group_dm"
    if e.get("is_dm"):
        return "dm"
    if e.get("is_self_authored"):
        return "self_post"
    if isinstance(act, str) and act.startswith("at_ai"):
        return "at_ai"
    if e.get("_planted_sensitive_event"):
        return "sensitive"
    if e.get("source_interaction_type") == "feed_visible":
        return "feed_visible"
    return None


def _stride_sample(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _richness(e: dict) -> int:
    """Preview desirability: content-bearing and preference-bearing rows first."""
    content = e.get("content") or {}
    score = 0
    if content.get("title") or content.get("caption") or content.get("text"):
        score += 2
    if e.get("conversation") is not None or e.get("messages") is not None:
        score += 2
    if e.get("preferences"):
        score += 1
    return score


def sample_history(events_by_app: dict[str, list[dict]], per_persona: int) -> dict[str, list[dict]]:
    """Preview sampler: ~70% of each app's quota from content/preference-rich
    events, ~30% stride over everything (so skips/lingers stay visible), plus
    one forced example of every special class present. Deterministic."""
    total = sum(len(v) for v in events_by_app.values()) or 1
    picked: dict[str, list[dict]] = {}
    for app, evs in events_by_app.items():
        quota = max(3, round(per_persona * len(evs) / total)) if evs else 0
        rich = [e for e in evs if _richness(e) >= 2]
        n_rich = min(len(rich), max(1, round(quota * 0.7))) if rich else 0
        chosen = _stride_sample(rich, n_rich)
        seen = {id(e) for e in chosen}
        rest = [e for e in evs if id(e) not in seen]
        chosen += _stride_sample(rest, max(0, quota - len(chosen)))
        picked[app] = chosen
    have = {_special_class(e) for evs in picked.values() for e in evs}
    for app, evs in events_by_app.items():
        for e in evs:
            cls = _special_class(e)
            if cls and cls not in have:
                picked[app].append(e)
                have.add(cls)
    for app in picked:
        picked[app].sort(key=lambda e: e.get("source_timestamp") or 0)
    return picked


def sample_queries(all_rows: list[tuple[str, dict]], per_type_floor: int = 5,
                   per_type_cap: int = 6) -> list[tuple[str, dict]]:
    """Trio-first, backfilled so every task_type appears; prefers rows with a
    non-empty gold+inferior pair (they preview best)."""
    by_type: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for uid, r in all_rows:
        by_type[r.get("task_type") or "?"].append((uid, r))

    def rank(item):
        uid, r = item
        trio = 0 if uid in TRIO else 1
        pair = 0 if (r.get("example_response") and r.get("inferior_response")) else 1
        return (trio, pair, uid, r.get("query_id") or "")

    out: list[tuple[str, dict]] = []
    for tt in sorted(by_type):
        rows = sorted(by_type[tt], key=rank)
        n_take = max(per_type_floor, min(per_type_cap, len(rows)))
        # Round-robin across personas (trio members first) so no single persona
        # dominates the preview; then guarantee >=1 mid-session row (non-empty
        # prior_conversation) per type when one exists.
        by_uid: dict[str, list] = defaultdict(list)
        order: list[str] = []
        for uid, r in rows:
            if uid not in by_uid:
                order.append(uid)
            by_uid[uid].append((uid, r))
        take: list[tuple[str, dict]] = []
        while len(take) < n_take and any(by_uid.values()):
            for uid in order:
                if by_uid[uid] and len(take) < n_take:
                    take.append(by_uid[uid].pop(0))
        def _pc(r):
            return r.get("prior_conversation") or (r.get("instance_full") or {}).get("prior_conversation")
        if not any(_pc(r) for _, r in take):
            mid = next(((u, r) for u, r in rows if _pc(r)), None)
            if mid:
                take[-1] = mid
        out.extend(take)
    return out


# ---------------------------------------------------------------- staging ----

def stage(out: Path, per_persona_hist: int) -> dict:
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (out / "backend").mkdir(exist_ok=True)

    stats: dict = {"events_per_app": Counter(), "n_pref_instances": 0,
                   "task_type_counts": Counter(), "task_family_counts": Counter(),
                   "problems": []}

    # -- backend/ verbatim copies + full-data census -------------------------
    for uid in USERS:
        src = REPO_ROOT / "backend" / uid
        dst = out / "backend" / uid
        dst.mkdir(exist_ok=True)
        for f in PER_PERSONA_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
            else:
                stats["problems"].append(f"MISSING source file backend/{uid}/{f}")

    # -- history flatten + coverage over ALL events; sample the trio ---------
    # Per-app previews sample INDEPENDENTLY from the full per-app pools (the
    # aggregate table's proportional quota would starve small apps like
    # AI Studio, which is a flagship surface).
    per_app_pp = {"instagram": 150, "facebook": 150, "threads": 150,
                  "chatbot": 130, "ai_studio": 400}
    all_rows_by_app: dict[str, list[OrderedDict]] = defaultdict(list)   # trio samples
    agg_rows: list[OrderedDict] = []
    for uid in USERS:
        events_by_app = {}
        for app in APPS:
            evs = json.loads((out / "backend" / uid / f"{app}.json").read_text())
            events_by_app[app] = evs
            stats["events_per_app"][app] += len(evs)
            for e in evs:
                stats["n_pref_instances"] += len(e.get("preferences") or [])
                row = flatten_event(uid, app, e)
                stats["problems"].extend(event_coverage_check(uid, app, e, row))
        if uid in TRIO:
            picked = sample_history(events_by_app, per_persona_hist)
            for app, evs in picked.items():
                agg_rows.extend(flatten_event(uid, app, e) for e in evs)
            for app, evs in events_by_app.items():
                chosen = sample_history({app: evs}, per_app_pp[app])[app]
                all_rows_by_app[app].extend(flatten_event(uid, app, e) for e in chosen)
    agg_rows.sort(key=lambda r: (TRIO.index(r["persona_id"]), int(r["timestamp"] or 0)))

    def write_csv(path: Path, cols: list[str], rows: list[OrderedDict]):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv(samples / "history_all_apps_sample.csv", HISTORY_COLUMNS, agg_rows)
    per_app_caps = {"instagram": 450, "facebook": 450, "threads": 450,
                    "chatbot": 400, "ai_studio": 400}
    for app in APPS:
        rows = all_rows_by_app[app][:per_app_caps[app]]
        write_csv(samples / f"history_{app}_sample.csv", HISTORY_COLUMNS, rows)
        stats[f"sample_{app}"] = len(rows)
    stats["sample_all_apps"] = len(agg_rows)

    # -- queries flatten + coverage over ALL rows; trio-first sample ---------
    all_q: list[tuple[str, dict]] = []
    for uid in USERS:
        rows = json.loads((out / "backend" / uid / "test.json").read_text())
        for r in rows:
            stats["task_type_counts"][r.get("task_type")] += 1
            stats["task_family_counts"][r.get("task_family")] += 1
            flat = flatten_query(uid, r)
            stats["problems"].extend(query_coverage_check(uid, r, flat))
            all_q.append((uid, r))
    q_sample = sample_queries(all_q)
    write_csv(samples / "queries_sample.csv", QUERY_COLUMNS,
              [flatten_query(uid, r) for uid, r in q_sample])
    stats["sample_queries"] = len(q_sample)
    stats["n_queries"] = len(all_q)
    stats["n_events"] = sum(stats["events_per_app"].values())
    return stats


# ------------------------------------------------------------------- card ----

TASK_DESCRIPTIONS = {
    "personalized_recommendation": "Rank a content slate so items matching the user's deep preferences beat surface-similar distractors",
    "hidden_persona_recommendation": "Surface the item resonating with an implicit (never-stated) persona layer",
    "hidden_persona_implicit_qa": "Answer questions whose ground truth is an implicit persona signal",
    "chatbot_personalized_response": "Free-form chatbot reply personalized to the user's history and voice",
    "new_suggestions_chatbot": "Propose something genuinely NEW that fits the persona (chatbot surface)",
    "personal_qa_hallucination": "Answer personal questions without inventing facts the history does not support",
    "short_vs_long_term_lifecycle": "Distinguish short-term intents (expired after their stop condition) from long-term tastes",
    "preference_shift_followthrough": "Follow the LATEST stance after a preference shift, not the outdated one",
    "at_ai_directive_followup": "Honor an earlier @ai comment directive (more-of / stop) when ranking later content",
    "over_personalization_chatbot_text": "Answer a generic question WITHOUT dragging in irrelevant personal details",
    "over_personalization_context_shift": "Stay on the NEW topic after a context shift instead of over-fitting to history",
    "over_personalization_sensitive_event": "Do not surface a sensitive life event in response to a benign query",
    "over_personalization_sycophancy": "Resist agreeing with false facts / fabricated memories / self-serving framing",
    "over_personalization_repetition_chatbot": "Avoid recommending the same saturated interest yet again (chatbot)",
    "over_personalization_repetition_recsys": "Avoid recommending the same saturated interest yet again (slate)",
    "restraint_sensitive_event_silence": "Proactive agent stays SILENT about a user's sensitive period",
    "active_mistake_prevention": "Proactively warn when the user is about to act against their own calendar/preferences",
    "agentic_auto_reply": "Draft an auto-reply in the user's own voice",
    "agentic_send_post": "Compose and send a post in the user's voice on the right app",
    "agentic_cross_app_repost": "Repost content across apps, re-voiced for the target app's audience",
    "agentic_community_post": "Write a community post matching the user's voice and the community's norms",
    "agentic_dm_digest": "Summarize a DM thread for the user",
    "agentic_group_dm_summary": "Summarize a group DM for the user",
    "agentic_vague_refind": "Re-find a specific past post from a vague description",
    "agentic_proactive_daily_catchup": "Compose the daily catch-up brief the user would want",
    "agentic_trending_alert": "Alert only on trending topics this user would care about (not ones they dislike)",
    "local_recommendation_geo_shift": "Adapt recommendations when the user travels to a different city",
    "proactive_close_friend_update": "React (or deliberately not) to a close friend's update",
    "proactive_friend_feed_react": "Choose whether/how to engage with a friend's feed item",
    "proactive_trending_feed_react": "Choose whether/how to engage with a trending feed item",
    "proactive_overactive_check": "Recognize when the proactive agent is doing TOO much and back off",
}


def write_card(out: Path, stats: dict):
    ev = stats["events_per_app"]
    tt = stats["task_type_counts"]
    tf = stats["task_family_counts"]
    trio_links = " · ".join(
        f"[persona {u}]({HF_RESOLVE}/backend/{u}/persona.html?download=true)" for u in TRIO)

    yaml = f"""---
license: cc-by-nc-4.0
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- personalization
- memory
- long-context
- agents
- recommendation
- user-modeling
pretty_name: PersonaMem-v3
size_categories:
- 10K<n<100K
configs:
- config_name: history_all_apps
  default: true
  data_files:
  - split: sample
    path: samples/history_all_apps_sample.csv
- config_name: history_per_app
  data_files:
  - split: instagram
    path: samples/history_instagram_sample.csv
  - split: facebook
    path: samples/history_facebook_sample.csv
  - split: threads
    path: samples/history_threads_sample.csv
  - split: chatbot
    path: samples/history_chatbot_sample.csv
  - split: ai_studio
    path: samples/history_ai_studio_sample.csv
- config_name: queries
  data_files:
  - split: sample
    path: samples/queries_sample.csv
---
"""

    task_rows = []
    for fam in sorted(tf):
        fam_types = sorted(t for t in tt if _family_of(t, fam))
        for t in fam_types:
            task_rows.append(f"| {fam} | `{t}` | {tt[t]} | {TASK_DESCRIPTIONS.get(t, '')} |")
    task_table = "\n".join(task_rows)

    body = f"""# PersonaMem-v3: Omni-Platform Personal Intelligence — Benchmark Data

Third release in the PersonaMem series
([v1](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) ·
[v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) ·
[code](https://github.com/bowen-upenn/PersonaMem-v3)).

## ⚠️ Warning

All data are LLM-generated (synthetic personas — no real users) with automated
quality gates and audits, but we cannot manually verify every sample; the
dataset may still contain stereotypical biases or other imperfect content.

## ✨ Key Features

- **20 rich personas × 5 platforms** — Instagram, Facebook, Threads, an AI
  Chatbot (utility assistant), and **AI Studio** (companion-character chat with
  cross-session memory and a relationship arc).
- **{stats['n_events']:,} engagement events** ({ev['instagram']:,} Instagram /
  {ev['facebook']:,} Facebook / {ev['threads']:,} Threads / {ev['chatbot']:,}
  Chatbot / {ev['ai_studio']:,} AI Studio): feed engagements with full content
  (titles, captions, media descriptions, transcripts), self-posts, DM threads,
  group DMs, sponsored ads, trending items, `@ai` comment directives, chatbot
  and companion conversations — each event stamped with time and location.
- **{stats['n_pref_instances']:,} inferred preference instances** attached to
  events, each with confidence scores, time horizon (short- vs long-term with
  stop conditions), update history (reinforced / contradicted / faded), and
  links to **hidden personas** — deeper motivational layers (aspirations,
  identity anchors, parasocial attachments, covert concerns, one synthetic
  sensitive-life-event episode per persona).
- **Per-user voice**: every user has ONE consistent writing voice (idiolect,
  emoji palette, register) modulated per app — self-posts, DMs, chatbot turns
  and `@ai` comments are all written in it.
- **{stats['n_queries']:,} benchmark queries across {len(tt)} task types / {len(tf)} families**
  — personalization, hidden-persona reasoning, over-personalization restraint
  (incl. sycophancy resistance), preference lifecycle, agentic voice-authoring
  tasks, geo-aware recommendation, and proactive-agent judgment calls.
- **Time-masked evaluation protocol**: for every query, the system under test
  may only see events with `timestamp < query.timestamp`. `profile.json` is
  the scorer-side ground truth — it is **never shown to the evaluated agent**.

## 👀 Browse a persona in one click

Each persona ships a self-contained HTML page with its full timeline, inferred
preference profile, and sample test cards. HF does not render HTML in the
browser, so the links download the file — open it locally with one click:
{trio_links}, or any `backend/{{persona_id}}/persona.html`. Every CSV row
carries its persona's download link in column 2 (`persona_html`).

## 🗂️ Dataset Contents

### Core: preview tables (`samples/` — what the Dataset Viewer shows)

The CSVs are **samples for browsing**, centered on three representative
personas (3, 282, 835 — chosen for richness and diversity of surfaces). The
history tables cover exactly those three; the queries table draws mainly from
them and additionally from other personas so that all 31 task types are
represented. The
complete data lives in `backend/`. One shared schema per table
(see `column_descriptions.md`):

| File | Rows | What a row is |
|---|---|---|
| `samples/history_all_apps_sample.csv` | {stats['sample_all_apps']} | One engagement event, all 5 apps interleaved chronologically |
| `samples/history_{{app}}_sample.csv` × 5 | {stats['sample_instagram']}/{stats['sample_facebook']}/{stats['sample_threads']}/{stats['sample_chatbot']}/{stats['sample_ai_studio']} | Same schema, split per app (load one split at a time — text columns that never apply to an app are empty in its split) |
| `samples/queries_sample.csv` | {stats['sample_queries']} | One benchmark query with ground truth; most rows carry a gold + inferior response pair (judge-scored tasks ship without an inferior by design). All {len(tt)} task types represented |

### Supplementary: the complete dataset (`backend/` — codebase-ready)

`backend/{{persona_id}}/` contains the **full** data for each of the 20
personas, in exactly the layout the
[PersonaMem-v3 codebase](https://github.com/bowen-upenn/PersonaMem-v3) reads:

| File | Contents |
|---|---|
| `instagram/facebook/threads/chatbot/ai_studio.json` | All engagement events (chronological) with nested inferred preferences |
| `calendar.json` | Calendar modification stream (fold entries with `ts <= T` to get the calendar at time T) |
| `profile.json` | Ground-truth persona: demographics, voice, app personas, AI-companion character, hidden personas, flat preference list. **Scorer-side only — never shown to the evaluated agent** |
| `test.json` | All benchmark queries for this persona (the queries CSV is a sample of these) |
| `persona.html` | Self-contained human-browsable view of everything above |

## 🚀 Run the benchmark

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; \\
           snapshot_download('{REPO_ID}', repo_type='dataset', local_dir='pm3_data')"
git clone https://github.com/bowen-upenn/PersonaMem-v3 && cd PersonaMem-v3
python evaluation/run_eval.py --backend_dir ../pm3_data/backend --user_id 8 --mode llm_longctx
```

## 📊 Task types

Families: `personalization` (core recommendation), `chatbot_response`,
`new_suggestions` (explorative recommendation), `over_personalization`
(restraint), `e_followup` (time-anchored follow-up: @ai directives, preference
lifecycle, sensitive-event restraint, geo shift), `proactive_actions`
(act-vs-stay-silent judgment), `agentic` (voice-authoring + tool tasks).

| Family | Task type | Rows (full dataset) | What it tests |
|---|---|---|---|
{task_table}

## 📜 License

The personas are derived from
[facebook/gistbench](https://huggingface.co/datasets/facebook/gistbench) and
inherit its **CC-BY-NC-4.0** license (attribution, non-commercial). The
[benchmark code](https://github.com/bowen-upenn/PersonaMem-v3) is MIT.

## Citation

If you use PersonaMem-v3, please cite the series:

- PersonaMem (v1), COLM 2025 — *Know Me, Respond to Me* ([arXiv:2504.14225](https://arxiv.org/abs/2504.14225))
- PersonaMem-v2 — *Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory* ([arXiv:2512.06688](https://arxiv.org/abs/2512.06688))
- PersonaMem-v3 — paper forthcoming; cite the dataset URL meanwhile.
"""
    (out / "README.md").write_text(yaml + body)
    write_column_descriptions(out)


def _family_of(task_type: str, fam: str) -> bool:
    # task_family is carried on rows; stats counters keyed independently —
    # regenerate the mapping from the staged queries at card time.
    return _TASK2FAM.get(task_type) == fam


_TASK2FAM: dict = {}


def write_column_descriptions(out: Path):
    txt = """# Column Descriptions

The seven preview files form three tables — the queries table, the all-apps
history table, and five per-app files sharing the history schema. Every table
puts `persona_id` first and `persona_html` (a link to that persona's browsable
HTML page inside this repo) second. Empty cells mean
the field does not apply to that row. Columns holding nested structures are
JSON-encoded strings.

## History tables (`history_all_apps_sample.csv` + the 5 per-app files, one shared schema)

### Identity & pointers
| Column | Description |
|---|---|
| `persona_id` | Persona identifier (also the `backend/{persona_id}/` folder name) |
| `persona_html` | Link to this persona's one-page browsable view |
| `event_id` | Source interaction id (unique per persona) |
| `event_summary` | **Start here** — one auto-composed human sentence describing the row (e.g. `Viewed more than 75% of the reel: "Canelo-Crawford Hype Reel" (#Boxing)`) |
| `source_file` | Repo path to the full JSON this row was flattened from |

### Time & place
| Column | Description |
|---|---|
| `timestamp` / `datetime` | Unix seconds / `YYYY-MM-DD HH:MM UTC` of the engagement (the original formatted string is kept in `extras_json.formatted_timestamp`) |
| `location` | "city, region, country" of the session (full geo object incl. lat/lon in `extras_json.event_location`) |

### What the user did
| Column | Description |
|---|---|
| `app` | Instagram, Facebook, Threads, Chatbot, or AI_Studio |
| `interaction_type` | Signal polarity, 5 values: `implicit_negative` (skipped / scrolled past — the majority, as in real feeds), `implicit_positive` (lingered, rewatched, viewed ≥75%), `explicit_positive` (liked, saved, shared), `explicit_negative` (hid, muted, dismissed), `feed_visible` (shown in feed, no engagement recorded). `implicit_negative` rows intentionally carry only hashtags + the skip action and no content body — the user scrolled past, so nothing was dwelled on; they are negative-signal stubs, not missing data |
| `action` / `action_label` | Concrete engagement from the platform's action catalog (e.g. `saved_to_collection`, Facebook reactions, `quote_repost`, `clicked_ad`). AI-Studio rows carry `unknown` — a companion-chat session IS the event; see `conversation_json` |
| `user_message` | Text the user typed: `@ai ...` comment directives on social apps, or the user's chat turn on the Chatbot |

### What the content was
| Column | Description |
|---|---|
| `content_type` | `short_video`, `image`, or `text` (empty for pure conversations) |
| `title` / `caption` | Content title / caption (text-post body appears here too) |
| `media_description` | Visual description of the image/video |
| `audio_transcript` | Transcript for videos |
| `hashtags` | Space-joined `#tags` of the source content |
| `conversation_json` | JSON turns: Chatbot session, AI-Studio companion session, or DM thread messages |

### Social context
| Column | Description |
|---|---|
| `author_id` / `recipient_id` / `relationship` | Who authored/received it and their relation to the user (`self`, `close_friend`, ...) |
| `is_self_authored` / `is_dm` | The user's own post / a direct message |
| `is_ad` / `is_trending` | Sponsored content (sponsor + CTA details in `extras_json.content_extra.ad_metadata`) / platform-trending item |

### What the system inferred
| Column | Description |
|---|---|
| `preferences_summary` | Plain-text preview of the inferred preferences (`Follows Minnesota Wild NHL hockey content; (+2 more)`) |
| `n_preferences` | Number of inferred preference instances attached |
| `preferences_json` | JSON list — each with `persona_item` (the preference statement), `category`, `confidence_score_init` (0–1 model confidence when the preference was first inferred), `confidence_cross_referenced` (unbounded corroboration score — how many independent engagements support it), `time_horizon` (+`stop_condition` when short-term), `stereotype_mark` (whether the preference aligns with / contradicts / is neutral w.r.t. demographic stereotypes — used for bias analysis, see the card's Warning), `hidden_persona_labels`, `update_history` (reinforced/contradicted/...), provenance fields. Fields prefixed `_` anywhere in the data are internal generator provenance |
| `extras_json` | JSON catch-all for every remaining field, guaranteeing the row is lossless: full `event_location`, AI-Studio session metadata (`prior_session_refs`, `memory_used_summary`, `oblique_reference_to_hidden_personas`, `ai_studio_metadata`), DM-thread fields (`thread_id`, `participants`, `is_group_dm`, ...), trending fields, `ask_to_forget`, remaining content fields (`key_frames`, `metadata`, `parts`, raw `text`) under `content_extra` |

## Queries table (`queries_sample.csv`)

**What the evaluated system sees vs. what is scorer-side:** at evaluation time
the system under test receives the persona's history strictly BEFORE the
query's `timestamp`, plus the query surface (`user_query`,
`prior_conversation`, and — for ranking tasks — the candidate slate inside
`instance_full`). Everything else (`groundtruth_*`, `example_response`,
`inferior_response`, `rubric_tags`, `internal_checks`) is ground truth for
scoring and is never shown to the evaluated system.


| Column | Description |
|---|---|
| `persona_id` / `persona_html` | As above |
| `query_id` | Unique query id (`{persona}:{seq}:{instance}`) |
| `task_family` / `task_type` | Family (7) and concrete task (31) — see the README table |
| `what_this_tests` | Plain-English one-liner of what a correct system must do on this task |
| `query_kind` | Surface form, 4 values: `user_query` (the user asks something), `proactive_recommendation` / `proactive_assistance` (no user turn — the system must decide whether/how to act), `agentic_task` (act on the user's behalf) |
| `expected_behavior` | What a correct system does — 8 values: `personalize`, `proactive_recommend`, `proactive_assist`, `agentic_action`, `avoid_overpersonalization`, `resist_sycophancy`, `restrain`, `abstain` |
| `timestamp` / `datetime` | The query's moment T — the evaluated system may only see history strictly before T |
| `user_query` | The user's message/request; for proactive tasks this holds a bracketed scenario label (e.g. `[system prompt] Decide: ping again, or stay silent.`) for browsing convenience — it is NOT the literal model input; the harness constructs the proactive prompt from `instance_full` |
| `prior_conversation` | JSON: conversation turns preceding the query, for tasks that anchor mid-session (surfaced from `instance_full.prior_conversation`; empty when the task has no preceding session) |
| `groundtruth_preference` / `groundtruth_preference_obj` | The preference(s) the answer must be grounded in (text / full JSON object). For voice-authoring agentic tasks this holds the user's voice spec rather than a topical preference |
| `distractor_preferences` | JSON: plausible-but-wrong preferences a shallow system confuses |
| `example_response` | Gold response (what a well-personalized system should say/do) |
| `inferior_response` | Contrast response — plausible but misses the axis under test. Plain text for chat tasks, JSON for structured tasks (the contrast text under `text`, the deliberately-injected flaw under `flaw_kind`); empty on judge-scored tasks (e.g. sycophancy), which have no canned inferior by design |
| `reference_example` | JSON: auxiliary reference material for some task types |
| `rubric_tags` | Scoring axes for the judge |
| `tool_call` | JSON: expected tool/action payload for agentic tasks |
| `internal_checks` | JSON: build-time QA artifacts (self-checks, voice-evidence audits) — not part of the task |
| `instance_full` | JSON: the COMPLETE task instance consumed by the eval harness (slates, pools, arms, anchors) |
| `source_file` | `backend/{persona_id}/test.json` — all queries for the persona |
"""
    (out / "column_descriptions.md").write_text(txt)


# ---------------------------------------------------------------- validate ---

def validate(out: Path, stats: dict) -> list[str]:
    problems = list(stats["problems"])
    # staging completeness
    for uid in USERS:
        for f in PER_PERSONA_FILES:
            if not (out / "backend" / uid / f).exists():
                problems.append(f"staged file missing: backend/{uid}/{f}")
    # CSVs parse + row counts + all task types present
    for name in ["history_all_apps_sample"] + [f"history_{a}_sample" for a in APPS] + ["queries_sample"]:
        p = out / "samples" / f"{name}.csv"
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            problems.append(f"{name}.csv is empty")
        if name == "queries_sample":
            types = {r["task_type"] for r in rows}
            missing = set(stats["task_type_counts"]) - types
            if missing:
                problems.append(f"queries_sample missing task types: {sorted(missing)}")
    # smoke: the staged copy is codebase-readable
    try:
        from evaluation.backend_query import BackendQuery
        bq = BackendQuery(out / "backend")
        for uid in USERS:
            for app in APPS:
                if not bq._load_events(uid, app):
                    problems.append(f"smoke: BackendQuery loaded 0 events for {uid}/{app}")
            if not bq._load_profile(uid):
                problems.append(f"smoke: BackendQuery loaded empty profile for {uid}")
    except Exception as exc:  # pragma: no cover
        problems.append(f"smoke: BackendQuery failed: {exc}")
    try:
        from evaluation.run_eval import _load_queries
        for uid in USERS:
            n = len(_load_queries(out / "backend" / uid / "test.json"))
            if n == 0:
                problems.append(f"smoke: _load_queries returned 0 rows for {uid}")
    except Exception as exc:  # pragma: no cover
        problems.append(f"smoke: _load_queries failed: {exc}")
    return problems


def upload(out: Path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        for line in (REPO_ROOT / ".env").read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not token:
        sys.exit("No HF_TOKEN in environment or .env — cannot upload.")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=str(out),
                      commit_message="PersonaMem-v3 release: 20 personas (histories + queries + card)")
    print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="release/hf")
    ap.add_argument("--per-persona-hist", type=int, default=250)
    ap.add_argument("--upload", action="store_true",
                    help="push the staged folder to HF after validation passes")
    args = ap.parse_args()

    out = (REPO_ROOT / args.out).resolve()
    print(f"Staging release to {out} ...")
    stats = stage(out, args.per_persona_hist)

    # build task->family mapping for the card
    for uid in USERS:
        for r in json.loads((out / "backend" / uid / "test.json").read_text()):
            _TASK2FAM[r.get("task_type")] = r.get("task_family")
    write_card(out, stats)

    problems = validate(out, stats)
    print(f"\nEvents: {stats['n_events']:,}  pref instances: {stats['n_pref_instances']:,}  "
          f"queries: {stats['n_queries']:,} / {len(stats['task_type_counts'])} task types")
    print(f"Samples: all_apps={stats['sample_all_apps']} " +
          " ".join(f"{a}={stats[f'sample_{a}']}" for a in APPS) +
          f" queries={stats['sample_queries']}")
    if problems:
        print(f"\nVALIDATION: {len(problems)} problem(s):")
        for p in problems[:40]:
            print("  -", p)
        sys.exit(1)
    print("VALIDATION: all checks passed (coverage over every event + every query row).")

    if args.upload:
        upload(out)


if __name__ == "__main__":
    main()
