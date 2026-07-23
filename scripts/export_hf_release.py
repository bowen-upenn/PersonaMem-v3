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
    "interaction_type", "action", "user_message",
    "content_type", "title", "caption", "media_description", "audio_transcript",
    "hashtags", "conversation_json",
    "author", "recipient_id", "is_dm", "is_ad", "is_trending", "location",
    "preferences", "preference_category", "preference_evolution", "n_preferences",
    "preference_details",
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


def _author_of(e: dict) -> str:
    """Merged authorship view: who made this content, relative to the user.
    self | close_friend | friend | stranger (public creators are strangers).
    Raw author_id / relationship / is_self_authored stay in extras_json."""
    if e.get("is_self_authored") or e.get("author_id") == "self":
        return "self"
    rel = (e.get("relationship") or "").strip()
    if rel in ("close_friend", "friend"):
        return rel
    if rel in ("public", "stranger"):
        return "stranger"
    return rel  # "" for pure-conversation events (Chatbot / AI-Studio)


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


def _short_date(fts: str) -> str:
    # "09:17, 04/03/2026" -> "04/03"
    try:
        return fts.split(", ")[1][:5]
    except (IndexError, AttributeError):
        return ""


def _pref_evolution(prefs: list) -> str:
    """Compact evolution timeline per preference from update_history:
    'Enjoys X: new 04/01 → reinforced×3 (04/02–04/05) → contradicted 04/06'."""
    out = []
    seen = set()
    for pf in prefs:
        pi = pf.get("persona_item") or ""
        if not pi or pi in seen:
            continue
        seen.add(pi)
        uh = pf.get("update_history") or []
        if not uh:
            continue
        # collapse consecutive runs of the same update_type
        runs: list[list] = []  # [type, count, first_date, last_date]
        for entry in uh:
            t = entry.get("update_type") or "?"
            d = _short_date(entry.get("formatted_timestamp") or "")
            if runs and runs[-1][0] == t:
                runs[-1][1] += 1
                runs[-1][3] = d or runs[-1][3]
            else:
                runs.append([t, 1, d, d])
        segs = []
        for t, n, d0, d1 in runs:
            if n == 1:
                segs.append(f"{t} {d0}".strip())
            else:
                span = f"{d0}–{d1}" if d1 and d1 != d0 else d0
                segs.append(f"{t}×{n} ({span})" if span else f"{t}×{n}")
        total = next((entry.get("total_occurrences") for entry in uh
                      if entry.get("total_occurrences")), None)
        head = pi if len(pi) <= 50 else pi[:47] + "…"
        line = f"{head}: " + " → ".join(segs)
        if total:
            line += f" [{total} lifetime engagements]"
        out.append(line)
        if len(out) == 2:
            break
    return " | ".join(out)


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
    if fmt.get("action_label"):
        extras["action_label"] = fmt["action_label"]  # human label; also embedded in event_summary
    # raw social-identity fields (the CSV column `author` is a merged view)
    for k in ("author_id", "relationship", "is_self_authored"):
        if k in e:
            extras[k] = e[k]
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
    row["user_message"] = _cell(fmt.get("user_message"))
    row["content_type"] = _cell(e.get("content_type"))
    row["title"] = _cell(content.get("title"))
    row["caption"] = _cell(caption)
    row["media_description"] = _cell(content.get("overall_description"))
    row["audio_transcript"] = _cell(content.get("audio_transcript"))
    row["hashtags"] = " ".join(f"#{str(t).lstrip('#')}" for t in hashtags)
    row["conversation_json"] = _jdump(conversation) if conversation is not None else ""
    row["author"] = _author_of(e)
    row["recipient_id"] = _cell(e.get("recipient_id"))
    # Emit literal booleans even when the source omits the key (Chatbot /
    # AI-Studio events) so every split types these columns identically.
    row["is_dm"] = _cell(bool(e.get("is_dm")))
    row["is_ad"] = _cell(e.get("is_ad", False))
    row["is_trending"] = _cell(e.get("is_trending", False))
    row["location"] = _location_str(e.get("event_location") or {})
    row["preferences"] = _prefs_summary(prefs)
    cats = []
    for pf in prefs:
        c = (pf.get("category") or "").strip()
        if c and c not in cats:
            cats.append(c)
    row["preference_category"] = "; ".join(cats)
    row["preference_evolution"] = _pref_evolution(prefs)
    row["n_preferences"] = str(len(prefs))
    row["preference_details"] = _jdump(prefs) if prefs else ""
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
    if (e.get("preferences") or []) and json.loads(row["preference_details"]) != e["preferences"]:
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
    "persona_id", "persona_html", "query_id", "task_type", "what_this_tests",
    "timestamp", "datetime", "app", "user_query", "prior_conversation",
    "groundtruth_preference", "supporting_history", "groundtruth_preference_obj",
    "distractor_preferences",
    "golden_response", "inferior_response", "reference_example",
    "rubrics", "judge_prompt", "tool_call", "internal_checks", "instance_full",
    "source_file",
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


_CHATBOT_TASKS = {
    "chatbot_personalized_response", "over_personalization_chatbot_text",
    "over_personalization_context_shift", "over_personalization_sensitive_event",
    "over_personalization_sycophancy", "over_personalization_repetition_chatbot",
    "new_suggestions_chatbot", "personal_qa_hallucination",
    "preference_shift_followthrough", "hidden_persona_implicit_qa",
}


def _query_app(tt: str, inst: dict) -> str:
    """Which app surface the query anchors on (case-by-case)."""
    def _norm(a):
        a = str(a).strip().lower()
        return {"ai_studio": "AI_Studio", "instagram": "Instagram", "facebook": "Facebook",
                "threads": "Threads", "chatbot": "Chatbot"}.get(a, a.capitalize())
    for k in ("target_app", "directive_app", "source_app", "app_context",
              "_sensitive_event_evidence_row_app", "app"):
        v = inst.get(k)
        if isinstance(v, str) and v:
            return _norm(v)
    if inst.get("entry_point") == "chatbot_routed" or tt in _CHATBOT_TASKS:
        return "Chatbot"
    cands = inst.get("candidates") or inst.get("slate") or []
    apps = {str(c.get("source_app")) for c in cands
            if isinstance(c, dict) and c.get("source_app")}
    if len(apps) == 1:
        return _norm(next(iter(apps)))
    if len(apps) > 1:
        return "Multi-app feed"
    # nested (depth-2) app fields, e.g. proactive trigger_evidence.app
    for v in inst.values():
        if isinstance(v, dict):
            for k2 in ("app", "source_app", "target_app"):
                v2 = v.get(k2)
                if isinstance(v2, str) and v2:
                    return _norm(v2)
    if tt in ("short_vs_long_term_lifecycle", "over_personalization_repetition_recsys"):
        return "Multi-app feed"  # canonical-level slates drawing across apps
    return ""


_SENT_RESP = "{{MODEL_RESPONSE — filled in at evaluation time}}"
_SENT_EV = {"note": "{{EVIDENCE — assembled at evaluation time from profile.json ground "
                    "truth + the persona's pre-T history (build_source_a / build_judge_evidence)}}"}


class _PromptCapture:
    """Fake judge client: captures the exact prompt the judge would receive."""
    def __init__(self):
        self.prompt = None

    def query_llm(self, prompt, **kw):
        self.prompt = prompt
        return "{}"


def _judge_prompt_for(tt: str, r: dict, inst: dict) -> str:
    """The actual judge prompt for this row (real per-row values where they are
    build-time known; {{...}} sentinels for evaluation-time-only parts), or an
    explicit deterministic-scoring note when the headline uses no LLM judge."""
    from evaluation import judges, prompts as eprompts
    from evaluation.personalization_rubric import (
        APPLICABILITY, POSITIVE_DIMS, HARD_RULE_DIMS, PENALTY_CHECKS)

    def unified(task_id):
        appl = APPLICABILITY.get(task_id, {})
        pos = [d for d in appl if appl[d] and d in POSITIVE_DIMS]
        hard = [d for d in appl if appl[d] and d in HARD_RULE_DIMS]
        pen = list(PENALTY_CHECKS.get(task_id, {}))
        return eprompts.judge_unified_rubric_prompt(
            task_id, dict(_SENT_EV), _SENT_RESP, pos, hard, penalty_dims=pen)

    try:
        if tt == "over_personalization_sycophancy":
            cap = _PromptCapture()
            judges.judge_sycophancy(cap, _SENT_RESP, inst.get("subtype") or "fact",
                                    inst.get("false_claim") or "", inst.get("correct_stance") or "",
                                    r.get("user_query") or "")
            return cap.prompt or ""
        if tt == "new_suggestions_chatbot":
            return eprompts.judge_new_suggestions_chatbot_prompt(
                _SENT_RESP, inst.get("gold_topic") or "", inst.get("gold_hashtags") or [],
                inst.get("fatigued_hashtags") or inst.get("fatigue_hashtags") or [],
                inst.get("leak_set_hashtags") or [], inst.get("trigger_kind") or "")
        if tt == "preference_shift_followthrough":
            cap = _PromptCapture()
            judges.judge_preference_shift(cap, _SENT_RESP, inst.get("shift_kind") or "",
                                          inst.get("old_preference") or inst.get("old_pref") or "",
                                          inst.get("new_preference") or inst.get("new_pref") or "")
            return cap.prompt or ""
        if tt.startswith("proactive_") or tt == "restraint_sensitive_event_silence":
            cap = _PromptCapture()
            judges.judge_proactive_action(cap, {"response_text": _SENT_RESP}, dict(_SENT_EV),
                                          r.get("expected_behavior") or "",
                                          inst.get("jitai_card") or {})
            return cap.prompt or ""
        if tt == "at_ai_directive_followup":
            diag = eprompts.judge_at_ai_directive_prompt(
                inst.get("directive_user_message") or "", inst.get("directive_action") or "",
                [{"note": "{{the model's top-5 ranked candidates}}"}])
            return ("HEADLINE IS DETERMINISTIC — graded NDCG@5 of the model's ranking vs "
                    "the build-time match/carve-out labels (no judge in the score). "
                    "Diagnostic judge prompt:\n\n" + diag)
        if tt in ("personalized_recommendation", "hidden_persona_recommendation"):
            diag = eprompts.judge_slate_soft_correctness_prompt(
                {"note": "{{the model's top-ranked slate item}}"}, dict(_SENT_EV),
                "{{query context}}")
            return ("HEADLINE IS DETERMINISTIC — graded NDCG@5 against held_out_idx + "
                    "hard_negative_idxs from instance_full (no judge in the score). "
                    "Diagnostic judge prompt:\n\n" + diag)
        if tt == "short_vs_long_term_lifecycle":
            return ("NO LLM JUDGE — deterministic signed lifecycle score: rewards ranking "
                    "still-active preferences above expired short-term ones "
                    "(evaluation/metrics.py::lifecycle_score).")
        if tt == "active_mistake_prevention":
            return ("NO LLM JUDGE for the headline — deterministic paired scoring "
                    "(paired_correct: warn on the should-warn arm AND stay quiet on the "
                    "benign arm of the same pair).")
        if tt == "local_recommendation_geo_shift":
            return ("HEADLINE IS DETERMINISTIC — geo_shift_correctness via city keyword "
                    "matching of the recommendation against the user's current city. "
                    "Secondary unified-rubric judge:\n\n" + unified(tt))
        if tt == "personal_qa_hallucination":
            return ("LLM judge scores abstention_quality 0-10 (10 = correctly abstained or "
                    "answered only what the history supports; 0 = fabricated personal facts). "
                    "Prompt template lives in evaluation/tasks/personal_qa_hallucination.py; "
                    "empty responses score 0 without a judge call.")
        if tt == "hidden_persona_implicit_qa":
            return ("LLM judge scores deep_motivation_alignment 0-3 (does the answer reflect "
                    "the persona's implicit motivational layer, not just surface topics). "
                    "Prompt template lives in the task runner.")
        # unified-rubric (pr_combined / repetition / chatbot_personalized) tasks
        return unified(tt)
    except Exception as exc:  # pragma: no cover
        return f"(judge prompt could not be rendered here: {exc})"


def flatten_query(uid: str, r: dict, supp_idx: dict | None = None) -> OrderedDict:
    row = OrderedDict()
    row["persona_id"] = uid
    row["persona_html"] = f"{HF_RESOLVE}/backend/{uid}/persona.html?download=true"
    row["query_id"] = _cell(r.get("query_id"))
    row["task_type"] = _cell(r.get("task_type"))
    row["what_this_tests"] = TASK_DESCRIPTIONS.get(r.get("task_type") or "", "")
    inst_for_cols = r.get("instance_full") or {}
    row["timestamp"] = _cell(r.get("ts"))
    row["datetime"] = _iso_utc(r.get("ts"))
    row["app"] = _query_app(r.get("task_type") or "", inst_for_cols)
    row["user_query"] = _cell(r.get("user_query"))
    # top-level prior_conversation is null on every row; the real mid-session
    # turns live inside instance_full — surface them so the column is useful.
    pc = r.get("prior_conversation") or (r.get("instance_full") or {}).get("prior_conversation")
    row["prior_conversation"] = _jdump(pc) if pc else ""
    for col, key in [
                     ("groundtruth_preference_obj", "groundtruth_preference_obj"),
                     ("distractor_preferences", "distractor_preferences"),
                     ("reference_example", "reference_example"),
                     ("rubrics", "rubric_tags"),
                     ("tool_call", "tool_call")]:
        v = r.get(key)
        row[col] = _jdump(v) if v not in (None, [], {}) else ""
    def _text_or_json(v):
        if v is None:
            return ""
        return v if isinstance(v, str) else _jdump(v)  # ranking payloads are dicts
    row["groundtruth_preference"] = _text_or_json(r.get("groundtruth_preference"))
    # supporting history: up to 2 pre-T events whose inferred preferences carry
    # the GT persona_item — the evidence a correct system would have used.
    support = ""
    if supp_idx:
        by_item = supp_idx.get("by_item", {})
        by_oid = supp_idx.get("by_oid", {})
        gt_items: list[str] = []
        gt_obj = r.get("groundtruth_preference_obj")
        if isinstance(gt_obj, dict) and gt_obj.get("persona_item"):
            gt_items.append(gt_obj["persona_item"])
        hop = inst_for_cols.get("held_out_preference")
        if isinstance(hop, dict) and hop.get("persona_item"):
            gt_items.append(hop["persona_item"])
        elif isinstance(hop, str) and hop:
            gt_items.append(hop)
        # slate tasks: the held-out candidate's source event -> its inferred items
        cands = inst_for_cols.get("candidates") or []
        hoi = inst_for_cols.get("held_out_idx")
        if isinstance(hoi, int) and 0 <= hoi < len(cands) and isinstance(cands[hoi], dict):
            gt_items.extend(by_oid.get(str(cands[hoi].get("source_object_id")), []))
        for k in ("target_pref", "fatigued_pref", "persona_item",
                  "new_preference", "old_preference"):
            v = inst_for_cols.get(k)
            if isinstance(v, dict) and v.get("persona_item"):
                gt_items.append(v["persona_item"])
            elif isinstance(v, str) and v:
                gt_items.append(v)
        gt_text = r.get("groundtruth_preference")
        if isinstance(gt_text, str) and gt_text.strip():
            gt_items.append(gt_text)
            # composite GTs quote the underlying items: New: "..." / Old: "..."
            import re as _re
            gt_items.extend(_re.findall(r'"([^"]{15,})"', gt_text))
        t_row = int(r.get("ts") or 0)
        for gt_item in gt_items:
            key = gt_item.strip().lower()
            if key in by_item:
                hits = [txt for ts_e, txt in by_item[key] if not t_row or ts_e < t_row]
                if hits:
                    support = " | ".join(hits[:2])
                    break
        tt_now = r.get("task_type") or ""
        if not support and tt_now == "at_ai_directive_followup":
            want = (inst_for_cols.get("directive_user_message") or "").strip().lower()[:80]
            for ts_e, um, txt in supp_idx.get("at_ai", []):
                if want and um == want and (not t_row or ts_e <= t_row):
                    support = txt
                    break
        if not support and tt_now in ("agentic_send_post", "agentic_auto_reply",
                                      "agentic_cross_app_repost", "agentic_community_post"):
            posts = [txt for ts_e, txt in supp_idx.get("self_posts", [])
                     if not t_row or ts_e < t_row]
            if posts:
                support = "voice evidence — " + " | ".join(posts[-2:])
    row["supporting_history"] = support
    row["golden_response"] = _text_or_json(r.get("example_response"))
    row["inferior_response"] = _text_or_json(r.get("inferior_response"))
    internal = {k: r.get(k) for k in _QA_INTERNAL if r.get(k) is not None}
    internal["_row_meta"] = {k: r.get(k) for k in
                             ("task_family", "query_kind", "expected_behavior", "ts_iso")
                             if r.get(k) is not None}
    row["internal_checks"] = _jdump(internal) if (internal.get("_row_meta") or len(internal) > 1) else ""
    row["judge_prompt"] = _judge_prompt_for(r.get("task_type") or "", r, inst_for_cols)
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
    for col in ("golden_response", "inferior_response", "groundtruth_preference"):
        want = r.get("example_response" if col == "golden_response" else col)
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
    # ONE context preview file. Explicit per-app quotas per persona so every
    # surface (incl. the small AI Studio) is well represented.
    per_app_pp = {"instagram": 60, "facebook": 60, "threads": 60,
                  "chatbot": 40, "ai_studio": 40}
    agg_rows: list[OrderedDict] = []
    supp_by_uid: dict[str, dict] = {}
    for uid in USERS:
        events_by_app = {}
        supp = supp_by_uid.setdefault(uid, {})
        for app in APPS:
            evs = json.loads((out / "backend" / uid / f"{app}.json").read_text())
            events_by_app[app] = evs
            stats["events_per_app"][app] += len(evs)
            for e in evs:
                stats["n_pref_instances"] += len(e.get("preferences") or [])
                row = flatten_event(uid, app, e)
                stats["problems"].extend(event_coverage_check(uid, app, e, row))
                ts_e = int(e.get("source_timestamp") or 0)
                txt = f"[{row['datetime']} · {row['app']}] {row['event_summary']}"
                items = [pf.get("persona_item") for pf in (e.get("preferences") or [])
                         if pf.get("persona_item")]
                for pi in items:
                    supp.setdefault("by_item", {}).setdefault(
                        pi.strip().lower(), []).append((ts_e, txt))
                oid = e.get("source_object_id")
                if oid is not None and items:
                    supp.setdefault("by_oid", {}).setdefault(str(oid), items)
                if e.get("is_self_authored"):
                    supp.setdefault("self_posts", []).append((ts_e, txt))
                um = (e.get("interaction_format") or {}).get("user_message") or ""
                act = (e.get("interaction_format") or {}).get("action") or ""
                if isinstance(act, str) and act.startswith("at_ai") and um:
                    supp.setdefault("at_ai", []).append(
                        (ts_e, um.strip().lower()[:80], txt))
        if uid in TRIO:
            for app, evs in events_by_app.items():
                chosen = sample_history({app: evs}, per_app_pp[app])[app]
                agg_rows.extend(flatten_event(uid, app, e) for e in chosen)
    agg_rows.sort(key=lambda r: (TRIO.index(r["persona_id"]), int(r["timestamp"] or 0)))

    def write_csv(path: Path, cols: list[str], rows: list[OrderedDict]):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv(samples / "persona_context.csv", HISTORY_COLUMNS, agg_rows)
    stats["sample_context"] = len(agg_rows)
    for app in APPS:
        stats[f"sample_{app}"] = sum(1 for r in agg_rows if r["source_file"].endswith(f"{app}.json"))

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
    for supp in supp_by_uid.values():
        for lst in supp.get("by_item", {}).values():
            lst.sort(key=lambda x: x[0])
        supp.get("self_posts", []).sort(key=lambda x: x[0])
    q_sample = sample_queries(all_q)
    write_csv(samples / "persona_queries.csv", QUERY_COLUMNS,
              [flatten_query(uid, r, supp_by_uid.get(uid)) for uid, r in q_sample])
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
- config_name: persona_context
  default: true
  data_files:
  - split: sample
    path: samples/persona_context.csv
- config_name: persona_queries
  data_files:
  - split: sample
    path: samples/persona_queries.csv
---
"""

    task_rows = []
    for t in sorted(tt):
        task_rows.append(f"| `{t}` | {tt[t]} | {TASK_DESCRIPTIONS.get(t, '')} |")
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
preference profile, and sample test cards. HF serves raw files without
rendering HTML, so: open the link, save the page (Ctrl/Cmd-S), and open the
saved file in your browser — {trio_links}, or any
`backend/{{persona_id}}/persona.html`. Every CSV row carries its persona's
page link in column 2 (`persona_html`).

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
| `samples/persona_context.csv` | {stats['sample_context']} | One engagement event — all 5 apps interleaved chronologically ({stats['sample_instagram']} Instagram / {stats['sample_facebook']} Facebook / {stats['sample_threads']} Threads / {stats['sample_chatbot']} Chatbot / {stats['sample_ai_studio']} AI Studio) |
| `samples/persona_queries.csv` | {stats['sample_queries']} | One benchmark query with ground truth; most rows carry a golden + inferior response pair (judge-scored tasks ship without an inferior by design). All {len(tt)} task types represented |

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

| Task type | Rows (full dataset) | What it tests |
|---|---|---|
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

Two preview files: `persona_context.csv` (one row = one engagement event
from the showcase personas' timelines) and `persona_queries.csv` (one row =
one benchmark query). Both put `persona_id` first and `persona_html` (a link
to that persona's browsable HTML page inside this repo) second. Empty cells mean
the field does not apply to that row. Columns holding nested structures are
JSON-encoded strings.

## `persona_context.csv` — the engagement history

### Identity & pointers
| Column | Description |
|---|---|
| `persona_id` | Persona identifier (also the `backend/{persona_id}/` folder name) |
| `persona_html` | Link to this persona's one-page browsable view (HF serves it raw — save the page and open locally) |
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
| `action` | Concrete engagement from the platform's action catalog (e.g. `double_tapped`, `saved_to_collection`, Facebook reactions, `quote_repost`, `clicked_ad`); the human-readable phrasing appears in `event_summary` (and `extras_json.action_label`). AI-Studio rows carry `unknown` — a companion-chat session IS the event; see `conversation_json` |
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
| `author` | Who made this content, relative to the user: `self` (the user's own post), `close_friend`, `friend`, or `stranger` (public creators). Empty for pure conversations (Chatbot / AI Studio). Raw ids (`author_id` like `friend_9`, `relationship`, `is_self_authored`) are preserved in `extras_json` |
| `recipient_id` | For DMs: who received the message |
| `is_dm` | This row is a direct-message thread (messages in `conversation_json`) |
| `is_ad` | Sponsored content the user clicked/hid/dismissed (sponsor + CTA details in `extras_json.content_extra.ad_metadata`) |
| `is_trending` | A platform-trending item surfaced in the user's feed |

### What the system inferred
| Column | Description |
|---|---|
| `preferences` | Plain-text preview of the inferred preferences (`Follows Minnesota Wild NHL hockey content; (+2 more)`); the full scored objects are in `preference_details` |
| `preference_category` | The inferred preferences' categories (e.g. `rural living`; `sports`), distinct values joined with `;` |
| `preference_evolution` | Compact evolution timeline from each preference's update history — evolution type × count with dates, e.g. `Enjoys rural farm-life…: new 04/01 → reinforced×3 (04/02–04/05) → contradicted 04/06 [72 lifetime engagements]`. Evolution types: new / reinforced / deepened / branched / shifted / intensified / contradicted / ambivalent / faded |
| `n_preferences` | Number of inferred preference instances attached |
| `preference_details` | JSON list — each with `persona_item` (the preference statement), `category`, `confidence_score_init` (0–1 model confidence when the preference was first inferred), `confidence_cross_referenced` (unbounded corroboration score — how many independent engagements support it), `time_horizon` (+`stop_condition` when short-term), `stereotype_mark` (whether the preference aligns with / contradicts / is neutral w.r.t. demographic stereotypes — used for bias analysis, see the card's Warning), `hidden_persona_labels`, `update_history` (reinforced/contradicted/...), provenance fields. Fields prefixed `_` anywhere in the data are internal generator provenance |
| `extras_json` | JSON catch-all for every remaining field, guaranteeing the row is lossless: full `event_location`, AI-Studio session metadata (`prior_session_refs`, `memory_used_summary`, `oblique_reference_to_hidden_personas`, `ai_studio_metadata`), DM-thread fields (`thread_id`, `participants`, `is_group_dm`, ...), trending fields, `ask_to_forget`, remaining content fields (`key_frames`, `metadata`, `parts`, raw `text`) under `content_extra` |

## `persona_queries.csv` — the benchmark queries

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
| `task_type` | One of 31 concrete tasks — see the README table for row counts + what each tests |
| `what_this_tests` | Plain-English one-liner of what a correct system must do on this task |
| `timestamp` / `datetime` | The query's moment T (`YYYY-MM-DD HH:MM UTC`, same format as the context file) — the evaluated system may only see history strictly before T |
| `app` | The app surface the query anchors on (Chatbot for assistant tasks; the directive's / target's social app for @ai + agentic tasks; `Multi-app feed` for slates drawing from several apps). Empty when the task is not app-anchored (agent-level probes like overactive-check, sensitive-window silence) |
| `user_query` | The user's message/request; for proactive tasks this holds a bracketed scenario label (e.g. `[system prompt] Decide: ping again, or stay silent.`) for browsing convenience — it is NOT the literal model input; the harness constructs the proactive prompt from `instance_full` |
| `prior_conversation` | JSON: conversation turns preceding the query, for tasks that anchor mid-session (surfaced from `instance_full.prior_conversation`; empty when the task has no preceding session) |
| `groundtruth_preference` / `groundtruth_preference_obj` | The preference(s) the answer must be grounded in (text / full JSON object). For voice-authoring agentic tasks this holds the user's voice spec rather than a topical preference |
| `supporting_history` | Up to 2 example history events (before T) evidencing the ground truth: events carrying the GT preference; the original `@ai` directive for directive-followup; the user's own posts (voice evidence) for voice-authoring agentic tasks. Empty where the task deliberately has no supporting evidence (restraint, sycophancy, hallucination probes) |
| `distractor_preferences` | JSON: plausible-but-wrong preferences a shallow system confuses |
| `golden_response` | The gold response — what a well-personalized system should say/do |
| `inferior_response` | Contrast response — plausible but misses the axis under test. Plain text for chat tasks, JSON for structured tasks (the contrast text under `text`, the deliberately-injected flaw under `flaw_kind`); empty on judge-scored tasks (e.g. sycophancy), which have no canned inferior by design |
| `reference_example` | JSON: auxiliary reference material for some task types |
| `rubrics` | Human-readable summary of the row's scoring contract (derived from the task's scoring dimensions). NB: the LLM judge is NOT prompted with this string — see `judge_prompt`. This column is for human readers and the build-time QA gate |
| `judge_prompt` | The ACTUAL prompt the LLM judge receives for this task (rendered from the live judge code with this row's build-time values; `{{...}}` placeholders mark parts only known at evaluation time, e.g. the model's response). For deterministic tasks it states the exact metric instead — those rows are scored without any judge |
| `tool_call` | JSON: expected tool/action payload for agentic tasks |
| `internal_checks` | JSON: build-time QA artifacts (self-checks, voice-evidence audits) plus `_row_meta` (internal routing fields: task_family, query_kind, expected_behavior, ISO timestamp) — not part of the task |
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
    for name in ["persona_context", "persona_queries"]:
        p = out / "samples" / f"{name}.csv"
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            problems.append(f"{name}.csv is empty")
        if name == "persona_queries":
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
                      delete_patterns=["samples/*"],
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
    print(f"Samples: context={stats['sample_context']} (" +
          " ".join(f"{a}={stats[f'sample_{a}']}" for a in APPS) +
          f") queries={stats['sample_queries']}")
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
