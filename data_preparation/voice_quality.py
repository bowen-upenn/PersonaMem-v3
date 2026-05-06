"""Voice-quality auto-verifier for user-voiced data-gen samples.

Every sample that the AI under evaluation can read as evidence of *how
this user writes* must actually carry the user's layered voice
signature — otherwise the agent has nothing to learn the voice from
(profile.user_voice is firewalled from the agent at test time).

This module wraps `prompts.voice_quality_check_prompt` with a small
retry loop so each generator (chatbot_conversation, self_posts,
dm_threads) can call one helper and get back a pass/fail verdict
plus a fix_hint that can be threaded into the regen prompt.

Scope:
  - Chatbot conversations (user turns + any embedded draft text).
  - Self-posts on social apps (caption text).
  - DM threads (user-side outbound messages, concatenated).
"""

from __future__ import annotations

from typing import Callable

from data_preparation import prompts, utils


def extract_user_text_from_chatbot_conversation(turns: list[dict]) -> str:
    """Concatenate user-side turns (with role=='user') into one
    auditable block. Assistant turns are skipped — the audit grades
    only what the USER wrote.

    The user-pasted draft (when the conversation type is `writing_help`
    / `correction` and the user pasted an email / caption / cover
    letter to improve) is INSIDE turn["content"] — it stays in the
    block, and the judge prompt's `embedded_drafts=True` knob tells
    the LLM to grade the pasted draft alongside the chat-turn voice.
    """
    if not isinstance(turns, list):
        return ""
    user_lines = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        if t.get("role") != "user":
            continue
        c = (t.get("content") or "").strip()
        if c:
            user_lines.append(c)
    return "\n\n---\n\n".join(user_lines)


def extract_user_text_from_dm_thread(thread_event: dict) -> str:
    """Concatenate the user's outbound DM messages from a thread event.

    Thread events have an embedded `messages: [{sender, text, ...}]`
    array. Pull every message where `sender == "self"` (or matches a
    self-id if available). Return concatenated text.
    """
    if not isinstance(thread_event, dict):
        return ""
    msgs = thread_event.get("messages") or []
    if not isinstance(msgs, list):
        return ""
    user_lines = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        sender = m.get("sender") or m.get("author_id") or ""
        if sender != "self":
            continue
        txt = (m.get("text") or m.get("content") or "").strip()
        if txt:
            user_lines.append(txt)
    return "\n".join(user_lines)


def validate_user_voiced_sample(
    sample_text: str,
    user_voice: dict,
    app_persona: dict | None,
    llm_query_fn: Callable[[str], str],
    *,
    surface_label: str = "user-voiced text",
    embedded_drafts: bool = False,
    score_floor: int = 3,
    soft_skip_short: int = 25,
) -> tuple[bool, dict]:
    """Run the voice-quality judge on one sample. Returns
    ``(passed, judgment_dict)``.

    ``judgment_dict`` has the keys ``score``, ``passes``,
    ``weakest_axis``, ``reason``, ``fix_hint``. On parse failure or
    LLM error, returns ``(True, {"audit_status": "skipped", ...})``
    — voice-quality is a soft gate, not a hard one; we'd rather emit
    the sample than block the whole pipeline on a flaky judge.

    `soft_skip_short`: samples shorter than this many characters
    (e.g., a 5-word reply, a one-line emoji reaction) carry too
    little signal for a meaningful voice judgment — auto-pass them
    rather than forcing the LLM to grade noise.
    """
    if not user_voice:
        return True, {"audit_status": "no_voice_to_check_against"}
    text = (sample_text or "").strip()
    if len(text) < soft_skip_short:
        return True, {
            "audit_status": "too_short_to_judge",
            "char_len": len(text),
        }

    prompt = prompts.voice_quality_check_prompt(
        sample_text=text,
        user_voice=user_voice,
        app_persona=app_persona,
        surface_label=surface_label,
        embedded_drafts=embedded_drafts,
    )
    try:
        response = llm_query_fn(prompt)
    except Exception as exc:
        return True, {
            "audit_status": "llm_error",
            "error": str(exc)[:160],
        }
    if not response:
        return True, {"audit_status": "no_llm_response"}
    parsed = utils.extract_json_from_response(response)
    if not isinstance(parsed, dict):
        return True, {
            "audit_status": "unparseable",
            "raw": (response or "")[:160],
        }
    try:
        score = int(parsed.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    score = max(1, min(5, score))
    passes_field = bool(parsed.get("passes", False))
    passed = passes_field and score >= score_floor
    return passed, {
        "audit_status": "audited",
        "score": score,
        "passes": passes_field,
        "weakest_axis": str(parsed.get("weakest_axis") or ""),
        "reason": str(parsed.get("reason") or "")[:280],
        "fix_hint": str(parsed.get("fix_hint") or "")[:240],
    }


def render_fix_hint_for_regen(judgment: dict) -> str:
    """Format a failed-judgment record into a steering line that the
    regen prompt can append. Used by chatbot_conversation,
    self_posts, dm_threads to get specific feedback on retry rather
    than a blind re-roll.
    """
    if not judgment:
        return ""
    reason = (judgment.get("reason") or "").strip()
    hint = (judgment.get("fix_hint") or "").strip()
    weakest = (judgment.get("weakest_axis") or "").strip()
    parts: list[str] = []
    if reason:
        parts.append(f"Voice-quality auditor flagged the previous output: {reason}")
    if weakest:
        parts.append(f"Weakest axis: `{weakest}`.")
    if hint:
        parts.append(f"To fix on retry: {hint}")
    if not parts:
        return ""
    return "## Voice-quality regen note\n\n" + " ".join(parts) + "\n"


__all__ = [
    "extract_user_text_from_chatbot_conversation",
    "extract_user_text_from_dm_thread",
    "render_fix_hint_for_regen",
    "validate_user_voiced_sample",
]
