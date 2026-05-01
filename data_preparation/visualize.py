"""
Generates a standalone HTML persona visualization for a user.

Reads backend/{user_id}/profile.json plus the four per-app JSON files
(instagram.json, facebook.json, threads.json, chatbot.json).

Supports both the new interaction-event format (nested preferences per
event) and the legacy flat format (one record per preference).

Design: minimalist, Apple/Anthropic-inspired aesthetic.
No external dependencies — pure HTML/CSS/JS.
"""

from __future__ import annotations

import csv
import os
import json
from datetime import datetime, timezone
from pathlib import Path

from data_preparation import utils


APPS = ["Instagram", "Facebook", "Threads", "Chatbot"]


# ---------------------------------------------------------------------------
# Test-sample annotation (Phase A4)
# ---------------------------------------------------------------------------

# Per-render persona context bank — populated by _load_test_samples and
# threaded into every GT extractor so abstract tasks (search, briefing,
# trending alert, etc.) can build CONCRETE expected-answer shapes that
# reference the user's actual recent preferences / hashtags / categories.
_PERSONA_CONTEXT: dict = {}


def _build_persona_context(uid: str, backend_dir: str = "backend") -> dict:
    """Walk backend/{uid}/*.json once; produce the lookup bank.

    Returns:
      top_prefs        : list[(persona_item, count)]  recency-weighted
      top_categories   : list[(category, count)]
      top_hashtags     : list[(hashtag, count)]
      recent_self_posts: list[caption-strings] (last 5)
      recent_reactions : list[(content_summary, action)] (last 10 explicit positives)
    """
    from collections import Counter
    pref_counter: Counter = Counter()
    pref_meta: dict = {}  # persona_item -> latest seen pref dict
    cat_counter: Counter = Counter()
    hashtag_counter: Counter = Counter()
    self_posts: list = []
    recent_pos: list = []
    for app_file in ("instagram.json", "facebook.json", "threads.json", "chatbot.json"):
        p = Path(backend_dir) / str(uid) / app_file
        if not p.exists():
            continue
        try:
            evs = json.loads(p.read_text())
        except Exception:
            continue
        # Sort recent-first within each app
        evs_sorted = sorted(evs, key=lambda e: e.get("source_timestamp", 0), reverse=True)
        for e in evs_sorted:
            for h in (e.get("source_hashtags") or []):
                if h:
                    hashtag_counter[h.lower().lstrip("#")] += 1
            for pref in (e.get("preferences") or []):
                if not isinstance(pref, dict):
                    continue
                pi = pref.get("persona_item") or ""
                if pi:
                    pref_counter[pi] += 1
                    pref_meta.setdefault(pi, pref)
                cat = pref.get("category")
                if cat:
                    cat_counter[cat] += 1
            if e.get("is_self_authored") and not e.get("is_dm") and len(self_posts) < 5:
                cap = (e.get("content") or {}).get("caption", "")
                if cap:
                    self_posts.append(cap)
            itype = e.get("source_interaction_type", "")
            if itype.startswith("explicit_positive") and len(recent_pos) < 10:
                cap = (e.get("content") or {}).get("caption", "") or (e.get("content") or {}).get("title", "")
                action = (e.get("interaction_format") or {}).get("action", "")
                if cap:
                    recent_pos.append((cap[:80], action))
    return {
        "top_prefs": [(pi, n) for pi, n in pref_counter.most_common(8)],
        "pref_meta": pref_meta,
        "top_categories": [(c, n) for c, n in cat_counter.most_common(6)],
        "top_hashtags": [(h, n) for h, n in hashtag_counter.most_common(15)],
        "recent_self_posts": self_posts,
        "recent_pos": recent_pos,
    }



# Per-task ground-truth extractor — given the parsed instance_json from
# benchmark/{uid}/queries.csv, return a {ground_truth, rubric_tags} dict.
# Default extractor returns the task_id only with empty GT.
# ---------------------------------------------------------------------------
# Per-task GROUND-TRUTH extractor — returns a rich dict the JS template
# renders as multiple sections on the test card. Keys (all optional):
#   ground_truth        : str   short headline blurb
#   candidates          : list[(idx, title, origin)]  for ranking tasks
#   held_out_pref       : str   the persona-item text the agent should align to
#   target_prefs        : list[str]  preferences the agent SHOULD surface
#   tool_call_rules     : list[str]  agentic write/read constraints
#   final_state_expected: dict  {must_contain_count, must_not_contain}
#   warn_frame          : {must_mention, must_not_mention, polarity}
#   signal_evidence     : list  active_mistake_prevention cross-signal trace
#   rubric_tags         : list[str]
# ---------------------------------------------------------------------------

def _gt_default(inst: dict) -> dict:
    return {"ground_truth": "", "rubric_tags": []}


def _truncate(s, n=120):
    s = s if isinstance(s, str) else str(s or "")
    return s[: n - 1] + "…" if len(s) > n else s


def _gt_personalized_feed_ranking(inst: dict) -> dict:
    held = inst.get("held_out_idx")
    slate = inst.get("slate") or []
    origins = inst.get("origin_by_idx") or []
    title = ""
    if isinstance(held, int) and 0 <= held < len(slate):
        title = slate[held].get("title") or slate[held].get("caption") or ""
    cands = []
    for i, c in enumerate(slate):
        origin = origins[i] if i < len(origins) else "?"
        cands.append({
            "idx": i,
            "title": _truncate(c.get("title") or c.get("caption") or "", 90),
            "hashtags": c.get("hashtags") or [],
            "origin": origin,
            "is_held_out": (i == held),
        })
    return {
        "ground_truth": f"Rank the held-out target at position 1 (the actual next item the user engaged with): {_truncate(title, 100)}",
        "candidates": cands,
        "rubric_tags": [
            "Place the held-out target at rank 1 (it is the actual next item the user will engage with).",
            "Among non-target items, prefer past_positive and future_positive over irrelevant/random/negative.",
            "Do not place known-disliked (negative) items in the top 3.",
            "If multiple plausible items exist, ordering should reflect the user's most recent engagement patterns.",
        ],
    }


def _gt_chatbot_proactive(inst: dict) -> dict:
    held = inst.get("held_out_preference") or {}
    held_pi = (held.get("persona_item") or "").strip()
    # Top-k relevant prefs are the query-relevant subset (hashtag-overlap
    # filtered at build time). Dedup defensively — same persona_item can
    # appear under multiple canonicals.
    seen: set[str] = set()
    top_k: list[str] = []
    for p in (inst.get("top_k_relevant_prefs") or []):
        pi = (p.get("persona_item") or "").strip()
        if pi and pi != held_pi and pi not in seen:
            top_k.append(pi)
            seen.add(pi)
        if len(top_k) >= 4:
            break
    prior = inst.get("prior_conversation") or []
    return {
        "ground_truth": _truncate(held_pi or "(no held-out preference; rubric checks restraint)", 200),
        "held_out_pref": held_pi,
        "top_k_relevant": top_k,
        "prior_conversation": prior[-6:] if prior else [],
        "rubric_tags": [
            "Weave in the held-out preference if (and only if) it fits the question.",
            "Surface at most 1-2 supporting preferences from the top-k list, only when relevant.",
            "Do not bring up unrelated preferences just because the user is known to like them.",
            "Match the user's tone; never lecture.",
        ],
    }


def _gt_chatbot_restraint(inst: dict) -> dict:
    """chatbot_restraint_control — testing whether the agent over-personalizes
    on a generic question. The user's question is unrelated to their personal
    preferences; the correct behavior is to answer generically without
    surfacing any persona context."""
    top_k = [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")]
    return {
        "ground_truth": "Over-personalization test on irrelevant preferences — the user asked a generic question; the agent must NOT volunteer any personal preferences.",
        "correct_but_irrelevant_prefs": top_k,
        "rubric_tags": [
            "Do NOT volunteer any personal preferences in the answer.",
            "Answer the user's question with general information only.",
            "Match the user's stated need without inferring related interests.",
            "Even though the listed preferences ARE correct facts about this user, they are not relevant to this question, so leave them out.",
        ],
    }


def _gt_at_ai_directive(inst: dict) -> dict:
    cands = inst.get("candidates") or []
    pos = set(inst.get("positive_indices") or [])
    carve = set(inst.get("carveout_indices") or [])
    cand_list = []
    for i, c in enumerate(cands):
        cand_list.append({
            "idx": i,
            "title": _truncate(c.get("title") or c.get("caption") or "", 90),
            "hashtags": c.get("hashtags") or [],
            "origin": "match" if i in pos else ("carve_out" if i in carve else "filler"),
            "is_held_out": (i in pos),
        })
    action = inst.get("directive_action", "")
    return {
        "ground_truth": f"User just sent the @ai directive '{action}' on hashtags {inst.get('directive_hashtags', [])}. Top-1 should align with the directive.",
        "candidates": cand_list,
        "rubric_tags": [
            f"Honor the @ai directive '{action}': for 'recommend more' / 'focus topic' surface positive_indices first; for 'stop recommending' / 'not interested' / 'feels off' do the opposite.",
            "Do NOT surface any candidate listed as carve_out (the user explicitly excluded those).",
            "Across the top-3 ranking, behavior should match the directive's intent — not just the top-1 match.",
        ],
    }


def _gt_active_mistake_prevention(inst: dict) -> dict:
    ef = inst.get("expected_warning_frame") or {}
    sigs = (inst.get("cross_signal_signals") or {}).get("signal_evidence") or []
    polarity = inst.get("polarity", "")
    if polarity == "warn":
        rubrics = [
            "Detect the cross-signal contradiction in the user's situation and proactively warn them.",
            "The warning should mention the items listed in must_mention.",
            "The warning must NOT mention items listed in must_not_mention.",
            "Phrase the warning respectfully — name the concern, do not pathologize the user.",
            "Be specific and actionable; do not produce a vague generic caution.",
        ]
        gt_blurb = f"WARN polarity — agent should detect the contradiction and warn: {_truncate(inst.get('mistake_summary', ''), 200)}"
    else:
        rubrics = [
            "There is NO actual contradiction in this foil scenario — the cross-signals all line up.",
            "Do NOT issue a warning. Stay silent on hypothetical concerns.",
            "Answer the user's question helpfully without flagging anything as risky.",
            "Items in must_not_mention must NOT appear in your response.",
        ]
        gt_blurb = f"FOIL polarity — no real contradiction exists; agent should NOT warn. (Mirror situation: {_truncate(inst.get('mistake_summary', ''), 160)})"
    return {
        "ground_truth": gt_blurb,
        "warn_frame": {
            "polarity": polarity,
            "must_mention": ef.get("must_mention") or [],
            "must_not_mention": ef.get("must_not_mention") or [],
        },
        "signal_evidence": [
            {
                "source": s.get("source", ""),
                "ts": s.get("ts", ""),
                "ref": s.get("ref", ""),
                "quote": _truncate(s.get("quote", ""), 140),
            } for s in sigs
        ][:6],
        "rubric_tags": rubrics,
    }


def _gt_irrelevant_query_restraint(inst: dict) -> dict:
    cands = inst.get("candidates") or []
    origins = inst.get("origin_by_idx") or []
    held_text = inst.get("held_out_persona_item") or ""
    irrels = inst.get("irrelevant_persona_items") or []
    cand_list = [{
        "idx": i,
        "title": _truncate(c.get("persona_item") or c.get("title") or str(c), 100),
        "origin": origins[i] if i < len(origins) else "?",
        "is_held_out": (origins[i] == "held_out") if i < len(origins) else False,
    } for i, c in enumerate(cands)]
    return {
        "ground_truth": f"On app={inst.get('app', '')}: only the held-out persona item is relevant. Reject the irrelevant ones.",
        "candidates": cand_list,
        "irrelevant_persona_items": [_truncate(s, 100) for s in irrels[:4]],
        "rubric_tags": [
            "Select only the held-out persona item from the candidate pool — it is the only one that fits this query.",
            "Reject all candidates marked as irrelevant; do NOT surface them even if they share some surface tokens with the query.",
        ],
    }


def _gt_preference_removal_regen(inst: dict) -> dict:
    held = inst.get("held_out_preference") or {}
    return {
        "ground_truth": f"The user previously asked you to forget this preference: '{_truncate(held.get('persona_item', ''), 160)}'. Regenerate without using it.",
        "held_out_pref": held.get("persona_item", ""),
        "top_k_relevant": [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")],
        "rubric_tags": [
            "Do NOT use the removed preference in your regenerated response.",
            "Generate content from the remaining top-k relevant preferences only.",
            "The regenerated response should be substantively different from one that uses the removed preference (do not produce near-identical text).",
        ],
    }


def _gt_repetition_fatigue_pairs(inst: dict) -> dict:
    return {
        "ground_truth": f"Pair test on {inst.get('target_app', '')}: dominant category PRE = '{inst.get('dominant_category_pre','')}', shift_category = '{inst.get('shift_category','')}'. Recommendations should diversify between t_early and t_late.",
        "extra_meta": {
            "dominant_category_pre": inst.get("dominant_category_pre"),
            "shift_category": inst.get("shift_category"),
            "t_early": inst.get("t_early"),
            "t_late": inst.get("t_late"),
        },
        "rubric_tags": [
            "Diversify recommendations between the early and late timepoint — do not keep recommending the same dominant category if the user's recent activity has shifted.",
            "Show recency sensitivity: as the user's recent engagement shifts toward shift_category, your top-1 should follow.",
            "Avoid producing identical or near-identical responses across the two timepoints.",
        ],
    }


def _gt_repetition_fatigue_sequences(inst: dict) -> dict:
    queries = inst.get("queries") or []
    return {
        "ground_truth": f"Sequence of {len(queries)} successive queries on the same persona. Across the sequence, agent must reduce repetition of the same preferences.",
        "extra_meta": {"n_queries": len(queries)},
        "rubric_tags": [
            "Across the sequence of queries, do not repeatedly surface the same 1-2 preferences each time.",
            "Show variety in your recommendations even when the topic is similar — vary which preferences you draw on.",
            "Do not reuse a preference that you already pushed in an earlier query if the user did not ask for more of it.",
        ],
    }


def _gt_context_shift_scenarios(inst: dict) -> dict:
    return {
        "ground_truth": f"Scenario '{inst.get('name', inst.get('scenario_id', ''))}': {_truncate(inst.get('notes', ''), 160)}",
        "carve_out": _truncate(inst.get("carve_out", ""), 200),
        "forbidden_items": [_truncate(s, 100) for s in (inst.get("forbidden_items") or [])[:4]],
        "rubric_tags": [
            "Recognize that the user's context has shifted (see carve_out) — do not assume their old preferences still apply.",
            "Do NOT surface any of the forbidden_items even though they relate to the user's earlier preferences.",
            "Adjust the response to fit the new context (e.g. talking to a different audience, in a different setting).",
            "Be respectful of relationships if the carve-out involves another person.",
        ],
    }


def _gt_daily_personalized_briefing(inst: dict) -> dict:
    day_label = inst.get('day_label', '')
    prior = inst.get('prior_day_label', 'prior days')
    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    top_cats = [c for c, _ in (_PERSONA_CONTEXT.get("top_categories") or [])][:4]
    pref_lines = "\n".join(f"  - {pi}" for pi in top_prefs) or "  (no recent prefs available)"
    template = (
        f"Example shape of a good briefing for {day_label}:\n\n"
        f"  \"Good morning! Here's what's worth your time today:\n"
        f"   1. Quick update on {top_cats[0] if top_cats else 'your top interest'} — this morning a few new items popped up.\n"
        f"   2. Something in {top_cats[1] if len(top_cats) > 1 else 'your second top area'} you'd probably want to see.\n"
        f"   3. One item from {top_cats[2] if len(top_cats) > 2 else 'a third interest area'} that fits your usual taste.\"\n\n"
        f"Concrete content: should reference at least 2 of these recent persona items:\n{pref_lines}"
    )
    return {
        "ground_truth": template,
        "rubric_tags": [
            f"The briefing must reference at least 2 distinct items from the user's recent top preferences (e.g. one of: {', '.join(top_prefs[:3]) if top_prefs else '...'}).",
            f"Length should be 3-5 distinct items, structured as a list or short paragraphs (no exhaustive dump).",
            f"Do NOT include items the user has clearly moved past from {prior} — only what's currently relevant.",
            "Each item should explain WHY it's relevant (1 sentence: \"based on your recent X\").",
            "Tone should be light, concise, conversational — not formal or list-of-headlines.",
        ],
    }


def _gt_personalized_search_ranking(inst: dict) -> dict:
    day_label = inst.get('day_label', '')
    recent = inst.get('recent_pref_summary', [])
    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    top_cats = [c for c, _ in (_PERSONA_CONTEXT.get("top_categories") or [])][:3]
    cat_str = ", ".join(top_cats) if top_cats else "the user's recent preferences"
    pref_lines = "\n".join(f"  - {pi}" for pi in top_prefs) or "  (no recent prefs available)"
    recent_str = "\n".join(f"  - {p.get('persona_item','?')} (recent count={p.get('count', '?')})"
                           for p in (recent or [])[:5]) or "  (none recorded)"
    template = (
        f"This is a generic search query at {day_label}; the e4 builder did NOT carry a literal user-typed search string.\n"
        f"What the agent receives instead: the user's recent preference summary.\n\n"
        f"Recent pref summary in this instance:\n{recent_str}\n\n"
        f"Example shape of an ideal personalized ranking (top-5 search results):\n"
        f"  rank 1: an item primarily on {top_cats[0] if top_cats else cat_str} (most-engaged recent topic)\n"
        f"  rank 2: an item on {top_cats[1] if len(top_cats) > 1 else 'a second top topic'} (cross-topic variety)\n"
        f"  rank 3: an item that bridges multiple of the user's interests\n"
        f"  rank 4: a high-quality item only loosely matching the user's preferences (universal relevance)\n"
        f"  rank 5: filler / generic relevance\n\n"
        f"Concrete acceptance — the top-3 should each map to one of these recent persona items:\n{pref_lines}"
    )
    return {
        "ground_truth": template,
        "rubric_tags": [
            f"Top-1 must align with the user's most-engaged category among: {', '.join(top_cats) if top_cats else 'top categories'}.",
            "Top-3 should collectively cover at least 2 distinct user preference categories — do not stack 3 items on the same topic.",
            "Items below rank 3 may be generically relevant; do NOT need to be heavily personalized.",
            "Do not over-personalize: if a search query were clearly factual or generic (and this one is generic), still keep some non-persona items in the lower ranks for variety.",
        ],
    }


def _gt_short_vs_long_term_lifecycle(inst: dict) -> dict:
    horizon = inst.get('horizon_type', '?')
    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    pref_meta = _PERSONA_CONTEXT.get("pref_meta") or {}
    short_examples = [pi for pi in top_prefs if (pref_meta.get(pi) or {}).get("time_horizon") == "short_term"][:2]
    long_examples = [pi for pi in top_prefs if (pref_meta.get(pi) or {}).get("time_horizon") != "short_term"][:2]
    template = (
        f"Lifecycle test (horizon={horizon}). The agent must distinguish ephemeral preferences from durable ones.\n\n"
        f"Long-term preferences in this user's profile (should still be surfaced when relevant):\n"
        + ("\n".join(f"  - {pi}" for pi in long_examples) or "  (none labeled long-term)") + "\n\n"
        f"Short-term preferences in this user's profile (should fade after their stop_condition):\n"
        + ("\n".join(f"  - {pi}" for pi in short_examples) or "  (none labeled short-term)") + "\n\n"
        f"Example correct behavior: at the test moment, surface long-term prefs naturally; for any short-term pref past its expected_stop_ts, treat it as expired and do NOT surface it."
    )
    return {
        "ground_truth": template,
        "rubric_tags": [
            "Long-term preferences (identity, hobbies, persistent interests) should persist across time — surface them when relevant.",
            "Short-term preferences (travel plans, event prep, time-bounded interests) should fade after expected_stop_ts.",
            "If a short-term pref is past its expected_stop_ts at this test moment, do NOT surface it (treat as expired).",
            "If a short-term pref is still within its window, surface it normally.",
            "Do not invent durability — if a pref's time_horizon=short_term, respect that even when it would be useful.",
        ],
    }


def _gt_agentic(inst: dict) -> dict:
    """Generic agentic GT — surfaces tool_call_rules + final_state_expected
    + a task-specific rubric + a CONCRETE example of what a good agent
    response should look like, built from persona context."""
    task_id = inst.get("task_id", "")
    target = inst.get("target_app") or ""
    bits: list[str] = []
    if target:
        bits.append(f"target_app={target}")
    for k in ("update", "context", "draft", "topic", "moment", "thread_id", "recipient_name", "inbound_message"):
        if inst.get(k):
            bits.append(f"{k}={_truncate(str(inst[k]), 100)}")
    if inst.get("source_post"):
        sp = inst["source_post"]
        bits.append(f"source_post.caption={_truncate(sp.get('caption', ''), 100)}")

    # Build a CONCRETE example response per task using persona context.
    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    top_cats = [c for c, _ in (_PERSONA_CONTEXT.get("top_categories") or [])][:3]
    top_hashtags = [h for h, _ in (_PERSONA_CONTEXT.get("top_hashtags") or [])][:8]
    recent_posts = (_PERSONA_CONTEXT.get("recent_self_posts") or [])[:3]
    voice_sample = recent_posts[0] if recent_posts else ""

    examples: dict[str, str] = {
        "agentic_user_voice_post":
            f"Example digest post on {target}: \"Catching up after the week — {top_hashtags[0] if top_hashtags else 'top topic'} had a few good moments, "
            f"the {top_hashtags[1] if len(top_hashtags) > 1 else 'next topic'} crowd is heating up again, and a few new {top_cats[0] if top_cats else 'interest'} clips dropped. "
            f"Anyone else watching?\" (~30-50 words, mentions ≥2 distinct community topics)",
        "agentic_moment_recommendation":
            f"Example response for {inst.get('moment','this moment')}: "
            f"\"Try the new {top_cats[0] if top_cats else 'interest'} clip from this morning — quick watch, fits the {inst.get('moment', 'moment')} vibe. "
            f"You'd also like the {top_cats[1] if len(top_cats) > 1 else 'second-interest'} thread from yesterday.\" (~20-40 words, fits time-of-day)",
        "agentic_dm_digest":
            f"Example DM digest: \"Recent DMs on {target}: friend_1 asked about Saturday plans (haven't replied yet), friend_2 shared a {top_hashtags[0] if top_hashtags else 'topic'} post, "
            f"and there's an unread thread from a third friend about an {top_cats[0] if top_cats else 'interest'} event. Three things waiting on you.\" "
            f"(names ≥2 distinct correspondents, summary only — no auto-replies)",
        "agentic_cross_app_repost":
            f"Example repost on {target}: paraphrase the source post's topic in the user's voice, keep the core hashtag, add 1-2 user-style adjectives. "
            f"Source: \"{_truncate((inst.get('source_post') or {}).get('caption', ''), 100)}\" → "
            f"Example output: \"crossposting from IG: still thinking about this — {top_hashtags[0] if top_hashtags else '#topic'}\"",
        "agentic_auto_reply":
            f"Example reply to inbound \"{_truncate(inst.get('inbound_message', ''), 80)}\": "
            f"a 1-3 sentence DM that addresses what the friend said, doesn't fabricate facts about them, "
            f"and matches the user's casual short style. E.g. \"yeah that works, see you saturday\" or \"hmm let me check and get back to you\".",
        "agentic_vague_refind":
            f"Example response for topic '{inst.get('topic','')}': surface a SPECIFIC past post the user previously engaged with on this topic, "
            f"with its source_object_id and a 1-line description. Don't return a vague \"here are some posts\" response — find the actual one.",
        "agentic_composed_post": (
            f"Example post on {target}, paraphrasing the user's update \"{_truncate(inst.get('update',''), 80)}\":\n"
            f"  \"{voice_sample[:120] if voice_sample else (chr(10).join(['use the users voice — short, casual, with hashtags from their typical pool']))}\"\n"
            f"Should match the user's typical voice (length, hashtag style). Acceptable hashtags: {top_hashtags[:4] if top_hashtags else '...'}"
        ),
        "agentic_chatbot_dispatch":
            f"Example dispatched post on {target} addressing context \"{_truncate(inst.get('context',''), 100)}\": "
            f"a single post that captures the gist of the context, in the user's voice, with appropriate hashtags ({top_hashtags[:3] if top_hashtags else '...'}). "
            f"Do NOT post on apps other than {target}.",
        "agentic_draft_audit":
            f"Example audit response for label='{inst.get('draft_label', '?')}': "
            f"name the specific issue you found (privacy/tone/leak/etc.) and suggest a 1-line fix. "
            f"For label='benign' say so honestly without inventing problems. Do NOT post the draft.",
        "agentic_collection_curation":
            f"Example curation on {target}: 3-4 thematic groups, each containing 4-8 items the user has saved/liked. "
            f"Group themes should follow the user's top categories: {', '.join(top_cats) if top_cats else '...'}. "
            f"Within each group, items should share ≥60% hashtag overlap. Read-only — no posts.",
        "agentic_group_dm_summary":
            f"Example summary of group thread on {target}: "
            f"\"3 friends in the thread discussed [topic1] and [topic2]; friend_X is pushing for [decision], friend_Y disagreed. "
            f"Outstanding question: [pending question].\" Names ≥2 participants + ≥2 topics, no unsolicited DMs.",
        "agentic_wrong_recipient_check":
            f"Example flag: \"Heads up — there are two contacts named '{inst.get('recipient_name', '?')}' "
            f"in your friends list. Did you mean [Name A] (recent thread about X) or [Name B] (you've been talking about Y with them)?\" "
            f"Do NOT send the DM.",
        "agentic_proactive_daily_catchup":
            f"Example catch-up: \"Three things to look at today: "
            f"(1) new {top_cats[0] if top_cats else 'topic-1'} content from yesterday you haven't seen, "
            f"(2) a friend's DM that's been sitting for a day, "
            f"(3) an item in {top_cats[1] if len(top_cats) > 1 else 'topic-2'} matching your recent interest.\" "
            f"≥3 distinct items, each tied to actual recent activity.",
        "agentic_trending_alert":
            f"Example alert: \"You'd probably care about these trending topics right now: "
            f"#{top_hashtags[0] if top_hashtags else 'topic1'} (matches your {top_cats[0] if top_cats else 'top interest'}), "
            f"#{top_hashtags[1] if len(top_hashtags) > 1 else 'topic2'} (cross-cuts your {top_cats[1] if len(top_cats) > 1 else 'second interest'}).\" "
            f"Skip trends the user has shown no interest in.",
    }
    example = examples.get(task_id, "")

    # Per-task rubric phrasing
    task_rubrics: dict[str, list[str]] = {
        "agentic_user_voice_post": [
            "Reference at least 2 distinct community topics from the user's last week of activity on target_app.",
            "Match the user's voice (tone, length, hashtag style).",
            "Do NOT post on apps other than target_app.",
        ],
        "agentic_moment_recommendation": [
            "Recommendation should fit the time-of-day moment (lunch / commute / shower / evening).",
            "Use at least one of the user's known interests so the recommendation feels personalized.",
            "Do not send a DM as part of this — it is read-only / chatbot-only.",
        ],
        "agentic_dm_digest": [
            "Summarize the user's recent DMs on target_app, naming at least 2 distinct correspondents.",
            "Do NOT send any DMs as part of this — read-only summary.",
            "Do NOT compose a new post.",
        ],
        "agentic_cross_app_repost": [
            "Compose a repost on target_app that references the source post's hashtag or topic.",
            "Match the user's voice (do not just copy the source post verbatim).",
            "Exactly 1 create_post on target_app; 0 posts on the source app.",
        ],
        "agentic_auto_reply": [
            "Reply must address the inbound DM specifically — do not respond with a generic acknowledgment.",
            "Do NOT fabricate facts about the sender or invent context that wasn't there.",
            "Exactly 1 send_dm on target_app; 0 posts.",
        ],
        "agentic_vague_refind": [
            "Surface a specific past post matching the topic — return a concrete reference, not a vague description.",
            "Do not create any new posts as part of this.",
        ],
        "agentic_composed_post": [
            "Post must reflect the user's update text (paraphrase or expand it).",
            "Match the user's voice (length, tone, hashtag style).",
            "Exactly 1 create_post on target_app; 0 send_dm.",
        ],
        "agentic_chatbot_dispatch": [
            "Dispatch the post to target_app (and only target_app).",
            "The dispatched content must address inst.context — do not produce off-topic content.",
            "Exactly 1 create_post on target_app; 0 posts on other apps.",
        ],
        "agentic_draft_audit": [
            "Identify at least 1 issue in the draft (privacy leak, tone mismatch, etc.) when one exists.",
            "Do NOT post the draft — this is audit-only, all writes must be 0.",
            "If the draft is benign, say so honestly without inventing problems.",
        ],
        "agentic_collection_curation": [
            "Group items into thematically coherent collections — within-group similarity should be high.",
            "Do not create any new posts; this is curation only.",
        ],
        "agentic_group_dm_summary": [
            "Summary should name the key participants in the group thread.",
            "Cover at least 2 key topics from the conversation.",
            "Do NOT send any DM without explicit user approval — summary only.",
        ],
        "agentic_wrong_recipient_check": [
            "Detect the recipient name collision and FLAG it — ask the user to confirm which person they meant.",
            "Do NOT send the DM until the user confirms the recipient.",
        ],
        "agentic_proactive_daily_catchup": [
            "Surface at least 3 catch-up items relevant to the user from recent activity.",
            "Do not create any posts as part of this; chatbot-routed surfacing only.",
        ],
        "agentic_trending_alert": [
            "Flag trending topics that intersect the user's positive hashtags.",
            "Skip trending topics the user has shown no interest in.",
            "Do not create posts as part of this; chatbot-routed alert only.",
        ],
    }
    rubric = task_rubrics.get(task_id, [
        "Call exactly the write tools specified in tool_call_rules — no more, no fewer.",
        "Produce overlay writes matching final_state_expected (must_contain_count + must_not_contain).",
        "Match the user's voice in any composed content.",
    ])
    # Always append the structural / write rules as additional rubric items
    rubric = rubric + [
        "Tool calls must satisfy tool_call_rules (counts of each tool).",
        "Overlay writes must satisfy final_state_expected (must_contain_count and must_not_contain).",
    ]

    setup = " | ".join(bits) if bits else "(agentic task; see tool rules + final state)"
    full_gt = setup + ("\n\n" + example if example else "")
    return {
        "ground_truth": full_gt,
        "tool_call_rules": inst.get("tool_call_rules") or [],
        "final_state_expected": inst.get("final_state_expected") or {},
        "rubric_tags": rubric,
    }


TEST_GT_EXTRACTORS = {
    "personalized_feed_ranking":           _gt_personalized_feed_ranking,
    "slate_ranking":                       _gt_personalized_feed_ranking,  # v1 alias
    "chatbot_proactive_personalization":   _gt_chatbot_proactive,
    "chatbot_response_proactive":          _gt_chatbot_proactive,           # v1 alias
    "over_personalization_chatbot_text":   _gt_chatbot_restraint,
    "chatbot_restraint_control":           _gt_chatbot_restraint,           # v2 alias
    "chatbot_response_control":            _gt_chatbot_restraint,           # v1 alias
    "at_ai_directive_followup":            _gt_at_ai_directive,
    "e2_at_ai_followup":                   _gt_at_ai_directive,             # v1 alias
    "active_mistake_prevention":           _gt_active_mistake_prevention,
    "e6_active_mistake_prevention":        _gt_active_mistake_prevention,   # v1 alias
    "over_personalization_distractor_reject": _gt_irrelevant_query_restraint,
    "irrelevant_query_restraint":          _gt_irrelevant_query_restraint,  # v2 alias
    "preference_removal_regen":            _gt_preference_removal_regen,
    "repetition_fatigue_pairs":            _gt_repetition_fatigue_pairs,
    "repetition_fatigue_sequences":        _gt_repetition_fatigue_sequences,
    "context_shift_scenarios":             _gt_context_shift_scenarios,
    "daily_personalized_briefing":         _gt_daily_personalized_briefing,
    "personalized_search_ranking":         _gt_personalized_search_ranking,
    "short_vs_long_term_lifecycle":        _gt_short_vs_long_term_lifecycle,
    # All agentic_* tasks share the generic agentic extractor
    "agentic_user_voice_post":            _gt_agentic,
    "agentic_moment_recommendation":       _gt_agentic,
    "agentic_dm_digest":                   _gt_agentic,
    "agentic_cross_app_repost":            _gt_agentic,
    "agentic_auto_reply":                  _gt_agentic,
    "agentic_vague_refind":                _gt_agentic,
    "agentic_composed_post":               _gt_agentic,
    "agentic_chatbot_dispatch":            _gt_agentic,
    "agentic_draft_audit":                 _gt_agentic,
    "agentic_collection_curation":         _gt_agentic,
    "agentic_group_dm_summary":            _gt_agentic,
    "agentic_wrong_recipient_check":       _gt_agentic,
    "agentic_proactive_daily_catchup":     _gt_agentic,
    "agentic_trending_alert":              _gt_agentic,
}


def _gt_agentic_default(inst: dict) -> dict:
    """Fallback for unknown agentic tasks — defers to the generic agentic extractor."""
    return _gt_agentic(inst)


# ---------------------------------------------------------------------------
# Per-task USER-QUERY extractor — what the test card SHOWS as the "user's
# message at this time and place." Some tasks carry a natural user message
# (chatbot, agentic_auto_reply, e6); for ranking-style tasks we synthesize a
# task-shaped intent ("what should I be shown next on Instagram?") so the
# card has something readable.
# ---------------------------------------------------------------------------

def _q_default(inst: dict) -> str:
    return inst.get("user_query") or inst.get("user_message") or inst.get("query_text") or ""


def _q_personalized_feed_ranking(inst: dict) -> str:
    app = inst.get("app") or "this app"
    return f"[ranking task] What should I be shown next on {app}?"


def _q_chatbot(inst: dict) -> str:
    return inst.get("user_query") or inst.get("user_message") or "[chatbot turn]"


def _q_at_ai_directive(inst: dict) -> str:
    msg = inst.get("directive_user_message") or ""
    action = inst.get("directive_action") or ""
    return f"@ai {action}: {msg}" if msg else f"@ai {action}"


def _q_active_mistake_prevention(inst: dict) -> str:
    return inst.get("user_query") or inst.get("triggering_user_query") or "[mistake-prevention probe]"


def _q_agentic_user_voice_post(inst: dict) -> str:
    return f"[agentic] compose a post in the user's voice on {inst.get('target_app', '')}"


def _q_agentic_moment_recommendation(inst: dict) -> str:
    return f"[agentic] recommend something for {inst.get('moment', '')}"


def _q_agentic_dm_digest(inst: dict) -> str:
    return f"[agentic] summarize my recent DMs on {inst.get('target_app', '')}"


def _q_agentic_cross_app_repost(inst: dict) -> str:
    src = inst.get("source_post") or {}
    cap = (src.get("caption") or "")[:120]
    return f"[agentic] repost this to {inst.get('target_app', '')}: {cap}"


def _q_agentic_auto_reply(inst: dict) -> str:
    sender = inst.get("sender_id") or "friend"
    msg = inst.get("inbound_message") or ""
    return f"[incoming DM from {sender}] {msg}"


def _q_agentic_vague_refind(inst: dict) -> str:
    return f"find that post I saw about {inst.get('topic', '')}"


def _q_agentic_composed_post(inst: dict) -> str:
    return f"[agentic] post on {inst.get('target_app', '')}: {inst.get('update', '')}"


def _q_agentic_chatbot_dispatch(inst: dict) -> str:
    return inst.get("context") or "[agentic dispatch]"


def _q_agentic_draft_audit(inst: dict) -> str:
    draft = (inst.get("draft") or "")[:160]
    return f"[agentic] audit this draft for {inst.get('target_app', '')}: {draft}"


def _q_agentic_collection_curation(inst: dict) -> str:
    return f"[agentic] curate collections on {inst.get('target_app', '')}"


def _q_agentic_group_dm_summary(inst: dict) -> str:
    return f"[agentic] summarize the group thread on {inst.get('target_app', '')}"


def _q_agentic_wrong_recipient_check(inst: dict) -> str:
    return f"[agentic] DM to {inst.get('recipient_name', '?')}: {(inst.get('draft') or '')[:120]}"


def _q_agentic_proactive_daily_catchup(inst: dict) -> str:
    return "what should I catch up on today?"


def _q_agentic_trending_alert(inst: dict) -> str:
    return "anything trending I care about right now?"


def _q_daily_personalized_briefing(inst: dict) -> str:
    return "[daily briefing] give me a personalized morning brief"


def _q_personalized_search_ranking(inst: dict) -> str:
    explicit = inst.get('query_text') or inst.get('user_query') or inst.get('query', '')
    if explicit:
        return f"[search] {explicit}"
    # The e4 builder doesn't carry a literal search query — synthesize a
    # plausible "what's good for me right now" query from the user's top
    # categories so the test card has something readable for the agent.
    cats = [c for c, _ in (_PERSONA_CONTEXT.get("top_categories") or [])][:2]
    if cats:
        return f"[search, no specific query — generic 'what should I look at right now' on {' / '.join(cats)} themes]"
    return "[search, no specific query — generic 'what should I look at right now']"


def _q_short_vs_long_term_lifecycle(inst: dict) -> str:
    return "[lifecycle ranking] short-term vs long-term preference test"


TEST_QUERY_EXTRACTORS = {
    "personalized_feed_ranking":           _q_personalized_feed_ranking,
    "slate_ranking":                       _q_personalized_feed_ranking,
    "chatbot_proactive_personalization":   _q_chatbot,
    "chatbot_response_proactive":          _q_chatbot,
    "over_personalization_chatbot_text":   _q_chatbot,
    "chatbot_restraint_control":           _q_chatbot,
    "chatbot_response_control":            _q_chatbot,
    "at_ai_directive_followup":            _q_at_ai_directive,
    "e2_at_ai_followup":                   _q_at_ai_directive,
    "active_mistake_prevention":           _q_active_mistake_prevention,
    "e6_active_mistake_prevention":        _q_active_mistake_prevention,
    "agentic_user_voice_post":            _q_agentic_user_voice_post,
    "agentic_moment_recommendation":       _q_agentic_moment_recommendation,
    "agentic_dm_digest":                   _q_agentic_dm_digest,
    "agentic_cross_app_repost":            _q_agentic_cross_app_repost,
    "agentic_auto_reply":                  _q_agentic_auto_reply,
    "agentic_vague_refind":                _q_agentic_vague_refind,
    "agentic_composed_post":               _q_agentic_composed_post,
    "agentic_chatbot_dispatch":            _q_agentic_chatbot_dispatch,
    "agentic_draft_audit":                 _q_agentic_draft_audit,
    "agentic_collection_curation":         _q_agentic_collection_curation,
    "agentic_group_dm_summary":            _q_agentic_group_dm_summary,
    "agentic_wrong_recipient_check":       _q_agentic_wrong_recipient_check,
    "agentic_proactive_daily_catchup":     _q_agentic_proactive_daily_catchup,
    "agentic_trending_alert":              _q_agentic_trending_alert,
    "daily_personalized_briefing":         _q_daily_personalized_briefing,
    "personalized_search_ranking":         _q_personalized_search_ranking,
    "short_vs_long_term_lifecycle":        _q_short_vs_long_term_lifecycle,
}


def _load_test_samples(
    uid: str,
    benchmark_dir: str = "benchmark",
    backend_dir: str = "backend",
    include_instance_full: bool = False,
) -> list[dict]:
    """Walk benchmark/{uid}/queries.csv → list of test-sample dicts.

    Each test sample is rendered as a STANDALONE timeline card at its own
    timestamp (sorted alongside regular events + calendar mods), with a
    distinct background color. Geo location is computed JS-side by walking
    backwards through events to find the nearest preceding event_location.

    Per-sample fields:
      ts (int)         — the moment the user is notionally asking
      ts_iso (str)     — formatted timestamp
      task_type        — e.g. "personalized_feed_ranking"
      task_family      — e.g. "agentic"
      query_id         — e.g. "115:0042:e6_115_p1_warn"
      query_text       — what the user (or the agent's prompt) effectively says
      ground_truth     — short blurb describing the expected answer
      rubric_tags      — list[str] of which rubric dimensions apply

    When ``include_instance_full=True``, each sample also carries
    ``instance_full`` — the parsed instance_json dict from the CSV row,
    used by ``dump_test_samples_json`` so downstream tooling can read
    every field the builder emitted (blind_check_score, arm, polarity,
    etc.).
    """
    qcsv = os.path.join(benchmark_dir, str(uid), "queries.csv")
    out: list[dict] = []
    if not os.path.exists(qcsv):
        return out
    # Build the persona context bank ONCE; extractors use it to fill in
    # concrete expected-answer shapes when the instance itself is sparse.
    global _PERSONA_CONTEXT
    _PERSONA_CONTEXT = _build_persona_context(uid, backend_dir)
    csv.field_size_limit(10_000_000)
    with open(qcsv, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        for r in csv.DictReader(f):
            try:
                inst = json.loads(r.get("instance_json") or "{}")
            except Exception:
                inst = {}
            task_type = r.get("task_type", "")
            task_family = r.get("task_family", "")
            gt_extractor = (
                TEST_GT_EXTRACTORS.get(task_type)
                or (_gt_agentic_default if task_family == "agentic" else _gt_default)
            )
            q_extractor = TEST_QUERY_EXTRACTORS.get(task_type, _q_default)
            try:
                gt = gt_extractor(inst)
            except Exception as exc:
                gt = {"ground_truth": f"(extractor crashed: {type(exc).__name__})", "rubric_tags": []}
            try:
                q_text = q_extractor(inst) or ""
            except Exception:
                q_text = ""
            try:
                ts_int = int(r.get("ts") or 0)
            except Exception:
                ts_int = 0
            sample = {
                "ts": ts_int,
                "ts_iso": r.get("ts_iso", ""),
                "task_type": task_type,
                "task_family": task_family,
                "query_id": r.get("query_id", ""),
                "query_text": q_text,
                "ground_truth": gt.get("ground_truth", ""),
                "rubric_tags": gt.get("rubric_tags") or (r.get("rubric_tags", "").split(";") if r.get("rubric_tags") else []),
            }
            # Pass through optional rich fields when present — JS template
            # renders each one as its own labeled section on the test card.
            for k in ("candidates", "held_out_pref",
                     "top_k_relevant", "correct_but_irrelevant_prefs",
                     "tool_call_rules", "final_state_expected",
                     "warn_frame", "signal_evidence", "irrelevant_persona_items",
                     "carve_out", "forbidden_items", "prior_conversation", "extra_meta"):
                if k in gt:
                    sample[k] = gt[k]
            if include_instance_full:
                sample["instance_full"] = inst
            out.append(sample)
    return out


# ---------------------------------------------------------------------------
# Phase 1.A — test.json dump
#
# Re-uses _load_test_samples and enriches each sample with:
#   - query_kind, expected_behavior        (from evaluation.task_registry)
#   - ground_truth_preference (normalized)
#   - reference_example       (looked up in the app JSONs by persona_item)
#   - distractor_preferences  (normalized union of top_k_relevant /
#     correct_but_irrelevant_prefs / irrelevant_persona_items, each
#     tagged with `role`)
#   - instance_full           (pass-through of the original instance_json)
# ---------------------------------------------------------------------------

def _normalize_held_out(sample: dict) -> dict | None:
    """Pull the held-out preference into a canonical {persona_item,
    category, polarity, source_hashtags} shape — or None if there isn't
    one for this task type."""
    inst = sample.get("instance_full") or {}
    held_obj = inst.get("held_out_preference") or {}
    if not held_obj:
        # Some extractors put it under held_out_pref (already-stringified)
        # — fall back to that, but it loses category/polarity info.
        held_str = sample.get("held_out_pref")
        if not held_str:
            return None
        return {
            "persona_item": held_str,
            "category": "",
            "polarity": "positive",
            "source_hashtags": inst.get("source_hashtags") or [],
        }
    pi = held_obj.get("persona_item") or ""
    if not pi:
        return None
    return {
        "persona_item": pi,
        "category": held_obj.get("category") or "",
        "polarity": held_obj.get("polarity") or "positive",
        "source_hashtags": held_obj.get("source_hashtags") or inst.get("source_hashtags") or [],
    }


def _find_reference_example(uid: str, persona_item: str, t_test: int,
                            backend_dir: str = "backend") -> dict | None:
    """Walk app JSONs and return the closest-by-timestamp event that
    contains a preference whose persona_item matches. Returns a compact
    evidence record (the full event would be too heavy)."""
    if not persona_item:
        return None
    user_dir = Path(backend_dir) / str(uid)
    best: tuple[int, dict, str] | None = None  # (abs_dt, event, app)
    for app in APPS:
        p = user_dir / (app.lower() + ".json")
        if not p.exists():
            continue
        try:
            evs = json.loads(p.read_text())
        except Exception:
            continue
        for e in evs:
            for pref in (e.get("preferences") or []):
                if (pref.get("persona_item") or "").strip().lower() == persona_item.strip().lower():
                    ts = int(e.get("source_timestamp") or 0)
                    dt = abs(ts - t_test)
                    if best is None or dt < best[0]:
                        best = (dt, e, app)
                    break
    if best is None:
        return None
    _, ev, app = best
    content = ev.get("content") or {}
    snippet = content.get("caption") or content.get("title") or content.get("overall_description") or ""
    return {
        "source_object_id": ev.get("source_object_id", ""),
        "source_app": app,
        "source_timestamp": ev.get("source_timestamp", 0),
        "source_hashtags": ev.get("source_hashtags") or [],
        "interaction_format": ev.get("interaction_format") or {},
        "content_snippet": _truncate(snippet, 200),
    }


def _normalize_distractors(sample: dict) -> list[dict]:
    """Merge the various near-miss / irrelevant / privacy-flagged pools
    into a single list of {persona_item, category, polarity, role}."""
    out: list[dict] = []
    seen: set[str] = set()

    def _push(items, role):
        for it in items or []:
            if isinstance(it, dict):
                pi = it.get("persona_item") or ""
                cat = it.get("category") or ""
                pol = it.get("polarity") or "positive"
            else:
                pi = str(it or "")
                cat = ""
                pol = "positive"
            if not pi or pi in seen:
                continue
            seen.add(pi)
            out.append({
                "persona_item": pi,
                "category": cat,
                "polarity": pol,
                "role": role,
            })

    _push(sample.get("top_k_relevant"), "near_miss")
    _push(sample.get("correct_but_irrelevant_prefs"), "irrelevant")
    _push(sample.get("irrelevant_persona_items"), "privacy_flagged")
    # forbidden_items can also act as a do-not-surface pool for C2
    _push(sample.get("forbidden_items"), "privacy_flagged")
    return out


def dump_test_samples_json(
    uid: str,
    output_path: str | None = None,
    benchmark_dir: str = "benchmark",
    backend_dir: str = "backend",
) -> str:
    """Build backend/{uid}/test.json — every test query in one place.

    See the plan in /vast/home/b/bwjiang/.claude/plans/ for the schema.
    """
    from evaluation import task_registry as _tr

    samples = _load_test_samples(uid, benchmark_dir, backend_dir, include_instance_full=True)
    records: list[dict] = []
    for s in samples:
        task_type = s["task_type"]
        inst = s.get("instance_full") or {}
        held = _normalize_held_out(s)
        ref_ex = _find_reference_example(
            uid,
            held["persona_item"] if held else "",
            int(s.get("ts") or 0),
            backend_dir=backend_dir,
        ) if held else None
        record = {
            "query_id": s.get("query_id", ""),
            "task_family": s.get("task_family", ""),
            "task_type": task_type,
            "query_kind": _tr.get_query_kind(task_type),
            "expected_behavior": _tr.get_expected_behavior(task_type),
            "ts": s.get("ts", 0),
            "ts_iso": s.get("ts_iso", ""),
            "user_query": s.get("query_text") or None,
            "prior_conversation": s.get("prior_conversation"),
            "ground_truth_preference": held,
            "reference_answer": None,  # reserved for Phase 2
            "reference_example": ref_ex,
            "distractor_preferences": _normalize_distractors(s),
            "rubric_tags": s.get("rubric_tags") or [],
            "instance_full": inst,
        }
        # Compact: drop empty optional fields so the file stays readable.
        for k in ("prior_conversation",):
            if record[k] in (None, [], {}):
                record[k] = None
        records.append(record)

    if output_path is None:
        output_path = os.path.join(backend_dir, str(uid), "test.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return output_path


def _load_app_events(user_dir: str) -> tuple[list[dict], list[dict]]:
    """Load per-app JSON files and return (events, flat_prefs).

    ``events`` is the interaction-event list (new format). Each event
    has event-level fields + a ``preferences`` list.

    ``flat_prefs`` is the flattened preference list (for backwards-compat
    counts and profile serialization).

    Both lists are sorted by source_timestamp ascending.
    """
    all_events: list[dict] = []
    flat_prefs: list[dict] = []

    for app in APPS:
        path = os.path.join(user_dir, app.lower() + ".json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            if "preferences" in entry:
                # New interaction-event format
                entry["_app"] = app
                all_events.append(entry)
                for pref in entry["preferences"]:
                    flat = dict(pref)
                    flat["assigned_app"] = app
                    flat["source_object_id"] = entry.get("source_object_id", "")
                    flat["source_timestamp"] = entry.get("source_timestamp", 0)
                    flat["formatted_timestamp"] = entry.get("formatted_timestamp", "")
                    flat["source_hashtags"] = entry.get("source_hashtags", [])
                    flat["source_interaction_type"] = entry.get("source_interaction_type", "")
                    flat["interaction_format"] = entry.get("interaction_format", {})
                    flat_prefs.append(flat)
            else:
                # Legacy flat format — wrap as single-pref event
                entry.setdefault("assigned_app", app)
                event = {
                    "source_object_id": entry.get("source_object_id", ""),
                    "source_timestamp": entry.get("source_timestamp", 0),
                    "formatted_timestamp": entry.get("formatted_timestamp", ""),
                    "source_hashtags": entry.get("source_hashtags", []),
                    "source_interaction_type": entry.get("source_interaction_type", ""),
                    "interaction_format": entry.get("interaction_format", {}),
                    "_app": app,
                    "preferences": [{
                        "persona_item": entry.get("persona_item", ""),
                        "category": entry.get("category", ""),
                        "confidence_score_init": entry.get("confidence_score_init", 0),
                        "confidence_cross_referenced": entry.get("confidence_cross_referenced", 0),
                        "stereotype_mark": entry.get("stereotype_mark", "neutral"),
                        "split": entry.get("split", ""),
                        "update_history": entry.get("update_history", []),
                        "over_personalization_irrelevant": entry.get("over_personalization_irrelevant", []),
                    }],
                    "conversation": entry.get("conversation"),
                    "conversation_type": entry.get("conversation_type"),
                    "ask_to_forget": entry.get("ask_to_forget", False),
                }
                all_events.append(event)
                flat_prefs.append(entry)

    all_events.sort(key=lambda e: (int(e.get("source_timestamp") or 0), e.get("source_object_id", "")))
    flat_prefs.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("persona_item", "")))
    return all_events, flat_prefs


# Keep legacy loader for any external callers
def _load_app_prefs(user_dir: str) -> list[dict]:
    """Load per-app JSONs and return a flat list of preferences (legacy compat)."""
    _, flat = _load_app_events(user_dir)
    return flat


def _load_profile(user_dir: str) -> dict | None:
    path = os.path.join(user_dir, "profile.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_persona_html(user_id: str, backend_dir: str = "backend") -> str:
    """Read backend/{user_id}/ JSON files and produce a self-contained HTML file."""
    user_dir = os.path.join(backend_dir, str(user_id))

    profile = _load_profile(user_dir)
    events, flat_prefs = _load_app_events(user_dir)

    # Load calendar modification stream (Step 11c output). Optional — older
    # backends may not have it.
    calendar_mods: list[dict] = []
    calendar_path = os.path.join(user_dir, "calendar.json")
    if os.path.exists(calendar_path):
        try:
            with open(calendar_path, "r", encoding="utf-8") as f:
                cal_doc = json.load(f)
            if isinstance(cal_doc, dict):
                mods = cal_doc.get("modifications", [])
                if isinstance(mods, list):
                    calendar_mods = mods
        except (ValueError, OSError):
            pass

    now_str = datetime.now(tz=timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    # Serialize events + calendar mods for JS
    events_json = json.dumps(events)
    profile_json = json.dumps(profile) if profile else "null"
    calendar_json = json.dumps(calendar_mods)

    # Test-sample annotation: load benchmark/{uid}/queries.csv (when present).
    # Each test sample becomes a standalone timeline card at its own ts +
    # nearest preceding event's geo location, with a distinct background color.
    test_samples = _load_test_samples(user_id)
    test_samples_json = json.dumps(test_samples)

    # Counts
    n_events = len(events)
    n_prefs = len(flat_prefs)
    n_unique = len(set(r.get("persona_item", "") for r in flat_prefs))
    n_stereo = sum(1 for r in flat_prefs if r.get("stereotype_mark") == "stereotypical")
    n_anti = sum(1 for r in flat_prefs if r.get("stereotype_mark") == "anti-stereotypical")
    # Pref-instance test counts (one per supporting event)
    # R8: no more test/train split in data-gen output. Count short-term
    # horizons instead — those are the actionable eval-facing signal.
    n_short_term_instances = sum(1 for r in flat_prefs if r.get("time_horizon") == "short_term")
    n_ad_events = sum(1 for e in events if e.get("is_ad"))
    short_term_canonicals = {r.get("persona_item", "") for r in flat_prefs if r.get("time_horizon") == "short_term"}
    short_term_canonicals.discard("")
    n_short_term_canonicals = len(short_term_canonicals)
    per_app_counts = {}
    for app in APPS:
        per_app_counts[app] = sum(1 for e in events if e.get("_app") == app)

    # Event counts split by source_interaction_type
    _TYPES = ("explicit_positive", "explicit_negative", "implicit_positive", "implicit_negative")
    event_type_counts = {t: 0 for t in _TYPES}
    for e in events:
        t = e.get("source_interaction_type", "")
        if t in event_type_counts:
            event_type_counts[t] += 1

    # Canonical-preference counts split by their dominant interaction type.
    # For each unique persona_item, classify by priority:
    #   explicit_negative > explicit_positive > implicit_positive > implicit_negative.
    # (In practice surviving negatives are all promoted to explicit_negative,
    # so the implicit_negative canonical count will usually be 0.)
    pref_types_by_canonical: dict[str, set[str]] = {}
    for r in flat_prefs:
        pi = r.get("persona_item", "")
        if not pi:
            continue
        pref_types_by_canonical.setdefault(pi, set()).add(r.get("source_interaction_type", ""))
    canonical_type_counts = {t: 0 for t in _TYPES}
    for types in pref_types_by_canonical.values():
        if "explicit_negative" in types:
            canonical_type_counts["explicit_negative"] += 1
        elif "explicit_positive" in types:
            canonical_type_counts["explicit_positive"] += 1
        elif "implicit_positive" in types:
            canonical_type_counts["implicit_positive"] += 1
        elif "implicit_negative" in types:
            canonical_type_counts["implicit_negative"] += 1

    # Time period: earliest → latest event's formatted timestamps.
    event_ts = [int(e.get("source_timestamp") or 0) for e in events if e.get("source_timestamp")]
    if event_ts:
        first_ts, last_ts = min(event_ts), max(event_ts)
        first_fmt = utils.unix_to_formatted(first_ts) if hasattr(utils, "unix_to_formatted") else datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%H:%M, %m/%d/%Y")
        last_fmt = utils.unix_to_formatted(last_ts) if hasattr(utils, "unix_to_formatted") else datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%H:%M, %m/%d/%Y")
        span_days = (last_ts - first_ts) / 86400.0
        time_period = f"{first_fmt} → {last_fmt} ({span_days:.1f} days)"
    else:
        time_period = "—"

    # Number of distinct preference categories
    n_categories = len({r.get("category", "") for r in flat_prefs if r.get("category")})

    # Unique geo locations across all events (ordered by frequency desc).
    location_counts: dict[tuple[str, str, str], int] = {}
    for e in events:
        loc = e.get("event_location") or {}
        city = (loc.get("city") or "").strip()
        if not city:
            continue
        key = (city, (loc.get("region") or "").strip(), (loc.get("country") or "").strip())
        location_counts[key] = location_counts.get(key, 0) + 1
    if location_counts:
        location_parts = []
        for (city, region, country), cnt in sorted(
            location_counts.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            label = city
            if region:
                label += f", {region}"
            if country and country not in ("USA", "US"):
                label += f", {country}"
            location_parts.append(f"<span>{label} ({cnt})</span>")
        locations_html = "".join(location_parts)
    else:
        locations_html = '<span>—</span>'

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Persona — User {user_id}</title>
<style>
  :root {{
    --bg: #F7F7F5;
    --bg-card: #FFFFFF;
    --text: #1D1D1F;
    --text-secondary: #86868B;
    --text-tertiary: #AEAEB2;
    --border: #E5E5EA;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-hover: 0 2px 8px rgba(0,0,0,0.07);
    --font: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }}
  .container {{ max-width: 860px; margin: 0 auto; padding: 56px 24px; }}

  .header {{ margin-bottom: 40px; }}
  .header h1 {{ font-size: 28px; font-weight: 600; letter-spacing: -0.4px; margin-bottom: 6px; color: var(--text); }}
  .header .meta {{ color: var(--text-secondary); font-size: 13px; display: flex; flex-wrap: wrap; gap: 6px 18px; }}

  .profile-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 40px; box-shadow: var(--shadow); }}
  .profile-card h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 10px; letter-spacing: -0.2px; }}
  .profile-card .bio {{ font-size: 14px; line-height: 1.65; margin-bottom: 14px; color: var(--text); }}
  .profile-card .details {{ font-size: 12px; color: var(--text-secondary); }}
  .profile-card .details span {{ margin-right: 14px; }}
  .profile-card .big-five {{ display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }}
  .profile-card .b5-item {{ font-size: 11px; padding: 3px 10px; border-radius: 20px; background: #F2F2F7; color: var(--text-secondary); }}
  .profile-card .mbti {{ margin-top: 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}

  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 16px; font-weight: 600; letter-spacing: -0.2px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--text); }}

  .event-grid {{ display: flex; flex-direction: column; gap: 12px; }}
  .event-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; box-shadow: var(--shadow); transition: box-shadow 0.15s ease; border-left: 3px solid var(--border); }}
  .event-card:hover {{ box-shadow: var(--shadow-hover); }}
  .event-card.app-Instagram {{ border-left-color: #C13584; }}
  .event-card.app-Facebook {{ border-left-color: #4A6FA5; }}
  .event-card.app-Threads {{ border-left-color: #636366; }}
  .event-card.app-Chatbot {{ border-left-color: #C8956C; }}
  .event-card.implicit-negative {{ background: #F0F0F0; border-left-color: #B0B0B0; opacity: 0.65; filter: grayscale(100%); }}
  .event-card.implicit-negative .event-meta {{ color: #999; }}
  .event-card.implicit-negative .event-header {{ border-bottom-color: #E0E0E0; }}
  .event-card.implicit-negative .hashtags {{ color: #888; }}
  .event-card.implicit-negative .badge {{ background: #E0E0E0 !important; color: #777 !important; }}
  .event-card.implicit-negative .pref-item {{ background: #E8E8E8; border-color: #D0D0D0; }}
  .event-card.implicit-negative .pref-item .item-text {{ color: #666; }}
  .event-card.implicit-negative .conf-inline {{ color: #999; }}

  .event-header {{ margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #F2F2F7; }}
  .event-header .event-meta {{ font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; }}
  .event-header .hashtags {{ font-size: 12px; color: var(--text); margin-top: 4px; line-height: 1.5; }}

  .pref-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .content-block + .pref-list {{ margin-top: 14px; }}
  .pref-list-label {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #6B7280; margin-bottom: 2px; }}
  .pref-item {{ padding: 10px 14px; border-radius: 8px; background: #FAFAFA; border: 1px solid #F2F2F7; }}
  .pref-item .item-text {{ font-size: 13px; font-weight: 500; line-height: 1.45; color: var(--text); margin-bottom: 4px; }}
  .pref-item .pref-meta {{ font-size: 10px; color: var(--text-secondary); }}

  .update-history {{ margin-top: 6px; padding-left: 10px; border-left: 2px solid #E8E8ED; }}
  .update-entry {{ font-size: 10px; color: var(--text-secondary); margin-bottom: 2px; }}
  .update-entry .ut-type {{ font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ut-reinforced {{ color: #2D6A4F; }}
  .ut-deepened {{ color: #1D4ED8; }}
  .ut-branched {{ color: #7C3AED; }}
  .ut-shifted {{ color: #B45309; }}
  .ut-intensified {{ color: #047857; }}
  .ut-contradicted {{ color: #B04050; }}
  .stance-res {{ font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; padding: 1px 6px; border-radius: 3px; margin-left: 4px; }}
  .stance-res.stance-passed {{ background: #FEE2E2; color: #7F1D1D; }}
  .stance-res.stance-ambivalence {{ background: #FFEDD5; color: #9A3412; }}
  .stance-res.stance-suppressed {{ background: #F3F4F6; color: #6B7280; text-decoration: line-through; }}
  .ut-ambivalent {{ color: #9A3412; }}
  .ut-faded {{ color: var(--text-tertiary); }}
  .ut-expanded {{ color: #1D4ED8; }}

  .conf-inline {{ font-size: 10px; color: var(--text-tertiary); font-variant-numeric: tabular-nums; }}
  .conf-inline span {{ margin-right: 10px; }}

  .badge {{ display: inline-block; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 4px; margin-right: 3px; letter-spacing: 0.1px; }}
  .badge.category {{ background: #F2F2F7; color: #636366; }}
  .badge.similar {{ background: #F2F2F7; color: #48854A; }}
  .badge.contradictory {{ background: #F2F2F7; color: #B04050; }}
  .badge.none {{ display: none; }}
  .badge.stereotypical {{ background: #FFF8E1; color: #8B6914; }}
  .badge.anti-stereotypical {{ background: #EEF2FF; color: #4A5DA8; }}
  .badge.train {{ background: #F2F2F7; color: var(--text-secondary); }}
  .badge.distractor {{ background: #FEF2F2; color: #9B2C2C; }}
  /* Standalone test-sample card — distinct gold-amber background so it stands
     out from regular events without needing inline annotations. */
  .event-card.test-sample-card {{
    background: #FFFBEB !important;
    border-left: 3px solid #d4af37 !important;
  }}
  .event-card.test-sample-card .event-header .event-meta code {{
    font-family: inherit;
    background: #FFF; padding: 1px 6px; border-radius: 3px; font-size: 11px;
    color: #7B5C00;
  }}
  .test-sample-query {{
    font-size: 14px; line-height: 1.45; color: var(--text);
    padding: 12px 14px; background: #fff; border-radius: 5px; margin: 8px 0;
    border: 1px solid #FBE9A1;
  }}
  .test-sample-meta {{
    font-size: 11px; color: var(--text-secondary); padding: 4px 6px;
  }}
  /* Rich-info sections inside a test card */
  .ts-section {{
    margin: 8px 0 4px 0; padding: 6px 10px; background: rgba(255,255,255,0.7);
    border-radius: 4px; border: 1px solid rgba(212,175,55,0.25);
  }}
  .ts-section-warn {{ background: #FEF2F2; border-color: #FCA5A5; }}
  .ts-section.ts-rubric-bar {{ background: #FFF8E1; }}
  .ts-label {{ font-weight: 600; font-size: 11px; color: #7B5C00; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 4px; }}
  .ts-section-warn .ts-label {{ color: #B91C1C; }}
  .ts-sublabel {{ font-size: 10px; font-weight: 500; color: var(--text-secondary); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ts-body {{ font-size: 12px; color: var(--text); line-height: 1.45; }}
  .ts-body.ts-mono {{ font-size: 11px; white-space: pre-wrap; color: var(--text-secondary); }}
  .ts-list {{ margin: 4px 0 0 0; padding-left: 18px; font-size: 12px; line-height: 1.5; }}
  .ts-list.ts-mono {{ font-size: 11px; color: var(--text-secondary); }}
  /* Inline <code> in test cards inherits the page font — keep visually distinct
     via subtle background + smaller size + grey, NOT a different font family. */
  .test-sample-card code,
  .ts-section code,
  .test-sample-meta code,
  .test-sample-card .event-meta code {{
    font-family: inherit; font-size: 0.92em; padding: 1px 5px;
    background: rgba(255,255,255,0.65); border-radius: 3px; color: var(--text-secondary);
  }}
  .ts-list li {{ margin: 3px 0; }}
  .ts-origin {{ display: inline-block; font-size: 9px; padding: 1px 5px; border-radius: 3px; background: #E5E7EB; color: #374151; margin: 0 2px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ts-origin-held_out {{ background: #D4AF37; color: #fff; }}
  .ts-origin-future_positive {{ background: #BFDBFE; color: #1E40AF; }}
  .ts-origin-past_positive {{ background: #BBF7D0; color: #166534; }}
  .ts-origin-negative {{ background: #FECACA; color: #7F1D1D; }}
  .ts-origin-irrelevant, .ts-origin-random, .ts-origin-filler, .ts-origin-filler_lowsim {{ background: #F3F4F6; color: #6B7280; }}
  .ts-origin-match {{ background: #D4AF37; color: #fff; }}
  .ts-origin-carve_out {{ background: #FECACA; color: #7F1D1D; }}
  .ts-target {{ font-size: 10px; color: #B45309; font-weight: 700; }}
  .badge.platform {{ font-weight: 600; font-size: 11px; padding: 2px 10px; }}
  .badge.platform.p-Instagram {{ background: #C13584; color: #fff; }}
  .badge.platform.p-Facebook {{ background: #4A6FA5; color: #fff; }}
  .badge.platform.p-Threads {{ background: #8E8E93; color: #fff; }}
  .badge.platform.p-Chatbot {{ background: #C8956C; color: #fff; }}
  .badge.action {{ background: #E8E8ED; color: #48484A; font-weight: 500; }}
  .badge.hidden-persona {{ background: #EDE9FE; color: #6D28D9; font-weight: 500; }}
  .badge.short-term {{ background: #EFE1FF; color: #7C3AED; font-weight: 600; }}
  .stop-condition {{ margin-top: 4px; font-size: 10px; color: #7C3AED; opacity: 0.85; font-style: italic; }}
  .stop-condition .sc-type {{ text-transform: uppercase; font-weight: 700; letter-spacing: 0.4px; margin-right: 6px; }}
  .badge.sponsored {{ background: #FFF7ED; color: #9A3412; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; border: 1px solid #FED7AA; }}
  .event-location {{ font-size: 11px; color: var(--text-tertiary); }}
  .calendar-card {{ background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 12px; color: #14532D; box-shadow: 0 1px 2px rgba(22,163,74,0.08); }}
  .calendar-card .cal-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-weight: 600; }}
  .calendar-card .cal-action {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; padding: 2px 8px; border-radius: 4px; color: #fff; background: #16A34A; font-weight: 700; }}
  .calendar-card .cal-action.removed {{ background: #DC2626; }}
  .calendar-card .cal-action.updated {{ background: #CA8A04; }}
  .calendar-card .cal-meta {{ font-size: 11px; opacity: 0.75; margin-top: 2px; }}
  .ad-meta {{ margin-top: 6px; padding: 6px 10px; background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 6px; font-size: 11px; color: #7C2D12; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
  .ad-meta .ad-sponsor {{ font-weight: 600; }}
  .ad-meta .ad-cta {{ background: #9A3412; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
  .event-card.is-ad .content-block {{ border-color: #FED7AA; background: #FFFBF5; }}

  .hidden-section {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 40px; box-shadow: var(--shadow); }}
  .hidden-section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 14px; letter-spacing: -0.2px; }}
  .hidden-summary {{ font-size: 13px; line-height: 1.7; color: var(--text); margin-bottom: 16px; padding: 12px 16px; background: #FAFAFA; border-radius: 8px; border-left: 3px solid #6D28D9; }}
  .hp-card {{ padding: 12px 16px; margin-bottom: 10px; border-radius: 8px; background: #FAFAFA; border: 1px solid #F2F2F7; }}
  .hp-card .hp-label {{ font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 2px; }}
  .hp-card .hp-type {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; color: #6D28D9; margin-bottom: 4px; }}
  .hp-card .hp-desc {{ font-size: 12px; color: var(--text); line-height: 1.6; margin-bottom: 6px; }}
  .hp-card .hp-meta {{ font-size: 10px; color: var(--text-secondary); }}
  .hp-card .hp-meta span {{ margin-right: 12px; }}
  .hp-card .hp-tags {{ font-size: 11px; color: var(--text-secondary); margin-top: 4px; }}
  .hp-card .hp-motivation {{ font-size: 11px; color: #6D28D9; margin-top: 4px; font-style: italic; }}
  .badge.interaction-type {{ font-weight: 600; padding: 2px 10px; }}
  .badge.interaction-type.explicit_positive {{ background: #D1FAE5; color: #065F46; }}
  .badge.interaction-type.implicit_positive {{ background: #EDF5E1; color: #3F6212; }}
  .badge.interaction-type.explicit_negative {{ background: #FEE2E2; color: #991B1B; }}
  .badge.interaction-type.implicit_negative {{ background: #FEF3C7; color: #92400E; }}

  .user-message {{ margin-top: 8px; padding: 8px 12px; background: #F2F2F7; border-left: 2px solid var(--text-tertiary); border-radius: 4px; font-size: 12px; color: var(--text); font-style: italic; }}

  /* Synthetic content (step 13b) rendering */
  .content-block {{ margin-top: 10px; padding: 12px 14px; background: #FAFAFC; border: 1px solid #ECECF1; border-radius: 8px; }}
  .content-block .c-type {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #6B7280; margin-bottom: 6px; }}
  .content-block .c-type.c-text {{ color: #4A5DA8; }}
  .content-block .c-type.c-image {{ color: #9B3068; }}
  .content-block .c-type.c-short_video {{ color: #7C3AED; }}
  .content-block .c-title {{ font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
  .content-block .c-caption {{ font-size: 12px; color: var(--text); margin-bottom: 6px; line-height: 1.5; }}
  .content-block .c-desc {{ font-size: 12px; color: var(--text-secondary); line-height: 1.55; margin-bottom: 8px; font-style: italic; }}
  .content-block .c-text-body {{ font-size: 13px; color: var(--text); line-height: 1.65; white-space: pre-wrap; }}
  .content-block details {{ margin-top: 6px; }}
  .content-block details summary {{ font-size: 11px; color: var(--text-secondary); cursor: pointer; padding: 2px 0; user-select: none; }}
  .content-block details summary:hover {{ color: var(--text); }}
  .content-block details[open] summary {{ color: var(--text); }}
  .content-block .c-parts, .content-block .c-frames {{ margin-top: 6px; padding-left: 2px; }}
  .content-block .c-part, .content-block .c-frame {{ font-size: 11px; color: var(--text); padding: 4px 8px; margin-bottom: 2px; background: #F2F2F7; border-radius: 4px; line-height: 1.45; }}
  .content-block .c-part .region, .content-block .c-frame .ts {{ font-weight: 600; color: #636366; margin-right: 6px; font-variant-numeric: tabular-nums; }}
  .content-block .c-transcript {{ font-size: 11px; color: var(--text); padding: 6px 10px; margin-top: 6px; background: #F2F2F7; border-radius: 4px; line-height: 1.5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }}
  .content-block .c-meta {{ margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .content-block .c-meta-chip {{ font-size: 10px; padding: 2px 8px; border-radius: 4px; background: #ECECF1; color: #636366; font-variant-numeric: tabular-nums; }}
  .event-card.implicit-negative .content-block {{ background: #ECECEC; border-color: #D8D8D8; }}

  .chat-thread {{ margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }}
  .chat-bubble {{ max-width: 85%; padding: 10px 14px; border-radius: 14px; font-size: 12px; line-height: 1.6; word-wrap: break-word; }}
  .chat-bubble.user-bubble {{ align-self: flex-end; background: #1B72E8; color: #fff; border-bottom-right-radius: 4px; }}
  .chat-bubble.assistant-bubble {{ align-self: flex-start; background: #E4E6EB; color: var(--text); border-bottom-left-radius: 4px; }}
  .chat-role {{ font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
  .chat-bubble.user-bubble .chat-role {{ color: rgba(255,255,255,0.55); }}
  .chat-bubble.assistant-bubble .chat-role {{ color: var(--text-tertiary); }}
  .chat-conv-label {{ font-size: 10px; color: var(--text-tertiary); margin-top: 6px; margin-bottom: 2px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.3px; }}

  .empty {{ text-align: center; padding: 40px; color: var(--text-secondary); font-size: 13px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>User {user_id}</h1>
    <div class="meta">
      <span>{n_events} events</span>
      <span>{n_prefs} pref instances ({n_short_term_instances} short-term)</span>
      <span>{n_unique} canonicals ({n_short_term_canonicals} short-term)</span>
      <span>{n_ad_events} ad events</span>
      <span>{n_categories} categories</span>
      <span>{n_stereo} stereo</span>
      <span>{n_anti} anti-stereo</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span title="Events by source_interaction_type">Events:</span>
      <span>expl+ {event_type_counts["explicit_positive"]}</span>
      <span>expl− {event_type_counts["explicit_negative"]}</span>
      <span>impl+ {event_type_counts["implicit_positive"]}</span>
      <span>impl− {event_type_counts["implicit_negative"]}</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span title="Canonical preferences by dominant supporting interaction type">Canonicals:</span>
      <span>expl+ {canonical_type_counts["explicit_positive"]}</span>
      <span>expl− {canonical_type_counts["explicit_negative"]}</span>
      <span>impl+ {canonical_type_counts["implicit_positive"]}</span>
      <span>impl− {canonical_type_counts["implicit_negative"]}</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span>Period: {time_period}</span>
      <span>IG: {per_app_counts.get("Instagram", 0)}</span>
      <span>FB: {per_app_counts.get("Facebook", 0)}</span>
      <span>TH: {per_app_counts.get("Threads", 0)}</span>
      <span>AI: {per_app_counts.get("Chatbot", 0)}</span>
      <span>Generated {now_str}</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span title="Geo locations across all events">Locations:</span>
      {locations_html}
    </div>
  </div>

  <div id="profile-section"></div>
  <div id="hidden-personas-section"></div>

  <div class="section">
    <div class="section-title">Interaction Events (earliest &rarr; latest)</div>
    <div id="timeline-section"></div>
  </div>

</div>

<script>
const eventsData = {events_json};
const profileData = {profile_json};
const calendarMods = {calendar_json};
// Test-sample annotation (Phase A4 v2): each test sample becomes a standalone
// timeline card at its own ts + nearest-preceding event's location, with a
// distinct background color. No annotations are merged into regular event cards.
const testSamples = {test_samples_json};

// Label -> motivation lookup for hidden persona badge tooltips.
const hpMotivation = {{}};
if (profileData && profileData.hidden_personas) {{
  profileData.hidden_personas.forEach(hp => {{
    if (hp && hp.label) {{
      hpMotivation[hp.label] = hp.inferred_motivation || hp.description || '';
    }}
  }});
}}

// -- Profile card --
const ps = document.getElementById('profile-section');
if (profileData) {{
  const b5 = profileData.big_five || {{}};
  const b5Html = Object.entries(b5).map(([k,v]) => `<span class="b5-item">${{k}}: ${{v}}</span>`).join('');

  // MBTI block — reuse the Big Five chip style. One chip per dimension,
  // formatted as "axis: dominant letter". The MBTI type itself is just the
  // concatenation of these four letters, so we don't repeat it as a chip.
  let mbtiHtml = '';
  const mbti = profileData.mbti;
  if (mbti && mbti.dimensions) {{
    const dimOrder = ['E_I', 'S_N', 'T_F', 'J_P'];
    const dimChips = dimOrder.map(key => {{
      const d = mbti.dimensions[key];
      if (!d) return '';
      const [letterA, letterB] = key.split('_');
      const pA = Number(d[letterA] || 0);
      const pB = Number(d[letterB] || 0);
      const dominant = pA >= pB ? letterA : letterB;
      const pct = Math.round((pA >= pB ? pA : pB) * 100);
      const axis = `${{letterA}}/${{letterB}}`;
      const reason = (d.reason || '').replace(/"/g, '&quot;');
      return `<span class="b5-item" title="${{reason}}">${{axis}}: ${{dominant}} ${{pct}}%</span>`;
    }}).join('');
    if (dimChips) {{
      mbtiHtml = `<div class="mbti">${{dimChips}}</div>`;
    }}
  }}

  ps.innerHTML = `
    <div class="profile-card">
      <h2>${{profileData.name || ''}}</h2>
      <div class="bio">${{profileData.bio || ''}}</div>
      <div class="details">
        <span>${{profileData.gender || ''}}</span>
        <span>${{profileData.race_ethnicity || ''}}</span>
        <span>${{profileData.career || ''}}</span>
        <span>${{profileData.education || ''}}</span>
      </div>
      <div class="big-five">${{b5Html}}</div>
      ${{mbtiHtml}}
    </div>
  `;
}}

// -- Hidden Personas section --
const hps = document.getElementById('hidden-personas-section');
if (profileData && profileData.hidden_personas && profileData.hidden_personas.length > 0) {{
  let html = '<div class="hidden-section"><h2>Hidden Personas</h2>';

  // Summary paragraph
  if (profileData.hidden_persona_summary) {{
    html += `<div class="hidden-summary">${{profileData.hidden_persona_summary}}</div>`;
  }}

  // Individual hidden persona cards
  profileData.hidden_personas.forEach(hp => {{
    const tags = (hp.evidence_hashtags || []).join('  ');
    const ib = hp.interaction_breakdown || {{}};
    const ibStr = Object.entries(ib).map(([k,v]) => `${{k.replace(/_/g,' ')}}: ${{v}}`).join(' · ');
    const appDist = hp.app_distribution || {{}};
    const appStr = Object.entries(appDist).map(([k,v]) => `${{k}}: ${{v}}`).join(' · ');
    html += `
      <div class="hp-card">
        <div class="hp-type">${{hp.type || ''}}</div>
        <div class="hp-label">${{hp.label || ''}}</div>
        <div class="hp-desc">${{hp.description || ''}}</div>
        <div class="hp-meta">
          <span>${{hp.evidence_rows || 0}} rows (${{((hp.evidence_row_fraction || 0) * 100).toFixed(1)}}%)</span>
          <span>privacy: ${{((hp.privacy_ratio || 0) * 100).toFixed(0)}}%</span>
          <span>${{hp.temporal_spread_days || 0}} days</span>
          ${{appStr ? `<span>${{appStr}}</span>` : ''}}
        </div>
        <div class="hp-tags">${{tags}}</div>
        ${{hp.inferred_motivation ? `<div class="hp-motivation">"${{hp.inferred_motivation}}"</div>` : ''}}
      </div>
    `;
  }});

  html += '</div>';
  hps.innerHTML = html;
}}

// -- Render synthetic content (step 13b) --
// Produces an HTML block describing what the user saw on screen: the text,
// image, or short video. Returns empty string when the event has no content
// (Chatbot events and implicit_negative stubs).
function escapeHtml(s) {{
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}}

function renderContentMeta(meta) {{
  if (!meta || typeof meta !== 'object') return '';
  const chips = Object.entries(meta)
    .filter(([k, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => {{
      const val = typeof v === 'object' ? JSON.stringify(v) : String(v);
      return `<span class="c-meta-chip"><b>${{escapeHtml(k)}}</b> ${{escapeHtml(val)}}</span>`;
    }})
    .join('');
  return chips ? `<div class="c-meta">${{chips}}</div>` : '';
}}

function renderAdMeta(content) {{
  if (!content || typeof content !== 'object') return '';
  const md = content.ad_metadata;
  if (!md || typeof md !== 'object') return '';
  const sponsor = md.sponsor_name ? `<span class="ad-sponsor">${{escapeHtml(md.sponsor_name)}}</span>` : '';
  const cat = md.ad_category ? `<span>${{escapeHtml(md.ad_category.replace(/_/g, ' '))}}</span>` : '';
  const cta = md.cta_label ? `<span class="ad-cta">${{escapeHtml(md.cta_label)}}</span>` : '';
  const dest = md.cta_destination_kind ? `<span style="opacity:0.7">${{escapeHtml(md.cta_destination_kind.replace(/_/g, ' '))}}</span>` : '';
  return `<div class="ad-meta">${{sponsor}}${{cat}}${{cta}}${{dest}}</div>`;
}}

function renderContent(ev) {{
  const ctype = ev.content_type;
  const content = ev.content;
  if (!ctype || !content || typeof content !== 'object') return '';

  const typeLabel = ctype.replace(/_/g, ' ');
  const header = `<div class="c-type c-${{ctype}}">${{typeLabel}}</div>`;
  const adMetaHtml = renderAdMeta(content);

  if (ctype === 'text') {{
    return `<div class="content-block">${{header}}<div class="c-text-body">${{escapeHtml(content.text || '')}}</div>${{adMetaHtml}}</div>`;
  }}

  if (ctype === 'image') {{
    const caption = content.caption ? `<div class="c-caption">${{escapeHtml(content.caption)}}</div>` : '';
    const desc = content.overall_description ? `<div class="c-desc">${{escapeHtml(content.overall_description)}}</div>` : '';
    const parts = (content.parts || []).map(p =>
      `<div class="c-part"><span class="region">${{escapeHtml(p.region || '')}}</span>${{escapeHtml(p.description || '')}}</div>`
    ).join('');
    const partsBlock = parts
      ? `<details><summary>parts (${{content.parts.length}})</summary><div class="c-parts">${{parts}}</div></details>`
      : '';
    const metaBlock = renderContentMeta(content.metadata);
    const metaWrapped = metaBlock
      ? `<details><summary>metadata</summary>${{metaBlock}}</details>`
      : '';
    return `<div class="content-block">${{header}}${{caption}}${{desc}}${{partsBlock}}${{metaWrapped}}${{adMetaHtml}}</div>`;
  }}

  if (ctype === 'short_video') {{
    const title = content.title ? `<div class="c-title">${{escapeHtml(content.title)}}</div>` : '';
    const caption = content.caption ? `<div class="c-caption">${{escapeHtml(content.caption)}}</div>` : '';
    const desc = content.overall_description ? `<div class="c-desc">${{escapeHtml(content.overall_description)}}</div>` : '';
    const frames = (content.key_frames || []).map(f => {{
      const ts = typeof f.timestamp_s === 'number' ? f.timestamp_s.toFixed(1) + 's' : String(f.timestamp_s || '');
      return `<div class="c-frame"><span class="ts">${{escapeHtml(ts)}}</span>${{escapeHtml(f.description || '')}}</div>`;
    }}).join('');
    const framesBlock = frames
      ? `<details open><summary>key frames (${{content.key_frames.length}})</summary><div class="c-frames">${{frames}}</div></details>`
      : '';
    const transcript = content.audio_transcript
      ? `<details><summary>audio transcript</summary><div class="c-transcript">${{escapeHtml(content.audio_transcript)}}</div></details>`
      : '';
    const metaBlock = renderContentMeta(content.metadata);
    const metaWrapped = metaBlock
      ? `<details><summary>metadata</summary>${{metaBlock}}</details>`
      : '';
    return `<div class="content-block">${{header}}${{title}}${{caption}}${{desc}}${{framesBlock}}${{transcript}}${{metaWrapped}}${{adMetaHtml}}</div>`;
  }}

  return '';
}}

// -- Render update history --
// Parse "HH:MM, MM/DD/YYYY" -> unix seconds (UTC). Used to filter
// update_history entries that lack a numeric `timestamp` field.
function _parseFormattedTs(s) {{
  if (!s || typeof s !== 'string') return 0;
  const m = s.match(/^(\d{{1,2}}):(\d{{2}}),\s*(\d{{1,2}})\/(\d{{1,2}})\/(\d{{4}})$/);
  if (!m) return 0;
  const [_, hh, mm, mo, dd, yyyy] = m;
  return Math.floor(Date.UTC(+yyyy, +mo - 1, +dd, +hh, +mm, 0) / 1000);
}}

function renderUpdateHistory(history, asOfTs) {{
  if (!history || !history.length) return '';
  // Filter rules for what shows up on a per-event render:
  //   1. Cross-ref entries (entries with a `resolution` field —
  //      "suppressed_weak_minority", "different_granularity", etc.) are
  //      GLOBAL findings about how this canonical relates to other
  //      canonicals across the whole persona. They don't represent events
  //      that happened at any specific time. We hide them from per-event
  //      renders entirely; they belong on a canonical-preference detail view.
  //   2. Temporal entries (`reinforced`, `new`, `faded`) ARE events. Show
  //      only those whose timestamp <= asOfTs (parse formatted_timestamp
  //      when no numeric ts is set).
  let visible = history.filter(h => !h.resolution);
  if (typeof asOfTs === 'number' && asOfTs > 0) {{
    visible = visible.filter(h => {{
      const ht = h.timestamp || h.ts || _parseFormattedTs(h.formatted_timestamp);
      return !ht || ht <= asOfTs;
    }});
  }}
  if (!visible.length) return '';
  const entries = visible.map(h => {{
    const cls = 'ut-' + (h.update_type || 'expanded');
    let text = `<span class="ut-type ${{cls}}">${{h.update_type}}</span>`;
    if (h.preference) text += ` ${{h.preference}}`;
    if (h.description) text += ` — ${{h.description}}`;
    if (h.formatted_timestamp) text += ` <span style="opacity:0.6">(${{h.formatted_timestamp}})</span>`;
    if (h.source_app) text += ` <span class="badge platform p-${{h.source_app}}" style="font-size:9px;padding:1px 6px;">${{h.source_app}}</span>`;
    if (h.total_occurrences) text += ` <span style="opacity:0.6">[occ ${{h.occurrence}}/${{h.total_occurrences}}]</span>`;
    if (h.resolution) {{
      const resCls = h.resolution === 'stance_shift_with_precedent' ? 'stance-passed'
                   : h.resolution === 'concurrent_ambivalence'     ? 'stance-ambivalence'
                   : h.resolution === 'different_granularity'      ? 'stance-ambivalence'
                   : 'stance-suppressed';
      text += ` <span class="stance-res ${{resCls}}">${{h.resolution.replace(/_/g, ' ')}}</span>`;
      if (typeof h.prior_corroboration_count === 'number') {{
        text += ` <span style="opacity:0.6">prior ${{h.prior_corroboration_count}}/${{h.required_precedent}}</span>`;
      }}
    }}
    return `<div class="update-entry">${{text}}</div>`;
  }}).join('');
  return `<div class="update-history">${{entries}}</div>`;
}}

// -- Calendar modification card renderer --
function renderCalendarMod(mod) {{
  const action = mod.action || '';
  const actionCls = `cal-action ${{action}}`;
  const actionLabel = action.toUpperCase();
  let title = '';
  let locChip = '';
  let extraLine = '';
  if (action === 'added' && mod.entry) {{
    const e = mod.entry;
    title = escapeHtml(e.title || '(untitled)');
    if (e.location && e.location.city) {{
      locChip = ` 📍 ${{escapeHtml(e.location.city)}}`;
    }}
    const start = e.start_ts ? new Date(e.start_ts * 1000).toISOString().slice(0, 16).replace('T', ' ') : '';
    const typeLbl = e.type ? `<span style="opacity:0.7">[${{escapeHtml(e.type)}}]</span>` : '';
    extraLine = `<div class="cal-meta">scheduled for ${{start}} ${{typeLbl}} ${{e.is_preference_driven ? '· preference-linked' : '· unrelated'}}</div>`;
  }} else if (action === 'updated') {{
    title = `update to ${{escapeHtml(mod.entry_id || '?')}}`;
    const diff = mod.diff || {{}};
    const fields = Object.keys(diff).join(', ');
    extraLine = `<div class="cal-meta">changed: ${{escapeHtml(fields)}}</div>`;
  }} else if (action === 'removed') {{
    title = `removed ${{escapeHtml(mod.entry_id || '?')}}`;
    if (mod.removal_reason) {{
      extraLine = `<div class="cal-meta">${{escapeHtml(mod.removal_reason)}}</div>`;
    }}
  }}
  const ts = mod.formatted_timestamp ? escapeHtml(mod.formatted_timestamp) : '';
  return `
    <div class="calendar-card">
      <div class="cal-head">
        <span class="${{actionCls}}">📅 Calendar ${{actionLabel}}</span>
        <span>${{title}}${{locChip}}</span>
      </div>
      <div style="opacity:0.6;font-size:11px;">${{ts}}</div>
      ${{extraLine}}
    </div>
  `;
}}

// -- Chronological timeline of interaction events + calendar modifications +
//    test-sample cards (Phase A4 v2). Test samples sit at their own ts in
//    the same timeline, with a distinct background color, no annotations on
//    regular event cards. --
const timeline = document.getElementById('timeline-section');

if (eventsData.length === 0) {{
  timeline.innerHTML = '<div class="empty">No interaction events available.</div>';
}} else {{
  const grid = document.createElement('div');
  grid.className = 'event-grid';

  // Pre-sort regular events by ts so test-sample location-lookup is O(log n).
  const sortedEvents = eventsData.slice().sort((a, b) => (a.source_timestamp || 0) - (b.source_timestamp || 0));
  // Find nearest preceding event's location for a given ts.
  function _locationAtTs(ts) {{
    let lo = 0, hi = sortedEvents.length - 1, best = null;
    while (lo <= hi) {{
      const mid = (lo + hi) >> 1;
      const evTs = sortedEvents[mid].source_timestamp || 0;
      if (evTs <= ts) {{ best = sortedEvents[mid]; lo = mid + 1; }}
      else hi = mid - 1;
    }}
    return (best && best.event_location) ? best.event_location : null;
  }}

  // Build merged timeline: events + calendar mods + test samples, sorted by ts.
  const timelineItems = [];
  eventsData.forEach((ev, i) => timelineItems.push({{ kind: 'event', ts: ev.source_timestamp || 0, data: ev, eventIdx: i }}));
  (calendarMods || []).forEach(mod => timelineItems.push({{ kind: 'cal', ts: mod.ts || 0, data: mod }}));
  (testSamples || []).forEach(t => timelineItems.push({{ kind: 'test', ts: t.ts || 0, data: t, location: _locationAtTs(t.ts || 0) }}));
  timelineItems.sort((a, b) => (a.ts || 0) - (b.ts || 0));

  timelineItems.forEach(item => {{
    if (item.kind === 'cal') {{
      const div = document.createElement('div');
      div.innerHTML = renderCalendarMod(item.data);
      grid.appendChild(div.firstElementChild);
      return;
    }}
    if (item.kind === 'test') {{
      // Standalone test-sample card — distinct background, same shape as
      // a regular event so the timeline reads naturally. Renders every
      // ground-truth field the extractor populated.
      const t = item.data;
      const loc = item.location;
      let locText = '';
      if (loc && typeof loc === 'object') {{
        const parts = [loc.city, loc.region].filter(x => x).map(escapeHtml);
        if (parts.length > 0) locText = `<span class="event-location">📍 ${{parts.join(', ')}}</span>`;
      }}
      // Match the regular event time format ("HH:MM, MM/DD/YYYY") instead
      // of the ISO string the queries.csv carries.
      let tsDisplay = '';
      if (t.ts) {{
        const d = new Date(t.ts * 1000);
        const pad = n => (n < 10 ? '0' : '') + n;
        tsDisplay = `${{pad(d.getUTCHours())}}:${{pad(d.getUTCMinutes())}}, ${{pad(d.getUTCMonth()+1)}}/${{pad(d.getUTCDate())}}/${{d.getUTCFullYear()}}`;
      }}

      // Build rich-info sections — only render keys the extractor populated.
      let sections = '';
      if (t.ground_truth) {{
        sections += `<div class="ts-section"><div class="ts-label">Expected (ground truth)</div><div class="ts-body" style="white-space:pre-wrap;">${{escapeHtml(t.ground_truth)}}</div></div>`;
      }}
      if (Array.isArray(t.candidates) && t.candidates.length > 0) {{
        const items = t.candidates.map(c => {{
          const tag = `<span class="ts-origin ts-origin-${{escapeHtml(c.origin || '')}}">${{escapeHtml(c.origin || '')}}</span>`;
          const star = c.is_held_out ? ' <span class="ts-target">★ target</span>' : '';
          const tags = (c.hashtags || []).slice(0, 4).map(escapeHtml).join(' ');
          return `<li><code>idx=${{c.idx}}</code> ${{tag}}${{star}} ${{escapeHtml(c.title || '')}}${{tags ? ` <small>${{tags}}</small>` : ''}}</li>`;
        }}).join('');
        sections += `<div class="ts-section"><div class="ts-label">Candidate pool (${{t.candidates.length}} items)</div><ul class="ts-list">${{items}}</ul></div>`;
      }}
      if (t.held_out_pref) {{
        sections += `<div class="ts-section"><div class="ts-label">Held-out preference</div><div class="ts-body">${{escapeHtml(t.held_out_pref)}}</div></div>`;
      }}
      if (Array.isArray(t.top_k_relevant) && t.top_k_relevant.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Other supporting preferences (use sparingly, only if relevant)</div><ul class="ts-list">${{t.top_k_relevant.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (Array.isArray(t.correct_but_irrelevant_prefs) && t.correct_but_irrelevant_prefs.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Correct but irrelevant preferences (do NOT surface these here)</div><ul class="ts-list">${{t.correct_but_irrelevant_prefs.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (Array.isArray(t.tool_call_rules) && t.tool_call_rules.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Tool-call rules</div><ul class="ts-list ts-mono">${{t.tool_call_rules.map(r => `<li><code>${{escapeHtml(r)}}</code></li>`).join('')}}</ul></div>`;
      }}
      if (t.final_state_expected && Object.keys(t.final_state_expected).length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Final-state expected (writes.jsonl diff)</div><div class="ts-body ts-mono">${{escapeHtml(JSON.stringify(t.final_state_expected, null, 2))}}</div></div>`;
      }}
      if (t.warn_frame) {{
        const wf = t.warn_frame;
        const mm = (wf.must_mention || []).map(escapeHtml).map(s => `<li>${{s}}</li>`).join('');
        const mn = (wf.must_not_mention || []).map(escapeHtml).map(s => `<li>${{s}}</li>`).join('');
        sections += `<div class="ts-section ts-section-warn"><div class="ts-label">Expected warning frame [polarity=${{escapeHtml(wf.polarity || '')}}]</div>` +
                    (mm ? `<div class="ts-sublabel">must_mention</div><ul class="ts-list">${{mm}}</ul>` : '') +
                    (mn ? `<div class="ts-sublabel">must_not_mention</div><ul class="ts-list">${{mn}}</ul>` : '') +
                    `</div>`;
      }}
      if (Array.isArray(t.signal_evidence) && t.signal_evidence.length > 0) {{
        const items = t.signal_evidence.map(s => `<li><code>${{escapeHtml(s.source || '')}}</code> @${{escapeHtml(String(s.ts || ''))}} <small>${{escapeHtml(s.ref || '')}}</small><br>${{escapeHtml(s.quote || '')}}</li>`).join('');
        sections += `<div class="ts-section"><div class="ts-label">Cross-signal evidence (mistake reasoning)</div><ul class="ts-list">${{items}}</ul></div>`;
      }}
      if (Array.isArray(t.irrelevant_persona_items) && t.irrelevant_persona_items.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Irrelevant prefs (distractors agent must reject)</div><ul class="ts-list">${{t.irrelevant_persona_items.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (t.carve_out) {{
        sections += `<div class="ts-section"><div class="ts-label">Carve-out (context shift)</div><div class="ts-body">${{escapeHtml(t.carve_out)}}</div></div>`;
      }}
      if (Array.isArray(t.forbidden_items) && t.forbidden_items.length > 0) {{
        sections += `<div class="ts-section ts-section-warn"><div class="ts-label">Forbidden items</div><ul class="ts-list">${{t.forbidden_items.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (t.extra_meta && Object.keys(t.extra_meta).length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Meta</div><div class="ts-body ts-mono">${{escapeHtml(JSON.stringify(t.extra_meta, null, 2))}}</div></div>`;
      }}
      const tags = (t.rubric_tags || []).filter(Boolean);
      if (tags.length > 0) {{
        sections += `<div class="ts-section ts-rubric-bar"><div class="ts-label">Rubric dimensions</div><ul class="ts-list">${{tags.map(s => `<li>${{escapeHtml(s)}}</li>`).join('')}}</ul></div>`;
      }}

      const card = document.createElement('div');
      card.className = 'event-card test-sample-card';
      // Render any preceding chat turns first so the User Query has context.
      let priorBlock = '';
      if (Array.isArray(t.prior_conversation) && t.prior_conversation.length > 0) {{
        const bubbles = t.prior_conversation.map(m => {{
          const role = m.role === 'user' ? 'You' : 'AI';
          const cls = m.role === 'user' ? 'user-bubble' : 'assistant-bubble';
          return `<div class="chat-bubble ${{cls}}"><div class="chat-role">${{role}}</div>${{escapeHtml(m.content || '')}}</div>`;
        }}).join('');
        priorBlock = `<div class="ts-section"><div class="ts-label">Prior conversation (last ${{t.prior_conversation.length}} turns)</div><div class="chat-thread">${{bubbles}}</div></div>`;
      }}
      // Render User Query as a regular ts-section (label INSIDE the
      // section block) so it visually matches every other section.
      const queryBlock = `<div class="ts-section"><div class="ts-label">User Query</div><div class="ts-body">${{escapeHtml(t.query_text || '')}}</div></div>`;
      card.innerHTML = `
        <div class="event-header">
          <div class="event-meta">
            <span style="font-weight:600;color:#7B5C00;">Test sample</span> &middot;
            ${{escapeHtml(tsDisplay)}} &middot;
            ${{locText}}${{locText ? ' &middot; ' : ''}}
            <code>${{escapeHtml(t.task_type || '')}}</code>
          </div>
        </div>
        ${{priorBlock}}
        ${{queryBlock}}
        ${{sections}}
      `;
      grid.appendChild(card);
      return;
    }}
    const ev = item.data;
    const idx = item.eventIdx;
    const app = ev._app || 'Instagram';
    const fmt = ev.interaction_format || {{}};
    const prefs = ev.preferences || [];
    const hashtags = ev.source_hashtags || [];
    const itype = ev.source_interaction_type || '';
    const isImplicitNeg = itype === 'implicit_negative';

    const isAd = !!ev.is_ad;
    const card = document.createElement('div');
    card.className = `event-card app-${{app}}${{isImplicitNeg ? ' implicit-negative' : ''}}${{isAd ? ' is-ad' : ''}}`;

    // Location string
    let locText = '';
    if (ev.event_location && typeof ev.event_location === 'object') {{
      const loc = ev.event_location;
      const parts = [loc.city, loc.region].filter(x => x).map(escapeHtml);
      if (parts.length > 0) {{
        locText = `<span class="event-location">📍 ${{parts.join(', ')}}</span>`;
      }}
    }}

    // Event header (test annotations now live on standalone test cards,
    // not on regular event cards — keeps regular events uncluttered.)
    let headerHtml = `
      <div class="event-header">
        <div class="event-meta">
          <span style="font-weight:600;color:var(--text);">Event #${{idx+1}}</span> &middot;
          ${{ev.formatted_timestamp || ''}} &middot;
          ${{locText}}${{locText ? ' &middot; ' : ''}}
          ${{prefs.length}} preference${{prefs.length !== 1 ? 's' : ''}}
        </div>
        <div>
          <span class="badge platform p-${{app}}">${{app}}</span>
          <span class="badge interaction-type ${{itype}}">${{itype.replace(/_/g, ' ')}}</span>
          ${{fmt.action_label ? `<span class="badge action">${{fmt.action_label}}</span>` : ''}}
          ${{isAd ? `<span class="badge sponsored">Ads</span>` : ''}}
        </div>
        ${{hashtags.length ? `<div class="hashtags">${{hashtags.join('  ')}}</div>` : ''}}
      </div>
    `;

    // Preferences list
    let prefsHtml = '<div class="pref-list">';
    if (prefs.length > 0) {{
      prefsHtml += '<div class="pref-list-label">Preferences</div>';
    }}
    prefs.forEach(p => {{
      let badges = `<span class="badge category">${{p.category || ''}}</span>`;
      if (p.time_horizon === 'short_term') badges += `<span class="badge short-term">short-term</span>`;
      if (p.stereotype_mark && p.stereotype_mark !== 'neutral') badges += `<span class="badge ${{p.stereotype_mark}}">${{p.stereotype_mark}}</span>`;
      if (p.hidden_persona_labels && p.hidden_persona_labels.length > 0) {{
        p.hidden_persona_labels.forEach(lbl => {{
          const motiv = (hpMotivation[lbl] || '').replace(/"/g, '&quot;');
          const titleAttr = motiv ? ` title="${{motiv}}"` : '';
          badges += `<span class="badge hidden-persona"${{titleAttr}}>${{lbl}}</span>`;
        }});
      }}

      // R8: split / over_personalization_irrelevant are no longer emitted
      // by data-gen (eval picks its own test moments from the full history).

      const historyHtml = renderUpdateHistory(p.update_history, ev.source_timestamp);

      let stopConditionLine = '';
      if (p.time_horizon === 'short_term' && p.stop_condition && typeof p.stop_condition === 'object') {{
        const sc = p.stop_condition;
        const typeLabel = sc.type ? `<span class="sc-type">${{escapeHtml(sc.type)}}</span>` : '';
        const desc = sc.description ? escapeHtml(sc.description) : '';
        const stopTs = sc.expected_stop_ts ? new Date(sc.expected_stop_ts * 1000).toISOString().slice(0, 16).replace('T', ' ') : '';
        const stopSuffix = stopTs ? ` <span style="opacity:0.7">(stops ~${{stopTs}})</span>` : '';
        if (desc || typeLabel) {{
          stopConditionLine = `<div class="stop-condition">${{typeLabel}}${{desc}}${{stopSuffix}}</div>`;
        }}
      }}

      prefsHtml += `
        <div class="pref-item">
          <div class="item-text">${{p.persona_item || ''}}</div>
          <div class="conf-inline"><span>init ${{(p.confidence_score_init || 0).toFixed(2)}}</span><span>xref ${{(p.confidence_cross_referenced || 0).toFixed(1)}}</span></div>
          <div class="pref-meta">${{badges}}</div>
          ${{stopConditionLine}}
          ${{historyHtml}}
        </div>
      `;
    }});
    prefsHtml += '</div>';

    // Chatbot conversation
    let convHtml = '';
    if (ev.conversation && ev.conversation.length > 0) {{
      let convLabel = ev.conversation_type ? `<div class="chat-conv-label">${{ev.conversation_type.replace(/_/g, ' ')}}${{ev.ask_to_forget ? ' &middot; ask-to-forget' : ''}}</div>` : '';
      let bubbles = ev.conversation.map(t => {{
        const cls = t.role === 'user' ? 'user-bubble' : 'assistant-bubble';
        const label = t.role === 'user' ? 'You' : 'AI';
        return `<div class="chat-bubble ${{cls}}"><div class="chat-role">${{label}}</div>${{t.content}}</div>`;
      }}).join('');
      convHtml = `${{convLabel}}<div class="chat-thread">${{bubbles}}</div>`;
    }} else if (fmt.user_message) {{
      convHtml = `<div class="user-message">${{fmt.user_message}}</div>`;
    }}

    const contentHtml = renderContent(ev);
    card.innerHTML = headerHtml + contentHtml + prefsHtml + convHtml;
    grid.appendChild(card);
  }});

  timeline.appendChild(grid);
}}
</script>
</body>
</html>"""

    output_path = os.path.join(user_dir, "persona.html")
    os.makedirs(user_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"{utils.Colors.OKGREEN}Visualization saved to {output_path}{utils.Colors.ENDC}")
    return output_path
