"""Generate self-authored post events for a user, per social app.

One LLM batch call per app. Each post carries:
- source_interaction_type = "explicit_positive" (authorship is the strongest explicit signal)
- author_id = "self", relationship = "self", is_self_authored = True, is_dm = False
- source_timestamp sampled across the user's existing event range
- content_type varies by app's content mix + app persona
- content = {title?, caption, overall_description?, ...} — in the user's voice

Target counts (per app, from plan's "minimum data-volume assertions"):
- Instagram: 10
- Facebook:  8
- Threads:   12
(Adjusted down if posting_frequency is 'rarely'.)
"""

from __future__ import annotations

import json
import random
from data_preparation.utils import extract_json_from_response


_DEFAULT_COUNTS = {"instagram": 10, "facebook": 8, "threads": 12}
_POSTING_FREQ_MULT = {
    "never":   0.0,
    "rarely":  0.5,
    "monthly": 0.8,
    "weekly":  1.0,
    "daily":   1.5,
}


SELF_POSTS_PROMPT = """You are generating realistic self-authored posts for a simulated user on {app_pretty}.

The posts should read as if THIS user wrote them — matching their voice, topical focus, and
everyday life. These are the user's OWN content, not content they consumed.

User profile:
- Name: {name}
- Gender: {gender}
- Career: {career}
- Bio: {bio}
- Big Five: {big_five}
- MBTI: {mbti}

{app_pretty} persona (how this user acts on {app_pretty}):
- Use purposes: {use_purposes}
- Audience type: {audience_type}
- Posting frequency: {posting_frequency}
- Style description: {style_description}
- Topical focus: {topical_focus}

Produce {n} distinct posts. Each must:
1. Sound like the user wrote it — vocabulary, register, length typical of {app_pretty}.
2. Cover a mix of the topical_focus (not all on the same topic).
3. Carry 1–4 hashtags relevant to the post content (use hashtags the user would realistically use).
4. Match platform format: Instagram posts have a caption + content_type image/video;
   Facebook posts are typically status text; Threads posts are short (under 300 chars).
5. Be grounded in the user's real persona — not generic marketing copy.

Return JSON only:
```json
[
  {{
    "content_type": "text|image|short_video",
    "title": "... (short, optional, usually empty for IG caption-only or FB status)",
    "caption": "the actual post text the user would publish",
    "hashtags": ["#tag1", "#tag2"],
    "overall_description": "one-sentence summary of what the post is about (for image/video) — else omit"
  }},
  ...
]
```
"""


def _target_count(app: str, app_persona: dict) -> int:
    """Scale the default count by the user's per-app posting_frequency."""
    base = _DEFAULT_COUNTS.get(app, 8)
    freq = (app_persona or {}).get("posting_frequency", "weekly")
    mult = _POSTING_FREQ_MULT.get(freq, 1.0)
    return max(3, int(round(base * mult)))


def _sample_timestamps(event_timestamps: list[int], n: int, seed: int) -> list[int]:
    """Pick n timestamps distributed across the user's activity range."""
    if not event_timestamps or n <= 0:
        return []
    lo, hi = min(event_timestamps), max(event_timestamps)
    rng = random.Random(seed)
    return sorted(rng.randint(lo, hi) for _ in range(n))


def generate_self_posts(
    user_id: str,
    app: str,
    profile: dict,
    existing_events: list[dict],
    llm_client,
    rng_seed: int = 0,
) -> list[dict]:
    """Generate self-posts for one app. Returns a list of event dicts ready to
    append to backend/{user_id}/{app}.json.
    """
    app_pretty = app.capitalize()
    app_persona = (profile.get("app_personas", {}) or {}).get(app_pretty, {}) or {}
    n = _target_count(app, app_persona)
    if n == 0:
        return []

    big_five = profile.get("big_five", {}) or {}
    prompt = SELF_POSTS_PROMPT.format(
        app_pretty=app_pretty,
        n=n,
        name=profile.get("name", "the user"),
        gender=profile.get("gender", ""),
        career=profile.get("career", ""),
        bio=(profile.get("bio", "") or "")[:400],
        big_five=", ".join(f"{k}={v}" for k, v in big_five.items()),
        mbti=profile.get("mbti", ""),
        use_purposes=", ".join(app_persona.get("use_purposes", [])) or "n/a",
        audience_type=app_persona.get("audience_type", "mixed"),
        posting_frequency=app_persona.get("posting_frequency", "weekly"),
        style_description=(app_persona.get("style_description", "") or "")[:300],
        topical_focus=", ".join(app_persona.get("topical_focus", [])) or "n/a",
    )
    resp = llm_client.query_llm(prompt)
    posts = extract_json_from_response(resp) or []
    if not isinstance(posts, list):
        return []

    # Timestamps sampled from the user's existing event range.
    event_ts = [int(e.get("source_timestamp", 0)) for e in existing_events if e.get("source_timestamp")]
    timestamps = _sample_timestamps(event_ts, len(posts), rng_seed)

    # Build a hashtag → canonical-preference index from the existing events
    # so self-posts can carry the same `preferences` list shape that regular
    # explicit_positive events do. Dedupe by persona_item + remember the
    # earliest occurrence of each pref so update_history reflects "this is a
    # post that REINFORCES an existing canonical preference".
    pref_index: dict[str, list[dict]] = {}
    for e in existing_events:
        for pref in (e.get("preferences") or []):
            if not isinstance(pref, dict):
                continue
            for h in (pref.get("source_hashtags") or e.get("source_hashtags") or []):
                pref_index.setdefault(_norm_tag(h), []).append(pref)

    # Sort existing events by ts once for the location-lookup binary search.
    sorted_evs = sorted(
        (e for e in existing_events if e.get("event_location")),
        key=lambda e: int(e.get("source_timestamp", 0)),
    )

    out: list[dict] = []
    for i, p in enumerate(posts):
        if not p.get("caption"):
            continue
        ts = timestamps[i] if i < len(timestamps) else (timestamps[-1] + 60 if timestamps else 0)
        formatted_ts = _format_ts(ts)
        post_hashtags = p.get("hashtags", []) or []
        # Match this post's hashtags against the canonical pref index.
        # A self-post with #LegDay #StrengthTraining picks up "Loves working
        # out" / "Strength training enthusiast" canonical prefs the same way
        # a non-self explicit_positive on those tags would.
        seen_pi: set[str] = set()
        post_prefs: list[dict] = []
        for h in post_hashtags:
            for cand in pref_index.get(_norm_tag(h), []):
                pi = cand.get("persona_item") or ""
                if not pi or pi in seen_pi:
                    continue
                seen_pi.add(pi)
                # Shallow-copy + add a "reinforced via self-post" update entry
                # so the timeline shows this self-post strengthens the pref.
                copy = {k: v for k, v in cand.items() if k != "update_history"}
                hist = list(cand.get("update_history") or [])
                hist.append({
                    "update_type": "reinforced",
                    "formatted_timestamp": formatted_ts,
                    "source_app": app_pretty,
                    "occurrence": "self_post",
                })
                copy["update_history"] = hist
                post_prefs.append(copy)
        event_location = _location_at(ts, sorted_evs)
        event = {
            "source_object_id": f"self_{app}_{user_id}_{i:03d}",
            "source_timestamp": ts,
            "formatted_timestamp": formatted_ts,
            "source_hashtags": post_hashtags,
            "source_interaction_type": "explicit_positive",
            "author_id": "self",
            "recipient_id": "",
            "relationship": "self",
            "is_self_authored": True,
            "is_dm": False,
            "interaction_format": {
                "app": app_pretty,
                "action": _action_for_content(app, p.get("content_type", "text")),
                "action_label": _action_label_for(app, p.get("content_type", "text")),
                "user_message": None,
            },
            "content_type": p.get("content_type", "text"),
            "content": {
                "title": p.get("title", "") or "",
                "caption": p.get("caption", ""),
                "overall_description": p.get("overall_description", "") or "",
            },
            "preferences": post_prefs,
        }
        if event_location:
            event["event_location"] = event_location
        out.append(event)
    return out


def _norm_tag(h: str) -> str:
    return (h or "").lower().lstrip("#").strip()


def _location_at(ts: int, sorted_events_with_loc: list[dict]) -> dict | None:
    """Binary-search the nearest preceding event with event_location set."""
    if not sorted_events_with_loc:
        return None
    lo, hi, best = 0, len(sorted_events_with_loc) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        ev_ts = int(sorted_events_with_loc[mid].get("source_timestamp", 0))
        if ev_ts <= ts:
            best = sorted_events_with_loc[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best.get("event_location") if best else sorted_events_with_loc[0].get("event_location")


def _format_ts(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M, %m/%d/%Y")


def _action_for_content(app: str, content_type: str) -> str:
    if app == "instagram":
        return {"image": "posted_image", "short_video": "posted_reel", "text": "posted_caption"}.get(content_type, "posted_image")
    if app == "facebook":
        return {"image": "posted_photo", "short_video": "posted_video", "text": "posted_status"}.get(content_type, "posted_status")
    if app == "threads":
        return {"image": "posted_image", "short_video": "posted_video", "text": "posted_thread"}.get(content_type, "posted_thread")
    return "posted"


def _action_label_for(app: str, content_type: str) -> str:
    action = _action_for_content(app, content_type)
    pretty = action.replace("_", " ").capitalize()
    return pretty
