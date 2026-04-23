"""Generate DM threads per social app — inbound, outbound, and group.

Writes backend/{user_id}/{app}_dms.json with structure:
    [
      {"thread_id": "ig_thr_001",
       "participants": ["self", "friend_3"],
       "is_group": false,
       "messages": [
         {"msg_id": "m_001", "sender": "friend_3", "timestamp": ..., "text": "..."},
         {"msg_id": "m_002", "sender": "self",     "timestamp": ..., "text": "..."}
       ]},
      ...
    ]

Also mirrors each top-level DM message into the main {app}.json as an event
with is_dm=True + thread_id so cross-reference by thread_id works from either
side. This is what the MCP server's list_dm_threads fallback reads.
"""

from __future__ import annotations

import json
import random
from data_preparation.utils import extract_json_from_response


DM_THREADS_PROMPT = """Generate realistic DM conversations on {app_pretty} for this user.

User:
- Name: {name}
- Career: {career}
- Bio: {bio}
- Style on {app_pretty}: {style_description}

Known friends (the user's actual contacts):
{friends_block}

Generate {n_threads} DM threads total, mixing:
- {n_inbound_friend} threads where a friend DMed the user; the user reacted variously
   (replied enthusiastically, replied briefly, or did not reply at all).
- {n_outbound} threads where the user DMed a friend first.
- {n_stranger} threads where a stranger (non-friend) DMed the user; user mostly does NOT reply.
- {n_group} threads that are group chats (3–4 participants including the user).

Each thread has 1–4 messages. Texts are short, casual — how people really DM. Topics should
connect to the user's topical interests where natural. Use the real friend_ids above.

Return JSON only:
```json
[
  {{
    "thread_kind": "inbound_friend | outbound | stranger | group",
    "participants": ["self", "friend_3"],
    "is_group": false,
    "messages": [
      {{"sender": "friend_3", "text": "..."}},
      {{"sender": "self",     "text": "..."}}
    ]
  }},
  ...
]
```

Strangers get placeholder sender ids like "stranger_1". For group threads, `participants`
is 3–4 ids mixing self + friends; `is_group` is true; messages can come from any participant.
"""


def _friends_block(friends: list[dict]) -> str:
    if not friends:
        return "(no friend graph available — use stranger placeholders)"
    return "\n".join(
        f"- {fr.get('friend_id')}: {fr.get('display_name', '?')} "
        f"({fr.get('relationship_depth', '?')}; interests: {', '.join(fr.get('shared_interests', []))})"
        for fr in friends
    )


def _sample_dm_timestamps(event_ts: list[int], n: int, seed: int) -> list[int]:
    if not event_ts or n <= 0:
        return []
    lo, hi = min(event_ts), max(event_ts)
    rng = random.Random(seed)
    return sorted(rng.randint(lo, hi) for _ in range(n))


def generate_dm_threads(
    user_id: str,
    app: str,
    profile: dict,
    friends: list[dict],
    existing_events: list[dict],
    llm_client,
    rng_seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Returns (threads_for_dms_json, events_for_main_app_json).

    - threads_for_dms_json: goes to `{app}_dms.json`.
    - events_for_main_app_json: one event per thread's latest message, tagged
      with is_dm=True + thread_id, appended to `{app}.json` so list_dm_threads
      fallback finds them.
    """
    app_pretty = app.capitalize()
    app_persona = (profile.get("app_personas", {}) or {}).get(app_pretty, {}) or {}
    n_inbound_friend = 4
    n_outbound = 3
    n_stranger = 2
    n_group = 2 if app == "instagram" else 1  # IG has more active groups; FB/Threads rarely
    n_threads = n_inbound_friend + n_outbound + n_stranger + n_group

    prompt = DM_THREADS_PROMPT.format(
        app_pretty=app_pretty,
        name=profile.get("name", ""),
        career=profile.get("career", ""),
        bio=(profile.get("bio", "") or "")[:300],
        style_description=(app_persona.get("style_description", "") or "")[:200],
        friends_block=_friends_block(friends),
        n_threads=n_threads,
        n_inbound_friend=n_inbound_friend,
        n_outbound=n_outbound,
        n_stranger=n_stranger,
        n_group=n_group,
    )
    resp = llm_client.query_llm(prompt)
    threads_raw = extract_json_from_response(resp) or []
    if not isinstance(threads_raw, list):
        return [], []

    event_ts = [int(e.get("source_timestamp", 0)) for e in existing_events if e.get("source_timestamp")]
    thread_start_ts = _sample_dm_timestamps(event_ts, len(threads_raw), rng_seed)

    threads: list[dict] = []
    mirror_events: list[dict] = []
    for i, t in enumerate(threads_raw):
        tid = f"{app[:2]}_thr_{user_id}_{i:03d}"
        msgs_raw = t.get("messages", []) or []
        if not msgs_raw:
            continue
        t0 = thread_start_ts[i] if i < len(thread_start_ts) else (event_ts[-1] if event_ts else 0)
        msgs = []
        for j, m in enumerate(msgs_raw):
            msg_ts = t0 + j * random.Random(f"{rng_seed}:{tid}:{j}").randint(60, 600)
            msgs.append({
                "msg_id": f"{tid}_m_{j:02d}",
                "sender": m.get("sender", "unknown"),
                "timestamp": msg_ts,
                "text": m.get("text", ""),
            })
        is_group = bool(t.get("is_group"))
        participants = t.get("participants") or []
        threads.append({
            "thread_id": tid,
            "participants": participants,
            "is_group": is_group,
            "thread_kind": t.get("thread_kind", "unspecified"),
            "messages": msgs,
        })
        # Mirror the LATEST message into the main app JSON so MCP list_dm_threads
        # fallback + Task B privacy evaluators can see the DM exists.
        last = msgs[-1]
        last_sender = last["sender"]
        is_inbound = last_sender != "self"
        # Interaction type heuristic: inbound-from-friend = implicit_positive,
        # inbound-from-stranger = implicit_negative, outbound = explicit_positive.
        if last_sender == "self":
            interaction_type = "explicit_positive"
        elif any(p.get("friend_id") == last_sender for p in friends or []):
            interaction_type = "implicit_positive"
        else:
            interaction_type = "implicit_negative"
        mirror_events.append({
            "source_object_id": last["msg_id"],
            "source_timestamp": last["timestamp"],
            "formatted_timestamp": _format_ts(last["timestamp"]),
            "source_hashtags": [],
            "source_interaction_type": interaction_type,
            "author_id": last_sender,
            "recipient_id": "self" if is_inbound else (next((p for p in participants if p != "self"), "unknown")),
            "relationship": _resolve_relationship(last_sender, friends),
            "is_self_authored": last_sender == "self",
            "is_dm": True,
            "thread_id": tid,
            "is_group_dm": is_group,
            "interaction_format": {
                "app": app_pretty,
                "action": "sent_dm" if last_sender == "self" else "received_dm",
                "action_label": "Sent a DM" if last_sender == "self" else "Received a DM",
                "user_message": None,
            },
            "content_type": "text",
            "content": {"caption": last["text"]},
            "preferences": [],
        })
    return threads, mirror_events


def _resolve_relationship(sender_id: str, friends: list[dict]) -> str:
    if sender_id == "self":
        return "self"
    if any(fr.get("friend_id") == sender_id for fr in (friends or [])):
        # Pick up relationship_depth from the friend record.
        for fr in friends:
            if fr.get("friend_id") == sender_id:
                depth = fr.get("relationship_depth", "acquaintance")
                return {"close": "friend", "acquaintance": "friend", "distant": "friend"}.get(depth, "friend")
        return "friend"
    return "stranger"


def _format_ts(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M, %m/%d/%Y")
