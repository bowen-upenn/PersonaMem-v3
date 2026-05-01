"""Generate DM threads per social app — inbound, outbound, and group.

Each DM thread is emitted as ONE event-shaped entry appended directly to
the main `{app}.json` list, with `is_dm: true` + the full `messages[]`
array embedded. No separate `{app}_dms.json` file is produced anymore —
a single merged list per app is simpler for consumers (MCP servers,
eval harness, HTML renderer) and sorts naturally by `source_timestamp`.
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
) -> list[dict]:
    """Return a list of DM-thread events to append to `{app}.json`.

    Each entry is one full DM thread shaped as an event with the complete
    `messages[]` embedded, `is_dm: true`, and a `dm_conversation` action.
    Consumers iterate the main app JSON and filter by `is_dm` for DM
    queries (MCP `list_dms` / `get_dm_thread`).
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
        return []

    event_ts = [int(e.get("source_timestamp", 0)) for e in existing_events if e.get("source_timestamp")]
    thread_start_ts = _sample_dm_timestamps(event_ts, len(threads_raw), rng_seed)

    merged_events: list[dict] = []
    for i, t in enumerate(threads_raw):
        tid = f"{app[:2]}_thr_{user_id}_{i:03d}"
        msgs_raw = t.get("messages", []) or []
        if not msgs_raw:
            continue
        t0 = thread_start_ts[i] if i < len(thread_start_ts) else (event_ts[-1] if event_ts else 0)
        msgs: list[dict] = []
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
        thread_kind = t.get("thread_kind", "unspecified")
        last = msgs[-1]
        last_sender = last["sender"]
        latest_ts = max(int(m.get("timestamp") or 0) for m in msgs)
        # Interaction-type rules:
        #   implicit_negative → DM from a stranger with NO user response
        #   implicit_positive → DM from a friend with NO user response
        #   explicit_positive → user responded positively (positive reply,
        #                       liked, like-equivalent emoji)
        #   implicit_positive → user responded but not positively (still
        #                       engagement, just neutral/lukewarm)
        # If the thread has only self messages (outbound), it's an
        # explicit_positive outreach — the user is initiating contact.
        non_self_msgs = [m for m in msgs if m.get("sender") != "self"]
        self_msgs = [m for m in msgs if m.get("sender") == "self"]
        if not non_self_msgs:
            # Outbound — user-initiated. Treat as explicit_positive.
            interaction_type = "explicit_positive"
        else:
            initiator = non_self_msgs[0].get("sender")
            initiator_is_friend = any(
                fr.get("friend_id") == initiator for fr in (friends or [])
            )
            if not self_msgs:
                # User did not respond.
                interaction_type = (
                    "implicit_positive" if initiator_is_friend
                    else "implicit_negative"
                )
            else:
                # User responded — check polarity.
                interaction_type = (
                    "explicit_positive" if _self_responded_positively(self_msgs)
                    else "implicit_positive"
                )
        merged_events.append({
            "source_object_id": tid,
            "source_timestamp": latest_ts,
            "formatted_timestamp": _format_ts(latest_ts),
            "source_hashtags": [],
            "source_interaction_type": interaction_type,
            "author_id": last_sender,
            "recipient_id": (
                "self" if last_sender != "self"
                else next((p for p in participants if p != "self"), "unknown")
            ),
            "relationship": _resolve_relationship(last_sender, friends),
            "is_self_authored": last_sender == "self",
            "is_dm": True,
            "thread_id": tid,
            "is_group_dm": is_group,
            "thread_kind": thread_kind,
            "participants": participants,
            "messages": msgs,
            "interaction_format": {
                "app": app_pretty,
                "action": "dm_conversation",
                "action_label": "DM conversation",
                "user_message": None,
            },
            "content_type": "text",
            "content": {
                "caption": last.get("text", ""),
                "n_messages": len(msgs),
            },
            "preferences": [],
        })
    return merged_events


# Tokens that a self message must carry to count as a "positive response".
# Captures explicit reply text + like / heart / laugh emoji equivalents.
_POSITIVE_TOKENS = (
    "lol", "haha", "lmao", "lmfao", "yes", "yeah", "yep", "yup",
    "love it", "love this", "loved it", "love that",
    "sounds good", "sounds great", "sg", "looks good",
    "nice", "amazing", "awesome", "great", "cool", "perfect", "fantastic",
    "agreed", "for sure", "definitely", "absolutely", "exactly",
    "thanks", "thank you", "ty", "tysm", "appreciate",
    "down", "i'm in", "let's go", "bet", "fr", "facts",
    "❤", "🧡", "💛", "💚", "💙", "💜", "🤍",
    "👍", "👌", "🙌", "🔥", "💯",
    "😂", "🤣", "😆", "😄", "😊", "🥰", "😍", "🤩",
)


def _self_responded_positively(self_msgs: list[dict]) -> bool:
    """True if any self-message contains a positive token (text reply,
    like, like-equivalent emoji). Heuristic-only — covers the user's
    rule: 'positive reply, liked, other emoji similar to like'."""
    for m in self_msgs:
        text = (m.get("text") or "").lower()
        if any(tok in text for tok in _POSITIVE_TOKENS):
            return True
    return False


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
