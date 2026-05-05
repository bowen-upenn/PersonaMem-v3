"""Generate DM threads per social app — every thread is anchored to a
REAL seed event from the user's interaction history. A friend / the user
forwards an actual post / video / image with short commentary; the
labeling rule (initiator + user response + polarity) determines the
interaction_type. Stranger threads are kept as cold outreach with no
preference link.

Each DM thread is emitted as ONE event-shaped entry appended directly to
the main `{app}.json` list, with `is_dm: true` + the full `messages[]`
array embedded. A message can carry `forwarded_content` (a copy of the
seed event's content + source_object_id + hashtags + source_app), which
the renderer treats as a forwarded post bubble. The thread inherits the
seed event's preferences[] so the DM event participates in
preference-anchored audits the same way as its underlying post.
"""

from __future__ import annotations

import json
import random
from data_preparation.utils import extract_json_from_response
from data_preparation.prompts import _render_voice_for_consumer


DM_THREAD_COMMENTARY_PROMPT = """Write a short, realistic DM exchange on {app_pretty} \
where {initiator_label} forwards / mentions a piece of content the user has actually \
engaged with, and the user reacts.

User:
- Name: {name}
- DM recipient: {recipient_name} (relationship depth: {recipient_depth})

{user_voice_block}
Recipient-aware register selection: pick a register from `active_registers` that fits the recipient above. Close friends license backstage register; family typically draws polite-warm; acquaintances draw task-direct. The user's underlying idiolect (function-word habits, hedge/booster ratio, sentence-length shape) does NOT change per recipient — only the register selection does. DM threads are intimate; emoji intensity may run a bit hotter than feed posts (palette only — never invent new emoji). Apply `idiolect.constructional_templates` ABSTRACTLY (slot-pattern shape — never recite verbatim); catchphrase residue may surface ZERO times across the thread. **Respect the negatives** — Voice avoid + Phrases to avoid + App avoid are hard constraints when present.

The forwarded content (real, from the user's feed):
- Topic / hashtags: {hashtags}
- Caption / title: {seed_caption}

Conversation kind: {thread_kind}
{conversation_guidance}

Output ONE fenced ```json block — exactly the messages, ≤4 turns, each
text 5–25 words, casual. Do NOT include the forwarded post itself in the
text — the forwarded post is attached separately. Just the human
commentary around it.

```json
{{
  "messages": [
    {{"sender": "{initiator_id}", "text": "<short commentary about the post>",
      "carries_forwarded_post": true}},
    {{"sender": "self", "text": "<short reaction>",
      "reaction_emoji": "<single emoji like ❤ or 🔥, OR null if not applicable>"}},
    ...
  ]
}}
```

Rules:
- The first message should set up the share: a short comment about the post.
- Only ONE message has `carries_forwarded_post: true` (the share itself).
- A message can be EMOJI-ONLY: leave `text` empty and set `reaction_emoji` to a single emoji.
- A message can MIX: short text + reaction_emoji at the end.
- `reaction_emoji` is ONLY a single small emoji (like ❤ / 🔥 / 👍 / 😂 / 🙌 / 💯 / 🤣). Never put it inside the text field.
- For "{thread_kind}", follow the guidance above for whether and how the user replies.
"""


_THREAD_GUIDANCE = {
    "inbound_friend_no_reply": (
        "A friend forwards the post. The user does NOT reply at all — "
        "produce ONLY the friend's message. (1 message total.)"
    ),
    "inbound_friend_positive_text": (
        "A friend forwards the post and the user replies POSITIVELY in "
        "TEXT — a short positive reply like 'lol love this' or 'yep "
        "agreed' or 'this is exactly my vibe'. NO emoji-only message. "
        "(2 messages total.)"
    ),
    "inbound_friend_emoji_react": (
        "A friend forwards the post and the user reacts with ONLY an "
        "emoji — no words, just one of: 🔥 / ❤ / 👍 / 😂 / 🙌 / 💯 / 🤣. "
        "Set `reaction_emoji` on the user's message and leave `text` "
        "empty. (2 messages total.)"
    ),
    "inbound_friend_positive_mixed": (
        "A friend forwards the post and the user replies with a SHORT "
        "positive text PLUS a like/heart emoji at the end. (2 messages "
        "total — the user's message has both `text` and `reaction_emoji`.)"
    ),
    "inbound_friend_neutral": (
        "A friend forwards the post and the user replies but NOT positively "
        "(a brief neutral observation, no enthusiasm). (2 messages total.)"
    ),
    "outbound_friend_share": (
        "The USER forwards the post to a friend with a short comment. The "
        "friend reacts back — sometimes with text, sometimes with just an "
        "emoji (use `reaction_emoji` on the friend's message). "
        "(2 messages total.)"
    ),
    "stranger_no_reply": (
        "A stranger DMs the user with cold outreach UNRELATED to the user's "
        "interests (recruiter pitch, generic ad, scam, etc.). The user does "
        "NOT reply. (1 message; ignore the forwarded post hashtags — the "
        "stranger does NOT forward content; carries_forwarded_post=false.)"
    ),
    "group_chat": (
        "A small group thread (3 participants incl. the user) discussing the "
        "forwarded post. Participants exchange short reactions; one of them "
        "may use just an emoji (`reaction_emoji`). The user replies "
        "positively. (3 messages total.)"
    ),
}


# DM voice rendering is now unified via prompts._render_voice_for_consumer.
# DMs foreground audience_design + active stances because per-recipient register
# selection (Bell's audience design at message granularity) is what genuinely
# varies in DMs — not voice mechanics.


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


def _pick_seed_events(
    events: list[dict],
    n: int,
    polarity: str,
    rng: random.Random,
) -> list[dict]:
    """Pick ``n`` real events to serve as seed posts for DM threads.
    Filters by interaction_type (positive engagements only) so the
    forwarded content is something the user actually liked. Skips
    events with no content (DMs, empty captions)."""
    pool = [
        e for e in events
        if not e.get("is_dm")
        and (e.get("source_interaction_type") or "") in {polarity, "explicit_positive"}
        and (e.get("content") or {})
        and (
            (e.get("content") or {}).get("caption")
            or (e.get("content") or {}).get("title")
        )
    ]
    rng.shuffle(pool)
    return pool[:n]


def _content_for_forward(seed: dict) -> dict:
    """Compact projection of a seed event's content for embedding into a
    DM message's forwarded_content field. Carries the source_object_id +
    hashtags + source_app so the renderer can treat the bubble as a real
    forwarded post."""
    return {
        "source_object_id": seed.get("source_object_id", ""),
        "source_app": seed.get("_app") or seed.get("interaction_format", {}).get("app", ""),
        "source_hashtags": seed.get("source_hashtags") or [],
        "content_type": seed.get("content_type", "text"),
        "content": seed.get("content") or {},
    }


def generate_dm_threads(
    user_id: str,
    app: str,
    profile: dict,
    friends: list[dict],
    existing_events: list[dict],
    llm_client,
    rng_seed: int = 0,
) -> list[dict]:
    """Return DM-thread events anchored to REAL seed events. Each
    inbound-friend / outbound / group thread carries a forwarded post
    (the seed) in one message; commentary text comes from a per-thread
    LLM call. Stranger threads are cold outreach with no seed.

    Thread inherits seed event's preferences[] + source_hashtags so the
    DM event participates in preference audits like any other event.
    """
    app_pretty = app.capitalize()
    app_persona = (profile.get("app_personas", {}) or {}).get(app_pretty, {}) or {}
    user_voice = profile.get("user_voice", {}) or {}
    voice_block = _render_voice_for_consumer(
        user_voice,
        app_persona,
        foreground=["audience_design", "stances"],
    )
    rng = random.Random(rng_seed or hash(user_id) % (2**31))

    # Tag every event with its source app so _content_for_forward can record it.
    for e in existing_events:
        e.setdefault("_app", app)

    # Plan thread mix per app (same overall counts as before).
    plans: list[dict] = []
    friends_for_app = [f for f in friends or [] if f.get("friend_id")]

    def _friend(): return rng.choice(friends_for_app) if friends_for_app else None

    # Mix of response styles per friend-inbound thread — text replies,
    # emoji-only reactions, and mixed text+emoji are all common in real DMs.
    # Roughly: 30% text-positive, 30% emoji-react, 20% mixed, 10% neutral,
    # 10% no-reply. Plus ~20% outbound (user-initiated → explicit_positive)
    # and ~10% stranger cold outreach.
    for _ in range(2):
        f = _friend()
        if f: plans.append({"kind": "inbound_friend_positive_text",
                            "initiator_id": f["friend_id"], "needs_seed": True})
    for _ in range(2):
        f = _friend()
        if f: plans.append({"kind": "inbound_friend_emoji_react",
                            "initiator_id": f["friend_id"], "needs_seed": True})
    for _ in range(1):
        f = _friend()
        if f: plans.append({"kind": "inbound_friend_positive_mixed",
                            "initiator_id": f["friend_id"], "needs_seed": True})
    for _ in range(1):
        f = _friend()
        if f: plans.append({"kind": "inbound_friend_no_reply",
                            "initiator_id": f["friend_id"], "needs_seed": True})
    for _ in range(2):
        f = _friend()
        if f: plans.append({"kind": "outbound_friend_share",
                            "initiator_id": "self", "needs_seed": True})
    for _ in range(1):
        f = _friend()
        if f: plans.append({"kind": "inbound_friend_neutral",
                            "initiator_id": f["friend_id"], "needs_seed": True})
    for _ in range(1):
        plans.append({"kind": "stranger_no_reply",
                      "initiator_id": f"stranger_{rng.randint(1, 99)}", "needs_seed": False})
    n_group = 2 if app == "instagram" else 1
    for _ in range(n_group):
        if len(friends_for_app) >= 2:
            picks = rng.sample(friends_for_app, 2)
            plans.append({
                "kind": "group_chat",
                "initiator_id": picks[0]["friend_id"],
                "group_participants": ["self", picks[0]["friend_id"], picks[1]["friend_id"]],
                "needs_seed": True,
            })

    # Pick seed events for every plan that needs one.
    needs_seed = [p for p in plans if p["needs_seed"]]
    seeds = _pick_seed_events(existing_events, len(needs_seed), "implicit_positive", rng)
    for plan, seed in zip(needs_seed, seeds):
        plan["seed"] = seed

    # Pick timestamps spread across the user's window.
    event_ts = [int(e.get("source_timestamp", 0)) for e in existing_events if e.get("source_timestamp")]
    thread_start_ts = _sample_dm_timestamps(event_ts, len(plans), rng_seed)

    merged_events: list[dict] = []
    for i, plan in enumerate(plans):
        kind = plan["kind"]
        if plan["needs_seed"] and "seed" not in plan:
            continue  # ran out of seeds
        tid = f"{app[:2]}_thr_{user_id}_{i:03d}"
        seed = plan.get("seed")
        seed_caption = ""
        seed_hashtags: list[str] = []
        if seed:
            content = seed.get("content") or {}
            seed_caption = (content.get("caption") or content.get("title") or "")[:200]
            seed_hashtags = seed.get("source_hashtags") or []

        # Resolve recipient identity for register selection (Bell's audience design).
        recipient_id = plan["initiator_id"] if plan["initiator_id"] != "self" else "(reply target)"
        recipient_friend = next(
            (f for f in (friends or []) if f.get("friend_id") == recipient_id),
            None,
        )
        recipient_name = (recipient_friend or {}).get("display_name") or recipient_id
        recipient_depth = (recipient_friend or {}).get("relationship_depth") or (
            "stranger" if recipient_id.startswith("stranger") else "(unknown)"
        )

        # LLM call: generate the human commentary around the forwarded post.
        prompt = DM_THREAD_COMMENTARY_PROMPT.format(
            app_pretty=app_pretty,
            name=profile.get("name", ""),
            recipient_name=recipient_name,
            recipient_depth=recipient_depth,
            user_voice_block=voice_block,
            initiator_id=plan["initiator_id"],
            initiator_label=("you" if plan["initiator_id"] == "self"
                              else f"a friend ({plan['initiator_id']})"
                              if plan["initiator_id"].startswith("friend_")
                              else "a stranger"),
            hashtags=", ".join(seed_hashtags[:6]) if seed_hashtags else "(no specific topic)",
            seed_caption=seed_caption or "(no specific post — cold outreach)",
            thread_kind=kind,
            conversation_guidance=_THREAD_GUIDANCE.get(kind, ""),
        )
        try:
            resp = llm_client.query_llm(prompt)
        except Exception:
            continue
        parsed = extract_json_from_response(resp) or {}
        msgs_raw = parsed.get("messages") or []
        if not msgs_raw:
            continue

        # Stamp messages with timestamps + msg_ids and attach forwarded content.
        t0 = thread_start_ts[i] if i < len(thread_start_ts) else (event_ts[-1] if event_ts else 0)
        msgs: list[dict] = []
        for j, m in enumerate(msgs_raw):
            msg_ts = t0 + j * rng.randint(60, 600)
            built = {
                "msg_id": f"{tid}_m_{j:02d}",
                "sender": m.get("sender", plan["initiator_id"]),
                "timestamp": msg_ts,
                "text": m.get("text", "") or "",
            }
            # Reaction-only messages carry an emoji instead of (or in
            # addition to) text. Stored as a separate field so the
            # renderer can show them as a small reaction bubble.
            reaction = m.get("reaction_emoji")
            if isinstance(reaction, str) and reaction.strip() and reaction.lower() not in ("null", "none"):
                built["reaction_emoji"] = reaction.strip()
            if m.get("carries_forwarded_post") and seed:
                built["forwarded_content"] = _content_for_forward(seed)
            # Drop the message if it has neither text nor reaction nor
            # forwarded content (LLM occasionally emits empty stubs).
            if not built["text"] and not built.get("reaction_emoji") and not built.get("forwarded_content"):
                continue
            msgs.append(built)

        is_group = (kind == "group_chat")
        participants = (
            plan.get("group_participants")
            or (["self", plan["initiator_id"]] if plan["initiator_id"] != "self"
                else ["self", "friend"])
        )
        last = msgs[-1]
        last_sender = last["sender"]
        latest_ts = max(int(m.get("timestamp") or 0) for m in msgs)

        # Interaction-type rules:
        #   explicit_positive → first message is from self (the user
        #                       initiated the share — user-initiated DMs
        #                       are positive engagement signals); OR
        #                       a friend/stranger sent first AND the user
        #                       responded positively (text or emoji react).
        #   implicit_positive → friend sent first, user responded but
        #                       not positively; OR friend sent first,
        #                       user did not respond.
        #   implicit_negative → stranger sent first, user did not respond.
        non_self_msgs = [m for m in msgs if m.get("sender") != "self"]
        self_msgs = [m for m in msgs if m.get("sender") == "self"]
        first_sender = msgs[0].get("sender")
        if first_sender == "self" or not non_self_msgs:
            interaction_type = "explicit_positive"  # user initiated the share
        else:
            initiator = non_self_msgs[0].get("sender")
            initiator_is_friend = any(
                fr.get("friend_id") == initiator for fr in (friends or [])
            )
            if not self_msgs:
                interaction_type = (
                    "implicit_positive" if initiator_is_friend
                    else "implicit_negative"
                )
            else:
                interaction_type = (
                    "explicit_positive" if _self_responded_positively(self_msgs)
                    else "implicit_positive"
                )

        # Inherit preferences[] + source_hashtags from the seed event so
        # the DM event participates in preference audits like any other
        # event. Stranger threads have no seed → empty prefs (and the
        # interaction_type stays implicit_negative).
        thread_prefs = list(seed.get("preferences") or []) if seed else []
        thread_hashtags = list(seed_hashtags) if seed else []

        merged_events.append({
            "source_object_id": tid,
            "source_timestamp": latest_ts,
            "formatted_timestamp": _format_ts(latest_ts),
            "source_hashtags": thread_hashtags,
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
            "thread_kind": kind,
            "participants": participants,
            "messages": msgs,
            "seed_object_id": seed.get("source_object_id") if seed else None,
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
            "preferences": thread_prefs,
        })
    return merged_events


# Tokens that mark an ENTHUSIASTIC response (not just a neutral
# acknowledgment). Single-word fillers like "yeah" / "ok" / "sure" are
# excluded — they often appear in lukewarm replies. Real positive
# signals are: laughs, love/heart language, strong-agreement phrases,
# and like-family emoji. Plain emoji reactions are handled separately
# via reaction_emoji.
_POSITIVE_TOKENS = (
    # Laughs (always positive)
    "lol", "haha", "lmao", "lmfao", "rofl",
    # Love language
    "love it", "love this", "love that", "loved it", "loved this",
    "love how", "obsessed",
    # Strong agreement
    "exactly", "100%", "agreed", "for sure", "absolutely", "definitely",
    "facts", "true that", "real", "spot on",
    # Excitement
    "let's go", "i'm in", "i'm down", "down for it", "so good", "so dope",
    "amazing", "incredible", "perfect", "fantastic", "stunning",
    # Positive emoji
    "❤", "🧡", "💛", "💚", "💙", "💜", "🤍",
    "👍", "👌", "🙌", "🔥", "💯",
    "😂", "🤣", "😆", "🥰", "😍", "🤩",
)


def _self_responded_positively(self_msgs: list[dict]) -> bool:
    """True if any self-message carries an explicit `reaction_emoji`,
    OR contains an enthusiastic positive token. Bare acknowledgments
    like 'yeah ok' / 'sure' do NOT count — those are implicit_positive."""
    for m in self_msgs:
        if m.get("reaction_emoji"):
            return True  # emoji reaction = positive response by itself
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
