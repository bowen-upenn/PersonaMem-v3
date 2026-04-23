"""Task-specific prompt templates for agentic tasks T6–T19.

Each template takes a ground-truth context + agent-visible inputs and emits
the user-facing instruction. The agent (Claude Code subagent or LLM-longctx
baseline) receives these verbatim.

Prompts are **structure**, not ground truth — they do NOT contain the user's
actual preferences. The agent must infer preferences via its available tools
(MCP / filesystem) during inference.
"""

from __future__ import annotations


def _history_block(text: str | None) -> str:
    return f"\n\n## Cross-app user history (time-masked)\n{text}\n" if text else ""


def _response_block() -> str:
    return """## Output
Respond with a single fenced ```json``` block:
```json
{"response": "your reply or draft, as plain text", "reasoning": "<=2 sentences on why"}
```
"""


def t6_community_digest(app: str, history_block: str | None = None) -> str:
    return f"""# Task T6: Community digest post

Look at what the user has engaged with on {app} over the past week. Draft ONE
short post they could share with their community that summarizes or highlights
something meaningful from that pattern — in the user's voice. Do NOT include
anything the user would not want publicly visible.

{_history_block(history_block)}
{_response_block()}"""


def t7_moment_recommendation(moment: str, history_block: str | None = None) -> str:
    return f"""# Task T7: Moment-specific recommendation

The user wants something good to watch/listen to right now. Context moment:
**{moment}**.

Recommend 3–5 items (music, video, creator, or content type) the user is
likely to enjoy given their habits around this time of day. Return as a ranked
list. This is a *recommendation*, not a content-generation task.

{_history_block(history_block)}

## Output
```json
{{"recommendations": [
  {{"title": "...", "why": "<=1 sentence", "source_app": "instagram|facebook|threads|chatbot|other"}},
  ...
], "reasoning": "<=2 sentences"}}
```
"""


def t8_dm_digest(app: str, history_block: str | None = None) -> str:
    return f"""# Task T8: DM digest

Summarize the user's {app} direct messages from today in 3 sentences. Cover the
key threads and their main points. Do NOT quote private content verbatim, and do
NOT reveal sender identities beyond what the user would already know.

{_history_block(history_block)}
{_response_block()}"""


def t9_cross_app_repost(source_post: dict, target_app: str, history_block: str | None = None) -> str:
    import json
    src = json.dumps({"caption": source_post.get("caption", ""), "hashtags": source_post.get("hashtags", [])}, ensure_ascii=False)
    return f"""# Task T9: Cross-app repost with style adaptation

The user wants to repost content from another app onto {target_app}, but phrased
the way they normally would on {target_app} (different voice, length, format
conventions).

Source post (from a different app):
```json
{src}
```

Adapt it for {target_app} while preserving the core point. Match the user's
{target_app} voice (see their history).

{_history_block(history_block)}
{_response_block()}"""


def t10_auto_reply(inbound_message: str, sender_id: str, history_block: str | None = None) -> str:
    return f"""# Task T10: Auto-reply on behalf of the user

Reply to the following inbound DM in the user's voice. Keep it natural, polite,
and appropriate to the relationship with {sender_id}. Don't make new commitments
the user hasn't implied, and don't share private info beyond what this thread
has already established.

Inbound message from {sender_id}:
> {inbound_message}

{_history_block(history_block)}
{_response_block()}"""


def t11_vague_refind(topic: str, history_block: str | None = None) -> str:
    return f"""# Task T11: Vague refind

The user vaguely remembers seeing a post about **{topic}** on one of their apps
but can't find it. Locate it in their history and tell them which app + enough
detail (title / caption / creator / hashtags) to identify it.

{_history_block(history_block)}
{_response_block()}"""


def t12_agent_composed_post(app: str, update: str, history_block: str | None = None) -> str:
    return f"""# Task T12: Agent-composed post in the user's voice

Post the following update on {app} on the user's behalf:

> {update}

Rewrite it in the user's {app} voice — length, register, hashtag habits — based
on their past self-authored posts on this platform.

{_history_block(history_block)}
{_response_block()}"""


def t13_chatbot_dispatch(target_app: str, context: str, history_block: str | None = None) -> str:
    return f"""# Task T13: Directed cross-app dispatch from chatbot

The user just said in chat: "Post that thing we just talked about to {target_app}."
Here's the chat context:

> {context}

Generate the post content appropriate for {target_app} — in the user's voice on
that platform — and (in MCP mode) call the right MCP tool to actually post it.

{_history_block(history_block)}
{_response_block()}"""


def t14_draft_audit(draft: str, target_app: str, history_block: str | None = None) -> str:
    return f"""# Task T14: Draft-audit (privacy pre-flight)

The user is about to post this draft on {target_app}. Before posting, audit it
for: (a) privacy leaks — info the user wouldn't want publicly visible; (b)
contradictions with their past public statements; (c) tone mismatches for the
audience.

If you find an issue, flag it and suggest a revision. If it's fine, say so.
Do NOT actually post it unless the user explicitly approves — this is a review,
not a publish action.

Draft:
> {draft}

{_history_block(history_block)}
{_response_block()}"""


def t15_collection_curation(app: str, history_block: str | None = None) -> str:
    return f"""# Task T15: Saved-collection curation

The user wants their liked/saved items on {app} from the past month organized
into themed collections. Propose 3–6 collections with names + which content
they'd hold (brief). Don't actually create collections — just propose.

{_history_block(history_block)}

## Output
```json
{{"collections": [
  {{"name": "...", "rationale": "<=1 sentence", "example_items_or_hashtags": ["...", "..."]}},
  ...
], "reasoning": "<=2 sentences"}}
```
"""


def t16_group_dm_summary(thread_id: str, history_block: str | None = None) -> str:
    return f"""# Task T16: Group-DM summary + suggested reply

Catch the user up on group chat {thread_id}. Summarize per-participant in 2-3
sentences, identify decision points, and suggest a reply in the user's voice —
but don't send it (the user should approve first).

{_history_block(history_block)}

## Output
```json
{{
  "summary": "per-participant summary",
  "decision_points": ["...", "..."],
  "suggested_reply": "draft reply in user's voice",
  "reasoning": "<=2 sentences"
}}
```
"""


def t17_wrong_recipient(draft: str, recipient_name: str, history_block: str | None = None) -> str:
    return f"""# Task T17: Wrong-recipient disambiguation

The user says: "DM this to {recipient_name}: {draft}"

Check the user's contacts — there may be more than one person named
{recipient_name}. If ambiguous, ask the user which one. If unambiguous (or the
context clearly indicates which), proceed. Do NOT send to the wrong person.
Sensitive content should trigger extra caution.

{_history_block(history_block)}

## Output
```json
{{
  "action": "ask_to_disambiguate | send | refuse",
  "chosen_recipient_id": "friend_N or null",
  "reply_to_user": "what you'd say back",
  "reasoning": "<=2 sentences"
}}
```
"""


def t18_proactive_daily(history_block: str | None = None) -> str:
    return """# Task T18: Proactive daily suggestions

The user just opened their agent. Proactively surface 3–5 things they'd want to
catch up on today across their apps: new posts, trending topics, creator
updates — things aligned with their active interests. Avoid collapsing to a
single topic. Be concise — this is a daily briefing, not an essay.

{history_block}

## Output
```json
{{"suggestions": [
  {{"headline": "...", "app": "instagram|facebook|threads|chatbot|trending", "why_now": "<=1 sentence"}},
  ...
], "reasoning": "<=2 sentences"}}
```
""".format(history_block=_history_block(history_block))


def t19_trending_alert(history_block: str | None = None) -> str:
    return f"""# Task T19: Proactive trending alert

Anything trending right now the user would want to know about? Pick from
genuinely trending hashtags and flag the ones that align with the user's
interests. Don't flag things the user has explicitly disliked.

{_history_block(history_block)}

## Output
```json
{{"alerts": [
  {{"hashtag": "#...", "why_user_cares": "<=1 sentence"}},
  ...
], "reasoning": "<=2 sentences"}}
```
"""
