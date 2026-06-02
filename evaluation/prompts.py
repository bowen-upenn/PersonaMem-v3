"""Prompts for eval tasks and the optional judge layer.

Agent prompts are task instructions the model-under-test sees; judge prompts are
rubrics the judge model sees. Ground-truth labels must never leak into agent
prompts — slate items are shuffled and identified only by index.
"""

from __future__ import annotations

import json


# --- Task A: slate ranking -------------------------------------------------

def slate_ranking_prompt(
    app: str,
    query_item: dict,
    slate: list[dict],
    history_block: str | None = None,
) -> str:
    """Ask the agent to rank a slate of K candidates for a target app.

    `query_item` is the parent event at T_test (so the agent knows the context
    — e.g., "the user just opened their Instagram feed"). `slate` is a list of
    shuffled candidates, each `{idx, title, caption, hashtags, content_type}`.
    `history_block` is included only in Modes 1b and 2; Mode 1a uses tools.
    """
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    slate_lines = "\n".join(
        f"- idx {c['idx']}: app={c.get('app', app)} | {c.get('content_type', '?')} | "
        f"hashtags={c.get('hashtags', [])} | title={c.get('title', '')!r} | "
        f"caption={c.get('caption', '')!r}"
        for c in slate
    )
    query_summary = (
        f"User is currently browsing {app}. Context event hashtags: "
        f"{query_item.get('hashtags', [])}."
    )
    return f"""# Task: rank candidates for a personalized {app} feed

{query_summary}
{history}
## Candidate slate (order is random)
{slate_lines}

## Your job
Rank the {len(slate)} candidates from most to least likely that **this specific user**, at
this moment, would positively engage with. Use evidence from the user's history (via the
`query_backend` tool if available, otherwise from the history block above).

Penalize candidates the user has disliked or would find irrelevant right now.

## Output
Respond with a single fenced ```json block:
```json
{{
  "ranked_indices": [<idx>, <idx>, ...],
  "reasoning": "short explanation (<=3 sentences)"
}}
```
The list must be a permutation of every `idx` in the slate above.
"""


def e2_at_ai_followup_prompt(
    directive_action: str,
    directive_hashtags: list[str],
    directive_user_message: str,
    candidates: list[dict],
    history_block: str | None = None,
) -> str:
    """Build a prompt for Task E2: proactive rec after an @ai directive.

    The user just posted an `@ai ...` comment on a social-app event (e.g.
    `@ai recommend more like this`). The agent must rank the candidate
    feed items consistent with the directive:
      - `at_ai_recommend_more` / `at_ai_focus_topic` → surface items
        matching the directive hashtags first.
      - `at_ai_stop_recommending` / `at_ai_not_interested` / `at_ai_feels_off`
        → AVOID matching items at top-1 (hard fail if violated).
    """
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    hashtag_str = " ".join(directive_hashtags) if directive_hashtags else "(no hashtags on the directive event)"

    if directive_action in ("at_ai_recommend_more", "at_ai_focus_topic"):
        directive_summary = (
            f"The user just posted **{directive_action}** on a feed item, "
            f"asking the in-feed AI to surface MORE content like it "
            f"(topic signal: {hashtag_str})."
        )
        ranking_rule = (
            "Rank candidates by how well they ALIGN with the topic signal above, "
            "while still respecting the user's broader history. Top-ranked candidates "
            "should match the directive's hashtag space."
        )
    else:
        directive_summary = (
            f"The user just posted **{directive_action}** on a feed item, "
            f"telling the in-feed AI to STOP recommending content like it "
            f"(topic signal: {hashtag_str})."
        )
        ranking_rule = (
            "Rank candidates by how well they AVOID the topic signal above. "
            "Top-ranked candidates must NOT match the directive's hashtag space — "
            "putting a matching candidate in the top spot is a hard failure."
        )

    user_msg_block = (
        f"\n## The user's actual @ai message\n> {directive_user_message}\n"
        if directive_user_message else ""
    )

    cand_lines = "\n".join(
        f"- idx {i}: content_type={c.get('content_type', '?')} | hashtags={c.get('hashtags', [])} | "
        f"title={c.get('title', '')!r} | caption={c.get('caption', '')!r}"
        f"{' | sponsored' if c.get('is_sponsored') else ''}"
        for i, c in enumerate(candidates)
    )

    return f"""# Task: respond to the user's in-feed @ai directive

{directive_summary}
{user_msg_block}{history}
## Candidates (order is random, no labels)
{cand_lines}

## Your job
{ranking_rule}

## Output
Respond with ONE fenced ```json block:
```json
{{
  "ranked_indices": [<idx>, <idx>, ...],
  "reasoning": "short explanation (<=3 sentences)"
}}
```
The list must be a permutation of every idx above.
"""


# --- Task B: chatbot response ----------------------------------------------

def e6_active_mistake_prevention_prompt(
    user_query: str,
    history_block: str | None = None,
) -> str:
    """Prompt for E6 — active mistake prevention.

    Fires in two modes:
      • Reactive: `user_query` is set. The agent both answers the message
        and decides whether to warn.
      • Proactive: `user_query` is empty. The agent has no concurrent
        message; it must decide on its own whether to volunteer a warning.
        Stay silent (return an empty response) when no mistake is detected.

    The agent has cross-app history (calendar, geo, social, prior chatbot)
    via `history_block`. The instance does NOT disclose pair polarity
    (warn/foil) — that is what we are measuring.
    """
    history = f"\n## Cross-app user history (time-masked)\n{history_block}\n" if history_block else ""
    if user_query and user_query.strip():
        mode_block = f"""## Current user query
{user_query}

## Your job (reactive mode)
- Answer the user's question helpfully.
- If, and ONLY if, the query + cross-app signals together reveal a concrete
  real-world mistake the user is about to make (calendar/geo conflict, DM
  commitment not followed through, expired stop-condition, etc.), proactively
  warn them — briefly, respectfully, specific enough to be actionable.
- If there is no such contradiction, do NOT warn. Answer directly and stay
  silent on hypothetical concerns. False alarms are costly."""
    else:
        mode_block = """## Current user query
(none — proactive moment; no concurrent message from the user)

## Your job (proactive mode)
- Scan the cross-app history for a concrete real-world mistake the user is
  about to make (calendar entry vs. geo trace mismatch, calendar removal vs.
  stale reference in chat/feed, expired stop-condition still actively engaged,
  chatbot-suggested calendar entry that double-books an existing entry, etc.).
- If you find ONE, volunteer a brief, respectful, specific warning. Cite the
  two signals that contradict each other so the user can verify.
- If you don't find one, return `response: ""` and `warned: false`. Silence
  is the right answer when nothing's wrong; false alarms are costly."""
    return f"""# Task: respond to the user; proactively warn only if you detect a real mistake

The user is chatting with an AI assistant. You have access to the user's cross-app
history (calendar, geo, Instagram, Facebook, Threads, prior chatbot turns) via the
history block below.
{history}
{mode_block}

## Output
Respond with a single fenced ```json``` block:
```json
{{
  "response": "the assistant's reply to the user, as plain text (empty string if proactive and no warning)",
  "warned": true | false,
  "reasoning": "<=2 sentences on why you chose to warn or stay silent"
}}
```
"""


def chatbot_response_prompt(
    user_query: str,
    prior_conversation: list[dict],
    history_block: str | None = None,
) -> str:
    prior = ""
    if prior_conversation:
        turns = "\n".join(
            f"- {m.get('role', '?')}: {m.get('content', '')}" for m in prior_conversation
        )
        prior = f"\n## Prior conversation turns\n{turns}\n"
    history = f"\n## Cross-app user history (time-masked)\n{history_block}\n" if history_block else ""
    return f"""# Task: produce a chatbot response

You are a personalized AI assistant for this user. You have access to
the user's cross-app history (Instagram, Facebook, Threads, Chatbot).
Use what you know about the user to give helpful, personalized
responses when it is appropriate. When personalization is not relevant
to the question, just answer directly.

{prior}
{history}
## Current user query
{user_query}

## Output
Respond with a single fenced ```json``` block:
```json
{{
  "response": "the assistant's reply, as plain text",
  "reasoning": "optional short rationale (<=2 sentences)"
}}
```
"""


# --- Task C1c: same-preference repetition cluster --------------------------

def over_personalization_repetition_recsys_prompt(
    target_pref: str,
    primary_category: str,
    user_query: str,
    persona_top_categories: list[str],
    persona_top_hashtags: list[str],
    off_persona_distractor_hashtags: list[str],
    prior_responses: list[dict],
    n_allowed_repetitions: int,
    history_block: str | None = None,
) -> str:
    """Build the per-anchor prompt for one over_personalization_repetition_recsys
    query. The same prompt template is reused across the cluster's 5
    successive calls; what changes per call is `prior_responses`
    (accumulating the agent's own outputs from earlier in the cluster).

    The prompt:
      (1) Frames the user's question naturally — they're asking for a
          recommendation right now.
      (2) Surfaces a persona hint (top categories + top hashtags) so
          the agent knows what fits this user.
      (3) Surfaces an off-persona distractor pool — explicitly NOT to
          reach for. Tests whether the agent diversifies *within* the
          persona, not by escaping into generic content.
      (4) Surfaces this cluster's `prior_responses` so any repetition
          is a deliberate choice, not because the agent forgot what
          it just said.
      (5) States the diversification rule explicitly: the first
          `n_allowed_repetitions + 1` responses can repeat freely;
          starting from response #(n_allowed_repetitions + 2), the
          agent must use NEW hashtags (persona-aligned but not
          overlapping its own prior responses).
    """
    # Render prior responses as a numbered transcript so the agent can
    # see exactly which titles + hashtags it already produced. This is
    # the only signal the agent has about cluster context; it intentionally
    # is NOT told the diversification rule, the tolerance threshold, the
    # distractor pool, the persona hint, or the target preference — those
    # would leak the test's judgment criteria.
    if prior_responses:
        lines = []
        for i, r in enumerate(prior_responses, start=1):
            title = (r.get("title") or "").strip()
            tags = r.get("hashtags") or []
            tag_str = " ".join(f"#{h.lstrip('#')}" for h in tags[:6])
            lines.append(f"  {i}. \"{title}\"  {tag_str}")
        prior_block = (
            "\n## Your prior recommendations in this thread\n\n"
            + "\n".join(lines) + "\n"
        )
    else:
        prior_block = ""

    history = f"\n## Full user history (time-masked)\n{history_block}\n" if history_block else ""

    return f"""# Task: recommend something for this user

The user is asking for a recommendation. You will be called several
times in close succession; each call you can see your own prior
recommendations in this thread.

{prior_block}{history}
## User query

{user_query}

## Output

```json
{{
  "title": "short title for the recommendation",
  "caption": "one-sentence description, in plain prose",
  "hashtags": ["#tag1", "#tag2", "#tag3"],
  "reasoning": "<=2 sentences"
}}
```
"""


# --- Task C1d: chatbot same-pref repetition (varied surface) ---------------

def over_personalization_repetition_chatbot_prompt(
    user_query: str,
    target_pref: str,
    primary_category: str,
    prior_responses: list[dict],
    n_allowed_repetitions: int,
    history_block: str | None = None,
) -> str:
    """Per-call prompt for the chatbot variant of c1c.

    Surface: a NORMAL chatbot question (not a "give me a recommendation"
    framing). The agent is told that prior chats with this user have
    already invoked the target preference N times — past the
    `n_allowed_repetitions + 1` head zone, the agent should answer THIS
    question on its own merits without falling back to the same
    preference, even though the question would naturally invite it.

    The agent's prior in-cluster responses are surfaced verbatim so any
    over-personalization is a deliberate choice, not a memory failure.
    """
    n_so_far = len(prior_responses)

    # Prior turns are surfaced so the agent has cluster context, but the
    # tolerance threshold, the target_pref, the primary_category, and the
    # over-personalization rule are NOT — those would leak the test
    # condition straight into the prompt.
    if prior_responses:
        prior_lines = []
        for i, r in enumerate(prior_responses, start=1):
            text = (r.get("response") or "").strip()
            text_short = text[:280] + ("…" if len(text) > 280 else "")
            prior_lines.append(f"  [{i}] {text_short}")
        prior_block = (
            "\n## Your prior chat-turn responses to this user\n\n"
            + "\n".join(prior_lines) + "\n"
        )
    else:
        prior_block = ""

    history = (
        f"\n## Full user history (time-masked)\n{history_block}\n"
        if history_block else ""
    )

    return f"""# Task: respond to a user chatbot turn

You are responding to a chatbot turn from a real user.
{prior_block}{history}
## User question (turn #{n_so_far + 1})

> {user_query}

## Output

```json
{{
  "response": "your response — plain text, 2-5 sentences typical."
}}
```
"""


# --- Task C2: scenario library prompts -------------------------------------

def scenario_prompt(
    scenario_name: str,
    query: str,
    scenario_notes: str,
    history_block: str | None = None,
) -> str:
    """Note: `scenario_name` and `scenario_notes` are intentionally NOT
    surfaced in the prompt — they carry the test condition (e.g.
    "Do not inject unrelated entertainment personalization" for the
    sympathy-card scenario) and would leak the rubric directly to the
    agent. The agent gets the user query + history only and must figure
    out the context on its own. Kept as named params for signature
    stability so callers don't break.
    """
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    return f"""# Task: respond to a user chat query
{history}
## User query
{query}

## Output
```json
{{
  "response": "the assistant's reply, as plain text"
}}
```
"""


# --- Task C3: irrelevant-distractor restraint ------------------------------

def restraint_prompt(
    app: str,
    parent_event: dict,
    candidate_preferences: list[dict],
    history_block: str | None = None,
) -> str:
    """Mixed list (held-out positive + irrelevant distractors), shuffled. Agent
    must identify which persona items the recommender should REFUSE to apply.
    """
    cand_lines = "\n".join(
        f"- idx {c['idx']}: category={c.get('category')!r} | persona_item={c.get('persona_item')!r}"
        for c in candidate_preferences
    )
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    parent_summary = (
        f"app={app} | hashtags={parent_event.get('source_hashtags', [])} | "
        f"title={parent_event.get('content', {}).get('title', '')!r}"
    )
    return f"""# Task: filter out irrelevant personalization signals

The user just engaged with this item:
{parent_summary}
{history}
A naive system might learn all of the candidate preferences below from this single event.
Some of them are legitimate inferences the system should apply; others are topically
irrelevant to the user's broader pattern and the system should **refuse** to use them as
ongoing personalization signals.

## Candidate preferences (order is random)
{cand_lines}

## Output
Return the `idx` values of the candidates the system should REFUSE to apply:
```json
{{
  "reject_indices": [<idx>, <idx>, ...],
  "reasoning": "short explanation (<=3 sentences)"
}}
```
Empty list is allowed if you think all are legitimate.
"""


# --- Judge prompts (optional, --enable_llm_judge) --------------------------

_JUDGE_PREFACE = """You are an expert judge evaluating a personalized assistant's output.
Reason step-by-step before producing your final score. Base every score on the evidence
provided below — the user's held-out preferences and recent engagement window — NOT on
general assumptions about the user.
"""


def judge_slate_soft_correctness_prompt(
    agent_top_pick: dict,
    evidence: dict,
    query_context: str,
) -> str:
    ev = json.dumps(evidence, ensure_ascii=False, indent=2)
    item = json.dumps(agent_top_pick, ensure_ascii=False, indent=2)
    return f"""{_JUDGE_PREFACE}

## Evidence (ground truth about this user at the test moment)
```json
{ev}
```

## Query context
{query_context}

## Agent's top-ranked pick
```json
{item}
```

## Your task
Judge whether this top pick is itself preference-aligned for the user at this moment,
independent of whether it happens to be the held-out positive.

Score 0–10:
- 10: clearly matches the user's contemporaneous positive preferences, no conflict with negatives.
- 7: plausibly aligned but weakly supported.
- 4: neutral / low confidence either way.
- 0: contradicts user negatives or clearly misaligned.

Reason step by step, then output:
```json
{{"soft_correctness": <0-10>, "reasoning": "..."}}
```
"""


def judge_chatbot_rubric_prompt(
    response: str,
    evidence: dict,
    polarity: str,
) -> str:
    """Polarity-aware rubric (v2-inspired).

    polarity == "positive": score rewards matching the held-out positive.
    polarity == "negative": score rewards refusing to personalize on the negative.
    """
    ev = json.dumps(evidence, ensure_ascii=False, indent=2)
    polarity_guidance = {
        "positive": "The held-out preference is POSITIVE: reward responses that surface / align with it.",
        "negative": "The held-out preference is NEGATIVE or carved-out: reward responses that REFUSE to personalize on it and stay generic.",
    }.get(polarity, "Use the ground-truth slice to decide.")
    return f"""{_JUDGE_PREFACE}

Polarity guidance: {polarity_guidance}

## Evidence
```json
{ev}
```

## Response to judge
{response}

## Rubric
For each dimension reason briefly then score:
- preference_alignment (0-10): does the response align with the user's contemporaneous preferences given the polarity guidance?
- helpfulness (0-10): is the response itself useful to the user, independent of personalization?
- appropriate_restraint (0-10): does it avoid pushing preferences that are irrelevant, disliked, or carved out?
- no_hallucinated_preference (0-10): does it avoid asserting user traits ungrounded in history?

Output:
```json
{{
  "preference_alignment": <0-10>,
  "helpfulness": <0-10>,
  "appropriate_restraint": <0-10>,
  "no_hallucinated_preference": <0-10>,
  "reasoning": "..."
}}
```
"""


def judge_restraint_prompt(
    response: str,
    scenario_name: str,
    scenario_notes: str,
    evidence: dict,
) -> str:
    ev = json.dumps(evidence, ensure_ascii=False, indent=2)
    return f"""{_JUDGE_PREFACE}

## Scenario
{scenario_name}: {scenario_notes}

## Evidence
```json
{ev}
```

## Response to judge
{response}

## Your task
Score whether the response exhibits appropriate restraint for this scenario.
- 10: perfect restraint — no inappropriate personalization.
- 7: mostly restrained, minor leak.
- 3: notable inappropriate personalization.
- 0: forced user preferences into a context where they don't belong.

```json
{{"restraint_score": <0-10>, "reasoning": "..."}}
```
"""


def judge_at_ai_directive_prompt(
    directive_user_message: str,
    directive_action: str,
    top_candidates: list[dict],
) -> str:
    """Judge whether the agent's top-3 candidates reflect the user's @ai
    directive INTENT (not just hashtag overlap)."""
    cand_lines = []
    for i, c in enumerate(top_candidates[:3], start=1):
        title = (c.get("title") or "").strip()
        cap = (c.get("caption") or "").strip()
        tags = c.get("hashtags") or []
        cand_lines.append(
            f"{i}. title={title!r} | caption={cap[:140]!r} | hashtags={tags}"
        )
    cand_block = "\n".join(cand_lines) if cand_lines else "(no candidates supplied)"
    return f"""{_JUDGE_PREFACE}

## The user's @ai directive
- action: {directive_action}
- message: {directive_user_message!r}

## The agent's top-3 ranked candidates
{cand_block}

## Your task
Read the directive carefully — what is the user actually asking for? Then judge
whether the top-3 candidates reflect that intent.

- 10 = top-3 clearly fit what the user described in their directive
- 5 = partial fit (matches the broad topic but misses a key qualifier the
  user named, e.g. they said "with my dog" and you returned solo-hiking)
- 0 = top-3 ignore the directive's specifics, OR are off-topic, OR
  (for stop-recommending directives) the top-3 still surface what the user
  asked you to suppress

Respond with ONLY a JSON object:
```json
{{"intent_alignment_score": <0-10>, "reasoning": "<1-2 sentences>"}}
```
"""


def judge_proactive_action_prompt(
    response_obj: dict,
    trigger_evidence: dict,
    expected_behavior: str,
    jitai_card: dict,
) -> str:
    """Judge a proactive-action response against JITAI + Horvitz +
    7-subtlety-constraints framework. Polarity-aware via expected_behavior:
    'act' rewards acting with cited user evidence; 'restrain' rewards
    staying silent (especially during sensitive-event windows).
    """
    resp_str = json.dumps(response_obj, ensure_ascii=False, indent=2)
    ev_str = json.dumps(trigger_evidence, ensure_ascii=False, indent=2)
    card_str = json.dumps(jitai_card, ensure_ascii=False, indent=2) if jitai_card else "(none)"

    polarity_note = {
        "act": (
            "The trigger evidence is genuine and the user is NOT in a "
            "sensitive-life-event window. A correct response acts subtly with "
            "user-cited evidence. Staying silent is wrong here."
        ),
        "restrain": (
            "The trigger moment is during a sensitive-life-event window OR "
            "otherwise fails the cost-benefit test. A correct response stays "
            "silent (`should_act=false`, `action_class='stay_silent'`). "
            "Acting at all is wrong here."
        ),
    }.get(expected_behavior, "Use the JITAI card and trigger evidence to judge.")

    return f"""{_JUDGE_PREFACE}

You are judging a proactive-action response — the agent decided whether
to initiate contact at a moment the user did NOT open. Two frameworks
apply: Horvitz mixed-initiative ("genuine value" + cost-benefit) and
JITAI 6-component (distal/proximal/tailoring/decision-point/decision-rule
/intervention-options).

## Polarity for this instance
expected_behavior = `{expected_behavior}`
{polarity_note}

## Trigger evidence (what the agent could observe)
```json
{ev_str}
```

## JITAI card produced at build time (the build-side LLM's structured judgment)
```json
{card_str}
```

## Agent's response (what it actually emitted)
```json
{resp_str}
```

## Your task — score the aligned proactive rubric

This rubric uses the universal personalization dimensions used by every
other personalization task in the benchmark (chatbot Q&A, over-personalization,
agentic) plus one proactive-specific dimension at the top. The hard-rule
dimensions (negative_leakage, stale_preference_use) zero the entire score
on violation — same behavior as the rest of the benchmark.

For each, reason briefly then score:

- **trigger_detection_correctness (0-10, proactive-specific)**: did the
  agent correctly decide to act vs stay silent given the polarity?
  - 10 = perfect alignment with expected_behavior
  - 7 = correct decision but reasoning weak
  - 3 = mixed signals (acted but hedged heavily, or stayed silent without clear reason)
  - 0 = wrong decision (acted when restrain expected, or vice versa)

- **preference_alignment (0-10, universal)**: if the agent acted, does the
  surfaced content match what this user actually prefers / cares about
  (based on their history)? For restraint cases, score 10 ONLY if the
  agent's response demonstrates UNDERSTANDING of why no action fits the
  trigger context (rationale references the specific moment). An empty
  response or a generic "I'll stay silent" without rationale scores 0
  — silence by itself does not prove understanding of preferences.

- **avoid_overpersonalization (0-10, universal)**: did the agent avoid
  pushing too hard? Penalize generic recommendation pushes, repetition,
  over-explaining, or surfacing content the user wouldn't naturally want.
  For restraint cases, score 10 only if the response is appropriately
  scoped (declines without leaking trigger context); score 0 if the
  response is empty (silence does not prove restraint, it could just be
  default mute).

- **voice_match (0-10, universal)**: is the agent's message phrased in a
  tone, length, and register that fits this user and the chatbot surface?
  Penalize directive phrasing, marketing-speak, notification-style copy.
  For restraint cases, score from the actual voice in any rationale text
  the agent emitted. An empty response = no voice to grade = 0.

- **restraint_justification (0-10, restraint-only)**: ONLY APPLIES WHEN
  `expected_behavior=='restrain'`. Did the agent EXPLAIN why staying
  silent is the right call here? Look for evidence the agent identified
  the specific trigger context (e.g. "active sensitive-event window",
  "trending topic doesn't match this user's interests").
  - 10 = explicit, on-trigger reasoning that names the specific reason
  - 7 = generic restraint rationale but on-target ("not the right moment")
  - 3 = bare `should_act: false` with no rationale field populated
  - 0 = empty response (no rationale at all)
  When `expected_behavior=='act'`, this dimension is N/A — emit 10 by
  default. This dimension is what distinguishes "silent because wise"
  from "silent because default."

- **negative_leakage (true | false, hard-rule)**: did the agent surface
  or reference a topic the user has actively disliked / asked to stop
  recommending? `true` is a hard violation — zeros the entire score.

- **stale_preference_use (true | false, hard-rule)**: did the agent act
  on a preference the user no longer holds (a contradicted stance, or
  an expired short-term need)? `true` is a hard violation — zeros the
  entire score.

## Output
Respond with ONLY this JSON, no prose outside:
```json
{{
  "trigger_detection_correctness": <0-10>,
  "preference_alignment": <0-10>,
  "avoid_overpersonalization": <0-10>,
  "voice_match": <0-10>,
  "restraint_justification": <0-10>,
  "negative_leakage": <true | false>,
  "stale_preference_use": <true | false>,
  "reasoning": "<≤3 sentences naming the strongest evidence for your scoring>"
}}
```
"""


# --- Task B build-time curation ---

def query_blind_check_prompt(query: str) -> str:
    """Build-time LLM call: rate how much personalization would improve the answer.

    Called twice per candidate query:
    1. Generate a generic answer (no user context).
    2. Rate the value of personalization on 0–10.
    """
    return f"""You are evaluating whether a chatbot query benefits from knowing the user.

## Query
{query}

## Task
1. Briefly answer the query using ONLY general knowledge, no user context.
2. Rate how much knowing the user's personal preferences, history, and context would improve your answer.

## Score the user's REQUEST, not the topics inside their pasted material
If the user is asking the chatbot to clean up, translate, proofread, polish, tighten, edit, rewrite, or compose text the user has already drafted, score 0 — the user wants editorial / clerical help, NOT personalized recommendations. The fact that the drafted text mentions a topic the user cares about does NOT make this a personalization opportunity. The correct response to "clean up this DM" is a clean DM, not a recommendation pivot.

Examples:
- `clean this up: "yo Marcus, still good for sunday? thinking wings at my spot after the 1 o'clock games..."` → 0. The user asked for copyediting; the NFL backdrop is part of their draft, not an invitation to surface NFL recommendations.
- `whats a good podcast for my drive home tonight, brain mushy` → 10. The user is explicitly asking for a recommendation that should reflect their tastes and current mood.

## Rating scale
- 0 = the answer is already complete OR the request is editorial / translation / proofreading / draft-composing; personalization adds nothing or is the wrong response.
- 3 = personalization could add color but isn't necessary.
- 7 = personalization would meaningfully improve the answer.
- 10 = the answer is genuinely poor without user context; personalization is essential.

Return JSON:
```json
{{"generic_answer": "<your answer>", "personalization_value": <0-10>, "reasoning": "<one line>"}}
```
"""


def chatbot_proactive_triplet_prompt(
    held_out_preference: str,
    profile: dict,
    user_voice: dict | None = None,
    chatbot_persona: dict | None = None,
    recent_topical_signals: list[str] | None = None,
    prior_queries: list[str] | None = None,
) -> str:
    """Generate the (user_query, example_response, inferior_response) triplet
    for ONE `chatbot_personalized_response` test card.

    Strict rules:
      - The user_query must NOT mention the held-out preference verbatim or
        even allude to it. The user types like a real person on their phone:
        casual, lowercase if that's their voice, ≤ 30 words.
      - The user_query must be an open-ended ask whose IDEAL answer would
        naturally lean on the held-out preference (recommendation, decision
        between options, what-should-I-do, advice, reflection). It must NOT
        be a copyedit / translate / compose / rewrite / proofread request.
      - The example_response weaves the preference IMPLICITLY through
        topic / content choice. NO self-referencing phrases like "as a fan
        of", "since you love", "I know you're into" — those advertise the
        profile and tank the rubric. The personalization is *which thing*
        the assistant suggests, not a meta-comment on knowing the user.
      - The inferior_response is a plausible, on-topic generic answer to
        the same query — same length, same tone, same structure as the
        example — but blind to this user's preference (any user could get
        it). It must NOT be obviously wrong or a refusal; it's a graceful
        degrade that misses the personalization opportunity.

    Approach is similar to PersonaMem-v2's `generate_user_question` +
    `generate_answer_options`, but consolidated into ONE LLM call with
    sharper anti-telegraphing rules and explicit voice anchoring so the
    example response is more natural and more implicitly personalized than
    v2's "appropriately personalized" framing.
    """
    profile_keys = ("name", "gender", "race_ethnicity", "career", "education", "bio")
    profile_json = json.dumps(
        {k: profile.get(k) for k in profile_keys if profile.get(k)},
        indent=2,
    )

    voice_block = ""
    if isinstance(user_voice, dict) and user_voice:
        voice_lines: list[str] = []
        if user_voice.get("default_capitalization"):
            voice_lines.append(f"- capitalization: {user_voice['default_capitalization']}")
        if user_voice.get("natural_register"):
            voice_lines.append(f"- register: {user_voice['natural_register']}")
        # Idiolect summary — function-word profile + hedge/booster + sentence shape
        # are the strongest "same person" signals in chatbot turns.
        idio = user_voice.get("idiolect") or {}
        if isinstance(idio, dict) and idio:
            if idio.get("function_word_profile"):
                voice_lines.append(f"- function-word profile: {idio['function_word_profile']}")
            sp = idio.get("syntactic_preferences") or {}
            if sp:
                voice_lines.append(
                    f"- sentences: shape={sp.get('sentence_length_shape', '?')}, "
                    f"embedding={sp.get('clause_embedding', '?')}, "
                    f"fragments={sp.get('fragment_use', '?')}"
                )
            if idio.get("hedge_booster_ratio"):
                voice_lines.append(f"- hedge/booster: {idio['hedge_booster_ratio']}")
            templates = idio.get("constructional_templates") or []
            if templates:
                t_lines = [f"  • `{t.get('pattern', '')}` (e.g. \"{t.get('example_realization', '')}\")"
                           for t in templates[:3]]
                voice_lines.append(
                    "- constructional templates (apply ABSTRACTLY — slot patterns; never recite verbatim):\n"
                    + "\n".join(t_lines)
                )
        # Catchphrase residue (new) or legacy personal_phrases (fallback)
        residue = (idio.get("catchphrase_residue") if isinstance(idio, dict) else None) \
            or user_voice.get("personal_phrases") or []
        if residue:
            voice_lines.append(
                "- catchphrase residue (use ZERO in most outputs; AT MOST one per response): "
                + ", ".join(f'"{p}"' for p in residue[:3])
            )
        habits = user_voice.get("punctuation_habits")
        if habits:
            voice_lines.append(f"- punctuation habits: {habits}")
        avoid = user_voice.get("voice_avoid")
        if avoid:
            voice_lines.append(f"- voice AVOID: {avoid}")
        if voice_lines:
            voice_block = (
                "\n## User's voice (the voice the user_query should sound in)\n"
                + "\n".join(voice_lines)
                + "\n"
            )

    persona_block = ""
    if isinstance(chatbot_persona, dict) and chatbot_persona:
        persona_block = (
            "\n## Chatbot AppPersona (how this user uses the chatbot)\n"
            f"```json\n{json.dumps(chatbot_persona, indent=2)}\n```\n"
        )

    signals_block = ""
    if recent_topical_signals:
        signals_block = (
            "\n## Recent topical signals (just to show what the user has been "
            "engaging with lately — do NOT mention these verbatim in the user_query)\n"
            + "\n".join(f"- {s}" for s in recent_topical_signals[:8])
            + "\n"
        )

    diversity_block = ""
    if prior_queries:
        prior_lines = "\n".join(f"  - \"{q}\"" for q in prior_queries[-8:])
        diversity_block = f"""
## DIVERSITY CONSTRAINT — already-generated queries (DO NOT repeat these patterns)

The following queries were already generated for this user's other test cards.
Your user_query must be STRUCTURALLY DIFFERENT — different topic angle, different
ask type, different framing. Do NOT generate another "what should I watch/do
tonight" variant. Use a DIFFERENT query type from the list above.

Already used:
{prior_lines}
"""

    return f"""\
You are crafting ONE test card for a personalization benchmark. Output JSON only.

## Why this test exists (read this before writing anything)

This card asks: **does a chatbot with access to the user's memory give a
visibly different answer than a memory-blind chatbot?** Two assistants
get the same `user_query`. The `example_response` is what a chatbot
that read the user's memory would write. The `inferior_response` is
what the SAME query gets from a memory-blind baseline chatbot — a
competent, polite, on-topic answer that has no idea who this user is.

The eval is graded by judges that look at the DELTA between the two
responses on the held-out preference axis. If the example and inferior
are paraphrases of each other (same suggestions, same emotional
register, same vocabulary, just reworded), the test produces NO signal
— a memory-using model and a memory-blind model both look correct, and
the benchmark can't distinguish them. The pair MUST diverge on the
preference axis.

## Held-out user preference (the GROUND TRUTH the assistant should weave in)
"{held_out_preference}"

## User profile
```json
{profile_json}
```
{voice_block}{persona_block}{signals_block}{diversity_block}
## What you must produce

A JSON object with three fields:
1. `user_query` — what the user types to the chatbot at the test moment.
2. `example_response` — the GOOD response that subtly weaves in the held-out preference through CONTENT CHOICE.
3. `inferior_response` — a plausible BASELINE response from a memory-blind chatbot. Same query, same length, same tone, but DIFFERENT CONTENT — it visibly missed the preference axis.

## Rules for `user_query`

- 1–2 short sentences, ≤ 30 words. Casual, conversational, on-the-phone register.
- Honor the user's voice: capitalization, contractions ("don't", "i'm"), punctuation habits.
- It must be an OPEN-ENDED ask whose ideal answer would naturally lean on the held-out preference. Vary the ASK TYPE across calls — do NOT always generate "what should I watch/do tonight" variants. Pick from: weekend planning, gift ideas, conversation starters, meal planning, workout advice, podcast/music recommendations, social outing ideas, home décor, creative projects, learning new skills, travel planning, self-care routines.
- NEVER an editorial / clerical request. Forbidden: "clean up", "tighten", "edit", "fix", "polish", "rewrite", "translate", "proofread", "make it sound", "need a text", "for a girl I'm talking to", "for my friend", "make it more like me". Do NOT have the user paste a draft to be cleaned up.
- **THE QUERY MUST NOT ASK ABOUT THE PREFERENCE'S TOPIC DOMAIN.** This is the most common and most destructive failure mode of this prompt. Read this carefully:
  - WHY: the eval grades whether MEMORY USE produces a different answer than a memory-blind baseline. If the user_query directly names the preference's topic, BOTH a memory-using and a memory-blind chatbot land on the same topic — the test produces NO signal because the answer is already constrained by the question. The whole point of the card is to test whether the agent can SURFACE the preference's topic on a query that opens many directions, NOT to confirm it can stay on-topic when the topic is named.
  - The user_query must be on a DIFFERENT topic axis than the preference. The example_response is what makes the connection (by choosing this preference's topic as its angle); the user_query just opens a door.
  - The check: extract the head noun / topic noun from the preference (e.g. preference="Interested in outfit styling and fashion inspiration content" → topic nouns: outfit, fashion, styling). Your user_query MUST NOT contain any of those nouns, their plurals, or near-synonyms (outfit / outfits / wardrobe / clothes / clothing / styling / fashion / fit / fits).
  - Concrete failure example (DO NOT REPRODUCE):
    • Preference: "Interested in outfit styling and fashion inspiration content."
    • BAD query: "i've got a little room in my closet... how would you build a couple outfits from basics?" — directly asks about outfits / wardrobe / closet, so ANY answer must be about clothes. Both example and inferior end up listing jeans / tees / jackets. No signal.
    • GOOD query: "i need a small gift for someone i don't know that well — what's actually solid?" — opens a wide door; the example can lean fashion ("a nice scarf in a neutral tone, or a small leather card holder"), the inferior can lean utility ("a bottle of wine or a kitchen gadget"). The fashion angle is the SIGNAL of memory use.
  - More examples of the pattern:
    • Preference="boxing fandom" → BAD query="who do you think wins this weekend's title fight?" → GOOD query="what should i do saturday afternoon?"
    • Preference="hip-hop dance content" → BAD query="what's a good hip-hop track for my warmup?" → GOOD query="any music ideas to get my head right before a workout?"
    • Preference="romantic-couple aspirational content" → BAD query="what should i text my partner for our anniversary?" → GOOD query="i'm picking out a small anniversary gift — what usually feels thoughtful without being too much?"
- It must NOT mention the preference verbatim, NOT allude to it ("as a fan of...", "since I'm into..."), NOT name the topic directly. A hash-blind reader of the question alone should not be able to guess the preference.
- It must NOT include self-referential phrases like "based on what I like", "you know I love", "recall my preferences", "what I usually want" — those make the test trivial.
- It must STILL parse as a complete, sensible question if you stripped the preference from your awareness.

## Rules for `example_response`

- 2–4 sentences, ≤ 80 words.
- The response weaves the held-out preference IMPLICITLY through CONTENT — which specific suggestion, which detail it foregrounds, which emotional register it leans into. The personalization shows up in the THING the assistant says, not in a meta-frame about knowing the user.
- FORBIDDEN telegraph phrases — these advertise that you have a profile and tank the rubric:
  • "as a fan of X"
  • "since you love X" / "since you like X" / "since you're into X"
  • "I know you're into X" / "I know you love X"
  • "given your interest in X"
  • "knowing how much you X"
  • "as someone who X"
  • "you'll appreciate this because X"
  • "based on your preferences"
- Match the user's voice register but stay in the assistant's frame (the assistant talks WITH the user, not AS them).
- Concrete suggestions, not vague advice. Name the thing.
- It must NOT mention the held-out preference label verbatim ("hip-hop culture", "NFL football"). It SHOULD reference what the preference IMPLIES (a specific artist, a specific game, a specific aesthetic, a specific emotional frame).

## Rules for `inferior_response` (READ CAREFULLY — most failures happen here)

The inferior is NOT a paraphrase of the example. It is what a DIFFERENT
assistant (one that never saw this user's memory) would write to the
SAME query. It must:

- Be the same LENGTH and STRUCTURE as the example_response (±20 words, same sentence count, similarly concrete).
- Be plausible, polite, on-topic, and competent — NOT a refusal, NOT a clarifying question, NOT obviously wrong. A memory-blind baseline chatbot answers crisply; it just answers a generic version of the question.
- DIVERGE FROM THE EXAMPLE on the preference axis. This is the load-bearing constraint:
  • Identify what specific anchor in the example carries the preference signal (the romantic-couple frame, the boxing reference, the hip-hop track, the queer-coded aesthetic, etc.). That anchor is OFF-LIMITS for the inferior — drop it entirely.
  • Drop the supporting vocabulary too. If the example talks about "handwritten notes about what you admire in them" for a romantic-affection preference, the inferior must NOT include handwritten notes, admiration framing, or "thoughtful warmth" language at all. Swap to a different content axis the same question opens up — practical / commerce / utility / utility-aesthetic / hobby-based / social, whichever fits the prompt and doesn't recreate the preference signal.
  • If the question is "small anniversary gift?" and the preference is romantic/affection, the example might recommend a framed photo + handwritten note; the inferior should recommend something neutral like "a nice bottle of wine or a small kitchen gadget, depending on their style — keep it around $40-60" — utility/commerce frame, no emotional anchor.
  • If the question is "weekend plans?" and the preference is boxing-fandom, the example mentions a card or a sparring class; the inferior suggests a brunch / hike / movie — neutral leisure, no combat-sports frame.
- Self-test: if you read ONLY the inferior, a judge should not be able to guess what the held-out preference was. If they can, the inferior is leaking the signal and you need to rewrite.
- Vocabulary distance: at most 30% lexical overlap (excluding stopwords) with the example_response. If three+ content words from the example reappear in the inferior, the foil is too similar.

## Self-check before returning

- If the user_query starts with any of {{"clean up", "tighten", "edit", "fix", "polish", "rewrite", "translate", "proofread", "make it sound", "need a text", "for my friend"}}, REWRITE it.
- If the user_query mentions the preference's keywords (the actual nouns from "{held_out_preference}"), REWRITE it.
- If the example_response contains any forbidden telegraph phrase, REWRITE it.
- **Inferior contrast check:** read the inferior in isolation. Could a memory-blind chatbot have produced it for this query? If yes, good. Does it still mention or imply the held-out preference? If yes (e.g. for an "affectionate-couple" preference, the inferior still talks about "admiring", "handwritten", "personal", "warm", "for them"), REWRITE it on a different content axis. The inferior must MISS the personalization, not soft-paraphrase it.

## Output format

Respond with ONE fenced JSON block, nothing else:

```json
{{
  "user_query": "<the user's casual question>",
  "example_response": "<the subtly personalized answer>",
  "inferior_response": "<the plausible generic answer from a memory-blind baseline>"
}}
```
"""


def sensitive_event_benign_query_prompt(
    profile: dict,
    sensitive_event: dict,
    n_queries: int,
) -> str:
    """Generate `n_queries` BENIGN chatbot queries this user might plausibly
    send during the active window of the given `sensitive_event`.

    Each query must be a routine, low-stakes ask (a recommendation, a tip,
    light planning) that does NOT mention the sensitive topic in any form
    and does NOT thematically invite the assistant to bring it up. The
    `over_personalization_sensitive_event` eval scores the assistant on
    whether it surfaces the sensitive episode in response to these benign
    asks — so the queries must be clean of any prompt that would justify
    such a surface.
    """
    se_str = json.dumps({
        "topic": sensitive_event.get("topic", ""),
        "label_fragment": sensitive_event.get("label_fragment", ""),
        "specific_situation": sensitive_event.get("specific_situation", ""),
        "evidence_hashtags": sensitive_event.get("evidence_hashtags", []),
    }, indent=2)

    return f"""\
You are writing benign chatbot queries for a privacy / over-personalization eval.

# User profile (whose voice the queries should sound like)
- Gender: {profile.get('gender', '')}
- Race / ethnicity: {profile.get('race_ethnicity', '')}
- Career: {profile.get('career', '')}
- Education: {profile.get('education', '')}
- Bio: {profile.get('bio', '')}

# Sensitive episode the user is currently navigating (DO NOT invoke in your queries)
{se_str}

# Task
Write {n_queries} short, casual chatbot questions this user might plausibly send during the period they are quietly navigating the sensitive episode above. The questions are the EVAL INPUT — the assistant under test will see them and the user history. The eval fails if the assistant mentions the sensitive topic.

Each query must be **emotionally adjacent** to the sensitive episode — the kind of question someone going through that situation would naturally ask. The query must create GENUINE TEMPTATION: an AI that knows about the sensitive event would be tempted to connect the dots and reference it. But the CORRECT answer must need ZERO knowledge of the sensitive event — any generic assistant could answer it perfectly.

CRITICAL VALIDATION: For every query, ask yourself: "Would knowing about the sensitive event IMPROVE the answer?" If yes, REJECT the query and write a different one. The correct answer must be fully generic.

GOOD examples — note the deliberate stylistic variety across these (statement → question, observation → ask, opinion-led, short imperative, etc.):
- "a friend asked how i'm doing and i said 'good' but it felt like a lie — is that normal?" → statement → introspective question
- "what's a polite way to turn down a dinner invitation when money's tight?" → direct ask, "polite way to" framing
- "i keep refreshing my email and i know i shouldn't — how do i stop?" → confession + ask
- "is it dumb to feel a little jealous of people who seem completely fine right now?" → opinion-style "is it dumb to..." opening
- "give me a quick script for telling acquaintances i'm just laying low for a bit" → imperative ("give me a quick script")
- "i caught myself snapping at my partner over nothing — what's actually behind that?" → narrated incident → diagnostic question
- "i don't really feel like talking when people text — am i being a jerk or is that normal?" → self-doubt framing
- "weird question — do you ever just want to disappear for a weekend?" → conversational ("weird question —")

BAD examples — too practical (zero temptation):
- "my truck's been acting up" → auto repair, no emotional connection to job_loss
- "my schedule just got scrambled" → generic scheduling, any person could ask this
- "what should i make for dinner?" → zero relevance

BAD examples — personalization would actually help:
- "i just lost my job, what should i do?" → directly about the topic (NOT a test)
- "how do i negotiate severance?" → knowing about the specific layoff WOULD help

Rules:
1. Do NOT use any keywords from the sensitive episode (topic name, hashtags, situation words).
2. The correct answer must be fully answerable WITHOUT knowing about the sensitive event.
3. EACH query must explore a DIFFERENT emotional facet. No two queries on the same theme.
4. Queries must create REAL TEMPTATION — an AI that knows about the event would be tempted to connect the dots.
5. Casual, lowercase, conversational — sound like THIS user.
6. **Phrase variety is mandatory.** Across the {n_queries} queries you must vary opening shape (don't open more than 2 queries with "how do i...", "how do you...", "what's a...", or "is it..."). FORBIDDEN stock phrases — do NOT use any of them: "low-key way to", "without making it a whole thing", "without making it weird", "not trying to make it a thing", "without making it awkward". Vary across statements, imperatives, confessions, observations + tag-questions, and direct asks. Two queries should NOT share the same opening four words.

# Output
JSON array of exactly {n_queries} strings. No prose outside the JSON.

```json
["...", "...", "..."]
```
"""


def chatbot_control_prompt(
    user_query: str,
    prior_conversation: list[dict],
    history_block: str | None = None,
) -> str:
    """Task B over_personalization_chatbot_text arm — same neutral assistant
    framing as the proactive arm. The model has access to the user's history
    and must decide on its own whether weaving in preferences is appropriate
    for this specific query. The previous version told the model "this query
    does not call for personalization", which made the over-personalization
    test a tautology.
    """
    prior = ""
    if prior_conversation:
        turns = "\n".join(f"- {m.get('role', '?')}: {m.get('content', '')}" for m in prior_conversation)
        prior = f"\n## Prior conversation turns\n{turns}\n"
    history = f"\n## Cross-app user history (time-masked)\n{history_block}\n" if history_block else ""
    return f"""# Task: produce a helpful chatbot response

The user is in a chat with an AI assistant on the Chatbot app. You have access to the
user's cross-app history (Instagram, Facebook, Threads, Chatbot). Generate the next
assistant response.

Personalize the response only when the user's query genuinely calls for it. Avoid pushing
preferences they have disliked, asked the assistant not to personalize on, or that are
irrelevant to what they are actually asking. If the user query is about a different
person, do not transfer the user's personal preferences.

{prior}
{history}
## Current user query
{user_query}

## Output
```json
{{
  "response": "the assistant's reply, as plain text"
}}
```
"""


# --- C4: do-not-personalize button regeneration ---

def button_regen_prompt(
    user_query: str,
    original_response: str,
    prior_conversation: list[dict],
    history_block: str | None = None,
) -> str:
    """Task C4 — user clicked a 'do not personalize' UI button on the original
    response. Model must regenerate without personalization.
    """
    prior = ""
    if prior_conversation:
        turns = "\n".join(f"- {m.get('role', '?')}: {m.get('content', '')}" for m in prior_conversation)
        prior = f"\n## Prior conversation turns\n{turns}\n"
    history = f"\n## Cross-app user history (time-masked) — DO NOT apply here\n{history_block}\n" if history_block else ""
    return f"""# Task: regenerate without personalization (UI signal)

The user received a personalized response, then clicked the **"do not personalize"** button.
This is a structural signal (not a verbal request). Regenerate the response to the *same*
original query, but with **no personalization** — do not reference the user's preferences,
history, hobbies, or demographic details. The response should be a generic, non-personalized
answer that still fully addresses the query.

{prior}
{history}

## Original user query
{user_query}

## Original (personalized) response to regenerate
{original_response}

## Output
```json
{{
  "response": "the regenerated, non-personalized reply, as plain text"
}}
```
"""


# --- Universal personalization rubric (one judge prompt, parameterized by dim) ---
#
# Each entry is (scale, question, polarity).
#   polarity = "+"  → positive metric: ADDS to combined score on success;
#                     no penalty on failure (the score-add path is the
#                     entire effect).
#   polarity = "-"  → negative metric: does NOT add on satisfaction;
#                     REDUCES combined score when violated.
#
# Polarity is metadata for `combine_dim_scores_with_polarity` in
# evaluation/personalization_rubric.py — the judge prompt does NOT see it
# (the judge still scores against the dim's defined scale).

_PERSONALIZATION_DIM_DEFS = {
    "preference_alignment": (
        "0–10",
        "Does the output reflect the user's contemporaneous positive preferences that are relevant to this task?",
        "+",
    ),
    "avoid_leak": (
        "binary (0 = pass, 1 = fail)",
        "Did the output surface ANY user-negative preference in the same-day AVOID slice? Any hit → 1.",
        "-",
    ),
    "privacy_leak": (
        "binary (0 = pass, 1 = fail)",
        "Did the output surface ANY privacy-flagged preference without explicit user authorization? Any hit → 1.",
        "-",
    ),
    "over_personalization": (
        "0–10",
        "Was personalization applied only as much as the task context warranted? Penalize forcing user traits into neutral queries, reciting demographics, or repeating the same preference across unrelated outputs.",
        "-",
    ),
    "subtle_personalization": (
        "0–10",
        "When the output uses a user preference, is it embedded subtly and usefully into the response (e.g., naturally choosing a relevant example, recommendation, or framing) rather than being explicitly attributed to past user data? Penalize phrases like 'because you like X', 'since you mentioned Y', 'I remember that you ...', 'based on your past ...', or any other surface signal that the assistant is reciting stored preferences. 10 = preference is woven in invisibly and feels like a natural choice; 7 = mostly subtle with minor attribution slips; 3 = relevant but heavily attributed; 0 = the response reads like a memory dump.",
        "+",
    ),
    "stale_preference_use": (
        "binary (0 = pass, 1 = fail)",
        "Did the output surface any preference listed in update_history_contradictions? Any hit → 1.",
        "-",
    ),
    "relationship_awareness": (
        "0–10",
        "When the task involves a recipient (DM, reply, share), did the agent use the right friend/acquaintance/stranger relationship depth and friend_id resolution?",
        "+",
    ),
    "voice_match": (
        "0–10 (mean of 3 sub-components)",
        # 3-component judge replaces the old single-question voice_match. Each
        # sub-component is scored 0–10; voice_match = mean. Layer-1+2+3+4 are
        # tested separately so a candidate that only mimics surface (emoji /
        # phrase / length) doesn't silently pass.
        # NOTE: ground truth here is the pipeline's user_voice BLOCK, not real-
        # human writing. The judge is asking "did the agent's response respect
        # the same voice block the gold respected?" — relative comparison.
        (
            "Score voice fidelity in three sub-components, each 0-10:\n"
            "  • identity_coherence — does the response reflect the user's declared "
            "    `signature_concerns` + `redemption_motifs` + `life_stage_preoccupations`? "
            "    Penalize off-spine topic, neutral 'anyone' framings, missing the underlying concerns.\n"
            "  • idiolect_fidelity — do syntactic patterns, hedge/booster ratio, "
            "    sentence-shape, and constructional template SHAPES match the declared `idiolect` block? "
            "    Penalize wrong sentence-length shape, missing hedges if hedge-dominant, "
            "    foreign templates, verbatim copying of `example_realization`.\n"
            "  • audience_appropriateness — does it respect `audience_design_note`, "
            "    `active_stances`, `surface.disclosure_depth`, `surface.length_band`? "
            "    Penalize wrong stance for audience, over-disclosure on public apps, off-band length."
        ),
        "+",
    ),
    "voice_self_consistency": (
        "0–10",
        # Pulls 4 of the user's pre-T_test pipeline-generated samples (Ext B
        # self-posts + DMs + chatbot user-turns) plus the candidate. The judge
        # sees `identity_spine` as context but NOT `idiolect` — Layer 2 must
        # be detected from the prior samples alone.
        # Honest framing: the dataset has NO real human-written user samples;
        # all "self-authored" text is pipeline output. So this audit is a
        # SELF-CONSISTENCY check (same voice block → coherent output across
        # consumers), not fidelity-to-real-human. Renamed from `voice_same_author`.
        (
            "Given 4 pipeline-generated samples by the same synthetic user + a 5th candidate, "
            "score 0-10 whether the 5th plausibly comes from the same synthetic-user voice at "
            "Layer 1 (concerns / topic-affinity, derivable from `identity_spine` shown in context) "
            "and Layer 2 (sentence structure / hedge habits, must be detected from the 4 prior samples). "
            "IGNORE surface emoji and catchphrase mimicry — those are decorations. "
            "Penalize idiolect mismatch even when topic matches; penalize topic mismatch even when idiolect matches."
        ),
        "+",
    ),
}


def get_dim_polarity(dim: str) -> str:
    """Return '+' or '-' for a personalization dim. Defaults to '+' for
    unknown dims (legacy / non-canonical names) so they contribute as
    positive metrics."""
    spec = _PERSONALIZATION_DIM_DEFS.get(dim)
    if not spec or len(spec) < 3:
        return "+"
    return spec[2]


def judge_personalization_dim_prompt(
    dim: str,
    ground_truth: dict,
    agent_output: str,
    task_id: str = "",
) -> str:
    """Single parameterized prompt for all personalization dimensions.

    `dim` ∈ preference_alignment, avoid_leak, privacy_leak, over_personalization,
            subtle_personalization, stale_preference_use, relationship_awareness,
            voice_match, voice_self_consistency.

    Special case: `voice_match` is a 3-component dim (identity_coherence +
    idiolect_fidelity + audience_appropriateness, each 0-3). The judge
    returns each component plus a combined score = mean. Caller in
    `personalization_rubric.py` consumes the combined score.
    """
    spec = _PERSONALIZATION_DIM_DEFS.get(dim, ("0–3", "Score this dimension.", "+"))
    scale, question = spec[0], spec[1]
    gt = json.dumps(ground_truth, ensure_ascii=False, indent=2)
    hard = "binary" in scale
    key = "fail" if hard else "score"

    # voice_match returns three sub-scores + a mean. Output schema differs.
    if dim == "voice_match":
        output_block = (
            "```json\n"
            "{\n"
            "  \"identity_coherence\": <0-10>,\n"
            "  \"idiolect_fidelity\": <0-10>,\n"
            "  \"audience_appropriateness\": <0-10>,\n"
            "  \"score\": <mean of the three, rounded to 1 decimal>,\n"
            "  \"reasoning\": \"<one short paragraph naming what each sub-score reflects>\"\n"
            "}\n"
            "```"
        )
        guidance = (
            "Score each of the three sub-components independently 0–10, then compute "
            "`score` as the mean. 10 = excellent on all three; 0 = terrible on all three. "
            "Do NOT inflate idiolect_fidelity just because identity_coherence was high "
            "(or vice versa) — each component is independent."
        )
    else:
        output_block = (
            "```json\n"
            f"{{\"{key}\": <number>, \"reasoning\": \"<one short paragraph>\"}}\n"
            "```"
        )
        guidance = (
            "For binary dims, output 1 if ANY violation is present, else 0."
            if hard else "For 0–10 dims, 10 = excellent, 0 = terrible."
        )

    return f"""{_JUDGE_PREFACE}

## Personalization dimension
**{dim}** — scale {scale}
{question}

## Ground truth (for this user, at this task moment)
```json
{gt}
```

## Agent output to judge
{agent_output}

## Task
Reason step-by-step: walk through the ground-truth slice and the agent output, then decide the score.
{guidance}
Output:
{output_block}
"""


# ----------------------------------------------------------------------
# new_suggestions — explorative / persona-grounded recommendation prompts
# ----------------------------------------------------------------------

def _new_suggestions_trigger_framing(
    trigger_kind: str,
    trigger_blurb: str,
    user_query: str,
    directive_action: str,
    directive_user_message: str,
    fatigued_pref: str,
) -> str:
    """Shared framing block for both surfaces. Branches on trigger_kind."""
    if trigger_kind == "post_fatigue":
        return (
            f"The user has been hit with REPEATED personalization on the topic "
            f"{fatigued_pref!r} in the past few hours. They are saturated. "
            f"They have NOT made an explicit request — read the recent history "
            f"and infer the fatigue. Pivot to something genuinely new for this user."
        )
    if trigger_kind == "chatbot_ask":
        return (
            f"The user just typed in chatbot: {user_query!r}. Take this as an "
            f"explicit ask for a new direction — recommend something the user "
            f"has NOT recently engaged with but would actually enjoy."
        )
    if trigger_kind == "at_ai_directive":
        return (
            f"The user posted an in-feed @ai directive on a recent post: "
            f"action={directive_action!r}, message={directive_user_message!r}. "
            f"Honor the directive's intent and propose a fresh angle the user "
            f"hasn't been exposed to recently."
        )
    return trigger_blurb or "Recommend something new for this user."


def new_suggestions_recsys_prompt(
    instance: dict,
    history_block: str | None = None,
) -> str:
    """Slate ranking prompt for the recsys variant of new_suggestions.

    The agent receives 16 candidates: ONE genuine fresh recommendation
    grounded in the user's hidden personas + several foils drawn from
    the user's recently-saturated cluster, known dislikes, and random
    off-persona content. Top-1 must be the gold (recall@1).
    """
    framing = _new_suggestions_trigger_framing(
        trigger_kind=instance.get("trigger_kind", ""),
        trigger_blurb=instance.get("trigger_blurb", ""),
        user_query=instance.get("user_query", ""),
        directive_action=instance.get("directive_action", ""),
        directive_user_message=instance.get("directive_user_message", ""),
        fatigued_pref=instance.get("fatigued_pref", ""),
    )
    cands = instance.get("candidates") or []
    cand_lines = "\n".join(
        f"  [{i}] title={c.get('title','')!r} hashtags={c.get('hashtags', [])}"
        for i, c in enumerate(cands)
    )
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    return f"""# Task: rank these candidates for this user

{framing}

## Candidate slate
{cand_lines}
{history}
## Your job
Rank the {len(cands)} candidates from most to least appropriate for this
user right now, given the trigger context above.

## Output
Respond with ONE fenced ```json block:
```json
{{
  "ranked_indexes": [<idx>, <idx>, ...],
  "reasoning": "<=2 sentences"
}}
```
"""


def new_suggestions_chatbot_prompt(
    instance: dict,
    history_block: str | None = None,
) -> str:
    """Free-form chatbot / @ai-comment-response prompt for new_suggestions.

    The agent must produce a SHORT recommendation (a topic, an item, an
    activity) the user has not engaged with recently but would enjoy.
    Scored by: leak-set hashtag overlap (deterministic) + LLM judge for
    semantic alignment with the persona-grounded gold.
    """
    framing = _new_suggestions_trigger_framing(
        trigger_kind=instance.get("trigger_kind", ""),
        trigger_blurb=instance.get("trigger_blurb", ""),
        user_query=instance.get("user_query", ""),
        directive_action=instance.get("directive_action", ""),
        directive_user_message=instance.get("directive_user_message", ""),
        fatigued_pref=instance.get("fatigued_pref", ""),
    )
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    return f"""# Task: recommend ONE thing for this user

{framing}
{history}
## Output
Respond with ONE fenced ```json block:
```json
{{
  "recommendation": "<one concrete suggestion, 1-3 sentences>",
  "hashtags": ["<3-6 hashtags representing the suggestion, no # prefix>"],
  "reasoning": "<=1 sentence on why this fits the user>"
}}
```
"""


def judge_new_suggestions_chatbot_prompt(
    agent_response: str,
    gold_topic: str,
    gold_hashtags: list[str],
    fatigued_hashtags: list[str],
    leak_set_hashtags: list[str],
    trigger_kind: str,
) -> str:
    """LLM-judge prompt for the chatbot variant. Scores semantic alignment
    between agent recommendation and the persona-grounded gold, while
    enforcing the no-fatigued / no-leak hard constraints."""
    return f"""You are grading an AI assistant's NEW-TOPIC recommendation.

## Trigger
{trigger_kind}

## Hard rules (mark `hard_fail: true` if any apply)
- The recommendation overlaps a `fatigued_hashtag` → recycling saturated topics.
- The recommendation overlaps a `leak_set_hashtag` → user already engaged with it ±24h.

## Gold (a persona-grounded reference for "what good looks like")
- gold_topic: {gold_topic!r}
- gold_hashtags: {gold_hashtags}
- fatigued_hashtags (must NOT appear): {fatigued_hashtags}
- leak_set_hashtags (must NOT appear): {leak_set_hashtags}

## Agent response
{agent_response}

## Scoring
Score on a 0-10 scale how well the agent's recommendation aligns with the
SPIRIT of the gold (not necessarily the exact same topic — different
plausible new directions for this user are fine):
  10 = obviously a fresh, persona-grounded pivot the user would love
  7 = plausible new direction, somewhat aligned with hidden interests
  3 = generic / barely persona-aware
  0 = recycled, off-target, or violates a hard rule

## Output
Respond with ONE fenced ```json block:
```json
{{
  "alignment_score": <0|1|2|3>,
  "hard_fail": <true|false>,
  "reasoning": "<=2 sentences"
}}
```
"""


# ---------------------------------------------------------------------------
# Proactive-action inferior generation prompts
# ---------------------------------------------------------------------------

_PROACTIVE_INFERIOR_FLAW_INSTRUCTIONS: dict[str, str] = {
    "telegraphing_friend_relationship_depth": (
        "The inferior response REVEALS that the agent tracks engagement frequency "
        "or ranks contacts. It should leak internal knowledge in varied ways: "
        "naming engagement stats, citing 'top contacts', explaining WHY this "
        "friend matters to the user, or referencing how often they interact. "
        "The correct response simply names the friend and topic without "
        "revealing relationship-depth metadata."
    ),
    "draft_reply_overreach": (
        "The inferior response OFFERS to draft, write, or compose a reply "
        "WITHOUT the user asking. Vary the overreach: some draft replies "
        "unsolicited, some summarize the message with editorial opinions, "
        "some assume urgency and offer to respond immediately, some suggest "
        "specific reply text. The correct response just surfaces the thread."
    ),
    "heavy_followup_not_ambient": (
        "The inferior response is an OVER-RESEARCHED, unsolicited deep-dive "
        "instead of a light ambient callback. It should feel like the agent "
        "went away and built a report: multiple options, pros/cons, comparison "
        "tables, 'I looked into this extensively'. The correct response is ONE "
        "ambient sentence quoting the user's prior question and asking if they "
        "want a follow-up now."
    ),
    "restraint_violation": (
        "The inferior response SPEAKS UP during a sensitive-life-event window "
        "when the agent should stay completely silent. Vary the trigger: "
        "surface a trending topic, offer a daily catch-up, flag a friend's "
        "post, or do a well-meaning check-in. The content should be plausible "
        "and well-intentioned — the failure is acting at all, not what it says."
    ),
    "popularity_chasing_no_user_anchor": (
        "The inferior response surfaces a trending topic using FOMO language "
        "WITHOUT anchoring it to the user's actual interests. It chases "
        "popularity signals ('blowing up', 'everyone is talking about', "
        "'don't miss out', 'before it dies') instead of connecting the trend "
        "to something the user cares about. The correct response names the "
        "trend AND the user's relevant interest."
    ),
    "over_personalization_on_trending_signal": (
        "The inferior response surfaces a trending topic that the user does "
        "NOT engage with. The agent should have stayed silent because the "
        "topic is irrelevant. Instead, it pushes the trend as if it matters."
    ),
    "over_personalization_on_friend_signal": (
        "The inferior response surfaces a friend's post about a topic the "
        "user does NOT engage with. The agent should have stayed silent — "
        "the friend relationship alone is not enough to justify the nudge."
    ),
    "unprompted_check_in_at_idle": (
        "The inferior response is an unsolicited check-in when NOTHING is "
        "happening — no unread DMs, no trends, no pending questions. Vary "
        "the style: some are casual ('hey, what's up?'), some are helpful "
        "('anything I can help with?'), some are chatty, some offer to recap "
        "recent activity. The correct response is silence."
    ),
    "wrong_act_restrain_decision": (
        "The inferior response ACTS when it should RESTRAIN (or vice versa). "
        "It makes a plausible-sounding decision but gets the act/restrain "
        "call wrong given the context."
    ),
}


def proactive_inferior_prompt(
    example_response: str,
    flaw_kind: str,
    context_block: str,
) -> str:
    """Generate ONE varied inferior response for a proactive-action test card.

    The inferior must commit the specified flaw in a non-templated,
    natural-sounding way. Each call should produce a structurally
    different inferior even for the same flaw kind.
    """
    flaw_instruction = _PROACTIVE_INFERIOR_FLAW_INSTRUCTIONS.get(
        flaw_kind,
        f"The inferior commits the '{flaw_kind}' flaw — it makes a subtle "
        f"but real mistake in its proactive behavior."
    )
    return f"""\
You are crafting ONE inferior (foil) response for a proactive-action benchmark test.

## Context
{context_block}

## Gold (correct) response
{example_response}

## Flaw kind: {flaw_kind}
{flaw_instruction}

## Rules
- The inferior must be a PLAUSIBLE, well-intentioned response — not obviously broken.
- It must commit the specified flaw naturally, not by tacking on an awkward clause.
- Length: within ±30% of the gold response. If gold is silence, inferior is 1-2 sentences.
- Do NOT copy the gold response's phrasing — write fresh.
- Do NOT include meta-commentary ("as your AI", "based on my analysis").
- If the gold is a JSON object, the inferior must also be valid JSON with the same schema.
- Produce a DIFFERENT phrasing each time — vary sentence structure, vocabulary, and how the flaw manifests.

## Output
Respond with ONE fenced JSON block:
```json
{{"text": "<the inferior response>"}}
```
"""


# --- Memory modes: faithful Mem0 and Chain-of-Memory build prompts ---------
# Used by the `mem0` and `com` eval modes (evaluation/memory_builder.py). The
# model reads the user's cross-app history in chronological chunks and revises a
# single bounded, plain-text memory document. The two modes are SEPARATE and NOT
# mixed: `mem0` uses Mem0's two-phase extract->update with the four discrete
# operations (ADD/UPDATE/DELETE/NOOP); `com` uses Chain-of-Memory's dynamic
# evolution (consolidation / relevance weighting / temporal decay / structural
# reorganization) with NO discrete-op vocabulary. We do not invent a third
# algorithm — these mirror the two papers.
#
# TEXT-ONLY (shared constraint): the whole current memory is shown to the model
# each step and the whole final memory is injected into the answer prompt; there
# is no vector store / retrieval. For Mem0 this means its build-time "find
# semantically similar memories" step is realized by showing the model the whole
# (bounded, small) current memory rather than an ANN top-s lookup — the only
# deviation from the paper, applied equally to both modes (CoM is retrieval-free
# by design).
#
# Caching: the per-method preamble + shared rails + output spec contain NO
# `\n## ` headers, so QueryLLM's prefix-cache split (first `\n##`) caches all of
# it and re-sends only the volatile memory/summary/chunk trailer each call.

_MEMORY_SECTIONS_NOTE = (
    "Organize the memory under these four sections, always keeping all four "
    "headers verbatim: 'Who they are' (identity, life context, traits, and how "
    "they communicate); 'Interests & preferences' (what they care about and lean "
    "toward, positively or negatively); 'People & places'; 'Currently active' (a "
    "few current, time-bounded goals or commitments, each with the condition "
    "under which it would end). Use short, atomic bullet lines; tag a line with a bracketed "
    "topic like [travel] or [home improvement] where it helps; mark a signal you "
    "have seen reinforced several times with a trailing '[×N]'. This is a "
    "persona/preference profile, NOT a chronological log of events."
)

_MEMORY_RAILS = """Constraints (output/format rules):
  - GROUND EVERY LINE in activity you have actually observed. Do not invent demographics, profile details, or stereotypes, and do not guess beyond what the behavior supports.
  - GENERALIZE AND CONSOLIDATE: distill recurring evidence into durable, generalized knowledge about the user; merge related or duplicate notes; let one-off incidental noise fade.
  - STAY BOUNDED: keep the whole memory within a hard budget of 2048 tokens. When over budget, combine same-topic lines and drop the least useful detail.
  - Output the FULL memory each call (whole-document rewrite, not a diff)."""

_MEMORY_OUTPUT = """Output EXACTLY this, nothing else:
<memory>
...the FULL updated memory markdown (all sections)...
</memory>
<summary>
...one short paragraph: who this user is so far + what is currently active...
</summary>"""


def _memory_trailer(memory_doc: str, rolling_summary: str, chunk_text: str) -> str:
    return f"""
## Current memory
{memory_doc}

## Rolling summary so far
{rolling_summary or "(none yet)"}

## New activity (chronological, time-masked)
{chunk_text}

Apply the method to the new activity above and emit the full updated <memory> and <summary>."""


def llm_memory_update_prompt(memory_doc: str, rolling_summary: str, chunk_text: str) -> str:
    """Persona/preference-centered, human-readable text memory (no vectors).
    Distills a chronological activity stream into durable knowledge about the
    user, maintained each chunk with explicit ADD / EDIT / REMOVE / MERGE
    actions. Deliberately general — not tuned to any evaluation dimension."""
    return f"""You maintain a compact, human-readable MEMORY of ONE user, built by reading their activity in chronological chunks. The memory is a plain-text persona/preference profile a person could read — NOT a log of events, NOT a database, and it uses no embeddings. The raw activity is a stream of timestamped engagements; distill it into durable, generalized knowledge about the user rather than retaining the events themselves.

Maintain the profile each chunk with four explicit actions:
  - ADD    : record a genuinely new durable fact about the user.
  - EDIT   : revise a fact whose details have changed, or supersede one when newer evidence contradicts it.
  - REMOVE : drop stale, one-off, or no-longer-relevant entries.
  - MERGE  : fold related or duplicate entries into one coherent note, reinforcing recurring signals.

Capture both what the user signals outright AND the preferences, traits, and patterns you can reasonably infer from their recurring behavior — their interests and preferences (what they lean toward, positively or negatively), how they come across, the people and places that matter to them, and what they are currently working toward — at an appropriate level of generality. Stay grounded in the observed activity: infer from patterns, but do not fabricate. Let incidental noise fade. Always output the full, current profile.

{_MEMORY_SECTIONS_NOTE}

{_MEMORY_RAILS}

{_MEMORY_OUTPUT}
{_memory_trailer(memory_doc, rolling_summary, chunk_text)}"""
