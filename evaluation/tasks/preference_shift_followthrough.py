"""Step 4.5 — preference_shift_followthrough.

Tests whether the agent uses the **latest** stance after a user's preference
shifts, instead of leaning on the outdated one. Two flavors:

- `chatbot`: a natural chatbot query whose right answer reflects the
  post-shift preference; the inferior response sticks to the old one.
- `recsys`: a feed-slate moment near T_test where the gold ranking puts
  new-preference items on top; the inferior puts old-preference items on
  top.

Shift sources (in order of preference for instance discovery):

  1. **Stance shifts** — canonicals with `update_history` containing an
     entry of `update_type="contradicted"` with `resolution ∈
     {"stance_shift_with_precedent", "suppressed_insufficient_precedent"}`.
     Use the entry's `timestamp` as `T_shift`.
  2. **Short-term expirations** — canonicals with `time_horizon=="short_term"`
     and `stop_condition.expected_stop_ts` set. Use that as `T_shift`.

`T_test ∈ (T_shift, T_shift + 14d]` so the new stance is live and recent
enough to test, but the old stance still feels "tempting" to surface.

Per the project plan, instance fields follow the canonical 5-field layout
(user_query, example_response, inferior_response, groundtruth_preference,
rubric_dimensions). Shift metadata lives inside `groundtruth_preference`.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import random
from typing import Any, Iterable

from evaluation.backend_query import BackendQuery
from data_preparation.utils import extract_json_from_response


# Maximum spread from T_shift to T_test (days). Past this, the shift is
# no longer "fresh" enough to be tempting.
T_TEST_WINDOW_DAYS = 14
# Minimum lag from T_shift to T_test (days). Below this, the shift hasn't
# had time to register as "old" yet.
T_TEST_MIN_LAG_DAYS = 1
DAY_SECONDS = 24 * 60 * 60

# Hard cap on emitted instances per user (chatbot + recsys combined).
# Bumped from 4 → 6 because audit found 3/5 users were emitting only
# 0-1 surviving rows after `_pick_t_test` + discovery-LLM attrition.
INSTANCES_PER_USER_CAP = 6
# Require this many distinct categories per user before emitting.
MIN_DISTINCT_CATEGORIES = 2

# Word-count bounds for discovery validation.
_USER_QUERY_MIN_WORDS = 5
_USER_QUERY_MAX_WORDS = 25
_RESPONSE_MIN_WORDS = 15
_RESPONSE_MAX_WORDS = 100

_DISCOVERY_RETRIES = 1

_WORD_RE = re.compile(r"\b[\w']+\b")


def _ts_iso(ts: int) -> str:
    try:
        return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).isoformat()
    except Exception:
        return ""


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _harvest_shift_candidates(
    bq: BackendQuery, user_id: str, rng: random.Random,
) -> list[dict]:
    """Return a list of shift candidates.

    NOTE: `update_history` and `stop_condition` live on per-app preference
    dicts inside `backend/{uid}/{app}.json` events, NOT on `profile.json`
    (where `preferences` is just a string list). We have to read the raw
    per-app JSONs to see these fields — `bq.get_events` strips them via
    `_LEAK_FIELDS_PREF`.

    Each candidate is a dict with:
      - `kind`: "stance_shift" | "short_term_expiration"
      - `category`: str
      - `t_shift`: int (unix)
      - `old_preference`: {text, category, polarity}
      - `new_preference`: {text, category, polarity}  # for stance_shift; None for short_term
    """
    from pathlib import Path

    out: list[dict] = []
    seen_keys: set[str] = set()
    base = Path(getattr(bq, "base", "backend")) / user_id
    apps = ("instagram", "facebook", "threads", "chatbot")
    for app in apps:
        path = base / f"{app}.json"
        if not path.exists():
            continue
        try:
            events = json.loads(path.read_text())
        except Exception:
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            for p in (ev.get("preferences") or []):
                if not isinstance(p, dict):
                    continue
                persona_item = (p.get("persona_item") or "").strip()
                if not persona_item:
                    continue
                key_base = persona_item.lower()

                # Stance shifts via update_history `contradicted` /
                # `shifted` entries. Audit (2026-05-28) found 0 surviving
                # `stance_shift_with_precedent` resolutions in all 5
                # users' data (the precedent gate is strict; most
                # contradictions land as `suppressed_weak_minority`),
                # so harvest from a wider set: also include `shifted`
                # entries (LLM-emitted cross-ref shifts where the
                # canonical changed form).
                for h in (p.get("update_history") or []):
                    if not isinstance(h, dict):
                        continue
                    ut = h.get("update_type")
                    res = h.get("resolution")
                    accepted = (
                        ut == "contradicted"
                        and res in (
                            "stance_shift_with_precedent",
                            "suppressed_insufficient_precedent",
                        )
                    ) or ut == "shifted"
                    if not accepted:
                        continue
                    t_shift = h.get("timestamp") or 0
                    if not t_shift:
                        continue
                    other = h.get("preference") or ""
                    if not other:
                        continue
                    key = f"shift|{key_base}|{other.lower()}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    out.append({
                        "kind": "stance_shift",
                        "resolution": res or ut,
                        "category": p.get("category", ""),
                        "t_shift": int(t_shift),
                        "new_preference": {
                            "text": persona_item,
                            "category": p.get("category", ""),
                            "polarity": p.get("polarity", "pos"),
                        },
                        "old_preference": {
                            "text": other,
                            "category": p.get("category", ""),
                            "polarity": "neg" if p.get("polarity", "pos") == "pos" else "pos",
                        },
                    })

                # Short-term expirations.
                if (p.get("time_horizon") or "long_term") == "short_term":
                    sc = p.get("stop_condition") or {}
                    if isinstance(sc, dict):
                        stop_ts = sc.get("expected_stop_ts")
                        if stop_ts:
                            key = f"expire|{key_base}"
                            if key not in seen_keys:
                                seen_keys.add(key)
                                out.append({
                                    "kind": "short_term_expiration",
                                    "stop_type": sc.get("type", "event"),
                                    "category": p.get("category", ""),
                                    "t_shift": int(stop_ts),
                                    "old_preference": {
                                        "text": persona_item,
                                        "category": p.get("category", ""),
                                        "polarity": p.get("polarity", "pos"),
                                    },
                                    "new_preference": None,
                                })

    rng.shuffle(out)
    return out


def _pick_t_test(t_shift: int, t_now: int, rng: random.Random) -> int:
    """Pick T_test ∈ (T_shift + min_lag, min(T_shift + window, T_now)].

    A future-stop candidate (`expected_stop_ts` after `t_now`) CANNOT be tested
    for expiration follow-through: at every observable moment the preference is
    still active, yet the GT template labels it "expired — no replacement" and
    penalizes an on-topic, still-active pref. An audit (2026-05-28) fallback
    tested such candidates at `t_now - 1h` to salvage coverage for user 760, but
    that shipped mislabeled "expired-when-active" rows (audit 2026-07-16, T2-4,
    confirmed personas 2/5/6). Correctness over coverage: skip future stops
    (return 0 → the caller drops the candidate). A user with only future stops
    simply gets no preference_shift instances from expirations — a data-dependent
    coverage gap, not an unfair test.
    """
    lo = t_shift + T_TEST_MIN_LAG_DAYS * DAY_SECONDS
    hi = min(t_shift + T_TEST_WINDOW_DAYS * DAY_SECONDS, t_now)
    if hi > lo:
        return rng.randint(lo, hi)
    return 0


def _build_instance(
    cand: dict,
    flavor: str,
    user_id: str,
    seq: int,
    t_test: int,
) -> dict:
    """Emit one instance in the canonical 5-field shape."""
    return {
        "instance_id": f"pshift_{user_id}_{seq:03d}_{flavor}",
        "task_type": "preference_shift_followthrough",
        "task_id": "preference_shift_followthrough",
        "flavor": flavor,
        "entry_point": "chatbot_routed" if flavor == "chatbot" else "app_native",
        "t_test": t_test,
        "user_query": "",
        "example_response": "",
        "inferior_response": "",
        "groundtruth_preference": {
            "t_shift": cand["t_shift"],
            "shift_kind": cand["kind"],
            "shift_resolution": cand.get("resolution"),
            "old_preference": cand["old_preference"],
            "new_preference": cand.get("new_preference"),
        },
    }


# ---------------------------------------------------------------------------
# Discovery prompt (build-time mini-tier LLM call)
# ---------------------------------------------------------------------------

DISCOVERY_PROMPT = """You are designing a benchmark probe for whether an AI assistant correctly tracks a preference shift — using the user's CURRENT stance instead of the outdated one.

You will be given metadata about a preference shift: what the user USED to like/do (`old_preference`) and what they NOW like/do (`new_preference`, or None if the old preference simply expired). Your job is to produce a triple:

  user_query        — a natural, casual chat-turn the user might send. It should touch the category/domain of the shifted preference WITHOUT directly asking "what do I like now?" The query invites the agent to demonstrate whether it knows the current stance.
  example_response  — a natural AI reply that correctly uses the NEW/current stance (or, for expirations, avoids surfacing the expired preference).
  inferior_response — a natural AI reply that incorrectly leans on the OLD/outdated stance. Same length and tone; the error is that it treats the old stance as still active.

## Shift metadata

  shift_kind:     {shift_kind}
  category:       {category}
  old_preference: {old_pref_text} ({old_pref_polarity})
  new_preference: {new_pref_text} ({new_pref_polarity})
  resolution:     {resolution}

## User context

  name:   {name}
  career: {career}
  bio:    {bio}

## Hard constraints

(A) `user_query` is {min_words}-{max_words} words, lowercase casual, real chat-turn length.
(B) `user_query` must NOT contain the preference text verbatim — it should ask about the domain/topic naturally.
(C) `example_response` uses the NEW stance — it suggests/recommends/discusses based on the current preference, NOT the old one.
(D) `inferior_response` uses the OLD stance — it suggests/recommends based on the outdated preference as if nothing changed.
(E) For short_term_expiration (new_preference is null): `example_response` should answer the query without leaning on the now-expired preference. `inferior_response` incorrectly still surfaces the expired preference as if it's active.
(F) Neither response should use "I know you…", "since you like X…", "based on your history…", or similar telegraphing phrases. Personalization should be visible in CONTENT CHOICE, not in self-referential framing.
(G) Both responses are 2-4 sentences, conversational. Same length and tone so the difference is content, not quality.
(H) `user_query` should NOT be a yes/no question — it should invite an open-ended response where the agent naturally reveals which stance it holds.

## Output

Return ONE JSON object inside a fenced block:

```json
{{
  "user_query": "...",
  "example_response": "...",
  "inferior_response": "..."
}}
```"""


def _format_discovery_prompt(cand: dict, profile: dict) -> str:
    old_pref = cand.get("old_preference") or {}
    new_pref = cand.get("new_preference")
    new_text = new_pref.get("text", "") if isinstance(new_pref, dict) else "(expired — no replacement)"
    new_pol = new_pref.get("polarity", "") if isinstance(new_pref, dict) else ""
    return DISCOVERY_PROMPT.format(
        shift_kind=cand.get("kind", ""),
        category=cand.get("category", ""),
        old_pref_text=old_pref.get("text", ""),
        old_pref_polarity=old_pref.get("polarity", ""),
        new_pref_text=new_text,
        new_pref_polarity=new_pol,
        resolution=cand.get("resolution", ""),
        name=profile.get("name", "the user"),
        career=profile.get("career", ""),
        bio=(profile.get("bio", "") or "")[:300],
        min_words=_USER_QUERY_MIN_WORDS,
        max_words=_USER_QUERY_MAX_WORDS,
    )


def _validate_discovery_output(
    parsed: dict,
    cand: dict,
) -> tuple[bool, str]:
    """Deterministic post-validator. Returns (passed, violation_reason)."""
    for key in ("user_query", "example_response", "inferior_response"):
        if not isinstance(parsed.get(key), str) or not parsed[key].strip():
            return False, f"missing or empty field: {key!r}"

    user_query = parsed["user_query"].strip()
    example = parsed["example_response"].strip()
    inferior = parsed["inferior_response"].strip()

    uq_wc = _word_count(user_query)
    if not (_USER_QUERY_MIN_WORDS <= uq_wc <= _USER_QUERY_MAX_WORDS):
        return False, f"user_query has {uq_wc} words; must be {_USER_QUERY_MIN_WORDS}-{_USER_QUERY_MAX_WORDS}"

    for name, text in (("example_response", example), ("inferior_response", inferior)):
        wc = _word_count(text)
        if not (_RESPONSE_MIN_WORDS <= wc <= _RESPONSE_MAX_WORDS):
            return False, f"{name} has {wc} words; must be {_RESPONSE_MIN_WORDS}-{_RESPONSE_MAX_WORDS}"

    old_text = (cand.get("old_preference") or {}).get("text", "").strip().lower()
    if old_text and len(old_text) > 15 and old_text in example.lower():
        return False, "example_response contains old preference text verbatim"

    if example.lower() == inferior.lower():
        return False, "example_response and inferior_response are identical"

    # Telegraph check
    from evaluation.llm_postprocess import _TELEGRAPH_PHRASE_RE
    for name, text in (("example_response", example), ("inferior_response", inferior)):
        m = _TELEGRAPH_PHRASE_RE.search(text)
        if m:
            return False, f"{name} contains telegraph phrase: {m.group(0)!r}"

    return True, ""


def _build_corrective_prompt(base_prompt: str, violation: str) -> str:
    return (
        base_prompt
        + "\n\n## Your last attempt failed validation\n\n"
        + f"Violation: {violation}\n\n"
        + "Try again, keeping every other constraint the same. "
        "Return ONE JSON object inside a fenced block, no prose outside."
    )


def _discover_shift_triplet(
    discovery_llm: Any,
    cand: dict,
    profile: dict,
    verbose: bool = False,
) -> dict | None:
    """One discovery LLM call + up to one corrective retry.

    Returns the parsed dict if validation passes, else None.
    """
    base_prompt = _format_discovery_prompt(cand, profile)

    raw = discovery_llm.query_llm(base_prompt)
    parsed = extract_json_from_response(raw) or {}
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if isinstance(parsed, dict):
        ok, why = _validate_discovery_output(parsed, cand)
        if ok:
            return parsed
    else:
        ok, why = False, f"LLM returned non-object JSON: {type(parsed).__name__}"

    if verbose:
        cat = cand.get("category", "?")
        print(f"[preference_shift_followthrough] retry ({cat}): {why}")

    for _ in range(_DISCOVERY_RETRIES):
        corrective = _build_corrective_prompt(base_prompt, why)
        raw = discovery_llm.query_llm(corrective)
        parsed = extract_json_from_response(raw) or {}
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if isinstance(parsed, dict):
            ok, why = _validate_discovery_output(parsed, cand)
            if ok:
                return parsed
        else:
            why = f"LLM returned non-object JSON: {type(parsed).__name__}"

    if verbose:
        cat = cand.get("category", "?")
        print(f"[preference_shift_followthrough] dropping ({cat}) after {_DISCOVERY_RETRIES + 1} attempts: {why}")
    return None


def build_preference_shift_followthrough(
    bq: BackendQuery,
    user_id: str,
    t_now: int,
    discovery_llm=None,
    rng_seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Build preference_shift_followthrough instances for one user.

    When `discovery_llm` is provided, populates user_query / example_response /
    inferior_response via LLM calls. When None, emits scaffolding stubs that
    the audit step drops automatically.
    """
    rng = random.Random(rng_seed)
    cands = _harvest_shift_candidates(bq, user_id, rng)
    if not cands:
        return []

    cats = {c.get("category") for c in cands if c.get("category")}
    if len(cats) < MIN_DISTINCT_CATEGORIES:
        pass

    profile = {}
    if discovery_llm is not None:
        try:
            from pathlib import Path
            profile_path = Path(getattr(bq, "base", "backend")) / user_id / "profile.json"
            if profile_path.exists():
                profile = json.loads(profile_path.read_text())
        except Exception:
            pass

    out: list[dict] = []
    flavor_cycle = iter(["chatbot", "recsys", "chatbot", "recsys"])
    for i, c in enumerate(cands[:INSTANCES_PER_USER_CAP]):
        t_test = _pick_t_test(c["t_shift"], t_now, rng)
        if not t_test:
            continue
        flavor = next(flavor_cycle, "chatbot")
        inst = _build_instance(c, flavor, user_id, i + 1, t_test)

        if discovery_llm is not None:
            parsed = _discover_shift_triplet(discovery_llm, c, profile, verbose=verbose)
            if parsed is not None:
                inst["user_query"] = parsed["user_query"].strip()
                inst["example_response"] = parsed["example_response"].strip()
                inst["inferior_response"] = parsed["inferior_response"].strip()
            else:
                inst["_scaffolding_stub"] = True
        else:
            inst["_scaffolding_stub"] = True

        if not inst.get("_scaffolding_stub"):
            out.append(inst)
        elif discovery_llm is None:
            out.append(inst)

    if discovery_llm is None:
        print(f"[preference_shift_followthrough] user {user_id}: "
              f"emitted {len(out)} scaffolded instance(s) — "
              f"discovery_llm not wired, user_query/example/inferior empty.")
    else:
        filled = sum(1 for inst in out if not inst.get("_scaffolding_stub"))
        print(f"[preference_shift_followthrough] user {user_id}: "
              f"{filled}/{len(out)} instances filled by discovery LLM.")
    return out


# ---------------------------------------------------------------------------
# Runner — dispatches to chatbot_response runner once instances carry
# non-empty user_query. Scaffolded instances return a placeholder result.
# ---------------------------------------------------------------------------


def run_preference_shift_followthrough(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    """Run preference_shift_followthrough instances.

    Both flavors (chatbot + recsys) present the discovery-generated
    `user_query` to the agent at `t_test` (time-masked history) and judge
    whether the response follows the user's CURRENT stance vs the outdated
    one — headline metric `preference_shift_consistency` (0-10 LLM judge).

    The instance is self-contained (its own `t_test` / `user_query` /
    `example_response` / `inferior_response` / `groundtruth_preference`), so it
    runs through the shared chatbot-answer path directly instead of being
    re-shaped into a Task-B instance (which expected `source_timestamp` /
    `test_id` and raised KeyError on these rows). The recsys flavor is scored
    identically — the shift-consistency question is flavor-independent.
    """
    from evaluation import prompts, judges
    from evaluation.inference_utils import dispatch_agent_run as _dispatch_agent

    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        base = {
            "task": "preference_shift_followthrough",
            "user_id": user_id,
            "instance_id": inst.get("instance_id", ""),
            "flavor": inst.get("flavor", ""),
            "mode": mode,
        }
        if inst.get("_scaffolding_stub"):
            results.append({**base, "metrics": {}, "status": "scaffolding_stub"})
            continue

        user_query = (inst.get("user_query") or "").strip()
        if not user_query:
            results.append({**base, "metrics": {}, "status": "skipped_no_query"})
            continue

        t_test = inst["t_test"]
        history_block = None
        if mode in ("llm_longctx", "llm_memory", "mem0"):
            history_block, _stats = snapshot_cache.get_or_build(
                bq, user_id, t_test, model_name, context_budget)

        prompt = prompts.chatbot_response_prompt(user_query, [], history_block)

        if dry_run:
            results.append({**base, "agent_response": None,
                            "metrics": None, "status": "ok"})
            continue

        raw_response, tool_call_count, subagent_stats = _dispatch_agent(
            mode, prompt, bq=bq, user_id=user_id, t=t_test,
            claude_model=claude_model, llm_client=llm_client,
        )
        parsed = extract_json_from_response(raw_response)
        if isinstance(parsed, dict) and parsed.get("response"):
            response_text = parsed["response"]
        else:
            response_text = raw_response

        metrics: dict = {}
        if enable_llm_judge and judge_client is not None:
            metrics = judges.judge_preference_shift(
                judge_client,
                response_text,
                inst.get("groundtruth_preference") or {},
                user_query,
                example_response=inst.get("example_response", ""),
                inferior_response=inst.get("inferior_response", ""),
            )

        results.append({
            **base,
            "agent_response": response_text,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "metrics": metrics,
            "status": "ok",
        })
    return results
