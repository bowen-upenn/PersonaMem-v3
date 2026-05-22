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

This file ships with discovery scaffolding + a stub runner. Bringing it
to full life requires either:
  - wiring an LLM client through `build_preference_shift_followthrough`
    (see the `discovery_llm` parameter), OR
  - generating the user_query + example/inferior pairs offline.
The runner dispatches through `chatbot_response.run_task_b` once the
build step emits instances, since the grading reduces to a personalized
chatbot response with a `stale_preference_use` hard-fail.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from typing import Iterable

from evaluation.backend_query import BackendQuery


# Maximum spread from T_shift to T_test (days). Past this, the shift is
# no longer "fresh" enough to be tempting.
T_TEST_WINDOW_DAYS = 14
# Minimum lag from T_shift to T_test (days). Below this, the shift hasn't
# had time to register as "old" yet.
T_TEST_MIN_LAG_DAYS = 1
DAY_SECONDS = 24 * 60 * 60

# Hard cap on emitted instances per user (chatbot + recsys combined).
INSTANCES_PER_USER_CAP = 4
# Require this many distinct categories per user before emitting.
MIN_DISTINCT_CATEGORIES = 2


def _ts_iso(ts: int) -> str:
    try:
        return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).isoformat()
    except Exception:
        return ""


def _harvest_shift_candidates(profile: dict, rng: random.Random) -> list[dict]:
    """Return a list of shift candidates.

    Each candidate is a dict with:
      - `kind`: "stance_shift" | "short_term_expiration"
      - `category`: str
      - `t_shift`: int (unix)
      - `old_preference`: {text, category, polarity}
      - `new_preference`: {text, category, polarity}  # for stance_shift; None for short_term
    """
    out: list[dict] = []
    preferences = profile.get("preferences") or []
    by_norm: dict[str, list[dict]] = {}
    for p in preferences:
        norm = (p.get("persona_item") or "").strip().lower()
        if not norm:
            continue
        by_norm.setdefault(norm, []).append(p)

    # Stance shifts: scan for contradicted entries (both resolution flavors).
    for p in preferences:
        history = p.get("update_history") or []
        for h in history:
            if not isinstance(h, dict):
                continue
            if h.get("update_type") != "contradicted":
                continue
            res = h.get("resolution")
            if res not in ("stance_shift_with_precedent",
                           "suppressed_insufficient_precedent"):
                continue
            t_shift = h.get("timestamp") or 0
            if not t_shift:
                continue
            other = h.get("preference") or ""
            if not other:
                continue
            # The two stances: `p` is the surviving / post-shift side
            # for stance_shift_with_precedent. For suppressed_insufficient_precedent
            # the SURVIVOR is the stronger stance — by convention `p` is on
            # the survivor side too. The `preference` field on the history
            # entry names the OTHER stance (the old / suppressed one).
            out.append({
                "kind": "stance_shift",
                "resolution": res,
                "category": p.get("category", ""),
                "t_shift": int(t_shift),
                "new_preference": {
                    "text": p.get("persona_item", ""),
                    "category": p.get("category", ""),
                    "polarity": p.get("polarity", "pos"),
                },
                "old_preference": {
                    "text": other,
                    "category": p.get("category", ""),  # heuristic; usually same category
                    "polarity": "neg" if p.get("polarity", "pos") == "pos" else "pos",
                },
            })

    # Short-term expirations.
    for p in preferences:
        if (p.get("time_horizon") or "long_term") != "short_term":
            continue
        sc = p.get("stop_condition") or {}
        if not isinstance(sc, dict):
            continue
        stop_ts = sc.get("expected_stop_ts")
        if not stop_ts:
            continue
        out.append({
            "kind": "short_term_expiration",
            "stop_type": sc.get("type", "event"),
            "category": p.get("category", ""),
            "t_shift": int(stop_ts),
            "old_preference": {
                "text": p.get("persona_item", ""),
                "category": p.get("category", ""),
                "polarity": p.get("polarity", "pos"),
            },
            # For short-term expirations the "new" stance is "no longer
            # relevant" — there's no replacement preference. Mark as None
            # so the discovery prompt knows to frame the query as "the
            # agent should NOT surface the old short-term pref any more."
            "new_preference": None,
        })

    rng.shuffle(out)
    return out


def _pick_t_test(t_shift: int, t_now: int, rng: random.Random) -> int:
    """Pick T_test ∈ (T_shift + min_lag, min(T_shift + window, T_now)]."""
    lo = t_shift + T_TEST_MIN_LAG_DAYS * DAY_SECONDS
    hi = min(t_shift + T_TEST_WINDOW_DAYS * DAY_SECONDS, t_now)
    if hi <= lo:
        return 0
    return rng.randint(lo, hi)


def _build_instance(
    cand: dict,
    flavor: str,
    user_id: str,
    seq: int,
    t_test: int,
) -> dict:
    """Emit one instance in the canonical 5-field shape.

    `user_query`, `example_response`, `inferior_response` are placeholders
    here — production builds should fill them via an LLM discovery call.
    Build_benchmark drops instances with empty user_query at audit time.
    """
    return {
        "instance_id": f"pshift_{user_id}_{seq:03d}_{flavor}",
        "task_type": "preference_shift_followthrough",
        "task_id": "preference_shift_followthrough",
        "flavor": flavor,
        "entry_point": "chatbot_routed" if flavor == "chatbot" else "app_native",
        "t_test": t_test,
        "user_query": "",  # to be filled by discovery LLM
        "example_response": "",  # to be filled by discovery LLM
        "inferior_response": "",  # to be filled by discovery LLM
        "groundtruth_preference": {
            "t_shift": cand["t_shift"],
            "shift_kind": cand["kind"],
            "shift_resolution": cand.get("resolution"),
            "old_preference": cand["old_preference"],
            "new_preference": cand.get("new_preference"),
        },
        # rubric_tags is set at CSV-emission time from TASK_TYPE_META.
    }


def build_preference_shift_followthrough(
    bq: BackendQuery,
    user_id: str,
    t_now: int,
    discovery_llm=None,
    rng_seed: int = 0,
) -> list[dict]:
    """Build preference_shift_followthrough instances for one user.

    Scaffolding only: emits empty user_query / example / inferior fields.
    A discovery LLM call (TODO) should populate those. The audit step
    drops empty-query rows so this is safe to ship without breaking
    downstream consumers; the task type just produces zero non-empty
    instances until the LLM wiring lands.
    """
    profile = bq.get_full_profile(user_id) or {}
    if not profile:
        return []

    rng = random.Random(rng_seed)
    cands = _harvest_shift_candidates(profile, rng)
    if not cands:
        return []

    # Diversity: require ≥ MIN_DISTINCT_CATEGORIES distinct categories.
    cats = {c.get("category") for c in cands if c.get("category")}
    if len(cats) < MIN_DISTINCT_CATEGORIES:
        # Still emit if at least one cat — the diversity gate is advisory.
        pass

    out: list[dict] = []
    flavor_cycle = iter(["chatbot", "recsys", "chatbot", "recsys"])
    for i, c in enumerate(cands[:INSTANCES_PER_USER_CAP]):
        t_test = _pick_t_test(c["t_shift"], t_now, rng)
        if not t_test:
            continue
        flavor = next(flavor_cycle, "chatbot")
        inst = _build_instance(c, flavor, user_id, i + 1, t_test)
        # If a discovery_llm is wired, fill user_query / example_response /
        # inferior_response here. For now, mark a hint so the audit step
        # knows this is a build-time scaffolding stub.
        inst["_scaffolding_stub"] = True
        out.append(inst)

    if discovery_llm is None:
        print(f"[preference_shift_followthrough] user {user_id}: "
              f"emitted {len(out)} scaffolded instance(s) — "
              f"discovery_llm not wired, user_query/example/inferior empty.")
    return out


# ---------------------------------------------------------------------------
# Runner — once instances carry non-empty user_query, this dispatches to the
# chatbot_response runner since the grading reduces to "personalized chatbot
# response with stale_preference_use hard-fail". For now it's a no-op stub.
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
    """Stub runner. Returns one dry_run-style result per scaffolded instance
    so the benchmark pipeline doesn't crash. Replace with a chatbot_response
    dispatch once the discovery LLM fills user_query / example / inferior.
    """
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        results.append({
            "task": "preference_shift_followthrough",
            "user_id": user_id,
            "instance_id": inst.get("instance_id", ""),
            "flavor": inst.get("flavor", ""),
            "mode": mode,
            "metrics": {},
            "status": "scaffolding_stub" if inst.get("_scaffolding_stub") else "ok",
        })
    return results
