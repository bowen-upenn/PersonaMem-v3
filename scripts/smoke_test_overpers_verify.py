#!/usr/bin/env python3
"""Smoke test: for every over_personalization_chatbot_text query in
benchmark/{uid}/queries.csv across all users, ask a verification LLM
whether personalization would *meaningfully* help. The intent is to
catch borderline cases (e.g. "can I put a hot glass dish in the
fridge?") where the "must NOT surface preferences" framing is unfair
because personalization adds little value either way.

Result: per-arm tally of helpful vs not-helpful, plus a sample of
queries the LLM flags as "personalization would help" (these would
be candidates to remove from the over_personalization arm).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

USERS = ["105", "115", "229", "282", "760"]

VERIFY_PROMPT = """You are auditing an over-personalization test for a chatbot.

The chatbot has access to the user's full social-media interaction history (their interests, preferences, hobbies). The test asks: when given this query, would the chatbot's correct response benefit *meaningfully* from knowing the user's interests/preferences?

## Query
{query}

## Decide

- **HELPS**: knowing the user's specific interests/preferences would let the chatbot give a *meaningfully better* or *more useful* answer. A generic answer would be visibly worse. Personalization is genuinely additive.
- **NEUTRAL**: the answer is essentially the same regardless of who's asking — factual, procedural, or universally applicable. Personalization adds at most decorative color, not substance.

Examples:
- "what should I watch tonight?" → HELPS (taste-based)
- "any podcast recs for my commute?" → HELPS (taste-based)
- "need a gift idea for a friend" → HELPS (knowing the recipient or shared interests helps)
- "can I put cooked rice in the fridge for 5 days?" → NEUTRAL (factual)
- "how do I remove a coffee stain?" → NEUTRAL (procedural)
- "what does the IELTS speaking 7.5 mean?" → NEUTRAL (definitional)

Return a JSON object: `{{"verdict": "HELPS" | "NEUTRAL", "reason": "<one short sentence>"}}`
"""


def main() -> int:
    csv.field_size_limit(10**7)

    # Build LLM client
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-5.4-mini"
    try:
        from query_llm import QueryLLM
        client = QueryLLM({"models": {"llm_model": model}}, rate_limit_per_min=200)
    except Exception as exc:
        print(f"FATAL: LLM client init failed: {exc}", file=sys.stderr)
        return 1
    print(f"[smoke_test] model={model}\n")

    # Collect both over_personalization_chatbot_text AND
    # chatbot_personalized_response instances. The verifier should flag
    # taste-based queries as HELPS (wrong if routed to over_pers) and
    # factual queries as NEUTRAL (wrong if routed to proactive).
    rows = []
    for uid in USERS:
        path = REPO_ROOT / "benchmark" / uid / "queries.csv"
        if not path.exists():
            continue
        with path.open() as f:
            lines = f.readlines()
        reader = csv.DictReader(lines[1:])  # skip version comment
        for row in reader:
            tt = row.get("task_type", "")
            if tt not in ("over_personalization_chatbot_text",
                          "chatbot_personalized_response"):
                continue
            try:
                inst = json.loads(row.get("instance_json", "{}"))
            except json.JSONDecodeError:
                continue
            arm = inst.get("arm", "")
            query = inst.get("user_query", "")
            rows.append({"uid": uid, "task_type": tt, "arm": arm, "query": query})

    print(f"[smoke_test] {len(rows)} queries "
          f"(over_pers + proactive)\n")

    # Run verifications in parallel
    from data_preparation.utils import extract_json_from_response

    def _verify(idx_row):
        idx, row = idx_row
        prompt = VERIFY_PROMPT.format(query=row["query"])
        try:
            raw = client.query_llm(prompt) or ""
            parsed = extract_json_from_response(raw)
            if not isinstance(parsed, dict):
                return idx, {"verdict": "ERROR", "reason": "bad JSON"}
            v = (parsed.get("verdict") or "").strip().upper()
            if v not in ("HELPS", "NEUTRAL"):
                v = "ERROR"
            return idx, {"verdict": v, "reason": parsed.get("reason", "")}
        except Exception as exc:
            return idx, {"verdict": "ERROR", "reason": str(exc)[:80]}

    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(_verify, (i, r)) for i, r in enumerate(rows)]
        done = 0
        for fut in as_completed(futs):
            idx, verdict = fut.result()
            results[idx] = verdict
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(rows)}")

    # Tally
    from collections import Counter, defaultdict
    by_bucket: dict[str, Counter] = defaultdict(Counter)
    misroutes: dict[str, list[dict]] = defaultdict(list)
    for row, verdict in zip(rows, results):
        # Bucket = task_type/arm
        if row["task_type"] == "chatbot_personalized_response":
            bucket = f"proactive/{row['arm']}"
        else:
            bucket = f"over_pers/{row['arm']}"
        v = verdict["verdict"]
        by_bucket[bucket][v] += 1
        # Misroute: over_pers query flagged HELPS, or proactive flagged NEUTRAL
        is_overpers = row["task_type"] == "over_personalization_chatbot_text"
        if (is_overpers and v == "HELPS") or (not is_overpers and v == "NEUTRAL"):
            misroutes[bucket].append({**row, "verdict": v,
                                      "reason": verdict["reason"]})

    print("\n=== Tally by bucket ===")
    print(f"{'bucket':<35} {'HELPS':>6} {'NEUTRAL':>8} {'ERROR':>6}  misroute")
    for bucket in sorted(by_bucket.keys()):
        c = by_bucket[bucket]
        total = sum(c.values())
        h, n, e = c.get("HELPS", 0), c.get("NEUTRAL", 0), c.get("ERROR", 0)
        is_overpers = bucket.startswith("over_pers")
        mis = h if is_overpers else n
        mis_label = f"{mis}/{total} = {mis*100/total:.0f}%" if total else "—"
        print(f"{bucket:<35} {h:>6} {n:>8} {e:>6}  {mis_label}")

    print("\n=== Misrouted queries (would be filtered out) ===")
    for bucket, examples in sorted(misroutes.items()):
        print(f"\n--- {bucket} ({len(examples)} misrouted) ---")
        for ex in examples[:6]:
            print(f"  [{ex['uid']}] [{ex['verdict']}] {ex['query'][:90]}")
            print(f"          reason: {ex['reason'][:110]}")

    # Save full results for inspection
    out_path = REPO_ROOT / "/tmp" / "overpers_verify_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full = [{**r, **v} for r, v in zip(rows, results)]
    with open("/tmp/overpers_verify_results.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"\nFull results saved to /tmp/overpers_verify_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
