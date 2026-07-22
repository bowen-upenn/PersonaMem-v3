#!/usr/bin/env python3
"""Study 2 (ablation): LLM-as-a-judge cross-model agreement.

Re-judges a FIXED set of responses with three judge models on the *exact
same* prompt string, so any score difference is the judge's, not the
answer's. The fixed responses are persona 1's real model-under-test
outputs (llm_longctx gpt-5.5) plus the matched benchmark foils.

We reconstruct the harness's own chatbot-rubric judge prompt (anchor
TestItem -> build_judge_evidence -> judge_chatbot_rubric_prompt) so the
prompt is byte-identical to production, then dispatch it to:
  - gpt-5.5         (Azure)        -- scored inline here
  - gemini-3.5-flash (Gemini API)  -- scored inline here
  - claude-opus-4.8 (Claude Code)  -- no API key; prompts are queued to
    opus_queue/*.jsonl for Claude Code subagents to score, then merged.

Outputs (under --out_dir, default results/audit/judge_agreement_p{uid}):
  prompts.jsonl       item_key, query_id, task_type, population, polarity, prompt
  api_scores.jsonl    one row per (item_key, judge in {gpt-5.5, gemini}) x 4 dims
  opus_queue/batch_NN.jsonl   prompt batches for the Opus subagents

Usage:
  python results/_scripts/judge_agreement.py --user_id 1 \
      --base results/llm_longctx_gpt5.5_judged
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_preparation.utils import extract_json_from_response  # noqa: E402
from evaluation import prompts  # noqa: E402
from evaluation.backend_query import BackendQuery  # noqa: E402
from evaluation.inference_utils import TestItem, build_judge_evidence  # noqa: E402

# The chatbot-rubric family: every task scored by judge_chatbot_rubric
# (preference_alignment / helpfulness / appropriate_restraint /
# no_hallucinated_preference, each 0-10). One rubric, one 0-10 scale ->
# the cleanest single population for a distribution + agreement figure.
CHATBOT_RUBRIC_TASKS = {
    "chatbot_personalized_response",
    "over_personalization_chatbot_text",
    "over_personalization_sensitive_event",
    "over_personalization_context_shift",
}
RUBRIC_DIMS = (
    "preference_alignment",
    "helpfulness",
    "appropriate_restraint",
    "no_hallucinated_preference",
)
# arms that flip the judge to "negative" polarity (reward restraint),
# copied verbatim from evaluation/tasks/chatbot_response.py.
_NEG_ARMS = {"control", "adversarial", "distractor_reject", "stale",
             "sensitive_event", "conversational_drift"}


def _as_text(x) -> str:
    """Coerce an example/inferior response (str | dict | repr-of-dict) to
    plain text. test.json stores some foils as a Python-repr string like
    "{'text': '...'}" -- recover the inner text."""
    if x is None:
        return ""
    if isinstance(x, dict):
        return str(x.get("text") or x.get("response") or "").strip()
    s = str(x).strip()
    if s[:1] in "{[" and ("'text'" in s or '"text"' in s):
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(s)
                if isinstance(v, dict):
                    return str(v.get("text") or v.get("response") or "").strip()
            except Exception:  # noqa: BLE001
                pass
    return s


def _load_real_responses(base: Path, uid: str) -> dict:
    src = base / uid / "results.csv"
    csv.field_size_limit(sys.maxsize)
    out = {}
    with src.open() as f:
        for r in csv.DictReader(f):
            out[r["query_id"]] = r.get("agent_response") or ""
    return out


def _flatten(item: dict, uid: str) -> dict:
    inst = dict(item.get("instance_full") or item)
    inst.setdefault("task_type", item.get("task_type") or "")
    inst.setdefault("query_id", item.get("query_id") or inst.get("test_id") or "")
    if item.get("user_query") and not inst.get("user_query"):
        inst["user_query"] = item["user_query"]
    inst["user_id"] = str(uid)
    ts = inst.get("source_timestamp")
    if isinstance(ts, str) and ts.isdigit():
        inst["source_timestamp"] = int(ts)
    return inst


def _polarity(inst: dict) -> str:
    pol = inst.get("polarity", "positive")
    if inst.get("action", "") == "asked_not_to_personalize" or inst.get("arm") in _NEG_ARMS:
        pol = "negative"
    return pol


def _build_prompt(bq: BackendQuery, inst: dict, response_text: str) -> str:
    anchor = TestItem(
        user_id=inst["user_id"],
        app="chatbot",
        source_object_id=inst.get("test_id", ""),
        source_timestamp=int(inst.get("source_timestamp") or 0),
        formatted_timestamp=inst.get("formatted_timestamp", ""),
        source_interaction_type=(
            "implicit_positive" if inst.get("polarity") == "positive" else "implicit_negative"
        ),
        source_hashtags=inst.get("source_hashtags", []) or [],
        content={},
        interaction_format={"action": inst.get("action", "")},
        preference=inst.get("held_out_preference") or {},
    )
    evidence = build_judge_evidence(bq, anchor, response_text)
    return prompts.judge_chatbot_rubric_prompt(response_text, evidence, _polarity(inst))


def _score(judge_client, prompt: str) -> dict:
    try:
        parsed = extract_json_from_response(judge_client.query_llm(prompt)) or {}
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)[:120]}
    out = {}
    for k in RUBRIC_DIMS:
        v = parsed.get(k)
        try:
            out[k] = max(0.0, min(10.0, float(v))) if v is not None else None
        except (TypeError, ValueError):
            out[k] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user_id", required=True)
    ap.add_argument("--base", default="results/llm_longctx_gpt5.5_judged",
                    help="dir holding {uid}/results.csv with real responses")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--rate_limit", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--opus_batch", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    uid = args.user_id
    out_dir = Path(args.out_dir or f"results/audit/judge_agreement_p{uid}")
    (out_dir / "opus_queue").mkdir(parents=True, exist_ok=True)

    items = json.loads((Path(args.backend_dir) / uid / "test.json").read_text())
    items = [it for it in items if it.get("task_type") in CHATBOT_RUBRIC_TASKS]
    if args.limit:
        items = items[: args.limit]
    real = _load_real_responses(Path(args.base), uid)
    bq = BackendQuery(args.backend_dir)

    from query_llm import QueryLLM
    judges = {
        "gpt-5.5": QueryLLM({"models": {"llm_model": "gpt-5.5"}}, rate_limit_per_min=args.rate_limit),
        "gemini-3.5-flash": QueryLLM({"models": {"llm_model": "gemini-3.5-flash"}}, rate_limit_per_min=args.rate_limit),
    }

    # Build the (item x population) work list with reconstructed prompts.
    work = []  # {item_key, query_id, task_type, population, polarity, response, prompt}
    for item in items:
        inst = _flatten(item, uid)
        qid = inst["query_id"]
        real_text = real.get(qid, "")
        foil_text = _as_text(inst.get("inferior_response"))
        for population, text in (("real", real_text), ("foil", foil_text)):
            if not text.strip():
                continue
            prompt = _build_prompt(bq, inst, text)
            work.append({
                "item_key": f"{qid}::{population}",
                "query_id": qid,
                "task_type": inst["task_type"],
                "population": population,
                "polarity": _polarity(inst),
                "prompt": prompt,
            })

    print(f"[judge_agree] persona {uid}: {len(items)} items -> {len(work)} "
          f"(item x population) prompts; scoring with {list(judges)} + queuing Opus")

    (out_dir / "prompts.jsonl").write_text(
        "\n".join(json.dumps({k: w[k] for k in
                  ("item_key", "query_id", "task_type", "population", "polarity", "prompt")},
                  ensure_ascii=False) for w in work) + "\n")

    # Score with the two API judges inline (identical prompt string).
    # QueryLLM is thread-safe (per-call thread_id + _usage_lock), so fan the
    # prompts out across a small thread pool; gemini latency dominates.
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0 = time.time()
    fh = (out_dir / "api_scores.jsonl").open("w")
    wlock = threading.Lock()
    done = [0]

    def _do(w):
        out = []
        for jname, jclient in judges.items():
            sc = _score(jclient, w["prompt"])
            out.append({"item_key": w["item_key"], "query_id": w["query_id"],
                        "task_type": w["task_type"], "population": w["population"],
                        "polarity": w["polarity"], "judge": jname, **sc})
        with wlock:
            for row in out:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[0] += 1
            if done[0] % 5 == 0 or done[0] == len(work):
                print(f"[judge_agree]   {done[0]}/{len(work)} prompts scored "
                      f"({time.time()-t0:.0f}s)", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_do, w) for w in work]):
            fut.result()
    fh.close()

    # Queue identical prompts for the Opus (Claude Code subagent) judge.
    batches = [work[i:i + args.opus_batch] for i in range(0, len(work), args.opus_batch)]
    for bi, batch in enumerate(batches):
        bp = out_dir / "opus_queue" / f"batch_{bi:02d}.jsonl"
        bp.write_text("\n".join(json.dumps(
            {"item_key": w["item_key"], "prompt": w["prompt"]}, ensure_ascii=False)
            for w in batch) + "\n")
    print(f"[judge_agree] wrote {len(batches)} Opus batch files to {out_dir/'opus_queue'} "
          f"({args.opus_batch}/batch). Fill each batch_NN.jsonl -> batch_NN.out.jsonl")
    print(f"[judge_agree] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
