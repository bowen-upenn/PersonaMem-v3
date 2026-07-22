#!/usr/bin/env python3
"""Split each model's failures into deterministic vs LLM-classified.

Deterministic (no LLM needed, the metric *is* the failure mode):
  - cat 9 "Non-substantive / error": status in the failed/error set, or the
    runner flagged the response non-substantive.
  - cat 1 "Retrieval / ranking miss": pure objective ranking tasks
    (personalized_recommendation, hidden_persona_recommendation,
    personalized_search_ranking) — valid-but-wrong orderings (low ndcg/hit).

Everything else (all judge-text rows + the remaining structured-but-textless
rows such as proactive / mistake-prevention / geo / lifecycle / fatigue) goes
to the LLM, which reads judge_text + metric_hint + response and assigns ONE of
the 9 categories.

Outputs:
  to_classify_<key>.jsonl   one compact row per LLM row, with a stable `idx`
  deterministic_<key>.json  {category: count} for the pre-assigned rows
  manifest.json             [{key,label,path,n_llm}] for the workflow
"""
import json, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    "longctx_gpt55":   "Long Context · GPT-5.5",
    "textmem_gpt55":   "Textual Memory · GPT-5.5",
    "mem0_gpt55":      "Mem0 w/ RAG · GPT-5.5",
    "codex_gpt55":     "Codex High · GPT-5.5",
    "longctx_gemini":  "Long Context · Gemini-3.5-Flash",
    "textmem_gemini":  "Textual Memory · Gemini-3.5-Flash",
    "claudecode_opus": "Claude Code High · Opus-4.8",
    "claudecode_sonnet":"Claude Code High · Sonnet-4.6",
}
ORDER = ["longctx_gpt55", "textmem_gpt55", "mem0_gpt55", "codex_gpt55",
         "longctx_gemini", "textmem_gemini", "claudecode_opus", "claudecode_sonnet"]

ERR_STATUS = {"failed_writes", "failed_quality", "error", "no_result", "timeout"}
PURE_RANKING = {"personalized_recommendation", "hidden_persona_recommendation",
                "personalized_search_ranking", "new_suggestions_recsys",
                "new_suggestions_chatbot"}

CAT_RETRIEVAL = "Retrieval / ranking miss"
CAT_ERROR = "Non-substantive / error"


def det_category(r):
    m = r["metric_hint"]
    if r["status"] in ERR_STATUS:
        return CAT_ERROR
    if m.get("non_substantive_response") or m.get("response_is_substantive") is False:
        return CAT_ERROR
    if r["task_type"] in PURE_RANKING:
        return CAT_RETRIEVAL
    return None


def main():
    manifest = []
    for key in ORDER:
        recs = [json.loads(l) for l in open(os.path.join(HERE, f"failures_{key}.jsonl"))]
        det_counts = {}
        llm_rows = []
        for r in recs:
            c = det_category(r)
            if c is not None:
                det_counts[c] = det_counts.get(c, 0) + 1
            else:
                llm_rows.append(r)
        # write compact LLM rows with stable idx
        out = os.path.join(HERE, f"to_classify_{key}.jsonl")
        with open(out, "w") as fh:
            for i, r in enumerate(llm_rows):
                fh.write(json.dumps({
                    "idx": i,
                    "task_type": r["task_type"],
                    "axis": r["axis"],
                    "accuracy": r["accuracy"],
                    "status": r["status"],
                    "judge_text": r["judge_text"][:950],
                    "metric_hint": r["metric_hint"],
                    "response": r["response"][:380],
                }, ensure_ascii=False) + "\n")
        json.dump(det_counts, open(os.path.join(HERE, f"deterministic_{key}.json"), "w"), indent=1)
        manifest.append({"key": key, "label": LABELS[key],
                         "path": out, "n_llm": len(llm_rows),
                         "n_total_fail": len(recs),
                         "n_det": sum(det_counts.values())})
        print(f"{key:20s} total_fail={len(recs):4d}  det={sum(det_counts.values()):4d} "
              f"{det_counts}  ->LLM={len(llm_rows):4d}")
    json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    print("\nLLM rows total:", sum(x["n_llm"] for x in manifest))


if __name__ == "__main__":
    main()
