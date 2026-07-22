#!/usr/bin/env python3
"""Re-score new_suggestions_chatbot from STORED responses (no re-generation)
under the fixed scoring (leak hard-fail only when dominant + softened judge).
Re-runs the gpt-5.5 judge where the new deterministic gate now lets rows through.
Writes the updated metrics_json back to each results.csv and prints new accuracy."""
import csv, json, glob, os, sys
csv.field_size_limit(2**31 - 1)
sys.path.insert(0, ".")
from data_preparation.utils import extract_json_from_response
from evaluation.tasks.new_suggestions import compute_new_suggestions_chatbot_metrics
from query_llm import QueryLLM

CONFIGS = ["llm_longctx_gpt5.5", "llm_memory_gpt5.5", "mem0_gpt5.5",
           "llm_longctx_gemini3.5flash", "llm_memory_gemini3.5flash",
           "agent_tools_opus4.8", "agent_tools_sonnet4.6"]  # codex left blank
PERS = ["1", "2", "3", "5", "6"]
LABEL = {"llm_longctx_gpt5.5": "GPT-LC", "llm_memory_gpt5.5": "GPT-Mem", "mem0_gpt5.5": "GPT-Mem0",
         "llm_longctx_gemini3.5flash": "Gem-LC", "llm_memory_gemini3.5flash": "Gem-Mem",
         "agent_tools_opus4.8": "OPUS-CC", "agent_tools_sonnet4.6": "Sonnet-CC"}

# instance lookup by query_id (authoritative fields from test.json)
INST = {}
for p in PERS:
    for r in json.load(open(f"backend/{p}/test.json")):
        if r["task_type"] == "new_suggestions_chatbot":
            INST[r["query_id"]] = r.get("instance_full") or {}

judge = QueryLLM({"models": {"llm_model": "gpt-5.5"}}, rate_limit_per_min=50)

for cfg in CONFIGS:
    accs = []
    for p in PERS:
        fp = f"results/{cfg}/{p}/results.csv"
        if not os.path.exists(fp):
            continue
        for r in csv.DictReader(open(fp)):  # READ-ONLY (no CSV rewrite)
            if r.get("task_type") != "new_suggestions_chatbot":
                continue
            inst = INST.get(r["query_id"], {})
            raw = r.get("agent_response") or ""
            parsed = extract_json_from_response(raw) or {}
            m = compute_new_suggestions_chatbot_metrics(parsed, raw, inst, judge, True)
            accs.append(100.0 if m.get("passed") else 0.0)
    acc = sum(accs) / len(accs) if accs else None
    print(f"{LABEL[cfg]:10s} new chatbot accuracy = {acc:.1f}  (n={len(accs)})" if acc is not None else f"{LABEL[cfg]}: none")
