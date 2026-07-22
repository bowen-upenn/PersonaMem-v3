#!/usr/bin/env python3
"""Recover per-row failure classifications from the workflow agent transcripts,
join task_type, fold in the deterministic assignments, and build:
  - perrow_failures.jsonl  : every failing row -> {key, idx, task_type, axis, category}
  - perrow_success.jsonl   : every passing row -> {key, task_type, axis, success_dim}
Both keyed to the SAME 9 shared dimensions for symmetric success/failure pies.
"""
import csv, json, os, glob, re
csv.field_size_limit(10**8)
ROOT = "/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"
import sys; sys.path.insert(0, ROOT)
from scripts.aggregate_eval import _accuracy_value
from evaluation.task_registry import get_capability_axis

HERE = os.path.join(ROOT, "results/_scripts/erroranalysis")
WF = "/vast/home/b/bwjiang/.claude/projects/-vast-projects-cjtaylor-occam-bwjiang-PersonaMem-v3/ae19265c-09b7-4b87-9e4b-32ee660be97a/subagents/workflows/wf_519ff5a4-660"
MATCHED = {1, 2, 3, 5, 6, 8, 9, 10, 13, 14}
FAIL = 60.0

RUNS = {
    "longctx_gpt55":   "results/llm_longctx_gpt5.5_judged",
    "textmem_gpt55":   "results/llm_memory_gpt5.5",
    "mem0_gpt55":      "results/mem0_gpt5.5",
    "codex_gpt55":     "results/codex_agent_gpt5.5",
    "longctx_gemini":  "results/llm_longctx_gemini3.5flash_judged",
    "textmem_gemini":  "results/llm_memory_gemini3.5flash_judged",
    "claudecode_opus": "results/agent_tools_opus4.8",
    "claudecode_sonnet":"results/agent_tools_sonnet4.6",
}

# ---- 9 shared dimensions (failure category string -> dimension) ----
FAILCAT_TO_DIM = {
    "Retrieval / ranking miss":            "Personalized ranking/retrieval",
    "Off-target response":                 "Task targeting (on-topic)",
    "Under-personalization / generic":     "Personalization depth",
    "Voice / style mismatch":              "Voice & style",
    "Over-personalization / privacy leak": "Privacy & over-personalization",
    "Stale / contradicted preference":     "Temporal currency",
    "Hallucination / failed abstention":   "Grounding & abstention",
    "Restraint / proactive misfire":       "Proactive judgment",
    "Non-substantive / error":             "Substantive delivery",
}

# ---- FIRST-PRINCIPLES cognitive cause (why the model erred) ----
# The failure CATEGORY says what went wrong; the CAUSE says the mechanism.
# The dominant "ranking miss" splits by whether ANY truly-relevant item even
# surfaced (recall@5): none surfaced -> a recall/retrieval failure (the model
# never brought the user's real prefs to bear); some surfaced but ranked low ->
# a salience/prioritisation failure (it had them, weighed the wrong things).
CAUSE_BY_CAT = {
    "Off-target response":                 "instruction_miss",
    "Under-personalization / generic":     "grounding_lapse",
    "Voice / style mismatch":              "grounding_lapse",
    "Over-personalization / privacy leak": "over_apply",
    "Stale / contradicted preference":     "temporal",
    "Hallucination / failed abstention":   "confabulation",
    "Restraint / proactive misfire":       "pragmatic_misjudge",
    "Non-substantive / error":             "degenerate",
}


def fp_cause(cat, task, m):
    if cat == "Retrieval / ranking miss":
        if task in PURE_RANKING:
            return "recall_miss" if (m.get("recall_at_5") or 0) == 0 else "salience_error"
        return "recall_miss"   # agentic re-find / judge-noted "couldn't surface it"
    return CAUSE_BY_CAT[cat]

# ---- success: task_type -> dimension it demonstrates when passed ----
RANKING = {"personalized_recommendation", "hidden_persona_recommendation",
           "personalized_search_ranking", "at_ai_directive_followup",
           "agentic_vague_refind", "new_suggestions_recsys", "new_suggestions_chatbot"}
AGENTIC_CONTENT = {"agentic_send_post", "agentic_community_post",
                   "agentic_cross_app_repost", "agentic_auto_reply",
                   "agentic_trending_alert", "agentic_proactive_daily_catchup",
                   "agentic_dm_digest", "agentic_group_dm_summary",
                   "agentic_draft_audit", "agentic_wrong_recipient_check",
                   "chatbot_personalized_response"}
# NB: success_dim must agree with how that task's FAILURES are classified, so the
# success/failure pies share one dimension axis. Verified failure landing zones:
#   over_personalization_*           -> 76% Privacy        => Privacy
#   restraint_sensitive_event_silence-> 97% Proactive      => Proactive (NOT Privacy)
#   proactive_* / mistake_prevention -> 97-98% Proactive   => Proactive
PRIVACY = {"over_personalization_chatbot_text", "over_personalization_context_shift",
           "over_personalization_sensitive_event", "over_personalization_repetition_chatbot",
           "over_personalization_repetition_recsys", "over_personalization_sycophancy"}
PROACTIVE = {"proactive_close_friend_update", "proactive_friend_feed_react",
             "proactive_overactive_check", "proactive_trending_feed_react",
             "active_mistake_prevention", "restraint_sensitive_event_silence"}
GROUNDING = {"personal_qa_hallucination"}
CURRENCY = {"local_recommendation_geo_shift", "preference_shift_followthrough",
            "short_vs_long_term_lifecycle"}
DEPTH = {"hidden_persona_implicit_qa"}


def success_dim(task, m):
    if task in RANKING:
        return "Personalized ranking/retrieval"
    if task in AGENTIC_CONTENT:
        pd = m.get("pr_primary_dim")
        if pd == "voice_match":
            return "Voice & style"
        return "Personalization depth"
    if task in PRIVACY:
        return "Privacy & over-personalization"
    if task in PROACTIVE:
        return "Proactive judgment"
    if task in GROUNDING:
        return "Grounding & abstention"
    if task in CURRENCY:
        return "Temporal currency"
    if task in DEPTH:
        return "Personalization depth"
    return "Personalization depth"


# ---------- 1. recover LLM per-row classifications from transcripts ----------
recovered = {}   # (key, idx) -> category
for jf in glob.glob(os.path.join(WF, "agent-*.jsonl")):
    key = None
    classifications = []
    for line in open(jf):
        try:
            o = json.loads(line)
        except Exception:
            continue
        msg = o.get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            mm = re.search(r"to_classify_(\w+)\.jsonl", content)
            if mm:
                key = mm.group(1)
        elif isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "text" and key is None:
                    mm = re.search(r"to_classify_(\w+)\.jsonl", blk.get("text", ""))
                    if mm:
                        key = mm.group(1)
                if blk.get("type") == "tool_use" and blk.get("name") == "StructuredOutput":
                    classifications = blk.get("input", {}).get("classifications", classifications)
    if key and classifications:
        for c in classifications:
            if isinstance(c, dict) and "idx" in c and "category" in c:
                recovered[(key, c["idx"])] = c["category"]

print("recovered LLM classifications:", len(recovered))

# map key -> {idx: task_type} from to_classify files (idx space == LLM rows)
llm_idx_task = {}
for key in RUNS:
    d = {}
    for line in open(os.path.join(HERE, f"to_classify_{key}.jsonl")):
        r = json.loads(line)
        d[r["idx"]] = r["task_type"]
    llm_idx_task[key] = d

# ---------- 2. rebuild ALL rows, assign final failure category OR success dim ----------
ERR_STATUS = {"failed_writes", "failed_quality", "error", "no_result", "timeout"}
PURE_RANKING = {"personalized_recommendation", "hidden_persona_recommendation",
                "personalized_search_ranking", "new_suggestions_recsys",
                "new_suggestions_chatbot"}

fail_out = open(os.path.join(HERE, "perrow_failures.jsonl"), "w")
succ_out = open(os.path.join(HERE, "perrow_success.jsonl"), "w")
unresolved = 0
align_mismatch = [0]
n_fail = n_succ = 0
for key, d in RUNS.items():
    dd = os.path.join(ROOT, d)
    # we must re-walk rows in the SAME order build_failures used to map LLM idx.
    # build_failures appended failing rows in user-sorted, file order; split_for_llm
    # then assigned LLM idx in that same order skipping deterministic rows.
    users = sorted(int(u) for u in os.listdir(dd) if u.isdigit() and int(u) in MATCHED)
    llm_counter = 0
    for u in users:
        f = os.path.join(dd, str(u), "results.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            m = json.loads(r["metrics_json"] or "{}")
            status = r["status"] or "ok"
            acc = _accuracy_value(r["task_type"], m, status)
            if acc is None:
                continue
            task = r["task_type"]
            axis = get_capability_axis(task)
            if acc >= FAIL:
                succ_out.write(json.dumps({"key": key, "task_type": task, "axis": axis,
                                           "success_dim": success_dim(task, m)}) + "\n")
                n_succ += 1
                continue
            # failing row — replicate split_for_llm's deterministic logic in order
            if status in ERR_STATUS or m.get("non_substantive_response") or m.get("response_is_substantive") is False:
                cat = "Non-substantive / error"
            elif task in PURE_RANKING:
                cat = "Retrieval / ranking miss"
            else:
                # alignment check: the task_type recorded for this llm idx in
                # to_classify must equal the task we're on, else ordering drifted
                exp_task = llm_idx_task[key].get(llm_counter)
                if exp_task is not None and exp_task != task:
                    align_mismatch[0] += 1
                cat = recovered.get((key, llm_counter))
                llm_counter += 1
                if cat is None:
                    unresolved += 1
                    cat = "Retrieval / ranking miss"  # safe fallback (rare)
            fail_out.write(json.dumps({"key": key, "idx_llm": llm_counter - 1,
                                       "task_type": task, "axis": axis,
                                       "category": cat,
                                       "dim": FAILCAT_TO_DIM[cat],
                                       "cause": fp_cause(cat, task, m)}) + "\n")
            n_fail += 1
fail_out.close(); succ_out.close()
print(f"failures written: {n_fail}  (unresolved LLM joins: {unresolved})")
print(f"successes written: {n_succ}")
print(f"alignment mismatches (should be 0): {align_mismatch[0]}")
