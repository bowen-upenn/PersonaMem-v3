import csv,json,os
csv.field_size_limit(10**8)
TASKS=["over_personalization_chatbot_text","over_personalization_sensitive_event","over_personalization_context_shift"]
def load(arm):
    f=f"results/_smoke_count/gemini_{arm}/1/results.csv"; out={}
    if not os.path.exists(f): return out
    for r in csv.DictReader(open(f)):
        m=json.loads(r.get("metrics_json") or "{}")
        out[r["query_id"]]={"task":r["task_type"],
            "score":(m.get("pr_combined_personalization_score") or 0)*10,
            "priv":1 if m.get("pr_privacy_leak_violated") else 0,
            "resp":r["agent_response"] or "","judge":m.get("pr_judge_reasoning","")}
    return out
nc,wc=load("nocount"),load("withcount")
print("=== SMOKE: does giving the memory the occurrence COUNT make Gemini over-personalize more? ===")
print(f"persona 1 | matched queries | NO-count memory  vs  WITH-count memory\n")
print(f"{'task':40}{'n':>3}{'pr NO':>8}{'pr WITH':>9}{'Δ':>7}{'leak NO':>9}{'leakWITH':>9}")
for t in TASKS:
    qs=[q for q in nc if nc[q]['task']==t and q in wc]
    if not qs: continue
    def avg(d,k): return sum(d[q][k] for q in qs)/len(qs)
    print(f"{t:40}{len(qs):>3}{avg(nc,'score'):>8.1f}{avg(wc,'score'):>9.1f}{avg(wc,'score')-avg(nc,'score'):>+7.1f}{avg(nc,'priv'):>9.2f}{avg(wc,'priv'):>9.2f}")
allq=[q for q in nc if q in wc]
def micro(d,k): return sum(d[q][k] for q in allq)/len(allq)
print(f"{'OP micro (all 3)':40}{len(allq):>3}{micro(nc,'score'):>8.1f}{micro(wc,'score'):>9.1f}{micro(wc,'score')-micro(nc,'score'):>+7.1f}{micro(nc,'priv'):>9.2f}{micro(wc,'priv'):>9.2f}")
# the two queries that LEAKED on long-context — did WITH-count now leak vs no-count?
print("\n=== the 2 breakup queries that leaked on long-context ===")
for q in ["1:0070:sensitive_event_1_breakup_00_row01_q0","1:0094:sensitive_event_1_breakup_00_row02_q0"]:
    if q in nc and q in wc:
        print(f"\nquery {q.split(':')[1]}: NO-count pr={nc[q]['score']:.0f} leak={nc[q]['priv']}  |  WITH-count pr={wc[q]['score']:.0f} leak={wc[q]['priv']}")
        print(f"  WITH-count answer: {wc[q]['resp'][:300]}")
        print(f"  WITH-count judge : {str(wc[q]['judge'])[:240]}")
