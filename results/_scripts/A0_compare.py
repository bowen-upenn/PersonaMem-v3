import csv,json,os,glob
csv.field_size_limit(10**8)
US=[1,10]; OP=["over_personalization_sensitive_event","over_personalization_chatbot_text","over_personalization_context_shift"]
def stat(d):
    sc=[]; lk=[]
    for u in US:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f): continue
        for r in csv.DictReader(open(f)):
            if r["task_type"] not in OP: continue
            m=json.loads(r.get("metrics_json") or "{}")
            v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): sc.append(v*10); lk.append(1 if m.get("pr_privacy_leak_violated") else 0)
    return (sum(sc)/len(sc) if sc else 0, sum(lk)/len(lk) if lk else 0, len(sc))
CELLS={
 "GPT answers GPT-doc (baseline)":   "results/_reeval_newmem/gpt_5_5",
 "GPT answers GEMINI-doc (swap)":    "results/_abl/A0_gpt_ans_gemdoc",
 "GEM answers GEM-doc (baseline)":   "results/_reeval_newmem/gemini_3_5_flash",
 "GEM answers GPT-doc (swap)":       "results/_abl/A0_gem_ans_gptdoc",
}
print("=== A0 builder x answerer swap | u1+u10 | OP micro (3 pr tasks) + leak rate ===")
print(f"{'cell':38}{'OPmicro':>8}{'leak':>7}{'n':>5}")
for name,d in CELLS.items():
    s,l,n=stat(d); print(f"{name:38}{s:>8.1f}{l:>7.2f}{n:>5}")
print("\nREAD: if score follows the ANSWERER (GPT cells alike, GEM cells alike) -> intrinsic model tendency.")
print("      if score follows the DOC (GPT-on-gem ~ GEM-on-gem) -> the document content/shape.")
