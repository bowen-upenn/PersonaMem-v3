import csv,json,os
csv.field_size_limit(10**8)
US=[1,10]; TASKS=["over_personalization_sensitive_event","over_personalization_chatbot_text"]
def stat(d):
    sc=[];lk=[]
    for u in US:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f):continue
        for r in csv.DictReader(open(f)):
            if r["task_type"] not in TASKS:continue
            m=json.loads(r.get("metrics_json") or "{}"); v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): sc.append(v*10); lk.append((r["task_type"],1 if m.get("pr_privacy_leak_violated") else 0))
    op=sum(sc)/len(sc) if sc else 0
    se=[x for t,x in lk if t=="over_personalization_sensitive_event"]
    return op,(sum(se)/len(se) if se else 0),len(sc)
print("=== FAST SMOKE: content-vs-form (Gemini, u1+u10, sensitive_event+chatbot_text) ===\n")
print("MIRROR (remove emotional posts from the DIARY -> does blurting stop?)")
for name,d in [("  baseline raw diary","results/_reeval_minprompt/gem_longctx"),("  diary w/ sensitive posts REMOVED","results/_abl/MirrorR")]:
    op,se,n=stat(d); print(f"  {name:36} OPmicro={op:5.1f}  sensEvLeak={se:.2f}  n={n}")
print("\nDOSE-UP (stuff emotional posts INTO the summary -> does blurting start?)")
for name,d in [("  baseline normal summary","results/_reeval_newmem/gemini_3_5_flash"),("  summary STUFFED w/ emotional posts","results/_abl/DoseUp")]:
    op,se,n=stat(d); print(f"  {name:36} OPmicro={op:5.1f}  sensEvLeak={se:.2f}  n={n}")
print("\nREAD: Mirror leak->0 = it's the content. DoseUp leak appears = content/volume. Both null = it's the FORM.")
