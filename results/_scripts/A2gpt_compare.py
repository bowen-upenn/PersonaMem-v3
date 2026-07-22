import csv,json,os
csv.field_size_limit(10**8)
US=[1,10]; OP=["over_personalization_sensitive_event","over_personalization_chatbot_text","over_personalization_context_shift"]
def stat(d):
    sc=[];lk=[]
    for u in US:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f):continue
        for r in csv.DictReader(open(f)):
            if r["task_type"] not in OP:continue
            m=json.loads(r.get("metrics_json") or "{}"); v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): sc.append(v*10); lk.append(1 if m.get("pr_privacy_leak_violated") else 0)
    return (sum(sc)/len(sc) if sc else 0, sum(lk)/len(lk) if lk else 0, len(sc))
print("=== GPT side: does flattening the to-do-list in GPT's summary calm it down? (u1+u10) ===")
print(f"{'arm':40}{'OPmicro':>8}{'leak':>7}{'n':>5}")
for name,d in [("GPT reads its normal summary (baseline)","results/_reeval_newmem/gpt_5_5"),
               ("GPT reads de-listed summary (A2_gpt)","results/_abl/A2_gpt")]:
    s,l,n=stat(d); print(f"{name:40}{s:>8.1f}{l:>7.2f}{n:>5}")
print("(for reference, GPT on the full raw diary scored ~76.9)")
