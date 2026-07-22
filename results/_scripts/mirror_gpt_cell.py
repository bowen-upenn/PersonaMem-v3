import csv,json,os
csv.field_size_limit(10**8)
P2=[1,10]; T2=["over_personalization_sensitive_event","over_personalization_chatbot_text"]
def stats(d):
    sc=[];lk=[]
    for u in P2:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f):continue
        for r in csv.DictReader(open(f)):
            if r["task_type"] not in T2:continue
            m=json.loads(r.get("metrics_json") or "{}");v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): sc.append(v*10)
            if r["task_type"]=="over_personalization_sensitive_event": lk.append(1 if m.get("pr_privacy_leak_violated") else 0)
    return (sum(sc)/len(sc) if sc else None, 100*sum(lk)/len(lk) if lk else None)
base_op,base_bl=stats("results/_reeval_minprompt/gpt_longctx")
mir_op,mir_bl =stats("results/_abl/MirrorR_gpt")
print("=== GPT Mirror cell (Long Context − posts), 2 personas ===")
print(f"  accuracy: GPT LongContext={base_op:.1f} -> Mirror={mir_op:.1f}  (delta {mir_op-base_op:+.1f})")
print(f"  blurt-rate: GPT LongContext={base_bl:.0f}% -> Mirror={mir_bl:.0f}%  (change {mir_bl-base_bl:+.0f})")
