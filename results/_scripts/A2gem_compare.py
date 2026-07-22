import csv,json,os
csv.field_size_limit(10**8)
US=[1,10]; OP=["over_personalization_sensitive_event","over_personalization_chatbot_text","over_personalization_context_shift"]
def stat(d,task=None):
    sc=[];lk=[]
    for u in US:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f):continue
        for r in csv.DictReader(open(f)):
            if r["task_type"] not in OP:continue
            if task and r["task_type"]!=task: continue
            m=json.loads(r.get("metrics_json") or "{}"); v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): sc.append(v*10); lk.append(1 if m.get("pr_privacy_leak_violated") else 0)
    return (sum(sc)/len(sc) if sc else 0, sum(lk)/len(lk) if lk else 0, len(sc))
ARMS={"neutral + boundaries (baseline)":"results/_reeval_newmem/gemini_3_5_flash",
      "RAW text + boundaries (A1a)":"results/_abl/A1a",
      "neutral - boundaries (A2)":"results/_abl/A2_gem",
      "RAW text - boundaries (A1a+A2)":"results/_abl/A1a_A2"}
print("=== Gemini memory 2x2: content (neutral/raw) x boundaries (on/off) | u1+u10 ===")
print(f"{'arm':36}{'OPmicro':>8}{'leak':>7}{'sensEvLeak':>11}{'n':>5}")
for name,d in ARMS.items():
    s,l,n=stat(d); _,sl,_=stat(d,"over_personalization_sensitive_event")
    print(f"{name:36}{s:>8.1f}{l:>7.2f}{sl:>11.2f}{n:>5}")
print("\nREAD: leak appears in (RAW - boundaries) only -> needs BOTH (content+no-guard). leak in both -boundary -> guards protect.")
