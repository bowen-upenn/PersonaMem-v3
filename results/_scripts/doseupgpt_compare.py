import csv,json,os
csv.field_size_limit(10**8)
US=[1,10]; TASKS=["over_personalization_sensitive_event","over_personalization_chatbot_text"]
def stat(d):
    sc=[];se=[]
    for u in US:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f):continue
        for r in csv.DictReader(open(f)):
            if r["task_type"] not in TASKS:continue
            m=json.loads(r.get("metrics_json") or "{}"); v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)):
                sc.append(v*10)
                if r["task_type"]=="over_personalization_sensitive_event": se.append(1 if m.get("pr_privacy_leak_violated") else 0)
    return (sum(sc)/len(sc) if sc else 0, sum(se)/len(se) if se else 0, len(sc))
print("=== ASYMMETRY TEST: stuff the summary with ~40 emotional posts — does each model leak? (u1+u10) ===\n")
print(f"{'':34}{'OPmicro':>9}{'sensEvLeak':>12}")
print("GPT-5.5:")
for name,d in [("  normal summary","results/_reeval_newmem/gpt_5_5"),("  summary STUFFED (Dose-up)","results/_abl/DoseUp_gpt")]:
    op,se,n=stat(d); print(f"{name:34}{op:>9.1f}{se:>12.2f}  (n={n})")
print("Gemini (from earlier smoke, for comparison):")
for name,d in [("  normal summary","results/_reeval_newmem/gemini_3_5_flash"),("  summary STUFFED (Dose-up)","results/_abl/DoseUp")]:
    op,se,n=stat(d); print(f"{name:34}{op:>9.1f}{se:>12.2f}  (n={n})")
print("\nREAD: Gemini stuffed -> leak up + score down. If GPT stuffed stays ~flat -> GPT is IMMUNE to content-dose (explains the asymmetry).")
