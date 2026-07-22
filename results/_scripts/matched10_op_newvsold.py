import csv,json,glob,os
csv.field_size_limit(10**8)
PERS=[1,2,3,5,6,8,9,10,13,14]
OP=["over_personalization_chatbot_text","over_personalization_sensitive_event","over_personalization_context_shift"]
def load(d,task,personas):
    vals=[]
    for u in personas:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f): continue
        for r in csv.DictReader(open(f)):
            if r["task_type"]!=task: continue
            m=json.loads(r.get("metrics_json") or "{}")
            v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): vals.append(v*10)
    return vals
def mean(x): return sum(x)/len(x) if x else 0
def pset(d): return set(int(os.path.basename(os.path.dirname(f))) for f in glob.glob(f"{d}/*/results.csv"))
for label,newd,oldd in [("GPT-5.5","results/_reeval_newmem/gpt_5_5","results/llm_memory_gpt5.5"),
                        ("Gemini","results/_reeval_newmem/gemini_3_5_flash","results/llm_memory_gemini3.5flash_ownmem")]:
    matched=sorted(set(PERS)&pset(newd)&pset(oldd))
    print(f"\n=== {label}: new memory vs old memory | MATCHED {len(matched)} personas {matched} ===")
    print(f"  {'task':40}{'OLD':>7}{'NEW':>7}{'Δ':>7}")
    on=[];oo=[]
    for t in OP:
        o=mean(load(oldd,t,matched)); n=mean(load(newd,t,matched))
        on+=load(newd,t,matched); oo+=load(oldd,t,matched)
        print(f"  {t:40}{o:>7.1f}{n:>7.1f}{n-o:>+7.1f}")
    print(f"  {'OP micro (3 pr tasks)':40}{mean(oo):>7.1f}{mean(on):>7.1f}{mean(on)-mean(oo):>+7.1f}")
