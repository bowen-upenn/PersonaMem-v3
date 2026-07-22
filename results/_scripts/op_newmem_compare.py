import csv, json, glob
csv.field_size_limit(10**8)
PERS=[1,2,3,5,6,8,9,10,13,14]
OP=["over_personalization_chatbot_text","over_personalization_sensitive_event","over_personalization_context_shift"]
def load(root, task, abs_dir=False):
    vals=[]
    pat=f"{root}/*/results.csv" if abs_dir else f"{root}/*/results.csv"
    for f in glob.glob(pat):
        for r in csv.DictReader(open(f)):
            if r["task_type"]!=task: continue
            try:m=json.loads(r.get("metrics_json") or "{}")
            except:continue
            v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): vals.append(v*10)
    return vals
def mean(x): return sum(x)/len(x) if x else 0
CFG=[("GPT-5.5", "results/_reeval_newmem/gpt_5_5", "results/llm_memory_gpt5.5"),
     ("Gemini",  "results/_reeval_newmem/gemini_3_5_flash", "results/llm_memory_gemini3.5flash_ownmem")]
print("=== OVER-PERSONALIZATION: new memory vs old memory (pr_combined %, all personas) ===")
print(f"{'model':9}{'task':40}{'OLD mem':>9}{'NEW mem':>9}{'Δ':>8}")
for label,newd,oldd in CFG:
    for t in OP:
        o=mean(load(oldd,t)); n=mean(load(newd,t))
        print(f"{label:9}{t:40}{o:>9.1f}{n:>9.1f}{n-o:>+8.1f}")
    # OP micro
    on=[v for t in OP for v in load(newd,t)]; oo=[v for t in OP for v in load(oldd,t)]
    print(f"{label:9}{'OP micro (3 pr tasks)':40}{mean(oo):>9.1f}{mean(on):>9.1f}{mean(on)-mean(oo):>+8.1f}\n")
