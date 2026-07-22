import csv,json,os
csv.field_size_limit(10**8)
PERS=[1,2,3,5,6,8,9,10,13,14]
T2=["over_personalization_sensitive_event","over_personalization_chatbot_text"]
def op(d):
    sc=[]
    for u in PERS:
        f=f"{d}/{u}/results.csv"
        if not os.path.exists(f):continue
        for r in csv.DictReader(open(f)):
            if r["task_type"] not in T2:continue
            m=json.loads(r.get("metrics_json") or "{}");v=m.get("pr_combined_personalization_score")
            if isinstance(v,(int,float)): sc.append(v*10)
    return (sum(sc)/len(sc) if sc else None, len(sc))
B={"LC gpt":"results/_reeval_minprompt/gpt_longctx","LC gem":"results/_reeval_minprompt/gem_longctx",
   "TM gpt":"results/_reeval_newmem/gpt_5_5","TM gem":"results/_reeval_newmem/gemini_3_5_flash"}
A={"counts":"+ frequency counts","p3":"+ 3 emotional posts","norules":"- do-not-personalize rules","p40":"+ ~40 emotional posts"}
print("=== 10-PERSONA ablation table (OP micro %, sensitive_event + chatbot_text) ===")
bl={}
for k,d in B.items(): v,n=op(d); bl[k]=v; print(f"  {k:8} {v:6.1f}  (n={n})")
tm={"gpt":bl["TM gpt"],"gem":bl["TM gem"]}
print(f"\n  {'arm':30}{'GPT Δ':>9}{'GEM Δ':>9}   (raw: GPT / GEM)")
for ak,al in A.items():
    g,ng=op(f"results/_abl10/{ak}_gpt"); e,ne=op(f"results/_abl10/{ak}_gem")
    gd=(g-tm['gpt']) if g is not None else None; ed=(e-tm['gem']) if e is not None else None
    print(f"  {al:30}{(('%+.1f'%gd) if gd is not None else '—'):>9}{(('%+.1f'%ed) if ed is not None else '—'):>9}   ({g if g else 0:.1f} / {e if e else 0:.1f})  n={ng},{ne}")
