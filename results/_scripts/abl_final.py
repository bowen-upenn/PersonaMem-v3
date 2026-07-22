import csv,json,os
csv.field_size_limit(10**8)
P=[1,2,3,5,6,8,9,10,13,14]
OP3=["over_personalization_chatbot_text","over_personalization_sensitive_event","over_personalization_context_shift"]
def micro(src):
    sc=[]
    for u in P:
        for t in OP3:
            d=src[t] if isinstance(src,dict) else src
            f=f"{d}/{u}/results.csv"
            if not os.path.exists(f):continue
            for r in csv.DictReader(open(f)):
                if r["task_type"]!=t:continue
                x=json.loads(r.get("metrics_json") or "{}").get("pr_combined_personalization_score")
                if isinstance(x,(int,float)): sc.append(x*10)
    return sum(sc)/len(sc) if sc else None
def lc(mk):
    mp=f"results/_reeval_minprompt/{'gpt' if mk=='gpt' else 'gem'}_longctx"; cs=f"results/_abl6/lcfull_cs_{mk}"
    return {"over_personalization_chatbot_text":mp,"over_personalization_sensitive_event":mp,"over_personalization_context_shift":cs}
LCg=micro(lc("gpt"));LCe=micro(lc("gem"));TMg=micro("results/_reeval_newmem/gpt_5_5");TMe=micro("results/_reeval_newmem/gemini_3_5_flash")
print("=== FINAL 3-task micro table ===")
print(f"  Long Context : GPT {LCg:.1f}  GEM {LCe:.1f}")
print(f"  Textual Memory: GPT {TMg:.1f}  GEM {TMe:.1f}   ({TMg-LCg:+.1f}/{TMe-LCe:+.1f})")
def d(n,gd,ed,gb,eb):
    g=micro(gd);e=micro(ed); print(f"  {n:20}{(('%+.1f'%(g-gb)) if g else '—'):>8}{(('%+.1f'%(e-eb)) if e else '—'):>8}  ({g if g else 0:.1f}/{e if e else 0:.1f})")
print("  STRONG drivers:")
d("swap (other doc)","results/_abl6/swap_gpt","results/_abl6/swap_gem",TMg,TMe)
d("+ SENSITIVE posts","results/_abl10/p40_gpt","results/_abl10/p40_gem",TMg,TMe)
print("  ruled out:")
d("+ ordinary posts","results/_abl6/dose40_gpt","results/_abl6/dose40_gem",TMg,TMe)
d("+ counts","results/_abl6/counts_gpt","results/_abl6/counts_gem",TMg,TMe)
