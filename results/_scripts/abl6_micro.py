import csv,json,os
csv.field_size_limit(10**8)
P=[1,2,3,5,6,8,9,10,13,14]
OP6=["over_personalization_chatbot_text","over_personalization_sensitive_event","over_personalization_context_shift","over_personalization_repetition_recsys","over_personalization_repetition_chatbot","preference_shift_followthrough"]
AFF={"over_personalization_chatbot_text","over_personalization_sensitive_event"}
def micro(src):   # src: dir-string OR dict task->dir ; pools ALL rows (row-weighted)
    sc=[]
    for u in P:
        for t in OP6:
            d=src[t] if isinstance(src,dict) else src
            f=f"{d}/{u}/results.csv"
            if not os.path.exists(f): continue
            for r in csv.DictReader(open(f)):
                if r["task_type"]!=t: continue
                m=json.loads(r.get("metrics_json") or "{}");v=m.get("pr_combined_personalization_score")
                if isinstance(v,(int,float)): sc.append(v*10)
    return sum(sc)/len(sc) if sc else None
def lc(model):
    mp=f"results/_reeval_minprompt/{model}_longctx"
    orig="results/llm_longctx_gpt5.5" if model=="gpt" else "results/llm_longctx_gemini3.5flash_judged"
    return {t:(mp if t in AFF else orig) for t in OP6}
TMg=micro("results/_reeval_newmem/gpt_5_5"); TMe=micro("results/_reeval_newmem/gemini_3_5_flash")
LCg=micro(lc("gpt")); LCe=micro(lc("gem"))
print("=== 6-TASK micro (row-pooled) OP accuracy ===")
print(f"  Long Context : GPT {LCg:.1f}   GEM {LCe:.1f}")
print(f"  Textual Memory: GPT {TMg:.1f}   GEM {TMe:.1f}   (GPT {TMg-LCg:+.1f}, GEM {TMe-LCe:+.1f} vs LC)")
print(f"\n  {'arm':22}{'GPT Δ':>8}{'GEM Δ':>8}   (raw GPT/GEM)")
def row(name,gd,ed,gbase,ebase):
    g=micro(gd) if gd else None; e=micro(ed) if ed else None
    gs=f"{g-gbase:+.1f}" if g is not None else "—"; es=f"{e-ebase:+.1f}" if e is not None else "—"
    print(f"  {name:22}{gs:>8}{es:>8}   ({g if g else 0:.1f}/{e if e else 0:.1f})")
row("Mirror (LC - posts)","results/_abl6/mirror_gpt","results/_abl6/mirror_gem",LCg,LCe)
row("Swap (other's doc)","results/_abl6/swap_gpt","results/_abl6/swap_gem",TMg,TMe)
row("+ counts","results/_abl6/counts_gpt","results/_abl6/counts_gem",TMg,TMe)
row("+ 3 posts","results/_abl6/dose3_gpt","results/_abl6/dose3_gem",TMg,TMe)
row("+ 40 posts","results/_abl6/dose40_gpt","results/_abl6/dose40_gem",TMg,TMe)
