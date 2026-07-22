import csv,json,os
csv.field_size_limit(10**8)
P=[1,2,3,5,6,8,9,10,13,14]
OP3=["over_personalization_chatbot_text","over_personalization_sensitive_event","over_personalization_context_shift"]
def micro(d):
    sc=[]
    for u in P:
        for t in OP3:
            f=f"{d}/{u}/results.csv"
            if not os.path.exists(f):continue
            for r in csv.DictReader(open(f)):
                if r["task_type"]!=t:continue
                x=json.loads(r.get("metrics_json") or "{}").get("pr_combined_personalization_score")
                if isinstance(x,(int,float)): sc.append(x*10)
    return sum(sc)/len(sc) if sc else None
Bg=micro("results/_ablOLD/base_gpt"); Be=micro("results/_ablOLD/base_gem")
print("=== OLD-MEMORY ablation table (deltas vs base; baselines from table 1) ===")
print(f"  [delta baseline: old-mem base micro GPT {Bg:.1f} / GEM {Be:.1f}]")
print(f"  Long Context (table 1)  : GPT 75.4  GEM 57.0")
print(f"  Textual Memory (table 1): GPT 71.0  GEM 69.3   (-4.4 / +12.3)")
print(f"\n  {'arm':18}{'GPT Δ':>8}{'GEM Δ':>8}  (raw)")
def d(n,a):
    g=micro(f"results/_ablOLD/{a}_gpt"); e=micro(f"results/_ablOLD/{a}_gem")
    print(f"  {n:18}{(('%+.1f'%(g-Bg)) if g else '—'):>8}{(('%+.1f'%(e-Be)) if e else '—'):>8}  ({g if g else 0:.1f}/{e if e else 0:.1f})")
d("swap","swap"); d("+ private posts","dpriv"); d("+ everyday posts","dord"); d("+ counts","counts")
