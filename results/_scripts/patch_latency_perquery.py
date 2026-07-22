#!/usr/bin/env python
"""Rebuild the Latency (s/query) table from CURRENT data, per standalone query.

Two value bugs are fixed at once, ONLY in the Latency <section>:
  1. Repetition tasks (c1c/c1d) bundle n_total sequential sub-calls into one
     row, so their stored duration is a CLUSTER SUM (5-6x a query) — divide by
     n_total.
  2. The gemini-mem column was rendered from older (≈2x slower) data and is
     stale across every cell — recompute from current results.csv.
Every cell = mean over the evaluated personas of (duration_ms/1000)/n_total;
Overall = micro mean over all rows. best = LOWEST latency. Shade = linear
white -> rgb(176,209,243) (the legend endpoint) by value/max. Other sections
(Accuracy / tokens / cost) are untouched.
"""
import os
import csv, glob, json, re
csv.field_size_limit(10**9)

HTML = "results/aggregate/html/results_tables.html"
MATCHED = {int(u) for u in os.environ.get("PERSONAS", "").split()} or \
          {int(u) for u in os.listdir("results/agent_tools_opus4.8") if u.isdigit()}
COLS = ["llm_longctx_gpt5.5_judged", "llm_memory_gpt5.5", "mem0_gpt5.5",
        "codex_agent_gpt5.5", "llm_longctx_gemini3.5flash_judged",
        "llm_memory_gemini3.5flash_judged", "agent_tools_opus4.8",
        "agent_tools_sonnet4.6"]
VDIV_IDX = {3, 5}
MAXRGB = (176, 209, 243)  # legend blue endpoint

# The `_judged` configs were built via --replay_from (re-score saved responses
# WITHOUT calling the model), so their duration is replay+judge time, NOT
# generation time — objective tasks collapse to ~0.1s (no judge) and judge
# tasks report judge latency. For LATENCY only, source these 3 columns from
# their real-generation (judge-off) dirs. Accuracy etc. stay from the judged
# dirs (unaffected — this map is consulted only in latency()).
LATENCY_DIR = {
    "llm_longctx_gpt5.5_judged": "llm_longctx_gpt5.5",
    "llm_longctx_gemini3.5flash_judged": "llm_longctx_gemini3.5flash",
    "llm_memory_gemini3.5flash_judged": "llm_memory_gemini3.5flash",
}

# HTML row label -> task_type (None = Overall, micro over all rows)
LABEL2TASK = {
    "Overall": None,
    "Personalized chatbot response": "chatbot_personalized_response",
    "Local recommendation after geo shift": "local_recommendation_geo_shift",
    "Personal-fact hallucination probe": "personal_qa_hallucination",
    "Proactive feed ranking": "personalized_recommendation",
    "@AI directive follow-up": "at_ai_directive_followup",
    "Hidden-persona recommendation": "hidden_persona_recommendation",
    "Generic chatbot restraint": "over_personalization_chatbot_text",
    "Sensitive-event chatbot restraint": "over_personalization_sensitive_event",
    "Do-not-personalize follow-up": "over_personalization_context_shift",
    "Repetitive feed personalization": "over_personalization_repetition_recsys",
    "Repetitive chatbot personalization": "over_personalization_repetition_chatbot",
    "QA on preference changes": "preference_shift_followthrough",
    "Community voice draft": "agentic_community_post",
    "DM inbox digest": "agentic_dm_digest",
    "Cross-app repost adaptation": "agentic_cross_app_repost",
    "Personalized DM reply": "agentic_auto_reply",
    "Vague memory refind": "agentic_vague_refind",
    "Cross-surface post composition": "agentic_send_post",
    "Proactive daily catch-up": "agentic_proactive_daily_catchup",
    "Personalized trend alert": "agentic_trending_alert",
    "Close-friend DM update": "proactive_close_friend_update",
    "Sensitive-event silence": "restraint_sensitive_event_silence",
    "Friend-post update": "proactive_friend_feed_react",
    "Trending-topic surfacing": "proactive_trending_feed_react",
    "Mistake-prevention alert": "active_mistake_prevention",
    "Idle-moment silence": "proactive_overactive_check",
    "QA on hidden personas": "hidden_persona_implicit_qa",
}


def latency(mode, task):
    vals = []
    src = LATENCY_DIR.get(mode, mode)
    for p in glob.glob(f"results/{src}/*/results.csv"):
        if int(p.split("/")[-2]) not in MATCHED:
            continue
        for r in csv.DictReader(open(p)):
            if task is not None and r.get("task_type") != task:
                continue
            try:
                d = float(r.get("duration_ms") or 0) / 1000.0
            except Exception:
                continue
            if d <= 0:
                continue
            try:
                n = max(1, int(json.loads(r.get("metrics_json") or "{}").get("n_total") or 1))
            except Exception:
                n = 1
            vals.append(d / n)
    return sum(vals) / len(vals) if vals else None


def main():
    html = open(HTML).read()
    s0 = html.index("<h2>Latency</h2>")
    s1 = html.index("</section>", s0)
    sec = html[s0:s1]

    # Compute all cells, find the global max for shading.
    data = {lab: [latency(m, task) for m in COLS] for lab, task in LABEL2TASK.items()}
    mx = max(v for vs in data.values() for v in vs if v is not None)

    def shade(v):
        f = max(0.0, min(1.0, v / mx))
        return tuple(round(255 + (c - 255) * f) for c in MAXRGB)

    def cells(vals):
        best = min(range(len(vals)), key=lambda i: (vals[i] if vals[i] is not None else 1e9))
        out = []
        for i, v in enumerate(vals):
            cls = "val" + (" best" if i == best else "") + (" vdiv" if i in VDIV_IDX else "")
            if v is None:
                out.append(f'<td class="{cls} na">&ndash;</td>')
            else:
                r, g, b = shade(v)
                out.append(f'<td class="{cls}" style="background:rgb({r},{g},{b});color:#243039">{v:.1f}s</td>')
        return "".join(out)

    for lab, vals in data.items():
        i = sec.index(f">{lab}</td>")
        rs = sec.rfind("<tr", 0, i)
        re_ = sec.index("</tr>", i) + len("</tr>")
        old = sec[rs:re_]
        cut = old.index(f">{lab}</td>") + len(f">{lab}</td>")
        sec = sec[:rs] + old[:cut] + cells(vals) + "</tr>" + sec[re_:]

    open(HTML, "w").write(html[:s0] + sec + html[s1:])
    # report
    print(f"max latency (shade ref) = {mx:.1f}s")
    for lab in ("Overall", "Repetitive feed personalization", "Repetitive chatbot personalization",
                "Personalized DM reply", "Idle-moment silence"):
        vs = data[lab]
        print(f"  {lab:36s} " + "  ".join(
            f"{c.split('_')[0][:4]}={v:.1f}" if v is not None else f"{c.split('_')[0][:4]}=-"
            for c, v in zip(COLS, vs)))


if __name__ == "__main__":
    main()
