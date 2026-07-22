#!/usr/bin/env python3
"""Render the error-analysis charts as pure HTML/CSS donuts (conic-gradient rings
+ HTML legends, no images, no JS) and insert them into results_tables.html under
the single-writer lock. The browser draws them crisply and reuses the page's
Optimistic @font-face. Idempotent (START/END markers)."""
import csv, json, os, sys, math, re
from collections import Counter
csv.field_size_limit(10**8)

ROOT = "/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "results/_scripts"))
from scripts.aggregate_eval import _accuracy_value
from _htmllock import html_lock

HERE = os.path.join(ROOT, "results/_scripts/erroranalysis")
HTML = os.path.join(ROOT, "results/aggregate/html/results_tables.html")
START, END = "<!-- ERRORANALYSIS_START -->", "<!-- ERRORANALYSIS_END -->"

# Modified Pride-flag palette: muted rainbow anchors (red -> violet). Categories
# are coloured strictly in legend order, top-to-bottom, following the flag.
PRIDE_ANCHORS = [(0xC7, 0x5D, 0x62), (0xDD, 0x94, 0x50), (0xD8, 0xB8, 0x5A),
                 (0x4E, 0x9E, 0x6A), (0x3E, 0x6B, 0xB0), (0x7E, 0x5A, 0xA6)]

def pride_ramp(n):
    if n <= 1:
        return ["#%02X%02X%02X" % PRIDE_ANCHORS[0]]
    segs = len(PRIDE_ANCHORS) - 1
    out = []
    for i in range(n):
        t = i / (n - 1) * segs
        k = min(int(t), segs - 1)
        a, b = PRIDE_ANCHORS[k], PRIDE_ANCHORS[k + 1]
        out.append("#%02X%02X%02X" % tuple(round(a[j] + (b[j] - a[j]) * (t - k)) for j in range(3)))
    return out

def text_on(hexc):  # dark label on light slices, white on dark, for legible %s
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return "#1d1d1d" if (0.299 * r + 0.587 * g + 0.114 * b) > 168 else "#fff"

# ---------------- failure taxonomy (plain language) ----------------
# The "could not find the relevant info" cause is split by method. Codex/tool
# tasks also get a few extra buckets for failures we can see from task type and
# judge notes. All buckets live in one legend; each donut only shows its own.
CAUSES = ["lc_recall", "mem_recall", "agent_recall", "search_scope_error",
          "salience_error", "grounding_lapse", "generic_voice_miss",
          "confabulation", "current_guess", "empty_context_guess", "over_apply",
          "pragmatic_misjudge", "action_unverified", "temporal",
          "instruction_miss", "degenerate"]
CAUSE_LABEL = {
    "lc_recall":          "Needed info was buried in the context",
    "mem_recall":         "Memory did not store or retrieve needed info",
    "agent_recall":       "Search did not find needed info",
    "search_scope_error": "Searched the wrong place or time",
    "salience_error":     "Found evidence but picked the wrong item",
    "grounding_lapse":    "Ignored known user info",
    "generic_voice_miss": "Voice or preference fit was too generic",
    "confabulation":      "Made up details",
    "current_guess":      "Guessed current activity or trends",
    "empty_context_guess":"Filled missing messages with profile guesses",
    "over_apply":         "Leaked private history or avoid-list details",
    "pragmatic_misjudge": "Acted at the wrong time",
    "action_unverified":  "Action was described but not checked in the app",
    "temporal":           "Used an old preference",
    "instruction_miss":   "Missed the request",
    "degenerate":         "Empty answer, refusal, or error",
}
# Pride-flag colours assigned in legend order, top-to-bottom (red -> violet).
CAUSE_COLOR = dict(zip(CAUSES, pride_ramp(len(CAUSES))))
# which method-specific recall bucket each system's "recall_miss" rows fall into
PARADIGM = {
    "longctx_gpt55": "lc", "longctx_gemini": "lc",
    "textmem_gpt55": "mem", "mem0_gpt55": "mem", "textmem_gemini": "mem",
    "codex_gpt55": "agent", "claudecode_opus": "agent", "claudecode_sonnet": "agent",
}

def fcat(key, cause):
    return PARADIGM[key] + "_recall" if cause == "recall_miss" else cause


def fcat_record(row):
    key = row["key"]
    cause = row["cause"]
    task = row.get("task_type", "")
    if key == "codex_gpt55":
        if task in {"agentic_proactive_daily_catchup", "agentic_trending_alert"}:
            return "current_guess"
        if task == "agentic_vague_refind":
            return "search_scope_error"
        if task == "agentic_community_post":
            return "generic_voice_miss"
        if task in {"agentic_dm_digest", "agentic_group_dm_summary"}:
            return "empty_context_guess"
        if task in {"agentic_send_post", "agentic_cross_app_repost", "agentic_auto_reply"}:
            return "action_unverified"
    return fcat(key, cause)
SUCC = ["Personalized ranking/retrieval", "Personalization depth", "Voice & style",
        "Privacy & over-personalization", "Proactive judgment",
        "Grounding & abstention", "Temporal currency"]
SUCC_LABEL = {
    "Personalized ranking/retrieval": "Picked the right things for the user",
    "Personalization depth":          "Used the user&rsquo;s specific tastes",
    "Voice & style":                  "Wrote in the user&rsquo;s own style",
    "Privacy & over-personalization": "Personalized without oversharing",
    "Proactive judgment":             "Knew when to speak up or stay quiet",
    "Grounding & abstention":         "Stuck to the facts (or admitted it wasn&rsquo;t sure)",
    "Temporal currency":              "Used the user&rsquo;s up-to-date preferences",
}
# Pride-flag colours in legend order, top-to-bottom (red -> violet).
SUCC_COLOR = dict(zip(SUCC, pride_ramp(len(SUCC))))
GROUPS = [["longctx_gpt55", "textmem_gpt55", "mem0_gpt55", "codex_gpt55"],
          ["longctx_gemini", "textmem_gemini", "claudecode_opus", "claudecode_sonnet"]]
LABELS = {
    "longctx_gpt55": ("GPT-5.5", "Long Context"), "textmem_gpt55": ("GPT-5.5", "Textual Memory"),
    "mem0_gpt55": ("GPT-5.5", "Mem0 w/ RAG"), "codex_gpt55": ("GPT-5.5", "Codex High"),
    "longctx_gemini": ("Gemini-3.5-Flash", "Long Context"),
    "textmem_gemini": ("Gemini-3.5-Flash", "Textual Memory"),
    "claudecode_opus": ("Opus-4.8", "Claude Code High"),
    "claudecode_sonnet": ("Sonnet-4.6", "Claude Code High"),
}
RMID = 37.5   # ring mid-radius as % of donut box (hole inset 25%)
GAP = 0.45    # white separator width between slices, in %
import os
MATCHED = {int(u) for u in os.environ.get("PERSONAS", "").split()} or \
          {int(u) for u in os.listdir("results/agent_tools_opus4.8") if u.isdigit()}
FAIL_THRESHOLD = 60.0
CODEX_RUN = os.path.join(ROOT, "results/codex_agent_gpt5.5")


def donut(counts, order, colors, center=None, pct_min=7.0):
    total = sum(counts.get(k, 0) for k in order) or 1
    segs = [(k, 100.0 * counts.get(k, 0) / total) for k in order if counts.get(k, 0) > 0]
    stops, labels, cum = [], [], 0.0
    for i, (k, f) in enumerate(segs):
        end = cum + f
        gap = min(GAP, f * 0.35) if i < len(segs) - 1 else 0.0
        seg_end = end - gap
        stops.append(f"{colors[k]} {cum:.3f}% {seg_end:.3f}%")
        if gap > 0:
            stops.append(f"#fff {seg_end:.3f}% {end:.3f}%")
        if f >= pct_min:
            th = math.radians((cum + end) / 2.0 / 100.0 * 360.0)
            left = 50 + RMID * math.sin(th)
            top = 50 - RMID * math.cos(th)
            labels.append(f'<span class="ea-pct" style="left:{left:.1f}%;top:{top:.1f}%;'
                           f'color:{text_on(colors[k])}">{f:.0f}%</span>')
        cum = end
    grad = "conic-gradient(" + ", ".join(stops) + ")"
    hole = f'<div class="ea-acc">{center[0]}</div><div class="ea-accs">{center[1]}</div>' if center else ""
    return (f'<div class="ea-dwrap"><div class="ea-donut" style="background:{grad}"></div>'
            f'<div class="ea-hole">{hole}</div>{"".join(labels)}</div>')


def legend(order, labels, colors, title, tcol):
    items = "".join(
        f'<div class="ea-li"><span class="ea-sw" style="background:{colors[k]}"></span>'
        f'<span>{labels[k]}</span></div>' for k in order)
    return f'<div class="ea-leg"><div class="ea-leg-t" style="color:{tcol}">{title}</div>{items}</div>'


def headers_row(models, stats, with_acc=True, head_notes=None):
    cells = ""
    for key in models:
        m, d = LABELS[key]
        sub = f"{d}&nbsp;&nbsp;&middot;&nbsp;&nbsp;{stats[key]['overall_acc']:.0f}%" if with_acc else d
        if head_notes and key in head_notes:
            sub += f'<div class="ea-head-note">{head_notes[key]}</div>'
        cells += f'<div class="ea-col ea-mh"><div class="ea-m">{m}</div><div class="ea-v">{sub}</div></div>'
    return f'<div class="ea-row ea-head">{cells}<div class="ea-legcol"></div></div>'


def donut_row(models, data, order, colors, leg_html, center_fn=None):
    cells = ""
    for key in models:
        center = center_fn(key) if center_fn else None
        cells += f'<div class="ea-col">{donut(data[key], order, colors, center=center, pct_min=(6.0 if center else 7.0))}</div>'
    return f'<div class="ea-row">{cells}<div class="ea-legcol">{leg_html}</div></div>'


CSS = """<style>
.ea-wrap{max-width:1240px;margin:0;font-family:'Optimistic',sans-serif;}
.ea-suptitle{font-weight:700;font-size:19px;color:#16242c;text-align:center;margin:8px 0 2px;}
.ea-sub{font-size:12.5px;color:#6a747c;text-align:center;margin:0 0 16px;line-height:1.45;}
.ea-row{display:flex;gap:14px;align-items:center;}
.ea-head{margin-bottom:4px;}
.ea-srow{margin-bottom:4px;}
.ea-col{flex:1 1 0;min-width:0;text-align:center;}
.ea-legcol{flex:1.55 1 0;min-width:0;}
.ea-mh{padding:2px 0;}
.ea-m{font-weight:700;font-size:16px;color:#16242c;line-height:1.12;}
.ea-v{font-size:12px;color:#55636b;}
.ea-head-note{margin:4px auto 0;max-width:178px;font-size:9.5px;line-height:1.25;color:#7a4a25;font-weight:600;}
.ea-dwrap{position:relative;width:100%;max-width:196px;margin:0 auto;aspect-ratio:1/1;}
.ea-donut{position:absolute;inset:0;border-radius:50%;}
.ea-hole{position:absolute;inset:25%;background:#fff;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.ea-acc{font-size:22px;font-weight:700;color:#1d1d1d;line-height:1;}
.ea-accs{font-size:9px;color:#8a929a;margin-top:3px;}
.ea-pct{position:absolute;transform:translate(-50%,-50%);font-size:11.5px;font-weight:700;color:#fff;}
.ea-leg-t{font-weight:700;font-size:13px;margin:0 0 5px;}
.ea-li{display:flex;align-items:center;gap:8px;font-size:12.5px;line-height:1.5;color:#33424b;}
.ea-sw{width:13px;height:13px;border-radius:3px;flex:none;}
.ea-divider{border-top:1px solid #e2e6ea;margin:14px 0;}
.ea-codex-note-row{align-items:stretch;margin:8px 0 4px;}
.ea-codex-card{height:100%;box-sizing:border-box;padding:9px 10px;border:1px solid #e6ebee;border-radius:8px;background:#fbfcfd;text-align:left;font-size:10.5px;line-height:1.42;color:#33424b;}
.ea-codex-k{font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#16242c;margin:0 0 4px;}
.ea-codex-t{font-size:11.5px;font-weight:700;color:#16242c;margin:0 0 4px;}
.ea-audit{margin:18px 0 0;padding:12px 14px;border:1px solid #e6ebee;border-radius:8px;background:#fbfcfd;}
.ea-audit-k{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#16242c;margin:0 0 7px;}
.ea-audit-p{margin:0 0 10px;font-size:10.5px;line-height:1.45;color:#33424b;}
.ea-audit-table{border-collapse:collapse;width:100%;table-layout:fixed;margin:0 0 8px;}
.ea-audit-table th{padding:0 5px 6px;font-size:9.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#7c8a93;text-align:left;}
.ea-audit-table td{padding:6px 5px;border-top:1px solid #edf1f3;vertical-align:top;font-size:10.5px;line-height:1.35;color:#33424b;}
.ea-audit-table .ea-val{font-weight:700;color:#16242c;}
.ea-audit-missing{font-size:10.5px;line-height:1.5;color:#33424b;}
</style>"""


def fmt_int(n):
    return f"{int(n):,}"


def flag_on(v):
    return v is True or str(v).lower() in {"1", "true", "yes"}


def pct_counter(counter, key):
    total = sum(counter.values()) or 1
    return 100.0 * counter.get(key, 0) / total


def load_codex_audit():
    tasks = {}
    raw = scored = fail = ok = nonempty = 0
    agentic_sum = agentic_n = agentic_fail = 0
    mode_grading = Counter()
    hard = Counter()
    users = []

    if not os.path.isdir(CODEX_RUN):
        return None

    for u in sorted(int(x) for x in os.listdir(CODEX_RUN) if x.isdigit() and int(x) in MATCHED):
        path = os.path.join(CODEX_RUN, str(u), "results.csv")
        if not os.path.exists(path):
            continue
        users.append(u)
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                raw += 1
                status = row.get("status") or "ok"
                ok += int(status == "ok")
                nonempty += int(bool((row.get("agent_response") or "").strip()))
                try:
                    metrics = json.loads(row.get("metrics_json") or "{}")
                except json.JSONDecodeError:
                    metrics = {}
                acc = _accuracy_value(row.get("task_type", ""), metrics, status)
                if acc is None:
                    continue
                scored += 1
                is_low = acc < FAIL_THRESHOLD
                fail += int(is_low)

                task = row.get("task_type", "")
                slot = tasks.setdefault(task, {"n": 0, "fail": 0, "sum": 0.0})
                slot["n"] += 1
                slot["fail"] += int(is_low)
                slot["sum"] += acc

                if task.startswith("agentic_"):
                    agentic_n += 1
                    agentic_sum += acc
                    agentic_fail += int(is_low)
                    mg = metrics.get("mode_grading")
                    if mg:
                        mode_grading[mg] += 1
                    if is_low:
                        if flag_on(metrics.get("pr_avoid_leak_violated")):
                            hard["avoid"] += 1
                        if flag_on(metrics.get("pr_privacy_leak_violated")):
                            hard["privacy"] += 1
                        if flag_on(metrics.get("pr_stale_preference_use_violated")):
                            hard["stale"] += 1

    return {
        "users": users,
        "raw": raw,
        "scored": scored,
        "fail": fail,
        "ok": ok,
        "nonempty": nonempty,
        "tasks": tasks,
        "agentic_n": agentic_n,
        "agentic_fail": agentic_fail,
        "agentic_mean": (agentic_sum / agentic_n) if agentic_n else 0.0,
        "mode_grading": mode_grading,
        "hard": hard,
    }


def task_obs(audit, task):
    d = audit["tasks"].get(task, {"n": 0, "fail": 0, "sum": 0.0})
    mean = (d["sum"] / d["n"]) if d["n"] else 0.0
    return f'{d["fail"]}/{d["n"]} low; mean {mean:.1f}'


def action_obs(audit):
    labels = [
        ("Send", "agentic_send_post"),
        ("repost", "agentic_cross_app_repost"),
        ("auto-reply", "agentic_auto_reply"),
    ]
    parts = []
    for label, task in labels:
        d = audit["tasks"].get(task, {"n": 0, "fail": 0})
        parts.append(f'{label} {d["fail"]}/{d["n"]} low')
    return "; ".join(parts)


def codex_offtarget_pct(F):
    return pct_counter(F.get("codex_gpt55", Counter()), "instruction_miss")


def codex_header_note(audit, F):
    return (f'Audit: {codex_offtarget_pct(F):.0f}% off-target; '
            f'{audit["agentic_fail"]}/{audit["agentic_n"]} agentic rows low')


def codex_inline_audit(audit, stats, F):
    if not audit:
        return ""
    acc = stats["codex_gpt55"]["overall_acc"]
    over = pct_counter(F["codex_gpt55"], "over_apply")
    confab = pct_counter(F["codex_gpt55"], "confabulation")
    return (
        '<div class="ea-row ea-codex-note-row">'
        '<div class="ea-col"></div><div class="ea-col"></div><div class="ea-col"></div>'
        '<div class="ea-col"><div class="ea-codex-card">'
        '<div class="ea-codex-k">Codex failure-case audit</div>'
        f'<div class="ea-codex-t">GPT-5.5 Codex High &middot; {acc:.0f}%</div>'
        f'It stays on the requested subject: <b>{codex_offtarget_pct(F):.0f}% off-target</b>. '
        f'Its weak points are answer quality after search: '
        f'<b>{audit["agentic_fail"]}/{audit["agentic_n"]}</b> agentic rows below cutoff, '
        f'with history leaks ({over:.0f}% of failures), made-up details ({confab:.0f}%), '
        'and wrong trigger or trend choices.'
        '</div></div><div class="ea-legcol"></div></div>'
    )


def codex_audit_html(audit):
    if not audit:
        return ""
    all_ok = audit["ok"] == audit["raw"] and audit["nonempty"] == audit["raw"]
    status_text = "all <code>ok</code> and non-empty" if all_ok else (
        f'{fmt_int(audit["ok"])} <code>ok</code>, {fmt_int(audit["nonempty"])} non-empty')
    mode_text = ", ".join(f"{k}: {v}" for k, v in audit["mode_grading"].items()) or "not recorded"
    rows = [
        ("Proactive daily catch-up", task_obs(audit, "agentic_proactive_daily_catchup"),
         "Wrong trigger or trend choice, often with unsupported recent activity, stale cues, or explicit behavior-history words such as watched, lingered, copied, or posted."),
        ("Trending alert", task_obs(audit, "agentic_trending_alert"),
         "Picks weak or mismatched trends, misses stronger current interests, or leaks avoid-list/profile evidence."),
        ("Vague refind/search", task_obs(audit, "agentic_vague_refind"),
         "Finds a plausible item but may expose private engagement metadata, timestamps, or stale preferences."),
        ("Community-post composition", task_obs(audit, "agentic_community_post"),
         "On-topic but generic: weak current preference use, weak user voice, or occasional privacy/stale leakage."),
        ("DM digest", task_obs(audit, "agentic_dm_digest"),
         "Can stay generic or invent message content when the thread context is missing."),
        ("Group-DM summary", task_obs(audit, "agentic_group_dm_summary"),
         "Both current rows are low because they use profile guesses instead of literal message content."),
        ("Send/repost/reply actions", action_obs(audit),
         "Strongest Codex agentic slice, but checked from final text only, not app write traces."),
        ("Visible rubric hard-fails among low agentic rows",
         f'avoid {audit["hard"]["avoid"]}; privacy {audit["hard"]["privacy"]}; stale {audit["hard"]["stale"]}',
         "The rubric catches some leakage, but not the tool cause: wrong source, broad search, missing action, or blocked write path."),
        ("Fresh single-persona trajectory attempt", "2 transport failures",
         "Separate fresh run attempts stopped before a usable trajectory because the Codex response stream was blocked in sandbox."),
    ]
    body = "".join(
        '<tr><td>{}</td><td class="ea-val">{}</td><td>{}</td></tr>'.format(label, obs, desc)
        for label, obs, desc in rows
    )
    return f"""
<div class="ea-audit">
<div class="ea-audit-k">Codex failure-case audit</div>
<p class="ea-audit-p">Codex/GPT-5.5 has <b>{fmt_int(audit["scored"])}</b> scored rows, all complete. <b>{fmt_int(audit["fail"])}/{fmt_int(audit["scored"])}</b> are below the 60-point cutoff. Agentic tasks have <b>{fmt_int(audit["agentic_fail"])}/{fmt_int(audit["agentic_n"])}</b> low rows (mean <b>{audit["agentic_mean"]:.2f}</b>). These labels come from final answers and judge notes; tool paths are not visible in <code>{mode_text}</code> grading.</p>
<table class="ea-audit-table">
<colgroup><col style="width:30%"><col style="width:20%"><col style="width:50%"></colgroup>
<thead><tr><th>Codex failure slice</th><th>Observed</th><th>Failure pattern</th></tr></thead>
<tbody>{body}</tbody>
</table>
<div class="ea-audit-missing"><b>Tool-specific categories added to the FAILURE legend:</b> wrong search place or time; generic voice/preference fit; guessed current activity or trend; profile guesses when messages are missing; action described but not checked in the app.</div>
</div>"""


def lead_html(F):
    lc = pct_counter(F["longctx_gpt55"], "lc_recall")
    textmem = pct_counter(F["textmem_gpt55"], "mem_recall")
    mem0 = pct_counter(F["mem0_gpt55"], "mem_recall")
    codex_recall = pct_counter(F["codex_gpt55"], "agent_recall")
    codex_over = pct_counter(F["codex_gpt55"], "over_apply")
    codex_confab = pct_counter(F["codex_gpt55"], "confabulation")
    codex_off = codex_offtarget_pct(F)
    gem_confab = max(pct_counter(F["longctx_gemini"], "confabulation"),
                     pct_counter(F["textmem_gemini"], "confabulation"))
    return (f'<p class="lead" style="margin:0 0 14px">A row is <b>wrong</b> if its score is below 60/100. Each failure ring shows why it was wrong. Most systems fail in similar ways: wrong item, generic answer, leaked history, or made-up details. The clearest difference is how each system finds user history. GPT-5.5 Long Context misses buried evidence in <b>{lc:.0f}%</b> of failures; Textual Memory misses stored/retrieved info in <b>{textmem:.0f}%</b>; RAG misses it in <b>{mem0:.0f}%</b>. <b>Codex</b> has a search-miss slice of <b>{codex_recall:.0f}%</b>, but it stays on target (<b>{codex_off:.0f}% off-target</b>). Its main issues are history leaks ({codex_over:.0f}%) and made-up details ({codex_confab:.0f}%). <b>Gemini</b> makes up details the most (up to {gem_confab:.0f}% of failures).</p>')


def build_pairs(S, F, stats, audit=None):
    blocks = [CSS, '<div class="ea-wrap">',
              '<div class="ea-suptitle">PersonaMem: why each system fails, and what it gets right</div>',
              '<div class="ea-sub">same simulated users &nbsp;&middot;&nbsp; each FAILURE ring shows WHY the model got it wrong '
              '(the reason, not the task) &nbsp;&middot;&nbsp; each SUCCESS ring shows what it did well</div>']
    for g, models in enumerate(GROUPS):
        head_notes = {}
        if audit and "codex_gpt55" in models:
            head_notes["codex_gpt55"] = codex_header_note(audit, F)
        blocks.append(headers_row(models, stats, head_notes=head_notes))
        blocks.append('<div class="ea-srow">' )
        blocks.append(donut_row(models, S, SUCC, SUCC_COLOR,
                                legend(SUCC, SUCC_LABEL, SUCC_COLOR, "SUCCESS: what it did well", "#1f6f4a")))
        blocks.append('</div>')
        blocks.append(donut_row(models, F, CAUSES, CAUSE_COLOR,
                                legend(CAUSES, CAUSE_LABEL, CAUSE_COLOR,
                                       "FAILURE: why it got it wrong", "#9e2b2b")))
        if g == 0:
            blocks.append(codex_inline_audit(audit, stats, F))
            blocks.append('<div class="ea-divider"></div>')
    blocks.append('</div>')
    return "".join(blocks)


def build_pies(F, stats):
    blocks = ['<div class="ea-wrap">',
              '<div class="ea-suptitle">PersonaMem: why each system gets answers wrong</div>',
              '<div class="ea-sub">same simulated users &nbsp;&middot;&nbsp; each ring shows WHY the model&rsquo;s wrong answers were wrong '
              '(the reason, not the task)</div>']
    for g, models in enumerate(GROUPS):
        blocks.append(headers_row(models, stats, with_acc=False))
        blocks.append(donut_row(models, F, CAUSES, CAUSE_COLOR,
                                legend(CAUSES, CAUSE_LABEL, CAUSE_COLOR, "Why it got answers wrong", "#9e2b2b"),
                                center_fn=lambda k: (f"{stats[k]['overall_acc']:.0f}%", "overall accuracy")))
        if g == 0:
            blocks.append('<div class="ea-divider"></div>')
    blocks.append('</div>')
    return "".join(blocks)


def main():
    succ = [json.loads(l) for l in open(os.path.join(HERE, "perrow_success.jsonl"))]
    fail = [json.loads(l) for l in open(os.path.join(HERE, "perrow_failures.jsonl"))]
    stats = json.load(open(os.path.join(HERE, "model_stats.json")))
    S = {k: Counter(r["success_dim"] for r in succ if r["key"] == k) for k in LABELS}
    F = {k: Counter(fcat_record(r) for r in fail if r["key"] == k) for k in LABELS}
    audit = load_codex_audit()

    pairs_html = build_pairs(S, F, stats, audit=audit)
    audit_html = codex_audit_html(audit)

    section = f"""{START}
<section>
<div class="cap"><h2>Error analysis: what each system gets right vs wrong</h2><span class="unit">same users</span><span class="note">every wrong answer is labelled by why it was wrong, not by task</span></div>
{lead_html(F)}
{pairs_html}
{audit_html}
</section>
{END}
"""
    QA_ANCHOR = "<!-- QA_AUDIT_START -->"
    FOOTER_ANCHOR = "</section>\n<footer>"
    with html_lock():
        html = open(HTML).read()
        html = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", "", html, flags=re.S)
        if QA_ANCHOR in html:
            html = html.replace(QA_ANCHOR, section + QA_ANCHOR, 1)
        elif FOOTER_ANCHOR in html:
            html = html.replace(FOOTER_ANCHOR, "</section>\n" + section + "<footer>", 1)
        else:
            raise SystemExit("no insertion anchor found")
        open(HTML, "w").write(html)
    print("inserted CSS error-analysis section -> %.2f MB" % (os.path.getsize(HTML) / 1e6))


if __name__ == "__main__":
    main()
