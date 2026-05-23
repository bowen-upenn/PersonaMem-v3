#!/usr/bin/env python3
"""Stage D: render the unified per-persona benchmark analysis HTML report.

Reads:
  benchmark/benchmark_analysis/query_stats.json
  benchmark/benchmark_analysis/schema_audit.json
  benchmark/benchmark_analysis/samples/{uid}.json
  backend/persona_analysis/similarity_matrix.json
  backend/persona_analysis/per_user/{uid}_qualitative.json
  backend/persona_analysis/synthesis.md  (optional)

Writes:
  benchmark/benchmark_analysis/report.html  (single self-contained file)
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.task_registry import TASK_TYPE_META  # noqa: E402

USER_IDS = ["105", "115", "229", "282", "760"]
BACKEND_DIR = REPO_ROOT / "backend"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
PA_DIR = BACKEND_DIR / "persona_analysis"
BA_DIR = BENCHMARK_DIR / "benchmark_analysis"
OUT_HTML = BA_DIR / "report.html"

# App accents (mirror visualize.py)
APP_COLOR = {
    "instagram": "#C13584",
    "facebook": "#4A6FA5",
    "threads": "#636366",
    "chatbot": "#8B5CF6",
    "ai_studio": "#16A34A",
    "": "#86868B",
}

# Family colors
FAMILY_COLOR = {
    "chatbot_response": "#8B5CF6",
    "over_personalization": "#B04050",
    "personalization": "#3B82F6",
    "new_suggestions": "#0EA5E9",
    "agentic": "#F59E0B",
    "e_followup": "#10B981",
    "proactive_actions": "#EC4899",
    "": "#86868B",
}


def esc(s) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def short_md_to_html(md: str) -> str:
    """Very small markdown → HTML converter (h1-h4, ul, p, inline code/bold/em).

    No external deps. Good enough for synthesis.md.
    """
    if not md:
        return "<p><em>No synthesis available.</em></p>"
    lines = md.splitlines()
    out: list[str] = []
    in_ul = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def inline(s: str) -> str:
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
        return s

    para_buf: list[str] = []

    def flush_para():
        if para_buf:
            out.append(f"<p>{' '.join(para_buf)}</p>")
            para_buf.clear()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_para()
            close_ul()
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush_para()
            close_ul()
            level = len(m.group(1))
            out.append(f"<h{level+1}>{inline(m.group(2))}</h{level+1}>")
            continue
        m = re.match(r"^[-*]\s+(.*)$", line)
        if m:
            flush_para()
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline(m.group(1))}</li>")
            continue
        close_ul()
        para_buf.append(inline(line))
    flush_para()
    close_ul()
    return "\n".join(out)


def heat_color(v: float, lo: float = 0.0, hi: float = 1.0) -> str:
    """Map a value to a soft heatmap color (green high → light → red low for similarity).

    For similarity matrices we want high = warm/saturated, low = pale.
    """
    if hi == lo:
        return "#F2F2F7"
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    # interpolate between pale (#F7F7F5) and the saturated end (#10B981 = green)
    # Lighter implementation: blend white → forest green
    r = int(247 + (16 - 247) * t)
    g = int(247 + (185 - 247) * t)
    b = int(245 + (129 - 245) * t)
    return f"rgb({r},{g},{b})"


def count_color(c: int, max_c: int) -> str:
    """Blue intensity for count heatmap (0 = white, max = saturated blue)."""
    if max_c == 0:
        return "#F7F7F5"
    t = min(1.0, c / max_c)
    r = int(247 + (74 - 247) * t)
    g = int(247 + (111 - 247) * t)
    b = int(245 + (165 - 245) * t)
    return f"rgb({r},{g},{b})"


def truncate(s, n=240):
    s = str(s or "")
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def load_inputs() -> dict:
    stats = json.loads((BA_DIR / "query_stats.json").read_text("utf-8"))
    audit = json.loads((BA_DIR / "schema_audit.json").read_text("utf-8"))
    sim = json.loads((PA_DIR / "similarity_matrix.json").read_text("utf-8"))

    samples = {}
    for uid in USER_IDS:
        p = BA_DIR / "samples" / f"{uid}.json"
        samples[uid] = json.loads(p.read_text("utf-8")) if p.exists() else []

    qualitative = {}
    for uid in USER_IDS:
        p = PA_DIR / "per_user" / f"{uid}_qualitative.json"
        qualitative[uid] = json.loads(p.read_text("utf-8")) if p.exists() else None

    synthesis_path = PA_DIR / "synthesis.md"
    synthesis = synthesis_path.read_text("utf-8") if synthesis_path.exists() else ""

    return {
        "stats": stats,
        "audit": audit,
        "sim": sim,
        "samples": samples,
        "qualitative": qualitative,
        "synthesis": synthesis,
    }


# --------------------------------------------------------------- CSS

CSS = """
:root {
  --bg: #F7F7F5;
  --bg-card: #FFFFFF;
  --text: #1D1D1F;
  --text-secondary: #86868B;
  --text-tertiary: #AEAEB2;
  --border: #E5E5EA;
  --radius: 10px;
  --shadow: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-hover: 0 2px 8px rgba(0,0,0,0.07);
  --font: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
  --green: #10B981;
  --red: #DC2626;
  --amber: #F59E0B;
  --gray: #9CA3AF;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }
.container { max-width: 1200px; margin: 0 auto; padding: 56px 24px; }
.header { margin-bottom: 40px; }
.header h1 { font-size: 32px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
.header .meta { color: var(--text-secondary); font-size: 13px; }
.section { margin-bottom: 48px; }
.section-title { font-size: 18px; font-weight: 700; letter-spacing: -0.2px; margin-bottom: 18px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--text); }
.section-title .hint { font-weight: 400; font-size: 12px; color: var(--text-secondary); margin-left: 10px; font-style: italic; }
.profile-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 14px; box-shadow: var(--shadow); }
.row { display: flex; gap: 14px; flex-wrap: wrap; }
.user-tile { flex: 1 1 200px; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; box-shadow: var(--shadow); min-width: 180px; }
.user-tile h3 { font-size: 14px; font-weight: 700; margin-bottom: 8px; }
.user-tile .uid { color: var(--text-tertiary); font-weight: 500; font-size: 11px; letter-spacing: 0.5px; }
.kv { display: flex; justify-content: space-between; align-items: baseline; margin: 4px 0; font-size: 12px; }
.kv .k { color: var(--text-secondary); }
.kv .v { font-weight: 600; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px; background: #F2F2F7; color: var(--text-secondary); margin-right: 4px; margin-bottom: 3px; letter-spacing: 0.1px; }
.pill.green { background: rgba(16,185,129,0.10); color: #047857; }
.pill.red { background: rgba(220,38,38,0.10); color: #991B1B; }
.pill.amber { background: rgba(245,158,11,0.12); color: #92400E; }
.pill.gray { background: #F2F2F7; color: var(--text-secondary); }
.bar { display: inline-block; height: 6px; border-radius: 3px; background: var(--green); vertical-align: middle; }
table { border-collapse: collapse; font-size: 12px; }
.heatmap { overflow-x: auto; }
.heatmap table { min-width: 100%; font-size: 11px; }
.heatmap th, .heatmap td { padding: 5px 6px; text-align: center; border: 1px solid var(--border); white-space: nowrap; font-variant-numeric: tabular-nums; }
.heatmap th { background: #F2F2F7; font-weight: 600; position: sticky; top: 0; }
.heatmap th.rotated { writing-mode: vertical-rl; text-orientation: mixed; transform: rotate(180deg); height: 160px; padding: 6px 4px; font-weight: 500; font-size: 10px; }
.heatmap td.label { font-weight: 700; background: #FAFAFA; text-align: left; padding: 6px 10px; }
/* Vertical heatmap layout — rows = task_types, narrow user_id columns + Σ column */
.heatmap-vert table { font-size: 12px; }
.heatmap-vert th.row-label-head { text-align: left; padding: 8px 14px; min-width: 280px; background: #F2F2F7; font-size: 12px; }
.heatmap-vert th.uid-col { min-width: 64px; width: 64px; padding: 8px 6px; font-size: 12px; }
.heatmap-vert th.total-col { min-width: 70px; width: 70px; padding: 8px 8px; background: #E5E5EA; font-size: 12px; }
.heatmap-vert td.label { font-family: ui-monospace, SFMono-Regular, "SF Mono", monospace; font-size: 11.5px; font-weight: 600; padding: 6px 14px; background: #FAFAFA; color: var(--text); min-width: 280px; }
.heatmap-vert td { padding: 6px 8px; }
.heatmap-vert td.total-col { font-weight: 700; background: #F2F2F7; min-width: 70px; }
.sim-matrix { display: inline-block; vertical-align: top; margin-right: 18px; margin-bottom: 14px; }
.sim-matrix h4 { font-size: 12px; font-weight: 600; margin-bottom: 6px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.4px; }
.sim-matrix table { font-size: 11px; }
.sim-matrix td { padding: 6px 9px; border: 1px solid var(--border); font-variant-numeric: tabular-nums; text-align: center; min-width: 40px; }
.sim-matrix td.label { font-weight: 700; background: #FAFAFA; }
.persona-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 22px; box-shadow: var(--shadow); }
.persona-panel .headline { font-size: 20px; font-weight: 700; letter-spacing: -0.2px; margin-bottom: 4px; }
.persona-panel .uid-line { color: var(--text-secondary); font-size: 12px; font-family: ui-monospace, SFMono-Regular, monospace; margin-bottom: 12px; }
.persona-panel .distinctive { font-size: 13px; color: var(--text); margin-bottom: 14px; line-height: 1.6; }
.persona-panel .demo-line { font-size: 11px; color: var(--text-secondary); margin-bottom: 6px; }
.hp-list { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
.hp-chip { font-size: 11px; padding: 4px 10px; border-radius: 6px; background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe; }
.hp-chip.privacy { background: #FEF3C7; color: #92400E; border-color: #FDE68A; }
.validity-bar { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0 14px 0; padding: 10px 14px; background: #FAFAFA; border-radius: 8px; border: 1px solid #F2F2F7; font-size: 11px; }
.validity-bar .axis { display: flex; flex-direction: column; gap: 2px; }
.validity-bar .axis .ax-name { font-size: 10px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.4px; font-weight: 600; }
.validity-bar .axis .ax-bar { display: flex; align-items: center; gap: 6px; }
.validity-bar .axis .ax-bar .bar-outer { width: 80px; height: 6px; border-radius: 3px; background: #E5E5EA; overflow: hidden; }
.validity-bar .axis .ax-bar .bar-inner { height: 100%; background: var(--green); border-radius: 3px; }
.samples { display: flex; flex-direction: column; gap: 10px; }
.sample-card { background: #FAFAFA; border: 1px solid var(--border); border-left: 3px solid var(--gray); border-radius: 8px; padding: 12px 14px; font-size: 12px; }
.sample-card.fail-1, .sample-card.fail-multi { border-left-color: var(--red); background: #FEF2F2; }
.sample-card.weak { border-left-color: var(--amber); background: #FFFBEB; }
.sample-card.valid { border-left-color: var(--green); background: #F0FDF4; }
.sample-card.pending { border-left-color: var(--gray); background: #F9FAFB; opacity: 0.85; }
.sample-card .sample-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.sample-card .qid { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 10px; color: var(--text-tertiary); }
.sample-card .tt { font-weight: 700; }
.sample-card .fam { font-size: 10px; color: var(--text-secondary); padding: 1px 6px; border-radius: 4px; background: #F2F2F7; }
.sample-card .app { font-size: 10px; padding: 1px 6px; border-radius: 4px; color: #fff; }
.sample-card .axis-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; padding: 2px 7px; border-radius: 10px; cursor: help; font-weight: 600; }
.sample-card .axis-badge.pass { background: rgba(16,185,129,0.10); color: #047857; }
.sample-card .axis-badge.fail { background: rgba(220,38,38,0.12); color: #991B1B; }
.sample-card .axis-badge.pending { background: #F2F2F7; color: var(--text-secondary); }
.sample-card .axis-badge.na { background: #F2F2F7; color: var(--text-tertiary); }
.sample-card .qtext { background: #fff; padding: 8px 10px; border: 1px solid var(--border); border-radius: 5px; margin: 6px 0; font-size: 12px; line-height: 1.5; }
.sample-card .body { display: grid; grid-template-columns: 1fr; gap: 6px; margin-top: 6px; }
.sample-card .body-row { font-size: 11px; }
.sample-card .body-row .label { font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.3px; font-size: 9px; }
.sample-card .body-row .val { display: block; background: #fff; padding: 6px 8px; border-radius: 4px; border: 1px solid #F2F2F7; margin-top: 2px; white-space: pre-wrap; word-break: break-word; color: var(--text); }
.sample-card .body-row.example .val { background: #F0FDF4; border-color: #BBF7D0; }
.sample-card .body-row.inferior .val { background: #FEF3C7; border-color: #FDE68A; }
.sample-card .body-row.gt .val { background: #EFF6FF; border-color: #BFDBFE; }
.sample-card .why { font-size: 10px; color: var(--text-secondary); margin-top: 4px; font-style: italic; }
.sample-card details { margin-top: 8px; font-size: 11px; }
.sample-card summary { cursor: pointer; color: var(--text-secondary); font-weight: 500; padding: 2px 0; }
.tooltip { position: relative; }
.tooltip:hover::after { content: attr(data-note); position: absolute; left: 50%; transform: translateX(-50%); bottom: 120%; background: #1F2937; color: #fff; font-size: 10px; padding: 5px 8px; border-radius: 4px; white-space: nowrap; max-width: 260px; white-space: normal; z-index: 100; }
.bar-chart { display: flex; align-items: flex-end; gap: 6px; height: 120px; padding: 8px 0; border-bottom: 1px solid var(--border); }
.bar-chart .bcol { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; gap: 4px; min-width: 50px; }
.bar-chart .bcol .br { width: 30px; border-radius: 4px 4px 0 0; }
.bar-chart .bcol .bv { font-size: 10px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.bar-chart .bcol .bl { font-size: 9px; color: var(--text-tertiary); }
.synthesis-block { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px 28px; box-shadow: var(--shadow); font-size: 13px; line-height: 1.7; }
.synthesis-block h2, .synthesis-block h3 { font-weight: 700; letter-spacing: -0.2px; margin: 14px 0 8px 0; }
.synthesis-block h2 { font-size: 16px; }
.synthesis-block h3 { font-size: 14px; }
.synthesis-block p { margin-bottom: 10px; color: var(--text); }
.synthesis-block ul { margin: 8px 0 14px 22px; }
.synthesis-block li { margin: 4px 0; }
.synthesis-block code { background: #F2F2F7; padding: 1px 5px; border-radius: 3px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.92em; }
.notes-block { background: #FFF8E1; border: 1px solid #FDE68A; padding: 10px 14px; border-radius: 6px; font-size: 12px; color: #78350F; margin-top: 8px; }
.method-note { background: #EFF6FF; border-left: 3px solid #3B82F6; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 0 0 20px 0; font-size: 12.5px; color: #1E3A5F; line-height: 1.65; }
.method-note .mn-label { display: inline-block; font-weight: 700; color: #1E3A8A; text-transform: uppercase; letter-spacing: 0.6px; font-size: 10.5px; margin-right: 6px; padding: 1px 7px; border-radius: 3px; background: rgba(59,130,246,0.14); }
.method-note p { margin: 0 0 10px 0; color: #1E3A5F; }
.method-note p:last-child { margin-bottom: 0; }
.method-note strong { color: #1E3A8A; font-weight: 600; }
.method-note em { color: #1E3A8A; font-style: italic; }
.method-note code { background: rgba(59,130,246,0.10); padding: 1px 5px; border-radius: 3px; font-family: ui-monospace,SFMono-Regular,monospace; font-size: 0.9em; color: #1E3A8A; }
.method-note ul { margin: 6px 0 10px 22px; }
.method-note li { margin: 3px 0; color: #1E3A5F; }
"""


METHOD_NOTE_OVERVIEW = """
<div class="method-note">
<p><span class="mn-label">Input</span> For each of the five user_ids I read two artifacts. The benchmark file <code>benchmark/{uid}/queries.csv</code> carries one row per generated query, with structured columns (<code>task_family</code>, <code>task_type</code>, <code>rubric_tags</code>, <code>ts</code>, <code>app_context</code>, …) and a nested <code>instance_json</code> payload holding the actual user query, example response, inferior response, and ground-truth preference. The five app-event JSONs <code>backend/{uid}/{instagram,facebook,threads,chatbot,ai_studio}.json</code> were also read — but only their <code>source_hashtags</code> fields, to build a per-user lifetime hashtag set used as the evidence-trace reference.</p>
<p><span class="mn-label">Analysis</span> I walked each row in <code>queries.csv</code>, JSON-decoded its <code>instance_json</code>, and asked three deterministic questions per row. <strong>(1) Has the row been LLM-postprocessed?</strong> — yes iff the <code>instance_json</code> carries both <code>example_response</code> and <code>inferior_response</code>; the downstream postprocess step is what writes those fields, so their absence pinpoints exactly which rows are still scaffolding (proactive- and repetition-family rows use different schemas and are treated as postprocessed by default since they don't use that pair). <strong>(2) Does the row pass the Axis-4 structural check?</strong> — the full per-category rule set is described in the Axis-4 heatmap note below; the tile-level "Axis-4 pass" is just the count of rows that satisfied it. <strong>(3) Does the row trace back to user history?</strong> — I recursively pulled every hashtag-looking string out of <code>instance_json</code> (e.g., <code>candidates[*].hashtags</code>, <code>held_out_preference.source_hashtags</code>, <code>directive_hashtags</code>, raw <code>#tags</code> embedded in free text) and intersected it with the user's lifetime hashtag set; "evidence trace" is the fraction of checkable rows whose intersection was non-empty.</p>
<p><span class="mn-label">Output</span> Each tile shows the four headline numbers per user with pills colored by fixed thresholds (<code>&gt;90%</code> green, <code>&gt;30%</code> amber, otherwise red). "Hashtag universe" is the size of that user's deduped lifetime hashtag set — the denominator behind every evidence-trace check.</p>
</div>
"""


METHOD_NOTE_FAMILY = """
<div class="method-note">
<p><span class="mn-label">Input</span> The CSV <code>task_family</code> column across all five users' <code>queries.csv</code> files. Every row carries exactly one family — one of <code>chatbot_response</code>, <code>over_personalization</code>, <code>personalization</code>, <code>new_suggestions</code>, <code>agentic</code>, <code>e_followup</code>, or <code>proactive_actions</code> — defined canonically in <code>evaluation/task_registry.py::TASK_TYPE_META</code>.</p>
<p><span class="mn-label">Analysis</span> I tallied a single global histogram by family — one count per family, summed across all 945 rows, ignoring user_id. The intent here is the highest-level question: <em>what kind of test is this benchmark, in aggregate?</em> Per-user breakdowns live in the two heatmaps below; this chart is purely the bird's-eye view.</p>
<p><span class="mn-label">Output</span> One bar per family, labeled with its absolute count. Heights scale relative to the largest family so the tallest bar reaches ≈110px and shorter ones scale proportionally. Family colors come from a small fixed palette that's reused on the per-persona sample cards lower in the report, so the eye can follow a family from chart to card.</p>
</div>
"""


METHOD_NOTE_COUNT_HEATMAP = """
<div class="method-note">
<p><span class="mn-label">Input</span> The same per-user <code>queries.csv</code> files as the overview, but grouped by <code>(user_id, task_type)</code> pairs this time. The <code>task_type</code> column is more granular than <code>task_family</code> — it carries one of 30 canonical names from <code>TASK_TYPE_META</code> (e.g., <code>personalized_recommendation</code>, <code>chatbot_personalized_response</code>, <code>agentic_dm_digest</code>, <code>at_ai_directive_followup</code>, <code>over_personalization_sensitive_event</code>, …).</p>
<p><span class="mn-label">Analysis</span> I built a nested counter <code>per_user_task_type[uid][task_type]</code> and rendered it as a 5-row × 30-column table. To make the table interpretable, columns are sorted by <em>global</em> popularity (the most-emitted task_type leftmost), so the densest part of the table sits at the left edge. Cell color is computed as <code>count / max_count_across_table</code>, fading from white (zero) to saturated blue (the largest single cell). Empty cells aren't missing data — they mean that user genuinely has zero rows of that task_type. This happens for data-dependent tasks like <code>local_recommendation_geo_shift</code> (only emitted when the user actually changes city in their event history) or <code>over_personalization_sensitive_event</code> (only emitted inside an active <code>sensitive_life_event</code> window).</p>
<p><span class="mn-label">Output</span> A glanceable map of which task_types are evenly distributed across users versus which are user-specific. A row of mostly-empty cells is a per-user coverage gap; an unusually saturated column is a task_type the generator favored. Use this to spot what a planned eval will and won't get coverage on.</p>
</div>
"""


METHOD_NOTE_AXIS4_HEATMAP = """
<div class="method-note">
<p><span class="mn-label">Input</span> The same <code>(user_id, task_type)</code> grid as the count heatmap above, but the cell value is now the Axis-4 <em>pass rate</em> — <code>passing_rows / total_rows</code> — instead of the raw count. The inputs to each row's pass/fail verdict are the full row record (CSV columns + decoded <code>instance_json</code>) from <code>queries.csv</code>.</p>
<p><span class="mn-label">Analysis</span> For every row I ran a deterministic structural validator implementing the verifier's Axis 4: <em>does this row carry all the fields required for its task category to be a valid personalization test?</em> The required field set is different per category, because different task_types use different <code>instance_json</code> schemas. The full rule set (implemented in <code>axis_4_check()</code> in <code>build_stats.py</code>):</p>
<ul>
  <li>Every row must have a non-empty <code>rubric_tags</code> CSV column.</li>
  <li><strong>Pair-scored tasks</strong> (<code>chatbot_response</code>, most <code>over_personalization</code> variants, <code>e_followup</code>, <code>agentic</code>, <code>personalization</code>, <code>new_suggestions</code>) need an <code>example_response</code>, an <code>inferior_response</code>, AND a ground-truth preference under any canonical name (<code>groundtruth_preference</code> / <code>groundtruth</code> / <code>held_out_preference</code> / <code>target_pref</code> / <code>gt_slice</code>).</li>
  <li><strong>Repetition tasks</strong> (<code>over_personalization_repetition_recsys</code> and <code>_chatbot</code>) use a multi-query schema instead — they need <code>queries[]</code> + <code>target_pref</code>.</li>
  <li><strong>Proactive-family rows</strong> use a JITAI-card schema — they need <code>expected_behavior</code> + <code>trigger_evidence</code> + <code>jitai_card</code> + <code>tool_call_rules</code>.</li>
  <li><strong>Ranking tasks</strong> (<code>personalized_recommendation</code>, <code>at_ai_directive_followup</code>, <code>new_suggestions_recsys</code>) additionally need a candidate slate (<code>candidates[]</code> with ≥5 entries) and a held-out index (<code>held_out_idx</code> or <code>positive_indices</code>).</li>
  <li><strong>Agentic-family rows</strong> additionally need <code>tool_call</code> or <code>tool_call_rules</code>.</li>
</ul>
<p>I deliberately did NOT require a surface user query for task_types whose surface naturally has none (ranking probes, agent-composed writes, proactive briefings) — those are listed in <code>NO_EXPLICIT_QUERY_TASKS</code> in <code>build_stats.py</code>, otherwise they'd all spuriously fail. After running the validator on every row I aggregated <code>pass / total</code> per <code>(user, task_type)</code> cell.</p>
<p><span class="mn-label">Output</span> Cells color from white (0%) to green (100%). A user's row turning systematically red means their benchmark is structurally incomplete — typically because the LLM postprocess step that fills in <code>example_response</code> / <code>inferior_response</code> hasn't run for that user yet. This is exactly what you see for users 105 / 229 / 282 / 760; user 115 is the only one whose postprocess has run, reaching 97% pass.</p>
</div>
"""


METHOD_NOTE_SIMILARITY = """
<div class="method-note">
<p><span class="mn-label">Input</span> Each user's <code>backend/{uid}/profile.json</code> — the master persona blob the data-prep pipeline emits. I pulled six clusters of fields out of it: (a) categorical demographics (<code>gender</code>, <code>race_ethnicity</code>, <code>career</code>, <code>education</code>); (b) Big-Five trait levels (5 categorical scores <code>high</code> / <code>medium</code> / <code>low</code>) plus MBTI 4-letter type; (c) the <code>user_voice</code> surface fields (<code>emoji_palette</code>, <code>formality_baseline</code>, <code>default_capitalization</code>, <code>humor_tone</code>, <code>emoji_intensity_default</code>); (d) the user's list of <code>hidden_personas</code>, each carrying a <code>type</code> from a fixed 12-element vocabulary; (e) the top-20 most-repeated hashtags from <code>exploration_exploitation.top_repeated_hashtags</code>; and (f) the free-text <code>hidden_persona_summary</code> paragraph the pipeline writes as a synthesis of the user's deep motivations.</p>
<p><span class="mn-label">Analysis</span> I picked six feature dimensions designed to capture orthogonal aspects of "how similar are two personas?" and computed a 5×5 pairwise similarity matrix for each. The four categorical dimensions use set Jaccard (<code>|A∩B| / |A∪B|</code>): <strong>demographic</strong> Jaccard runs over the 4-element string set <code>{gender, race_ethnicity, career, education}</code>; <strong>hidden-persona</strong> Jaccard runs over the set of <code>hidden_personas[].type</code> strings each user has; <strong>hashtag-interests</strong> Jaccard runs over the top-20 hashtag set (case-normalized, leading <code>#</code> stripped); <strong>voice</strong> averages four sub-scores — <code>emoji_palette</code> Jaccard, formality distance-to-similarity (<code>1 − |Δformality|</code>), and equality matches on <code>default_capitalization</code> and <code>emoji_intensity_default</code>. The <strong>personality</strong> matrix averages two scores: cosine on the Big-Five vector after mapping <code>high</code> / <code>medium</code> / <code>low</code> to <code>1.0</code> / <code>0.5</code> / <code>0.0</code>, and Hamming similarity on the MBTI 4-letter string (matching positions / 4). The <strong>semantic</strong> dimension is the only one using a real embedding model: I ran each user's <code>hidden_persona_summary</code> paragraph through <code>sentence-transformers/all-MiniLM-L6-v2</code> via the existing <code>evaluation/metrics.py::embed</code> helper and took the cosine of the normalized 384-dim vectors. The <strong>combined</strong> matrix is the equal-weight mean of all six.</p>
<p><span class="mn-label">Output</span> Six 5×5 matrices plus a combined one, all symmetric with 1.0 on the diagonal (verified by a Stage-A invariant test). The combined matrix answers "which personas cluster?"; the per-dimension matrices answer <em>why</em>. For example, users 105 and 760 cluster largely on shared MBTI (both INFJ) and voice, not on demographics — you can see that by comparing their cells in the personality and voice matrices versus the demographic matrix.</p>
</div>
"""


METHOD_NOTE_PANELS = """
<div class="method-note">
<p><span class="mn-label">Input</span> Each panel below is the output of a separately-launched general-purpose Claude subagent — five subagents in parallel, one per user_id. Each subagent received two files. The first is the user's full <code>backend/{uid}/profile.json</code> — every demographic, voice, hidden-persona, hashtag, and personality field the pipeline emitted — so its rubric application is grounded in <em>that specific persona</em>, not in generic priors. The second is <code>benchmark/benchmark_analysis/samples/{uid}.json</code>, a stratified-random subset of that user's queries that Stage A wrote out for review. The sampling rule guarantees coverage: I take <code>min(2, available)</code> rows per <code>task_type</code> that the user has any rows of (so every represented task_type contributes at least one sample), then uniformly top up toward 30 (seed = 42 for reproducibility). Because several users have many distinct task_types, the per-task-type minimum pushes them above 30 — actual subset sizes ran from 51 to 58 rows per user.</p>
<p><span class="mn-label">Analysis</span> Each subagent's prompt embedded the full validity rubric verbatim and gave it four jobs. <strong>First</strong>, read the profile carefully — internalize the persona before grading anything, so judgements about leakage and naturalness are persona-aware (e.g., "does the inferior response actually fit this user's voice?"). <strong>Second</strong>, walk every sampled query and grade it on axes 1, 2, and 3 of the rubric: <em>axis 1</em> (does the query require user context AND avoid leaking the answer in the prompt itself?), <em>axis 2</em> (is the <code>example_response</code> natural, and does it use the GT preference implicitly rather than telegraphing it with phrases like "I know you like X"?), <em>axis 3</em> (is the <code>inferior_response</code> natural, comparable in length and format to the example, and does it actually fail on the specific tested capability — missing preference, stale signal, wrong rank, etc.?). When <code>example_response</code> or <code>inferior_response</code> are absent in <code>instance_json</code> (the postprocess gap), axes 2-3 are explicitly marked <code>postprocess_pending</code>; for proactive-family rows that use a JITAI-card schema instead of the example/inferior pair, axes 2-3 are marked <code>not_applicable</code>. <strong>Third</strong>, pick 3-6 illustrative queries that best showcase that persona's testing challenges across task families. <strong>Fourth</strong>, write a ≤12-word headline plus a one-paragraph distinctive_features summary. The full grading was written to <code>backend/persona_analysis/per_user/{uid}_qualitative.json</code>.</p>
<p><span class="mn-label">Output</span> The renderer reads each <code>_qualitative.json</code> together with the deterministic Stage-A <code>schema_audit.json</code> and assembles the panel: headline + distinctive_features paragraph + hidden-persona chips (privacy-flagged types styled amber) + a validity bar summarizing axes 1-3 pass rates across the user's sampled subset + the actual sample-query cards. Each card carries five traffic-light badges: three from the subagent (axes 1-3, with hover tooltips showing the subagent's short note), one Axis-4 schema badge from the Stage-A deterministic check (NOT the subagent), and one ⊕/⊖ evidence-trace badge. Cards are sorted failing-first so problems jump to the top of each panel.</p>
</div>
"""


METHOD_NOTE_SYNTHESIS = """
<div class="method-note">
<p><span class="mn-label">Input</span> A sixth general-purpose Claude subagent received the prior wave's five <code>_qualitative.json</code> files plus the three deterministic Stage-A/B artifacts: <code>query_stats.json</code> (per-user totals + family/type matrices + overall task_type ranking), <code>schema_audit.json</code> (per-row Axis-4 + evidence-trace verdicts, plus the compact <code>per_user_task_type_pass_rate</code> summary), and <code>similarity_matrix.json</code> (the six dimension matrices + per-user demographic/voice/hidden-persona snapshots).</p>
<p><span class="mn-label">Analysis</span> The synthesis subagent's job was deliberately <em>synthesis</em>, not repetition — its prompt told it not to rewrite what was already in the per-user JSONs, but to reason across them. Five concrete targets were specified: (a) identify the strongest discriminator dimension in the similarity matrix and back the claim with a concrete user_id pair and number; (b) summarize the validity state across all five users, leading with the postprocess gap if present; (c) name the task_types with the lowest Axis-4 pass rates across multiple users (draw from <code>per_user_task_type_pass_rate</code>); (d) call out 3-5 individual query_ids worth manual review, pulling from each user's illustrative or anomalous flags; and (e) write 3-5 prioritized next-step recommendations.</p>
<p><span class="mn-label">Output</span> The subagent wrote <code>backend/persona_analysis/synthesis.md</code> (≈750 words, ≈44 lines). The renderer parses that markdown with a small in-house converter (h2/h3 headings, paragraphs, bulleted lists, inline <code>code</code> / <code>**bold**</code> / <code>*italic*</code>; no external markdown library, so <code>report.html</code> stays self-contained and offline-viewable) and drops the result below.</p>
</div>
"""


# --------------------------------------------------------------- per-section


def render_header(stats: dict, audit: dict) -> str:
    total = stats["grand_total"]
    n_users = len(stats["user_ids"])
    pp_total = sum(audit["per_user"][u]["n_postprocessed"] for u in stats["user_ids"])
    return f"""
<div class="header">
  <h1>PersonaMem-v3 — Benchmark Analysis</h1>
  <div class="meta">{n_users} personas · {total} total queries · {pp_total} ({100*pp_total/max(1,total):.0f}%) LLM-postprocessed</div>
</div>
"""


def render_overview(stats: dict, audit: dict) -> str:
    tiles = []
    for uid in stats["user_ids"]:
        n = audit["per_user"][uid]["n_rows"]
        pp = audit["per_user"][uid]["n_postprocessed"]
        ax = audit["per_user"][uid]["n_pass"]
        ev_c = audit["per_user"][uid]["n_evidence_checked"]
        ev_p = audit["per_user"][uid]["n_evidence_pass"]
        hs = audit["per_user"][uid]["user_hashtag_universe_size"]
        pp_pct = 100 * pp / max(1, n)
        pp_cls = "green" if pp_pct > 90 else ("amber" if pp_pct > 30 else "red")
        ax_pct = 100 * ax / max(1, n)
        ax_cls = "green" if ax_pct > 90 else ("amber" if ax_pct > 30 else "red")
        ev_pct = 100 * ev_p / max(1, ev_c)
        ev_cls = "green" if ev_pct > 90 else ("amber" if ev_pct > 50 else "red")
        tiles.append(f"""
<div class="user-tile">
  <h3><span class="uid">USER</span> {uid}</h3>
  <div class="kv"><span class="k">Total queries</span><span class="v">{n}</span></div>
  <div class="kv"><span class="k">Postprocessed</span><span class="v"><span class="pill {pp_cls}">{pp}/{n} ({pp_pct:.0f}%)</span></span></div>
  <div class="kv"><span class="k">Axis-4 pass</span><span class="v"><span class="pill {ax_cls}">{ax}/{n} ({ax_pct:.0f}%)</span></span></div>
  <div class="kv"><span class="k">Evidence trace</span><span class="v"><span class="pill {ev_cls}">{ev_p}/{ev_c} ({ev_pct:.0f}%)</span></span></div>
  <div class="kv"><span class="k">Hashtag universe</span><span class="v">{hs}</span></div>
</div>
""")
    return f"""
<div class="section">
  <h2 class="section-title">Per-persona overview <span class="hint">click in for full panels below</span></h2>
  {METHOD_NOTE_OVERVIEW}
  <div class="row">{''.join(tiles)}</div>
</div>
"""


def render_family_chart(stats: dict) -> str:
    fam_totals: dict[str, int] = defaultdict(int)
    for uid, fams in stats["per_user_task_family"].items():
        for f, c in fams.items():
            fam_totals[f] += c
    if not fam_totals:
        return ""
    max_v = max(fam_totals.values()) or 1
    cols = []
    for fam, c in sorted(fam_totals.items(), key=lambda x: -x[1]):
        h = max(4, int(110 * c / max_v))
        color = FAMILY_COLOR.get(fam, "#86868B")
        cols.append(f"""
<div class="bcol" title="{esc(fam)}: {c}">
  <div class="bv">{c}</div>
  <div class="br" style="height:{h}px; background:{color};"></div>
  <div class="bl">{esc(fam)}</div>
</div>
""")
    return f"""
<div class="section">
  <h2 class="section-title">Task-family distribution <span class="hint">aggregated across all {len(stats['user_ids'])} personas</span></h2>
  {METHOD_NOTE_FAMILY}
  <div class="bar-chart">{''.join(cols)}</div>
</div>
"""


def render_task_type_heatmap(stats: dict, audit: dict) -> str:
    user_ids = stats["user_ids"]
    type_cols = [tt for tt, _ in stats["overall_task_type_ranking"]]  # sort by overall popularity

    # Count heatmap
    max_c = max(
        (stats["per_user_task_type"].get(uid, {}).get(tt, 0))
        for uid in user_ids for tt in type_cols
    ) or 1

    # Transposed layout: rows = task_types (horizontal label), cols = user_ids.
    head = (
        '<tr><th class="row-label-head">Task type</th>'
        + "".join(f'<th class="uid-col">{esc(uid)}</th>' for uid in user_ids)
        + '<th class="total-col">Σ</th></tr>'
    )

    # --- Count heatmap (rows = task_types) ---
    rows = []
    for tt in type_cols:
        cells = []
        row_total = 0
        for uid in user_ids:
            c = stats["per_user_task_type"].get(uid, {}).get(tt, 0)
            row_total += c
            color = count_color(c, max_c)
            txt_color = "#1D1D1F" if c < max_c * 0.6 else "#fff"
            cells.append(f'<td style="background:{color}; color:{txt_color};">{c if c else ""}</td>')
        rows.append(
            f'<tr><td class="label">{esc(tt)}</td>'
            f'{"".join(cells)}'
            f'<td class="total-col">{row_total}</td></tr>'
        )

    # --- Axis-4 pass-rate heatmap (rows = task_types) ---
    rows_v = []
    for tt in type_cols:
        cells = []
        agg_pass = 0
        agg_total = 0
        for uid in user_ids:
            v = audit["per_user_task_type_pass_rate"].get(uid, {}).get(tt, {})
            total = v.get("total", 0)
            passed = v.get("pass", 0)
            agg_pass += passed
            agg_total += total
            if total == 0:
                cells.append('<td style="background:#FAFAFA; color:#AEAEB2;">—</td>')
                continue
            rate = passed / total
            color = heat_color(rate)
            txt = f"{int(rate*100)}%"
            cells.append(f'<td style="background:{color};" title="{passed}/{total}">{txt}</td>')
        if agg_total:
            agg_rate = agg_pass / agg_total
            agg_color = heat_color(agg_rate)
            agg_cell = f'<td class="total-col" style="background:{agg_color};" title="{agg_pass}/{agg_total}">{int(agg_rate*100)}%</td>'
        else:
            agg_cell = '<td class="total-col" style="background:#FAFAFA; color:#AEAEB2;">—</td>'
        rows_v.append(
            f'<tr><td class="label">{esc(tt)}</td>'
            f'{"".join(cells)}'
            f'{agg_cell}</tr>'
        )

    return f"""
<div class="section">
  <h2 class="section-title">Task-type distribution <span class="hint">rows = task_types, columns = users; sorted by overall popularity</span></h2>
  {METHOD_NOTE_COUNT_HEATMAP}
  <div class="heatmap heatmap-vert"><table>{head}{''.join(rows)}</table></div>
</div>
<div class="section">
  <h2 class="section-title">Axis-4 schema/format pass rate <span class="hint">rows = task_types, columns = users — green = passes structural rubric, lights up when LLM-postprocessed</span></h2>
  {METHOD_NOTE_AXIS4_HEATMAP}
  <div class="heatmap heatmap-vert"><table>{head}{''.join(rows_v)}</table></div>
</div>
"""


def render_similarity(sim: dict) -> str:
    user_ids = sim["user_ids"]
    def fmt(matrix, title):
        thead = "<tr><td></td>" + "".join(f"<td class='label'>{u}</td>" for u in user_ids) + "</tr>"
        body = ""
        for i, ui in enumerate(user_ids):
            row = f"<tr><td class='label'>{ui}</td>"
            for j in range(len(user_ids)):
                v = matrix[i][j]
                color = heat_color(v)
                txt_color = "#1D1D1F" if v < 0.7 else "#fff"
                row += f"<td style='background:{color}; color:{txt_color};'>{v:.2f}</td>"
            row += "</tr>"
            body += row
        return f"""<div class="sim-matrix"><h4>{title}</h4><table>{thead}{body}</table></div>"""

    parts = [fmt(sim["dimensions"][d], d.replace("_", " ")) for d in sim["dimensions"]]
    parts.append(fmt(sim["combined"], "COMBINED (mean)"))
    return f"""
<div class="section">
  <h2 class="section-title">Persona similarity matrices <span class="hint">6 feature dimensions + equal-weight combined</span></h2>
  {METHOD_NOTE_SIMILARITY}
  <div>{''.join(parts)}</div>
</div>
"""


def render_validity_bar(grades: list[dict]) -> str:
    axes = ["axis_1_setup", "axis_2_example", "axis_3_inferior"]
    labels = ["Axis 1 · setup", "Axis 2 · example", "Axis 3 · inferior"]
    rows = []
    for ax, lbl in zip(axes, labels):
        n_total = 0
        n_pass = 0
        n_pending = 0
        for g in grades:
            v = g.get(ax) or {}
            p = v.get("pass")
            if p is True:
                n_pass += 1; n_total += 1
            elif p is False:
                n_total += 1
            elif (v.get("note") or "") in ("postprocess_pending",):
                n_pending += 1
        pct = (100 * n_pass / n_total) if n_total else 0
        rows.append(f"""
<div class="axis">
  <span class="ax-name">{lbl}</span>
  <span class="ax-bar">
    <span class="bar-outer"><span class="bar-inner" style="width:{pct:.0f}%;"></span></span>
    <span style="font-weight:600;">{n_pass}/{n_total}</span>
    <span style="color:#AEAEB2;font-size:10px;">({n_pending} pending)</span>
  </span>
</div>
""")
    return f'<div class="validity-bar">{"".join(rows)}</div>'


def render_axis_badges(grade: dict) -> str:
    badges = []
    for ax, short in [
        ("axis_1_setup", "Ax1·setup"),
        ("axis_2_example", "Ax2·example"),
        ("axis_3_inferior", "Ax3·inferior"),
    ]:
        v = grade.get(ax) or {}
        p = v.get("pass")
        note = v.get("note") or ""
        if note in ("not_applicable",):
            cls = "na"
            mark = "—"
        elif note in ("postprocess_pending",) or p is None:
            cls = "pending"
            mark = "⋯"
        elif p is True:
            cls = "pass"
            mark = "✓"
        else:
            cls = "fail"
            mark = "✗"
        badges.append(
            f'<span class="axis-badge {cls} tooltip" data-note="{esc(note or short)}">{mark} {short}</span>'
        )
    # Axis-4 from schema_audit gets stamped separately at render time
    return "".join(badges)


def render_sample_card(sample: dict, grade: dict | None, axis4: dict | None, ev: dict | None) -> str:
    inst = sample.get("instance_json") or {}
    qid = sample.get("query_id", "")
    tt = sample.get("task_type", "")
    fam = sample.get("task_family", "")
    app = (sample.get("app_context") or "").lower()
    app_color = APP_COLOR.get(app, "#86868B")

    user_query = (
        sample.get("query_text")
        or inst.get("user_query")
        or inst.get("query_text")
        or inst.get("query")
        or inst.get("topic")
        or "<em>(no surface query — proactive/ranking/agentic-write task)</em>"
    )

    if isinstance(user_query, list):
        user_query = " · ".join(str(x) for x in user_query[:5])

    example_resp = inst.get("example_response", "")
    inferior_resp = inst.get("inferior_response", "")
    gt = inst.get("groundtruth_preference") or inst.get("groundtruth") or ""
    if isinstance(gt, dict):
        gt = json.dumps(gt, ensure_ascii=False)

    # Card border based on overall verdict
    verdict = (grade or {}).get("overall_verdict") if grade else None
    if axis4 and not axis4.get("pass"):
        cls = "fail-1" if len(axis4.get("hard_issues", [])) <= 1 else "fail-multi"
    elif verdict == "valid":
        cls = "valid"
    elif verdict == "weak":
        cls = "weak"
    elif verdict in ("postprocess_pending", None):
        cls = "pending"
    elif verdict == "invalid":
        cls = "fail-multi"
    else:
        cls = ""

    badges_html = render_axis_badges(grade) if grade else (
        '<span class="axis-badge pending">⋯ not graded</span>'
    )

    # Axis-4 badge (deterministic from schema_audit)
    if axis4 is not None:
        if axis4.get("pass"):
            badges_html += '<span class="axis-badge pass tooltip" data-note="schema/format complete">✓ Ax4·schema</span>'
        else:
            issues = "; ".join(axis4.get("hard_issues", []))[:200]
            badges_html += f'<span class="axis-badge fail tooltip" data-note="{esc(issues)}">✗ Ax4·schema</span>'

    # Evidence traceback badge
    if ev:
        if ev.get("trace_pass") is True:
            badges_html += '<span class="axis-badge pass tooltip" data-note="hashtag overlap with user history">⊕ trace</span>'
        elif ev.get("trace_pass") is False:
            badges_html += '<span class="axis-badge fail tooltip" data-note="no hashtag overlap with user history">⊖ trace</span>'

    why = (grade or {}).get("why") or ""
    if not why and grade:
        # If illustrative, optionally pull from illustrative_queries — handled at caller level
        pass

    body_rows = []
    if example_resp:
        body_rows.append(f'<div class="body-row example"><span class="label">Example</span><span class="val">{esc(truncate(example_resp, 600))}</span></div>')
    if inferior_resp:
        body_rows.append(f'<div class="body-row inferior"><span class="label">Inferior</span><span class="val">{esc(truncate(inferior_resp, 600))}</span></div>')
    if gt:
        body_rows.append(f'<div class="body-row gt"><span class="label">Ground-truth preference</span><span class="val">{esc(truncate(gt, 300))}</span></div>')

    notes = (grade or {}).get("notes") or ""
    if notes:
        body_rows.append(f'<div class="body-row"><span class="label">Reviewer note</span><span class="val">{esc(notes)}</span></div>')

    illustrative_marker = ""
    if grade and grade.get("is_illustrative"):
        illustrative_marker = '<span class="pill amber">★ illustrative</span>'

    return f"""
<div class="sample-card {cls}">
  <div class="sample-head">
    <span class="qid">{esc(qid)}</span>
    <span class="tt">{esc(tt)}</span>
    <span class="fam">{esc(fam)}</span>
    {'<span class="app" style="background:'+app_color+';">'+esc(app)+'</span>' if app else ''}
    {illustrative_marker}
  </div>
  <div>{badges_html}</div>
  <div class="qtext">{esc(truncate(user_query, 400))}</div>
  <div class="body">{''.join(body_rows)}</div>
</div>
"""


def render_persona_panel(uid: str, *, audit: dict, samples: list[dict], qualitative: dict | None, sim_snapshot: dict) -> str:
    if qualitative is None:
        headline = sim_snapshot.get("name") or f"User {uid}"
        distinctive = "(per-user subagent did not produce a qualitative report)"
        hp_chips = ""
        grades_by_qid = {}
        illustrative_marks = {}
        notes = ""
    else:
        headline = qualitative.get("headline") or sim_snapshot.get("name") or f"User {uid}"
        distinctive = qualitative.get("distinctive_features") or ""
        hps = qualitative.get("top_hidden_personas") or []
        chips = []
        for hp in hps:
            typ = hp.get("type") or ""
            label = hp.get("label") or ""
            why = hp.get("why_it_matters") or ""
            is_privacy = typ in {"intimate_interest", "covert_concern", "compensatory_need", "medical_aesthetic_concern", "sensitive_life_event"}
            cls = "hp-chip privacy" if is_privacy else "hp-chip"
            chips.append(f'<span class="{cls}" title="{esc(why)}">{esc(typ)} · {esc(label)}</span>')
        hp_chips = '<div class="hp-list">' + "".join(chips) + "</div>" if chips else ""
        grades_by_qid = {g.get("query_id"): g for g in (qualitative.get("validity_grades") or [])}
        illustrative_marks = {iq.get("query_id"): iq for iq in (qualitative.get("illustrative_queries") or [])}
        # Inject is_illustrative flag from the illustrative list
        for qid_i, _ in illustrative_marks.items():
            if qid_i in grades_by_qid:
                grades_by_qid[qid_i]["is_illustrative"] = True
        notes = qualitative.get("notes") or ""

    # Demographics line
    demo_parts = []
    for k in ("mbti", "gender", "race_ethnicity", "career", "education"):
        v = sim_snapshot.get(k)
        if v:
            demo_parts.append(esc(v))
    demo_line = " · ".join(demo_parts)
    bf = sim_snapshot.get("big_five") or {}
    bf_chips = " ".join(
        f'<span class="pill">{esc(k)}: {esc(v)}</span>' for k, v in bf.items()
    )

    # Validity bar
    grades = list(grades_by_qid.values())
    validity_bar = render_validity_bar(grades) if grades else ""

    # Build axis_4 lookup from schema_audit
    axis4_lookup: dict[str, dict] = {}
    ev_lookup: dict[str, dict] = {}
    for v in audit["per_user"][uid]["row_verdicts"]:
        axis4_lookup[v["query_id"]] = v["axis_4"]
        ev_lookup[v["query_id"]] = v["evidence_traceback"]

    # Sort: failing cards first, then by task_family
    def sort_key(s):
        qid = s.get("query_id", "")
        ax4 = axis4_lookup.get(qid, {})
        g = grades_by_qid.get(qid, {})
        verdict = g.get("overall_verdict") or "ungraded"
        # rank: fails > weak/pending > valid
        order_v = {"invalid": 0, "weak": 1, "postprocess_pending": 2, "ungraded": 2, None: 2, "valid": 3}.get(verdict, 2)
        ax4_fail = 0 if not ax4.get("pass") else 1
        return (ax4_fail, order_v, s.get("task_family", ""), s.get("task_type", ""))

    sorted_samples = sorted(samples, key=sort_key)
    cards = []
    for s in sorted_samples:
        qid = s.get("query_id")
        grade = grades_by_qid.get(qid)
        ax4 = axis4_lookup.get(qid)
        ev = ev_lookup.get(qid)
        cards.append(render_sample_card(s, grade, ax4, ev))

    notes_html = f'<div class="notes-block"><strong>Reviewer notes:</strong> {esc(notes)}</div>' if notes else ""

    return f"""
<div class="persona-panel" id="user-{uid}">
  <div class="uid-line">USER {uid} · {esc(sim_snapshot.get("name") or "")}</div>
  <div class="headline">{esc(headline)}</div>
  <div class="demo-line">{demo_line}</div>
  <div style="margin: 6px 0 12px;">{bf_chips}</div>
  <div class="distinctive">{esc(distinctive)}</div>
  {hp_chips}
  {validity_bar}
  {notes_html}
  <details>
    <summary>Show {len(cards)} sampled queries</summary>
    <div class="samples" style="margin-top: 12px;">{''.join(cards)}</div>
  </details>
</div>
"""


def render_synthesis(synthesis_md: str) -> str:
    return f"""
<div class="section">
  <h2 class="section-title">Cross-persona synthesis <span class="hint">from the synthesis subagent</span></h2>
  {METHOD_NOTE_SYNTHESIS}
  <div class="synthesis-block">{short_md_to_html(synthesis_md)}</div>
</div>
"""


def main() -> None:
    d = load_inputs()
    stats, audit, sim, samples, qual, syn = (
        d["stats"], d["audit"], d["sim"], d["samples"], d["qualitative"], d["synthesis"]
    )

    persona_panels = []
    for uid in USER_IDS:
        persona_panels.append(
            render_persona_panel(
                uid,
                audit=audit,
                samples=samples.get(uid, []),
                qualitative=qual.get(uid),
                sim_snapshot=sim["snapshots"].get(uid, {}),
            )
        )

    body = "\n".join([
        render_header(stats, audit),
        render_overview(stats, audit),
        render_family_chart(stats),
        render_task_type_heatmap(stats, audit),
        render_similarity(sim),
        '<div class="section"><h2 class="section-title">Per-persona panels <span class="hint">click each to expand sampled queries</span></h2>' + METHOD_NOTE_PANELS,
        "\n".join(persona_panels),
        '</div>',
        render_synthesis(syn),
    ])

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PersonaMem-v3 — Benchmark Analysis</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"[Stage D] Wrote {OUT_HTML}")
    print(f"  size: {OUT_HTML.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
