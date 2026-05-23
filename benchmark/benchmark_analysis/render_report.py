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
.method-note { background: #EFF6FF; border-left: 3px solid #3B82F6; border-radius: 0 6px 6px 0; padding: 10px 16px; margin: 0 0 18px 0; font-size: 12px; color: #1E40AF; line-height: 1.6; }
.method-note strong { color: #1E3A8A; font-weight: 600; }
.method-note code { background: rgba(59,130,246,0.10); padding: 1px 5px; border-radius: 3px; font-family: ui-monospace,SFMono-Regular,monospace; font-size: 0.9em; color: #1E3A8A; }
.method-note em { color: #1E40AF; font-style: italic; }
.method-note ul { margin: 6px 0 0 18px; }
.method-note li { margin: 2px 0; }
"""


METHOD_NOTE_OVERVIEW = """
<div class="method-note">
<strong>How this was computed.</strong> Each tile aggregates one user's queries from <code>benchmark/{uid}/queries.csv</code>.
<strong>Total queries</strong> = CSV row count.
<strong>Postprocessed</strong> = pair-scored rows where <code>instance_json</code> carries both <code>example_response</code> and <code>inferior_response</code> (proactive + repetition schemas count as postprocessed by default — they don't use that pair).
<strong>Axis-4 pass</strong> = the row passes the full structural check for its task category (see the Axis-4 section below for the exact rules).
<strong>Evidence trace</strong> = fraction of rows whose <code>instance_json</code> hashtags overlap the user's app-history hashtag universe (collected from all five <code>backend/{uid}/{instagram,facebook,threads,chatbot,ai_studio}.json</code> files).
<strong>Hashtag universe</strong> = size of that user's deduped lifetime hashtag set.
</div>
"""


METHOD_NOTE_FAMILY = """
<div class="method-note">
<strong>How this was computed.</strong> Each bar sums one <code>task_family</code> across all 5 users, taken directly from the CSV <code>task_family</code> column. Heights scale to the largest family. Colors match the family palette used in the per-persona sample cards below.
</div>
"""


METHOD_NOTE_COUNT_HEATMAP = """
<div class="method-note">
<strong>How this was computed.</strong> Each cell is the raw count of one user's rows of one <code>task_type</code>. Columns are sorted by overall popularity (most-emitted task_types first). Cell color = count / max_count, fading from white (0) to saturated blue (max). Empty cells mean the user has zero rows of that task_type — common for data-dependent tasks like <code>local_recommendation_geo_shift</code> on home-only users.
</div>
"""


METHOD_NOTE_AXIS4_HEATMAP = """
<div class="method-note">
<strong>How this was computed.</strong> Each cell is the <em>pass rate</em> (<code>pass / total</code>) of that user's rows of that <code>task_type</code> under the deterministic Axis-4 schema check. A row passes iff it carries <strong>all</strong> required fields for its category:
<ul>
  <li><strong>All rows:</strong> CSV <code>rubric_tags</code> column non-empty.</li>
  <li><strong>Pair-scored tasks</strong> (chatbot_response, over_personalization except repetition, e_followup, agentic, personalization, new_suggestions): <code>example_response</code> + <code>inferior_response</code> + one of <code>groundtruth_preference</code> / <code>groundtruth</code> / <code>held_out_preference</code> / <code>target_pref</code> / <code>gt_slice</code>.</li>
  <li><strong>Repetition tasks</strong> (<code>over_personalization_repetition_recsys</code> / <code>_chatbot</code>): <code>queries[]</code> + <code>target_pref</code> (no example/inferior pair).</li>
  <li><strong>Proactive family:</strong> <code>expected_behavior</code> + <code>trigger_evidence</code> + <code>jitai_card</code> + <code>tool_call_rules</code>.</li>
  <li><strong>Ranking tasks</strong> (<code>personalized_recommendation</code>, <code>at_ai_directive_followup</code>, <code>new_suggestions_recsys</code>): <code>candidates[]</code> (≥5) + <code>held_out_idx</code> (or <code>positive_indices</code>).</li>
  <li><strong>Agentic family:</strong> <code>tool_call</code> or <code>tool_call_rules</code>.</li>
</ul>
A surface user query is required only for task types whose surface naturally has one — ranking/agentic-write/proactive task types are exempt (see <code>NO_EXPLICIT_QUERY_TASKS</code> in <code>build_stats.py</code>).
</div>
"""


METHOD_NOTE_SIMILARITY = """
<div class="method-note">
<strong>How this was computed.</strong> Each 5×5 matrix scores one feature dimension; entry (i, j) is the pairwise similarity of personas i and j (diagonal = 1.0).
<ul>
  <li><strong>Demographic</strong>: Jaccard over {<code>gender</code>, <code>race_ethnicity</code>, <code>career</code>, <code>education</code>}.</li>
  <li><strong>Personality</strong>: cosine on Big-Five (high/medium/low → 1.0 / 0.5 / 0.0) averaged with MBTI 4-letter Hamming.</li>
  <li><strong>Hidden personas</strong>: Jaccard over <code>hidden_personas[].type</code> set (12 possible types).</li>
  <li><strong>Hashtag interests</strong>: Jaccard over the top 20 <code>exploration_exploitation.top_repeated_hashtags</code>.</li>
  <li><strong>Voice</strong>: mean of <code>emoji_palette</code> Jaccard, <code>formality_baseline</code> distance-to-similarity, <code>default_capitalization</code> equality, <code>emoji_intensity_default</code> equality.</li>
  <li><strong>Semantic</strong>: sentence-transformer <code>all-MiniLM-L6-v2</code> cosine on the <code>hidden_persona_summary</code> text.</li>
</ul>
<strong>Combined</strong> = equal-weight mean across the six dimensions.
</div>
"""


METHOD_NOTE_PANELS = """
<div class="method-note">
<strong>How this was computed.</strong> Each panel is the output of one <em>general-purpose subagent</em> launched in parallel for that user_id. The subagent received the user's full <code>backend/{uid}/profile.json</code> plus ~30 stratified-random sampled queries from <code>benchmark/{uid}/queries.csv</code> (≥2 per task_type to guarantee coverage, then uniform top-up to 30; seed = 42). It graded each sampled query on the first 3 rubric axes — <em>task setup integrity</em>, <em>example response quality</em>, <em>inferior response quality</em> — with a short note per axis, picked 3-6 illustrative queries, and wrote the result to <code>backend/persona_analysis/per_user/{uid}_qualitative.json</code>. The Axis-4 schema badge and ⊕/⊖ evidence-trace badge on each card come from the deterministic Stage A check, not the subagent. The validity bar at the top of each panel summarizes the subagent's axes 1-3 across the sampled subset. Sample cards sort failing-first.
</div>
"""


METHOD_NOTE_SYNTHESIS = """
<div class="method-note">
<strong>How this was computed.</strong> A 6th general-purpose subagent read all five per-user qualitative JSONs plus <code>query_stats.json</code>, <code>schema_audit.json</code>, and <code>similarity_matrix.json</code>, and wrote <code>backend/persona_analysis/synthesis.md</code> (~750 words). It was asked to surface the strongest discriminator dimension across personas, the weakest task types by validity, standout queries worth manual review, and concrete next-step recommendations — leading with the postprocess gap if present.
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

    head = "<tr><th>User</th>" + "".join(
        f'<th class="rotated">{esc(tt)}</th>' for tt in type_cols
    ) + "</tr>"

    rows = []
    for uid in user_ids:
        cells = []
        for tt in type_cols:
            c = stats["per_user_task_type"].get(uid, {}).get(tt, 0)
            color = count_color(c, max_c)
            txt_color = "#1D1D1F" if c < max_c * 0.6 else "#fff"
            cells.append(f'<td style="background:{color}; color:{txt_color};">{c if c else ""}</td>')
        rows.append(f'<tr><td class="label">{uid}</td>{"".join(cells)}</tr>')

    # Axis-4 pass-rate heatmap
    rows_v = []
    for uid in user_ids:
        cells = []
        pt = audit["per_user_task_type_pass_rate"].get(uid, {})
        for tt in type_cols:
            v = pt.get(tt, {})
            total = v.get("total", 0)
            passed = v.get("pass", 0)
            if total == 0:
                cells.append('<td style="background:#FAFAFA; color:#AEAEB2;">—</td>')
                continue
            rate = passed / total
            color = heat_color(rate)
            txt = f"{int(rate*100)}%"
            cells.append(f'<td style="background:{color};" title="{passed}/{total}">{txt}</td>')
        rows_v.append(f'<tr><td class="label">{uid}</td>{"".join(cells)}</tr>')

    return f"""
<div class="section">
  <h2 class="section-title">Task-type distribution <span class="hint">cells colored by count; sorted by overall popularity</span></h2>
  {METHOD_NOTE_COUNT_HEATMAP}
  <div class="heatmap"><table>{head}{''.join(rows)}</table></div>
</div>
<div class="section">
  <h2 class="section-title">Axis-4 schema/format pass rate <span class="hint">per (user × task_type) — green = passes structural rubric, lights up when LLM-postprocessed</span></h2>
  {METHOD_NOTE_AXIS4_HEATMAP}
  <div class="heatmap"><table>{head}{''.join(rows_v)}</table></div>
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
