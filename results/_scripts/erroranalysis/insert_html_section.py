#!/usr/bin/env python3
"""Insert (or refresh) the Error-analysis section into results_tables.html.

Holds the single-writer lock (results/_scripts/_htmllock.py) across the whole
read-modify-write. Idempotent: the section is wrapped in START/END markers, so
re-running replaces it rather than duplicating. Figures are referenced by
relative path (../figures/...) to keep this shared file lean.
"""
import os, re, sys, base64

ROOT = "/vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3"
sys.path.insert(0, os.path.join(ROOT, "results/_scripts"))
from _htmllock import html_lock

HTML = os.path.join(ROOT, "results/aggregate/html/results_tables.html")
FIG = os.path.join(ROOT, "results/aggregate/figures")
START = "<!-- ERRORANALYSIS_START -->"
END = "<!-- ERRORANALYSIS_END -->"


def data_uri(name):
    """Inline a PNG as a base64 data: URI so the HTML is fully standalone."""
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


PAIRS_URI = data_uri("success_failure_pairs_by_model.png")
PIES_URI = data_uri("error_analysis_by_model.png")

CAP = (
    'font-size:9.5px;color:var(--muted);margin:6px 2px 0;line-height:1.5;'
    'max-width:1180px'
)
IMG = ('display:block;width:100%;max-width:1180px;margin:0;'
       'border:1px solid var(--hair);border-radius:6px')

SECTION = f"""{START}
<section>
<div class="cap"><h2>Error analysis &mdash; why each system gets answers wrong</h2><span class="unit">same 10 users</span><span class="note">every wrong answer labelled by the underlying reason it was wrong (read from the saved judge notes + model responses), not by the task</span></div>
<p class="lead" style="margin:0 0 12px">A row counts as <b>wrong</b> when its answer scored below 60/100. We sort each wrong answer by <b>why</b> the model got it wrong, in plain terms. The biggest reason everywhere is picking the wrong things for the user &mdash; either it <b>couldn&rsquo;t find</b> the relevant info, or it <b>found it but ranked the wrong things on top</b>. The split is revealing: memory- and agent-based systems fail more by <i>not finding</i> the info (Claude Code 28&ndash;29%, Mem0 23%) while Long Context, which has everything in front of it, fails more by <i>finding it but mis-ranking</i> (29%). <b>Codex</b> never ignores the request (0%), but instead <b>over-shares private info</b> and <b>makes details up</b>. <b>Gemini</b> systems make things up the most. The success rings (left/top of each pair) look alike &mdash; every system does well at the same things; they differ in <i>how</i> they fail.</p>
<figure style="margin:0 0 18px">
<img src="{PAIRS_URI}" alt="Per-system success and failure donut pairs" style="{IMG}">
<figcaption style="{CAP}">Each system&rsquo;s <b>SUCCESS</b> ring (what it did well) sits directly above its <b>FAILURE</b> ring (the underlying reason its wrong answers were wrong). Labels are plain-language; read the legend on each row.</figcaption>
</figure>
<figure style="margin:0">
<img src="{PIES_URI}" alt="Per-system failure-cause donuts" style="{IMG}">
<figcaption style="{CAP}">Failure-only view: why each model&rsquo;s wrong answers were wrong. Centre = the model&rsquo;s overall accuracy.</figcaption>
</figure>
</section>
{END}
"""

ANCHOR = "</section>\n<footer>"

with html_lock():
    html = open(HTML).read()
    # drop any prior copy of our section (idempotent)
    html = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", "",
                  html, flags=re.S)
    if ANCHOR not in html:
        raise SystemExit("anchor '</section>\\n<footer>' not found — aborting (file changed?)")
    # preserve the ablation section's </section>; insert our block BETWEEN it and <footer>
    html = html.replace(ANCHOR, "</section>\n" + SECTION + "<footer>", 1)
    open(HTML, "w").write(html)

print("inserted error-analysis section ->", HTML)
