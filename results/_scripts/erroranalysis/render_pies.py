#!/usr/bin/env python3
"""Failure-only view: one donut per (model, mode) showing the first-principles
cognitive CAUSE of its errors (why the model got it wrong), in plain language.
Optimistic font, model-name-first headers, per-row legend, tight large layout."""
import json, os, math
from collections import Counter
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "aggregate", "figures")
FONT = os.path.join(HERE, "..", "..", "aggregate", "html", "Optimistic.ttf")
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
os.makedirs(OUT, exist_ok=True)

CAUSES = ["recall_miss", "salience_error", "grounding_lapse", "confabulation",
          "over_apply", "pragmatic_misjudge", "temporal", "instruction_miss", "degenerate"]
CAUSE_LABEL = {
    "recall_miss":        "Couldn't find the relevant info about the user",
    "salience_error":     "Found the info but ranked the wrong things on top",
    "grounding_lapse":    "Gave a generic answer, ignoring what it knew",
    "confabulation":      "Made up details that weren't there",
    "over_apply":         "Overdid it — shared private info it shouldn't",
    "pragmatic_misjudge": "Spoke up when it should've held back (or vice-versa)",
    "temporal":           "Used old info the user had moved on from",
    "instruction_miss":   "Didn't do what was actually asked",
    "degenerate":         "Gave an empty answer, refused, or errored",
}
CAUSE_COLOR = {
    "recall_miss": "#2F5C9E", "salience_error": "#86B0DC", "grounding_lapse": "#9AA0A6",
    "confabulation": "#9C7A54", "over_apply": "#C8484C", "pragmatic_misjudge": "#5AA469",
    "temporal": "#E0894A", "instruction_miss": "#B57BB0", "degenerate": "#4D4D4D",
}

GRID = [
    ["longctx_gpt55", "textmem_gpt55", "mem0_gpt55", "codex_gpt55"],
    ["longctx_gemini", "textmem_gemini", "claudecode_opus", "claudecode_sonnet"],
]
LABELS = {
    "longctx_gpt55":   ("GPT-5.5", "Long Context"),
    "textmem_gpt55":   ("GPT-5.5", "Textual Memory"),
    "mem0_gpt55":      ("GPT-5.5", "Mem0 w/ RAG"),
    "codex_gpt55":     ("GPT-5.5", "Codex High"),
    "longctx_gemini":  ("Gemini-3.5-Flash", "Long Context"),
    "textmem_gemini":  ("Gemini-3.5-Flash", "Textual Memory"),
    "claudecode_opus": ("Opus-4.8", "Claude Code High"),
    "claudecode_sonnet":("Sonnet-4.6", "Claude Code High"),
}


def donut(ax, counts, acc):
    total = sum(counts.get(k, 0) for k in CAUSES)
    vals = [counts.get(k, 0) for k in CAUSES]
    cols = [CAUSE_COLOR[k] for k in CAUSES]
    wedges, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False, radius=1.30,
                       wedgeprops=dict(width=0.50, edgecolor="white", linewidth=1.7))
    for w, v in zip(wedges, vals):
        pct = 100.0 * v / total if total else 0
        if pct >= 6.0:
            ang = (w.theta2 + w.theta1) / 2.0
            ax.text(1.05 * math.cos(math.radians(ang)), 1.05 * math.sin(math.radians(ang)),
                    f"{pct:.0f}%", ha="center", va="center",
                    fontsize=12.5, fontweight="bold", color="white")
    ax.text(0, 0.10, f"{acc:.0f}%", ha="center", va="center", fontsize=21, color="#222")
    ax.text(0, -0.20, "overall accuracy", ha="center", va="center", fontsize=9.5, color="#888")
    ax.set_xlim(-1.34, 1.34); ax.set_ylim(-1.34, 1.34); ax.set_aspect("equal")


def row_legend(ax):
    ax.axis("off")
    handles = [Patch(facecolor=CAUSE_COLOR[k], edgecolor="white") for k in CAUSES]
    leg = ax.legend(handles, [CAUSE_LABEL[k] for k in CAUSES], loc="center left",
                    frameon=False, fontsize=13, handlelength=1.15, handleheight=1.15,
                    labelspacing=0.55, borderaxespad=0, bbox_to_anchor=(-0.10, 0.5))
    leg.set_title("Why it got answers wrong", prop={"size": 14, "weight": "bold"})
    leg.get_title().set_color("#9e2b2b")


def main():
    fail = [json.loads(l) for l in open(os.path.join(HERE, "perrow_failures.jsonl"))]
    stats = json.load(open(os.path.join(HERE, "model_stats.json")))
    F = {k: Counter(r["cause"] for r in fail if r["key"] == k) for k in LABELS}

    fig, axes = plt.subplots(2, 5, figsize=(22, 9.0),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
    fig.patch.set_facecolor("white")
    for r in range(2):
        for c in range(4):
            key = GRID[r][c]
            model, detail = LABELS[key]
            ax = axes[r][c]
            ax.text(0, 1.66, model, ha="center", va="center", fontsize=16.5,
                    fontweight="bold", color="#16242c", clip_on=False)
            ax.text(0, 1.45, detail, ha="center", va="center", fontsize=12.5,
                    color="#55636b", clip_on=False)
            donut(ax, F[key], stats[key]["overall_acc"])
        row_legend(axes[r][4])

    fig.suptitle("PersonaMem — why each system gets answers wrong",
                 fontsize=23, fontweight="bold", y=0.975)
    fig.text(0.5, 0.918,
             "same simulated users · each ring shows WHY the model's wrong answers were wrong — the underlying reason, not the task",
             ha="center", fontsize=13, color="#666")
    fig.subplots_adjust(left=0.006, right=0.995, top=0.84, bottom=0.02,
                        wspace=0.03, hspace=0.40)
    png = os.path.join(OUT, "error_analysis_by_model.png")
    fig.savefig(png, dpi=150, facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), facecolor="white")
    print("saved:", os.path.abspath(png))


if __name__ == "__main__":
    main()
