#!/usr/bin/env python3
"""Paired success/failure donuts per (model, mode).
FAILURE donut = first-principles cognitive CAUSE of the error (why the model
erred), not the task type. SUCCESS donut = capability the passing rows
demonstrate. Per-row legends, Optimistic font, model-name-first headers, tight
spacing, no count annotations."""
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
FAM = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["font.family"] = FAM
os.makedirs(OUT, exist_ok=True)

# ---------------- FAILURE: first-principles causes ----------------
CAUSES = ["recall_miss", "salience_error", "grounding_lapse", "confabulation",
          "over_apply", "pragmatic_misjudge", "temporal", "instruction_miss", "degenerate"]
CAUSE_LABEL = {  # plain language — a general reader should get it at a glance
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
    "recall_miss":        "#2F5C9E",
    "salience_error":     "#86B0DC",
    "grounding_lapse":    "#9AA0A6",
    "confabulation":      "#9C7A54",
    "over_apply":         "#C8484C",
    "pragmatic_misjudge": "#5AA469",
    "temporal":           "#E0894A",
    "instruction_miss":   "#B57BB0",
    "degenerate":         "#4D4D4D",
}

# ---------------- SUCCESS: capability demonstrated ----------------
SUCC = ["Personalized ranking/retrieval", "Personalization depth", "Voice & style",
        "Privacy & over-personalization", "Proactive judgment",
        "Grounding & abstention", "Temporal currency"]
SUCC_LABEL = {  # plain language
    "Personalized ranking/retrieval": "Picked the right things for the user",
    "Personalization depth":          "Used the user's specific tastes",
    "Voice & style":                  "Wrote in the user's own style",
    "Privacy & over-personalization": "Personalized without oversharing",
    "Proactive judgment":             "Knew when to speak up or stay quiet",
    "Grounding & abstention":         "Stuck to the facts (or admitted it wasn't sure)",
    "Temporal currency":              "Used the user's up-to-date preferences",
}
SUCC_COLOR = {
    "Personalized ranking/retrieval": "#3B6FB0",
    "Personalization depth":          "#3FA08D",
    "Voice & style":                  "#C9A94E",
    "Privacy & over-personalization": "#7E6BB3",
    "Proactive judgment":             "#5AA469",
    "Grounding & abstention":         "#59A6C9",
    "Temporal currency":              "#E0894A",
}

GROUPS = [
    ["longctx_gpt55", "textmem_gpt55", "mem0_gpt55", "codex_gpt55"],
    ["longctx_gemini", "textmem_gemini", "claudecode_opus", "claudecode_sonnet"],
]
LABELS = {  # model name FIRST, then mode (mirrors the Accuracy table headers)
    "longctx_gpt55":   ("GPT-5.5", "Long Context"),
    "textmem_gpt55":   ("GPT-5.5", "Textual Memory"),
    "mem0_gpt55":      ("GPT-5.5", "Mem0 w/ RAG"),
    "codex_gpt55":     ("GPT-5.5", "Codex High"),
    "longctx_gemini":  ("Gemini-3.5-Flash", "Long Context"),
    "textmem_gemini":  ("Gemini-3.5-Flash", "Textual Memory"),
    "claudecode_opus": ("Opus-4.8", "Claude Code High"),
    "claudecode_sonnet":("Sonnet-4.6", "Claude Code High"),
}


def donut(ax, counts, order, colmap):
    total = sum(counts.get(k, 0) for k in order)
    vals = [counts.get(k, 0) for k in order]
    cols = [colmap[k] for k in order]
    wedges, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False,
                       radius=1.30,
                       wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.6))
    for w, v in zip(wedges, vals):
        pct = 100.0 * v / total if total else 0
        if pct >= 7.0:
            ang = (w.theta2 + w.theta1) / 2.0
            ax.text(1.04 * math.cos(math.radians(ang)), 1.04 * math.sin(math.radians(ang)),
                    f"{pct:.0f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")
    ax.set_xlim(-1.34, 1.34); ax.set_ylim(-1.34, 1.34); ax.set_aspect("equal")


def row_legend(ax, kind):
    ax.axis("off")
    if kind == "succ":
        order, lab, col = SUCC, SUCC_LABEL, SUCC_COLOR
        title, tcol = "SUCCESS  —  what it did well", "#1f6f4a"
    else:
        order, lab, col = CAUSES, CAUSE_LABEL, CAUSE_COLOR
        title, tcol = "FAILURE  —  the underlying reason it got it wrong", "#9e2b2b"
    handles = [Patch(facecolor=col[k], edgecolor="white") for k in order]
    labels = [lab[k] for k in order]
    leg = ax.legend(handles, labels, loc="center left", frameon=False,
                    fontsize=12.5, handlelength=1.15, handleheight=1.15,
                    labelspacing=0.5, borderaxespad=0, bbox_to_anchor=(-0.10, 0.5))
    leg.set_title(title, prop={"size": 13.5, "weight": "bold"})
    leg.get_title().set_color(tcol)


def main():
    succ = [json.loads(l) for l in open(os.path.join(HERE, "perrow_success.jsonl"))]
    fail = [json.loads(l) for l in open(os.path.join(HERE, "perrow_failures.jsonl"))]
    stats = json.load(open(os.path.join(HERE, "model_stats.json")))
    S = {k: Counter(r["success_dim"] for r in succ if r["key"] == k) for k in LABELS}
    F = {k: Counter(r["cause"] for r in fail if r["key"] == k) for k in LABELS}

    fig, axes = plt.subplots(4, 5, figsize=(22, 16),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
    fig.patch.set_facecolor("white")

    for g, models in enumerate(GROUPS):
        rs, rf = g * 2, g * 2 + 1
        for c, key in enumerate(models):
            st = stats[key]
            model, detail = LABELS[key]
            ax_s = axes[rs][c]
            ax_s.text(0, 1.66, model, ha="center", va="center", fontsize=16.5,
                      fontweight="bold", color="#16242c", clip_on=False)
            ax_s.text(0, 1.45, f"{detail}   ·   {st['overall_acc']:.0f}%",
                      ha="center", va="center", fontsize=12.5, color="#55636b", clip_on=False)
            donut(ax_s, S[key], SUCC, SUCC_COLOR)
            donut(axes[rf][c], F[key], CAUSES, CAUSE_COLOR)
        row_legend(axes[rs][4], "succ")
        row_legend(axes[rf][4], "fail")

    fig.suptitle("PersonaMem — why each system fails (and what it gets right)",
                 fontsize=23, fontweight="bold", y=0.987)
    fig.text(0.5, 0.953,
             "same 10 simulated users · each FAILURE ring shows WHY the model got it wrong (the underlying reason, "
             "not the task) · each SUCCESS ring shows what it did well",
             ha="center", fontsize=13, color="#666")
    fig.add_artist(plt.Line2D([0.02, 0.995], [0.498, 0.498], color="#dddddd", lw=1.3))

    fig.subplots_adjust(left=0.006, right=0.995, top=0.915, bottom=0.02,
                        wspace=0.03, hspace=0.42)
    png = os.path.join(OUT, "success_failure_pairs_by_model.png")
    fig.savefig(png, dpi=150, facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), facecolor="white")
    print("saved:", os.path.abspath(png), "| font:", FAM)


if __name__ == "__main__":
    main()
