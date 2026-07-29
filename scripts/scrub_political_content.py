#!/usr/bin/env python3
"""Scrub direct political-figure/movement references from generated persona data.

Content policy (2026-07-23): the benchmark ships no direct support for any
political party, movement, or partisan media figure. Rural / faith /
conservative LIFESTYLES are fine; partisan anchors are not. The real-world
engagement seeds carried a viral wave of Charlie Kirk memorial content
(hashtags on ~74 personas' events, one persona's canonical preference and
chatbot session) plus a few Gutfeld/Fox late-night references. This scrubber
swaps those anchors for fictional, apolitical equivalents while preserving the
surrounding narrative (memorial/faith framing, late-night comedy framing).

Replacements are raw-text on each JSON file (replacement strings contain no
JSON metacharacters), validated by re-parsing every touched file.

Usage:
    python scripts/scrub_political_content.py            # dry-run report
    python scripts/scrub_political_content.py --apply    # rewrite files
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ordered longest-first so specific forms win before generic ones.
REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"#TheCharlieKirkShow", re.I), "#TheMarlonReevesShow"),
    (re.compile(r"#CharlieKirk\b", re.I), "#MarlonReeves"),
    (re.compile(r"Charlie[\s_-]?Kirk", re.I), "Marlon Reeves"),
    (re.compile(r"CharlieKirk", re.I), "MarlonReeves"),
    (re.compile(r"#TurningPointUSA\b", re.I), "#FaithForward"),
    (re.compile(r"Turning[\s_-]?Point[\s_-]?USA", re.I), "Faith Forward"),
    (re.compile(r"TurningPointUSA", re.I), "FaithForward"),
    (re.compile(r"#TPUSA\b", re.I), "#FaithForward"),
    (re.compile(r"\bTPUSA\b", re.I), "FaithForward"),
    (re.compile(r"Gutfeld/Fox News late-night talk show content", re.I),
     "late-night comedy talk show content"),
    (re.compile(r"#GregGutfeld\b", re.I), "#LateNightLaughs"),
    (re.compile(r"#Gutfeld\b", re.I), "#LateNightLaughs"),
    (re.compile(r"\bGutfeld\b", re.I), "late-night comedy"),
]
# Post-scrub leftovers that demand a human look if they ever match.
LEFTOVER = re.compile(r"\bkirk\b|\btpusa\b|turning ?point ?usa|gutfeld|fox news", re.I)

FILES = ["instagram", "facebook", "threads", "chatbot", "ai_studio",
         "calendar", "profile", "test"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backend_dir", default="backend")
    args = ap.parse_args()

    base = REPO_ROOT / args.backend_dir
    uids = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.isdigit()], key=int)
    touched: dict[str, int] = {}
    leftovers: list[str] = []
    for u in uids:
        for f in FILES:
            fp = base / u / f"{f}.json"
            if not fp.exists():
                continue
            text = fp.read_text()
            total = 0
            for pat, rep in REPLACEMENTS:
                text, n = pat.subn(rep, text)
                total += n
            if total:
                json.loads(text)  # must still parse before we write
                touched[f"{u}/{f}"] = total
                if args.apply:
                    fp.write_text(text)
            for m in LEFTOVER.finditer(text):
                i = max(0, m.start() - 60)
                leftovers.append(f"{u}/{f}: ...{text[i:m.end()+80]}...")

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] {sum(touched.values())} replacements across "
          f"{len({k.split('/')[0] for k in touched})} personas / {len(touched)} files")
    for k, n in sorted(touched.items(), key=lambda x: -x[1])[:12]:
        print(f"   {n:>4}  {k}")
    if leftovers:
        print(f"\nLEFTOVERS NEEDING REVIEW ({len(leftovers)}):")
        for s in leftovers[:20]:
            print("  ", s[:200])
    else:
        print("no leftovers — post-scrub sweep clean")
    return 1 if leftovers else 0


if __name__ == "__main__":
    sys.exit(main())
