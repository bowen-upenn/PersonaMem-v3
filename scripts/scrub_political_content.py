#!/usr/bin/env python3
"""Apply maintainer-defined content normalization rules to generated persona data.

The benchmark follows a content policy for its published personas. The concrete
normalization rules (regex -> replacement pairs plus a post-run watch pattern)
are maintained locally in ``data/content_policy_rules.json`` and are not part
of the repository. When the rules file is absent this tool is a no-op.

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
RULES_PATH = REPO_ROOT / "data" / "content_policy_rules.json"

FILES = ["instagram", "facebook", "threads", "chatbot", "ai_studio",
         "calendar", "profile", "test"]


def _load_rules():
    if not RULES_PATH.exists():
        return [], None
    doc = json.loads(RULES_PATH.read_text())
    reps = [(re.compile(pat, re.I), rep) for pat, rep in doc.get("replacements", [])]
    watch = doc.get("leftover_watch")
    return reps, (re.compile(watch, re.I) if watch else None)


REPLACEMENTS, LEFTOVER = _load_rules()


def scrub_user(uid: str, backend_dir: str = "backend") -> int:
    """Scrub one persona's files in place. Returns replacement count.
    Called by scripts/run_persona_pipeline.py after each persona is built
    (BEFORE persona.html renders). No-op when no local rules file exists."""
    if not REPLACEMENTS:
        return 0
    base = REPO_ROOT / backend_dir
    total = 0
    for f in FILES:
        fp = base / str(uid) / f"{f}.json"
        if not fp.exists():
            continue
        text = fp.read_text()
        n_f = 0
        for pat, rep in REPLACEMENTS:
            text, n = pat.subn(rep, text)
            n_f += n
        if n_f:
            json.loads(text)
            fp.write_text(text)
            total += n_f
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backend_dir", default="backend")
    args = ap.parse_args()

    if not REPLACEMENTS:
        print(f"no rules file at {RULES_PATH} — nothing to do")
        return 0

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
                json.loads(text)
                touched[f"{u}/{f}"] = total
                if args.apply:
                    fp.write_text(text)
            if LEFTOVER is not None:
                for m in LEFTOVER.finditer(text):
                    i = max(0, m.start() - 60)
                    leftovers.append(f"{u}/{f}: ...{text[i:m.end()+80]}...")

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] {sum(touched.values())} replacements across "
          f"{len({k.split('/')[0] for k in touched})} personas / {len(touched)} files")
    if leftovers:
        print(f"LEFTOVERS NEEDING REVIEW ({len(leftovers)}):")
        for s in leftovers[:20]:
            print("  ", s[:200])
    return 1 if leftovers else 0


if __name__ == "__main__":
    sys.exit(main())
