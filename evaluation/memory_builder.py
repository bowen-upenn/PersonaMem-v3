"""Memory mode — iterative, text-only memory construction for the eval harness.

The `memory` eval mode follows the `llm_longctx` style (single answer call,
context injected into the prompt) but, instead of dumping the raw cross-app
event history, a memory agent reads that history in chronological chunks and
consolidates it into ONE bounded, plain-text memory document, which is then
injected into the answering prompt.

Algorithm: Chain-of-Memory (CoM, arXiv 2601.14287, 2026) dynamic-evolution
construction, grounded on Mem0's (arXiv 2504.19413) ADD/UPDATE/DELETE/NOOP edit
contract — reimplemented natively against this repo's `QueryLLM` +
`BackendQuery`.

TEXT-ONLY HARD INVARIANT
------------------------
The memory module is a plain-text (markdown) document and nothing else. NO RAG,
NO embeddings, NO vector/similarity search, NO graph DB, NO top-k retrieval —
not at build time and not at answer time. The LLM sees the ENTIRE current memory
each build step and the ENTIRE final memory in the answer prompt. All
dedup/merge/eviction here is literal string + `[topic]`-tag + recency/occurrence
logic (text-derived); never vector similarity. No `faiss`/`chromadb`/`qdrant`/
`sentence-transformers`/`neo4j` import is added. Mem0's "retrieve top-s similar"
step is intentionally replaced by "show the whole memory to the LLM".

Firewall
--------
- The memory at boundary `T_test` reflects ONLY events with
  `source_timestamp < T_test` (strict `<`, matching BackendQuery's mask).
- `profile.json` is NEVER read — only `bq.get_events` (same masked, leak-stripped
  view `serialize_history_for_context` uses). No demographics/hidden-personas.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from evaluation.backend_query import APPS, BackendQuery
from evaluation.inference_utils import _compact_event, count_tokens
from evaluation.prompts import llm_memory_update_prompt

# ONE persona/preference-centered, human-readable text memory (no vectors). The
# real `mem0ai` library is the OTHER memory baseline and lives in
# `mem0_backend.py` — it is NOT built through this module.
MEMORY_ALGOS = ("llm_memory",)

# Stable tie-break so the merged stream is byte-deterministic across runs.
APP_ORDER = {app: i for i, app in enumerate(APPS)}

EMPTY_MEMORY = """# USER MEMORY (last activity seen: none yet)

## Who they are
(none yet)

## Interests & preferences
(none yet)

## People & places
(none yet)

## Currently active
(none yet)
"""


def default_memory_config() -> dict:
    """Default knobs for the memory builder. Mirrors the run_eval CLI flags."""
    return {
        "token_cap": 2048,
        "chunk_k": 40,
        "chunk_gap": 900,          # seconds; >15-min gap ends a chunk (if big enough)
        "chunk_tok_budget": 4000,
        "builder_temperature": 0.0,
        "builder_model": None,     # metadata only; the passed llm_client owns the model
    }


# ---------------------------------------------------------------------------
# Chronological merge across apps
# ---------------------------------------------------------------------------

def build_global_stream(bq: BackendQuery, user_id: str, t_max: int) -> list[dict]:
    """Merge all 4 `APPS` into ONE chronological stream of compacted events
    with `source_timestamp < t_max`.

    Uses the IDENTICAL `_compact_event` view the llm_longctx baseline uses
    (fairness), then stamps each event with its `app` (compact drops it) and
    `oid` (for a stable tie-break). Sorted ascending by
    `(t, APP_ORDER[app], oid)` for byte-determinism.
    """
    rows: list[dict] = []
    for app in APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_max):
            c = _compact_event(e)
            c["app"] = app
            c["oid"] = str(e.get("source_object_id") or "")
            # `_compact_event` may drop t if missing; default to 0 so sort is safe.
            c["t"] = int(c.get("t") or e.get("source_timestamp") or 0)
            rows.append(c)
    rows.sort(key=lambda c: (c["t"], APP_ORDER.get(c["app"], 99), c["oid"]))
    return rows


def _should_close_chunk(buffer: list[dict], cfg: dict, model: str | None) -> bool:
    """True when `buffer` should be flushed as one chunk (K / token / gap)."""
    if len(buffer) >= cfg["chunk_k"]:
        return True
    # Session gap: a big jump between the last two events closes a chunk, but
    # only once the chunk is already substantial (avoid 1-event chunks).
    if len(buffer) >= 8:
        gap = buffer[-1]["t"] - buffer[-2]["t"]
        if gap > cfg["chunk_gap"]:
            return True
    # Soft token budget (chatbot/conversation events are large).
    if count_tokens(_render_chunk(buffer), model) >= cfg["chunk_tok_budget"]:
        return True
    return False


# ---------------------------------------------------------------------------
# Rendering events for the build prompt
# ---------------------------------------------------------------------------

def _trunc(s, n: int = 220) -> str:
    s = "" if s is None else str(s).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_event_line(c: dict) -> str:
    when = c.get("when") or c.get("t")
    app = c.get("app", "?")
    typ = c.get("type", "")
    action = c.get("action", "")
    tags = ",".join(c.get("hashtags", []) or [])
    bits = [f"{when}", f"[{app}]", typ, f"| {action}"]
    if tags:
        bits.append(f"| tags=[{tags}]")
    loc = c.get("location") or {}
    if isinstance(loc, dict) and loc.get("city"):
        bits.append(f"| loc={loc.get('city')}")
    title = c.get("title")
    if title:
        bits.append(f"| title={_trunc(title, 120)}")
    caption = c.get("caption")
    if caption:
        bits.append(f"| {_trunc(caption, 160)}")
    msg = c.get("user_message")
    if msg:
        bits.append(f"| msg={_trunc(msg, 200)}")
    convo = c.get("conversation")
    if convo:
        flat = " ".join(
            f"{m.get('role', '?')}:{_trunc(m.get('content', ''), 120)}"
            for m in convo[:6]
        )
        bits.append(f"| convo={_trunc(flat, 360)}")
    prefs = c.get("inferred_prefs")
    if prefs:
        ps = "; ".join(_trunc(p.get("persona_item", ""), 80) for p in prefs[:4])
        bits.append(f"| inferred=[{ps}]")
    return " ".join(b for b in bits if b)


def _render_chunk(chunk: list[dict]) -> str:
    return "\n".join(_render_event_line(c) for c in chunk)


# ---------------------------------------------------------------------------
# One memory-evolution step (the iterative update call)
# ---------------------------------------------------------------------------

def _parse_memory_summary(resp: str, fallback_m: str, fallback_s: str) -> tuple[str, str]:
    m = re.search(r"<memory>(.*?)</memory>", resp, re.S)
    s = re.search(r"<summary>(.*?)</summary>", resp, re.S)
    if m:
        new_m = m.group(1).strip()
    elif resp and "# USER MEMORY" in resp:
        # Model emitted the doc without tags — salvage it.
        new_m = resp.strip()
    else:
        new_m = fallback_m
    new_s = s.group(1).strip() if s else fallback_s
    return new_m or fallback_m, new_s


def update_step(
    memory: str,
    summary: str,
    chunk: list[dict],
    llm_client,
    *,
    algo: str = "llm_memory",
    temperature: float = 0.0,
    token_cap: int = 2048,
    model: str | None = None,
) -> tuple[str, str, int, int]:
    """Run ONE memory build LLM call over `chunk` using the persona/preference
    memory prompt.

    Returns `(new_memory, new_summary, input_tokens, output_tokens)`. The model
    rewrites the whole doc; we then enforce the token cap with text-only
    consolidation as a safety net. `algo` is retained only as a state-file label.
    """
    prompt = llm_memory_update_prompt(memory, summary, _render_chunk(chunk))
    resp = llm_client.query_llm(prompt, temperature=temperature) or ""
    new_m, new_s = _parse_memory_summary(resp, memory, summary)
    new_m = consolidate_evict(new_m, token_cap, model=model)
    return new_m, new_s, count_tokens(prompt, model), count_tokens(resp, model)


# ---------------------------------------------------------------------------
# Text-only consolidation / eviction (NO embeddings)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"·\s*last\s*(\d{1,2})/(\d{1,2})", re.I)
_OCC_RE = re.compile(r"[×x]\s*(\d+)")


def _line_salience(line: str) -> float:
    """Text-derived salience: reinforcement (from a trailing `[×N]`) and, if a
    line happens to carry a `· last MM/DD` date, recency. Higher = keep. No
    embeddings, and content-neutral — no line is privileged by topic/polarity."""
    recency = 0.0
    md = _DATE_RE.search(line)
    if md:
        mm, dd = int(md.group(1)), int(md.group(2))
        recency = float(mm * 31 + dd) / 400.0  # normalized within-year ordinal
    occ = 1
    mo = _OCC_RE.search(line)
    if mo:
        occ = max(1, int(mo.group(1)))
    return (1.0 + recency) * (1.0 + math.log(occ))


def consolidate_evict(md: str, cap: int, model: str | None = None) -> str:
    """Enforce the token cap with text-only, CONTENT-NEUTRAL logic: drop the
    lowest-salience bullet lines until under cap. No pinning, no topic/polarity
    privilege, no embeddings — fairness invariant for the memory baseline."""
    if count_tokens(md, model) <= cap:
        return md

    lines = md.split("\n")
    evictable: list[tuple[int, float]] = []  # (line_idx, salience)
    for i, ln in enumerate(lines):
        if ln.strip().startswith("- "):
            evictable.append((i, _line_salience(ln)))

    # Drop lowest-salience bullets until under cap.
    evictable.sort(key=lambda x: x[1])
    dropped: set[int] = set()
    for idx, _sal in evictable:
        if count_tokens(
            "\n".join(l for j, l in enumerate(lines) if j not in dropped), model
        ) <= cap:
            break
        dropped.add(idx)
    return "\n".join(l for j, l in enumerate(lines) if j not in dropped)


# ---------------------------------------------------------------------------
# Incremental checkpointing — build once, snapshot at each T_test boundary
# ---------------------------------------------------------------------------

@dataclass
class MemoryLedger:
    """Per-user monotonic memory: T_test boundary -> consolidated memory string."""

    user_id: str
    checkpoints: dict[int, str] = field(default_factory=dict)
    build_stats: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "calls": 0, "n_events": 0,
    })

    def get(self, t_test: int) -> str:
        """Memory for `t_test`. Exact hit preferred; else the nearest boundary
        <= t_test (firewall-safe: a subset of events < t_test). EMPTY if none."""
        t = int(t_test)
        if t in self.checkpoints:
            return self.checkpoints[t]
        prior = [b for b in self.checkpoints if b <= t]
        if prior:
            return self.checkpoints[max(prior)]
        return EMPTY_MEMORY


def build_checkpoints(
    bq: BackendQuery,
    user_id: str,
    t_tests,
    llm_client,
    cfg: dict | None = None,
    *,
    algo: str = "llm_memory",
    run_dir=None,
    existing: dict[int, str] | None = None,
) -> MemoryLedger:
    """Walk the user's global event stream ONCE in ascending order and snapshot
    the consolidated persona/preference memory at each ascending `T_test`
    boundary.

    Correctness/firewall: a boundary `b`'s snapshot is taken AFTER folding every
    event with `t < b` and BEFORE folding any event with `t >= b`. Because the
    stream is sorted ascending, that is a clean prefix cut.

    `existing` (resume): boundaries already present are reused; the walk still
    folds their events so later boundaries are correct, but skips re-spending on
    boundaries we already have.
    """
    if algo not in MEMORY_ALGOS:
        raise ValueError(f"unknown memory algo {algo!r}; expected one of {MEMORY_ALGOS}")
    cfg = {**default_memory_config(), **(cfg or {})}
    cap = cfg["token_cap"]
    model = cfg.get("builder_model")
    ledger = MemoryLedger(user_id=user_id)
    if existing:
        ledger.checkpoints.update({int(k): v for k, v in existing.items()})

    boundaries = sorted({int(t) for t in t_tests})
    if not boundaries:
        return ledger
    rows = build_global_stream(bq, user_id, boundaries[-1])
    ledger.build_stats["n_events"] = len(rows)

    memory, summary = EMPTY_MEMORY, ""
    buffer: list[dict] = []
    bi = 0

    def _flush() -> None:
        nonlocal memory, summary, buffer
        if not buffer:
            return
        memory, summary, in_tok, out_tok = update_step(
            memory, summary, buffer, llm_client, algo=algo,
            temperature=cfg["builder_temperature"], token_cap=cap, model=model,
        )
        ledger.build_stats["input_tokens"] += in_tok
        ledger.build_stats["output_tokens"] += out_tok
        ledger.build_stats["calls"] += 1
        buffer = []

    def _snapshot(b: int) -> None:
        # Reuse an existing checkpoint string if resuming; else render now.
        snap = ledger.checkpoints.get(b)
        if snap is None:
            snap = consolidate_evict(memory, cap, model=model)
            ledger.checkpoints[b] = snap
        if run_dir is not None:
            _dump_state(run_dir, user_id, b, snap, ledger.build_stats, algo)

    for row in rows:
        # Before consuming a row at time t, snapshot every boundary <= t: all
        # buffered/folded events so far are strictly < that boundary.
        while bi < len(boundaries) and boundaries[bi] <= row["t"]:
            _flush()
            _snapshot(boundaries[bi])
            bi += 1
        buffer.append(row)
        if _should_close_chunk(buffer, cfg, model):
            _flush()
    # Drain remaining buffer + snapshot any boundaries after the last event.
    _flush()
    while bi < len(boundaries):
        _snapshot(boundaries[bi])
        bi += 1
    return ledger


def _dump_state(run_dir, user_id: str, t_test: int, memory: str, build_stats: dict, algo: str) -> None:
    """Write a debug/resume checkpoint under run_dir ONLY (never backend/).
    Namespaced by `algo` (the memory-mode label) for stable state filenames."""
    from pathlib import Path
    states_dir = Path(run_dir) / "memory_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    (states_dir / f"{user_id}_{algo}_T{t_test}.json").write_text(
        json.dumps(
            {
                "user_id": user_id,
                "algo": algo,
                "t_test": t_test,
                "memory": memory,
                "build_calls": build_stats.get("calls"),
                "build_input_tokens": build_stats.get("input_tokens"),
                "build_output_tokens": build_stats.get("output_tokens"),
                "n_events": build_stats.get("n_events"),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def load_existing_checkpoints(run_dir, user_id: str, algo: str) -> dict[int, str]:
    """Reload persisted checkpoints for `algo` (for --resume) so we skip rebuilding."""
    from pathlib import Path
    states_dir = Path(run_dir) / "memory_states"
    out: dict[int, str] = {}
    if not states_dir.exists():
        return out
    for p in states_dir.glob(f"{user_id}_{algo}_T*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out[int(d["t_test"])] = d["memory"]
        except Exception:
            continue
    return out
