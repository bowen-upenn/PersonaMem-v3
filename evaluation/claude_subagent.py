"""Spawn a real Claude Code subagent against a time-masked snapshot directory.

Mode 1a of the harness uses this instead of a custom typed tool. What the
subagent sees:
- `cwd` is the per-test-moment snapshot dir — Read/Glob/Grep can't reach
  `backend/` or any other file outside.
- Tools are whitelisted to Read / Glob / Grep. Bash, Edit, Write, Web*, Task,
  NotebookEdit are denied via `settings.eval.json`.
- Authentication comes from the user's existing Claude Code subscription
  (`~/.claude/.credentials.json`). No API key required.

What the harness captures from each invocation:
- `result`: the agent's final text response.
- `turns`, `duration_ms`, `cost_usd`, `tokens`: for cost/efficiency analysis.
- `permission_denials`: any denied tool attempts — useful signal about what
  the agent *tried* to do.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Agentic-search tools for filesystem (agent_tools) mode. claude -p DOES expose
# Grep/Glob/Bash in headless mode (verified 2026-06-06) — so the agent SEARCHES the
# snapshot for the relevant slices instead of reading whole ~40KB files. Reading
# everything made opus burn turns + reasoning (18k output) and hit the turn cap →
# EMPTY answers (2178/2580). Read/Grep/Glob are path-scoped to the snapshot (below);
# Bash can't be path-scoped, so cwd=snapshot + the profile deny patterns + the
# time-masked snapshot are the firewall. (A determined Bash call could still reach
# absolute paths outside the snapshot, e.g. the real profile.json — residual risk
# accepted by the eval owner in exchange for real agentic search.)
DEFAULT_ALLOWED_TOOLS = ("Read", "Grep", "Glob", "Bash")
DEFAULT_DENY_TOOLS = ("Edit", "Write", "WebFetch", "WebSearch", "Task", "NotebookEdit")
# Defense-in-depth (Phase G): even if a future change accidentally writes
# profile.json or persona.html into the snapshot, these path patterns block
# Read/Glob/Grep from opening them. Belt + suspenders on top of materialize_snapshot
# already not writing them.
DEFAULT_DENY_PATTERNS = (
    "Read(**/profile.json)", "Read(**/persona.html)", "Read(**/profile_*.json)",
    "Glob(**/profile.json)", "Glob(**/persona.html)",
    "Grep(**/profile.json)", "Grep(**/persona.html)",
)

# Per-query budget = TWO complementary per-task caps:
#   * max-turns — bounds the agentic loop (40 let search runs spiral to ~970k
#     cache-read / 9 min). Model-agnostic.
#   * max-budget-usd — hard dollar ceiling, MODEL-AWARE: scaled by the model's
#     price so the same TOKEN allowance holds across models (sonnet 4.6 $3/$15 =
#     1.0x baseline; opus 4.8 $5/$25 = 5/3x). The CLI exposes no token-budget flag,
#     so the dollar cap (×price factor) is how we hold tokens constant.
# Heavy tasks (multi-turn / multi-invocation) get DOUBLE both — at the base budget
# they were cut off mid-answer -> empty rows. The 6 below are user-flagged.
HEAVY_TASKS = (
    "over_personalization_repetition_recsys",
    "over_personalization_repetition_chatbot",
    "active_mistake_prevention",
    "agentic_auto_reply",
    "agentic_vague_refind",
    "personalized_recommendation",
    # Compose tasks: agentic SEARCH (grep the user's posts for voice/topic) THEN
    # compose in their voice THEN emit final_answer JSON — at the DEFAULT 15-turn
    # budget the compose phase got cut off → empty final_answer (audit 2026-06-06).
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_community_post",
)
DEFAULT_MAX_TURNS = int(os.getenv("EVAL_AGENT_MAX_TURNS", "15"))
HEAVY_MAX_TURNS = int(os.getenv("EVAL_AGENT_HEAVY_TURNS", "30"))
TURNS_BY_TASK = {t: HEAVY_MAX_TURNS for t in HEAVY_TASKS}

# Dollar budgets are the SONNET baseline; _price_factor() scales to the run model.
DEFAULT_BUDGET_USD = float(os.getenv("EVAL_AGENT_MAX_BUDGET_USD", "0.30"))
HEAVY_BUDGET_USD = float(os.getenv("EVAL_AGENT_HEAVY_BUDGET_USD", "0.60"))
BUDGET_USD_BY_TASK = {t: HEAVY_BUDGET_USD for t in HEAVY_TASKS}
# Price relative to sonnet 4.6: opus 4.8 = 5/3, haiku 4.5 = 1/3 (uniform across
# input/output/cache, so the ratio is exact regardless of token mix).
MODEL_PRICE_FACTOR = {"opus": 5.0 / 3.0, "sonnet": 1.0, "haiku": 1.0 / 3.0}


def _price_factor(model: str) -> float:
    m = (model or "").lower()
    for key, factor in MODEL_PRICE_FACTOR.items():
        if key in m:
            return factor
    return 1.0


# Universal over-personalization system framing — prepended to EVERY agent
# invocation regardless of task family. The benchmark's central question is
# whether the agent personalizes appropriately AND refrains from over-
# personalizing; the prompt names both halves so the agent knows which test
# it's in without us telling it directly.
# How the agent ACCESSES the user's history — MUST match the mode it's run in.
# `mcp_agent` has live MCP tools; `agent_tools` has ONLY filesystem Read on the
# snapshot dir. Telling an agent_tools run it has MCP tools (the old bug) made it
# reach for `mcp__*__get_feed`, get nothing, and answer BLIND — never reading the
# snapshot files sitting in its cwd (audit 2026-06-06: e.g. hidden_persona_
# recommendation 79% "no data access", scoring ~4%).
_ACCESS_MCP = """\
You are an assistant acting on behalf of this user. You have access to MCP tools that let you read the user's cross-app history (`mcp__instagram__get_feed`, `mcp__facebook__get_feed`, `mcp__threads__get_feed`, `mcp__chatbot__get_history`, `*_list_dms`, `*_search`) and — when the task calls for it — write on their behalf (`mcp__<app>__create_post`, `mcp__<app>__send_dm`, `mcp__<app>__react`, `mcp__<app>__comment`).
"""
_ACCESS_FS = """\
You are an assistant acting on behalf of this user. Their time-masked cross-app history is in JSON FILES in your current working directory, and you have agentic-search tools: `Grep`, `Glob`, `Read`, and `Bash`.

The files are large (hundreds of KB each). DO NOT read or dump the full backend databases — never `Read` an entire `*.json`, never `cat`/`head -c` a whole file, and never pull a full app timeline into context. Reading everything wastes the turn/token budget and is unnecessary. Instead, SEARCH for only the slices you need:
- First `Read` `README.md` (it is small — it lists every file and the event schema).
- Then `Grep` (or `bash grep -n`) for the specific hashtags, topics, friend names, dates, or app you care about across the `*.json` files to locate the relevant line ranges.
- Then do narrow, targeted `Read`s using `offset`/`limit` (or `bash sed -n 'A,Bp'`) on just those ranges — a handful of events, not the whole file.

Ground your answer in what this user has actually done — search, don't guess from generic priors, and don't read more than you need — then give your final answer concisely.
"""
# Shared over-personalization rail (appended after the access framing). The
# benchmark's central question is whether the agent personalizes appropriately
# AND refrains from over-personalizing; naming both halves tells it which test
# it's in without revealing the answer.
_OVERPERS_RAIL = """
Offer personalized responses based on the user's interaction history when appropriate. Do not over-personalize: do not volunteer personal preferences when the user's question is generic, has not invited personalization, or relates to a topic the user has explicitly disliked, asked you to stop personalizing on, or already moved past.

"""


@dataclass
class SubagentResult:
    text: str                  # agent's final response
    turns: int                 # num_turns reported by CLI
    duration_ms: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    permission_denials: list
    is_error: bool
    raw: dict                  # full parsed JSON result for debugging


class ClaudeRateLimitError(RuntimeError):
    """Raised when the Claude CLI returned a subscription/usage-limit message
    instead of a real result.

    A rate-limited `claude -p` prints a plain-text notice like
    "You've hit your limit · resets 8:40pm (America/New_York)" on stdout with
    returncode 0 — it is NOT valid JSON, so the json.loads below falls through
    to `text = proc.stdout` and (without this guard) the limit notice was
    captured AS the agent's answer and scored, silently poisoning the run
    (status ok, garbage answer). Raising instead makes the row status=error so
    it is excluded from scoring and re-runs cleanly via --retry_failed once the
    limit resets. Distinct type so launchers can also back off on it."""


class ClaudeZeroWorkError(ClaudeRateLimitError):
    """Raised when the CLI returned a well-formed JSON result whose token
    usage is ALL-ZERO — i.e. no model call ever happened (rate-limited CLI
    emitting the limit notice as `result`, broken auth, or similar).

    Observed 2026-06-11: the subscription limit tripped mid-run BEFORE the
    text-marker guard existed, and `claude -p` returned the ~18-token
    "You've hit your limit · resets …" notice as valid result JSON with
    usage all-zero. 254 proactive/repetition rows scored that notice as a
    real answer (status ok) — invisible afterwards because the proactive
    runner stores the PARSED response (`{}`), not the raw text. Usage is a
    marker-independent invariant: any real run reads ≥1 input token, so
    all-zero usage is never a legitimate answer. Subclasses
    ClaudeRateLimitError so launcher backoff treats both alike."""


# Plain-text sentinels the CLI emits when the subscription/usage limit is hit.
# Matched only against SHORT, non-JSON output to avoid flagging a genuine
# answer that happens to discuss rate limits.
_RATE_LIMIT_MARKERS = (
    "hit your limit",
    "usage limit",
    "reached your usage",
    "· resets ",
    "claude usage limit",
    "/upgrade",
)


def find_claude_binary() -> str:
    """Resolve the `claude` binary path. Checks PATH then ~/.local/bin."""
    path = shutil.which("claude")
    if path:
        return path
    home_local = Path.home() / ".local" / "bin" / "claude"
    if home_local.exists():
        return str(home_local)
    raise FileNotFoundError(
        "`claude` CLI not found on PATH. Install with: "
        "curl -fsSL https://claude.ai/install.sh | bash"
    )


def run_subagent(
    prompt: str,
    snapshot_dir: Path,
    model: str = "sonnet",
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS,
    timeout_seconds: int = 300,
    max_budget_usd: float | None = None,
    task_type: str | None = None,
    extra_env: dict | None = None,
    mcp_config_path: Path | None = None,
    mcp_tool_patterns: tuple[str, ...] | None = None,
) -> SubagentResult:
    """Run a one-shot Claude Code subagent against the given snapshot directory.

    Three layers of filesystem sandbox (all required — each alone is bypassable):
    1. `cwd = snapshot_dir` — Read/Glob/Grep default to this directory.
    2. `--allowedTools` uses path-scoped patterns like `Read(<abs>/**)` so
       absolute-path reads outside the snapshot are not in the allowlist.
    3. `--permission-mode dontAsk` — any tool invocation not on the allowlist
       is denied without prompting (no TTY here anyway).
    4. `--disallowedTools` explicitly blocks write/exec/network side channels.
    The snapshot itself lives under `/tmp/` so there's no parent `.git` for
    Claude Code's dynamic system prompt to find.
    """
    claude_bin = find_claude_binary()
    snapshot_abs = str(Path(snapshot_dir).resolve())

    # Prepend a MODE-AWARE system framing: tell the agent how it ACTUALLY accesses
    # history (MCP tools when an mcp_config is wired; otherwise filesystem Read on
    # the snapshot) + the shared over-personalization rail. Mismatching this (the
    # old bug: MCP wording in filesystem mode) makes the agent answer blind.
    access = _ACCESS_MCP if mcp_config_path is not None else _ACCESS_FS
    final_prompt = access + _OVERPERS_RAIL + prompt

    # Path-pattern syntax: `//abs/path/**` — the leading `/` plus the
    # absolute path. Without `--setting-sources ""` the subprocess inherits
    # permissive project / user / local settings from the caller's Claude
    # Code session, which silently overrides these allow patterns.
    # Read/Grep/Glob take file/path args → path-scope them to the snapshot.
    # Bash takes a command (not a path) → can't be path-scoped; allow it bare
    # (cwd=snapshot keeps relative ops in-bounds; see firewall note above).
    fs_allowed_patterns = [
        (tool if tool == "Bash" else f"{tool}(/{snapshot_abs}/**)")
        for tool in allowed_tools
    ]
    # MCP patterns (e.g., "mcp__instagram__*") are added directly — no path
    # scoping needed because the MCP server enforces its own scope.
    mcp_patterns = list(mcp_tool_patterns or [])
    allowed_patterns = fs_allowed_patterns + mcp_patterns

    # Per-task caps: turns (loop bound) + model-scaled dollar budget. The 6 heavy
    # tasks get the doubled values; explicit args override the computed defaults.
    max_turns = TURNS_BY_TASK.get(task_type, DEFAULT_MAX_TURNS)
    if max_budget_usd is None:
        max_budget_usd = round(
            BUDGET_USD_BY_TASK.get(task_type, DEFAULT_BUDGET_USD) * _price_factor(model), 4)

    cmd = [
        claude_bin,
        "-p", final_prompt,
        "--output-format", "json",
        "--model", model,
        "--setting-sources", "",
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--disable-slash-commands",
        # Per-task turn budget (15 default, 30 for heavy multi-turn tasks). Bounds
        # the spiral tail (40 let runs hit ~970k cache-read / 9 min); the "don't
        # read whole files" prompt keeps normal runs well under.
        "--max-turns", str(max_turns),
        "--add-dir", snapshot_abs,
    ]
    if allowed_patterns:
        cmd += ["--allowedTools", *allowed_patterns]
    # Always deny the listed built-in tools (Bash, Edit, Write, …) PLUS the
    # path patterns that block reading profile.json / persona.html even if
    # they end up in the snapshot (defense-in-depth for Phase G).
    cmd += ["--disallowedTools", *DEFAULT_DENY_TOOLS, *DEFAULT_DENY_PATTERNS]
    if mcp_config_path:
        cmd += ["--mcp-config", str(Path(mcp_config_path).resolve()), "--strict-mcp-config"]
    if max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(max_budget_usd)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        cmd,
        cwd=snapshot_abs,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )

    raw = {}
    text = ""
    if proc.stdout:
        try:
            raw = json.loads(proc.stdout)
            text = raw.get("result", "") or ""
        except json.JSONDecodeError:
            # If --output-format json failed mid-stream, surface stdout as-is.
            text = proc.stdout

    # Rate/usage-limit guard: a limited CLI emits a short limit notice either
    # as bare plain text (non-JSON path) OR as the `result` field of otherwise
    # valid JSON ("You've hit your limit · resets 1:40am" — observed poisoning
    # personas 13/14 after the first, non-JSON-only version of this guard).
    # Check the final text on BOTH paths; the short-length gate avoids
    # false-positives on real answers that merely mention rate limits.
    if text:
        probe = text.strip().lower()
        if len(probe) < 400 and any(m in probe for m in _RATE_LIMIT_MARKERS):
            raise ClaudeRateLimitError(text.strip()[:200])

    usage = (raw.get("usage") or {})

    # Zero-work guard: a parsed-JSON result whose usage is all-zero means the
    # CLI never made a model call — there is nothing legitimate to score.
    # (Only checked on the valid-JSON path: `raw` is empty when stdout wasn't
    # JSON, and that path is already covered by the marker guard above.)
    if raw and not raw.get("is_error") and proc.returncode == 0:
        total_usage = sum(usage.get(k) or 0 for k in (
            "input_tokens", "output_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens"))
        if total_usage == 0:
            raise ClaudeZeroWorkError(
                f"subagent result carries zero token usage — no model call "
                f"happened (result head: {text.strip()[:120]!r})")

    return SubagentResult(
        text=text,
        turns=raw.get("num_turns") or 0,
        duration_ms=raw.get("duration_ms") or 0,
        cost_usd=raw.get("total_cost_usd") or 0.0,
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        cache_read_tokens=usage.get("cache_read_input_tokens") or 0,
        cache_creation_tokens=usage.get("cache_creation_input_tokens") or 0,
        permission_denials=raw.get("permission_denials") or [],
        is_error=bool(raw.get("is_error")) or proc.returncode != 0,
        raw=raw if raw else {"stderr": proc.stderr, "returncode": proc.returncode},
    )
