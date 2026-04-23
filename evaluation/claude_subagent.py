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


# Headless `claude -p` doesn't expose Glob/Grep as standalone tools — those only
# exist in interactive sessions. Filesystem navigation in headless mode is
# ordinarily done via Bash (find/ls/grep), but Bash opens a trivial sandbox
# escape (`cat /etc/passwd`). Solution: restrict to Read only, and put a README
# in the snapshot directory that enumerates every file the agent would want.
DEFAULT_ALLOWED_TOOLS = ("Read",)
DEFAULT_DENY_TOOLS = ("Bash", "Edit", "Write", "WebFetch", "WebSearch", "Task", "NotebookEdit")


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

    # Path-pattern syntax: `//abs/path/**` — the leading `/` plus the
    # absolute path. Without `--setting-sources ""` the subprocess inherits
    # permissive project / user / local settings from the caller's Claude
    # Code session, which silently overrides these allow patterns.
    fs_allowed_patterns = [f"{tool}(/{snapshot_abs}/**)" for tool in allowed_tools]
    # MCP patterns (e.g., "mcp__instagram__*") are added directly — no path
    # scoping needed because the MCP server enforces its own scope.
    mcp_patterns = list(mcp_tool_patterns or [])
    allowed_patterns = fs_allowed_patterns + mcp_patterns

    cmd = [
        claude_bin,
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
        "--setting-sources", "",
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--add-dir", snapshot_abs,
    ]
    if allowed_patterns:
        cmd += ["--allowedTools", *allowed_patterns]
    # Always deny the listed built-in tools (Bash, Edit, Write, WebFetch, etc.).
    cmd += ["--disallowedTools", *DEFAULT_DENY_TOOLS]
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

    usage = (raw.get("usage") or {})
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
