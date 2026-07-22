"""Spawn a real Codex CLI agent against a time-masked snapshot directory.

This is the Codex-harness sibling of ``evaluation.claude_subagent`` for the
filesystem snapshot mode. It deliberately does not use the OpenAI API client in
``query_llm.py``: each row is answered by ``codex exec`` with GPT-5.5, read-only
filesystem access, and the same snapshot materialized for Claude ``agent_tools``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from evaluation.claude_subagent import _OVERPERS_RAIL


_ACCESS_CODEX_FS = """\
You are an assistant acting on behalf of this user. Their time-masked cross-app
history is in JSON files in your current working directory. You may inspect the
snapshot with read-only shell/file operations.

The files are large (hundreds of KB each). Do not dump full backend databases
or read whole app timelines into context. Search for only the slices you need:
- First inspect README.md; it is small and lists every file and the event schema.
- Then search for specific hashtags, topics, friend names, dates, or apps across
  the JSON files using tools such as rg/grep when available.
- Then read narrow line ranges around the matches, for example with sed -n.

Ground your answer in what this user has actually done. Search first, avoid
generic guessing, and keep the final answer concise.
"""


FEATURE_DISABLE_FLAGS = (
    "plugins",
    "remote_plugin",
    "apps",
    "enable_mcp_apps",
    "browser_use",
    "browser_use_external",
    "image_generation",
    "multi_agent",
)

_RATE_LIMIT_MARKERS = (
    "hit your limit",
    "usage limit",
    "rate limit",
    "reached your usage",
    "try again later",
    "resets ",
)


@dataclass
class CodexSubagentResult:
    text: str
    turns: int
    duration_ms: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    permission_denials: list
    is_error: bool
    raw: dict


class CodexRateLimitError(RuntimeError):
    """Raised when Codex CLI returns a usage/rate-limit notice as the answer."""


def find_codex_binary() -> str:
    path = shutil.which("codex")
    if path:
        return path
    raise FileNotFoundError("`codex` CLI not found on PATH.")


def _source_codex_home() -> Path:
    explicit = os.getenv("EVAL_CODEX_AUTH_HOME") or os.getenv("EVAL_CODEX_SOURCE_HOME")
    if explicit:
        return Path(explicit).expanduser()
    env_home = os.getenv("CODEX_HOME")
    if env_home and not str(env_home).startswith("/tmp/pm3_codex_home"):
        return Path(env_home).expanduser()
    return Path.home() / ".codex"


def _prepare_codex_home() -> Path:
    """Return a writable per-worker CODEX_HOME with auth copied in.

    The cluster mounts the real home read-only inside sandboxed commands. Codex
    still needs to create state sqlite files even with ``--ephemeral``, so each
    eval worker gets a private temp CODEX_HOME. Auth is copied from the user's
    real Codex home; generated state remains under /tmp.
    """
    override = os.getenv("EVAL_CODEX_HOME")
    if override:
        dest = Path(override).expanduser()
    else:
        base = Path(os.getenv("EVAL_CODEX_HOME_BASE", "/tmp/pm3_codex_home"))
        dest = base / f"worker_{os.getpid()}"
    dest.mkdir(parents=True, exist_ok=True)

    src = _source_codex_home()
    for name in ("auth.json", "installation_id"):
        src_file = src / name
        dest_file = dest / name
        if dest_file.exists():
            continue
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            if name == "auth.json":
                try:
                    dest_file.chmod(0o600)
                except OSError:
                    pass
    if not (dest / "auth.json").exists():
        raise FileNotFoundError(
            f"Codex auth not found at {dest / 'auth.json'} and source {src / 'auth.json'}"
        )
    return dest


def _json_events(stdout: str) -> list[dict]:
    events: list[dict] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _content_text(value) -> str:
    """Best-effort text extraction from Codex/OpenAI-style content payloads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_content_text(v) for v in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        for key in ("text", "output_text", "content"):
            text = _content_text(value.get(key))
            if text:
                return text
    return ""


def _item_text(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("text", "output_text", "content", "message"):
        text = _content_text(item.get(key))
        if text:
            return text
    return ""


def _last_agent_message(events: list[dict]) -> str:
    for ev in reversed(events):
        if ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            if isinstance(item, dict) and item.get("type") in (
                "agent_message",
                "assistant_message",
                "message",
            ):
                text = _item_text(item)
                if text:
                    return text
        if ev.get("type") in ("agent_message", "assistant_message", "message"):
            text = _item_text(ev)
            if text:
                return text
    return ""


def _usage_from_events(events: list[dict]) -> dict:
    usage: dict = {}
    for ev in events:
        if ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
            usage = ev["usage"]
    total_input = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    # Codex/OpenAI usage reports cached tokens as a subset of input tokens. The
    # eval schema expects fresh input and cache-read tokens separately.
    fresh_input = max(0, total_input - cached)
    return {
        "input_tokens": fresh_input,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_tokens": cached,
        "cache_creation_tokens": 0,
        "raw_usage": usage,
    }


def _write_trace(reason: str, *, events: list[dict], stderr: str, returncode: int, usage: dict) -> None:
    """Persist a small local trace for failed/empty Codex turns."""
    run_dir = os.getenv("PM3_RUN_DIR")
    if not run_dir:
        return
    trace_dir = Path(run_dir) / "codex_traces"
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        qid = (os.getenv("PM3_QUERY_ID") or "unknown").replace("/", "_").replace(":", "_")
        path = trace_dir / f"{qid}.{reason}.{uuid4().hex[:8]}.json"
        path.write_text(
            json.dumps(
                {
                    "reason": reason,
                    "returncode": returncode,
                    "stderr_tail": (stderr or "")[-4000:],
                    "usage": usage,
                    "events_tail": events[-20:],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        return


def _write_success_trace(
    *,
    text: str,
    events: list[dict],
    stderr: str,
    returncode: int,
    usage: dict,
    duration_ms: int,
) -> None:
    """Persist full Codex JSON event streams for explicit trajectory audits."""
    if os.getenv("EVAL_CODEX_TRACE_SUCCESS", "").lower() not in {"1", "true", "yes"}:
        return
    run_dir = os.getenv("PM3_RUN_DIR")
    if not run_dir:
        return
    trace_dir = Path(run_dir) / "codex_success_traces"
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        qid = (os.getenv("PM3_QUERY_ID") or "unknown").replace("/", "_").replace(":", "_")
        path = trace_dir / f"{qid}.success.{uuid4().hex[:8]}.json"
        path.write_text(
            json.dumps(
                {
                    "reason": "success",
                    "query_id": os.getenv("PM3_QUERY_ID") or "",
                    "task_type": os.getenv("PM3_TASK_TYPE") or "",
                    "returncode": returncode,
                    "duration_ms": duration_ms,
                    "stderr_tail": (stderr or "")[-4000:],
                    "usage": usage,
                    "final_text": text or "",
                    "event_count": len(events),
                    "events": events,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        return


def run_codex_subagent(
    prompt: str,
    snapshot_dir: Path,
    model: str = "gpt-5.5",
    timeout_seconds: int = 600,
    task_type: str | None = None,
    extra_env: dict | None = None,
) -> CodexSubagentResult:
    del task_type  # Reserved for parity with run_subagent.

    codex_bin = find_codex_binary()
    codex_home = _prepare_codex_home()
    snapshot_abs = str(Path(snapshot_dir).resolve())
    final_prompt = _ACCESS_CODEX_FS + _OVERPERS_RAIL + prompt

    base_cmd = [codex_bin, "exec"]
    for feature in FEATURE_DISABLE_FLAGS:
        base_cmd += ["--disable", feature]
    base_cmd += [
        "--model", model,
        "--cd", snapshot_abs,
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--json",
    ]

    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    if extra_env:
        env.update(extra_env)

    attempts = max(1, int(os.getenv("EVAL_CODEX_ATTEMPTS", "2") or "2"))
    total_duration_ms = 0
    total_turns = 0
    agg_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    last_events: list[dict] = []
    last_stderr = ""
    last_returncode = 0

    for attempt in range(1, attempts + 1):
        out_path = Path(tempfile.gettempdir()) / f"pm3_codex_last_{os.getpid()}_{uuid4().hex}.txt"
        cmd = base_cmd + ["--output-last-message", str(out_path), "-"]
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd=snapshot_abs,
            input=final_prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        duration_ms = int((time.time() - t0) * 1000)
        total_duration_ms += duration_ms

        events = _json_events(proc.stdout)
        last_events = events
        last_stderr = proc.stderr
        last_returncode = proc.returncode
        text = ""
        if out_path.exists():
            try:
                text = out_path.read_text(encoding="utf-8")
            finally:
                try:
                    out_path.unlink()
                except OSError:
                    pass
        if not text:
            text = _last_agent_message(events)

        probe = (text or proc.stdout or proc.stderr or "").strip().lower()
        if len(probe) < 500 and any(m in probe for m in _RATE_LIMIT_MARKERS):
            raise CodexRateLimitError(probe[:240])

        usage = _usage_from_events(events)
        for key in agg_usage:
            agg_usage[key] += int(usage.get(key) or 0)
        total_turns += sum(1 for ev in events if ev.get("type") == "turn.completed")

        is_error = proc.returncode != 0 or any(ev.get("type") == "turn.failed" for ev in events)
        if is_error:
            _write_trace(
                "error",
                events=events,
                stderr=proc.stderr,
                returncode=proc.returncode,
                usage=usage.get("raw_usage") or {},
            )
            err_msg = ""
            for ev in reversed(events):
                if ev.get("type") == "turn.failed":
                    err = ev.get("error") or {}
                    if isinstance(err, dict):
                        err_msg = err.get("message") or ""
                    break
            raise RuntimeError(
                f"codex exec failed rc={proc.returncode}: "
                f"{(err_msg or proc.stderr or proc.stdout or 'unknown error')[:500]}"
            )

        if (text or "").strip():
            _write_success_trace(
                text=text,
                events=events,
                stderr=proc.stderr,
                returncode=proc.returncode,
                usage=usage.get("raw_usage") or {},
                duration_ms=duration_ms,
            )
            return CodexSubagentResult(
                text=text,
                turns=total_turns,
                duration_ms=total_duration_ms,
                cost_usd=0.0,
                input_tokens=agg_usage["input_tokens"],
                output_tokens=agg_usage["output_tokens"],
                cache_read_tokens=agg_usage["cache_read_tokens"],
                cache_creation_tokens=agg_usage["cache_creation_tokens"],
                permission_denials=[],
                is_error=False,
                raw={
                    "events": events,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                    "usage": usage.get("raw_usage") or {},
                    "attempts": attempt,
                },
            )

        _write_trace(
            "empty",
            events=events,
            stderr=proc.stderr,
            returncode=proc.returncode,
            usage=usage.get("raw_usage") or {},
        )
        if attempt < attempts:
            time.sleep(min(2 * attempt, 6))

    raise RuntimeError(
        "codex exec produced an empty final answer after "
        f"{attempts} attempt(s); rc={last_returncode}; "
        f"stderr={(last_stderr or '')[-240:]}; events={len(last_events)}"
    )
