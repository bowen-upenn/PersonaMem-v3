"""Generate per-invocation `--mcp-config` JSON for `claude -p --mcp-config`.

Given a test moment (user_id, t_test) and the set of apps the agent is
allowed to touch, emit a config file listing the stdio commands to start
the right mock MCP servers with the right time-mask / overlay paths.

Claude Code's `--mcp-config` format:
    {"mcpServers": {"<name>": {"command": "python", "args": [...], "env": {...}}}}
"""

from __future__ import annotations

import json
from pathlib import Path


SOCIAL_APP_MODULES = {
    "instagram": "evaluation.mcp_servers.instagram_mcp_server",
    "facebook":  "evaluation.mcp_servers.facebook_mcp_server",
    "threads":   "evaluation.mcp_servers.threads_mcp_server",
}

CHATBOT_MODULE = "evaluation.mcp_servers.chatbot_mcp_server"


def build_mcp_config(
    user_id: str,
    t_test: int,
    overlay_path: Path,
    backend_dir: str = "backend",
    enabled_apps: tuple[str, ...] = ("instagram", "facebook", "threads", "chatbot"),
    python_exe: str = "python",
) -> dict:
    servers: dict[str, dict] = {}
    base_env = {
        "PM3_USER_ID": user_id,
        "PM3_T_TEST": str(t_test),
        "PM3_BACKEND_DIR": backend_dir,
        "PM3_OVERLAY_PATH": str(overlay_path),
    }
    for app in enabled_apps:
        if app == "chatbot":
            servers["chatbot"] = {
                "command": python_exe,
                "args": ["-m", CHATBOT_MODULE],
                "env": dict(base_env),
            }
        elif app in SOCIAL_APP_MODULES:
            servers[app] = {
                "command": python_exe,
                "args": ["-m", SOCIAL_APP_MODULES[app]],
                "env": {**base_env, "PM3_APP": app},
            }
    return {"mcpServers": servers}


def write_mcp_config(path: Path, config: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


def mcp_allowed_tools(enabled_apps: tuple[str, ...]) -> list[str]:
    """Return the `--allowedTools` patterns for the given app set.

    MCP tools show up as `mcp__<server_name>__<tool_name>`; we allow all
    tools from each server (path-scoping would be meaningless here since
    the servers already scope by time-mask + overlay).
    """
    return [f"mcp__{app}__*" for app in enabled_apps]
