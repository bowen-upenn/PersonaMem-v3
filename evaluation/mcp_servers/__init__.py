"""Mock MCP servers for eval Mode 1a′ (mcp_agent).

Each server exposes read+write+search tools scoped to one app's view of the
user's backend. Write tools append to a per-run `writes.jsonl` via OverlayView;
they never mutate `backend/{user_id}/`.
"""
