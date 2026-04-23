"""Instagram mock MCP server entry point. See `_social_server.make_server`."""
import os
os.environ.setdefault("PM3_APP", "instagram")
from evaluation.mcp_servers._social_server import run_server

if __name__ == "__main__":
    run_server()
