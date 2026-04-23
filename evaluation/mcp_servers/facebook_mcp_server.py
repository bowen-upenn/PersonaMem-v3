"""Facebook mock MCP server entry point."""
import os
os.environ.setdefault("PM3_APP", "facebook")
from evaluation.mcp_servers._social_server import run_server

if __name__ == "__main__":
    run_server()
