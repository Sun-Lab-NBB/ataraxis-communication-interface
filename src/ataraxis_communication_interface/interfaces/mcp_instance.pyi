from typing import Any
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp: FastMCP

def read_tracker_status(tracker_path: Path) -> dict[str, Any]: ...
