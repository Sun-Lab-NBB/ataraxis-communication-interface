from typing import Any
from pathlib import Path

from mcp.server import MCPServer

mcp: MCPServer

def read_tracker_status(tracker_path: Path) -> dict[str, Any]: ...
