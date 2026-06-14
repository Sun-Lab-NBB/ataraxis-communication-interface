from typing import Literal

from . import (
    config_tools as config_tools,
    output_tools as output_tools,
    discovery_tools as discovery_tools,
    processing_tools as processing_tools,
)
from .mcp_instance import mcp as mcp

def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
def run_mcp_server() -> None: ...
