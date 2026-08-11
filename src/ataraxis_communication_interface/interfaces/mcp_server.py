"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

Exposes microcontroller discovery, MQTT broker connectivity checking, log archive assembly, recording discovery,
microcontroller manifest management, extraction configuration management, microcontroller data log processing, output
verification, output cleanup, and extracted event querying through the MCP protocol, enabling AI agents to
programmatically interact with the library's core features.
"""

from __future__ import annotations

from typing import Literal

from . import (
    config_tools,  # noqa: F401
    output_tools,  # noqa: F401
    discovery_tools,  # noqa: F401
    processing_tools,  # noqa: F401
)
from .mcp_instance import mcp


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication, 'sse' for Server-Sent Events, and 'streamable-http' for HTTP-based communication.
    """
    # Delegates to the MCPServer run loop, which blocks until the transport connection is closed. For 'stdio', the
    # server runs until the parent process closes stdin. For 'streamable-http', runs an HTTP server that accepts
    # connections until explicitly terminated.
    if transport == "streamable-http":
        # Frames each response as a single JSON body instead of an event stream. Only the streamable-http transport
        # accepts this flag, so it stays out of the call below.
        mcp.run(transport=transport, json_response=True)
        return

    mcp.run(transport=transport)


def run_mcp_server() -> None:
    """Starts the MCP server with stdio transport.

    Wraps run_server() with the stdio transport protocol recommended for Claude Desktop integration.
    """
    run_server(transport="stdio")
