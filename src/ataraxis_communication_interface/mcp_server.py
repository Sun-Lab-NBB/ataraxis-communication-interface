"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

This module exposes microcontroller discovery and MQTT broker connectivity checking functionality through the MCP
protocol, enabling AI agents to programmatically interact with the library's core features.
"""

from typing import TYPE_CHECKING, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

from mcp.server.fastmcp import FastMCP
from ataraxis_transport_layer_pc import list_available_ports

from .communication import MQTTCommunication
from .microcontroller_interface import _evaluate_port

if TYPE_CHECKING:
    from serial.tools.list_ports_common import ListPortInfo

# Initializes the MCP server instance.
mcp = FastMCP(name="ataraxis-communication-interface", json_response=True)


@mcp.tool()
def list_microcontrollers(baudrate: int = 115200) -> str:
    """Discovers all available serial ports and identifies which ones are connected to Arduino or Teensy
    microcontrollers running the ataraxis-micro-controller library.

    Uses parallel processing to simultaneously query all ports for microcontroller identification.

    Args:
        baudrate: The baudrate to use for communication during identification. Note, the same baudrate value is used
            to evaluate all available microcontrollers. The baudrate is only used by microcontrollers that communicate
            via the UART serial interface and is ignored by microcontrollers that use the USB interface.
    """
    # Gets all available serial ports.
    available_ports = list_available_ports()

    # Filters out invalid ports (PID == None) - primarily for Linux systems.
    valid_ports = [port for port in available_ports if port.pid is not None]

    # If there are no valid candidates to evaluate, returns early.
    if not valid_ports:
        return "No valid serial ports detected."

    # Prepares the parallel evaluation tasks.
    port_names = [port.device for port in valid_ports]

    # Uses ProcessPoolExecutor to evaluate all ports in parallel.
    results: dict[str, tuple[ListPortInfo, int, str | None]] = {}

    with ProcessPoolExecutor() as executor:
        # Submits all port evaluation tasks.
        future_to_port = {
            executor.submit(_evaluate_port, port_name, baudrate): (port_name, port_info)
            for port_name, port_info in zip(port_names, valid_ports, strict=True)
        }

        # Collects results as they complete.
        for future in as_completed(future_to_port):
            port_name, port_info = future_to_port[future]
            controller_id, error_msg = future.result()
            results[port_name] = (port_info, controller_id, error_msg)

    # Builds the output string.
    lines: list[str] = [f"Evaluated {len(valid_ports)} serial port(s) at baudrate {baudrate}:"]
    count = 0
    for port_name in port_names:
        if port_name in results:
            port_info, controller_id, error_msg = results[port_name]
            count += 1

            if error_msg is not None:
                # Port encountered a connection error.
                lines.append(f"{count}: {port_info.device} -> {port_info.description} [Connection Failed: {error_msg}]")
            elif controller_id == -1:
                # Port did not respond or is not a valid microcontroller.
                lines.append(f"{count}: {port_info.device} -> {port_info.description} [No microcontroller]")
            else:
                # Port is connected to a valid microcontroller with identified ID.
                lines.append(
                    f"{count}: {port_info.device} -> {port_info.description} [Microcontroller ID: {controller_id}]"
                )

    return "\n".join(lines)


@mcp.tool()
def check_mqtt_broker(host: str = "127.0.0.1", port: int = 1883) -> str:
    """Checks whether an MQTT broker is reachable at the specified host and port.

    Attempts to connect to the MQTT broker and reports the result. Use this tool to verify MQTT broker availability
    before running code that depends on MQTT communication.

    Args:
        host: The IP address or hostname of the MQTT broker.
        port: The socket port used by the MQTT broker.
    """
    # Creates a temporary MQTTCommunication instance to test connectivity.
    mqtt_client = MQTTCommunication(ip=host, port=port)

    # Attempts to connect to the MQTT broker.
    try:
        mqtt_client.connect()
        mqtt_client.disconnect()
    except ConnectionError:
        return (
            f"MQTT broker at {host}:{port} is not reachable. Ensure the broker is running and the host/port "
            f"are correct."
        )
    else:
        return f"MQTT broker at {host}:{port} is reachable."


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output communication
            (recommended for Claude Desktop integration), 'sse' for Server-Sent Events, and 'streamable-http' for
            HTTP-based communication.
    """
    mcp.run(transport=transport)


def run_mcp_server() -> None:
    """Starts the MCP server with stdio transport.

    This function is intended to be used as a CLI entry point. It starts the MCP server using the stdio transport
    protocol, which is the recommended transport for Claude Desktop integration.
    """
    run_server(transport="stdio")
