"""Provides the Command Line Interface (CLI) installed into the Python environment together with the library."""

from typing import TYPE_CHECKING, Literal  # pragma: no cover
from pathlib import Path  # pragma: no cover
from concurrent.futures import ProcessPoolExecutor, as_completed  # pragma: no cover

if TYPE_CHECKING:  # pragma: no cover
    from serial.tools.list_ports_common import ListPortInfo  # pragma: no cover

import click  # pragma: no cover
from ataraxis_base_utilities import LogLevel, console  # pragma: no cover
from ataraxis_transport_layer_pc import list_available_ports  # pragma: no cover

from .mcp_server import run_server as run_mcp  # pragma: no cover
from .dataclasses import ExtractionConfig, create_extraction_config  # pragma: no cover
from .communication import MQTTCommunication  # pragma: no cover
from .log_processing import run_log_processing_pipeline  # pragma: no cover
from .microcontroller_interface import _evaluate_port  # pragma: no cover

# Enables console output.
console.enable()  # pragma: no cover

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}  # pragma: no cover
"""Ensures that displayed Click help messages are formatted according to the lab standard."""  # pragma: no cover


@click.group("axci", context_settings=CONTEXT_SETTINGS)
def axci_cli() -> None:  # pragma: no cover
    """Serves as the entry-point for interfacing with all interactive components of the
    ataraxis-communication-interface (AXCI) library.
    """


@axci_cli.command("id")
@click.option(
    "-b",
    "--baudrate",
    type=int,
    default=115200,
    show_default=True,
    help="The baudrate to use for communication during identification. Only used by microcontrollers that communicate "
    "via the UART serial interface; ignored by microcontrollers that use the USB interface.",
)
def identify(baudrate: int) -> None:  # pragma: no cover
    """Discovers all connected Arduino or Teensy microcontrollers running the ataraxis-micro-controller library.

    Use this command to identify the hardware available to the local host-machine.
    """
    # Gets all available serial ports.
    available_ports = list_available_ports()

    # Filters out invalid ports (PID == None) - primarily for Linux systems.
    valid_ports = [port for port in available_ports if port.pid is not None]

    # If there are no valid candidates to evaluate, aborts the runtime early.
    if not valid_ports:
        console.echo("No valid serial ports detected.")
        return

    console.echo(f"Evaluating {len(valid_ports)} serial port(s) at baudrate {baudrate}...")

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
            controller_id, error_message = future.result()
            results[port_name] = (port_info, controller_id, error_message)

    # Prints the results in the original port order.
    count = 0
    for port_name in port_names:
        if port_name in results:
            port_info, controller_id, error_message = results[port_name]
            count += 1

            if error_message is not None:
                # Port encountered a connection error.
                console.echo(
                    f"{count}: {port_info.device} -> {port_info.description} [Connection Failed: {error_message}]"
                )
            elif controller_id == -1:
                # Port did not respond or is not a valid microcontroller.
                console.echo(f"{count}: {port_info.device} -> {port_info.description} [No microcontroller]")
            else:
                # Port is connected to a valid microcontroller with identified ID.
                console.echo(
                    f"{count}: {port_info.device} -> {port_info.description} [Microcontroller ID: {controller_id}]"
                )


@axci_cli.command("mqtt")
@click.option(
    "-h",
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="The IP address or hostname of the MQTT broker.",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=1883,
    show_default=True,
    help="The socket port used by the MQTT broker.",
)
def check_mqtt(host: str, port: int) -> None:  # pragma: no cover
    """Checks whether an MQTT broker is reachable at the specified host and port.

    Attempts to connect to the MQTT broker and reports the result. Use this command to verify MQTT broker
    availability before running code that depends on MQTT communication.
    """
    console.echo(f"Checking MQTT broker connectivity at {host}:{port}...")

    # Creates a temporary MQTTCommunication instance to test connectivity.
    mqtt_client = MQTTCommunication(ip=host, port=port)

    # Attempts to connect to the MQTT broker.
    try:
        mqtt_client.connect()
        console.echo(f"MQTT broker at {host}:{port} is reachable.", level=LogLevel.SUCCESS)
        mqtt_client.disconnect()
    except ConnectionError:
        console.echo(
            f"MQTT broker at {host}:{port} is not reachable. Ensure the broker is running and the "
            f"host/port are correct.",
            level=LogLevel.ERROR,
        )


@axci_cli.group("config")
def config_group() -> None:  # pragma: no cover
    """Manages extraction configuration files for the log processing pipeline."""


@config_group.command("create")
@click.option(
    "-m",
    "--manifest-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="The path to the microcontroller_manifest.yaml file to generate the config from.",
)
@click.option(
    "-o",
    "--output-path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="The path to the output .yaml file where to save the generated configuration data.",
)
def config_create(manifest_path: Path, output_path: Path) -> None:  # pragma: no cover
    """Generates a precursor extraction configuration from a microcontroller manifest.

    Creates an extraction_config.yaml with all controllers and modules populated from the manifest,
    but with empty event codes that must be filled in before processing. Edit the generated file to
    specify the event codes for each module and kernel entry.
    """
    config = create_extraction_config(manifest_path=manifest_path)
    config.save(file_path=output_path)
    console.echo(
        message=f"Extraction config written to {output_path}. Fill in event_codes before processing.",
        level=LogLevel.SUCCESS,
    )


@config_group.command("show")
@click.option(
    "-c",
    "--config-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="The path to the extraction configuration .yaml file to display.",
)
def config_show(config_path: Path) -> None:  # pragma: no cover
    """Displays the contents of an extraction configuration file.

    Reads the specified .yaml file and prints each controller's modules, event codes, command codes,
    and kernel settings.
    """
    config = ExtractionConfig.load(file_path=config_path)

    console.echo(message=f"Extraction config: {config_path}", level=LogLevel.INFO)
    for controller in config.controllers:
        console.echo(message=f"  Controller ID: {controller.controller_id}")
        for module in controller.modules:
            console.echo(
                message=f"    Module ({module.module_type}, {module.module_id}): events={list(module.event_codes)}"
            )
        if controller.kernel is not None:
            console.echo(message=f"    Kernel: events={list(controller.kernel.event_codes)}")
        else:
            console.echo(message="    Kernel: not configured")


@axci_cli.command("process")
@click.option(
    "-ld",
    "--log-directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    help="The path to the root directory to search for .npz log archives. Typically this is the root directory of the "
    "processed recording session.",
)
@click.option(
    "-od",
    "--output-directory",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="The path to the directory where processed output files are written. Created automatically if it "
    "does not exist. All processed data is saved under microcontroller_data subdirectory created under this target "
    "output directory.",
)
@click.option(
    "-c",
    "--config",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path),
    help="The path to the .yaml file specifying which controllers, modules, and events to extract.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help="The unique hexadecimal identifier for this processing job. If provided, runs only the matching "
    "job (remote mode).",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of worker processes to use. Set to -1 (default) to use all available CPU cores.",
)
@click.option(
    "-p",
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Determines whether to display progress bars during data extraction.",
)
def process(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None,
    *,
    workers: int,
    progress: bool,
) -> None:  # pragma: no cover
    """Processes MicroControllerInterface log archives to extract hardware module and kernel message data.

    Extracts data as specified by the extraction configuration and writes the results to feather (IPC) files.
    Controller IDs in the extraction config determine which archives are processed. Requires an
    extraction_config.yaml file -- use 'axci config create' to generate one from a manifest.
    """
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
    )


@axci_cli.command("mcp")
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    show_default=True,
    help="The transport protocol to use for MCP communication. Use 'stdio' for standard input/output communication "
    "(default, recommended for Claude Desktop integration) or 'streamable-http' for HTTP-based communication.",
)
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None:  # pragma: no cover
    """Starts the Model Context Protocol (MCP) server for agentic interaction with the library.

    The MCP server exposes microcontroller discovery, MQTT connectivity checking, and data processing
    functionality through the MCP protocol, enabling AI agents to programmatically interact with the library.
    """
    console.echo(message=f"Starting AXCI MCP server with {transport} transport...", level=LogLevel.INFO)
    run_mcp(transport=transport)
