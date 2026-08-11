"""Provides the Command Line Interface (CLI) installed into the Python environment together with the library."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

if TYPE_CHECKING:
    from serial.tools.list_ports_common import ListPortInfo

import click
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import limit_worker_threads, initialize_worker_threads
from ataraxis_transport_layer_pc import list_available_ports

from .mcp_server import run_server as run_mcp
from ..communication import MQTTCommunication
from ..orchestration import run_log_processing_pipeline
from ..microcontroller import ExtractionConfig, evaluate_port, create_extraction_config

console.enable()

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""

_WORKER_THREAD_CEILING: int = 1
"""The number of threads each port evaluation worker pins its numeric backends to. A worker spends its runtime waiting
on one serial port, so the backends it imports repay no pool wider than a single thread."""


@click.group("axci", context_settings=_CONTEXT_SETTINGS)
def axci_cli() -> None:
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
    "via the UART serial interface, and ignored by microcontrollers that use the USB interface.",
)
def identify(baudrate: int) -> None:
    """Discovers all connected Arduino or Teensy microcontrollers running the ataraxis-micro-controller library.

    Use this command to identify the hardware available to the local host-machine.
    """
    available_ports = list_available_ports()

    # Filters out invalid ports (PID == None) - primarily for Linux systems.
    valid_ports = [port for port in available_ports if port.pid is not None]

    if not valid_ports:
        console.echo(message="No valid serial ports detected.")
        return

    console.echo(message=f"Evaluating {len(valid_ports)} serial port(s) at baudrate {baudrate}...")

    port_names = [port.device for port in valid_ports]

    results: dict[str, tuple[ListPortInfo, int, str | None]] = {}

    # Pins every worker from both sides for the pool's whole lifetime. The environment limit reaches the backends that
    # size their pool while importing, which a worker does after it is spawned, and the initializer reaches the ones
    # that read their width the first time they are asked to do work.
    with (
        limit_worker_threads(thread_count=_WORKER_THREAD_CEILING),
        ProcessPoolExecutor(
            max_workers=min(len(valid_ports), resolve_worker_count()),
            initializer=initialize_worker_threads,
            initargs=(_WORKER_THREAD_CEILING,),
        ) as executor,
    ):
        future_to_port = {
            executor.submit(evaluate_port, port=port_name, baudrate=baudrate): (port_name, port_info)
            for port_name, port_info in zip(port_names, valid_ports, strict=True)
        }

        for future in as_completed(future_to_port):
            port_name, port_info = future_to_port[future]
            controller_id, error_message = future.result()
            results[port_name] = (port_info, controller_id, error_message)

    count = 0
    for port_name in port_names:
        if port_name in results:
            port_info, controller_id, error_message = results[port_name]
            count += 1

            if error_message is not None:
                # Port encountered a connection error.
                console.echo(
                    message=f"{count}: {port_info.device} -> {port_info.description} "
                    f"[Connection Failed: {error_message}]"
                )
            elif controller_id == -1:
                # Port did not respond or is not a valid microcontroller.
                console.echo(message=f"{count}: {port_info.device} -> {port_info.description} [No microcontroller]")
            else:
                # Port is connected to a valid microcontroller with identified ID.
                console.echo(
                    message=f"{count}: {port_info.device} -> {port_info.description} "
                    f"[Microcontroller ID: {controller_id}]"
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
def check_mqtt(host: str, port: int) -> None:
    """Checks whether an MQTT broker is reachable at the specified host and port.

    Attempts to connect to the MQTT broker and reports the result. Use this command to verify MQTT broker
    availability before running code that depends on MQTT communication.
    """
    console.echo(message=f"Checking MQTT broker connectivity at {host}:{port}...")

    mqtt_client = MQTTCommunication(ip=host, port=port)

    try:
        mqtt_client.connect()
        console.echo(message=f"MQTT broker at {host}:{port} is reachable.", level=LogLevel.SUCCESS)
        mqtt_client.disconnect()
    except ConnectionError:
        console.echo(
            message=f"MQTT broker at {host}:{port} is not reachable. Ensure the broker is running and the "
            f"host/port are correct.",
            level=LogLevel.ERROR,
        )


@axci_cli.group("config")
def config_group() -> None:
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
def config_create(manifest_path: Path, output_path: Path) -> None:
    """Generates a precursor extraction configuration from a microcontroller manifest.

    Writes the configuration to the requested output path with all controllers and modules populated from the
    manifest, but with empty event codes that must be filled in before processing. Edit the generated file to
    specify the event codes for each module entry. Kernel extraction is left unconfigured, so a user who wants
    kernel messages adds a kernel entry with its own event codes.
    """
    config = create_extraction_config(manifest_path=manifest_path)
    config.to_yaml(file_path=output_path)
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
def config_show(config_path: Path) -> None:
    """Displays the contents of an extraction configuration file.

    Reads the specified .yaml file and prints each controller's modules, event codes, and kernel settings.
    """
    config = ExtractionConfig.from_yaml(file_path=config_path)

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
    help="The canonical hexadecimal identifier of the single job to run. If provided, runs only the matching job, "
    "which is the target an external scheduler names when it dispatches one unit of work.",
)
@click.option(
    "-s",
    "--specifier",
    type=str,
    multiple=True,
    help="Controller ID to process. Repeat to specify multiple IDs. If not provided, processes every controller the "
    "extraction config declares. Ignored when a job ID selects the work.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help="The ceiling on the worker processes any single job receives. Set to -1 (default) to resolve the ceiling "
    "from every available CPU core minus the cores reserved for the host system. The resolved ceiling is capped at "
    "the declared per-job allocation of 8 cores.",
)
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    default=False,
    show_default=True,
    help="Determines whether to suppress the progress bars during data extraction. The progress bars are displayed by "
    "default.",
)
def process(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None,
    specifier: tuple[str, ...],
    *,
    workers: int,
    no_progress: bool,
) -> None:
    """Processes MicroControllerInterface log archives to extract hardware module and kernel message data.

    Extracts data as specified by the extraction configuration and writes the results to feather (IPC) files.
    Targets a single recording and runs its archives one at a time. Controller IDs in the extraction config
    determine which archives are processed. Passing a job ID runs that single job alone, which is how an external
    scheduler dispatches one unit of work. Requires an extraction configuration .yaml file, which 'axci config
    create' generates from a manifest. Use the MCP server to orchestrate batches spanning many recordings.
    """
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config,
        job_id=job_id,
        source_ids=list(specifier) if specifier else None,
        workers=workers,
        display_progress=not no_progress,
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
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None:
    """Starts the Model Context Protocol (MCP) server for agentic interaction with the library.

    The MCP server exposes microcontroller discovery, MQTT connectivity checking, log archive assembly, recording
    discovery, microcontroller manifest management, extraction configuration management, log data processing, output
    verification, output cleanup, and event querying through the MCP protocol. The exposed tools enable AI agents to
    programmatically interact with the library.
    """
    # The stdio transport carries the JSON-RPC message stream over stdout, which is also where the console writes
    # every message up to the WARNING level. Silencing the console keeps library output out of that stream, as a
    # single logged line renders the message it interleaves with unparsable for the connected client.
    if transport == "stdio":
        console.disable()
    else:
        console.echo(message=f"Starting AXCI MCP server with {transport} transport...", level=LogLevel.INFO)

    run_mcp(transport=transport)
