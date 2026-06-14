"""Provides MCP tools for discovering microcontrollers, checking MQTT brokers, assembling log archives, managing
microcontroller manifests, and discovering confirmed microcontroller recordings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from ataraxis_data_structures import assemble_log_archives
from ataraxis_transport_layer_pc import list_available_ports

from .mcp_instance import mcp
from ..communication import MQTTCommunication
from ..microcontroller.interface import evaluate_port
from ..microcontroller.dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData,
    MicroControllerManifest,
    write_microcontroller_manifest,
)
from ..microcontroller.log_processing import LOG_ARCHIVE_SUFFIX, resolve_recording_roots

if TYPE_CHECKING:
    from serial.tools.list_ports_common import ListPortInfo

_UNIDENTIFIED_CONTROLLER_ID: int = -1
"""The sentinel value returned by evaluate_port when a serial port is not connected to a recognized microcontroller."""


@mcp.tool()
def list_microcontrollers_tool(baudrate: int = 115200) -> str:
    """Discovers all available serial ports and identifies which ones are connected to Arduino or Teensy
    microcontrollers running the ataraxis-micro-controller library.

    Uses parallel processing to simultaneously query all ports for microcontroller identification.

    Args:
        baudrate: The baudrate to use for communication during identification. Note, the same baudrate value is used
            to evaluate all available microcontrollers. The baudrate is only used by microcontrollers that communicate
            via the UART serial interface and is ignored by microcontrollers that use the USB interface.

    Returns:
        A numbered list of evaluated serial ports with their device descriptions and identified microcontroller IDs,
        or a message indicating no valid ports were detected.
    """
    available_ports = list_available_ports()

    # Filters out invalid ports (PID is None) — primarily for Linux systems.
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
            executor.submit(evaluate_port, port_name, baudrate): (port_name, port_info)
            for port_name, port_info in zip(port_names, valid_ports, strict=True)
        }

        for future in as_completed(future_to_port):
            port_name, port_info = future_to_port[future]
            controller_id, error_message = future.result()
            results[port_name] = (port_info, controller_id, error_message)

    lines: list[str] = [f"Evaluated {len(valid_ports)} serial port(s) at baudrate {baudrate}:"]
    count = 0
    for port_name in port_names:
        if port_name in results:
            port_info, controller_id, error_message = results[port_name]
            count += 1

            if error_message is not None:
                # Reports the connection error for this port.
                lines.append(
                    f"{count}: {port_info.device} -> {port_info.description} [Connection Failed: {error_message}]"
                )
            elif controller_id == _UNIDENTIFIED_CONTROLLER_ID:
                # Reports unrecognized ports that did not respond or lack a valid microcontroller.
                lines.append(f"{count}: {port_info.device} -> {port_info.description} [No microcontroller]")
            else:
                # Reports identified microcontrollers with their controller ID.
                lines.append(
                    f"{count}: {port_info.device} -> {port_info.description} [Microcontroller ID: {controller_id}]"
                )

    return "\n".join(lines)


@mcp.tool()
def check_mqtt_broker_tool(host: str = "127.0.0.1", port: int = 1883) -> str:
    """Checks whether an MQTT broker is reachable at the specified host and port.

    Attempts to connect to the MQTT broker and reports the result. Use this tool to verify MQTT broker availability
    before running code that depends on MQTT communication.

    Args:
        host: The IP address or hostname of the MQTT broker.
        port: The socket port used by the MQTT broker.

    Returns:
        A message indicating whether the MQTT broker is reachable at the specified host and port.
    """
    mqtt_client = MQTTCommunication(ip=host, port=port)

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


@mcp.tool()
def assemble_log_archives_tool(
    log_directory: str,
    *,
    remove_sources: bool = True,
    verify_integrity: bool = False,
) -> dict[str, Any]:
    """Consolidates raw .npy log entries in a DataLogger output directory into .npz archives by source ID.

    Assembles the raw .npy files produced by a DataLogger instance into consolidated .npz archives, one per unique
    source ID. This is required before the log processing pipeline can extract microcontroller data.

    This tool is useful when log archives need to be assembled independently of a runtime stop operation,
    for example when processing log directories from previous sessions or when the automatic assembly was skipped or
    failed.

    Important:
        The AI agent calling this tool MUST ask the user to provide the log_directory path before calling this
        tool. Do not assume or guess the log directory path.

    Args:
        log_directory: The absolute path to the DataLogger output directory containing raw .npy log entries. Must
            be provided by the user.
        remove_sources: Determines whether to remove the original .npy files after successful archive assembly.
        verify_integrity: Determines whether to verify archive integrity against original log entries before
            removing sources.

    Returns:
        A dictionary containing the assembly status, directory path, list of created archive filenames, extracted
        source IDs, and archive count. Returns an error dictionary if the directory does not exist or assembly
        fails.
    """
    directory_path = Path(log_directory)

    if not directory_path.exists():
        return {"error": f"Directory not found: {log_directory}"}

    if not directory_path.is_dir():
        return {"error": f"Not a directory: {log_directory}"}

    # Consolidates raw .npy log entries into .npz archives grouped by source ID.
    try:
        assemble_log_archives(
            log_directory=directory_path,
            remove_sources=remove_sources,
            verify_integrity=verify_integrity,
            verbose=False,
        )
    except Exception as error:  # noqa: BLE001
        return {"error": f"Archive assembly failed: {error}"}

    # Scans for created archives and extracts source IDs from filenames.
    source_ids = _scan_archive_source_ids(directory=directory_path)
    archives = [f"{source_id}{LOG_ARCHIVE_SUFFIX}" for source_id in source_ids]

    return {
        "status": "assembled",
        "directory": log_directory,
        "archives": archives,
        "source_ids": source_ids,
        "archive_count": len(archives),
    }


@mcp.tool()
def read_microcontroller_manifest_tool(manifest_path: str) -> dict[str, Any]:
    """Reads a microcontroller manifest file and returns its contents.

    The manifest identifies which MicroControllerInterface instances logged data to a DataLogger output directory
    and enumerates the hardware modules managed by each controller.

    Args:
        manifest_path: The absolute path to the microcontroller_manifest.yaml file.

    Returns:
        A dictionary containing the manifest path, a list of controller entries with their modules, and the
        total controller count.
    """
    path = Path(manifest_path)

    if not path.exists():
        return {"error": f"Manifest file not found: {manifest_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {manifest_path}"}

    try:
        manifest = MicroControllerManifest.load(file_path=path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to read manifest: {error}"}

    # Serializes each controller and its modules into a dictionary representation.
    controllers: list[dict[str, Any]] = []
    for controller in manifest.controllers:
        module_entries = [
            {"module_type": source_module.module_type, "module_id": source_module.module_id, "name": source_module.name}
            for source_module in controller.modules
        ]
        controllers.append({"id": controller.id, "name": controller.name, "modules": module_entries})

    return {"manifest_path": manifest_path, "controllers": controllers, "total_controllers": len(controllers)}


@mcp.tool()
def write_microcontroller_manifest_tool(
    log_directory: str,
    controller_id: int,
    controller_name: str,
    modules: list[dict[str, Any]],
) -> dict[str, Any]:
    """Registers a microcontroller source in the manifest file within a DataLogger output directory.

    If the manifest already exists (another MicroControllerInterface has already registered), appends the new
    controller entry. Otherwise, creates a new manifest. Each module entry must have 'module_type' (int type code),
    'module_id' (int ID code), and 'name' (str) keys.

    Important:
        The AI agent calling this tool MUST know the controller ID, name, and module details. Do not guess
        these values.

    Args:
        log_directory: The absolute path to the DataLogger output directory where the manifest file is stored.
        controller_id: The controller_id used by the MicroControllerInterface instance.
        controller_name: A colloquial human-readable name for the microcontroller.
        modules: A list of module descriptors, each with 'module_type' (int type code), 'module_id' (int ID code),
            and 'name' (str) keys.

    Returns:
        A dictionary containing a 'success' flag, the manifest path, and a summary of the registered entry.
    """
    log_path = Path(log_directory)

    if not log_path.exists():
        return {"error": f"Directory does not exist: {log_directory}"}

    if not log_path.is_dir():
        return {"error": f"Path is not a directory: {log_directory}"}

    # Converts the raw module dictionaries into typed ModuleSourceData instances.
    try:
        module_entries = tuple(
            ModuleSourceData(
                module_type=int(module["module_type"]),
                module_id=int(module["module_id"]),
                name=str(module["name"]),
            )
            for module in modules
        )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "error": (
                f"Invalid module descriptor: {error}. Each module must have 'module_type' (type code), "
                f"'module_id' (ID code), and 'name' keys."
            ),
        }

    # Writes or appends the controller entry to the manifest file.
    try:
        write_microcontroller_manifest(
            log_directory=log_path,
            controller_id=controller_id,
            controller_name=controller_name,
            modules=module_entries,
        )
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to write manifest: {error}"}

    manifest_path = log_path / MICROCONTROLLER_MANIFEST_FILENAME
    return {
        "success": True,
        "manifest_path": str(manifest_path),
        "controller_id": controller_id,
        "controller_name": controller_name,
        "module_count": len(module_entries),
    }


@mcp.tool()
def discover_microcontroller_data_tool(root_directory: str) -> dict[str, Any]:
    """Discovers confirmed microcontroller recordings under a root directory.

    Recursively searches for microcontroller_manifest.yaml files to identify controller sources. Only sources whose
    log archives (``{source_id}_log.npz``) exist on disk are included. For each confirmed source, returns the
    controller metadata and associated hardware modules from the manifest.

    Args:
        root_directory: The absolute path to the root directory to search. Searched recursively.

    Returns:
        A dictionary containing a 'sources' list where each entry has 'recording_root', 'source_id', 'name',
        'log_archive', 'log_directory', and 'modules' keys, a flat 'log_directories' list for batch processing,
        and aggregate counts.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    # Discovers all microcontroller manifests and collects only sources whose log archives exist on disk.
    confirmed_sources: list[tuple[Path, int, str, Path, list[dict[str, Any]]]] = []
    log_dirs_with_archives: set[Path] = set()

    try:
        for manifest_path in sorted(root_path.rglob(MICROCONTROLLER_MANIFEST_FILENAME)):
            log_dir = manifest_path.parent

            try:
                manifest = MicroControllerManifest.load(file_path=manifest_path)
            except Exception:  # noqa: BLE001, S112
                continue

            if not manifest.controllers:
                continue

            for controller in manifest.controllers:
                archive_path = log_dir / f"{controller.id}{LOG_ARCHIVE_SUFFIX}"
                if not archive_path.exists():
                    continue

                module_entries = [
                    {
                        "module_type": source_module.module_type,
                        "module_id": source_module.module_id,
                        "name": source_module.name,
                    }
                    for source_module in controller.modules
                ]
                confirmed_sources.append((log_dir, controller.id, controller.name, archive_path, module_entries))
                log_dirs_with_archives.add(log_dir)
    except PermissionError as error:
        return {"error": f"Permission denied during search: {error}"}

    if not confirmed_sources:
        return {
            "sources": [],
            "log_directories": [],
            "total_sources": 0,
            "total_log_directories": 0,
        }

    # Resolves recording roots and builds the log-directory-to-root mapping.
    log_dir_paths = sorted(log_dirs_with_archives)
    log_dir_to_root = _resolve_log_dir_roots(log_dir_paths=log_dir_paths)

    # Builds the flat list of resolved source entries.
    sources_output: list[dict[str, Any]] = []
    for log_dir, source_id, name, archive_path, module_entries in confirmed_sources:
        sources_output.append(
            {
                "recording_root": str(log_dir_to_root[log_dir]),
                "source_id": str(source_id),
                "name": name,
                "log_archive": str(archive_path),
                "log_directory": str(log_dir),
                "modules": module_entries,
            }
        )

    return {
        "sources": sources_output,
        "log_directories": sorted(str(log_dir) for log_dir in log_dir_paths),
        "total_sources": len(sources_output),
        "total_log_directories": len(log_dir_paths),
    }


def _scan_archive_source_ids(directory: Path) -> list[str]:
    """Scans a directory for assembled log archives and extracts source IDs from their filenames.

    Matches files ending with the log archive suffix and strips the suffix to recover the source ID string. Returns
    results in sorted order.

    Args:
        directory: The directory to scan for log archives.

    Returns:
        A sorted list of source ID strings extracted from archive filenames.
    """
    return sorted(
        source_id
        for archive_path in directory.glob(f"*{LOG_ARCHIVE_SUFFIX}")
        if (source_id := archive_path.name.removesuffix(LOG_ARCHIVE_SUFFIX))
    )


def _resolve_log_dir_roots(log_dir_paths: list[Path]) -> dict[Path, Path]:
    """Resolves each log directory to its recording root.

    Uses unique path component detection to identify recording session boundaries. Falls back to using each
    log directory's parent when unique component detection fails (e.g., single log directory).

    Args:
        log_dir_paths: The sorted list of log directory paths to resolve.

    Returns:
        A mapping from each log directory to its recording root path.
    """
    try:
        recording_roots = resolve_recording_roots(paths=log_dir_paths)
    except RuntimeError:
        recording_roots = tuple(dict.fromkeys(log_dir.parent for log_dir in log_dir_paths))

    log_dir_to_root: dict[Path, Path] = {}
    for log_dir in log_dir_paths:
        for root in recording_roots:
            if log_dir == root or root in log_dir.parents:
                log_dir_to_root[log_dir] = root
                break
        else:
            log_dir_to_root[log_dir] = log_dir.parent

    return log_dir_to_root
