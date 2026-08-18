"""Provides MCP tools for discovering microcontrollers, checking MQTT brokers, assembling log archives, managing
microcontroller manifests, and discovering confirmed microcontroller recordings.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    resolve_unique_roots,
    assemble_log_archives,
    discover_log_archives,
    discover_marker_files,
)

from .responses import (
    page_fields,
    project_item,
    resolve_page,
    item_breakdown,
    reject_unknown,
    resolve_detail_limit,
)
from .mcp_instance import mcp
from ..communication import MQTTCommunication
from ..microcontroller import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData,
    MicroControllerManifest,
    discover_microcontrollers,
    write_microcontroller_manifest,
)

_SOURCE_AXES: tuple[str, ...] = ("source_id", "name")
"""The source keys a caller filters the recording listing by, which a bare call reports the counts of."""

_SOURCE_SEMI_DETAIL_FIELDS: tuple[str, ...] = ("recording_root", "source_id", "name", "log_directory")
"""The fields every listed source carries."""

_SOURCE_DETAIL_FIELDS: tuple[str, ...] = ("log_archive", "modules")
"""The fields a listed source carries once detail is requested. One entry per registered module makes the module list
the term that grows a whole-project listing fastest, so it is withheld until a caller asks for it."""


@mcp.tool()
def list_microcontrollers_tool(baudrate: int = 115200) -> str:
    """Discovers all available serial ports and identifies which ones are connected to Arduino or Teensy
    microcontrollers running the ataraxis-micro-controller library.

    Queries the ports in parallel across a pool sized to the smaller of the port count and the host's worker budget.

    Args:
        baudrate: The baudrate to use for communication during identification. Note, the same baudrate value is used
            to evaluate all available microcontrollers. The baudrate is only used by microcontrollers that communicate
            via the UART serial interface and is ignored by microcontrollers that use the USB interface.

    Returns:
        A numbered list of evaluated serial ports with their device descriptions, each entry reporting the identified
        microcontroller ID, that no microcontroller responded, or the connection error encountered on that port, or a
        message indicating no valid ports were detected.
    """
    controllers = discover_microcontrollers(baudrate=baudrate)

    if not controllers:
        return "No valid serial ports detected."

    lines: list[str] = [f"Evaluated {len(controllers)} serial port(s) at baudrate {baudrate}:"]
    for count, controller in enumerate(controllers, start=1):
        if controller.error_message is not None:
            lines.append(
                f"{count}: {controller.port} -> {controller.description} "
                f"[Connection Failed: {controller.error_message}]"
            )
        elif controller.controller_id is None:
            lines.append(f"{count}: {controller.port} -> {controller.description} [No microcontroller]")
        else:
            lines.append(
                f"{count}: {controller.port} -> {controller.description} "
                f"[Microcontroller ID: {controller.controller_id}]"
            )

    return "\n".join(lines)


@mcp.tool()
def check_mqtt_broker_tool(host: str = "127.0.0.1", port: int = 1883) -> str:
    """Checks whether an MQTT broker is reachable at the specified host and port.

    Use this tool to verify MQTT broker availability before running code that depends on MQTT communication.

    Args:
        host: The IP address or hostname of the MQTT broker.
        port: The socket port used by the MQTT broker.

    Returns:
        A message indicating whether the MQTT broker is reachable at the specified host and port, or naming the
        address the client rejected before any connection was attempted.
    """
    try:
        mqtt_client = MQTTCommunication(ip=host, port=port)
        mqtt_client.connect()
        mqtt_client.disconnect()
    except ValueError as error:
        # An address the client rejects never reaches a socket, so it is reported as the caller error it is rather
        # than as an unreachable broker, which would send the caller looking for a broker to start.
        return f"Unable to check an MQTT broker at {host}:{port}. {error}"
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

    Runs ahead of the log processing pipeline, which reads the assembled archives to extract microcontroller data.
    Serves a log directory from an earlier session, and a directory whose automatic assembly was skipped or failed.

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
        A dictionary containing the assembly status, directory path, the list of archive filenames present in the
        directory after assembly, extracted source IDs, and archive count. Returns an error dictionary if the
        directory does not exist or assembly fails.
    """
    directory_path = Path(log_directory)

    if not directory_path.exists():
        return {"error": f"Directory not found: {log_directory}"}

    if not directory_path.is_dir():
        return {"error": f"Not a directory: {log_directory}"}

    try:
        assemble_log_archives(
            log_directory=directory_path,
            remove_sources=remove_sources,
            verify_integrity=verify_integrity,
            verbose=False,
        )
    except Exception as error:
        return {"error": f"Archive assembly failed: {error}"}

    source_ids = sorted(discover_log_archives(log_directory=directory_path))
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
        total controller count. Returns an error dictionary if the manifest file is missing, is not a file, or
        cannot be parsed.
    """
    path = Path(manifest_path)

    if not path.exists():
        return {"error": f"Manifest file not found: {manifest_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {manifest_path}"}

    try:
        manifest = MicroControllerManifest.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Unable to read manifest: {error}"}

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

    If the manifest already exists (another MicroControllerInterface has already registered), replaces the entry
    registered under the same controller_id or appends a new one when the manifest carries none. Otherwise, creates a
    new manifest.

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
        A dictionary containing a 'success' flag, the manifest path, and a summary of the registered entry. Returns
        an error dictionary if the log directory is missing, is not a directory, a module descriptor is malformed, or
        the manifest cannot be written.
    """
    log_path = Path(log_directory)

    if not log_path.exists():
        return {"error": f"Directory does not exist: {log_directory}"}

    if not log_path.is_dir():
        return {"error": f"Path is not a directory: {log_directory}"}

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

    try:
        write_microcontroller_manifest(
            log_directory=log_path,
            controller_id=controller_id,
            controller_name=controller_name,
            modules=module_entries,
        )
    except Exception as error:
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
def discover_microcontroller_data_tool(
    root_directory: str,
    source_ids: list[str] | None = None,
    name: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Discovers confirmed microcontroller recordings under a root directory, in three widening stages.

    Recursively searches for microcontroller_manifest.yaml files to identify controller sources. Only sources whose
    log archives (``{source_id}_log.npz``) exist on disk are included.

    A bare call reports the counts and the flat log directory list a batch is prepared from, alongside a ``breakdown``
    naming every controller id and name the scan found. Naming a filter adds a page of sources carrying their
    identity and their directories. Opting into detail adds each source's archive path and its module list, which is
    what makes a whole-project listing large.

    The counts, the breakdown, and the log directory list span every confirmed source regardless of the filters, so
    narrowing what is listed never distorts what is reported.

    Args:
        root_directory: The absolute path to the root directory to search. Searched recursively.
        source_ids: Restricts the listing to these controller ids.
        name: Restricts the listing to one controller name.
        limit: The sources to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match, which is how a caller reading under a tight filter takes the whole result at once.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list sources when no filter is named.
        detailed: Determines whether the listed sources report their archive path and module list.

    Returns:
        A dictionary carrying 'log_directories' for batch processing, 'total_sources', 'total_log_directories', and a
        'breakdown' per axis. Adds a 'sources' list alongside top-level 'rows', 'matched_rows', 'start_row', and
        'next_start_row' paging fields whenever a filter is named or the listing is requested. A scan confirming no
        source returns an empty 'sources' list and an empty 'breakdown' whatever the filters name. Returns an error
        dictionary if the root directory is missing, is not a directory, cannot be searched, or a filter names a value
        the scan found no source for.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    # A manifest can register a controller whose archive was never written, so only sources with an archive are kept.
    confirmed_sources: list[tuple[Path, int, str, Path, list[dict[str, Any]]]] = []
    log_directories_with_archives: set[Path] = set()

    try:
        manifest_paths = discover_marker_files(directory=root_path, marker_name=MICROCONTROLLER_MANIFEST_FILENAME)
    except OSError as error:
        return {"error": f"Unable to search '{root_directory}': {error}"}

    for manifest_path in manifest_paths:
        log_directory = manifest_path.parent

        try:
            manifest = MicroControllerManifest.from_yaml(file_path=manifest_path)
            # Resolves every archive the logger wrote beside the manifest in one flat scan, instead of probing the
            # filesystem once per registered controller.
            archives = discover_log_archives(log_directory=log_directory)
        except Exception:  # noqa: S112 - a manifest that cannot be read contributes no sources, so the scan skips it.
            continue

        # Collapses a repeated controller id, since one controller addresses one archive and one tracker entry. A
        # manifest written before the writer replaced a re-registered entry can still hold several rows for one id,
        # and every id-keyed consumer of this tool keeps one of them, so reporting each row would disagree with them.
        for controller in {controller.id: controller for controller in manifest.controllers}.values():
            archive_path = archives.get(str(controller.id))
            if archive_path is None:
                continue

            module_entries = [
                {
                    "module_type": source_module.module_type,
                    "module_id": source_module.module_id,
                    "name": source_module.name,
                }
                for source_module in controller.modules
            ]
            confirmed_sources.append((log_directory, controller.id, controller.name, archive_path, module_entries))
            log_directories_with_archives.add(log_directory)

    if not confirmed_sources:
        return {
            "sources": [],
            "log_directories": [],
            "total_sources": 0,
            "total_log_directories": 0,
            "breakdown": {},
        }

    log_directory_paths = sorted(log_directories_with_archives)
    log_directory_to_root = _resolve_log_directory_roots(log_directory_paths=log_directory_paths)

    sources_output: list[dict[str, Any]] = [
        {
            "recording_root": str(log_directory_to_root[log_directory]),
            "source_id": str(source_id),
            "name": controller_name,
            "log_archive": str(archive_path),
            "log_directory": str(log_directory),
            "modules": module_entries,
        }
        for log_directory, source_id, controller_name, archive_path, module_entries in confirmed_sources
    ]

    response: dict[str, Any] = {
        "log_directories": sorted(str(log_directory) for log_directory in log_directory_paths),
        "total_sources": len(sources_output),
        "total_log_directories": len(log_directory_paths),
        "breakdown": item_breakdown(items=sources_output, axes=_SOURCE_AXES),
    }

    if source_ids is None and name is None and not include_items:
        return response

    matched = sources_output
    if source_ids is not None:
        rejection = reject_unknown(items=sources_output, key="source_id", values=source_ids, subject="source")
        if rejection is not None:
            return rejection
        matched = [source for source in matched if source["source_id"] in source_ids]
    if name is not None:
        rejection = reject_unknown(items=sources_output, key="name", values=[name], subject="source")
        if rejection is not None:
            return rejection
        matched = [source for source in matched if source["name"] == name]

    fields = (*_SOURCE_SEMI_DETAIL_FIELDS, *_SOURCE_DETAIL_FIELDS) if detailed else _SOURCE_SEMI_DETAIL_FIELDS
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["sources"] = [project_item(item=source, fields=fields) for source in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


def _resolve_log_directory_roots(log_directory_paths: list[Path]) -> dict[Path, Path]:
    """Resolves each log directory to its recording root.

    Uses unique path component detection to identify recording session boundaries. Falls back to using each
    log directory's parent when unique component detection fails (e.g., several directories sharing every component).

    Args:
        log_directory_paths: The sorted list of log directory paths to resolve.

    Returns:
        A mapping from each log directory to its recording root path.
    """
    try:
        recording_roots = resolve_unique_roots(paths=log_directory_paths)
    except ValueError:
        recording_roots = tuple(dict.fromkeys(log_directory.parent for log_directory in log_directory_paths))

    log_directory_to_root: dict[Path, Path] = {}
    for log_directory in log_directory_paths:
        for root in recording_roots:
            if log_directory == root or root in log_directory.parents:
                log_directory_to_root[log_directory] = root
                break
        else:
            log_directory_to_root[log_directory] = log_directory.parent

    return log_directory_to_root
