"""Provides a Model Context Protocol (MCP) server for agentic interaction with the library.

Exposes microcontroller discovery, MQTT broker connectivity checking, and microcontroller data log processing
functionality through the MCP protocol, enabling AI agents to programmatically interact with the library's
core features.
"""

from typing import TYPE_CHECKING, Any, Literal  # pragma: no cover
from pathlib import Path  # pragma: no cover
from threading import Lock, Thread  # pragma: no cover
import contextlib  # pragma: no cover
from dataclasses import field, dataclass  # pragma: no cover
from concurrent.futures import ProcessPoolExecutor, as_completed  # pragma: no cover

import numpy as np  # pragma: no cover
from ataraxis_time import (  # pragma: no cover
    TimeUnits,
    PrecisionTimer,
    TimerPrecisions,
    TimestampFormats,
    TimestampPrecisions,
    convert_time,
    get_timestamp,
)
from mcp.server.fastmcp import FastMCP  # pragma: no cover
from ataraxis_base_utilities import resolve_worker_count  # pragma: no cover
from ataraxis_data_structures import (
    ProcessingStatus,
    ProcessingTracker,
    delete_directory,
    assemble_log_archives,
)  # pragma: no cover
from ataraxis_transport_layer_pc import list_available_ports  # pragma: no cover

from .manifest import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData,
    MicroControllerManifest,
    write_microcontroller_manifest,
)  # pragma: no cover
from .communication import MQTTCommunication  # pragma: no cover
from .log_processing import (
    TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX,
    PARALLEL_PROCESSING_THRESHOLD,
    MICROCONTROLLER_DATA_DIRECTORY,
    execute_job,
    find_log_archive,
    resolve_recording_roots,
    initialize_processing_tracker,
)  # pragma: no cover
from .extraction_config import (
    ExtractionConfig,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    ControllerExtractionConfig,
    create_extraction_config,
)  # pragma: no cover
from .microcontroller_interface import _evaluate_port  # pragma: no cover

if TYPE_CHECKING:
    from serial.tools.list_ports_common import ListPortInfo

# Initializes the MCP server instance.
mcp: FastMCP = FastMCP(name="ataraxis-communication-interface", json_response=True)  # pragma: no cover
"""Stores the MCP server instance used to expose tools to AI agents."""  # pragma: no cover

_WORKER_SCALING_FACTOR: int = 1000  # pragma: no cover
"""Controls the saturation floor via the formula ``ceil(sqrt(messages / factor))``. The square root models diminishing
returns from process parallelism. This value sets the minimum workers a job receives before the budget division can
push it lower. With a factor of 1,000, a 648,000-message archive has a saturation floor of 25
workers."""  # pragma: no cover

_WORKER_MULTIPLE: int = 5  # pragma: no cover
"""Worker counts above 1 are rounded down to the nearest multiple of this value for clean allocation."""

_RESERVED_CORES: int = 2  # pragma: no cover
"""The number of CPU cores reserved for system operations. The worker budget is computed as available cores minus this
value, with a minimum of 1."""


@dataclass(slots=True)  # pragma: no cover
class _PendingJob:  # pragma: no cover
    """Describes a single data extraction job queued for execution."""

    log_directory: Path
    """The path to the DataLogger output directory containing the log archive."""
    output_directory: Path
    """The path to the output directory for this log directory's processed data."""
    tracker_path: Path
    """The path to the ProcessingTracker file that tracks this job."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    source_id: str
    """The source ID string identifying the log archive to process."""
    config_path: Path
    """The path to the ExtractionConfig YAML file for this job's controller."""

    @property  # pragma: no cover
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the tracker
        path with the job ID.
        """
        return str(self.tracker_path), self.job_id


@dataclass(slots=True)  # pragma: no cover
class _ActiveGroup:  # pragma: no cover
    """Tracks a group of jobs executing sequentially with a shared ProcessPoolExecutor."""

    source_id: str
    """The shared source ID for all jobs in this group, or a unique identifier for single-job groups."""
    jobs: list[_PendingJob]
    """The jobs in this group, processed sequentially by the group worker thread."""
    workers: int
    """The number of CPU cores allocated to this group's ProcessPoolExecutor."""
    thread: Thread
    """The background thread executing the group."""


@dataclass(slots=True)  # pragma: no cover
class _JobExecutionState:  # pragma: no cover
    """Tracks runtime state for batch job execution with budget-based worker allocation.

    The execution manager groups pending jobs by source ID so that archives with similar sizes share a single
    ProcessPoolExecutor. Each group is dispatched as one thread that processes its jobs sequentially, reusing the
    pool across all archives in the group. This avoids the overhead of repeatedly spawning and tearing down worker
    processes for archives of the same size.
    """

    all_jobs: dict[tuple[str, str], _PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by (tracker_path, job_id) dispatch key."""
    pending_queue: list[_PendingJob] = field(default_factory=list)
    """Jobs awaiting dispatch."""
    active_groups: list[_ActiveGroup] = field(default_factory=list)
    """Currently executing job groups, each with its own thread and worker allocation."""
    job_message_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    """Maps each dispatch key to its archive message count, probed before execution."""
    worker_budget: int = 1
    """Total CPU cores available for the execution session."""
    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for execution state access."""
    manager_thread: Thread | None = None
    """Background execution manager thread reference."""
    canceled: bool = False
    """Indicates whether the execution session has been canceled."""


_job_execution_state: _JobExecutionState | None = None  # pragma: no cover
"""Stores the active execution state for batch log processing jobs."""  # pragma: no cover


@mcp.tool()  # pragma: no cover
def list_microcontrollers(baudrate: int = 115200) -> str:  # pragma: no cover
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
            controller_id, error_message = future.result()
            results[port_name] = (port_info, controller_id, error_message)

    # Builds the output string.
    lines: list[str] = [f"Evaluated {len(valid_ports)} serial port(s) at baudrate {baudrate}:"]
    count = 0
    for port_name in port_names:
        if port_name in results:
            port_info, controller_id, error_message = results[port_name]
            count += 1

            if error_message is not None:
                # Port encountered a connection error.
                lines.append(
                    f"{count}: {port_info.device} -> {port_info.description} [Connection Failed: {error_message}]"
                )
            elif controller_id == -1:
                # Port did not respond or is not a valid microcontroller.
                lines.append(f"{count}: {port_info.device} -> {port_info.description} [No microcontroller]")
            else:
                # Port is connected to a valid microcontroller with identified ID.
                lines.append(
                    f"{count}: {port_info.device} -> {port_info.description} [Microcontroller ID: {controller_id}]"
                )

    return "\n".join(lines)


@mcp.tool()  # pragma: no cover
def check_mqtt_broker(host: str = "127.0.0.1", port: int = 1883) -> str:  # pragma: no cover
    """Checks whether an MQTT broker is reachable at the specified host and port.

    Attempts to connect to the MQTT broker and reports the result. Use this tool to verify MQTT broker availability
    before running code that depends on MQTT communication.

    Args:
        host: The IP address or hostname of the MQTT broker.
        port: The socket port used by the MQTT broker.

    Returns:
        A message indicating whether the MQTT broker is reachable at the specified host and port.
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


@mcp.tool()  # pragma: no cover
def assemble_log_archives_tool(  # pragma: no cover
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


@mcp.tool()  # pragma: no cover
def read_microcontroller_manifest_tool(manifest_path: str) -> dict[str, Any]:  # pragma: no cover
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
        manifest = MicroControllerManifest.from_yaml(file_path=path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to read manifest: {error}"}

    controllers: list[dict[str, Any]] = []
    for controller in manifest.controllers:
        module_entries = [
            {"module_type": source_module.module_type, "module_id": source_module.module_id, "name": source_module.name}
            for source_module in controller.modules
        ]
        controllers.append({"id": controller.id, "name": controller.name, "modules": module_entries})

    return {"manifest_path": manifest_path, "controllers": controllers, "total_controllers": len(controllers)}


@mcp.tool()  # pragma: no cover
def write_microcontroller_manifest_tool(  # pragma: no cover
    log_directory: str,
    controller_id: int,
    controller_name: str,
    modules: list[dict[str, Any]],
) -> dict[str, Any]:
    """Registers a microcontroller source in the manifest file within a DataLogger output directory.

    If the manifest already exists (another MicroControllerInterface has already registered), appends the new
    controller entry. Otherwise, creates a new manifest. Each module entry must have 'module_type' (int),
    'module_id' (int), and 'name' (str) keys.

    Important:
        The AI agent calling this tool MUST know the controller ID, name, and module details. Do not guess
        these values.

    Args:
        log_directory: The absolute path to the DataLogger output directory where the manifest file is stored.
        controller_id: The controller_id used by the MicroControllerInterface instance.
        controller_name: A colloquial human-readable name for the microcontroller.
        modules: A list of module descriptors, each with 'module_type' (int), 'module_id' (int), and 'name' (str) keys.

    Returns:
        A dictionary containing a 'success' flag, the manifest path, and a summary of the registered entry.
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
                f"Invalid module descriptor: {error}. Each module must have 'module_type', 'module_id', "
                f"and 'name' keys."
            ),
        }

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


@mcp.tool()  # pragma: no cover
def discover_microcontroller_data_tool(root_directory: str) -> dict[str, Any]:  # pragma: no cover
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
                manifest = MicroControllerManifest.from_yaml(file_path=manifest_path)
            except Exception:  # noqa: BLE001, S112
                continue

            if not manifest.controllers:
                continue

            for controller in manifest.controllers:
                archive_path = log_dir / f"{controller.id}{LOG_ARCHIVE_SUFFIX}"
                if not archive_path.exists():
                    # Also check by name as the source ID may be the controller name.
                    archive_path = log_dir / f"{controller.name}{LOG_ARCHIVE_SUFFIX}"
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


@mcp.tool()  # pragma: no cover
def create_extraction_config_tool(manifest_path: str, output_path: str) -> dict[str, Any]:  # pragma: no cover
    """Generate a precursor extraction configuration from a microcontroller manifest.

    Read the manifest file and create an ExtractionConfig with placeholder empty event codes for each
    controller and module. Write the resulting configuration to the specified output path as a YAML file.
    The user must fill in the actual event codes before the configuration is usable for processing.

    Args:
        manifest_path: The absolute path to the microcontroller_manifest.yaml file.
        output_path: The absolute path where the extraction configuration YAML file will be written.

    Returns:
        A dictionary containing a 'success' flag, the config file path, controller count, and a reminder
        message to fill in event codes. Returns an error dictionary if the manifest is missing or invalid.
    """
    path = Path(manifest_path)

    if not path.exists():
        return {"error": f"Manifest file not found: {manifest_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {manifest_path}"}

    try:
        config = create_extraction_config(manifest_path=path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to create extraction config: {error}"}

    output = Path(output_path)

    try:
        config.to_yaml(file_path=output)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to write extraction config: {error}"}

    return {
        "success": True,
        "config_path": output_path,
        "controller_count": len(config.controllers),
        "message": "Precursor config created. Fill in event_codes before processing.",
    }


@mcp.tool()  # pragma: no cover
def read_extraction_config_tool(config_path: str) -> dict[str, Any]:  # pragma: no cover
    """Read an extraction configuration from a YAML file and return its contents.

    Parse the ExtractionConfig file and return a structured dictionary representation of all controller,
    module, and kernel extraction settings.

    Args:
        config_path: The absolute path to the extraction configuration YAML file.

    Returns:
        A dictionary containing the config path, a list of controller entries with their modules and
        kernel settings, and the total controller count. Returns an error dictionary if the file is
        missing or cannot be parsed.
    """
    path = Path(config_path)

    if not path.exists():
        return {"error": f"Config file not found: {config_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {config_path}"}

    try:
        config = ExtractionConfig.from_yaml(file_path=path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to read extraction config: {error}"}

    controllers: list[dict[str, Any]] = []
    for controller in config.controllers:
        module_entries = [
            {
                "module_type": module.module_type,
                "module_id": module.module_id,
                "event_codes": list(module.event_codes),
            }
            for module in controller.modules
        ]

        controller_entry: dict[str, Any] = {
            "controller_id": controller.controller_id,
            "modules": module_entries,
        }

        if controller.kernel is not None:
            controller_entry["kernel"] = {
                "event_codes": list(controller.kernel.event_codes),
            }
        else:
            controller_entry["kernel"] = None

        controllers.append(controller_entry)

    return {
        "config_path": config_path,
        "controllers": controllers,
        "total_controllers": len(controllers),
    }


@mcp.tool()  # pragma: no cover
def write_extraction_config_tool(  # pragma: no cover
    config_path: str, controllers: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write an extraction configuration to a YAML file from structured controller data.

    Accept a list of controller dictionaries, construct an ExtractionConfig instance, and serialize it
    to the specified YAML file path. Each controller dictionary must contain 'controller_id' and 'modules'
    keys. Each module must have 'module_type', 'module_id', and 'event_codes' keys.
    An optional 'kernel' key may contain a dictionary with 'event_codes'.

    Args:
        config_path: The absolute path where the extraction configuration YAML file will be written.
        controllers: A list of controller dictionaries, each with 'controller_id', 'modules' (list of
            dicts with 'module_type', 'module_id', 'event_codes'), and optionally
            'kernel' (dict with 'event_codes').

    Returns:
        A dictionary containing a 'success' flag, the config file path, and the controller count.
        Returns an error dictionary if the input data is invalid or the file cannot be written.
    """
    try:
        controller_configs: list[ControllerExtractionConfig] = []
        for controller_dict in controllers:
            module_configs = tuple(
                ModuleExtractionConfig(
                    module_type=int(module["module_type"]),
                    module_id=int(module["module_id"]),
                    event_codes=tuple(int(code) for code in module["event_codes"]),
                )
                for module in controller_dict["modules"]
            )

            kernel_config = None
            kernel_data = controller_dict.get("kernel")
            if kernel_data is not None:
                kernel_config = KernelExtractionConfig(
                    event_codes=tuple(int(code) for code in kernel_data["event_codes"]),
                )

            controller_configs.append(
                ControllerExtractionConfig(
                    controller_id=int(controller_dict["controller_id"]),
                    modules=module_configs,
                    kernel=kernel_config,
                )
            )

        config = ExtractionConfig(controllers=controller_configs)
    except (KeyError, TypeError, ValueError) as error:
        return {"error": f"Invalid controller data: {error}"}

    output = Path(config_path)

    try:
        config.to_yaml(file_path=output)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to write extraction config: {error}"}

    return {
        "success": True,
        "config_path": config_path,
        "controller_count": len(controller_configs),
    }


@mcp.tool()  # pragma: no cover
def prepare_log_processing_batch_tool(  # pragma: no cover
    log_directories: list[str],
    source_ids: list[str],
    output_directories: list[str],
) -> dict[str, Any]:
    """Prepares an execution manifest for batch log processing without starting execution.

    Accepts log directories, source IDs, and output directories from the caller and initializes a
    ProcessingTracker with one data-extraction job per source ID for each log directory. Idempotent: if a
    tracker already exists for a log directory, returns the existing manifest with current job statuses instead
    of reinitializing. Requires prior discovery -- the caller must provide confirmed source IDs rather than
    relying on implicit archive or manifest discovery.

    Important:
        The AI agent calling this tool MUST run discover_microcontroller_data_tool first to obtain log directory
        paths and confirmed source IDs. The agent MUST ask the user for the output directory paths before calling
        this tool. Do not assume or guess directory paths or source IDs.

    Args:
        log_directories: The list of absolute paths to DataLogger output directories containing log archives.
            Accepts paths from the 'log_directories' list returned by discover_microcontroller_data_tool.
        source_ids: The list of confirmed source IDs to process. Accepts IDs from the 'source_id' field of
            entries in the 'sources' list returned by discover_microcontroller_data_tool. Applied uniformly: each
            log directory creates jobs for every source ID in this list that has a matching archive on disk.
        output_directories: The list of absolute paths for per-log-directory output. Must match the length of
            log_directories. Each output directory receives a ``microcontroller_data/`` subdirectory containing
            the processing tracker and output files.

    Returns:
        A dictionary containing per-log-directory manifests in 'log_directories' with tracker paths and job
        lists, total counts, and any invalid paths.
    """
    if len(output_directories) != len(log_directories):
        return {
            "error": (
                f"Length mismatch: {len(log_directories)} log directories but "
                f"{len(output_directories)} output directories."
            ),
        }

    source_id_set = set(source_ids)
    result_log_dirs: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for entry_index, log_dir_str in enumerate(log_directories):
        log_dir_path = Path(log_dir_str)

        if not log_dir_path.exists() or not log_dir_path.is_dir():
            invalid_paths.append(log_dir_str)
            continue

        # Filters the requested source IDs to those that have a matching archive in this log directory.
        # Discovery already confirmed these archives exist, but the check guards against stale data.
        filtered_ids = sorted(
            source_id for source_id in source_id_set if (log_dir_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}").exists()
        )

        if not filtered_ids:
            result_log_dirs[log_dir_str] = {"source_ids": [], "jobs": [], "tracker_path": None, "summary": {}}
            continue

        # Resolves the output directory for this log directory.
        output_path = Path(output_directories[entry_index])

        # Creates the microcontroller_data subdirectory under the output path for tracker and output files.
        data_path = output_path / MICROCONTROLLER_DATA_DIRECTORY
        data_path.mkdir(parents=True, exist_ok=True)
        tracker_path = data_path / TRACKER_FILENAME

        if tracker_path.exists():
            # Idempotent path: returns existing tracker state.
            try:
                tracker_status = _read_tracker_status(tracker_path=tracker_path)
            except Exception:  # noqa: BLE001
                tracker_status = {"jobs": [], "summary": {}}

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(data_path),
                "source_ids": filtered_ids,
                **tracker_status,
            }
            total_jobs += len(tracker_status.get("jobs", []))
        else:
            # Initializes a new tracker with jobs for the filtered source IDs.
            job_ids = initialize_processing_tracker(output_directory=data_path, source_ids=filtered_ids)

            jobs: list[dict[str, str]] = [
                {
                    "job_id": job_ids[source_id],
                    "source_id": source_id,
                    "status": "SCHEDULED",
                    "log_directory": log_dir_str,
                    "output_directory": str(data_path),
                    "tracker_path": str(tracker_path),
                }
                for source_id in filtered_ids
            ]

            result_log_dirs[log_dir_str] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(data_path),
                "source_ids": filtered_ids,
                "jobs": jobs,
                "summary": {
                    "total": len(jobs),
                    "succeeded": 0,
                    "failed": 0,
                    "running": 0,
                    "scheduled": len(jobs),
                },
            }
            total_jobs += len(jobs)

    result: dict[str, Any] = {
        "success": True,
        "log_directories": result_log_dirs,
        "total_log_directories": len(result_log_dirs),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()  # pragma: no cover
def execute_log_processing_jobs_tool(  # pragma: no cover
    jobs: list[dict[str, str]],
    config_path: str,
    *,
    worker_budget: int = -1,
) -> dict[str, Any]:
    """Dispatches log processing jobs for background execution with budget-based worker allocation.

    Takes job descriptors from the manifest produced by prepare_log_processing_batch_tool and starts a background
    execution manager that allocates CPU cores to each job based on its archive size. The worker budget directly
    controls memory footprint since each worker spawns a separate process. Large archives (>= 2000 messages) receive
    more workers, while small archives receive 1 worker since they process sequentially regardless. The manager fills
    available budget greedily, dispatching smaller jobs alongside large ones when cores are available.

    Important:
        Only one execution session can be active at a time. Use cancel_log_processing_tool to cancel an active
        session before starting a new one.

    Args:
        jobs: The list of job descriptors, each a dictionary with 'log_directory', 'output_directory',
            'tracker_path', 'job_id', 'source_id', and 'config_path' keys.
        config_path: The absolute path to the ExtractionConfig YAML file that specifies which events and
            commands to extract for each controller.
        worker_budget: The total number of CPU cores available for the execution session. Directly controls memory
            footprint. Set to -1 for automatic resolution via resolve_worker_count.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', resolved worker budget, per-job message counts,
        and any invalid jobs.
    """
    global _job_execution_state

    # Enforces single-session constraint.
    if (
        _job_execution_state is not None
        and _job_execution_state.manager_thread is not None
        and _job_execution_state.manager_thread.is_alive()
    ):
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    # Validates and builds pending jobs.
    required_keys = {"log_directory", "output_directory", "tracker_path", "job_id", "source_id", "config_path"}
    pending: list[_PendingJob] = []
    all_jobs: dict[tuple[str, str], _PendingJob] = {}
    invalid_jobs: list[dict[str, str]] = []

    for job_dict in jobs:
        if not required_keys.issubset(job_dict.keys()):
            invalid_jobs.append({**job_dict, "error": f"Missing required keys: {required_keys - job_dict.keys()}"})
            continue

        tracker_path = Path(job_dict["tracker_path"])
        if not tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {job_dict['tracker_path']}"})
            continue

        pending_job = _PendingJob(
            log_directory=Path(job_dict["log_directory"]),
            output_directory=Path(job_dict["output_directory"]),
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            source_id=job_dict["source_id"],
            config_path=Path(job_dict.get("config_path", config_path)),
        )
        pending.append(pending_job)
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves the total worker budget.
    resolved_budget = resolve_worker_count(requested_workers=worker_budget, reserved_cores=_RESERVED_CORES)

    # Probes archive message counts for all pending jobs. This reads only the zip directory of each .npz file,
    # which is fast and does not load message data into memory.
    job_message_counts: dict[tuple[str, str], int] = {}
    for job in pending:
        job_message_counts[job.dispatch_key] = _probe_archive_message_count(job=job)

    # Creates execution state and starts the manager thread.
    _job_execution_state = _JobExecutionState(
        all_jobs=all_jobs,
        pending_queue=pending,
        job_message_counts=job_message_counts,
        worker_budget=resolved_budget,
    )

    manager = Thread(target=_job_execution_manager, daemon=True)
    manager.start()
    _job_execution_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "worker_budget": resolved_budget,
        "job_message_counts": job_message_counts,
    }

    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs

    return result


@mcp.tool()  # pragma: no cover
def get_log_processing_status_tool() -> dict[str, Any]:  # pragma: no cover
    """Returns the current status of the active log processing execution session.

    Reads ProcessingTracker files from disk for each job to report per-job progress. If no execution session
    exists, returns an inactive status.

    Returns:
        A dictionary containing an 'active' flag, per-job status entries in 'jobs', and a 'summary' with counts
        for pending, running, succeeded, and failed jobs.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    state = _job_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    # Reads status from tracker files for each job.
    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for tracker_path, path_jobs in _group_jobs_by_tracker(state=state).items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:  # noqa: BLE001
            job_details.extend(
                {"job_id": job.job_id, "source_id": job.source_id, "status": "UNKNOWN"} for job in path_jobs
            )
            continue

        for job in path_jobs:
            if job.job_id in tracker.jobs:
                job_state = tracker.jobs[job.job_id]
                status = job_state.status

                if status == ProcessingStatus.SUCCEEDED:
                    succeeded_count += 1
                elif status == ProcessingStatus.FAILED:
                    failed_count += 1
                elif status == ProcessingStatus.RUNNING:
                    running_count += 1
                else:
                    scheduled_count += 1

                entry: dict[str, Any] = {"job_id": job.job_id, "source_id": job.source_id, "status": status.name}
                if job_state.error_message is not None:
                    entry["error_message"] = job_state.error_message
                job_details.append(entry)
            else:
                job_details.append({"job_id": job.job_id, "source_id": job.source_id, "status": "UNKNOWN"})

    return {
        "active": manager_alive,
        "canceled": state.canceled,
        "jobs": job_details,
        "summary": {
            "total": len(state.all_jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


@mcp.tool()  # pragma: no cover
def get_log_processing_timing_tool() -> dict[str, Any]:  # pragma: no cover
    """Returns timing information for all jobs in the active execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps from ProcessingTracker.

    Returns:
        A dictionary containing an 'active' flag, per-job timing in 'jobs', and a 'session' summary with
        total elapsed seconds and throughput.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    state = _job_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()
    current_us = int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))

    job_timing: list[dict[str, Any]] = []
    earliest_start: int | None = None
    completed_count = 0
    failed_count = 0

    for tracker_path, path_jobs in _group_jobs_by_tracker(state=state).items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:  # noqa: BLE001, S112
            continue

        for job in path_jobs:
            if job.job_id not in tracker.jobs:
                continue

            job_info = tracker.jobs[job.job_id]
            entry: dict[str, Any] = {"job_id": job.job_id, "source_id": job.source_id}

            if job_info.started_at is not None:
                started_at_us = int(job_info.started_at)
                entry["started_at"] = started_at_us
                if earliest_start is None or started_at_us < earliest_start:
                    earliest_start = started_at_us

            if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
                elapsed_seconds = convert_time(
                    time=current_us - int(job_info.started_at),
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.SECOND,
                    as_float=True,
                )
                entry["elapsed_seconds"] = round(elapsed_seconds, 2)

            if job_info.completed_at is not None:
                entry["completed_at"] = int(job_info.completed_at)
                if job_info.started_at is not None:
                    duration_seconds = convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    )
                    entry["duration_seconds"] = round(duration_seconds, 2)

            if job_info.status == ProcessingStatus.SUCCEEDED:
                completed_count += 1
            elif job_info.status == ProcessingStatus.FAILED:
                failed_count += 1

            job_timing.append(entry)

    # Computes session-level statistics.
    total_elapsed_seconds = 0.0
    if earliest_start is not None:
        total_elapsed_seconds = round(
            convert_time(
                time=current_us - earliest_start,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.SECOND,
                as_float=True,
            ),
            2,
        )

    session: dict[str, Any] = {
        "total_elapsed_seconds": total_elapsed_seconds,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": sum(1 for job_entry in job_timing if "elapsed_seconds" in job_entry),
        "pending_count": len(state.all_jobs)
        - completed_count
        - failed_count
        - sum(1 for job_entry in job_timing if "elapsed_seconds" in job_entry),
    }

    if completed_count > 0 and earliest_start is not None:
        elapsed_hours = convert_time(
            time=current_us - earliest_start,
            from_units=TimeUnits.MICROSECOND,
            to_units=TimeUnits.HOUR,
            as_float=True,
        )
        if elapsed_hours > 0:
            session["throughput_jobs_per_hour"] = round(completed_count / elapsed_hours, 2)

    return {"active": manager_alive, "jobs": job_timing, "session": session}


@mcp.tool()  # pragma: no cover
def cancel_log_processing_tool() -> dict[str, Any]:  # pragma: no cover
    """Cancels the active log processing execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a 'canceled' flag, a 'message', and 'final_state' with counts for succeeded,
        failed, and active jobs at the time of cancellation.
    """
    if _job_execution_state is None:
        return {"canceled": False, "message": "No execution session is active."}

    state = _job_execution_state

    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_queue)
        state.pending_queue.clear()
        active_count = len(state.active_groups)

    # Counts final job statuses from tracker files.
    succeeded = 0
    failed = 0
    tracker_paths: set[Path] = {job.tracker_path for job in state.all_jobs.values()}

    for tracker_path in tracker_paths:
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
            for job_state in tracker.jobs.values():
                if job_state.status == ProcessingStatus.SUCCEEDED:
                    succeeded += 1
                elif job_state.status == ProcessingStatus.FAILED:
                    failed += 1
        except Exception:  # noqa: BLE001, S110
            pass

    return {
        "canceled": True,
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} group(s) still completing.",
        "final_state": {
            "succeeded_jobs": succeeded,
            "failed_jobs": failed,
            "active_jobs_at_cancel": active_count,
        },
    }


@mcp.tool()  # pragma: no cover
def reset_log_processing_jobs_tool(  # pragma: no cover
    tracker_path: str,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resets specific jobs or all jobs in a tracker to scheduled status for re-runs.

    Args:
        tracker_path: The absolute path to the ProcessingTracker YAML file.
        source_ids: An optional list of source IDs whose jobs should be reset. If not provided, all jobs are reset.

    Returns:
        A dictionary containing a 'reset' flag, the number of jobs reset, and updated job statuses.
    """
    path = Path(tracker_path)

    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    try:
        tracker = ProcessingTracker.from_yaml(file_path=path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to read tracker: {error}"}

    # Identifies which job IDs to reset based on source_ids filter.
    target_ids: set[str] = set()
    if source_ids is not None:
        source_id_set = set(source_ids)
        for job_id, job_state in tracker.jobs.items():
            if job_state.specifier in source_id_set:
                target_ids.add(job_id)
    else:
        target_ids = set(tracker.jobs.keys())

    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Collects (job_name, specifier) tuples for the jobs to reset.
    reset_jobs: list[tuple[str, str]] = [
        (tracker.jobs[job_id].job_name, tracker.jobs[job_id].specifier) for job_id in target_ids
    ]

    # Removes target jobs and re-initializes them.
    for job_id in target_ids:
        del tracker.jobs[job_id]
    tracker.to_yaml(file_path=path)

    # Re-initializes the reset jobs.
    reset_tracker = ProcessingTracker(file_path=path)
    reset_tracker.initialize_jobs(jobs=reset_jobs)

    # Reads back the updated state for the response.
    try:
        updated_status = _read_tracker_status(tracker_path=path)
    except Exception:  # noqa: BLE001
        updated_status = {"jobs": [], "summary": {}}

    return {"reset": True, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()  # pragma: no cover
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]:  # pragma: no cover
    """Discovers and summarizes processing status for all log directories under a root directory.

    Recursively searches for microcontroller_processing_tracker.yaml files and aggregates their status. Each tracker
    corresponds to a single DataLogger output directory.

    Args:
        root_directory: The absolute path to the root directory to search for tracker files.

    Returns:
        A dictionary containing per-log-directory status summaries and aggregate counts.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    log_dir_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    for found_tracker_path in sorted(root_path.rglob(TRACKER_FILENAME)):
        log_dir = str(found_tracker_path.parent)
        try:
            status = _read_tracker_status(tracker_path=found_tracker_path)
            summary = status.get("summary", {})

            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            dir_status = _derive_tracker_status(summary=summary)

            log_dir_statuses.append(
                {
                    "log_directory": log_dir,
                    "tracker_path": str(found_tracker_path),
                    "status": dir_status,
                    **status,
                }
            )
        except Exception:  # noqa: BLE001
            log_dir_statuses.append(
                {
                    "log_directory": log_dir,
                    "tracker_path": str(found_tracker_path),
                    "status": "error",
                    "error": "Unable to read tracker file.",
                }
            )

    return {
        "log_directories": log_dir_statuses,
        "total_log_directories": len(log_dir_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
    }


@mcp.tool()  # pragma: no cover
def clean_log_processing_output_tool(output_directories: list[str]) -> dict[str, Any]:  # pragma: no cover
    """Deletes the microcontroller_data subdirectory under one or more output directories.

    Removes each ``microcontroller_data/`` subdirectory and all of its contents, including processed output files
    and the processing tracker. Uses ``delete_directory`` from ataraxis-data-structures for parallel file deletion
    with platform-safe retry logic. After cleanup, the output directories can be passed to
    prepare_log_processing_batch_tool to reinitialize from scratch. Accepts the 'log_directories' list returned
    by discover_microcontroller_data_tool.

    Args:
        output_directories: The list of absolute paths to output directories containing ``microcontroller_data/``
            subdirectories to delete.

    Returns:
        A dictionary containing a 'results' list with per-directory outcomes (each with 'output_directory',
        'cleaned' flag, and either 'data_path' or 'error') and a 'total_cleaned' count.
    """
    results = [_clean_single_output(output_directory=directory) for directory in output_directories]
    total_cleaned = sum(1 for result in results if result.get("cleaned", False))

    return {"results": results, "total_cleaned": total_cleaned, "total_directories": len(results)}


# Placed after all @mcp.tool() definitions so that all tools are registered with the FastMCP instance before the
# server run loop is callable.
def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:  # pragma: no cover
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output communication
            and 'streamable-http' for HTTP-based communication.
    """
    # Delegates to the FastMCP run loop, which blocks until the transport connection is closed. For 'stdio' this
    # means the server runs until the parent process closes stdin; for 'streamable-http' it runs an HTTP server
    # that accepts connections until explicitly terminated.
    mcp.run(transport=transport)


def run_mcp_server() -> None:  # pragma: no cover
    """Starts the MCP server with stdio transport.

    This function is intended to be used as a CLI entry point. It starts the MCP server using the stdio transport
    protocol, which is the recommended transport for Claude Desktop integration.
    """
    run_server(transport="stdio")


def _read_tracker_status(tracker_path: Path) -> dict[str, Any]:  # pragma: no cover
    """Reads a log processing tracker file and returns structured per-job status information.

    Args:
        tracker_path: The path to the ProcessingTracker YAML file.

    Returns:
        A dictionary containing per-job status details and summary counts.
    """
    tracker = ProcessingTracker.from_yaml(file_path=tracker_path)

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for job_id, job_state in tracker.jobs.items():
        source_id = job_state.specifier or job_id[:8]
        status = job_state.status

        if status == ProcessingStatus.SUCCEEDED:
            succeeded_count += 1
        elif status == ProcessingStatus.FAILED:
            failed_count += 1
        elif status == ProcessingStatus.RUNNING:
            running_count += 1
        else:
            scheduled_count += 1

        entry: dict[str, Any] = {"job_id": job_id, "source_id": source_id, "status": status.name}
        if job_state.error_message is not None:
            entry["error_message"] = job_state.error_message
        job_details.append(entry)

    return {
        "jobs": job_details,
        "summary": {
            "total": len(tracker.jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


def _derive_tracker_status(summary: dict[str, Any]) -> str:  # pragma: no cover
    """Derives a high-level processing status label from a tracker summary's job counts.

    Applies a fixed priority: ``failed`` if any job failed, ``completed`` if all succeeded, ``processing`` if any
    are running, ``not_started`` if all are scheduled, and ``in_progress`` otherwise.

    Args:
        summary: A dictionary containing 'total', 'succeeded', 'failed', 'running', and 'scheduled' counts.

    Returns:
        A status string: one of 'failed', 'completed', 'processing', 'not_started', or 'in_progress'.
    """
    total = summary.get("total", 0)
    if summary.get("failed", 0) > 0:
        return "failed"
    if summary.get("succeeded", 0) == total and total > 0:
        return "completed"
    if summary.get("running", 0) > 0:
        return "processing"
    if summary.get("scheduled", 0) == total and total > 0:
        return "not_started"
    return "in_progress"


def _group_jobs_by_tracker(state: _JobExecutionState) -> dict[Path, list[_PendingJob]]:  # pragma: no cover
    """Groups all jobs in an execution state by their tracker file path.

    Minimizes redundant file reads by batching jobs that share the same tracker, so each tracker YAML file is
    deserialized only once when iterating over the groups.

    Args:
        state: The active job execution state containing the job registry.

    Returns:
        A dictionary mapping each tracker path to its list of pending job descriptors.
    """
    tracker_jobs: dict[Path, list[_PendingJob]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)
    return tracker_jobs


def _scan_archive_source_ids(directory: Path) -> list[str]:  # pragma: no cover
    """Scans a directory for assembled log archives and extracts source IDs from their filenames.

    Matches files ending with the log archive suffix and strips the suffix to recover the source ID string. Results
    are returned in sorted order.

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


def _resolve_log_dir_roots(log_dir_paths: list[Path]) -> dict[Path, Path]:  # pragma: no cover
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


def _probe_archive_message_count(job: _PendingJob) -> int:  # pragma: no cover
    """Probes the message count of a job's log archive by reading the .npz zip directory.

    Reconstructs the archive path from the job's log directory and source ID, then reads the file list from the .npz
    archive without loading any message data. The message count is the total entry count minus one (excluding the onset
    message).

    Args:
        job: The pending job descriptor containing the log directory and source ID.

    Returns:
        The number of data messages in the archive, or 0 if the archive cannot be read.
    """
    archive_path = job.log_directory / f"{job.source_id}{LOG_ARCHIVE_SUFFIX}"
    if not archive_path.exists():
        return 0

    try:
        with np.load(file=archive_path, allow_pickle=False) as archive:
            return max(0, len(archive.files) - 1)
    except Exception:  # noqa: BLE001
        return 0


def _compute_sqrt_minimum(message_count: int) -> int:  # pragma: no cover
    """Computes the minimum useful worker count for an archive based on square root scaling.

    The formula ``ceil(sqrt(messages / _WORKER_SCALING_FACTOR))`` models diminishing returns from additional
    workers. The result is snapped to the nearest multiple of ``_WORKER_MULTIPLE`` for clean allocation. Archives
    below the parallel processing threshold always return 1.

    Args:
        message_count: The number of data messages in the job's archive.

    Returns:
        The minimum number of workers that meaningfully benefit this archive size.
    """
    if message_count < PARALLEL_PROCESSING_THRESHOLD:
        return 1

    raw = int(np.ceil(np.sqrt(message_count / _WORKER_SCALING_FACTOR)))
    if raw <= 1:
        return 1

    return max(_WORKER_MULTIPLE, round(raw / _WORKER_MULTIPLE) * _WORKER_MULTIPLE)


def _clean_single_output(output_directory: str) -> dict[str, Any]:  # pragma: no cover
    """Deletes the microcontroller_data subdirectory under a single output directory.

    Args:
        output_directory: The absolute path to the output directory.

    Returns:
        A dictionary containing 'output_directory', 'cleaned' flag, and either 'data_path' or 'error' keys.
    """
    output_path = Path(output_directory)

    if not output_path.exists():
        return {"output_directory": output_directory, "cleaned": False, "error": "Directory does not exist."}

    if not output_path.is_dir():
        return {"output_directory": output_directory, "cleaned": False, "error": "Path is not a directory."}

    data_path = output_path / MICROCONTROLLER_DATA_DIRECTORY

    if not data_path.exists():
        return {"output_directory": output_directory, "cleaned": True, "message": "Nothing to clean."}

    try:
        delete_directory(directory_path=data_path)
    except Exception as error:  # noqa: BLE001
        return {
            "output_directory": output_directory,
            "cleaned": False,
            "data_path": str(data_path),
            "error": f"Unable to delete: {error}",
        }

    return {"output_directory": output_directory, "cleaned": True, "data_path": str(data_path)}


def _group_worker(jobs: list[_PendingJob], workers: int, state: _JobExecutionState) -> None:  # pragma: no cover
    """Executes a group of jobs sequentially using a shared ProcessPoolExecutor.

    Creates one ProcessPoolExecutor for the entire group and processes each job in sequence, reusing the pool
    across all archives. This avoids the overhead of spawning and tearing down worker processes for each individual
    archive. Checks for cancellation between jobs to allow responsive shutdown. If a job's tracker is not updated
    to a terminal state, marks it as failed.

    Args:
        jobs: The list of pending job descriptors to process sequentially.
        workers: The number of CPU cores allocated to this group's ProcessPoolExecutor.
        state: The execution state, checked for cancellation between jobs.
    """
    shared_executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        for job in jobs:
            # Checks for cancellation between jobs so the group stops promptly.
            if state.canceled:
                break

            tracker = ProcessingTracker(file_path=job.tracker_path)

            # Loads the extraction config and finds the matching controller configuration.
            try:
                config = ExtractionConfig.from_yaml(file_path=job.config_path)
                controller_configs = {str(c.controller_id): c for c in config.controllers}
                controller_config = controller_configs.get(job.source_id)
            except Exception:  # noqa: BLE001
                tracker.fail_job(
                    job_id=job.job_id, error_message=f"Unable to load extraction config from '{job.config_path}'."
                )
                continue

            if controller_config is None:
                tracker.fail_job(job_id=job.job_id, error_message=f"No controller config for source '{job.source_id}'.")
                continue

            # execute_job already calls tracker.fail_job on exception, so the tracker state is updated.
            with contextlib.suppress(Exception):
                log_path = find_log_archive(log_directory=job.log_directory, source_id=job.source_id)
                execute_job(
                    log_path=log_path,
                    output_directory=job.output_directory,
                    source_id=job.source_id,
                    job_id=job.job_id,
                    workers=workers,
                    tracker=tracker,
                    controller_config=controller_config,
                    display_progress=False,
                    executor=shared_executor,
                )

            # Failsafe: if the tracker was not updated to a terminal state, marks the job as failed.
            try:
                reloaded = ProcessingTracker.from_yaml(file_path=job.tracker_path)
                if job.job_id in reloaded.jobs:
                    status = reloaded.jobs[job.job_id].status
                    if status not in (ProcessingStatus.SUCCEEDED, ProcessingStatus.FAILED):
                        tracker.fail_job(
                            job_id=job.job_id, error_message="Job terminated without updating tracker status."
                        )
            except Exception:  # noqa: BLE001, S110
                pass
    finally:
        if shared_executor is not None:
            shared_executor.shutdown(wait=True)


def _job_execution_manager() -> None:  # pragma: no cover
    """Dispatches queued jobs as worker-tier groups with shared process pools.

    Runs as a daemon thread, polling at 1-second intervals. Each dispatch cycle classifies pending jobs into
    small (< 2000 messages, 1 worker each) and parallel (>= 2000 messages). Parallel jobs are grouped by their
    precomputed worker tier from ``_compute_sqrt_minimum``, which snaps archive sizes to discrete worker counts
    (multiples of 5). Jobs in the same tier share a single ProcessPoolExecutor sized exactly to that tier.
    Each tier is split into as many concurrent groups as the budget allows. Small jobs are dispatched individually.
    Exits when the queue is empty and no active groups remain.
    """
    if _job_execution_state is None:
        return

    state = _job_execution_state
    poll_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    while True:
        with state.lock:
            # Removes completed groups and frees their budget.
            state.active_groups = [group for group in state.active_groups if group.thread.is_alive()]

            # Exits when no pending jobs and no active groups remain.
            if not state.pending_queue and not state.active_groups:
                break

            # Stops dispatching new groups if canceled. Waits for active groups to finish.
            if state.canceled:
                if not state.active_groups:
                    break
            else:
                available = state.worker_budget - sum(group.workers for group in state.active_groups)
                if available < 1:
                    poll_timer.delay(delay=1, allow_sleep=True)
                    continue

                # Classifies pending jobs into small (sequential) and parallel.
                small_pending: list[_PendingJob] = []
                parallel_pending: list[_PendingJob] = []
                for job in state.pending_queue:
                    message_count = state.job_message_counts.get(job.dispatch_key, 0)
                    if message_count < PARALLEL_PROCESSING_THRESHOLD:
                        small_pending.append(job)
                    else:
                        parallel_pending.append(job)

                dispatch_groups: list[tuple[list[_PendingJob], int]] = []

                # Phase 1: Groups parallel jobs by worker tier. Each job's optimal worker count is precomputed
                # via _compute_sqrt_minimum, which snaps archive sizes to discrete tiers (multiples of 5). Jobs
                # in the same tier share a ProcessPoolExecutor sized exactly to that tier. Each tier is split
                # into as many concurrent groups as the available budget allows.
                if parallel_pending and available >= _WORKER_MULTIPLE:
                    worker_tiers: dict[int, list[_PendingJob]] = {}
                    for job in parallel_pending:
                        tier = _compute_sqrt_minimum(message_count=state.job_message_counts.get(job.dispatch_key, 0))
                        worker_tiers.setdefault(tier, []).append(job)

                    # Dispatches tiers from largest to smallest so large archives get budget priority.
                    for tier_workers in sorted(worker_tiers, reverse=True):
                        if available < tier_workers:
                            continue

                        tier_jobs = worker_tiers[tier_workers]
                        max_concurrent = available // tier_workers
                        concurrent = min(max_concurrent, len(tier_jobs))

                        # Splits tier jobs evenly across concurrent groups via chunking.
                        chunk_size = -(-len(tier_jobs) // concurrent)  # Ceiling division.
                        for start in range(0, len(tier_jobs), chunk_size):
                            chunk = tier_jobs[start : start + chunk_size]
                            dispatch_groups.append((chunk, tier_workers))
                            available -= tier_workers

                # Phase 2: Fills remaining budget with small jobs (1 worker each, dispatched individually).
                for job in small_pending:
                    if available < 1:
                        break
                    dispatch_groups.append(([job], 1))
                    available -= 1

                # Dispatches all groups.
                for group_jobs, group_workers in dispatch_groups:
                    for job in group_jobs:
                        state.pending_queue.remove(job)

                    thread = Thread(
                        target=_group_worker,
                        kwargs={"jobs": group_jobs, "workers": group_workers, "state": state},
                        daemon=True,
                    )
                    thread.start()
                    state.active_groups.append(
                        _ActiveGroup(
                            source_id=group_jobs[0].source_id,
                            jobs=group_jobs,
                            workers=group_workers,
                            thread=thread,
                        )
                    )

        # Polls at 1-second intervals outside the lock to avoid blocking other threads.
        poll_timer.delay(delay=1, allow_sleep=True)
