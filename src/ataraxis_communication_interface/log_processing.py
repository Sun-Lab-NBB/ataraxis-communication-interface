"""Provides the log data processing pipeline for extracting hardware module and kernel message data from
MicroControllerInterface log archives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import LogArchiveReader, ProcessingTracker

from .dataclasses import MICROCONTROLLER_MANIFEST_FILENAME, MicroControllerManifest, ExtractionConfig, ControllerExtractionConfig
from .communication import SerialProtocols, SerialPrototypes

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""Naming convention suffix for log archives produced by assemble_log_archives()."""

TRACKER_FILENAME: str = "microcontroller_processing_tracker.yaml"
"""Filename for the processing tracker file placed in the output directory."""

MICROCONTROLLER_DATA_DIRECTORY: str = "microcontroller_data"
"""Name of the subdirectory created under the output path for microcontroller data processing results. All tracker
files and processed outputs are written into this subdirectory."""

PARALLEL_PROCESSING_THRESHOLD: int = 2000
"""The minimum number of messages in a log archive required to enable parallel processing. Archives with fewer messages
are processed sequentially to avoid multiprocessing overhead."""

CONTROLLER_FEATHER_PREFIX: str = "controller_"
"""Filename prefix for controller feather output files."""

MODULE_FEATHER_INFIX: str = "_module_"
"""Filename infix separating the controller source ID from the module type and ID."""

KERNEL_FEATHER_INFIX: str = "_kernel"
"""Filename infix identifying kernel message feather files."""

FEATHER_SUFFIX: str = ".feather"
"""Filename suffix for all feather (IPC) output files."""

_EXTRACTION_JOB_NAME: str = "microcontroller_data_extraction"
"""The job name used by the processing pipeline for microcontroller data extraction."""


@dataclass(slots=True)
class ExtractedMessageData:
    """Stores the data parsed from a single incoming message received by the PC from the microcontroller during
    runtime.
    """

    timestamp: np.uint64
    """The number of microseconds elapsed since the UTC epoch onset when the message was received by the PC."""
    command: np.uint8
    """The code of the command that the module or kernel was executing when it sent the message to the PC."""
    event: np.uint8
    """The event code of the message."""
    prototype: np.uint8 | None
    """The SerialPrototypes code identifying the data layout, or None for state-only messages."""
    data: None | np.number | NDArray[np.number]
    """The parsed data object transmitted with the message (or None, for state-only messages)."""


@dataclass(slots=True)
class ExtractedModuleData:
    """Stores the data parsed from all messages sent to the PC by a hardware module instance during runtime that
    matched the caller's event code filter.
    """

    module_type: int
    """The type (family) code of the hardware module instance whose data is stored in the 'event_data' attribute."""
    module_id: int
    """The unique identifier code of the hardware module instance whose data is stored in the 'event_data' attribute."""
    # noinspection PyTypeHints
    event_data: dict[np.uint8, tuple[ExtractedMessageData, ...]]
    """A dictionary that uses message event codes as keys and tuples of ExtractedMessageData instances as values."""


type _BatchResult = tuple[
    dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]],
    list[ExtractedMessageData],
]
"""Describes the return type of _process_message_batch: a module data dictionary mapping (type, id) tuples to
event-grouped message lists, and a chronological list of kernel messages."""


def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    config: ExtractionConfig | Path,
    job_id: str | None = None,
    log_ids: list[str] | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None:
    """Processes the requested MicroControllerInterface log archives from a single DataLogger output directory.

    Extracts hardware module and kernel message data as specified by the extraction configuration and writes the
    results to feather (IPC) files. Supports both local and remote processing modes. In local mode (job_id is None),
    resolves each requested log archive by source ID, initializes a processing tracker in the output directory, and
    executes all requested jobs sequentially. In remote mode (job_id is provided), generates all possible job IDs for
    the requested source IDs and executes only the matching job.

    All resolved archives must reside in the same directory. If the log_directory contains archives from multiple
    DataLogger instances (in separate subdirectories), each must be processed independently. Use the MCP batch
    processing tools to orchestrate multi-directory workflows.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``microcontroller_data/`` subdirectory is created
            automatically under this path, and all tracker and output files are written there.
        config: The extraction configuration specifying which controllers, modules, events, and commands to extract.
            Accepts an ExtractionConfig instance or a Path to an extraction_config.yaml file.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all requested jobs are run sequentially
            with automatic tracker management (local mode).
        log_ids: A list of source log IDs to process. Each ID must correspond to exactly one archive under the
            log directory, and all archives must reside in the same parent directory. If not provided, reads the
            microcontroller_manifest.yaml file from the log directory to resolve all registered source IDs.
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores. Setting this to 1 conducts processing sequentially.
        display_progress: Determines whether to display progress bars during extraction. Defaults to True for
            interactive CLI use. Set to False for MCP batch processing.

    Raises:
        FileNotFoundError: If the log_directory does not exist, a requested log ID has no matching archive, or no
            microcontroller manifest is found when log_ids is not provided.
        ValueError: If the provided job_id does not match any discoverable job, if no source IDs can be resolved,
            if a requested log ID matches multiple archives, if resolved archives span multiple directories, or if
            the extraction config has empty event codes.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = f"Unable to process logs in '{log_directory}'. The path does not exist or is not a directory."
        console.error(message=message, error=FileNotFoundError)

    # Resolves the extraction configuration from a file path if needed.
    if isinstance(config, Path):
        if not config.exists() or not config.is_file():
            message = f"Unable to load extraction config from '{config}'. The path does not exist or is not a file."
            console.error(message=message, error=FileNotFoundError)
        resolved_config = ExtractionConfig.load(file_path=config)
    else:
        resolved_config = config

    # Builds a lookup from controller ID to its extraction configuration.
    controller_configs: dict[str, ControllerExtractionConfig] = {
        str(controller.controller_id): controller for controller in resolved_config.controllers
    }

    # Locates the microcontroller manifest to resolve or validate source IDs. The manifest ensures only
    # axci-produced log archives are processed, preventing accidental processing of logs from other libraries.
    candidates = sorted(log_directory.rglob(MICROCONTROLLER_MANIFEST_FILENAME))
    if not candidates:
        message = (
            f"Unable to process logs in '{log_directory}'. No {MICROCONTROLLER_MANIFEST_FILENAME} was found. "
            f"A microcontroller manifest is required to identify which log archives were produced by "
            f"ataraxis-communication-interface."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest_path = candidates[0]
    manifest = MicroControllerManifest.load(file_path=manifest_path)
    manifest_ids = {str(controller.id) for controller in manifest.controllers} | {
        controller.name for controller in manifest.controllers
    }

    if not manifest_ids:
        message = (
            f"Unable to process logs in '{log_directory}'. The {MICROCONTROLLER_MANIFEST_FILENAME} at "
            f"'{manifest_path}' contains no controller entries."
        )
        console.error(message=message, error=ValueError)

    # Resolves source IDs from the manifest when none are explicitly provided. When IDs are provided,
    # validates them against the manifest to prevent processing non-microcontroller logs.
    if log_ids is None or not log_ids:
        log_ids = sorted(manifest_ids)
        console.echo(message=f"Resolved {len(log_ids)} source ID(s) from manifest: {', '.join(log_ids)}")
    else:
        invalid_ids = [source_id for source_id in log_ids if source_id not in manifest_ids]
        if invalid_ids:
            message = (
                f"Unable to process logs in '{log_directory}'. The following source IDs are not registered "
                f"in the {MICROCONTROLLER_MANIFEST_FILENAME}: {', '.join(invalid_ids)}. Registered source IDs: "
                f"{', '.join(sorted(manifest_ids))}."
            )
            console.error(message=message, error=ValueError)

    source_ids = sorted(log_ids)

    # Validates that each source ID has a matching controller extraction configuration.
    for source_id in source_ids:
        if source_id not in controller_configs:
            message = (
                f"Unable to process logs in '{log_directory}'. Source ID '{source_id}' has no matching controller "
                f"in the extraction config. Configured controller IDs: {sorted(controller_configs.keys())}."
            )
            console.error(message=message, error=ValueError)

    # Resolves all archive paths upfront and validates they belong to the same DataLogger output directory.
    archive_paths = {
        source_id: find_log_archive(log_directory=log_directory, source_id=source_id) for source_id in source_ids
    }
    parent_directories = {path.parent for path in archive_paths.values()}
    if len(parent_directories) > 1:
        message = (
            f"Unable to process logs in '{log_directory}'. The requested log archives span multiple directories: "
            f"{sorted(str(parent) for parent in parent_directories)}. Each DataLogger output directory must be "
            f"processed independently."
        )
        console.error(message=message, error=ValueError)

    # Creates the microcontroller_data subdirectory under the output path.
    data_path = output_directory / MICROCONTROLLER_DATA_DIRECTORY
    data_path.mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(file_path=data_path / TRACKER_FILENAME)

    if job_id is not None:
        # Generates all possible job IDs and executes only the one matching the provided job_id (remote mode).
        all_job_ids = _generate_job_ids(source_ids=source_ids)
        id_to_source: dict[str, str] = {v: k for k, v in all_job_ids.items()}

        if job_id not in id_to_source:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for the provided log IDs. Valid job IDs: "
                f"{list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        source_id = id_to_source[job_id]
        execute_job(
            log_path=archive_paths[source_id],
            output_directory=data_path,
            source_id=source_id,
            job_id=job_id,
            workers=workers,
            tracker=tracker,
            controller_config=controller_configs[source_id],
            display_progress=display_progress,
        )
    else:
        # Initializes the tracker and runs all requested jobs sequentially (local mode). Resolves workers once and
        # creates a shared ProcessPoolExecutor to reuse across all jobs, avoiding repeated process pool creation.
        console.echo(message=f"Initializing processing tracker for {len(source_ids)} job(s)...")
        job_ids = initialize_processing_tracker(output_directory=data_path, source_ids=source_ids)

        resolved_workers = resolve_worker_count(requested_workers=workers)
        shared_executor = ProcessPoolExecutor(max_workers=resolved_workers) if resolved_workers > 1 else None

        try:
            for source_id in source_ids:
                execute_job(
                    log_path=archive_paths[source_id],
                    output_directory=data_path,
                    source_id=source_id,
                    job_id=job_ids[source_id],
                    workers=resolved_workers,
                    tracker=tracker,
                    controller_config=controller_configs[source_id],
                    display_progress=display_progress,
                    executor=shared_executor,
                )
        finally:
            if shared_executor is not None:
                shared_executor.shutdown(wait=True)

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)


def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    controller_config: ControllerExtractionConfig,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> None:
    """Executes a single data extraction job for the target log archive.

    Extracts hardware module and kernel data from the log archive in a single pass as specified by the controller
    extraction configuration, then writes the results to feather (IPC) files in the output directory. Each module
    produces a separate feather file, and kernel messages (when configured) produce an additional feather file.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where feather output files and the tracker are written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        controller_config: The extraction configuration for the controller whose archive is being processed.
        display_progress: Determines whether to display a progress bar during extraction.
        executor: An optional pre-created ProcessPoolExecutor to reuse for parallel processing.

    Raises:
        ValueError: If the controller config has modules with empty event codes.
    """
    # Converts the controller config into the parameters expected by extract_log_data.
    module_type_id: tuple[tuple[int, int], ...] | None = None
    module_event_codes: frozenset[int] | None = None
    kernel_event_codes: frozenset[int] | None = None

    if controller_config.modules:
        module_type_id = tuple((module.module_type, module.module_id) for module in controller_config.modules)
        # Collects all event codes across all modules into a single frozenset for extract_log_data.
        all_module_events: set[int] = set()
        for module in controller_config.modules:
            if not module.event_codes:
                message = (
                    f"Unable to execute the data extraction job for source '{source_id}'. Module "
                    f"({module.module_type}, {module.module_id}) has empty event_codes."
                )
                console.error(message=message, error=ValueError)
            all_module_events.update(module.event_codes)
        module_event_codes = frozenset(all_module_events)

    if controller_config.kernel is not None:
        if not controller_config.kernel.event_codes:
            message = (
                f"Unable to execute the data extraction job for source '{source_id}'. Kernel extraction "
                f"has empty event_codes."
            )
            console.error(message=message, error=ValueError)
        kernel_event_codes = frozenset(controller_config.kernel.event_codes)

    # Validates that at least one extraction target is configured.
    if module_type_id is None and kernel_event_codes is None:
        message = (
            f"Unable to execute the data extraction job for source '{source_id}'. The controller config "
            f"has no modules and no kernel extraction configured."
        )
        console.error(message=message, error=ValueError)

    console.echo(message=f"Running '{_EXTRACTION_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Extracts both module and kernel data in a single pass through the archive.
        module_data, kernel_messages = extract_log_data(
            log_path=log_path,
            module_type_id=module_type_id,
            module_event_codes=module_event_codes,
            kernel_event_codes=kernel_event_codes,
            n_workers=workers,
            display_progress=display_progress,
            executor=executor,
        )

        # Writes each extracted module's data to a separate feather file.
        for extracted_module in module_data:
            _write_module_feather(
                module_data=extracted_module,
                source_id=source_id,
                output_directory=output_directory,
            )

        # Writes kernel messages to a feather file when kernel extraction was configured.
        if kernel_messages:
            _write_kernel_feather(
                kernel_messages=kernel_messages,
                source_id=source_id,
                output_directory=output_directory,
            )

        tracker.complete_job(job_id=job_id)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def extract_log_data(
    log_path: Path,
    module_type_id: tuple[tuple[int, int], ...] | None = None,
    module_event_codes: frozenset[int] | None = None,
    kernel_event_codes: frozenset[int] | None = None,
    n_workers: int = -1,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[tuple[ExtractedModuleData, ...], tuple[ExtractedMessageData, ...]]:
    """Extracts hardware module and kernel message data from a MicroControllerInterface .npz log archive in a single
    pass.

    Reads the archive once and routes each incoming message (MODULE_DATA, MODULE_STATE, KERNEL_DATA, KERNEL_STATE)
    through event code filters. Only messages whose event codes match the caller's filter are extracted. Event codes
    are guaranteed to be unique within each module and the kernel, so command code filtering is not required.

    Notes:
        At this time, the function exclusively works with incoming messages sent by the microcontroller to the PC.

        If the target .npz archive contains fewer than 2000 messages, the processing is carried out sequentially
        regardless of the specified worker-count.

        When an external executor is provided, batch processing is submitted to that executor instead of creating a
        new ProcessPoolExecutor. The caller is responsible for executor lifecycle management.

    Args:
        log_path: The path to the .npz archive file that stores the logged data generated by the
            MicroControllerInterface instance during runtime.
        module_type_id: A tuple of tuples, where each inner tuple stores the type and ID codes of a specific hardware
            module whose data should be extracted, e.g.: ``((3, 1), (4, 2))``. Required when module_event_codes is
            provided. Set to None to skip module extraction.
        module_event_codes: The event codes to extract for hardware module messages. Required when module_type_id is
            provided. Only messages with matching event codes are extracted.
        kernel_event_codes: The event codes to extract for kernel messages. Set to None to skip kernel extraction.
        n_workers: The number of parallel worker processes (CPU cores) to use for processing. Setting this to a value
            below 1 uses all available CPU cores. Setting this to a value of 1 conducts the processing sequentially.
        display_progress: Determines whether to display a progress bar during parallel batch processing.
        executor: An optional pre-created ProcessPoolExecutor to use for parallel batch processing. When provided,
            the function submits work to this executor instead of creating its own. The caller must ensure the
            executor's worker count matches the n_workers value used for batch generation.

    Returns:
        A tuple of two elements: (module_data, kernel_messages). module_data is a tuple of ExtractedModuleData
        instances (one per requested module that had matching messages). kernel_messages is a tuple of
        ExtractedMessageData instances in chronological order. Either component can be empty.

    Raises:
        ValueError: If the log archive does not exist, if module_type_id is provided without module_event_codes
            (or vice versa), or if no extraction is requested.
    """
    # Validates parameter consistency.
    has_module = module_type_id is not None or module_event_codes is not None
    has_kernel = kernel_event_codes is not None

    if module_type_id is not None and module_event_codes is None:
        message = (
            "Unable to extract log data. The module_type_id parameter requires module_event_codes to also be specified."
        )
        console.error(message=message, error=ValueError)
    if module_event_codes is not None and module_type_id is None:
        message = (
            "Unable to extract log data. The module_event_codes parameter requires module_type_id to also be specified."
        )
        console.error(message=message, error=ValueError)
    if not has_module and not has_kernel:
        message = "Unable to extract log data. At least one of module or kernel extraction must be requested."
        console.error(message=message, error=ValueError)

    # Validates the log archive path.
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        message = (
            f"Unable to extract data from the log file {log_path}, as it does not exist or does "
            f"not point to a valid .npz archive."
        )
        console.error(message=message, error=ValueError)

    # Uses LogArchiveReader to handle archive access, onset timestamp discovery, and batch creation.
    reader = LogArchiveReader(archive_path=log_path)

    # Returns empty results if there are no messages to process.
    if reader.message_count == 0:
        return (), ()

    onset_us = reader.onset_timestamp_us

    # Processes small archives sequentially to avoid the unnecessary overhead of setting up the multiprocessing
    # runtime. Also applies when the user explicitly requests a single worker process.
    if n_workers == 1 or reader.message_count < PARALLEL_PROCESSING_THRESHOLD:
        all_keys = reader.get_batches(workers=1, batch_multiplier=1)
        module_batch, kernel_batch = _process_message_batch(
            log_path=log_path,
            file_names=all_keys[0],
            onset_us=onset_us,
            module_type_id=module_type_id,
            module_event_codes=module_event_codes,
            kernel_event_codes=kernel_event_codes,
        )

        # Converts the lists of ExtractedMessageData instances to tuples for the final data structure.
        module_event_data = {
            module: {event: tuple(message_list) for event, message_list in event_dict.items()}
            for module, event_dict in module_batch.items()
        }

        module_results = tuple(
            ExtractedModuleData(
                module_type=module[0],
                module_id=module[1],
                event_data=module_event_data[module],
            )
            for module in (module_type_id or ())
            if module_event_data.get(module)
        )

        return module_results, tuple(kernel_batch)

    # Resolves the number of worker processes to use for parallel processing.
    if n_workers < 1:
        n_workers = resolve_worker_count(requested_workers=0)

    # Uses LogArchiveReader to create batches optimized for parallel processing. The batch multiplier of 4
    # creates many smaller batches for better load distribution across workers.
    batch_keys = reader.get_batches(workers=n_workers, batch_multiplier=4)
    batch_arguments = [
        {
            "log_path": log_path,
            "file_names": keys,
            "onset_us": onset_us,
            "module_type_id": module_type_id,
            "module_event_codes": module_event_codes,
            "kernel_event_codes": kernel_event_codes,
        }
        for keys in batch_keys
    ]

    # Uses the provided executor or creates a managed one for this invocation.
    managed = executor is None
    active_executor = executor if executor is not None else ProcessPoolExecutor(max_workers=n_workers)

    try:
        # Submits all tasks.
        future_to_index = {
            active_executor.submit(_process_message_batch, **batch_kwargs): index
            for index, batch_kwargs in enumerate(batch_arguments)
        }

        # Collects results while maintaining message order.
        results: list[_BatchResult | None] = [None] * len(batch_arguments)

        if display_progress:
            with console.progress(
                total=len(batch_arguments), description="Extracting microcontroller log data", unit="batch"
            ) as pbar:
                for future in as_completed(future_to_index):
                    results[future_to_index[future]] = future.result()
                    pbar.update(1)
        else:
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

    finally:
        if managed:
            active_executor.shutdown(wait=True)

    # Combines module processing results from all batches.
    combined_module_data: dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]] = {
        module: {} for module in (module_type_id or ())
    }

    # Combines kernel processing results from all batches.
    all_kernel_messages: list[ExtractedMessageData] = []

    # Processes results from each batch to maintain chronological ordering.
    for batch_result in results:
        if batch_result is None:
            continue

        module_batch, kernel_batch = batch_result

        for module, event_dict in module_batch.items():
            for event, message_list in event_dict.items():
                if event not in combined_module_data[module]:
                    combined_module_data[module][event] = message_list
                else:
                    combined_module_data[module][event].extend(message_list)

        all_kernel_messages.extend(kernel_batch)

    # Converts all lists to tuples for the final data structure.
    module_results = tuple(
        ExtractedModuleData(
            module_type=module[0],
            module_id=module[1],
            event_data={event: tuple(msg_list) for event, msg_list in event_dict.items()},
        )
        for module in (module_type_id or ())
        if (event_dict := combined_module_data.get(module)) and event_dict
    )

    return module_results, tuple(all_kernel_messages)


def resolve_recording_roots(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]:
    """Resolves a set of discovered log directories to their recording root directories.

    Recording roots are the meaningful top-level directories that uniquely identify each recording session. Log
    archives and pipeline outputs may be nested at arbitrary depths below the root, but the root itself is essential
    for proper recording identification and display labels. Uses _extract_unique_components to identify the first path
    component (from the end) that uniquely distinguishes each path, then truncates each path at that component to
    strip shared structural subdirectories without assuming a fixed directory hierarchy.

    Args:
        paths: The directories containing discovered log archives. Each path is resolved to its recording root
            by walking up to the ancestor matching its unique component.

    Returns:
        A deduplicated tuple of recording root paths, one per unique recording.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    unique_ids = _extract_unique_components(paths=list(paths))
    roots: list[Path] = []
    for path, unique_id in zip(paths, unique_ids, strict=True):
        current = path
        while current.name != unique_id and current != current.parent:
            current = current.parent
        if current not in roots:
            roots.append(current)
    return tuple(roots)


def find_log_archive(log_directory: Path, source_id: str) -> Path:
    """Searches for a single log archive matching the target source ID under the log directory.

    Recursively searches the log_directory and all subdirectories for an archive file matching the
    ``{source_id}_log.npz`` naming convention. Expects exactly one match per source ID within the directory tree.

    Args:
        log_directory: The path to the root directory to search. The directory is searched recursively, so archives
            may be nested at any depth below this path.
        source_id: The source ID string to match. Corresponds to the filename prefix before the ``_log.npz`` suffix.

    Returns:
        The path to the discovered log archive.

    Raises:
        FileNotFoundError: If the log_directory does not exist, is not a directory, or no archive matching the
            source ID is found.
        ValueError: If multiple archives matching the source ID are found under the log directory.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = (
            f"Unable to find log archive for source '{source_id}' in '{log_directory}'. The path does not exist or "
            f"is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    matches = sorted(log_directory.rglob(f"{source_id}{LOG_ARCHIVE_SUFFIX}"))

    if not matches:
        message = (
            f"Unable to find log archive for source '{source_id}' in '{log_directory}'. No file matching "
            f"'{source_id}{LOG_ARCHIVE_SUFFIX}' was found."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to find log archive for source '{source_id}' in '{log_directory}'. Found {len(matches)} "
            f"matching archives, but expected exactly one: {[str(p) for p in matches]}."
        )
        console.error(message=message, error=ValueError)

    return matches[0]


def initialize_processing_tracker(
    output_directory: Path,
    source_ids: list[str],
) -> dict[str, str]:
    """Initializes the processing tracker file with data extraction jobs for each source ID.

    Notes:
        Used to process data in the 'local' processing mode. During remote data processing, the tracker file is
        pre-generated before submitting the processing jobs to the remote compute server.

    Args:
        output_directory: The path to the output directory where the tracker file is created.
        source_ids: The source ID strings for the log archives to track.

    Returns:
        A dictionary mapping source IDs to their generated hexadecimal job identifiers.
    """
    tracker = ProcessingTracker(file_path=output_directory / TRACKER_FILENAME)

    # Builds the (job_name, specifier) tuples required by the tracker's initialization interface.
    jobs: list[tuple[str, str]] = [(_EXTRACTION_JOB_NAME, source_id) for source_id in source_ids]
    tracker.initialize_jobs(jobs=jobs)

    return _generate_job_ids(source_ids=source_ids)


def _process_message_batch(
    log_path: Path,
    file_names: list[str],
    onset_us: np.uint64,
    module_type_id: tuple[tuple[int, int], ...] | None,
    module_event_codes: frozenset[int] | None,
    kernel_event_codes: frozenset[int] | None,
) -> _BatchResult:  # pragma: no cover
    """Processes a batch of messages from a MicroControllerInterface log archive, extracting both hardware module
    and kernel messages in a single pass.

    This worker function is used by extract_log_data() to process multiple message batches in parallel. Each message
    is routed to module or kernel results based on its protocol code, then filtered by the caller's event codes.
    Event codes are guaranteed to be unique within each module and the kernel, so command code filtering is not
    required.

    Args:
        log_path: The path to the processed .npz log file.
        file_names: The names of the individual message .npy files stored in the target archive.
        onset_us: The onset of the data acquisition, in microseconds elapsed since UTC epoch onset.
        module_type_id: The module type and ID codes to extract, or None to skip module extraction.
        module_event_codes: The event codes to extract for module messages, or None to skip module extraction.
        kernel_event_codes: The event codes to extract for kernel messages, or None to skip kernel extraction.

    Returns:
        A tuple of (module_results, kernel_results). module_results maps module (type, id) tuples to dicts of
        event code -> list of ExtractedMessageData. kernel_results is a list of ExtractedMessageData in
        chronological order.
    """
    # Pre-creates the result structures.
    extract_modules = module_type_id is not None and module_event_codes is not None
    extract_kernel = kernel_event_codes is not None

    module_data: dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]] = (
        {module: {} for module in module_type_id} if extract_modules else {}
    )
    kernel_messages: list[ExtractedMessageData] = []

    # Uses LogArchiveReader to iterate over the batch messages. Passing the pre-discovered onset_us avoids redundant
    # onset scanning in each worker process.
    reader = LogArchiveReader(archive_path=log_path, onset_us=onset_us)
    for log_msg in reader.iter_messages(keys=file_names):
        payload = log_msg.payload
        protocol = payload[0]

        # Routes module messages (MODULE_DATA / MODULE_STATE) through the extraction pipeline.
        if extract_modules and protocol in {SerialProtocols.MODULE_DATA, SerialProtocols.MODULE_STATE}:
            # Checks if this message comes from one of the requested modules.
            current_module = None
            for module in module_type_id:  # type: ignore[union-attr]
                if payload[1] == module[0] and payload[2] == module[1]:
                    current_module = module
                    break

            if current_module is None:
                continue

            # Extracts only messages with requested event codes.
            event_code_int = int(payload[4])
            if event_code_int not in module_event_codes:  # type: ignore[operator]
                continue

            # Extracts command and event codes.
            command_code = np.uint8(payload[3])
            event = np.uint8(payload[4])

            # Extracts the data object for MODULE_DATA messages (protocol code distinguishes DATA from STATE).
            data: None | np.number | NDArray[np.number] = None
            prototype_code: np.uint8 | None = None
            if protocol == SerialProtocols.MODULE_DATA:
                prototype_code = np.uint8(payload[5])
                # noinspection PyTypeChecker
                prototype_object = SerialPrototypes.get_prototype_for_code(code=payload[5])

                if isinstance(prototype_object, np.ndarray):
                    data = payload[6:].view(prototype_object.dtype)[:].copy()  # type: ignore[assignment]
                elif prototype_object is not None:
                    data = payload[6:].view(prototype_object.dtype)[0].copy()

            message_data = ExtractedMessageData(
                timestamp=log_msg.timestamp_us,
                command=command_code,
                event=event,
                prototype=prototype_code,
                data=data,
            )

            # Adds the extracted message data to the batch results, grouped by event code.
            if event not in module_data[current_module]:
                module_data[current_module][event] = [message_data]
            else:
                module_data[current_module][event].append(message_data)

        # Routes kernel messages (KERNEL_DATA / KERNEL_STATE) through the extraction pipeline.
        elif extract_kernel and protocol in {SerialProtocols.KERNEL_DATA, SerialProtocols.KERNEL_STATE}:
            # Extracts only messages with requested event codes.
            if int(payload[2]) not in kernel_event_codes:  # type: ignore[operator]
                continue

            command_code = np.uint8(payload[1])
            kernel_event = np.uint8(payload[2])

            # Extracts the data object for KERNEL_DATA messages.
            data = None
            prototype_code = None
            if protocol == SerialProtocols.KERNEL_DATA:
                prototype_code = np.uint8(payload[3])
                # noinspection PyTypeChecker
                prototype_object = SerialPrototypes.get_prototype_for_code(code=payload[3])

                if isinstance(prototype_object, np.ndarray):
                    data = payload[4:].view(prototype_object.dtype)[:].copy()  # type: ignore[assignment]
                elif prototype_object is not None:
                    data = payload[4:].view(prototype_object.dtype)[0].copy()

            kernel_messages.append(
                ExtractedMessageData(
                    timestamp=log_msg.timestamp_us,
                    command=command_code,
                    event=kernel_event,
                    prototype=prototype_code,
                    data=data,
                )
            )

    return module_data, kernel_messages


def _extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """Extracts the first component from the end of each input path that uniquely identifies each path globally.

    Adapts the processing pipeline to directory structures where the unique recording identifier appears at different
    levels of the path hierarchy. For example, given paths like ``/data/day1/recording`` and ``/data/day2/recording``,
    identifies ``day1`` and ``day2`` as the unique components (not ``recording``, which is shared).

    Args:
        paths: The list or tuple of Path objects to extract unique components from.

    Returns:
        A tuple of unique component strings, one for each path, stored in the same order as the input paths.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    paths_list = list(paths)
    unique_components: list[str] = []

    for index, path in enumerate(paths_list):
        # Iterates components from right to left to find the first one unique to this path.
        components = list(path.parts)[::-1]
        found_unique = False

        for component in components:
            # Checks whether this component appears in any other path.
            is_unique = all(
                component not in other_path.parts
                for other_index, other_path in enumerate(paths_list)
                if other_index != index
            )

            if is_unique:
                unique_components.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"Unable to extract a unique component from the given path: {path}."
            console.error(message=message, error=RuntimeError)

    return tuple(unique_components)


def _generate_job_ids(source_ids: list[str]) -> dict[str, str]:
    """Generates unique processing job identifiers for each source ID.

    Args:
        source_ids: The list of source ID strings for which to generate job IDs.

    Returns:
        A dictionary mapping source IDs to their generated hexadecimal job identifiers.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=_EXTRACTION_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }


def _write_module_feather(
    module_data: ExtractedModuleData,
    source_id: str,
    output_directory: Path,
) -> None:
    """Serializes extracted module data to a feather (IPC) file.

    Converts the event-grouped message data for a single hardware module into a columnar feather file with
    timestamp, command, event, prototype, and binary payload columns. Writes the file to the output directory
    using the ``controller_{source_id}_module_{type}_{id}.feather`` naming convention.

    Args:
        module_data: The extracted data for a single hardware module.
        source_id: The source ID string identifying the originating log archive.
        output_directory: The path to the directory where the feather file is written.
    """
    # Flattens the event-grouped structure into columnar arrays.
    timestamps: list[int] = []
    commands: list[int] = []
    events: list[int] = []
    prototypes: list[int | None] = []
    data_bytes_column: list[bytes | None] = []

    for messages in module_data.event_data.values():
        for message in messages:
            timestamps.append(int(message.timestamp))
            commands.append(int(message.command))
            events.append(int(message.event))
            prototypes.append(int(message.prototype) if message.prototype is not None else None)
            if message.data is not None:
                data_bytes_column.append(np.asarray(message.data).tobytes())
            else:
                data_bytes_column.append(None)

    dataframe = pl.DataFrame(
        {
            "timestamp_us": pl.Series(name="timestamp_us", values=timestamps, dtype=pl.UInt64),
            "command": pl.Series(name="command", values=commands, dtype=pl.UInt8),
            "event": pl.Series(name="event", values=events, dtype=pl.UInt8),
            "prototype": pl.Series(name="prototype", values=prototypes, dtype=pl.UInt8),
            "data": pl.Series(name="data", values=data_bytes_column, dtype=pl.Binary),
        }
    )

    filename = (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}"
        f"{MODULE_FEATHER_INFIX}{module_data.module_type}_{module_data.module_id}"
        f"{FEATHER_SUFFIX}"
    )
    dataframe.write_ipc(file=output_directory / filename)


def _write_kernel_feather(
    kernel_messages: tuple[ExtractedMessageData, ...],
    source_id: str,
    output_directory: Path,
) -> None:
    """Serializes extracted kernel messages to a feather (IPC) file.

    Converts the kernel message tuple into a columnar feather file with timestamp, command, event, prototype,
    and binary payload columns. Writes the file to the output directory using the
    ``controller_{source_id}_kernel.feather`` naming convention.

    Args:
        kernel_messages: The extracted kernel messages in chronological order.
        source_id: The source ID string identifying the originating log archive.
        output_directory: The path to the directory where the feather file is written.
    """
    timestamps: list[int] = []
    commands: list[int] = []
    events: list[int] = []
    prototypes: list[int | None] = []
    data_bytes_column: list[bytes | None] = []

    for message in kernel_messages:
        timestamps.append(int(message.timestamp))
        commands.append(int(message.command))
        events.append(int(message.event))
        prototypes.append(int(message.prototype) if message.prototype is not None else None)
        if message.data is not None:
            data_bytes_column.append(np.asarray(message.data).tobytes())
        else:
            data_bytes_column.append(None)

    dataframe = pl.DataFrame(
        {
            "timestamp_us": pl.Series(name="timestamp_us", values=timestamps, dtype=pl.UInt64),
            "command": pl.Series(name="command", values=commands, dtype=pl.UInt8),
            "event": pl.Series(name="event", values=events, dtype=pl.UInt8),
            "prototype": pl.Series(name="prototype", values=prototypes, dtype=pl.UInt8),
            "data": pl.Series(name="data", values=data_bytes_column, dtype=pl.Binary),
        }
    )

    filename = f"{CONTROLLER_FEATHER_PREFIX}{source_id}{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}"
    dataframe.write_ipc(file=output_directory / filename)
