"""Provides the log data processing pipeline for extracting hardware module and kernel message data from
MicroControllerInterface log archives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import LogArchiveReader, ProcessingTracker

from .manifest import MICROCONTROLLER_MANIFEST_FILENAME, MicroControllerManifest
from .communication import SerialProtocols, SerialPrototypes

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

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

_EXTRACTION_JOB_NAME: str = "microcontroller_data_extraction"
"""The job name used by the processing pipeline for microcontroller data extraction."""

_SERVICE_CODE_THRESHOLD: int = 50
"""The highest code-value used by 'service' (system-reserved) Module messages."""

_COMMAND_COMPLETE_CODE: int = 2
"""The event code used by hardware modules to report command completion."""

_MINIMUM_MODULE_DATA_SIZE: int = 5
"""The smallest non-service data payload size currently used by hardware module instances to communicate with the PC."""

_MINIMUM_KERNEL_DATA_SIZE: int = 3
"""The smallest payload size for KernelState messages (command + event + no data). KernelData messages with a data
object have payloads larger than this threshold."""


@dataclass(slots=True)
class ExtractedMessageData:
    """Stores the data parsed from a message sent to the PC by a hardware module instance during runtime."""

    timestamp: np.uint64
    """The number of microseconds elapsed since the UTC epoch onset when the message was received by the PC."""
    command: np.uint8
    """The code of the command that the module was executing when it sent the message to the PC."""
    data: None | np.number | NDArray[np.number]
    """The parsed data object transmitted with the message (or None, for state-only messages)."""


@dataclass(slots=True)
class ExtractedModuleData:
    """Stores the data parsed from all non-service messages sent to the PC by a hardware module instance during
    runtime.
    """

    module_type: int
    """The type (family) code of the hardware module instance whose data is stored in the 'data' attribute."""
    module_id: int
    """The unique identifier code of the hardware module instance whose data is stored in the 'data' attribute."""
    # noinspection PyTypeHints
    event_data: dict[np.uint8, tuple[ExtractedMessageData, ...]]
    """A dictionary that uses message event codes as keys and tuples of ExtractedMessageData instances as values."""


def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    job_id: str | None = None,
    log_ids: list[str] | None = None,
    module_callbacks: dict[tuple[int, int], Callable[[ExtractedModuleData, Path], None]] | None = None,
    kernel_callback: Callable[[tuple[ExtractedMessageData, ...], Path], None] | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None:
    """Processes the requested MicroControllerInterface log archives from a single DataLogger output directory.

    Supports both local and remote processing modes. In local mode (job_id is None), resolves each requested log
    archive by source ID, initializes a processing tracker in the output directory, and executes all requested jobs
    sequentially. In remote mode (job_id is provided), generates all possible job IDs for the requested source IDs
    and executes only the matching job.

    All resolved archives must reside in the same directory. If the log_directory contains archives from multiple
    DataLogger instances (in separate subdirectories), each must be processed independently. Use the MCP batch
    processing tools to orchestrate multi-directory workflows.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``microcontroller_data/`` subdirectory is created
            automatically under this path, and all tracker and output files are written there.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all requested jobs are run sequentially
            with automatic tracker management (local mode).
        log_ids: A list of source log IDs to process. Each ID must correspond to exactly one archive under the
            log directory, and all archives must reside in the same parent directory. If not provided, reads the
            microcontroller_manifest.yaml file from the log directory to resolve all registered source IDs.
        module_callbacks: An optional dictionary mapping (module_type, module_id) tuples to callback functions for
            post-extraction processing. Passed through to each execute_job() invocation.
        kernel_callback: An optional callback function for processing kernel messages. Passed through to each
            execute_job() invocation.
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores. Setting this to 1 conducts processing sequentially.
        display_progress: Determines whether to display progress bars during extraction. Defaults to True for
            interactive CLI use. Set to False for MCP batch processing.

    Raises:
        FileNotFoundError: If the log_directory does not exist, a requested log ID has no matching archive, or no
            microcontroller manifest is found when log_ids is not provided.
        ValueError: If the provided job_id does not match any discoverable job, if no source IDs can be resolved,
            if a requested log ID matches multiple archives, or if resolved archives span multiple directories.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = f"Unable to process logs in '{log_directory}'. The path does not exist or is not a directory."
        console.error(message=message, error=FileNotFoundError)

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
    manifest = MicroControllerManifest.from_yaml(file_path=manifest_path)
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
            module_callbacks=module_callbacks,
            kernel_callback=kernel_callback,
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
                    module_callbacks=module_callbacks,
                    kernel_callback=kernel_callback,
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
    module_callbacks: dict[tuple[int, int], Callable[[ExtractedModuleData, Path], None]] | None = None,
    kernel_callback: Callable[[tuple[ExtractedMessageData, ...], Path], None] | None = None,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> None:
    """Executes a single data extraction job for the target log archive.

    Extracts hardware module data from the log archive, invokes any registered module callbacks for post-extraction
    processing, and optionally extracts and processes kernel messages.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output files are written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        module_callbacks: An optional dictionary mapping (module_type, module_id) tuples to callback functions. Each
            callback receives the ExtractedModuleData for that module and the output directory path. The callbacks are
            invoked after the extraction phase completes.
        kernel_callback: An optional callback function for processing kernel messages. Receives a tuple of
            ExtractedMessageData instances and the output directory path.
        display_progress: Determines whether to display a progress bar during extraction.
        executor: An optional pre-created ProcessPoolExecutor to reuse for parallel processing.
    """
    console.echo(message=f"Running '{_EXTRACTION_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Determines which module (type, id) tuples to extract from the callbacks dictionary.
        if module_callbacks:
            module_type_id = tuple(module_callbacks.keys())

            # Extracts hardware module data from the log archive.
            extracted_data = extract_logged_hardware_module_data(
                log_path=log_path,
                module_type_id=module_type_id,
                n_workers=workers,
                display_progress=display_progress,
                executor=executor,
            )

            # Invokes the registered callback for each extracted module's data.
            for module_data in extracted_data:
                callback_key = (module_data.module_type, module_data.module_id)
                if callback_key in module_callbacks:
                    module_callbacks[callback_key](module_data, output_directory)

        # Extracts and processes kernel messages only if a kernel callback is provided.
        if kernel_callback is not None:
            kernel_messages = extract_kernel_messages(
                log_path=log_path,
                n_workers=workers,
                display_progress=display_progress,
                executor=executor,
            )
            kernel_callback(kernel_messages, output_directory)

        tracker.complete_job(job_id=job_id)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def extract_logged_hardware_module_data(
    log_path: Path,
    module_type_id: tuple[tuple[int, int], ...],
    n_workers: int = -1,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[ExtractedModuleData, ...]:
    """Extracts the received message data for the requested hardware module instances from the .npz log file generated
    by a MicroControllerInterface instance during runtime.

    This function reads the '.npz' archive generated by the DataLogger's assemble_log_archives() method for a
    MicroControllerInterface instance and extracts the data for all non-system messages transmitted by the requested
    hardware module instances from the microcontroller to the PC.

    Notes:
        At this time, the function exclusively works with the data sent by the microcontroller to the PC.

        The extracted data does not contain library-reserved events and messages. This includes all Kernel messages
        and module messages with event codes 0 through 50. The only exceptions to this rule are messages with event
        code 2, which report command completion. These messages are parsed in addition to custom messages sent by each
        hardware module.

        If the target .npz archive contains fewer than 2000 messages, the processing is carried out sequentially
        regardless of the specified worker-count.

        When an external executor is provided, batch processing is submitted to that executor instead of creating a
        new ProcessPoolExecutor. The caller is responsible for executor lifecycle management. This allows multiple
        archives with similar sizes to share a single process pool, avoiding the overhead of repeatedly spawning and
        tearing down worker processes.

    Args:
        log_path: The path to the .npz archive file that stores the logged data generated by the
            MicroControllerInterface instance during runtime.
        module_type_id: A tuple of tuples, where each inner tuple stores the type and ID codes of a specific hardware
            module, whose data should be extracted from the archive, e.g.: ((3, 1), (4, 2)).
        n_workers: The number of parallel worker processes (CPU cores) to use for processing. Setting this to a value
            below 1 uses all available CPU cores. Setting this to a value of 1 conducts the processing sequentially.
        display_progress: Determines whether to display a progress bar during parallel batch processing.
        executor: An optional pre-created ProcessPoolExecutor to use for parallel batch processing. When provided,
            the function submits work to this executor instead of creating its own. The caller must ensure the
            executor's worker count matches the n_workers value used for batch generation.

    Returns:
        A tuple of ExtractedModuleData instances. Each instance stores all data extracted from the log archive for one
        specific hardware module instance.

    Raises:
        ValueError: If the target .npz archive does not exist.
    """
    # Raises an error if the specified compressed log archive does not exist.
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        message = (
            f"Unable to extract hardware module data from the log file {log_path}, as it does not exist or does "
            f"not point to a valid .npz archive."
        )
        console.error(message=message, error=ValueError)

    # Uses LogArchiveReader to handle archive access, onset timestamp discovery, and batch creation.
    reader = LogArchiveReader(archive_path=log_path)

    # Returns an empty tuple if there are no messages to process.
    if reader.message_count == 0:
        return ()

    onset_us = reader.onset_timestamp_us

    # Processes small archives sequentially to avoid the unnecessary overhead of setting up the multiprocessing
    # runtime. Also applies when the user explicitly requests a single worker process.
    module_event_data: dict[tuple[int, int], dict[np.uint8, tuple[ExtractedMessageData, ...]]]
    if n_workers == 1 or reader.message_count < PARALLEL_PROCESSING_THRESHOLD:
        # Processes all messages in a single batch sequentially.
        all_keys = reader.get_batches(workers=1, batch_multiplier=1)
        batch_results = _process_module_message_batch(
            log_path=log_path, file_names=all_keys[0], onset_us=onset_us, module_type_id=module_type_id
        )

        # Converts the lists of ExtractedMessageData instances to tuples for the final data structure.
        module_event_data = {
            module: {event: tuple(message_list) for event, message_list in event_dict.items()}
            for module, event_dict in batch_results.items()
        }
    else:
        # Resolves the number of worker processes to use for parallel processing.
        if n_workers < 1:
            n_workers = resolve_worker_count(requested_workers=0)

        # Uses LogArchiveReader to create batches optimized for parallel processing. The batch multiplier of 4
        # creates many smaller batches for better load distribution across workers.
        batch_keys = reader.get_batches(workers=n_workers, batch_multiplier=4)
        batches = [(log_path, keys, onset_us, module_type_id) for keys in batch_keys]

        # Uses the provided executor or creates a managed one for this invocation.
        managed = executor is None
        active_executor = executor if executor is not None else ProcessPoolExecutor(max_workers=n_workers)

        try:
            # Submits all tasks.
            future_to_index = {
                active_executor.submit(_process_module_message_batch, *batch_args): idx
                for idx, batch_args in enumerate(batches)
            }

            # Collects results while maintaining message order.
            results: list[dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]] | None] = [None] * len(
                batches
            )

            if display_progress:
                # Creates a progress bar for batch processing.
                with console.progress(
                    total=len(batches), description="Extracting microcontroller hardware module data", unit="batch"
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

        # Combines processing results from all batches.
        combined_module_data: dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]] = {
            module: {} for module in module_type_id
        }

        # Processes results from each batch to maintain chronological ordering.
        for batch_result in results:
            if batch_result is not None:
                for module, event_dict in batch_result.items():
                    for event, message_list in event_dict.items():
                        if event not in combined_module_data[module]:
                            combined_module_data[module][event] = message_list
                        else:
                            combined_module_data[module][event].extend(message_list)

        # Converts all lists to tuples for the final data structure.
        module_event_data = {
            module: {event: tuple(message_list) for event, message_list in event_dict.items()}
            for module, event_dict in combined_module_data.items()
        }

    # Creates ExtractedModuleData instances for each module and returns the tuple to the caller.
    return tuple(
        ExtractedModuleData(
            module_type=module[0],
            module_id=module[1],
            event_data=module_event_data[module],
        )
        for module in module_type_id
        if module_event_data[module]
    )


def extract_kernel_messages(
    log_path: Path,
    n_workers: int = -1,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[ExtractedMessageData, ...]:
    """Extracts kernel-level messages (KernelData and KernelState) from the .npz log file generated by a
    MicroControllerInterface instance during runtime.

    This function reads the '.npz' archive and extracts all non-service Kernel messages. Service messages with event
    codes 0 through 50 are excluded, with the exception of command completion messages (event code 2).

    Notes:
        If the target .npz archive contains fewer than 2000 messages, the processing is carried out sequentially
        regardless of the specified worker-count.

    Args:
        log_path: The path to the .npz archive file that stores the logged data.
        n_workers: The number of parallel worker processes (CPU cores) to use for processing. Setting this to a value
            below 1 uses all available CPU cores. Setting this to a value of 1 conducts the processing sequentially.
        display_progress: Determines whether to display a progress bar during parallel batch processing.
        executor: An optional pre-created ProcessPoolExecutor to use for parallel batch processing.

    Returns:
        A tuple of ExtractedMessageData instances containing the extracted kernel messages in chronological order.

    Raises:
        ValueError: If the target .npz archive does not exist.
    """
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        message = (
            f"Unable to extract kernel message data from the log file {log_path}, as it does not exist or does "
            f"not point to a valid .npz archive."
        )
        console.error(message=message, error=ValueError)

    reader = LogArchiveReader(archive_path=log_path)

    if reader.message_count == 0:
        return ()

    onset_us = reader.onset_timestamp_us

    # Processes sequentially for small archives or explicit single-worker requests.
    if n_workers == 1 or reader.message_count < PARALLEL_PROCESSING_THRESHOLD:
        all_keys = reader.get_batches(workers=1, batch_multiplier=1)
        return tuple(_process_kernel_message_batch(log_path=log_path, file_names=all_keys[0], onset_us=onset_us))

    # Resolves workers and generates batches for parallel processing.
    if n_workers < 1:
        n_workers = resolve_worker_count(requested_workers=0)

    batch_keys = reader.get_batches(workers=n_workers, batch_multiplier=4)

    managed = executor is None
    active_executor = executor if executor is not None else ProcessPoolExecutor(max_workers=n_workers)

    try:
        future_to_index = {
            active_executor.submit(_process_kernel_message_batch, log_path, keys, onset_us): index
            for index, keys in enumerate(batch_keys)
        }

        results: list[list[ExtractedMessageData] | None] = [None] * len(batch_keys)

        if display_progress:
            with console.progress(
                total=len(batch_keys), description="Extracting kernel messages", unit="batch"
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

    # Concatenates batch results while maintaining chronological order.
    all_messages: list[ExtractedMessageData] = []
    for batch_result in results:
        if batch_result is not None:
            all_messages.extend(batch_result)

    return tuple(all_messages)


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


def _process_module_message_batch(
    log_path: Path,
    file_names: list[str],
    onset_us: np.uint64,
    module_type_id: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]]:  # pragma: no cover
    """Processes the target batch of MicroControllerInterface-generated messages stored in the .npz log file.

    This worker function is used by the extract_logged_hardware_module_data() function to process multiple message
    batches in parallel to speed up the hardware module data extraction.

    Args:
        log_path: The path to the processed .npz log file.
        file_names: The names of the individual message .npy files stored in the target archive.
        onset_us: The onset of the data acquisition, in microseconds elapsed since UTC epoch onset.
        module_type_id: The module type and ID codes to extract.

    Returns:
        A dictionary mapping module (type, id) tuples to lists of extracted data dictionaries. Each data dictionary
        contains a list of ExtractedMessageData instances for each processed event code.
    """
    # Pre-creates the dictionary to store the extracted data for this batch.
    batch_data: dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]] = {
        module: {} for module in module_type_id
    }

    # Uses LogArchiveReader to iterate over the batch messages. Passing the pre-discovered onset_us avoids redundant
    # onset scanning in each worker process.
    reader = LogArchiveReader(archive_path=log_path, onset_us=onset_us)
    for log_msg in reader.iter_messages(keys=file_names):
        payload = log_msg.payload

        # Filters messages to exclusively process custom Data and State messages (event codes 51 and above).
        # The only exception to this rule is the CommandComplete state message, which uses the system-reserved code
        # '2'. In the future, if enough interest is shown, this list may be extended to also include outgoing
        # messages. For now, these messages need to be parsed manually by users that need this data.
        if (payload[0] != SerialProtocols.MODULE_STATE and payload[0] != SerialProtocols.MODULE_DATA) or (
            payload[4] != _COMMAND_COMPLETE_CODE and payload[4] <= _SERVICE_CODE_THRESHOLD
        ):
            continue

        # Checks if this message comes from one of the processed modules.
        current_module = None
        for module in module_type_id:
            if payload[1] == module[0] and payload[2] == module[1]:
                current_module = module
                break

        if current_module is None:
            continue

        # Extracts command, event, and, if supported, data object from the message payload.
        command_code = np.uint8(payload[3])
        event = np.uint8(payload[4])

        # Handles MessageData payloads exclusively. MessageState payloads are only 5 bytes in size and do not
        # contain a data object.
        data: None | np.number | NDArray[np.number] = None
        if len(payload) > _MINIMUM_MODULE_DATA_SIZE:
            # noinspection PyTypeChecker
            prototype = SerialPrototypes.get_prototype_for_code(code=payload[5])

            # Depending on the prototype, reads the data object as an array or scalar.
            if isinstance(prototype, np.ndarray):
                data = payload[6:].view(prototype.dtype)[:].copy()  # type: ignore[assignment]
            elif prototype is not None:
                data = payload[6:].view(prototype.dtype)[0].copy()
            else:
                data = None  # Marks as an error case.

        # Creates an ExtractedMessageData instance with the extracted information.
        message_data = ExtractedMessageData(
            timestamp=log_msg.timestamp_us,
            command=command_code,
            data=data,
        )

        # Adds the extracted message data to the batch results, grouped by event code.
        if event not in batch_data[current_module]:
            batch_data[current_module][event] = [message_data]
        else:
            batch_data[current_module][event].append(message_data)

    return batch_data


def _process_kernel_message_batch(
    log_path: Path,
    file_names: list[str],
    onset_us: np.uint64,
) -> list[ExtractedMessageData]:  # pragma: no cover
    """Processes a batch of Kernel-originated messages from a MicroControllerInterface log archive.

    This worker function is used by the extract_kernel_messages() function to process multiple message batches in
    parallel to speed up the kernel message data extraction.

    Args:
        log_path: The path to the processed .npz log file.
        file_names: The names of the individual message .npy files stored in the target archive.
        onset_us: The onset of the data acquisition, in microseconds elapsed since UTC epoch onset.

    Returns:
        A list of ExtractedMessageData instances for the kernel messages found in this batch.
    """
    batch_messages: list[ExtractedMessageData] = []

    reader = LogArchiveReader(archive_path=log_path, onset_us=onset_us)
    for log_msg in reader.iter_messages(keys=file_names):
        payload = log_msg.payload

        # Only processes KernelData and KernelState messages.
        if payload[0] != SerialProtocols.KERNEL_DATA and payload[0] != SerialProtocols.KERNEL_STATE:
            continue

        # KernelState and KernelData share the same header layout: command (byte 1), event (byte 2).
        command_code = np.uint8(payload[1])
        event = np.uint8(payload[2])

        # Skips service codes below the threshold (except command complete).
        if event != _COMMAND_COMPLETE_CODE and event <= _SERVICE_CODE_THRESHOLD:
            continue

        # Extracts the data object for KernelData messages (payload > _MINIMUM_KERNEL_DATA_SIZE includes
        # prototype + data).
        data: None | np.number | NDArray[np.number] = None
        if payload[0] == SerialProtocols.KERNEL_DATA and len(payload) > _MINIMUM_KERNEL_DATA_SIZE:
            # noinspection PyTypeChecker
            prototype = SerialPrototypes.get_prototype_for_code(code=payload[_MINIMUM_KERNEL_DATA_SIZE])

            if isinstance(prototype, np.ndarray):
                data = payload[_MINIMUM_KERNEL_DATA_SIZE + 1 :].view(prototype.dtype)[:].copy()  # type: ignore[assignment]
            elif prototype is not None:
                data = payload[_MINIMUM_KERNEL_DATA_SIZE + 1 :].view(prototype.dtype)[0].copy()

        batch_messages.append(
            ExtractedMessageData(
                timestamp=log_msg.timestamp_us,
                command=command_code,
                data=data,
            )
        )

    return batch_messages


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
