"""Provides the log data processing pipeline for extracting hardware module and kernel message data from
MicroControllerInterface log archives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import LogArchiveReader, ProcessingTracker

from .dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    MicroControllerManifest,
    ControllerExtractionConfig,
)
from .communication import SerialProtocols, SerialPrototypes

if TYPE_CHECKING:
    from pathlib import Path

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
"""Filename infix separating the controller source ID from the module type code and ID code."""

KERNEL_FEATHER_INFIX: str = "_kernel"
"""Filename infix identifying kernel message feather files."""

FEATHER_SUFFIX: str = ".feather"
"""Filename suffix for all feather (IPC) output files."""

EXTRACTION_JOB_NAME: str = "microcontroller_data_extraction"
"""The job name used by the processing pipeline for microcontroller data extraction."""


@dataclass(slots=True)
class _ExtractedMessages:
    """Stores the data parsed from a set of incoming messages received by the PC from the microcontroller during
    runtime, in columnar form. All arrays share the same length, with each index position corresponding to a single
    message.
    """

    timestamps: NDArray[np.uint64]
    """Microseconds elapsed since the UTC epoch onset when each message was received by the PC."""
    commands: NDArray[np.uint8]
    """The command code that the module or kernel was executing when it sent each message."""
    events: NDArray[np.uint8]
    """The event code of each message."""
    dtypes: tuple[str | None, ...]
    """The numpy dtype string for the data payload of each message (e.g., ``'float32'``, ``'uint16'``), or None for
    state-only messages that carry no data. Combined with ``data_payloads``, this allows reconstruction of the
    original numpy array via ``np.frombuffer(payload, dtype=dtype_str)`` without any library dependency."""
    data_payloads: tuple[bytes | None, ...]
    """The serialized binary payload of each message, or None for state-only messages. Each entry is the raw byte
    representation of the numpy data array, decodable via the corresponding ``dtypes`` entry."""

    @property
    def count(self) -> int:
        """Returns the number of messages stored in this columnar block."""
        return len(self.timestamps)


@dataclass(slots=True)
class _ExtractedModuleData:
    """Stores the data extracted from all messages sent to the PC by a hardware module instance during runtime that
    matched the caller's event code filter, in columnar form.
    """

    module_type: int
    """The type (family) code of the hardware module instance."""
    module_id: int
    """The unique identifier code of the hardware module instance."""
    messages: _ExtractedMessages
    """Columnar storage for all extracted messages from this module."""


@dataclass(slots=True)
class _ColumnAccumulator:
    """Accumulates message data in parallel lists during batch extraction, then converts to finalized numpy arrays
    via ``_finalize_accumulator``.
    """

    timestamps: list[int]
    """Microseconds elapsed since the UTC epoch onset when each message was received by the PC."""
    commands: list[int]
    """The command code for each message."""
    events: list[int]
    """The event code for each message."""
    dtypes: list[str | None]
    """The numpy dtype string for each message's data payload, or None for state-only messages."""
    data_payloads: list[bytes | None]
    """The serialized binary payload of each message, or None for state-only messages."""


type _BatchResult = tuple[
    dict[tuple[int, int], _ColumnAccumulator],
    _ColumnAccumulator,
]
"""Describes the return type of _process_message_batch: a module data dictionary mapping (type, id) tuples to
column accumulators, and a column accumulator for kernel messages."""


def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None:
    """Processes the requested MicroControllerInterface log archives from a single DataLogger output directory.

    Extracts hardware module and / or kernel message data as specified by the extraction configuration and writes the
    results to feather (IPC) files. The controller IDs to process are resolved directly from the extraction
    configuration. Each controller ID is validated against the microcontroller manifest to confirm the corresponding
    log archives were produced by ataraxis-communication-interface.

    Supports both local and remote processing modes. In local mode (job_id is None), resolves each requested log
    archive by controller ID, initializes a processing tracker in the output directory, and executes all jobs
    sequentially. In remote mode (job_id is provided), generates all possible job IDs for the configured controllers
    and executes only the matching job.

    All resolved archives must reside in the same directory. If the log_directory contains archives from multiple
    DataLogger instances (in separate subdirectories), each must be processed independently. Use the MCP batch
    processing tools to orchestrate multi-directory workflows.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``microcontroller_data/`` subdirectory is created
            automatically under this path, and all tracker and output files are written there.
        config: The path to the extraction_config.yaml file specifying which controllers, modules, events, and
            commands to extract. Controller IDs in the config determine which archives are processed.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all configured jobs are run sequentially
            with automatic tracker management (local mode).
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores. Setting this to 1 conducts processing sequentially.
        display_progress: Determines whether to display progress bars during extraction. Defaults to True for
            interactive CLI use. Set to False for MCP batch processing.

    Raises:
        FileNotFoundError: If the log_directory does not exist, the config path does not exist, a controller ID
            has no matching archive, or no microcontroller manifest is found.
        ValueError: If the provided job_id does not match any discoverable job, if controller IDs are not
            registered in the microcontroller manifest, if resolved archives span multiple directories, or if the
            extraction config has empty event codes.
    """
    if not log_directory.exists() or not log_directory.is_dir():
        message = f"Unable to process logs in '{log_directory}'. The path does not exist or is not a directory."
        console.error(message=message, error=FileNotFoundError)

    # Loads the extraction configuration from the provided file path.
    if not config.exists() or not config.is_file():
        message = f"Unable to load extraction config from '{config}'. The path does not exist or is not a file."
        console.error(message=message, error=FileNotFoundError)
    resolved_config = ExtractionConfig.load(file_path=config)

    # Builds a lookup from controller ID to its extraction configuration. Controller IDs from the config are the
    # sole source of truth for which archives to process.
    controller_configs: dict[str, ControllerExtractionConfig] = {
        str(controller.controller_id): controller for controller in resolved_config.controllers
    }
    source_ids = sorted(controller_configs.keys())
    console.echo(message=f"Resolved {len(source_ids)} controller ID(s) from config: {', '.join(source_ids)}")

    # Validates controller IDs against the microcontroller manifest to confirm they are axci-produced log archives,
    # preventing accidental processing of archives from other libraries (e.g., ataraxis-video-system).
    candidates = sorted(log_directory.rglob(MICROCONTROLLER_MANIFEST_FILENAME))
    if not candidates:
        message = (
            f"Unable to process logs in '{log_directory}'. No {MICROCONTROLLER_MANIFEST_FILENAME} was found. "
            f"A microcontroller manifest is required to confirm the log archives were produced by "
            f"ataraxis-communication-interface."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest = MicroControllerManifest.load(file_path=candidates[0])
    manifest_ids = {str(controller.id) for controller in manifest.controllers}

    unregistered_ids = [source_id for source_id in source_ids if source_id not in manifest_ids]
    if unregistered_ids:
        message = (
            f"Unable to process logs in '{log_directory}'. The following controller IDs are not registered in "
            f"the {MICROCONTROLLER_MANIFEST_FILENAME}: {', '.join(unregistered_ids)}. The corresponding log "
            f"archives were not produced by ataraxis-communication-interface. Registered IDs: "
            f"{sorted(manifest_ids)}."
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

    # Aligns the tracker with the current invocation's job set in both local and remote modes. Foreign entries
    # on disk indicate that the extraction configuration has diverged from what was previously tracked and are
    # reset before processing begins.
    jobs: list[tuple[str, str]] = [(EXTRACTION_JOB_NAME, source_id) for source_id in source_ids]
    prepare_tracker(tracker=tracker, jobs=jobs)

    if job_id is not None:
        # Generates all possible job IDs and executes only the one matching the provided job_id (remote mode).
        all_job_ids = generate_job_ids(source_ids=source_ids)
        id_to_source: dict[str, str] = {generated_id: source for source, generated_id in all_job_ids.items()}

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
        # Runs all requested jobs sequentially (local mode). Resolves workers once and creates a shared
        # ProcessPoolExecutor to reuse across all jobs, avoiding repeated process pool creation.
        job_ids = generate_job_ids(source_ids=source_ids)

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

    Reads the archive once and routes each incoming message (MODULE_DATA, MODULE_STATE, KERNEL_DATA, KERNEL_STATE)
    through event code filters specified by the controller extraction configuration. Only messages whose event codes
    match the configuration are extracted. Writes the results to feather (IPC) files in the output directory — each
    module produces a separate feather file, and kernel messages (when configured) produce an additional feather file.
    Manages tracker state (start / complete / fail) for the job.

    Notes:
        At this time, the function exclusively works with incoming messages sent by the microcontroller to the PC.

        If the target .npz archive contains fewer than 2000 messages, the processing is carried out sequentially
        regardless of the specified worker-count.

        When an external executor is provided, batch processing is submitted to that executor instead of creating a
        new ProcessPoolExecutor. The caller is responsible for executor lifecycle management.

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
        ValueError: If the controller config has modules with empty event codes, has no extraction targets configured,
            or the log archive does not exist.
    """
    # Unpacks the controller config into per-module event code filters. Each module maps to its own frozenset of
    # event codes to prevent off-target extraction across modules that share the same controller.
    module_filters: dict[tuple[int, int], frozenset[int]] | None = None
    kernel_event_codes: frozenset[int] | None = None

    if controller_config.modules:
        module_filters = {}
        for module in controller_config.modules:
            if not module.event_codes:
                message = (
                    f"Unable to execute the data extraction job for source '{source_id}'. Module with type code "
                    f"{module.module_type} and ID code {module.module_id} has empty event_codes."
                )
                console.error(message=message, error=ValueError)
            module_filters[(module.module_type, module.module_id)] = frozenset(module.event_codes)

    if controller_config.kernel is not None:
        if not controller_config.kernel.event_codes:
            message = (
                f"Unable to execute the data extraction job for source '{source_id}'. Kernel extraction "
                f"has empty event_codes."
            )
            console.error(message=message, error=ValueError)
        kernel_event_codes = frozenset(controller_config.kernel.event_codes)

    if module_filters is None and kernel_event_codes is None:
        message = (
            f"Unable to execute the data extraction job for source '{source_id}'. The controller config "
            f"has no modules and no kernel extraction configured."
        )
        console.error(message=message, error=ValueError)

    # Validates the log archive path.
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        message = (
            f"Unable to execute the data extraction job for source '{source_id}'. The log archive "
            f"'{log_path}' does not exist or is not a valid .npz file."
        )
        console.error(message=message, error=ValueError)

    console.echo(message=f"Running '{EXTRACTION_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Uses LogArchiveReader to handle archive access, onset timestamp discovery, and batch creation.
        reader = LogArchiveReader(archive_path=log_path)

        # Skips extraction and writes no output files if the archive contains no messages.
        if not reader.message_count:
            tracker.complete_job(job_id=job_id)
            return

        onset_us = reader.onset_timestamp_us
        worker_count = workers

        # Processes small archives sequentially to avoid the unnecessary overhead of setting up the multiprocessing
        # runtime. Also applies when the caller explicitly requests a single worker process.
        if worker_count == 1 or reader.message_count < PARALLEL_PROCESSING_THRESHOLD:
            all_keys = reader.get_batches(workers=1, batch_multiplier=1)
            module_batch, kernel_accumulator = _process_message_batch(
                log_path=log_path,
                file_names=all_keys[0],
                onset_us=onset_us,
                module_filters=module_filters,
                kernel_event_codes=kernel_event_codes,
            )

            module_data = tuple(
                _ExtractedModuleData(
                    module_type=module[0],
                    module_id=module[1],
                    messages=_finalize_accumulator(module_batch[module]),
                )
                for module in (module_filters or {})
                if module_batch.get(module) and module_batch[module].timestamps
            )
            kernel_data = _finalize_accumulator(kernel_accumulator)

        else:
            # Resolves the number of worker processes to use for parallel processing.
            if worker_count < 1:
                worker_count = resolve_worker_count(requested_workers=0)

            # Uses LogArchiveReader to create batches optimized for parallel processing. The batch multiplier of 4
            # creates many smaller batches for better load distribution across workers.
            batch_keys = reader.get_batches(workers=worker_count, batch_multiplier=4)

            # Uses the provided executor or creates a managed one for this invocation.
            managed = executor is None
            active_executor = executor if executor is not None else ProcessPoolExecutor(max_workers=worker_count)

            try:
                future_to_index = {
                    active_executor.submit(
                        _process_message_batch,
                        log_path=log_path,
                        file_names=keys,
                        onset_us=onset_us,
                        module_filters=module_filters,
                        kernel_event_codes=kernel_event_codes,
                    ): index
                    for index, keys in enumerate(batch_keys)
                }

                # Collects results while maintaining message order.
                results: list[_BatchResult | None] = [None] * len(batch_keys)

                if display_progress:
                    with console.progress(
                        total=len(batch_keys), description="Extracting microcontroller log data", unit="batch"
                    ) as progress_bar:
                        for future in as_completed(future_to_index):
                            results[future_to_index[future]] = future.result()
                            progress_bar.update(1)
                else:
                    for future in as_completed(future_to_index):
                        results[future_to_index[future]] = future.result()

            finally:
                if managed:
                    active_executor.shutdown(wait=True)

            # Combines columnar accumulators from all batches, maintaining chronological ordering.
            combined_module_data: dict[tuple[int, int], _ColumnAccumulator] = {
                module: _ColumnAccumulator(timestamps=[], commands=[], events=[], dtypes=[], data_payloads=[])
                for module in (module_filters or {})
            }
            combined_kernel = _ColumnAccumulator(timestamps=[], commands=[], events=[], dtypes=[], data_payloads=[])

            for batch_result in results:
                if batch_result is None:  # pragma: no cover
                    continue

                module_batch, kernel_accumulator = batch_result

                for module_key, accumulator in module_batch.items():
                    target = combined_module_data[module_key]
                    target.timestamps.extend(accumulator.timestamps)
                    target.commands.extend(accumulator.commands)
                    target.events.extend(accumulator.events)
                    target.dtypes.extend(accumulator.dtypes)
                    target.data_payloads.extend(accumulator.data_payloads)

                combined_kernel.timestamps.extend(kernel_accumulator.timestamps)
                combined_kernel.commands.extend(kernel_accumulator.commands)
                combined_kernel.events.extend(kernel_accumulator.events)
                combined_kernel.dtypes.extend(kernel_accumulator.dtypes)
                combined_kernel.data_payloads.extend(kernel_accumulator.data_payloads)

            module_data = tuple(
                _ExtractedModuleData(
                    module_type=module[0],
                    module_id=module[1],
                    messages=_finalize_accumulator(combined_module_data[module]),
                )
                for module in (module_filters or {})
                if combined_module_data[module].timestamps
            )
            kernel_data = _finalize_accumulator(combined_kernel)

        # Writes each extracted module's data to a separate feather file.
        for extracted_module in module_data:
            _write_module_feather(
                module_data=extracted_module,
                source_id=source_id,
                output_directory=output_directory,
            )

        # Writes kernel messages to a feather file when kernel extraction was configured.
        if kernel_data.count > 0:
            _write_kernel_feather(
                kernel_data=kernel_data,
                source_id=source_id,
                output_directory=output_directory,
            )

        tracker.complete_job(job_id=job_id)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


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


def prepare_tracker(tracker: ProcessingTracker, jobs: list[tuple[str, str]]) -> None:
    """Aligns the processing tracker's job registry with the jobs discovered for the current pipeline invocation.

    Notes:
        Applies the same regeneration strategy in local and remote modes so that foreign or stale tracker
        entries consistently trigger a reset instead of silently persisting across invocations. Foreign entries
        are treated as architectural drift (the current invocation's job set has changed since the tracker was
        last written) and surfaced through a warning before the tracker is rebuilt.

        If the tracker file does not yet exist on disk, the helper initializes it from scratch with the current
        jobs. If the file exists and contains job IDs that are not part of the current invocation's expected
        set, those entries are classified as foreign and the helper emits a warning before resetting and
        reinitializing the tracker. If the file already contains a strict subset of the expected IDs, the
        helper performs an additive ``initialize_jobs`` call that registers the missing entries without
        clobbering any existing state for previously-tracked jobs. If the file already contains exactly the
        expected ID set, the helper is a no-op, which keeps ``initialize_jobs`` from emitting duplicate-entry
        warnings for the fully-aligned case.

    Args:
        tracker: The ProcessingTracker instance bound to the microcontroller_data output directory.
        jobs: The list of (job_name, specifier) tuples the current pipeline invocation intends to execute.
    """
    expected_ids = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in jobs
    }

    if not tracker.file_path.exists():
        tracker.initialize_jobs(jobs=jobs)
        return

    existing_ids = set(tracker.find_jobs(job_name="").keys())
    foreign_ids = existing_ids - expected_ids
    missing_ids = expected_ids - existing_ids

    if foreign_ids:
        console.echo(
            message=(
                f"The processing tracker at '{tracker.file_path}' contains {len(foreign_ids)} job entries "
                f"that are not part of the current invocation's job set. Resetting and reinitializing the "
                f"tracker to match the requested jobs. Foreign job IDs: {sorted(foreign_ids)}."
            ),
            level=LogLevel.WARNING,
        )
        tracker.reset()
        tracker.initialize_jobs(jobs=jobs)
        return

    if missing_ids:
        tracker.initialize_jobs(jobs=jobs)


def _process_message_batch(
    log_path: Path,
    file_names: list[str],
    onset_us: np.uint64,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
    kernel_event_codes: frozenset[int] | None,
) -> _BatchResult:  # pragma: no cover
    """Processes a batch of messages from a MicroControllerInterface log archive, extracting both hardware module
    and kernel messages in a single pass into columnar accumulators.

    This worker function is used by execute_job() to process multiple message batches in parallel. Each message
    is routed to module or kernel accumulators based on its protocol code, then filtered by per-module event codes.
    Each module is filtered against its own event code set to prevent off-target extraction across modules. Data
    payloads are converted to bytes immediately to avoid storing intermediate numpy objects.

    Args:
        log_path: The path to the processed .npz log file.
        file_names: The names of the individual message .npy files stored in the target archive.
        onset_us: The onset of the data acquisition, in microseconds elapsed since UTC epoch onset.
        module_filters: A mapping from (module_type, module_id) tuples to their per-module event code frozensets,
            or None to skip module extraction.
        kernel_event_codes: The event codes to extract for kernel messages, or None to skip kernel extraction.

    Returns:
        A tuple of (module_accumulators, kernel_accumulator). module_accumulators maps module (type, id) tuples
        to column accumulators. kernel_accumulator stores kernel messages in chronological order.
    """
    # Pre-creates columnar accumulators for each requested module and the kernel.
    extract_modules = module_filters is not None
    extract_kernel = kernel_event_codes is not None

    module_data: dict[tuple[int, int], _ColumnAccumulator] = {
        module: _ColumnAccumulator(timestamps=[], commands=[], events=[], dtypes=[], data_payloads=[])
        for module in (module_filters or {})
    }
    kernel_accumulator = _ColumnAccumulator(timestamps=[], commands=[], events=[], dtypes=[], data_payloads=[])

    # Pre-creates protocol sets outside the per-message loop to avoid re-creating them on every iteration.
    module_protocols = frozenset({SerialProtocols.MODULE_DATA, SerialProtocols.MODULE_STATE})
    kernel_protocols = frozenset({SerialProtocols.KERNEL_DATA, SerialProtocols.KERNEL_STATE})

    # Uses LogArchiveReader to iterate over the batch messages. Passing the pre-discovered onset_us avoids redundant
    # onset scanning in each worker process.
    reader = LogArchiveReader(archive_path=log_path, onset_us=onset_us)
    for log_msg in reader.iter_messages(keys=file_names):
        payload = log_msg.payload
        protocol = payload[0]

        # Routes module messages (MODULE_DATA / MODULE_STATE) through the extraction pipeline.
        if extract_modules and protocol in module_protocols:
            # Looks up the per-module event codes in a single dict access. Returns None if the module is not
            # requested, combining module membership and event filter retrieval into one O(1) operation.
            current_module = (int(payload[1]), int(payload[2]))
            module_events = module_filters.get(current_module)  # type: ignore[union-attr]
            if module_events is None:
                continue

            # Filters against this specific module's event codes, preventing off-target extraction.
            if int(payload[4]) not in module_events:
                continue

            # Resolves the numpy dtype string and extracts the raw data bytes for MODULE_DATA messages. Uses the
            # pre-built dtype lookup to avoid per-message prototype object allocation.
            dtype_str: str | None = None
            data_payload: bytes | None = None
            if protocol == SerialProtocols.MODULE_DATA:
                dtype_str = SerialPrototypes.get_dtype_for_code(code=int(payload[5]))
                if dtype_str is not None:
                    data_payload = payload[6:].tobytes()

            # Appends directly to the module's columnar accumulator.
            accumulator = module_data[current_module]
            accumulator.timestamps.append(int(log_msg.timestamp_us))
            accumulator.commands.append(int(payload[3]))
            accumulator.events.append(int(payload[4]))
            accumulator.dtypes.append(dtype_str)
            accumulator.data_payloads.append(data_payload)

        # Routes kernel messages (KERNEL_DATA / KERNEL_STATE) through the extraction pipeline.
        elif extract_kernel and protocol in kernel_protocols:
            # Extracts only messages with requested event codes.
            if int(payload[2]) not in kernel_event_codes:  # type: ignore[operator]
                continue

            # Resolves the numpy dtype string and extracts the raw data bytes for KERNEL_DATA messages.
            dtype_str = None
            data_payload = None
            if protocol == SerialProtocols.KERNEL_DATA:
                dtype_str = SerialPrototypes.get_dtype_for_code(code=int(payload[3]))
                if dtype_str is not None:
                    data_payload = payload[4:].tobytes()

            # Appends directly to the kernel's columnar accumulator.
            kernel_accumulator.timestamps.append(int(log_msg.timestamp_us))
            kernel_accumulator.commands.append(int(payload[1]))
            kernel_accumulator.events.append(int(payload[2]))
            kernel_accumulator.dtypes.append(dtype_str)
            kernel_accumulator.data_payloads.append(data_payload)

    return module_data, kernel_accumulator


def _extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """Extracts the first component from the end of each input path that uniquely identifies each path globally.

    Adapts the processing pipeline to directory structures where the unique recording identifier appears at different
    levels of the path hierarchy. For example, given paths like ``/data/day1/recording`` and ``/data/day2/recording``,
    identifies ``day1`` and ``day2`` as the unique components (not ``recording``, which is shared).

    Uses a frequency counter to count how many distinct paths contain each component, then selects the first
    component (from the end) that appears in exactly one path.

    Args:
        paths: The list or tuple of Path objects to extract unique components from.

    Returns:
        A tuple of unique component strings, one for each path, stored in the same order as the input paths.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    # Builds per-path component sets and a global frequency counter in a single pass. Each component is counted
    # once per path (not once per occurrence), so shared components get count == number of paths containing them.
    path_parts: list[tuple[str, ...]] = [path.parts for path in paths]
    component_path_count: dict[str, int] = {}
    for parts in path_parts:
        for component in set(parts):
            component_path_count[component] = component_path_count.get(component, 0) + 1

    # For each path, finds the first component from the end that appears in exactly one path.
    unique_components: list[str] = []
    for path, parts in zip(paths, path_parts, strict=True):
        found_unique = False
        for component in reversed(parts):
            if component_path_count[component] == 1:
                unique_components.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"Unable to extract a unique component from the given path: {path}."
            console.error(message=message, error=RuntimeError)

    return tuple(unique_components)


def generate_job_ids(source_ids: list[str]) -> dict[str, str]:
    """Generates unique processing job identifiers for each source ID.

    Args:
        source_ids: The list of source ID strings for which to generate job IDs.

    Returns:
        A dictionary mapping source IDs to their generated hexadecimal job identifiers.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=EXTRACTION_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }


def _finalize_accumulator(accumulator: _ColumnAccumulator) -> _ExtractedMessages:
    """Converts a growable column accumulator into a finalized _ExtractedMessages instance with numpy arrays.

    Args:
        accumulator: The column accumulator to finalize.

    Returns:
        An _ExtractedMessages instance with numpy arrays built from the accumulator's lists.
    """
    return _ExtractedMessages(
        timestamps=np.array(accumulator.timestamps, dtype=np.uint64),
        commands=np.array(accumulator.commands, dtype=np.uint8),
        events=np.array(accumulator.events, dtype=np.uint8),
        dtypes=tuple(accumulator.dtypes),
        data_payloads=tuple(accumulator.data_payloads),
    )


def _build_message_dataframe(messages: _ExtractedMessages) -> pl.DataFrame:
    """Builds a polars DataFrame directly from an _ExtractedMessages columnar structure.

    Passes pre-built numpy arrays to polars Series constructors with zero re-iteration. The ``dtype`` column stores
    the numpy dtype string for each message's data payload, enabling reconstruction via
    ``np.frombuffer(payload, dtype=dtype_str)`` without any library dependency.

    Args:
        messages: The columnar message data to serialize.

    Returns:
        A polars DataFrame with columns: timestamp_us (UInt64), command (UInt8), event (UInt8),
        dtype (String), and data (Binary).
    """
    return pl.DataFrame(
        {
            "timestamp_us": pl.Series(name="timestamp_us", values=messages.timestamps, dtype=pl.UInt64),
            "command": pl.Series(name="command", values=messages.commands, dtype=pl.UInt8),
            "event": pl.Series(name="event", values=messages.events, dtype=pl.UInt8),
            "dtype": pl.Series(name="dtype", values=list(messages.dtypes), dtype=pl.String),
            "data": pl.Series(name="data", values=list(messages.data_payloads), dtype=pl.Binary),
        }
    )


def _write_module_feather(
    module_data: _ExtractedModuleData,
    source_id: str,
    output_directory: Path,
) -> None:
    """Serializes extracted module data to a feather (IPC) file.

    Builds a polars DataFrame directly from the module's columnar message data and writes it to the output directory
    using the ``controller_{source_id}_module_{type}_{id}.feather`` naming convention.

    Args:
        module_data: The extracted data for a single hardware module.
        source_id: The source ID string identifying the originating log archive.
        output_directory: The path to the directory where the feather file is written.
    """
    dataframe = _build_message_dataframe(messages=module_data.messages)

    filename = (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}"
        f"{MODULE_FEATHER_INFIX}{module_data.module_type}_{module_data.module_id}"
        f"{FEATHER_SUFFIX}"
    )
    dataframe.write_ipc(file=output_directory / filename)


def _write_kernel_feather(
    kernel_data: _ExtractedMessages,
    source_id: str,
    output_directory: Path,
) -> None:
    """Serializes extracted kernel messages to a feather (IPC) file.

    Builds a polars DataFrame directly from the kernel's columnar message data and writes it to the output directory
    using the ``controller_{source_id}_kernel.feather`` naming convention.

    Args:
        kernel_data: The extracted kernel messages in columnar form.
        source_id: The source ID string identifying the originating log archive.
        output_directory: The path to the directory where the feather file is written.
    """
    dataframe = _build_message_dataframe(messages=kernel_data)

    filename = f"{CONTROLLER_FEATHER_PREFIX}{source_id}{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}"
    dataframe.write_ipc(file=output_directory / filename)
