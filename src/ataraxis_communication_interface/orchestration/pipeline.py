"""Provides the single-job runner every scheduler dispatches and the local pipeline entry point that runs a whole
DataLogger output directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import (
    ProcessingTracker,
    atomic_write,
    find_log_archive,
    find_log_archives,
)

from .jobs import (
    TRACKER_FILENAME,
    EXTRACTION_JOB_NAME,
    MICROCONTROLLER_DATA_DIRECTORY,
    generate_job_ids,
    resolve_kernel_feather_path,
    resolve_module_feather_path,
    discover_microcontroller_jobs,
)
from .allocation import resolve_core_budget, resolve_job_workers, resolve_archive_footprint
from ..microcontroller import ExtractionConfig, build_message_dataframe, extract_logged_microcontroller_data

if TYPE_CHECKING:
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor

    from ..microcontroller import ExtractedMessages, ControllerExtractionConfig


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

    Reads the archive once, routes each incoming message through the event code filters the controller configuration
    declares, and writes one feather (Arrow IPC) file per module that produced data, plus one holding the kernel
    messages when kernel extraction is configured.

    Notes:
        Delegates the job's state transitions to the tracker's run_job() context manager, which marks the job as
        running, completes it when the block returns, and marks it as failed with the exception's message before
        re-raising when the block raises an Exception.

        Writes the feather files directly into the output directory, creating it when it does not exist, and
        registers no job on the tracker, so a scheduler owning its own tracker and output layout dispatches this
        function unchanged.

        Publishes each feather file through a temporary file and a rename, so a job killed mid-write leaves the
        previously written file intact rather than a truncated one a reader cannot decode.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output feather files are written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        workers: The number of worker processes to use for parallel processing.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        controller_config: The extraction configuration for the controller whose archive is being processed.
        display_progress: Determines whether to display a progress bar during extraction.
        executor: When provided, parallel processing reuses this pool instead of creating a new one. The pool is
            passed through to extract_logged_microcontroller_data to avoid spawning a redundant process pool.

    Raises:
        ValueError: If the controller config declares a module or a kernel entry with empty event codes, or if it
            declares no extraction targets at all.
    """
    module_filters, kernel_event_codes = _resolve_event_filters(
        controller_config=controller_config, source_id=source_id
    )

    console.echo(message=f"Running '{EXTRACTION_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")

    with tracker.run_job(job_id=job_id):
        extracted = extract_logged_microcontroller_data(
            log_path=log_path,
            module_filters=module_filters,
            kernel_event_codes=kernel_event_codes,
            workers=workers,
            display_progress=display_progress,
            executor=executor,
        )

        for module in extracted.modules:
            _write_messages(
                messages=module.messages,
                file_path=resolve_module_feather_path(
                    output_directory=output_directory,
                    source_id=source_id,
                    module_type=module.module_type,
                    module_id=module.module_id,
                ),
            )

        if extracted.kernel.count:
            _write_messages(
                messages=extracted.kernel,
                file_path=resolve_kernel_feather_path(output_directory=output_directory, source_id=source_id),
            )


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

    Extracts hardware module and kernel message data as the extraction configuration specifies and writes the results
    to feather (Arrow IPC) files. The controller IDs to process are resolved from the extraction configuration and
    validated against the microcontroller manifest, which confirms the archives were produced by
    ataraxis-communication-interface.

    Supports both local and remote processing modes. In local mode (job_id is None), resolves each requested log
    archive by controller ID, aligns a processing tracker in the output directory with the requested jobs, and
    executes them sequentially. In remote mode (job_id is provided), aligns the tracker with the full job universe
    the manifest defines, then resolves and executes only the single archive matching the requested job ID. The
    universe alignment lets independent remote jobs share one tracker without resetting each other's state, which
    supports running every controller in parallel under an external scheduler.

    In local mode, all resolved archives must reside in the same directory. If the log_directory contains archives
    from multiple DataLogger instances (in separate subdirectories), each must be processed independently. Use the
    MCP batch processing tools to orchestrate multi-directory workflows.

    Notes:
        Every job's width is resolved from the archive it reads, so a run mixing a long recording with a short one
        gives each the cores its own archive repays rather than one width chosen for the whole run.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``microcontroller_data/`` subdirectory is created
            automatically under this path, and all tracker and output files are written there.
        config: The path to the extraction configuration .yaml file specifying which controllers, modules, and events
            to extract. Controller IDs in the config determine which archives are processed.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all configured jobs are run sequentially
            with automatic tracker management (local mode).
        workers: The ceiling on the cores any single job receives. Setting this to a value less than 1 resolves the
            ceiling from the host's core count. Setting this to 1 conducts every job sequentially.
        display_progress: Determines whether to display progress bars during extraction. Defaults to True for
            interactive CLI use. Set to False for MCP batch processing.

    Raises:
        FileNotFoundError: If the log_directory does not exist, the config path does not exist, a controller ID has
            no matching archive, or no microcontroller manifest is found.
        OSError: If any directory beneath the log_directory cannot be read.
        ValueError: If the provided job_id does not match any job in the manifest universe, if controller IDs are not
            registered in the microcontroller manifest, if resolved archives span multiple directories, or if the
            microcontroller manifest registers no controllers.
    """
    if not config.is_file():
        message = f"Unable to load the extraction config from '{config}'. The path does not exist or is not a file."
        console.error(message=message, error=FileNotFoundError)

    # Builds the universe of every job the manifest could produce: one extraction job per registered controller ID.
    # The universe is a manifest fingerprint, not an invocation fingerprint, so every invocation (full, subset, or
    # single remote job) aligns the tracker against the same set and never resets sibling jobs.
    universe, _ = discover_microcontroller_jobs(log_directory=log_directory)
    universe_ids = [specifier for _, specifier in universe]

    # Creates the microcontroller_data subdirectory under the output path. All tracker and feather files go here.
    data_path = output_directory / MICROCONTROLLER_DATA_DIRECTORY
    data_path.mkdir(parents=True, exist_ok=True)

    tracker = ProcessingTracker(file_path=data_path / TRACKER_FILENAME)

    # Bounds every job resolved below, so no job is dispatched at a width the host cannot supply.
    ceiling = resolve_core_budget(requested_budget=workers)

    controller_configs = _resolve_controller_configs(config=config, universe_ids=universe_ids)

    if job_id is not None:
        # Remote mode: selects the job to run solely by ID, validated against the manifest universe. Aligns the
        # tracker with the full universe so start_job finds the requested ID and concurrent remote jobs do not
        # treat each other's entries as foreign. Resolves only the matched archive so a missing or late sibling
        # archive cannot fail this job.
        _, source_id = tracker.resolve_job(job_id=job_id, universe=universe)

        if source_id not in controller_configs:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The extraction config at '{config}' "
                f"declares no entry for the controller with ID '{source_id}'. Configured controller IDs: "
                f"{sorted(controller_configs)}."
            )
            console.error(message=message, error=ValueError)

        tracker.align_jobs(jobs=universe, universe=universe)

        _execute_sized_job(
            log_path=find_log_archive(log_directory=log_directory, source_id=source_id),
            output_directory=data_path,
            source_id=source_id,
            job_id=job_id,
            ceiling=ceiling,
            tracker=tracker,
            controller_config=controller_configs[source_id],
            display_progress=display_progress,
        )
    else:
        source_ids = sorted(controller_configs)
        console.echo(message=f"Resolved {len(source_ids)} controller ID(s) from config: {', '.join(source_ids)}")

        # Resolves all requested archive paths in one pass and validates they belong to the same DataLogger directory.
        archive_paths = find_log_archives(log_directory=log_directory, source_ids=source_ids)
        parent_directories = {path.parent for path in archive_paths.values()}
        if len(parent_directories) > 1:
            message = (
                f"Unable to process logs in '{log_directory}'. The requested log archives span multiple "
                f"directories: {sorted(str(parent) for parent in parent_directories)}. Each DataLogger output "
                f"directory must be processed independently."
            )
            console.error(message=message, error=ValueError)

        # Aligns the tracker with the requested subset while detecting foreign entries against the full universe.
        jobs: list[tuple[str, str]] = [(EXTRACTION_JOB_NAME, source_id) for source_id in source_ids]
        tracker.align_jobs(jobs=jobs, universe=universe)

        job_ids = generate_job_ids(source_ids=source_ids)

        for source_id in source_ids:
            _execute_sized_job(
                log_path=archive_paths[source_id],
                output_directory=data_path,
                source_id=source_id,
                job_id=job_ids[source_id],
                ceiling=ceiling,
                tracker=tracker,
                controller_config=controller_configs[source_id],
                display_progress=display_progress,
            )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)


def _resolve_controller_configs(config: Path, universe_ids: list[str]) -> dict[str, ControllerExtractionConfig]:
    """Reads the extraction configuration and validates its controllers against the manifest job universe.

    Args:
        config: The path to the extraction configuration .yaml file.
        universe_ids: The controller source identifiers the microcontroller manifest registers.

    Returns:
        The extraction configuration of each configured controller, keyed by that controller's source identifier.

    Raises:
        ValueError: If the configuration declares no controllers, or if it declares a controller the manifest does
            not register.
    """
    resolved_config = ExtractionConfig.from_yaml(file_path=config)
    controller_configs = {str(controller.controller_id): controller for controller in resolved_config.controllers}

    if not controller_configs:
        message = f"Unable to process logs using the extraction config at '{config}'. It declares no controllers."
        console.error(message=message, error=ValueError)

    unregistered_ids = sorted(set(controller_configs) - set(universe_ids))
    if unregistered_ids:
        message = (
            f"Unable to process logs using the extraction config at '{config}'. The following controller IDs are "
            f"not registered in the microcontroller manifest: {', '.join(unregistered_ids)}. The corresponding log "
            f"archives were not produced by ataraxis-communication-interface. Registered IDs: {sorted(universe_ids)}."
        )
        console.error(message=message, error=ValueError)

    return controller_configs


def _resolve_event_filters(
    controller_config: ControllerExtractionConfig,
    source_id: str,
) -> tuple[dict[tuple[int, int], frozenset[int]] | None, frozenset[int] | None]:
    """Unpacks a controller's extraction configuration into the per-module and kernel event code filters.

    Notes:
        Each module maps to its own frozenset of event codes, which prevents off-target extraction across modules
        that share the same controller and reuse an event code with different semantics.

    Args:
        controller_config: The extraction configuration of the controller whose archive is processed.
        source_id: The source ID string identifying the log archive, used to attribute a configuration error.

    Returns:
        The per-module event code filters keyed by the module type and identifier pair, and the kernel event code
        filter, with either replaced by None when the configuration requests no extraction for that target.

    Raises:
        ValueError: If a configured module or the kernel declares empty event codes, or if the configuration
            declares no extraction targets at all.
    """
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
                f"Unable to execute the data extraction job for source '{source_id}'. Kernel extraction has empty "
                f"event_codes."
            )
            console.error(message=message, error=ValueError)
        kernel_event_codes = frozenset(controller_config.kernel.event_codes)

    if module_filters is None and kernel_event_codes is None:
        message = (
            f"Unable to execute the data extraction job for source '{source_id}'. The controller config has no "
            f"modules and no kernel extraction configured."
        )
        console.error(message=message, error=ValueError)

    return module_filters, kernel_event_codes


def _execute_sized_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    ceiling: int,
    tracker: ProcessingTracker,
    controller_config: ControllerExtractionConfig,
    *,
    display_progress: bool,
) -> None:
    """Sizes one job from the archive it reads and executes it at the resolved width.

    Args:
        log_path: The path to the .npz log archive to process.
        output_directory: The path to the directory where the output feather files are written.
        source_id: The source ID string identifying the log archive.
        job_id: The unique hexadecimal identifier for this processing job.
        ceiling: The cores available to this job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        controller_config: The extraction configuration for the controller whose archive is being processed.
        display_progress: Determines whether to display a progress bar during extraction.
    """
    job_workers = resolve_job_workers(footprint=resolve_archive_footprint(archive_path=log_path), ceiling=ceiling)

    execute_job(
        log_path=log_path,
        output_directory=output_directory,
        source_id=source_id,
        job_id=job_id,
        workers=job_workers,
        tracker=tracker,
        controller_config=controller_config,
        display_progress=display_progress,
    )


def _write_messages(messages: ExtractedMessages, file_path: Path) -> None:
    """Serializes an extracted columnar message block to a feather (Arrow IPC) file.

    Args:
        messages: The columnar message data to serialize.
        file_path: The path to the feather file to write.
    """
    dataframe = build_message_dataframe(messages=messages)
    with atomic_write(file_path=file_path, binary=True) as feather_file:
        dataframe.write_ipc(file=feather_file)
