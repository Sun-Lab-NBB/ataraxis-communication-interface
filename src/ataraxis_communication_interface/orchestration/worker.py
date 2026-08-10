"""Provides the single-job runner every scheduler dispatches and the picklable descriptor-addressed entry point a
process pool submits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingTracker, atomic_write

from .jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    resolve_kernel_path,
    resolve_module_path,
)
from ..microcontroller import ExtractionConfig, build_message_dataframe, extract_logged_microcontroller_data

if TYPE_CHECKING:
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor

    from .jobs import JobDescriptor
    from ..microcontroller import ExtractedMessages, ControllerExtractionConfig


def run_extraction_job(job: JobDescriptor) -> None:
    """Runs one microcontroller data extraction job described entirely by its descriptor.

    Notes:
        This is the picklable entry point a process pool submits. It takes one flat descriptor, so the only state
        crossing the process boundary is paths, strings, and integers.

        Opens the tracker from the descriptor's own path, because a tracker holds a file lock that cannot cross a
        process boundary.

    Args:
        job: The descriptor of the job to run.
    """
    execute_job(
        log_path=job.archive_path,
        output_directory=job.output_directory,
        source_id=job.source_id,
        job_id=job.job_id,
        workers=job.core_weight,
        tracker=ProcessingTracker(file_path=job.tracker_path),
        config_path=job.config_path,
        display_progress=False,
    )


def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    config_path: Path,
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
        re-raising when the block raises an Exception. The configuration is read inside that context, so a
        configuration error is recorded against the job rather than escaping it.

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
        config_path: The path to the ExtractionConfig .yaml file declaring this controller's extraction targets.
        display_progress: Determines whether to display a progress bar during extraction.
        executor: When provided, parallel processing reuses this pool instead of creating a new one. The pool is
            passed through to extract_logged_microcontroller_data to avoid spawning a redundant process pool.

    Raises:
        ValueError: If the configuration declares no entry for this controller, if it declares a module or a kernel
            entry with empty event codes, or if it declares no extraction targets at all.
    """
    console.echo(message=f"Running '{CONTROLLER_EXTRACTION_JOB_NAME}' job for source '{source_id}' (ID: {job_id})...")

    with tracker.run_job(job_id=job_id):
        controller_config = resolve_controller_config(config_path=config_path, source_id=source_id)
        module_filters, kernel_event_codes = _resolve_event_filters(
            controller_config=controller_config, source_id=source_id
        )

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
                file_path=resolve_module_path(
                    output_directory=output_directory,
                    source_id=source_id,
                    module_type=module.module_type,
                    module_id=module.module_id,
                ),
            )

        if extracted.kernel.count:
            _write_messages(
                messages=extracted.kernel,
                file_path=resolve_kernel_path(output_directory=output_directory, source_id=source_id),
            )


def resolve_controller_config(config_path: Path, source_id: str) -> ControllerExtractionConfig:
    """Reads the extraction configuration and returns the entry declaring the target controller's targets.

    Args:
        config_path: The path to the extraction configuration .yaml file.
        source_id: The identifier of the controller whose entry is read.

    Returns:
        The extraction configuration of the requested controller.

    Raises:
        ValueError: If the configuration declares no entry for the requested controller.
    """
    resolved_config = ExtractionConfig.from_yaml(file_path=config_path)
    controller_configs = {str(controller.controller_id): controller for controller in resolved_config.controllers}

    if source_id not in controller_configs:
        message = (
            f"Unable to execute the data extraction job for source '{source_id}'. The extraction config at "
            f"'{config_path}' declares no entry for that controller. Configured controller IDs: "
            f"{', '.join(sorted(controller_configs))}."
        )
        console.error(message=message, error=ValueError)

    return controller_configs[source_id]


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


def _write_messages(messages: ExtractedMessages, file_path: Path) -> None:
    """Serializes an extracted columnar message block to a feather (Arrow IPC) file.

    Args:
        messages: The columnar message data to serialize.
        file_path: The path to the feather file to write.
    """
    dataframe = build_message_dataframe(messages=messages)
    with atomic_write(file_path=file_path, binary=True) as feather_file:
        dataframe.write_ipc(file=feather_file)
