"""Provides the sequential processing pipeline that runs the microcontroller data extraction jobs of one recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker

from .worker import execute_job
from .discovery import prepare_jobs
from .allocation import resolve_job_workers, resolve_archive_footprint

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None = None,
    source_ids: Sequence[str] | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None:
    """Processes the requested MicroControllerInterface log archives from a single DataLogger output directory.

    Extracts hardware module and kernel message data as the extraction configuration specifies and writes the results
    to feather (Arrow IPC) files. The controller IDs to process are resolved from the extraction configuration and
    validated against the microcontroller manifest, which confirms the archives were produced by
    ataraxis-communication-interface.

    Supports both local and external processing modes. In local mode (job_id is None), resolves each requested log
    archive by controller ID, aligns a processing tracker in the output directory, and executes the jobs sequentially.
    In external mode (job_id is provided), resolves and executes only the single archive matching the requested ID.

    Notes:
        Serves the command-line interface and any external driver that dispatches one job by its identifier. The module
        defining this pipeline imports no batch engine, and batch orchestration across many recordings belongs to the
        MCP server.

        The tracker is aligned against the full job universe the microcontroller manifest defines in both modes, which
        lets independent external jobs share one tracker without resetting each other's state. That is what supports
        running every controller of one recording in parallel under an external scheduler.

        Each job runs at the width the caller named, or at the width its own archive resolves to when the caller
        named none. Jobs run one at a time, so this path weighs nothing against a core or a memory budget.

    Args:
        log_directory: The path to the root directory to search for .npz log archives. The directory is searched
            recursively, so archives may be nested at any depth below this path.
        output_directory: The path to the root output directory. A ``microcontroller_data/`` subdirectory is created
            automatically under this path, and all tracker and output files are written there.
        config: The path to the extraction configuration .yaml file specifying which controllers, modules, and events
            to extract. Controller IDs in the config determine which archives are processed.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (external mode). If not provided, all configured jobs are run sequentially
            with automatic tracker management (local mode).
        source_ids: The controller IDs to process in local mode. Each ID must be declared in the extraction
            configuration and resolve to exactly one archive. If not provided, processes every configured controller.
            This argument is ignored in external mode.
        workers: The workers every job receives. A positive value is used verbatim. A non-positive value resolves the
            width from each archive, which is one worker below the parallel extraction threshold and the declared
            per-job allocation above it.
        display_progress: Determines whether to display progress bars during parallel batch processing. A job that
            runs sequentially displays nothing.

    Raises:
        FileNotFoundError: If the log directory or the configuration file does not exist, if a requested controller's
            archive is absent, or if the recording resolves no job to run.
        ValueError: If the tree holds more than one microcontroller manifest, if a manifest registers no controllers,
            if the configuration declares no controllers, if a requested controller or job identifier is not
            registered, or if the resolved archives span several directories. Also raised once a job runs, if a
            configured module or the kernel declares empty event codes, if a controller declares no extraction
            targets, if the archive carries no onset timestamp message, or if a logged data message's payload size
            disagrees with its prototype code.
        OSError: If any directory beneath the log directory cannot be read.
        TimeoutError: If the processing tracker's lock cannot be acquired, which a batch running concurrently over
            the same output directory can cause.
    """
    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_directory,
        config_path=config,
        source_ids=source_ids,
        job_id=job_id,
    )

    # A caller reaching this function asked for work to be carried out, so resolving nothing is a failure here even
    # though the resolution itself reports a recording holding no microcontroller data as an ordinary answer.
    if not job_set.jobs:
        message = (
            f"Unable to process microcontroller log archives in '{log_directory}'. The recording resolved no "
            f"extraction job. Its tree holds no microcontroller manifest, or the extraction config declares no "
            f"controller whose log archive resolves to exactly one file beneath it."
        )
        console.error(message=message, error=FileNotFoundError)

    console.echo(
        message=(
            f"Resolved {len(job_set.jobs)} job(s) for controller ID(s): "
            f"{', '.join(job.source_id for job in job_set.jobs)}"
        )
    )

    tracker = ProcessingTracker(file_path=job_set.tracker_path)

    for job in job_set.jobs:
        # An unset width is the one choice the caller left open, so it is resolved from each archive in turn.
        job_workers = (
            workers
            if workers > 0
            else resolve_job_workers(footprint=resolve_archive_footprint(archive_path=job.archive_path))
        )
        execute_job(
            log_path=job.archive_path,
            output_directory=job.output_directory,
            source_id=job.source_id,
            job_id=job.job_id,
            workers=job_workers,
            tracker=tracker,
            config_path=job.config_path,
            display_progress=display_progress,
        )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)
