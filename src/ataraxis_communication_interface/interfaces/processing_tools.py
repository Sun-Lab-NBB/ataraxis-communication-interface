"""Provides MCP tools for preparing, executing, monitoring, canceling, and resetting batch log processing jobs."""

from __future__ import annotations

from typing import Any
from pathlib import Path
from threading import Thread

import numpy as np
from ataraxis_time import TimeUnits, TimestampFormats, TimestampPrecisions, convert_time, get_timestamp
from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .mcp_instance import mcp, read_tracker_status
from .mcp_execution import (
    PendingJob,
    JobExecutionState,
    get_execution_state,
    set_execution_state,
    job_execution_manager,
)
from ..microcontroller.dataclasses import ExtractionConfig
from ..microcontroller.log_processing import (
    TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX,
    EXTRACTION_JOB_NAME,
    MICROCONTROLLER_DATA_DIRECTORY,
    prepare_tracker,
    generate_job_ids,
)

_RESERVED_CORES: int = 2
"""The number of CPU cores reserved for system operations. The worker budget is computed as available cores minus this
value, with a minimum of 1."""


@mcp.tool()
def prepare_log_processing_batch_tool(
    log_directories: list[str],
    source_ids: list[str],
    output_directories: list[str],
    config_path: str,
) -> dict[str, Any]:
    """Prepares an execution manifest for batch log processing without starting execution.

    Accepts log directories, source IDs, output directories, and an extraction configuration path from the caller
    and initializes a ProcessingTracker with one data-extraction job per source ID for each log directory. The
    configuration path is validated up front and embedded in every job descriptor so that downstream execution
    tools receive a self-contained manifest. Idempotent: if a tracker already exists for a log directory, returns
    the existing manifest with current job statuses instead of reinitializing. Requires prior discovery -- the
    caller must provide confirmed source IDs rather than relying on implicit archive or manifest discovery.

    Important:
        The AI agent calling this tool MUST run discover_microcontroller_data_tool first to obtain log directory
        paths and confirmed source IDs. The agent MUST ask the user for the output directory paths and extraction
        configuration path before calling this tool. Do not assume or guess directory paths or source IDs.

    Args:
        log_directories: The list of absolute paths to DataLogger output directories containing log archives.
            Accepts paths from the 'log_directories' list returned by discover_microcontroller_data_tool.
        source_ids: The list of confirmed source IDs to process. Accepts IDs from the 'source_id' field of
            entries in the 'sources' list returned by discover_microcontroller_data_tool. Applied uniformly: each
            log directory creates jobs for every source ID in this list that has a matching archive on disk.
        output_directories: The list of absolute paths for per-log-directory output. Must match the length of
            log_directories. Each output directory receives a ``microcontroller_data/`` subdirectory containing
            the processing tracker and output files.
        config_path: The absolute path to the ExtractionConfig YAML file that specifies which events to extract
            for each controller. Validated before batch preparation and embedded in every job descriptor.

    Returns:
        A dictionary containing per-log-directory manifests in 'log_directories' with tracker paths and job
        lists, total counts, and any invalid paths.
    """
    config_file = Path(config_path)
    if not config_file.exists() or not config_file.is_file():
        return {"error": f"Extraction config not found: {config_path}"}

    try:
        ExtractionConfig.load(file_path=config_file)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Invalid extraction config: {error}"}

    if len(output_directories) != len(log_directories):
        return {
            "error": (
                f"Length mismatch: {len(log_directories)} log directories but "
                f"{len(output_directories)} output directories."
            ),
        }

    source_id_set = set(source_ids)
    result_log_directories: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for entry_index, log_directory_string in enumerate(log_directories):
        log_directory_path = Path(log_directory_string)

        if not log_directory_path.exists() or not log_directory_path.is_dir():
            invalid_paths.append(log_directory_string)
            continue

        # Filters the requested source IDs to those that have a matching archive in this log directory.
        # Discovery already confirmed these archives exist, but the check guards against stale data.
        filtered_ids = sorted(
            source_id
            for source_id in source_id_set
            if (log_directory_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}").exists()
        )

        if not filtered_ids:
            result_log_directories[log_directory_string] = {
                "source_ids": [],
                "jobs": [],
                "tracker_path": None,
                "summary": {},
            }
            continue

        output_path = Path(output_directories[entry_index])

        data_path = output_path / MICROCONTROLLER_DATA_DIRECTORY
        data_path.mkdir(parents=True, exist_ok=True)
        tracker_path = data_path / TRACKER_FILENAME

        if tracker_path.exists():
            # Idempotent path: returns existing tracker state.
            try:
                tracker_status = read_tracker_status(tracker_path=tracker_path)
            except Exception:  # noqa: BLE001
                tracker_status = {"jobs": [], "summary": {}}

            result_log_directories[log_directory_string] = {
                "tracker_path": str(tracker_path),
                "output_directory": str(data_path),
                "source_ids": filtered_ids,
                **tracker_status,
            }
            total_jobs += len(tracker_status.get("jobs", []))
        else:
            # Initializes a new tracker with jobs for the filtered source IDs. Uses prepare_tracker so the MCP
            # batch-preparation path inherits the same regeneration logic as run_log_processing_pipeline.
            tracker = ProcessingTracker(file_path=tracker_path)
            tracker_jobs: list[tuple[str, str]] = [(EXTRACTION_JOB_NAME, source_id) for source_id in filtered_ids]
            prepare_tracker(tracker=tracker, jobs=tracker_jobs, universe=tracker_jobs)
            job_ids = generate_job_ids(source_ids=filtered_ids)

            jobs: list[dict[str, str]] = [
                {
                    "job_id": job_ids[source_id],
                    "source_id": source_id,
                    "status": "SCHEDULED",
                    "log_directory": log_directory_string,
                    "output_directory": str(data_path),
                    "tracker_path": str(tracker_path),
                    "config_path": config_path,
                }
                for source_id in filtered_ids
            ]

            result_log_directories[log_directory_string] = {
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
        "log_directories": result_log_directories,
        "total_log_directories": len(result_log_directories),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def execute_log_processing_jobs_tool(
    jobs: list[dict[str, str]],
    *,
    worker_budget: int = -1,
) -> dict[str, Any]:
    """Dispatches log processing jobs for background execution with budget-based worker allocation.

    Takes job descriptors from the manifest produced by prepare_log_processing_batch_tool and starts a background
    execution manager that allocates CPU cores to each job based on its archive size. Each job descriptor must
    include its own 'config_path' key pointing to the ExtractionConfig YAML file for that job. The worker budget
    directly controls memory footprint since each worker spawns a separate process. Large archives (>= 2000
    messages) receive more workers, while small archives receive 1 worker since they process sequentially
    regardless. The manager fills available budget greedily, dispatching smaller jobs alongside large ones when
    cores are available.

    Important:
        Only one execution session can be active at a time. Use cancel_log_processing_tool to cancel an active
        session before starting a new one.

    Args:
        jobs: The list of job descriptors from prepare_log_processing_batch_tool. Each dictionary must have
            'log_directory', 'output_directory', 'tracker_path', 'job_id', 'source_id', and 'config_path' keys.
        worker_budget: The total number of CPU cores available for the execution session. Directly controls memory
            footprint. Set to -1 for automatic resolution via resolve_worker_count.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', resolved worker budget, per-job message counts,
        and any invalid jobs.
    """
    # Enforces single-session constraint.
    active_state = get_execution_state()
    if active_state is not None and active_state.manager_thread is not None and active_state.manager_thread.is_alive():
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    # Validates and builds pending jobs.
    required_keys = {"log_directory", "output_directory", "tracker_path", "job_id", "source_id", "config_path"}
    pending: list[PendingJob] = []
    all_jobs: dict[tuple[str, str], PendingJob] = {}
    invalid_jobs: list[dict[str, str]] = []

    for job_dict in jobs:
        if not required_keys.issubset(job_dict.keys()):
            invalid_jobs.append({**job_dict, "error": f"Missing required keys: {required_keys - job_dict.keys()}"})
            continue

        tracker_path = Path(job_dict["tracker_path"])
        if not tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {job_dict['tracker_path']}"})
            continue

        pending_job = PendingJob(
            log_directory=Path(job_dict["log_directory"]),
            output_directory=Path(job_dict["output_directory"]),
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            source_id=job_dict["source_id"],
            config_path=Path(job_dict["config_path"]),
        )
        pending.append(pending_job)
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    resolved_budget = resolve_worker_count(requested_workers=worker_budget, reserved_cores=_RESERVED_CORES)

    # Probes archive message counts for all pending jobs. Reads only the zip directory of each .npz file,
    # avoiding loading message data into memory.
    job_message_counts: dict[tuple[str, str], int] = {}
    for job in pending:
        job_message_counts[job.dispatch_key] = _probe_archive_message_count(job=job)

    execution_state = JobExecutionState(
        all_jobs=all_jobs,
        pending_queue=pending,
        job_message_counts=job_message_counts,
        worker_budget=resolved_budget,
    )
    set_execution_state(execution_state)

    manager = Thread(target=job_execution_manager, daemon=True)
    manager.start()
    execution_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "worker_budget": resolved_budget,
        "job_message_counts": job_message_counts,
    }

    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs

    return result


@mcp.tool()
def get_log_processing_status_tool() -> dict[str, Any]:
    """Returns the current status of the active log processing execution session.

    Reads ProcessingTracker files from disk for each job to report per-job progress. If no execution session
    exists, returns an inactive status.

    Returns:
        A dictionary containing an 'active' flag, a 'canceled' flag, per-job status entries in 'jobs', and a
        'summary' with counts for scheduled, running, succeeded, and failed jobs.
    """
    state = get_execution_state()
    if state is None:
        return {"active": False, "message": "No execution session exists."}

    # Checks whether the background execution manager thread is still running.
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
                if job_state.executor_id is not None:
                    entry["executor_id"] = job_state.executor_id
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


@mcp.tool()
def get_log_processing_timing_tool() -> dict[str, Any]:
    """Returns timing information for all jobs in the active execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps from ProcessingTracker.

    Returns:
        A dictionary containing an 'active' flag, per-job timing in 'jobs', and a 'session' summary with total
        elapsed seconds and completed, failed, running, and pending counts. The session also includes a throughput in
        jobs per hour once at least one job has completed.
    """
    state = get_execution_state()
    if state is None:
        return {"active": False, "message": "No execution session exists."}

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    # Captures the current timestamp for computing elapsed time on running jobs.
    current_us = int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))

    # Collects per-job timing entries and tracks the earliest start for session-level statistics.
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

            if job_info.executor_id is not None:
                entry["executor_id"] = job_info.executor_id

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
                entry["elapsed_seconds"] = round(number=elapsed_seconds, ndigits=2)

            if job_info.completed_at is not None:
                entry["completed_at"] = int(job_info.completed_at)
                if job_info.started_at is not None:
                    duration_seconds = convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    )
                    entry["duration_seconds"] = round(number=duration_seconds, ndigits=2)

            if job_info.status == ProcessingStatus.SUCCEEDED:
                completed_count += 1
            elif job_info.status == ProcessingStatus.FAILED:
                failed_count += 1

            job_timing.append(entry)

    # Computes session-level statistics.
    total_elapsed_seconds = 0.0
    if earliest_start is not None:
        total_elapsed_seconds = round(
            number=convert_time(
                time=current_us - earliest_start,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.SECOND,
                as_float=True,
            ),
            ndigits=2,
        )

    running_count = sum(1 for job_entry in job_timing if "elapsed_seconds" in job_entry)
    session: dict[str, Any] = {
        "total_elapsed_seconds": total_elapsed_seconds,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "pending_count": len(state.all_jobs) - completed_count - failed_count - running_count,
    }

    if completed_count and earliest_start is not None:
        elapsed_hours = convert_time(
            time=current_us - earliest_start,
            from_units=TimeUnits.MICROSECOND,
            to_units=TimeUnits.HOUR,
            as_float=True,
        )
        if elapsed_hours > 0:
            session["throughput_jobs_per_hour"] = round(number=completed_count / elapsed_hours, ndigits=2)

    return {"active": manager_alive, "jobs": job_timing, "session": session}


@mcp.tool()
def cancel_log_processing_tool() -> dict[str, Any]:
    """Cancels the active log processing execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a 'canceled' flag, a 'message', and 'final_state' with counts for succeeded,
        failed, and active jobs at the time of cancellation.
    """
    state = get_execution_state()
    if state is None:
        return {"canceled": False, "message": "No execution session is active."}

    # Sets the canceled flag and clears pending jobs under the lock. Active groups complete naturally.
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


@mcp.tool()
def reset_log_processing_jobs_tool(
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

    reset_tracker = ProcessingTracker(file_path=path)
    reset_tracker.initialize_jobs(jobs=reset_jobs)

    # Reads back the updated state for the response.
    try:
        updated_status = read_tracker_status(tracker_path=path)
    except Exception:  # noqa: BLE001
        updated_status = {"jobs": [], "summary": {}}

    return {"reset": True, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]:
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

    # Discovers all tracker files recursively and aggregates their job statuses.
    log_dir_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    for found_tracker_path in sorted(root_path.rglob(TRACKER_FILENAME)):
        log_directory = str(found_tracker_path.parent)
        try:
            status = read_tracker_status(tracker_path=found_tracker_path)
            summary = status.get("summary", {})

            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            dir_status = _derive_tracker_status(summary=summary)

            log_dir_statuses.append(
                {
                    "log_directory": log_directory,
                    "tracker_path": str(found_tracker_path),
                    "status": dir_status,
                    **status,
                }
            )
        except Exception:  # noqa: BLE001
            log_dir_statuses.append(
                {
                    "log_directory": log_directory,
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


def _derive_tracker_status(summary: dict[str, Any]) -> str:
    """Derives a high-level processing status label from a tracker summary's job counts.

    Applies a fixed priority: ``failed`` if any job failed, ``completed`` if all succeeded, ``processing`` if any
    are running, ``not_started`` if all are scheduled, and ``in_progress`` otherwise.

    Args:
        summary: A dictionary containing 'total', 'succeeded', 'failed', 'running', and 'scheduled' counts.

    Returns:
        A status string: one of 'failed', 'completed', 'processing', 'not_started', or 'in_progress'.
    """
    total = summary.get("total", 0)
    if summary.get("failed", 0):
        return "failed"
    if summary.get("succeeded", 0) == total and total > 0:
        return "completed"
    if summary.get("running", 0):
        return "processing"
    if summary.get("scheduled", 0) == total and total > 0:
        return "not_started"
    return "in_progress"


def _group_jobs_by_tracker(state: JobExecutionState) -> dict[Path, list[PendingJob]]:
    """Groups all jobs in an execution state by their tracker file path.

    Minimizes redundant file reads by batching jobs that share the same tracker, so each tracker YAML file is
    deserialized only once when iterating over the groups.

    Args:
        state: The active job execution state containing the job registry.

    Returns:
        A dictionary mapping each tracker path to its list of pending job descriptors.
    """
    tracker_jobs: dict[Path, list[PendingJob]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)
    return tracker_jobs


def _probe_archive_message_count(job: PendingJob) -> int:
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
