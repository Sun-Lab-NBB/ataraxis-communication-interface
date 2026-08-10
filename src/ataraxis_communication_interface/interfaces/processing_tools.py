"""Provides MCP tools for preparing, executing, monitoring, canceling, and resetting batch log processing jobs."""

from __future__ import annotations

from typing import Any
from pathlib import Path
from dataclasses import replace

from ataraxis_time import TimeUnits, TimestampFormats, TimestampPrecisions, convert_time, get_timestamp
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, discover_marker_files

from .mcp_instance import mcp, read_tracker_status
from ..orchestration import (
    JobSizing,
    OutputLayout,
    JobDescriptor,
    ArchiveFootprint,
    JobExecutionState,
    size_job,
    prepare_jobs,
    resolve_pool_size,
    get_execution_state,
    resolve_core_budget,
    resolve_job_workers,
    group_jobs_by_tracker,
    estimate_job_memory_mb,
    start_execution_session,
    resolve_memory_budget_mb,
)
from ..microcontroller import ExtractionConfig


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
            log directory creates a job for every source ID in this list that both has a matching archive on disk
            and is declared in the extraction configuration. A source failing either condition is reported under
            that directory's 'skipped_sources'.
        output_directories: The list of absolute paths for per-log-directory output. Must match the length of
            log_directories. Each output directory receives a ``microcontroller_data/`` subdirectory containing
            the processing tracker and output files.
        config_path: The absolute path to the ExtractionConfig YAML file that specifies which events to extract
            for each controller. Validated before batch preparation and embedded in every job descriptor.

    Returns:
        A dictionary containing a 'success' flag and per-log-directory manifests in 'log_directories', each carrying
        'tracker_path', 'output_directory', 'source_ids', 'jobs', 'summary', and 'skipped_sources' keys, together
        with total counts. A path that is not a directory is listed under 'invalid_paths', and a directory whose
        preparation raised is listed under 'failed_directories' as an entry carrying 'log_directory' and 'error'. The
        'success' flag reads False when no directory prepared and at least one failed. Returns an error dictionary if
        the extraction config is missing or unreadable, or if the log directory and output directory lists differ in
        length.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        return {"error": f"Extraction config not found: {config_path}"}

    try:
        ExtractionConfig.from_yaml(file_path=config_file)
    except Exception as error:
        return {"error": f"Invalid extraction config: {error}"}

    if len(output_directories) != len(log_directories):
        return {
            "error": (
                f"Length mismatch: {len(log_directories)} log directories but "
                f"{len(output_directories)} output directories."
            ),
        }

    result_log_directories: dict[str, Any] = {}
    invalid_paths: list[str] = []
    failed_directories: list[dict[str, str]] = []
    total_jobs = 0

    for entry_index, log_directory_string in enumerate(log_directories):
        log_directory_path = Path(log_directory_string)

        if not log_directory_path.is_dir():
            invalid_paths.append(log_directory_string)
            continue

        # Prepares every requested controller the configuration declares and whose archive resolves under this log
        # directory. Lenient sourcing records the controllers it cannot prepare rather than failing the whole batch,
        # since one caller applies one controller list across several recordings.
        try:
            job_set = prepare_jobs(
                log_directory=log_directory_path,
                output_directory=Path(output_directories[entry_index]),
                config_path=config_file,
                source_ids=source_ids or None,
                strict_sources=False,
            )
            sized_jobs = [size_job(job=job, core_ceiling=job_set.core_ceiling) for job in job_set.jobs]
            sized_jobs.sort(key=lambda entry: entry[1].memory_mb, reverse=True)
        except Exception as error:
            failed_directories.append({"log_directory": log_directory_string, "error": str(error)})
            continue

        # Merges the tracker's live state over the prepared set, so a directory prepared twice reports what its jobs
        # have already done rather than presenting every job as freshly scheduled.
        try:
            tracker_status = read_tracker_status(tracker_path=job_set.tracker_path)
        except Exception:
            tracker_status = {"jobs": [], "summary": {}}

        recorded = {entry["job_id"]: entry for entry in tracker_status.get("jobs", [])}

        jobs: list[dict[str, Any]] = []
        for descriptor, sizing in sized_jobs:
            entry: dict[str, Any] = dict(descriptor.to_mapping())
            entry["memory_mb"] = sizing.memory_mb
            entry["message_count"] = sizing.message_count
            entry["archive_bytes"] = sizing.archive_bytes
            entry["modeled"] = sizing.modeled
            entry["status"] = recorded.get(descriptor.job_id, {}).get("status", "SCHEDULED")
            error_message = recorded.get(descriptor.job_id, {}).get("error_message")
            if error_message is not None:
                entry["error_message"] = error_message
            jobs.append(entry)

        result_log_directories[log_directory_string] = {
            "tracker_path": str(job_set.tracker_path),
            "output_directory": str(job_set.output_directory),
            "source_ids": [descriptor.source_id for descriptor, _ in sized_jobs],
            "jobs": jobs,
            "summary": tracker_status.get("summary", {}),
            "skipped_sources": [
                {"source_id": source_id, "reason": reason} for source_id, reason in job_set.skipped_sources
            ],
        }
        total_jobs += len(jobs)

    # A request that prepared no directory while at least one raised reports no success, since the caller holds no
    # manifest to execute and the recorded failures are the reason.
    result: dict[str, Any] = {
        "success": bool(result_log_directories) or not failed_directories,
        "log_directories": result_log_directories,
        "total_log_directories": len(result_log_directories),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    if failed_directories:
        result["failed_directories"] = failed_directories

    return result


@mcp.tool()
def execute_log_processing_jobs_tool(
    jobs: list[dict[str, Any]],
    *,
    core_budget: int = -1,
    memory_budget_mb: int = -1,
) -> dict[str, Any]:
    """Dispatches log processing jobs for background execution against a core and a memory budget.

    Takes job descriptors from the manifest produced by prepare_log_processing_batch_tool and starts a background
    execution manager. Each job descriptor must include its own 'config_path' key pointing to the ExtractionConfig
    YAML file for that job. Each job's cores and memory are resolved from the archive it reads before dispatch, so a
    long recording and a short one are admitted at their own sizes. A job runs at the declared stage width narrowed
    to the workers its own archive repays, and an archive below the parallel processing threshold takes a single
    core. The manager admits a job once the running set has room for both its cores and its memory, and it admits an
    oversized job alone rather than leaving it queued forever.

    Important:
        Only one execution session can be active at a time. Use cancel_log_processing_tool to cancel an active
        session before starting a new one.

    Args:
        jobs: The list of job descriptors from prepare_log_processing_batch_tool, each passed through unchanged.
            Every key that tool emits is required, which is 'log_directory', 'archive_path', 'output_directory',
            'config_path', 'tracker_path', 'job_name', 'job_id', 'source_id', 'core_weight', 'message_count',
            'archive_bytes', and 'modeled'.
        core_budget: The total number of CPU cores available for the execution session. Set to -1 to auto-resolve
            to every available core minus the reserved host cores.
        memory_budget_mb: The total memory in megabytes available for the execution session. Set to -1 to
            auto-resolve to a share of the host's physical memory.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', the resolved 'core_budget', 'memory_budget_mb',
        and 'pool_size', a 'job_allocations' entry per job naming its 'job_id', 'source_id', resolved 'cores',
        'memory_mb', archive 'message_count', and 'modeled' flag, and any invalid jobs. Returns an error dictionary
        when a session is already active, and an error dictionary carrying 'invalid_jobs' when no job is valid.
    """
    # Enforces single-session constraint.
    existing_state = get_execution_state()
    if (
        existing_state is not None
        and existing_state.manager_thread is not None
        and existing_state.manager_thread.is_alive()
    ):
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    # Resolves both budgets before sizing, since the core budget bounds the width any single job receives.
    resolved_cores = resolve_core_budget(requested_budget=core_budget)
    resolved_memory = resolve_memory_budget_mb(requested_budget_mb=memory_budget_mb)

    # Rebuilds every descriptor and its footprint from the mapping the preparation emitted, then resolves this
    # session's own width and memory from those figures. The preparation already read each archive, so re-deriving
    # the width against a different budget costs no filesystem access.
    pending: list[tuple[JobDescriptor, JobSizing]] = []
    all_jobs: dict[tuple[str, str], JobDescriptor] = {}
    invalid_jobs: list[dict[str, Any]] = []
    job_allocations: list[dict[str, Any]] = []

    for job_dict in jobs:
        try:
            descriptor = JobDescriptor.from_mapping(mapping=job_dict)
        except Exception as error:
            invalid_jobs.append({**job_dict, "error": str(error)})
            continue

        if not descriptor.tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {descriptor.tracker_path}"})
            continue

        try:
            footprint = ArchiveFootprint(
                message_count=int(job_dict["message_count"]),
                archive_bytes=int(job_dict["archive_bytes"]),
                modeled=bool(job_dict["modeled"]),
            )
        except (KeyError, TypeError, ValueError):
            invalid_jobs.append({**job_dict, "error": "Missing or unreadable sizing keys from the prepared manifest."})
            continue

        core_weight = resolve_job_workers(footprint=footprint, ceiling=resolved_cores)
        descriptor = replace(descriptor, core_weight=core_weight)
        sizing = JobSizing(
            memory_mb=estimate_job_memory_mb(footprint=footprint, cores=core_weight),
            message_count=footprint.message_count,
            archive_bytes=footprint.archive_bytes,
            modeled=footprint.modeled,
        )

        pending.append((descriptor, sizing))
        all_jobs[descriptor.dispatch_key] = descriptor
        job_allocations.append(
            {
                "job_id": descriptor.job_id,
                "source_id": descriptor.source_id,
                "cores": core_weight,
                "memory_mb": sizing.memory_mb,
                "message_count": sizing.message_count,
                "modeled": sizing.modeled,
            }
        )

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Creates the execution state and claims it as the session of record. The claim tests the incumbent, publishes the
    # replacement, and starts the manager thread under one lock, so a second caller arriving mid-start finds a live
    # session rather than an empty slot. The guard above rejects the common sequential case with a specific message,
    # while this claim closes the window two concurrent tool calls open.
    pool_size = resolve_pool_size(job_count=len(pending), core_budget=resolved_cores, memory_budget_mb=resolved_memory)
    state = JobExecutionState(
        all_jobs=all_jobs,
        pending_jobs=pending,
        core_budget=resolved_cores,
        memory_budget_mb=resolved_memory,
        pool_size=pool_size,
    )

    if not start_execution_session(state=state):
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "core_budget": resolved_cores,
        "memory_budget_mb": resolved_memory,
        "pool_size": pool_size,
        "job_allocations": job_allocations,
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
        'summary' with the job total alongside counts for scheduled, running, succeeded, and failed jobs. A call
        made with no execution session returns only an 'active' flag set to False and an explanatory 'message'.
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

    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            registry = ProcessingTracker(file_path=tracker_path).snapshot()
        except Exception:
            job_details.extend(
                {"job_id": job.job_id, "source_id": job.source_id, "status": "UNKNOWN"} for job in path_jobs
            )
            continue

        for job in path_jobs:
            if job.job_id in registry:
                job_state = registry[job.job_id]
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
        jobs per hour once at least one job has completed. A call made with no execution session returns only an
        'active' flag set to False and an explanatory 'message'.
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

    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            registry = ProcessingTracker(file_path=tracker_path).snapshot()
        except Exception:  # noqa: S112
            continue

        for job in path_jobs:
            if job.job_id not in registry:
                continue

            job_info = registry[job.job_id]
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

    # Sets the canceled flag and clears pending jobs under the lock. Active jobs complete naturally.
    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_jobs)
        state.pending_jobs.clear()
        active_count = len(state.active_jobs)

    # Counts the final status of this session's own jobs alone. A tracker records every job that ever wrote to its
    # directory, so counting its whole registry would credit this session with the outcomes of earlier ones.
    succeeded = 0
    failed = 0

    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            registry = ProcessingTracker(file_path=tracker_path).snapshot()
        except Exception:  # noqa: S112
            continue

        for job in path_jobs:
            if job.job_id not in registry:
                continue

            job_state = registry[job.job_id]
            if job_state.status == ProcessingStatus.SUCCEEDED:
                succeeded += 1
            elif job_state.status == ProcessingStatus.FAILED:
                failed += 1

    return {
        "canceled": True,
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} job(s) still completing.",
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
        A dictionary containing a 'reset' flag, the number of jobs reset, and updated job statuses. Returns an error
        dictionary if the tracker file is missing or unreadable, and a 'reset' flag set to False with an explanatory
        'message' when no job matches the requested source IDs or when the active execution session holds one of the
        targeted jobs.
    """
    path = Path(tracker_path)

    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    tracker = ProcessingTracker(file_path=path)
    try:
        registry = tracker.snapshot()
    except Exception as error:
        return {"error": f"Unable to read tracker: {error}"}

    # Identifies which job IDs to reset based on the source_ids filter. The comparison is exact, since the tracker's
    # own substring search would also match a source whose identifier merely contains a requested one.
    if source_ids is not None:
        source_id_set = set(source_ids)
        target_ids = [job_id for job_id, job_state in registry.items() if job_state.specifier in source_id_set]
    else:
        target_ids = list(registry)

    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Refuses a reset that targets a job the live session holds, since the session's manager and the job's own worker
    # both write that job's outcome over the reset entry, discarding the re-run the operator asked for.
    state = get_execution_state()
    if state is not None and state.manager_thread is not None and state.manager_thread.is_alive():
        session_job_ids = {job_id for tracker_key, job_id in state.all_jobs if tracker_key == str(path)}
        contested_ids = [job_id for job_id in target_ids if job_id in session_job_ids]

        if contested_ids:
            message = (
                f"Unable to reset {len(contested_ids)} job(s) currently held by the active execution session. Cancel "
                f"the session with cancel_log_processing_tool before resetting these jobs."
            )
            return {"reset": False, "message": message}

    # Resets the targeted jobs back to SCHEDULED under the tracker's lock, leaving every other job untouched.
    tracker.reset_jobs(job_ids=target_ids)

    # Reads back the updated state for the response.
    try:
        updated_status = read_tracker_status(tracker_path=path)
    except Exception:
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
    log_directory_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    try:
        tracker_paths = discover_marker_files(directory=root_path, marker_name=OutputLayout.TRACKER_FILENAME)
    except OSError as error:
        return {"error": f"Unable to search '{root_directory}': {error}"}

    for found_tracker_path in tracker_paths:
        log_directory = str(found_tracker_path.parent)
        try:
            status = read_tracker_status(tracker_path=found_tracker_path)
            summary = status.get("summary", {})

            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            directory_status = ProcessingTracker.resolve_status(summary=summary).value

            log_directory_statuses.append(
                {
                    "log_directory": log_directory,
                    "tracker_path": str(found_tracker_path),
                    "status": directory_status,
                    **status,
                }
            )
        except Exception:
            log_directory_statuses.append(
                {
                    "log_directory": log_directory,
                    "tracker_path": str(found_tracker_path),
                    "status": "error",
                    "error": "Unable to read tracker file.",
                }
            )

    return {
        "log_directories": log_directory_statuses,
        "total_log_directories": len(log_directory_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
    }
