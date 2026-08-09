"""Provides the local batch execution engine that admits queued data extraction jobs against a core and a memory
budget and dispatches each one at the width its own archive resolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from threading import Lock, Thread
import contextlib
from dataclasses import field, dataclass

from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_data_structures import (
    ProcessingStatus,
    ProcessingTracker,
    find_log_archive,
    limit_worker_threads,
)

from .pipeline import execute_job
from .allocation import ArchiveFootprint, resolve_job_workers, estimate_job_memory_mb, resolve_archive_footprint
from ..microcontroller import ExtractionConfig

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from .jobs import PendingJob
    from ..microcontroller import ControllerExtractionConfig

_WORKER_THREAD_CEILING: int = 1
"""The number of threads every pool worker of the session pins its numeric backends to. The manager holds the pin for
the whole session, so a worker spawned by any job inherits it whatever else is running at the time."""

_DISPATCH_POLL_SECONDS: int = 1
"""The interval at which the manager re-examines the running set for freed capacity."""


@dataclass(slots=True)
class _ActiveJob:
    """Tracks a single job executing on its own worker thread."""

    job: PendingJob
    """The pending job descriptor associated with the running thread."""
    thread: Thread
    """The background thread executing the job."""


@dataclass(slots=True)
class JobExecutionState:
    """Tracks runtime state for one batch execution session budgeted by both cores and memory.

    Notes:
        Each admitted job runs on its own thread and owns a process pool sized to the cores it was admitted at. A
        long archive and a short one therefore run side by side, each at its own width.
    """

    all_jobs: dict[tuple[str, str], PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by their tracker path and job identifier pair."""
    pending_jobs: list[PendingJob] = field(default_factory=list)
    """Jobs awaiting dispatch."""
    active_jobs: list[_ActiveJob] = field(default_factory=list)
    """Jobs currently executing, each on its own thread with its own process pool."""
    core_budget: int = 1
    """The cores the batch may commit across all concurrently running jobs."""
    memory_budget_mb: int = 1024
    """The memory the batch may commit across all concurrently running jobs."""
    lock: Lock = field(default_factory=Lock)
    """The lock guarding every mutation of the job queues."""
    manager_thread: Thread | None = None
    """The background thread running the execution manager, or None before the session starts it."""
    canceled: bool = False
    """Determines whether the execution session has been canceled."""


_execution_state: JobExecutionState | None = None
"""Stores the active execution state for batch log processing jobs, or None when no session exists."""


def get_execution_state() -> JobExecutionState | None:
    """Returns the active batch log processing execution state, or None when no session exists."""
    return _execution_state


def set_execution_state(state: JobExecutionState | None) -> None:
    """Stores the active batch log processing execution state, replacing any existing session reference.

    Args:
        state: The execution state to store, or None to clear the active session.
    """
    global _execution_state
    _execution_state = state


def size_pending_job(job: PendingJob, core_budget: int) -> ArchiveFootprint:
    """Resolves the cores and the memory one queued job occupies from the archive it will read.

    Notes:
        Resolves the archive through the same recursive search the job itself uses at dispatch, so a nested archive
        is sized at the size it is processed at. An archive that cannot be resolved yields an unmodeled footprint,
        which sizes the job at the single-core baseline rather than dropping it.

    Args:
        job: The queued job to size. Its archive path, core weight, and memory weight are set in place.
        core_budget: The cores the batch may commit, which bounds the width this job receives.

    Returns:
        The footprint the weights were resolved from.
    """
    try:
        archive_path = find_log_archive(log_directory=job.log_directory, source_id=job.source_id)
    except Exception:
        footprint = ArchiveFootprint(message_count=0, archive_bytes=0, modeled=False)
    else:
        # Holds the resolved path on the job, so the dispatch that follows reads it instead of searching the tree a
        # second time for the archive this search already found.
        job.archive_path = archive_path
        footprint = resolve_archive_footprint(archive_path=archive_path)

    job.core_weight = resolve_job_workers(footprint=footprint, ceiling=core_budget)
    job.memory_mb = estimate_job_memory_mb(footprint=footprint, cores=job.core_weight)
    return footprint


def job_execution_manager(state: JobExecutionState) -> None:
    """Dispatches queued jobs under the batch's core and memory budgets until the queue and the running set empty.

    Notes:
        Runs as a daemon thread for the lifetime of one execution session, polling at one-second intervals. The pin
        on the worker threading layers is held for the whole session, so every pool any job opens inherits it and no
        job restores the environment while another is still spawning its workers.

        Cancellation stops new admissions and lets the running jobs finish, so a canceled session ends once the
        running set empties rather than interrupting work already in flight.

    Args:
        state: The active job execution state. Mutated under its own lock as jobs move between the queues.
    """
    poll_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    with limit_worker_threads(thread_count=_WORKER_THREAD_CEILING):
        while True:
            with state.lock:
                # Reaps finished jobs and frees their share of both budgets.
                state.active_jobs = [active for active in state.active_jobs if active.thread.is_alive()]

                if not state.pending_jobs and not state.active_jobs:
                    break

                if state.canceled:
                    if not state.active_jobs:
                        break
                else:
                    admitted, deferred = _select_admissible_jobs(
                        pending=state.pending_jobs,
                        core_budget=state.core_budget,
                        memory_budget_mb=state.memory_budget_mb,
                        used_cores=sum(active.job.core_weight for active in state.active_jobs),
                        used_memory_mb=sum(active.job.memory_mb for active in state.active_jobs),
                    )
                    state.pending_jobs = deferred

                    for job in admitted:
                        thread = Thread(target=_run_job, kwargs={"job": job}, daemon=True)
                        thread.start()
                        state.active_jobs.append(_ActiveJob(job=job, thread=thread))

            # Polls outside the lock to avoid blocking the cancellation tool.
            poll_timer.delay(delay=_DISPATCH_POLL_SECONDS, allow_sleep=True)


def group_jobs_by_tracker(state: JobExecutionState) -> dict[Path, list[PendingJob]]:
    """Groups every job in an execution state by the tracker file that records it.

    Batches the jobs sharing a tracker so each tracker file is deserialized once when iterating over the groups.

    Args:
        state: The active job execution state holding the job registry.

    Returns:
        The jobs recorded by each tracker, keyed by that tracker's path.
    """
    tracker_jobs: dict[Path, list[PendingJob]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)
    return tracker_jobs


def _select_admissible_jobs(
    pending: Sequence[PendingJob],
    core_budget: int,
    memory_budget_mb: int,
    used_cores: int,
    used_memory_mb: int,
) -> tuple[list[PendingJob], list[PendingJob]]:
    """Selects the queued jobs whose cores and memory the budgets still allow.

    Notes:
        A job is weighed against both budgets, and the one that runs out first is whichever the batch's mix makes
        scarce. The scan considers the heaviest job first so it is placed while the budgets are still open, and it
        continues past anything that does not fit, so lighter jobs backfill the capacity a heavier one leaves spare.

        A job is admitted alone when nothing is running and nothing has been admitted in this pass, so a job larger
        than the whole budget still makes progress instead of holding the queue forever.

    Args:
        pending: The queued jobs to consider, each already carrying its resolved core and memory weights.
        core_budget: The cores the batch may commit across all concurrently running jobs.
        memory_budget_mb: The memory the batch may commit across all concurrently running jobs.
        used_cores: The cores the already-running jobs commit.
        used_memory_mb: The memory the already-running jobs commit.

    Returns:
        The jobs to admit now and the jobs to leave queued, in that order.
    """
    admitted: list[PendingJob] = []
    deferred: list[PendingJob] = []

    for job in sorted(pending, key=lambda candidate: (candidate.memory_mb, candidate.core_weight), reverse=True):
        forced = used_cores == 0 and not admitted
        fits = used_cores + job.core_weight <= core_budget and used_memory_mb + job.memory_mb <= memory_budget_mb

        if not (fits or forced):
            deferred.append(job)
            continue

        admitted.append(job)
        used_cores += job.core_weight
        used_memory_mb += job.memory_mb

    return admitted, deferred


def _run_job(job: PendingJob) -> None:
    """Executes one admitted job at the width it was admitted at.

    Notes:
        Suppresses the job's exception, because execute_job wraps the work in the tracker's run_job() context, which
        records the failure before re-raising. Letting the exception escape would terminate this thread without
        adding anything the tracker does not already hold.

    Args:
        job: The admitted job to execute.
    """
    tracker = ProcessingTracker(file_path=job.tracker_path)

    controller_config = _resolve_controller_config(job=job, tracker=tracker)
    if controller_config is None:
        return

    with contextlib.suppress(Exception):
        # Falls back to a fresh search only for a job whose sizing failed to resolve the archive, where the search
        # repeats that failure and the tracker records it below.
        log_path = job.archive_path
        if log_path is None:
            log_path = find_log_archive(log_directory=job.log_directory, source_id=job.source_id)

        execute_job(
            log_path=log_path,
            output_directory=job.output_directory,
            source_id=job.source_id,
            job_id=job.job_id,
            workers=job.core_weight,
            tracker=tracker,
            controller_config=controller_config,
            display_progress=False,
        )

    # Records a terminal outcome for a job that ended without reaching the tracker, which happens when the archive
    # cannot be resolved before run_job() opens.
    try:
        reloaded = ProcessingTracker(file_path=job.tracker_path).snapshot()
        if job.job_id in reloaded and reloaded[job.job_id].status not in (
            ProcessingStatus.SUCCEEDED,
            ProcessingStatus.FAILED,
        ):
            _record_failure(
                tracker=tracker, job_id=job.job_id, error_message="Job terminated without updating tracker status."
            )
    except Exception:  # noqa: S110
        pass


def _resolve_controller_config(job: PendingJob, tracker: ProcessingTracker) -> ControllerExtractionConfig | None:
    """Reads the extraction configuration of the controller the target job processes.

    Notes:
        Records the failure on the tracker rather than raising it, since the caller runs on the job's own thread and
        the tracker is the only channel a batch consumer reads the outcome from.

    Args:
        job: The admitted job whose controller configuration is resolved.
        tracker: The tracker recording this job's outcome.

    Returns:
        The controller's extraction configuration, or None when the file cannot be read or registers no entry for
        the job's source.
    """
    try:
        config = ExtractionConfig.from_yaml(file_path=job.config_path)
    except Exception:
        _record_failure(
            tracker=tracker,
            job_id=job.job_id,
            error_message=f"Unable to load the extraction config from '{job.config_path}'.",
        )
        return None

    for controller in config.controllers:
        if str(controller.controller_id) == job.source_id:
            return controller

    _record_failure(
        tracker=tracker, job_id=job.job_id, error_message=f"No controller config for source '{job.source_id}'."
    )
    return None


def _record_failure(tracker: ProcessingTracker, job_id: str, error_message: str) -> None:
    """Records a job's failure on its tracker, absorbing a tracker that cannot be written.

    Notes:
        The caller runs on the job's own thread with no supervisor to catch what escapes it, so a tracker that is
        itself unreadable or unwritable leaves the job unrecorded rather than terminating the thread with an
        exception nothing observes.

    Args:
        tracker: The tracker recording this job's outcome.
        job_id: The unique identifier of the job that failed.
        error_message: The reason recorded alongside the failure.
    """
    with contextlib.suppress(Exception):
        tracker.fail_job(job_id=job_id, error_message=error_message)
