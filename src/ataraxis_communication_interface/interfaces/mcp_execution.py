"""Provides the batch execution engine that dispatches log processing jobs with budget-based worker allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from threading import Lock, Thread
import contextlib
from dataclasses import field, dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ..microcontroller.dataclasses import ExtractionConfig
from ..microcontroller.log_processing import PARALLEL_PROCESSING_THRESHOLD, execute_job, find_log_archive

if TYPE_CHECKING:
    from pathlib import Path

_WORKER_SCALING_FACTOR: int = 1000
"""Controls the saturation floor via the formula ``ceil(sqrt(messages / factor))``. The square root models diminishing
returns from process parallelism. This value sets the minimum workers a job receives before the budget division can
push it lower. With a factor of 1,000, a 648,000-message archive has a saturation floor of 25 workers."""

_WORKER_MULTIPLE: int = 5
"""Worker counts above 1 are rounded down to the nearest multiple of this value for clean allocation."""


@dataclass(slots=True)
class PendingJob:
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

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the tracker
        path with the job ID.
        """
        return str(self.tracker_path), self.job_id


@dataclass(slots=True)
class JobExecutionState:
    """Tracks runtime state for batch job execution with budget-based worker allocation.

    The execution manager groups pending jobs by source ID so that archives with similar sizes share a single
    ProcessPoolExecutor. Each group is dispatched as one thread that processes its jobs sequentially, reusing the
    pool across all archives in the group. This avoids the overhead of repeatedly spawning and tearing down worker
    processes for archives of the same size.
    """

    all_jobs: dict[tuple[str, str], PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by (tracker_path, job_id) dispatch key."""
    pending_queue: list[PendingJob] = field(default_factory=list)
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
    """Determines whether the execution session has been canceled."""


@dataclass(slots=True)
class _ActiveGroup:
    """Tracks a group of jobs executing sequentially with a shared ProcessPoolExecutor."""

    source_id: str
    """The shared source ID for all jobs in this group, or a unique identifier for single-job groups."""
    jobs: list[PendingJob]
    """The jobs in this group, processed sequentially by the group worker thread."""
    workers: int
    """The number of CPU cores allocated to this group's ProcessPoolExecutor."""
    thread: Thread
    """The background thread executing the group."""


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


def job_execution_manager() -> None:
    """Dispatches queued jobs as worker-tier groups with shared process pools.

    Runs as a daemon thread, polling at 1-second intervals. Each dispatch cycle classifies pending jobs into
    small (< 2000 messages, 1 worker each) and parallel (>= 2000 messages). Parallel jobs are grouped by their
    precomputed worker tier from ``_compute_sqrt_minimum``, which snaps archive sizes to discrete worker counts
    (multiples of 5). Jobs in the same tier share a single ProcessPoolExecutor sized exactly to that tier.
    Each tier is split into as many concurrent groups as the budget allows. Small jobs are dispatched individually.
    Exits when the queue is empty and no active groups remain.
    """
    if _execution_state is None:
        return

    state = _execution_state
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
                small_pending: list[PendingJob] = []
                parallel_pending: list[PendingJob] = []
                for job in state.pending_queue:
                    message_count = state.job_message_counts.get(job.dispatch_key, 0)
                    if message_count < PARALLEL_PROCESSING_THRESHOLD:
                        small_pending.append(job)
                    else:
                        parallel_pending.append(job)

                dispatch_groups: list[tuple[list[PendingJob], int]] = []

                # Phase 1: Groups parallel jobs by worker tier. Each job's optimal worker count is precomputed
                # via _compute_sqrt_minimum, which snaps archive sizes to discrete tiers (multiples of 5). Jobs
                # in the same tier share a ProcessPoolExecutor sized exactly to that tier. Each tier is split
                # into as many concurrent groups as the available budget allows.
                if parallel_pending and available >= _WORKER_MULTIPLE:
                    worker_tiers: dict[int, list[PendingJob]] = {}
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


def _group_worker(jobs: list[PendingJob], workers: int, state: JobExecutionState) -> None:
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
                config = ExtractionConfig.load(file_path=job.config_path)
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

            # Suppresses exceptions because execute_job calls tracker.fail_job internally on failure.
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


def _compute_sqrt_minimum(message_count: int) -> int:
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
