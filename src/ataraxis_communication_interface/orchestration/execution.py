"""Provides the batch execution engine that admits sized extraction jobs against a core and a memory budget and runs
each one in a worker of a single shared process pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from threading import Lock, Thread
import contextlib
from dataclasses import field, dataclass
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import (
    ProcessingStatus,
    ProcessingTracker,
    limit_worker_threads,
    initialize_worker_threads,
)

from .worker import run_extraction_job
from .allocation import SPAWNED_CHILD_MEMORY_MB, resolve_host_memory_mb

if TYPE_CHECKING:
    from pathlib import Path
    from threading import Barrier
    from collections.abc import Sequence
    from concurrent.futures import Future
    from multiprocessing.context import SpawnContext

    from .jobs import JobSizing, JobDescriptor

_MULTIPROCESSING_CONTEXT: SpawnContext = get_context("spawn")
"""The spawn-based multiprocessing context the session's shared job pool is created over. Spawn is the library-wide
policy, so the pool behaves identically on every supported platform and every job body re-imports its module rather
than inheriting the parent's state."""

_POOL_WORKER_THREAD_CEILING: int = 1
"""The threads every pool worker pins its numeric backends to. Each worker either runs a job sequentially or opens an
extraction pool whose own children are pinned the same way, so no worker repays a backend pool wider than one
thread."""

_POOL_WARMUP_TIMEOUT_SECONDS: float = 120.0
"""The time the pool's warm-up allows every worker to spawn and reach the barrier before the creation is abandoned."""

_DISPATCH_POLL_SECONDS: int = 1
"""The interval at which the manager re-examines the running set for freed capacity."""

_MAXIMUM_POOL_REBUILDS: int = 3
"""The times one session rebuilds its shared pool before abandoning the batch. A single transient kill is worth
absorbing, while a host that kills workers three times over is misconfigured and every further attempt fails the same
way at the same cost."""

_MAXIMUM_JOB_REQUEUES: int = 2
"""The times one job is requeued after a break it is provably responsible for. A job running alone when the pool
broke is the only job a break can be attributed to, so only such a job spends this budget."""

_EXECUTION_LOCK: Lock = Lock()
"""Serializes the check-and-reserve that admits one batch execution session at a time. The test of the reference below
and its replacement sit on opposite sides of a bytecode boundary. Two callers finding the slot free would otherwise both
publish a session, and the second would strand the first session's worker pool beyond the reach of every cancellation
tool."""


@dataclass(slots=True)
class JobExecutionState:
    """Tracks runtime state for one batch execution session budgeted by both cores and memory.

    Notes:
        Every job body runs in a worker of one shared pool that outlives it. A body admitted at more than one core
        opens its own extraction pool at that width, and a body admitted at one core reads its archive sequentially.
        Total live processes are the pool's slot count plus the cores of the running set that opened a pool, and both
        terms are budgeted.
    """

    all_jobs: dict[tuple[str, str], JobDescriptor] = field(default_factory=dict)
    """Every submitted job, keyed by its dispatch key."""
    pending_jobs: list[tuple[JobDescriptor, JobSizing]] = field(default_factory=list)
    """Jobs awaiting admission, each paired with the figures it was sized at."""
    active_jobs: dict[tuple[str, str], _ActiveJob] = field(default_factory=dict)
    """Jobs currently executing, keyed by dispatch key so a broken future is matched to its descriptor."""
    core_budget: int = 1
    """The cores the batch may commit across all concurrently running jobs."""
    memory_budget_mb: int = 1024
    """The memory the batch may commit across all concurrently running jobs."""
    pool_size: int = 1
    """The job slots the shared pool opens, every one of which is warmed when the pool is created."""
    lock: Lock = field(default_factory=Lock)
    """The lock guarding every mutation of the job queues."""
    manager_thread: Thread | None = None
    """The background thread running the execution manager, or None before the session starts it."""
    canceled: bool = False
    """Determines whether the execution session has been canceled."""
    pool_broken: bool = False
    """Determines whether the shared pool broke and awaits a rebuild."""
    broken_jobs: list[tuple[JobDescriptor, JobSizing]] = field(default_factory=list)
    """The jobs a pool break killed, awaiting requeue once the pool is rebuilt."""
    pool_rebuilds: int = 0
    """The times the shared pool has been rebuilt during this session."""
    requeue_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    """The requeues charged to each job, keyed by dispatch key. Only a job that broke the pool while running alone
    is charged, since a break fails every in-flight job whatever caused it."""


@dataclass(slots=True)
class _ActiveJob:
    """Tracks one job executing in a worker of the shared job pool."""

    job: JobDescriptor
    """The descriptor the pool was handed."""
    sizing: JobSizing
    """The resource figures this job was admitted at."""
    future: Future[None]
    """The future the pool returned, which carries the job body's outcome."""


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

    with _EXECUTION_LOCK:
        _execution_state = state


def start_execution_session(state: JobExecutionState) -> bool:
    """Publishes one execution state as the session of record and starts the thread that manages it.

    Notes:
        The incumbent test, the publication, and the thread start all happen under one lock. A thread reports itself
        alive only once it has started, so a state published before its thread runs reads as a finished session.
        Splitting these steps lets two callers each start a manager and double-commit the host's cores and memory.

        A session whose manager thread has ended is replaced, so a completed or an abandoned batch does not block
        every later batch.

    Args:
        state: The execution state to publish. Its manager thread is created, recorded, and started here.

    Returns:
        True when this state became the session of record, and False when a live session already holds that place.
    """
    global _execution_state

    with _EXECUTION_LOCK:
        active = _execution_state
        if active is not None and active.manager_thread is not None and active.manager_thread.is_alive():
            return False

        manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
        state.manager_thread = manager
        _execution_state = state
        manager.start()

    return True


def job_execution_manager(state: JobExecutionState) -> None:
    """Dispatches queued jobs into one shared process pool under the batch's core and memory budgets.

    Notes:
        Runs as a daemon thread for the lifetime of one execution session. Creates the pool once and keeps it, so a
        job body starts in a worker that is already alive. A body admitted at more than one core opens its own
        extraction pool at that width.

        Cancellation stops new admissions and lets the running jobs finish.

        Every exit path leaves each job it dispatched in a terminal tracker state, since the tracker is the only
        channel a status reader consults. A canceled batch's queued jobs stay scheduled and remain re-runnable.

    Args:
        state: The active job execution state. Mutated under its own lock as jobs move between the queues.
    """
    poll_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    executor: ProcessPoolExecutor | None = None

    try:
        executor = _create_job_pool(pool_size=state.pool_size)

        while True:
            rebuild_needed = False

            with state.lock:
                _reap_finished_jobs(state=state)

                if state.pool_broken:
                    rebuild_needed = True
                elif not state.pending_jobs and not state.active_jobs:
                    break
                elif state.canceled:
                    if not state.active_jobs:
                        break
                else:
                    _admit_pending_jobs(state=state, executor=executor)

            # Rebuilds outside the lock, because the replacement pool blocks on its warm-up and the cancellation
            # tool takes the same lock. Holding it here would stall a cancel for the whole warm-up timeout.
            if rebuild_needed:
                executor = _handle_broken_pool(state=state, executor=executor)
                if executor is None:
                    break
                continue

            poll_timer.delay(delay=_DISPATCH_POLL_SECONDS, allow_sleep=True)
    except Exception as error:
        _abandon_batch(state=state, reason=f"The batch's execution manager stopped: {error}.")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def group_jobs_by_tracker(state: JobExecutionState) -> dict[Path, list[JobDescriptor]]:
    """Groups every job in an execution state by the tracker file that records it.

    Batches the jobs sharing a tracker so each tracker file is deserialized once when iterating over the groups.

    Args:
        state: The active job execution state holding the job registry.

    Returns:
        The jobs recorded by each tracker, keyed by that tracker's path.
    """
    tracker_jobs: dict[Path, list[JobDescriptor]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)
    return tracker_jobs


def _select_admissible_jobs(
    pending: Sequence[tuple[JobDescriptor, JobSizing]],
    core_budget: int,
    memory_budget_mb: int,
    used_cores: int,
    used_memory_mb: int,
) -> tuple[list[tuple[JobDescriptor, JobSizing]], list[tuple[JobDescriptor, JobSizing]]]:
    """Selects the queued jobs whose cores and memory the budgets still allow.

    Notes:
        The scan considers the heaviest job first and continues past anything that does not fit, so lighter jobs
        backfill the capacity a heavier one leaves spare.

        A job is admitted alone when nothing is running, so a job larger than the whole budget still makes progress.
        An admitted job takes over the memory of the idle pool slot it occupies, so its baseline is not charged twice.

    Args:
        pending: The queued jobs to consider, each paired with the figures it was sized at.
        core_budget: The cores the batch may commit across all concurrently running jobs.
        memory_budget_mb: The memory the batch may commit across all concurrently running jobs.
        used_cores: The cores the already-running jobs commit.
        used_memory_mb: The memory the already-running jobs commit, including every idle pool slot.

    Returns:
        The jobs to admit now and the jobs to leave queued, in that order.
    """
    admitted: list[tuple[JobDescriptor, JobSizing]] = []
    deferred: list[tuple[JobDescriptor, JobSizing]] = []

    for job, sizing in sorted(pending, key=lambda entry: (entry[1].memory_mb, entry[0].core_weight), reverse=True):
        forced = used_cores == 0 and not admitted
        fits = used_cores + job.core_weight <= core_budget and used_memory_mb + sizing.memory_mb <= memory_budget_mb

        if not (fits or forced):
            deferred.append((job, sizing))
            continue

        if forced and not fits:
            message = (
                f"Job '{job.job_id}' for source '{job.source_id}' is estimated to hold {sizing.memory_mb} MB against "
                f"a batch memory budget of {memory_budget_mb} MB. It runs alone so the batch still progresses, and "
                f"the host may kill it."
            )
            console.echo(message=message, level=LogLevel.WARNING)

        admitted.append((job, sizing))
        used_cores += job.core_weight
        used_memory_mb += sizing.memory_mb - SPAWNED_CHILD_MEMORY_MB

    return admitted, deferred


def _create_job_pool(pool_size: int) -> ProcessPoolExecutor:
    """Creates the session's shared job pool with every worker already spawned and pinned.

    Notes:
        The environment pin is held for the construction and the warm-up alone, then released. The pin travels to a
        spawned child through the environment it inherits, so without a warm-up it would have to be held for the
        batch's whole runtime in the process that started it.

    Args:
        pool_size: The job slots the pool opens.

    Returns:
        The created pool, with every worker alive and pinned.

    Raises:
        BrokenBarrierError: If a worker fails to start within the warm-up timeout.
    """
    barrier = _MULTIPROCESSING_CONTEXT.Barrier(parties=pool_size + 1)

    with limit_worker_threads(thread_count=_POOL_WORKER_THREAD_CEILING):
        pool = ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=_MULTIPROCESSING_CONTEXT,
            initializer=_pin_pool_worker,
            initargs=(_POOL_WORKER_THREAD_CEILING, barrier),
        )
        warmups = [pool.submit(_warm_pool_worker) for _ in range(pool_size)]
        barrier.wait(timeout=_POOL_WARMUP_TIMEOUT_SECONDS)
        for warmup in warmups:
            warmup.result()

    return pool


def _pin_pool_worker(thread_count: int, barrier: Barrier) -> None:
    """Pins one shared-pool worker's numeric backends and holds it until every sibling worker has started.

    Notes:
        Pins from both sides. The inherited environment reaches the backends that size their pool while importing,
        and this call reaches the ones that read their width when first asked to do work.

        A pool spawns a worker only when work arrives and reuses an idle worker over spawning a new one. Holding every
        worker at the barrier is therefore what forces one spawn per slot while the parent's pin is still in force.

    Args:
        thread_count: The threads this worker's numeric backends may open.
        barrier: The barrier every worker and the creating parent meet on.
    """
    initialize_worker_threads(thread_count=thread_count)
    barrier.wait(timeout=_POOL_WARMUP_TIMEOUT_SECONDS)


def _warm_pool_worker() -> None:
    """Serves as the task whose submission forces one shared-pool worker to spawn."""


def _reap_finished_jobs(state: JobExecutionState) -> None:
    """Removes every finished job from the running set and records what a pool break killed.

    Notes:
        A job that raised on its own terms is left to its tracker, which recorded the failure before the exception
        reached the future.

        A break of the shared pool is recognized by two facts together. The exception is a BrokenProcessPool, which
        a job body's own extraction pool can also raise, and the job's tracker entry still reads running, which a
        body that recorded its own failure would not leave behind.

    Args:
        state: The active job execution state, mutated in place.
    """
    for dispatch_key, active in list(state.active_jobs.items()):
        if not active.future.done():
            continue

        del state.active_jobs[dispatch_key]

        try:
            active.future.result()
        except BrokenProcessPool:
            if _job_is_unrecorded(job=active.job):
                state.pool_broken = True
                state.broken_jobs.append((active.job, active.sizing))
                continue
            _reconcile_unrecorded_job(job=active.job)
        except Exception:
            _reconcile_unrecorded_job(job=active.job)
        else:
            _reconcile_unrecorded_job(job=active.job)


def _admit_pending_jobs(state: JobExecutionState, executor: ProcessPoolExecutor) -> None:
    """Admits every queued job the budgets still allow and submits it to the shared pool.

    Notes:
        A job estimated above the host's total physical memory is failed rather than admitted, since it cannot
        complete wherever it is dispatched.

        A submit that raises means the pool broke between the reap and this pass. The job returns to the queue
        without a requeue charge, since it never reached a worker.

    Args:
        state: The active job execution state, mutated in place.
        executor: The shared pool the admitted jobs are submitted to.
    """
    host_memory_mb = resolve_host_memory_mb()
    runnable: list[tuple[JobDescriptor, JobSizing]] = []

    for job, sizing in state.pending_jobs:
        if sizing.memory_mb > host_memory_mb:
            message = (
                f"Unable to run the microcontroller data extraction job for source '{job.source_id}' (job ID: "
                f"{job.job_id}). The job is estimated to hold {sizing.memory_mb} MB, which passes the host's total "
                f"physical memory of {host_memory_mb} MB."
            )
            _fail_job(job=job, error_message=message)
            continue
        runnable.append((job, sizing))

    idle_slots = max(0, state.pool_size - len(state.active_jobs))
    admitted, deferred = _select_admissible_jobs(
        pending=runnable,
        core_budget=state.core_budget,
        memory_budget_mb=state.memory_budget_mb,
        used_cores=sum(active.job.core_weight for active in state.active_jobs.values()),
        used_memory_mb=(
            sum(active.sizing.memory_mb for active in state.active_jobs.values()) + idle_slots * SPAWNED_CHILD_MEMORY_MB
        ),
    )

    for index, (job, sizing) in enumerate(admitted):
        try:
            future = executor.submit(run_extraction_job, job=job)
        except BrokenProcessPool:
            state.pool_broken = True
            deferred.extend(admitted[index:])
            break
        state.active_jobs[job.dispatch_key] = _ActiveJob(job=job, sizing=sizing, future=future)

    state.pending_jobs = deferred


def _handle_broken_pool(state: JobExecutionState, executor: ProcessPoolExecutor) -> ProcessPoolExecutor | None:
    """Replaces a broken shared pool and returns the jobs it killed to the queue.

    Notes:
        Runs outside the state lock, because building the replacement blocks on its warm-up while the cancellation
        tool waits on the same lock.

        A break fails every job the pool was running, so a requeue is charged only to a job that was running alone,
        which is the one case the break is attributable and the case an oversized job reaches. Every other requeued
        job returns to the queue free of charge.

    Args:
        state: The active job execution state, mutated in place.
        executor: The broken pool, shut down here.

    Returns:
        The replacement pool, or None when the session gives up and the batch has been abandoned.
    """
    executor.shutdown(wait=False, cancel_futures=True)

    with state.lock:
        broken = list(state.broken_jobs)
        state.broken_jobs.clear()
        state.pool_broken = False
        rebuilds = state.pool_rebuilds

    if rebuilds >= _MAXIMUM_POOL_REBUILDS:
        _abandon_batch(
            state=state,
            orphaned=broken,
            reason=(
                f"The batch's shared worker pool broke {rebuilds + 1} times, which passes the "
                f"{_MAXIMUM_POOL_REBUILDS} rebuilds one session allows. The host is killing worker processes, most "
                f"commonly because the batch's memory budget passes what it can actually supply."
            ),
        )
        return None

    try:
        replacement = _create_job_pool(pool_size=state.pool_size)
    except Exception as error:
        _abandon_batch(
            state=state,
            orphaned=broken,
            reason=f"The batch's shared worker pool broke and could not be rebuilt: {error}.",
        )
        return None

    # A break with a single job in flight is the one case the break is attributable to that job.
    attributable = broken[0][0].dispatch_key if len(broken) == 1 else None

    with state.lock:
        state.pool_rebuilds = rebuilds + 1

        for job, sizing in broken:
            charged = state.requeue_counts.get(job.dispatch_key, 0)

            if job.dispatch_key == attributable:
                if charged >= _MAXIMUM_JOB_REQUEUES:
                    _fail_job(
                        job=job,
                        error_message=(
                            f"The worker running this job was killed {charged + 1} times while the job ran alone, "
                            f"which passes the {_MAXIMUM_JOB_REQUEUES} requeues one job allows. The job was not "
                            f"retried again."
                        ),
                    )
                    continue
                state.requeue_counts[job.dispatch_key] = charged + 1

            _reset_job(job=job)
            state.pending_jobs.append((job, sizing))

    return replacement


def _abandon_batch(
    state: JobExecutionState,
    reason: str,
    orphaned: Sequence[tuple[JobDescriptor, JobSizing]] = (),
) -> None:
    """Fails every job the batch has not completed and stops it from admitting anything further.

    Notes:
        Every in-flight and every queued job is failed on its tracker, because a job left reading scheduled or
        running after the batch stopped would never resolve.

    Args:
        state: The active job execution state, mutated in place.
        reason: The message recorded against every unfinished job.
        orphaned: The jobs a caller has already removed from the state, which no longer reach it through the queues.
    """
    console.echo(message=reason, level=LogLevel.ERROR)

    with state.lock:
        state.canceled = True
        for active in state.active_jobs.values():
            _fail_job(job=active.job, error_message=reason)
        for job, _ in [*state.pending_jobs, *state.broken_jobs, *orphaned]:
            _fail_job(job=job, error_message=reason)
        state.active_jobs.clear()
        state.pending_jobs.clear()
        state.broken_jobs.clear()


def _job_is_unrecorded(job: JobDescriptor) -> bool:
    """Returns True when the target job's tracker entry still reads running rather than a terminal outcome.

    Notes:
        A scheduled entry is excluded. A job that reached a worker leaves a running entry behind, so a finished job
        reading scheduled was reset by an operator asking for a re-run, and recording an outcome over it would
        discard that re-run.

    Args:
        job: The job whose tracker entry is inspected.

    Returns:
        True when the entry reads running, and False when it reads any other status or cannot be read at all.
    """
    try:
        snapshot = ProcessingTracker(file_path=job.tracker_path).snapshot()
    except Exception:
        return False

    state = snapshot.get(job.job_id)
    return state is not None and state.status == ProcessingStatus.RUNNING


def _reconcile_unrecorded_job(job: JobDescriptor) -> None:
    """Records a terminal outcome for a job whose body ended without reaching its tracker.

    Notes:
        A body killed after the tracker's run_job() context opened leaves its entry reading running with no outcome
        of its own.

    Args:
        job: The finished job to reconcile.
    """
    if not _job_is_unrecorded(job=job):
        return

    _fail_job(job=job, error_message="Job terminated without updating tracker status.")


def _reset_job(job: JobDescriptor) -> None:
    """Returns one job's tracker entry to the scheduled state so a requeued job starts from a clean record.

    Args:
        job: The job whose tracker entry is returned to the scheduled state.
    """
    with contextlib.suppress(Exception):
        ProcessingTracker(file_path=job.tracker_path).reset_jobs(job_ids=[job.job_id])


def _fail_job(job: JobDescriptor, error_message: str) -> None:
    """Records one job's terminal failure, absorbing a tracker that cannot be written.

    Args:
        job: The job whose terminal failure is recorded.
        error_message: The explanation recorded against the failed job.
    """
    with contextlib.suppress(Exception):
        ProcessingTracker(file_path=job.tracker_path).fail_job(job_id=job.job_id, error_message=error_message)
