"""Contains tests for the classes and functions provided by the orchestration/execution.py module."""

from typing import Any
from pathlib import Path
from threading import Event, Thread
from collections.abc import Callable, Generator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest
from tests.log_archives import setup_test_environment
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingStatus, ProcessingTracker

from ataraxis_communication_interface.orchestration import execution
from ataraxis_communication_interface.orchestration.jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    JobSizing,
    OutputLayout,
    JobDescriptor,
    resolve_module_path,
)
from ataraxis_communication_interface.orchestration.discovery import JobSet, size_job, prepare_jobs
from ataraxis_communication_interface.orchestration.execution import (
    _MAXIMUM_JOB_REQUEUES,
    _MAXIMUM_POOL_REBUILDS,
    JobExecutionState,
    _fail_job,
    _ActiveJob,
    _reset_job,
    _abandon_batch,
    _job_is_unrecorded,
    _admit_pending_jobs,
    _handle_broken_pool,
    _reap_finished_jobs,
    get_execution_state,
    set_execution_state,
    group_jobs_by_tracker,
    job_execution_manager,
    _select_admissible_jobs,
    start_execution_session,
    _reconcile_unrecorded_job,
)
from ataraxis_communication_interface.orchestration.allocation import (
    SPAWNED_CHILD_MEMORY_MB,
    resolve_pool_size,
    resolve_host_memory_mb,
)

_MANAGER_TIMEOUT_SECONDS: int = 180
"""Stores the time a test waits for the execution manager thread to drain its queues before failing."""

_UNRECORDED_JOB_MESSAGE: str = "Job terminated without updating tracker status."
"""Stores the message the engine records against a job whose body ended without reaching its tracker."""

_SOURCE_ID: str = "1"
"""Stores the controller source identifier the end-to-end manager tests build their single-job batch around."""

_MODULE_TYPE: int = 1
"""Stores the type code of the hardware module the synthetic archive of the manager tests holds messages for."""

_MODULE_ID: int = 2
"""Stores the identifier code of the hardware module the synthetic archive of the manager tests holds messages for."""


@pytest.fixture(autouse=True)
def execution_state_guard() -> Generator[None, None, None]:
    """Clears the module-global execution state after every test, so no session leaks into the next test."""
    yield
    set_execution_state(state=None)


@pytest.fixture
def replacement_pool(monkeypatch: pytest.MonkeyPatch) -> "_StandInPool":
    """Points the rebuild pass at a stand-in pool factory and returns the pool that factory installs."""
    pool = _StandInPool()

    def _build_replacement_pool(pool_size: int) -> "_StandInPool":
        """Builds the stand-in replacement pool opening the requested number of slots."""
        pool.slot_count = pool_size
        return pool

    monkeypatch.setattr(execution, "_create_job_pool", _build_replacement_pool)
    return pool


@pytest.fixture
def session_manager(monkeypatch: pytest.MonkeyPatch) -> Generator["_StandInManager", None, None]:
    """Points a started session at a stand-in manager, then releases and joins every thread that session started."""
    manager = _StandInManager()
    monkeypatch.setattr(execution, "Thread", manager.build_thread)
    monkeypatch.setattr(execution, "job_execution_manager", manager.run_session)

    yield manager

    manager.settle()


@pytest.mark.xdist_group(name="orchestration")
def test_active_job_fields(tmp_path: Path) -> None:
    """Verifies that an active job pairs the job descriptor with the figures it was admitted at and its future."""
    job = _build_descriptor(directory=tmp_path, source_id="1")
    sizing = _build_sizing(memory_mb=1024)
    future = Future()

    active = _ActiveJob(job=job, sizing=sizing, future=future)

    assert active.job is job
    assert active.sizing is sizing
    assert active.future is future


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_empty_pending() -> None:
    """Verifies that selecting from an empty queue returns an empty admitted list and an empty deferred list."""
    admitted, deferred = _select_admissible_jobs(
        pending=[],
        core_budget=8,
        memory_budget_mb=8192,
        used_cores=0,
        used_memory_mb=0,
    )

    assert admitted == []
    assert deferred == []


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_admits_a_fitting_job(tmp_path: Path) -> None:
    """Verifies that a job fitting both budgets is admitted alongside the jobs already running."""
    job = _build_entry(directory=tmp_path, source_id="1", core_weight=2, memory_mb=1024)

    admitted, deferred = _select_admissible_jobs(
        pending=[job],
        core_budget=8,
        memory_budget_mb=8192,
        used_cores=1,
        used_memory_mb=1024,
    )

    assert admitted == [job]
    assert deferred == []


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_forces_oversized_job_when_idle(tmp_path: Path) -> None:
    """Verifies that a job wider than the whole budget is still admitted while nothing else is running."""
    oversized = _build_entry(directory=tmp_path, source_id="1", core_weight=16, memory_mb=8192)

    admitted, deferred = _select_admissible_jobs(
        pending=[oversized],
        core_budget=4,
        memory_budget_mb=1024,
        used_cores=0,
        used_memory_mb=0,
    )

    assert admitted == [oversized]
    assert deferred == []


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_forces_only_one_job_when_idle(tmp_path: Path) -> None:
    """Verifies that only the first oversized job is forced through, so the idle batch admits it on its own."""
    heavier = _build_entry(directory=tmp_path, source_id="1", core_weight=16, memory_mb=9216)
    lighter = _build_entry(directory=tmp_path, source_id="2", core_weight=16, memory_mb=8192)

    admitted, deferred = _select_admissible_jobs(
        pending=[lighter, heavier],
        core_budget=4,
        memory_budget_mb=1024,
        used_cores=0,
        used_memory_mb=0,
    )

    assert admitted == [heavier]
    assert deferred == [lighter]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_defers_oversized_job_when_busy(tmp_path: Path) -> None:
    """Verifies that a job wider than the whole budget is deferred while another job is already running."""
    oversized = _build_entry(directory=tmp_path, source_id="1", core_weight=16, memory_mb=8192)

    admitted, deferred = _select_admissible_jobs(
        pending=[oversized],
        core_budget=4,
        memory_budget_mb=1024,
        used_cores=1,
        used_memory_mb=1024,
    )

    assert admitted == []
    assert deferred == [oversized]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_backfills_past_unfitting_job(tmp_path: Path) -> None:
    """Verifies that the scan continues past a job that does not fit, so a lighter job takes the spare capacity."""
    heavy = _build_entry(directory=tmp_path, source_id="1", core_weight=8, memory_mb=1000)
    light = _build_entry(directory=tmp_path, source_id="2", core_weight=2, memory_mb=500)

    admitted, deferred = _select_admissible_jobs(
        pending=[heavy, light],
        core_budget=4,
        memory_budget_mb=10000,
        used_cores=1,
        used_memory_mb=0,
    )

    assert admitted == [light]
    assert deferred == [heavy]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_gated_by_memory(tmp_path: Path) -> None:
    """Verifies that a job whose memory exceeds the remaining budget is deferred while another job is running."""
    job = _build_entry(directory=tmp_path, source_id="1", core_weight=1, memory_mb=512)

    admitted, deferred = _select_admissible_jobs(
        pending=[job],
        core_budget=64,
        memory_budget_mb=1024,
        used_cores=1,
        used_memory_mb=1000,
    )

    assert admitted == []
    assert deferred == [job]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_gated_by_cores(tmp_path: Path) -> None:
    """Verifies that a job whose cores exceed the remaining budget is deferred even when memory is abundant."""
    job = _build_entry(directory=tmp_path, source_id="1", core_weight=1, memory_mb=10)

    admitted, deferred = _select_admissible_jobs(
        pending=[job],
        core_budget=4,
        memory_budget_mb=100000,
        used_cores=4,
        used_memory_mb=0,
    )

    assert admitted == []
    assert deferred == [job]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_considers_heaviest_first(tmp_path: Path) -> None:
    """Verifies that the heaviest job is weighed against the budgets first, so it is placed ahead of a lighter one."""
    light = _build_entry(directory=tmp_path, source_id="1", core_weight=1, memory_mb=1024)
    heavy = _build_entry(directory=tmp_path, source_id="2", core_weight=1, memory_mb=2048)

    admitted, deferred = _select_admissible_jobs(
        pending=[light, heavy],
        core_budget=100,
        memory_budget_mb=2048,
        used_cores=1,
        used_memory_mb=0,
    )

    assert admitted == [heavy]
    assert deferred == [light]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_breaks_memory_ties_by_core_weight(tmp_path: Path) -> None:
    """Verifies that two jobs holding equal memory are ordered by their core weight, widest first."""
    narrow = _build_entry(directory=tmp_path, source_id="1", core_weight=1, memory_mb=1024)
    wide = _build_entry(directory=tmp_path, source_id="2", core_weight=3, memory_mb=1024)

    admitted, deferred = _select_admissible_jobs(
        pending=[narrow, wide],
        core_budget=4,
        memory_budget_mb=100000,
        used_cores=1,
        used_memory_mb=0,
    )

    assert admitted == [wide]
    assert deferred == [narrow]


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_credits_the_occupied_pool_slot(tmp_path: Path) -> None:
    """Verifies that an admitted job takes over its pool slot's baseline, so that baseline is not charged twice."""
    first = _build_entry(directory=tmp_path, source_id="1", core_weight=1, memory_mb=1000)
    second = _build_entry(directory=tmp_path, source_id="2", core_weight=1, memory_mb=1000)

    # Both jobs together hold 2000 MB. The first admitted job releases the baseline of the slot it takes over, so the
    # second job is weighed at 1848 MB against the budget and both are admitted.
    admitted, deferred = _select_admissible_jobs(
        pending=[first, second],
        core_budget=8,
        memory_budget_mb=1000 + 1000 - SPAWNED_CHILD_MEMORY_MB,
        used_cores=1,
        used_memory_mb=0,
    )

    assert len(admitted) == 2
    assert deferred == []


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_state_defaults() -> None:
    """Verifies that a batch execution state starts with empty queues, unit budgets, and no recovery history."""
    state = JobExecutionState()

    assert state.all_jobs == {}
    assert state.pending_jobs == []
    assert state.active_jobs == {}
    assert state.core_budget == 1
    assert state.memory_budget_mb == 1024
    assert state.pool_size == 1
    assert not state.lock.locked()
    assert state.manager_thread is None
    assert not state.canceled
    assert not state.pool_broken
    assert state.broken_jobs == []
    assert state.pool_rebuilds == 0
    assert state.requeue_counts == {}


@pytest.mark.xdist_group(name="orchestration")
def test_execution_state_round_trip() -> None:
    """Verifies that the stored execution state is returned unchanged and can be cleared back to None."""
    assert get_execution_state() is None

    state = JobExecutionState(core_budget=4, memory_budget_mb=4096)
    set_execution_state(state=state)
    assert get_execution_state() is state

    replacement = JobExecutionState(core_budget=2, memory_budget_mb=2048)
    set_execution_state(state=replacement)
    assert get_execution_state() is replacement

    set_execution_state(state=None)
    assert get_execution_state() is None


@pytest.mark.xdist_group(name="orchestration")
def test_start_execution_session_claims_a_free_slot(session_manager: "_StandInManager") -> None:
    """Verifies that a session started with no incumbent is published and runs the manager thread recorded on it."""
    state = JobExecutionState(core_budget=4, memory_budget_mb=4096)

    assert start_execution_session(state=state)
    assert get_execution_state() is state
    assert isinstance(state.manager_thread, Thread)

    # Pins the ordering the handshake rests on. The thread sits on the state before it is started, so the body
    # running in it never reads its own session as one that has already finished.
    assert session_manager.recorded_threads == [state.manager_thread]

    session_manager.settle()

    assert session_manager.sessions == [state]
    assert not state.manager_thread.is_alive()


@pytest.mark.xdist_group(name="orchestration")
def test_start_execution_session_refuses_a_live_session(session_manager: "_StandInManager") -> None:
    """Verifies that a second session is refused while the incumbent's manager thread is still running."""
    incumbent = JobExecutionState(core_budget=4, memory_budget_mb=4096)
    contender = JobExecutionState(core_budget=2, memory_budget_mb=2048)

    assert start_execution_session(state=incumbent)
    assert incumbent.manager_thread.is_alive()

    # The incumbent stays the session of record, so every cancellation and status tool still reaches it, and the
    # contender gets no thread of its own to commit the host's cores and memory a second time.
    assert not start_execution_session(state=contender)
    assert get_execution_state() is incumbent
    assert contender.manager_thread is None
    assert session_manager.threads == [incumbent.manager_thread]


@pytest.mark.xdist_group(name="orchestration")
def test_start_execution_session_replaces_a_dead_session(session_manager: "_StandInManager") -> None:
    """Verifies that a session whose manager thread has ended is replaced by the next batch asking for the slot."""
    incumbent = JobExecutionState(core_budget=4, memory_budget_mb=4096)

    assert start_execution_session(state=incumbent)

    session_manager.settle()
    assert not incumbent.manager_thread.is_alive()

    replacement = JobExecutionState(core_budget=2, memory_budget_mb=2048)

    assert start_execution_session(state=replacement)
    assert get_execution_state() is replacement
    assert replacement.manager_thread is not incumbent.manager_thread
    assert session_manager.recorded_threads == [incumbent.manager_thread, replacement.manager_thread]

    session_manager.settle()

    assert session_manager.sessions == [incumbent, replacement]


@pytest.mark.xdist_group(name="orchestration")
def test_group_jobs_by_tracker(tmp_path: Path) -> None:
    """Verifies that every job in the registry is grouped under the tracker path that records it."""
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()

    first_jobs = [_build_descriptor(directory=first_directory, source_id=str(index)) for index in (1, 2, 3)]
    second_jobs = [_build_descriptor(directory=second_directory, source_id=str(index)) for index in (4, 5)]

    state = JobExecutionState()
    for job in first_jobs + second_jobs:
        state.all_jobs[job.dispatch_key] = job

    grouped = group_jobs_by_tracker(state=state)

    first_tracker = first_directory / OutputLayout.TRACKER_FILENAME
    second_tracker = second_directory / OutputLayout.TRACKER_FILENAME

    assert set(grouped.keys()) == {first_tracker, second_tracker}
    assert grouped[first_tracker] == first_jobs
    assert grouped[second_tracker] == second_jobs


@pytest.mark.xdist_group(name="orchestration")
def test_resolve_pool_size_reflects_the_memory_budget() -> None:
    """Verifies that the slot count a batch opens is held to the warmed bodies half its memory budget can hold."""
    # Half of 1024 MB holds three spawned children, which is fewer than the jobs or the cores on offer.
    assert resolve_pool_size(job_count=4, core_budget=8, memory_budget_mb=1024) == 3

    # A budget that cannot afford a single body still opens one slot, so the batch is never stalled by its own model.
    assert resolve_pool_size(job_count=4, core_budget=8, memory_budget_mb=SPAWNED_CHILD_MEMORY_MB) == 1

    # The job count and the core budget bound the count from the other side.
    assert resolve_pool_size(job_count=1, core_budget=8, memory_budget_mb=65536) == 1
    assert resolve_pool_size(job_count=8, core_budget=3, memory_budget_mb=65536) == 3


@pytest.mark.xdist_group(name="orchestration")
def test_admit_pending_jobs_fills_every_resolved_pool_slot(tmp_path: Path) -> None:
    """Verifies that a batch sized by resolve_pool_size admits a running set that fills the slots it opened."""
    pool_size = resolve_pool_size(job_count=4, core_budget=8, memory_budget_mb=4096)
    state = JobExecutionState(core_budget=8, memory_budget_mb=4096, pool_size=pool_size)
    state.pending_jobs = [
        _build_entry(directory=tmp_path, source_id=str(index), core_weight=1, memory_mb=512) for index in range(4)
    ]

    pool = _RecordingPool()
    _admit_pending_jobs(state=state, executor=pool)

    assert pool_size == 4
    assert len(state.active_jobs) == pool_size
    assert len(pool.submissions) == pool_size
    assert state.pending_jobs == []


@pytest.mark.xdist_group(name="orchestration")
@pytest.mark.parametrize("pool_size, expected_admitted", [(1, 2), (3, 1)])
def test_admit_pending_jobs_charges_every_idle_pool_slot(
    tmp_path: Path, pool_size: int, expected_admitted: int
) -> None:
    """Verifies that each idle pool slot holds a spawned child's baseline against the batch's memory budget."""
    state = JobExecutionState(core_budget=8, memory_budget_mb=2200, pool_size=pool_size)
    state.pending_jobs = [
        _build_entry(directory=tmp_path, source_id=str(index), core_weight=1, memory_mb=1024) for index in range(2)
    ]

    pool = _RecordingPool()
    _admit_pending_jobs(state=state, executor=pool)

    assert len(state.active_jobs) == expected_admitted
    assert len(state.pending_jobs) == 2 - expected_admitted


@pytest.mark.xdist_group(name="orchestration")
def test_admit_pending_jobs_fails_a_job_above_host_memory(tmp_path: Path) -> None:
    """Verifies that a job estimated above the host's physical memory is failed instead of being dispatched."""
    job = _register_job(directory=tmp_path, source_id="1")
    sizing = _build_sizing(memory_mb=resolve_host_memory_mb() + 1024)

    state = JobExecutionState(core_budget=8, memory_budget_mb=65536, pool_size=1)
    state.all_jobs[job.dispatch_key] = job
    state.pending_jobs = [(job, sizing)]

    pool = _RecordingPool()
    _admit_pending_jobs(state=state, executor=pool)

    assert pool.submissions == []
    assert state.pending_jobs == []
    assert state.active_jobs == {}
    assert _job_status(job=job) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_admit_pending_jobs_requeues_a_job_a_broken_pool_rejected(tmp_path: Path) -> None:
    """Verifies that a submission the pool rejects returns to the queue and flags the pool for a rebuild."""
    entry = _build_entry(directory=tmp_path, source_id="1", core_weight=1, memory_mb=512)

    state = JobExecutionState(core_budget=8, memory_budget_mb=65536, pool_size=1)
    state.pending_jobs = [entry]

    pool = _RecordingPool(error=BrokenProcessPool("The pool is broken."))
    _admit_pending_jobs(state=state, executor=pool)

    assert state.pool_broken
    assert state.pending_jobs == [entry]
    assert state.active_jobs == {}


@pytest.mark.xdist_group(name="orchestration")
def test_job_is_unrecorded_running_entry(tmp_path: Path) -> None:
    """Verifies that only a running tracker entry reads as unrecorded, while a scheduled one does not.

    Notes:
        A job that reached a worker and finished can never legitimately read as scheduled, so a scheduled entry means
        an external reset landed while the job ran. Reporting it as unrecorded would let the reap overwrite the
        re-run the operator asked for with a fabricated failure.
    """
    job = _register_job(directory=tmp_path, source_id="1")

    assert not _job_is_unrecorded(job=job)

    ProcessingTracker(file_path=job.tracker_path).start_job(job_id=job.job_id)

    assert _job_is_unrecorded(job=job)


@pytest.mark.xdist_group(name="orchestration")
@pytest.mark.parametrize("terminal_status", [ProcessingStatus.SUCCEEDED, ProcessingStatus.FAILED])
def test_job_is_unrecorded_terminal_entry(tmp_path: Path, terminal_status: ProcessingStatus) -> None:
    """Verifies that a job whose tracker entry holds a terminal outcome is not reported as unrecorded."""
    job = _register_job(directory=tmp_path, source_id="1")
    tracker = ProcessingTracker(file_path=job.tracker_path)

    if terminal_status == ProcessingStatus.SUCCEEDED:
        tracker.start_job(job_id=job.job_id)
        tracker.complete_job(job_id=job.job_id)
    else:
        tracker.fail_job(job_id=job.job_id, error_message="boom")

    assert not _job_is_unrecorded(job=job)


@pytest.mark.xdist_group(name="orchestration")
def test_job_is_unrecorded_unknown_job(tmp_path: Path) -> None:
    """Verifies that a job the tracker does not hold at all is not reported as unrecorded."""
    _register_job(directory=tmp_path, source_id="1")
    stranger = _build_descriptor(directory=tmp_path, source_id="2")

    assert not _job_is_unrecorded(job=stranger)


@pytest.mark.xdist_group(name="orchestration")
def test_job_is_unrecorded_unreadable_tracker(tmp_path: Path) -> None:
    """Verifies that a tracker file that cannot be deserialized reports no unrecorded job instead of raising."""
    job = _build_descriptor(directory=tmp_path, source_id="1")
    job.tracker_path.write_text("- not a tracker mapping\n")

    assert not _job_is_unrecorded(job=job)


@pytest.mark.xdist_group(name="orchestration")
def test_reconcile_unrecorded_job_fails_an_unrecorded_entry(tmp_path: Path) -> None:
    """Verifies that a job whose body ended without reaching its tracker is recorded as failed."""
    job = _register_job(directory=tmp_path, source_id="1")
    ProcessingTracker(file_path=job.tracker_path).start_job(job_id=job.job_id)

    _reconcile_unrecorded_job(job=job)

    assert _job_status(job=job) == ProcessingStatus.FAILED
    assert ProcessingTracker(file_path=job.tracker_path).get_job_info(job_id=job.job_id).error_message == (
        _UNRECORDED_JOB_MESSAGE
    )


@pytest.mark.xdist_group(name="orchestration")
def test_reconcile_unrecorded_job_keeps_a_recorded_outcome(tmp_path: Path) -> None:
    """Verifies that a job that already recorded its own outcome is left exactly as its body recorded it."""
    job = _register_job(directory=tmp_path, source_id="1")
    tracker = ProcessingTracker(file_path=job.tracker_path)
    tracker.fail_job(job_id=job.job_id, error_message="The job recorded this failure itself.")

    _reconcile_unrecorded_job(job=job)

    assert _job_status(job=job) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job.job_id).error_message == "The job recorded this failure itself."


@pytest.mark.xdist_group(name="orchestration")
def test_reset_job_returns_a_failed_entry_to_scheduled(tmp_path: Path) -> None:
    """Verifies that resetting a requeued job clears its recorded failure so it starts from a clean record."""
    job = _register_job(directory=tmp_path, source_id="1")
    ProcessingTracker(file_path=job.tracker_path).fail_job(job_id=job.job_id, error_message="boom")

    _reset_job(job=job)

    assert _job_status(job=job) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_reset_job_absorbs_an_unreadable_tracker(tmp_path: Path) -> None:
    """Verifies that resetting a job whose tracker cannot be deserialized returns instead of raising."""
    job = _build_descriptor(directory=tmp_path, source_id="1")
    job.tracker_path.write_text("- not a tracker mapping\n")

    _reset_job(job=job)

    assert job.tracker_path.read_text() == "- not a tracker mapping\n"


@pytest.mark.xdist_group(name="orchestration")
def test_fail_job_records_the_error_message(tmp_path: Path) -> None:
    """Verifies that failing a job records both the terminal status and the message explaining it."""
    job = _register_job(directory=tmp_path, source_id="1")

    _fail_job(job=job, error_message="The host killed the worker.")

    assert _job_status(job=job) == ProcessingStatus.FAILED
    assert ProcessingTracker(file_path=job.tracker_path).get_job_info(job_id=job.job_id).error_message == (
        "The host killed the worker."
    )


@pytest.mark.xdist_group(name="orchestration")
def test_fail_job_absorbs_an_unreadable_tracker(tmp_path: Path) -> None:
    """Verifies that failing a job whose tracker cannot be deserialized returns instead of raising."""
    job = _build_descriptor(directory=tmp_path, source_id="1")
    job.tracker_path.write_text("- not a tracker mapping\n")

    _fail_job(job=job, error_message="The host killed the worker.")

    assert job.tracker_path.read_text() == "- not a tracker mapping\n"


@pytest.mark.xdist_group(name="orchestration")
def test_reap_finished_jobs_keeps_a_running_job(tmp_path: Path) -> None:
    """Verifies that a job whose future has not resolved stays in the running set."""
    job = _register_job(directory=tmp_path, source_id="1")
    sizing = _build_sizing(memory_mb=1024)

    state = JobExecutionState()
    state.active_jobs[job.dispatch_key] = _ActiveJob(job=job, sizing=sizing, future=Future())

    _reap_finished_jobs(state=state)

    assert list(state.active_jobs) == [job.dispatch_key]
    assert _job_status(job=job) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_reap_finished_jobs_reconciles_an_unrecorded_return(tmp_path: Path) -> None:
    """Verifies that a job returning without recording its outcome is failed rather than left unfinished."""
    job = _register_job(directory=tmp_path, source_id="1")
    ProcessingTracker(file_path=job.tracker_path).start_job(job_id=job.job_id)
    future = Future()
    future.set_result(None)

    state = JobExecutionState()
    state.active_jobs[job.dispatch_key] = _ActiveJob(job=job, sizing=_build_sizing(memory_mb=1024), future=future)

    _reap_finished_jobs(state=state)

    assert state.active_jobs == {}
    assert _job_status(job=job) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_reap_finished_jobs_keeps_an_externally_reset_job(tmp_path: Path) -> None:
    """Verifies that a job reset to scheduled while it ran keeps that state instead of being failed by the reap.

    Notes:
        Pins the tracker-reset race directly. A job that completed and was then reset by an operator must survive the
        reap that follows, because overwriting it would record a failure that never happened and discard the re-run
        the reset requested.
    """
    job = _register_job(directory=tmp_path, source_id="1")
    tracker = ProcessingTracker(file_path=job.tracker_path)
    tracker.start_job(job_id=job.job_id)
    tracker.complete_job(job_id=job.job_id)
    tracker.reset_jobs(job_ids=[job.job_id])

    future = Future()
    future.set_result(None)

    state = JobExecutionState()
    state.active_jobs[job.dispatch_key] = _ActiveJob(job=job, sizing=_build_sizing(memory_mb=1024), future=future)

    _reap_finished_jobs(state=state)

    assert state.active_jobs == {}
    assert _job_status(job=job) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_reap_finished_jobs_keeps_a_recorded_failure(tmp_path: Path) -> None:
    """Verifies that a job that recorded its own failure before raising keeps the message its body wrote."""
    job = _register_job(directory=tmp_path, source_id="1")
    ProcessingTracker(file_path=job.tracker_path).fail_job(job_id=job.job_id, error_message="The archive is empty.")

    future = Future()
    future.set_exception(RuntimeError("The archive is empty."))

    state = JobExecutionState()
    state.active_jobs[job.dispatch_key] = _ActiveJob(job=job, sizing=_build_sizing(memory_mb=1024), future=future)

    _reap_finished_jobs(state=state)

    assert state.active_jobs == {}
    assert not state.pool_broken
    assert ProcessingTracker(file_path=job.tracker_path).get_job_info(job_id=job.job_id).error_message == (
        "The archive is empty."
    )


@pytest.mark.xdist_group(name="orchestration")
def test_reap_finished_jobs_records_a_pool_break(tmp_path: Path) -> None:
    """Verifies that a broken pool killing an unrecorded job flags the rebuild and queues the job it killed."""
    job = _register_job(directory=tmp_path, source_id="1")
    ProcessingTracker(file_path=job.tracker_path).start_job(job_id=job.job_id)

    sizing = _build_sizing(memory_mb=1024)
    future = Future()
    future.set_exception(BrokenProcessPool("A worker process terminated abruptly."))

    state = JobExecutionState()
    state.active_jobs[job.dispatch_key] = _ActiveJob(job=job, sizing=sizing, future=future)

    _reap_finished_jobs(state=state)

    assert state.pool_broken
    assert state.broken_jobs == [(job, sizing)]
    assert state.active_jobs == {}
    assert _job_status(job=job) == ProcessingStatus.RUNNING


@pytest.mark.xdist_group(name="orchestration")
def test_reap_finished_jobs_ignores_a_break_after_a_recorded_outcome(tmp_path: Path) -> None:
    """Verifies that a pool break reported after a job recorded its outcome does not trigger a rebuild."""
    job = _register_job(directory=tmp_path, source_id="1")
    tracker = ProcessingTracker(file_path=job.tracker_path)
    tracker.start_job(job_id=job.job_id)
    tracker.complete_job(job_id=job.job_id)

    future = Future()
    future.set_exception(BrokenProcessPool("A worker process terminated abruptly."))

    state = JobExecutionState()
    state.active_jobs[job.dispatch_key] = _ActiveJob(job=job, sizing=_build_sizing(memory_mb=1024), future=future)

    _reap_finished_jobs(state=state)

    assert not state.pool_broken
    assert state.broken_jobs == []
    assert _job_status(job=job) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_handle_broken_pool_charges_the_job_that_ran_alone(tmp_path: Path, replacement_pool: "_StandInPool") -> None:
    """Verifies that a break with a single job in flight is attributed to that job and charged one requeue."""
    job = _register_job(directory=tmp_path, source_id="1")
    ProcessingTracker(file_path=job.tracker_path).start_job(job_id=job.job_id)
    sizing = _build_sizing(memory_mb=1024)

    broken_pool = _StandInPool()
    state = JobExecutionState(broken_jobs=[(job, sizing)], pool_size=2, pool_broken=True)

    result = _handle_broken_pool(state=state, executor=broken_pool)

    assert result is replacement_pool
    assert replacement_pool.slot_count == 2
    assert broken_pool.shutdown_requests == [(False, True)]
    assert state.requeue_counts == {job.dispatch_key: 1}
    assert state.pending_jobs == [(job, sizing)]
    assert state.broken_jobs == []
    assert not state.pool_broken
    assert state.pool_rebuilds == 1
    assert _job_status(job=job) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_handle_broken_pool_charges_no_job_for_a_multi_job_break(
    tmp_path: Path, replacement_pool: "_StandInPool"
) -> None:
    """Verifies that a break killing several jobs at once requeues every one of them free of charge."""
    first = _register_job(directory=tmp_path / "first", source_id="1")
    second = _register_job(directory=tmp_path / "second", source_id="2")
    sizing = _build_sizing(memory_mb=1024)
    for job in (first, second):
        ProcessingTracker(file_path=job.tracker_path).start_job(job_id=job.job_id)

    state = JobExecutionState(broken_jobs=[(first, sizing), (second, sizing)], pool_broken=True)

    result = _handle_broken_pool(state=state, executor=_StandInPool())

    assert result is replacement_pool
    # The break fails every job the pool was running, so it is attributable to none of them.
    assert state.requeue_counts == {}
    assert state.pending_jobs == [(first, sizing), (second, sizing)]
    assert state.broken_jobs == []
    assert _job_status(job=first) == ProcessingStatus.SCHEDULED
    assert _job_status(job=second) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_handle_broken_pool_spares_the_budget_of_an_unattributable_break(
    tmp_path: Path, replacement_pool: "_StandInPool"
) -> None:
    """Verifies that a job already at its requeue ceiling is requeued when the break killed a second job with it."""
    spent = _register_job(directory=tmp_path / "spent", source_id="1")
    companion = _register_job(directory=tmp_path / "companion", source_id="2")
    sizing = _build_sizing(memory_mb=1024)
    for job in (spent, companion):
        ProcessingTracker(file_path=job.tracker_path).start_job(job_id=job.job_id)

    state = JobExecutionState(
        broken_jobs=[(spent, sizing), (companion, sizing)],
        pool_broken=True,
        requeue_counts={spent.dispatch_key: _MAXIMUM_JOB_REQUEUES},
    )

    result = _handle_broken_pool(state=state, executor=_StandInPool())

    assert result is replacement_pool
    # An unattributable break spends no budget, so the job holding a spent one is requeued rather than failed.
    assert state.requeue_counts == {spent.dispatch_key: _MAXIMUM_JOB_REQUEUES}
    assert state.pending_jobs == [(spent, sizing), (companion, sizing)]
    assert _job_status(job=spent) == ProcessingStatus.SCHEDULED
    assert _job_status(job=companion) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_abandon_batch_fails_every_unfinished_job(tmp_path: Path) -> None:
    """Verifies that abandoning a batch fails every running, queued, and killed job and stops further admission."""
    active_job = _register_job(directory=tmp_path / "active", source_id="1")
    pending_job = _register_job(directory=tmp_path / "pending", source_id="2")
    broken_job = _register_job(directory=tmp_path / "broken", source_id="3")

    sizing = _build_sizing(memory_mb=1024)
    state = JobExecutionState()
    state.active_jobs[active_job.dispatch_key] = _ActiveJob(job=active_job, sizing=sizing, future=Future())
    state.pending_jobs.append((pending_job, sizing))
    state.broken_jobs.append((broken_job, sizing))

    _abandon_batch(state=state, reason="The host is out of memory.")

    assert state.canceled
    assert state.active_jobs == {}
    assert state.pending_jobs == []
    assert state.broken_jobs == []
    for job in (active_job, pending_job, broken_job):
        assert _job_status(job=job) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_abandon_batch_fails_the_jobs_a_caller_already_drained(tmp_path: Path) -> None:
    """Verifies that jobs held outside the state queues still reach a terminal status when a batch is abandoned."""
    tracker_path = tmp_path / "tracker.yaml"
    job = JobDescriptor.for_archive(
        archive_path=tmp_path / f"5{LOG_ARCHIVE_SUFFIX}",
        output_directory=tmp_path,
        config_path=tmp_path / "extraction_config.yaml",
        tracker_path=tracker_path,
        source_id="5",
    )
    ProcessingTracker(file_path=tracker_path).align_jobs(
        jobs=[(CONTROLLER_EXTRACTION_JOB_NAME, "5")],
        universe=[(CONTROLLER_EXTRACTION_JOB_NAME, "5")],
    )
    state = JobExecutionState(pool_rebuilds=_MAXIMUM_POOL_REBUILDS)

    _abandon_batch(
        state=state,
        reason="the pool broke",
        orphaned=[(job, JobSizing(cores=1, memory_mb=0))],
    )

    assert ProcessingTracker(file_path=tracker_path).snapshot()[job.job_id].status is ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_runs_a_prepared_batch(tmp_path: Path) -> None:
    """Verifies that the manager dispatches a prepared job into its shared pool and drains its queues when it ends."""
    job_set, descriptor, sizing = _build_single_job_batch(tmp_path=tmp_path)

    state = JobExecutionState(
        all_jobs={descriptor.dispatch_key: descriptor},
        pending_jobs=[(descriptor, sizing)],
        core_budget=1,
        memory_budget_mb=8192,
        pool_size=1,
    )
    set_execution_state(state=state)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.pending_jobs == []
    assert state.active_jobs == {}
    assert not state.pool_broken
    assert state.pool_rebuilds == 0
    assert _job_status(job=descriptor) == ProcessingStatus.SUCCEEDED
    assert _module_output_path(job_set=job_set).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_cancellation_skips_dispatch(tmp_path: Path) -> None:
    """Verifies that a canceled session exits without dispatching any of the jobs still queued for execution."""
    job_set, descriptor, sizing = _build_single_job_batch(tmp_path=tmp_path)

    state = JobExecutionState(
        all_jobs={descriptor.dispatch_key: descriptor},
        pending_jobs=[(descriptor, sizing)],
        core_budget=1,
        memory_budget_mb=8192,
        pool_size=1,
        canceled=True,
    )
    set_execution_state(state=state)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.pending_jobs == [(descriptor, sizing)]
    assert state.active_jobs == {}
    assert _job_status(job=descriptor) == ProcessingStatus.SCHEDULED
    assert not _module_output_path(job_set=job_set).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_rebuilds_a_broken_pool(tmp_path: Path) -> None:
    """Verifies that a session whose pool broke rebuilds it, requeues the job it killed, and runs that job."""
    job_set, descriptor, sizing = _build_single_job_batch(tmp_path=tmp_path)

    # Seeds the state a break leaves behind, which is what the reaping pass writes when a worker is killed.
    state = JobExecutionState(
        all_jobs={descriptor.dispatch_key: descriptor},
        broken_jobs=[(descriptor, sizing)],
        core_budget=1,
        memory_budget_mb=8192,
        pool_size=1,
        pool_broken=True,
    )
    set_execution_state(state=state)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.pool_rebuilds == 1
    assert not state.pool_broken
    assert state.broken_jobs == []
    assert state.pending_jobs == []
    assert state.active_jobs == {}

    # The job ran alone when the pool broke, so the break is attributed to it and charged against its requeue budget.
    assert state.requeue_counts == {descriptor.dispatch_key: 1}
    assert _job_status(job=descriptor) == ProcessingStatus.SUCCEEDED
    assert _module_output_path(job_set=job_set).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_fails_a_job_past_the_requeue_ceiling(tmp_path: Path) -> None:
    """Verifies that a job that has spent its requeue budget is failed instead of being dispatched again."""
    job_set, descriptor, sizing = _build_single_job_batch(tmp_path=tmp_path)

    state = JobExecutionState(
        all_jobs={descriptor.dispatch_key: descriptor},
        broken_jobs=[(descriptor, sizing)],
        core_budget=1,
        memory_budget_mb=8192,
        pool_size=1,
        pool_broken=True,
        requeue_counts={descriptor.dispatch_key: _MAXIMUM_JOB_REQUEUES},
    )
    set_execution_state(state=state)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.pool_rebuilds == 1
    assert state.pending_jobs == []
    assert state.active_jobs == {}
    assert _job_status(job=descriptor) == ProcessingStatus.FAILED
    assert not _module_output_path(job_set=job_set).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_abandons_past_the_rebuild_ceiling(tmp_path: Path) -> None:
    """Verifies that a session that has spent its rebuild budget fails every job it did not finish."""
    job_set, descriptor, sizing = _build_single_job_batch(tmp_path=tmp_path)

    state = JobExecutionState(
        all_jobs={descriptor.dispatch_key: descriptor},
        pending_jobs=[(descriptor, sizing)],
        core_budget=1,
        memory_budget_mb=8192,
        pool_size=1,
        pool_broken=True,
        pool_rebuilds=_MAXIMUM_POOL_REBUILDS,
    )
    set_execution_state(state=state)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.canceled
    assert state.pending_jobs == []
    assert state.active_jobs == {}
    assert state.pool_rebuilds == _MAXIMUM_POOL_REBUILDS
    assert _job_status(job=descriptor) == ProcessingStatus.FAILED
    assert not _module_output_path(job_set=job_set).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_abandons_when_the_pool_cannot_be_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a session whose replacement pool cannot be built fails every job it did not finish."""
    job_set, descriptor, sizing = _build_single_job_batch(tmp_path=tmp_path)

    creations = []
    build_pool = execution._create_job_pool

    def failing_rebuild(pool_size: int) -> ProcessPoolExecutor:
        """Builds the session's first pool and refuses every replacement it is asked for afterwards."""
        creations.append(pool_size)
        if len(creations) == 1:
            return build_pool(pool_size=pool_size)

        message = "The host refused to spawn a worker."
        raise RuntimeError(message)

    monkeypatch.setattr(execution, "_create_job_pool", failing_rebuild)

    state = JobExecutionState(
        all_jobs={descriptor.dispatch_key: descriptor},
        pending_jobs=[(descriptor, sizing)],
        core_budget=1,
        memory_budget_mb=8192,
        pool_size=1,
        pool_broken=True,
    )
    set_execution_state(state=state)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert creations == [1, 1]
    assert state.canceled
    assert state.pending_jobs == []
    assert _job_status(job=descriptor) == ProcessingStatus.FAILED
    assert not _module_output_path(job_set=job_set).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_abandons_when_the_pool_cannot_be_created(tmp_path: Path) -> None:
    """Verifies that a session whose shared pool cannot be created at all fails every job it holds."""
    job_set, descriptor, sizing = _build_single_job_batch(tmp_path=tmp_path)

    # A pool of zero slots cannot be built, so the manager's own creation raises before any job is dispatched.
    state = JobExecutionState(
        all_jobs={descriptor.dispatch_key: descriptor},
        pending_jobs=[(descriptor, sizing)],
        core_budget=1,
        memory_budget_mb=8192,
        pool_size=0,
    )
    set_execution_state(state=state)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.canceled
    assert state.pending_jobs == []
    assert state.active_jobs == {}
    assert _job_status(job=descriptor) == ProcessingStatus.FAILED
    assert not _module_output_path(job_set=job_set).exists()


class _RecordingPool:
    """Stands in for the session's shared process pool, recording every submission it accepts.

    Notes:
        Only the admission pass is exercised through this stand-in. The manager itself is tested end to end against a
        real spawned pool, so the pool's own behavior is never simulated.

    Attributes:
        submissions: Stores the function and job pair of every submission this pool accepted.
        error: Stores the error every submission raises instead of accepting the work, or None to accept every
            submission.
    """

    def __init__(self, error: BaseException | None = None) -> None:
        self.submissions = []
        self.error = error

    def submit(self, function: Callable[[JobDescriptor], None], job: JobDescriptor) -> Future[None]:
        """Records one submission and returns an already-resolved future, or raises the configured error."""
        if self.error is not None:
            raise self.error

        self.submissions.append((function, job))
        future = Future()
        future.set_result(None)
        return future


class _StandInPool:
    """Stands in for the shared process pool a rebuild retires and for the replacement it installs in its place.

    Notes:
        The rebuild pass only disposes of the broken pool and hands back its replacement, so neither pool is ever
        asked to run work and only the disposal request needs recording.

    Attributes:
        slot_count: Stores the number of slots the rebuild pass requested of the factory that installed this pool.
        shutdown_requests: Stores the wait and cancel_futures flag pair of every disposal request this pool received.
    """

    def __init__(self) -> None:
        self.slot_count = 0
        self.shutdown_requests = []

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        """Records the disposal flags one shutdown request carries."""
        self.shutdown_requests.append((wait, cancel_futures))


class _StandInManager:
    """Stands in for the session's execution manager, recording how each session starts the thread that runs it.

    Notes:
        A real manager builds a shared process pool, which the session handshake itself never needs. Holding every
        body on one event instead keeps a published session readable as live for as long as a test requires.

        The thread the state already carries is read at the moment that thread is started, because a session that
        started the thread before recording it would leave the running body reading its own session as finished.

    Attributes:
        threads: Stores every manager thread this stand-in built, in the order the sessions requested them.
        recorded_threads: Stores the thread each session's state already carried when that session started its own
            manager thread.
        sessions: Stores the execution state of every session whose manager body this stand-in ran.
        release: Stores the event that ends every held manager body, letting the tests decide when a session stops
            reading as live.
    """

    def __init__(self) -> None:
        self.threads = []
        self.recorded_threads = []
        self.sessions = []
        self.release = Event()

    def build_thread(self, target: Callable[..., None], kwargs: dict[str, Any], *, daemon: bool) -> Thread:
        """Builds one session's manager thread, wrapping its start with the record of what the state held."""
        state = kwargs["state"]
        thread = Thread(target=target, kwargs=kwargs, daemon=daemon)
        start_thread = thread.start

        def start() -> None:
            """Records the thread the state already carries, then starts the manager thread."""
            self.recorded_threads.append(state.manager_thread)
            start_thread()

        thread.start = start
        self.threads.append(thread)
        return thread

    def run_session(self, state: JobExecutionState) -> None:
        """Serves as one session's manager body, holding that session alive until the test releases it."""
        self.sessions.append(state)
        self.release.wait(timeout=_MANAGER_TIMEOUT_SECONDS)

    def settle(self) -> None:
        """Releases every held session and waits for the thread running it to end."""
        self.release.set()
        for thread in self.threads:
            thread.join(timeout=_MANAGER_TIMEOUT_SECONDS)


def _build_descriptor(directory: Path, source_id: str, core_weight: int = 1) -> JobDescriptor:
    """Builds a job descriptor rooted in the target directory and carrying the requested core weight."""
    return JobDescriptor.for_archive(
        archive_path=directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
        output_directory=directory,
        config_path=directory / "extraction_config.yaml",
        tracker_path=directory / OutputLayout.TRACKER_FILENAME,
        source_id=source_id,
        log_directory=directory,
        core_weight=core_weight,
    )


def _build_sizing(memory_mb: int, cores: int = 1) -> JobSizing:
    """Builds a sizing record carrying the requested memory figure."""
    return JobSizing(cores=cores, memory_mb=memory_mb)


def _build_entry(
    directory: Path, source_id: str, core_weight: int = 1, memory_mb: int = 0
) -> tuple[JobDescriptor, JobSizing]:
    """Builds the descriptor and sizing pair the pending queue holds for one job."""
    return (
        _build_descriptor(directory=directory, source_id=source_id, core_weight=core_weight),
        _build_sizing(memory_mb=memory_mb),
    )


def _register_job(directory: Path, source_id: str, core_weight: int = 1) -> JobDescriptor:
    """Registers one scheduled job on a real tracker in the target directory and returns its descriptor."""
    directory.mkdir(parents=True, exist_ok=True)
    job = _build_descriptor(directory=directory, source_id=source_id, core_weight=core_weight)
    ProcessingTracker(file_path=job.tracker_path).initialize_jobs(jobs=[(CONTROLLER_EXTRACTION_JOB_NAME, source_id)])
    return job


def _job_status(job: JobDescriptor) -> ProcessingStatus:
    """Reads the tracker status the target job's entry currently holds."""
    return ProcessingTracker(file_path=job.tracker_path).get_job_status(job_id=job.job_id)


def _build_single_job_batch(tmp_path: Path) -> tuple[JobSet, JobDescriptor, JobSizing]:
    """Prepares and sizes the single-source batch the end-to-end manager tests dispatch."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_directory,
        config_path=config_path,
    )
    descriptor, sizing, _ = size_job(job=job_set.jobs[0])

    return job_set, descriptor, sizing


def _module_output_path(job_set: JobSet) -> Path:
    """Resolves the module output file the single-source batch writes once its job succeeds."""
    return resolve_module_path(
        output_directory=job_set.output_directory,
        source_id=_SOURCE_ID,
        module_type=_MODULE_TYPE,
        module_id=_MODULE_ID,
    )
