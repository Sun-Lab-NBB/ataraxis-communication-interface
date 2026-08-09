"""Contains tests for the classes and functions provided by the orchestration/execution.py module."""

from time import sleep
from threading import Thread

import pytest
from tests.log_archives import setup_test_environment, write_extraction_config
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ataraxis_communication_interface.orchestration.jobs import (
    TRACKER_FILENAME,
    EXTRACTION_JOB_NAME,
    PendingJob,
    generate_job_ids,
    resolve_module_feather_path,
)
from ataraxis_communication_interface.orchestration.execution import (
    JobExecutionState,
    _run_job,
    _ActiveJob,
    size_pending_job,
    get_execution_state,
    set_execution_state,
    group_jobs_by_tracker,
    job_execution_manager,
    _select_admissible_jobs,
    _resolve_controller_config,
)

_MANAGER_TIMEOUT_SECONDS: int = 180
"""Stores the time a test waits for the execution manager thread to drain its queues before failing."""

_ACTIVE_JOB_SECONDS: float = 0.5
"""Stores the time the synthetic active job of the cancellation test occupies its thread for."""


@pytest.fixture
def execution_state_guard():
    """Clears the module-global execution state after a test that replaces it, so no state leaks between tests."""
    set_execution_state(state=None)
    yield
    set_execution_state(state=None)


def _build_job(directory, source_id, config_path=None, core_weight=1, memory_mb=0):
    """Builds a pending job rooted in the target directory and carrying the requested core and memory weights."""
    return PendingJob(
        log_directory=directory,
        output_directory=directory,
        tracker_path=directory / TRACKER_FILENAME,
        job_id=f"job_{source_id}",
        source_id=source_id,
        config_path=directory / "config.yaml" if config_path is None else config_path,
        core_weight=core_weight,
        memory_mb=memory_mb,
    )


def _build_registered_job(log_directory, output_directory, config_path, source_id):
    """Builds a pending job whose identifier matches the extraction job the tracker registers for the target source."""
    return PendingJob(
        log_directory=log_directory,
        output_directory=output_directory,
        tracker_path=output_directory / TRACKER_FILENAME,
        job_id=generate_job_ids(source_ids=[source_id])[source_id],
        source_id=source_id,
        config_path=config_path,
    )


def _initialize_tracker(output_directory, source_id):
    """Creates a tracker in the output directory and registers the extraction job of the target source on it."""
    tracker = ProcessingTracker(file_path=output_directory / TRACKER_FILENAME)
    jobs = [(EXTRACTION_JOB_NAME, source_id)]
    tracker.align_jobs(jobs=jobs, universe=jobs)
    return tracker


def _occupy_thread():
    """Occupies a worker thread long enough for the manager to observe it as an in-flight job."""
    sleep(_ACTIVE_JOB_SECONDS)


@pytest.mark.xdist_group(name="orchestration")
def test_active_job_fields(tmp_path):
    """Verifies that an active job pairs the pending job descriptor with the thread executing it."""
    job = _build_job(directory=tmp_path, source_id="1")
    thread = Thread(target=lambda: None, daemon=True)

    active = _ActiveJob(job=job, thread=thread)

    assert active.job is job
    assert active.thread is thread


@pytest.mark.xdist_group(name="orchestration")
def test_select_admissible_jobs_empty_pending():
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
def test_select_admissible_jobs_admits_fitting_job(tmp_path):
    """Verifies that a job fitting both budgets is admitted alongside the jobs already running."""
    job = _build_job(directory=tmp_path, source_id="1", core_weight=2, memory_mb=1024)

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
def test_select_admissible_jobs_defers_job_without_cores(tmp_path):
    """Verifies that a job whose cores exceed the remaining budget is deferred even when memory is abundant."""
    job = _build_job(directory=tmp_path, source_id="1", core_weight=1, memory_mb=10)

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
def test_select_admissible_jobs_defers_job_without_memory(tmp_path):
    """Verifies that a job whose memory exceeds the remaining budget is deferred even when every core is free."""
    job = _build_job(directory=tmp_path, source_id="1", core_weight=1, memory_mb=512)

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
def test_select_admissible_jobs_forces_oversized_job_when_idle(tmp_path):
    """Verifies that a job wider than the whole budget is still admitted while nothing else is running."""
    oversized = _build_job(directory=tmp_path, source_id="1", core_weight=16, memory_mb=8192)

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
def test_select_admissible_jobs_backfills_past_unfitting_job(tmp_path):
    """Verifies that the scan continues past a job that does not fit, so a lighter job takes the spare capacity."""
    heavy = _build_job(directory=tmp_path, source_id="1", core_weight=8, memory_mb=1000)
    light = _build_job(directory=tmp_path, source_id="2", core_weight=2, memory_mb=500)

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
def test_select_admissible_jobs_considers_heaviest_first(tmp_path):
    """Verifies that the heaviest job is weighed against the budgets first, so it is placed ahead of a lighter one."""
    light = _build_job(directory=tmp_path, source_id="1", core_weight=1, memory_mb=1024)
    heavy = _build_job(directory=tmp_path, source_id="2", core_weight=1, memory_mb=2048)

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
def test_size_pending_job_resolves_archive(tmp_path):
    """Verifies that sizing a job backed by a real archive reads the archive and sets the job's weights in place."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)

    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="1"
    )
    job.core_weight = 99
    job.memory_mb = 99

    footprint = size_pending_job(job=job, core_budget=8)

    assert footprint.modeled
    assert footprint.message_count == 3
    assert footprint.archive_bytes > 0
    assert job.archive_path is not None
    assert job.archive_path.is_file()
    assert job.core_weight == 1
    assert job.memory_mb >= 1024
    assert job.memory_mb % 1024 == 0


@pytest.mark.xdist_group(name="orchestration")
def test_size_pending_job_missing_archive(tmp_path):
    """Verifies that sizing a job whose archive cannot be resolved falls back to the single-core baseline."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    job = _build_job(directory=log_directory, source_id="1", core_weight=99, memory_mb=99)

    footprint = size_pending_job(job=job, core_budget=8)

    assert not footprint.modeled
    assert footprint.message_count == 0
    assert footprint.archive_bytes == 0
    assert job.archive_path is None
    assert job.core_weight == 1
    assert job.memory_mb >= 1024


@pytest.mark.xdist_group(name="orchestration")
def test_group_jobs_by_tracker(tmp_path):
    """Verifies that every job in the registry is grouped under the tracker path that records it."""
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()

    first_jobs = [_build_job(directory=first_directory, source_id=str(index)) for index in (1, 2, 3)]
    second_jobs = [_build_job(directory=second_directory, source_id=str(index)) for index in (4, 5)]

    state = JobExecutionState()
    for job in first_jobs + second_jobs:
        state.all_jobs[job.dispatch_key] = job

    grouped = group_jobs_by_tracker(state=state)

    assert set(grouped.keys()) == {first_directory / TRACKER_FILENAME, second_directory / TRACKER_FILENAME}
    assert grouped[first_directory / TRACKER_FILENAME] == first_jobs
    assert grouped[second_directory / TRACKER_FILENAME] == second_jobs


@pytest.mark.xdist_group(name="orchestration")
def test_execution_state_round_trip(execution_state_guard):
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
def test_job_execution_manager_empty_queues_exit_immediately():
    """Verifies that the manager returns at once when the queue and the running set are both empty."""
    state = JobExecutionState(core_budget=4, memory_budget_mb=8192)

    job_execution_manager(state=state)

    assert state.pending_jobs == []
    assert state.active_jobs == []


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_dispatches_pending_jobs(tmp_path):
    """Verifies that the manager dispatches every queued job and drains its queues once the jobs finish."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)
    tracker = _initialize_tracker(output_directory=output_directory, source_id="1")

    state = JobExecutionState(core_budget=4, memory_budget_mb=8192)
    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="1"
    )
    size_pending_job(job=job, core_budget=state.core_budget)
    state.all_jobs[job.dispatch_key] = job
    state.pending_jobs.append(job)

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    manager.join(timeout=_MANAGER_TIMEOUT_SECONDS)

    assert not manager.is_alive()
    assert state.pending_jobs == []
    assert state.active_jobs == []
    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.SUCCEEDED
    assert resolve_module_feather_path(
        output_directory=output_directory, source_id="1", module_type=1, module_id=2
    ).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_cancellation_skips_dispatch(tmp_path):
    """Verifies that a canceled session exits without dispatching any of the jobs still queued for execution."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)
    tracker = _initialize_tracker(output_directory=output_directory, source_id="1")

    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="1"
    )
    size_pending_job(job=job, core_budget=4)

    state = JobExecutionState(core_budget=4, memory_budget_mb=8192, canceled=True)
    state.all_jobs[job.dispatch_key] = job
    state.pending_jobs.append(job)

    job_execution_manager(state=state)

    assert state.pending_jobs == [job]
    assert state.active_jobs == []
    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.SCHEDULED
    assert not resolve_module_feather_path(
        output_directory=output_directory, source_id="1", module_type=1, module_id=2
    ).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_job_execution_manager_cancellation_waits_for_active_jobs(tmp_path):
    """Verifies that a canceled session keeps polling until the jobs already in flight finish."""
    job = _build_job(directory=tmp_path, source_id="1")
    thread = Thread(target=_occupy_thread, daemon=True)
    thread.start()

    state = JobExecutionState(core_budget=4, memory_budget_mb=8192, canceled=True)
    state.all_jobs[job.dispatch_key] = job
    state.active_jobs.append(_ActiveJob(job=job, thread=thread))

    job_execution_manager(state=state)

    assert not thread.is_alive()
    assert state.active_jobs == []
    assert state.pending_jobs == []


@pytest.mark.xdist_group(name="orchestration")
def test_run_job_extracts_configured_data(tmp_path):
    """Verifies that running an admitted job writes its feather output and completes the job on the tracker."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)
    tracker = _initialize_tracker(output_directory=output_directory, source_id="1")

    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="1"
    )
    size_pending_job(job=job, core_budget=1)

    _run_job(job=job)

    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.SUCCEEDED
    assert resolve_module_feather_path(
        output_directory=output_directory, source_id="1", module_type=1, module_id=2
    ).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_run_job_marks_unreadable_config_as_failed(tmp_path):
    """Verifies that a job whose extraction config cannot be read is recorded as failed without any extraction."""
    log_directory, _, output_directory = setup_test_environment(tmp_path=tmp_path)
    tracker = _initialize_tracker(output_directory=output_directory, source_id="1")

    missing_config = tmp_path / "missing_config.yaml"
    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=missing_config, source_id="1"
    )

    _run_job(job=job)

    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job.job_id).error_message == (
        f"Unable to load the extraction config from '{missing_config}'."
    )
    assert not resolve_module_feather_path(
        output_directory=output_directory, source_id="1", module_type=1, module_id=2
    ).exists()


@pytest.mark.xdist_group(name="orchestration")
def test_run_job_marks_unconfigured_source_as_failed(tmp_path):
    """Verifies that a job whose source is absent from the extraction config is recorded as failed."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)
    tracker = _initialize_tracker(output_directory=output_directory, source_id="2")

    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="2"
    )

    _run_job(job=job)

    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job.job_id).error_message == "No controller config for source '2'."


@pytest.mark.xdist_group(name="orchestration")
def test_run_job_marks_unresolvable_archive_as_failed(tmp_path):
    """Verifies that a job whose archive cannot be resolved is recorded as failed rather than left unfinished."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    config_path = write_extraction_config(config_path=tmp_path / "config.yaml", source_id=1)
    tracker = _initialize_tracker(output_directory=output_directory, source_id="1")

    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="1"
    )

    _run_job(job=job)

    assert job.archive_path is None
    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job.job_id).error_message == "Job terminated without updating tracker status."


@pytest.mark.xdist_group(name="orchestration")
def test_run_job_survives_unreadable_tracker(tmp_path):
    """Verifies that a job whose tracker file cannot be deserialized returns instead of raising out of its thread."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)

    tracker_path = output_directory / TRACKER_FILENAME
    tracker_path.write_text("- not a tracker mapping\n")

    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="1"
    )
    size_pending_job(job=job, core_budget=1)

    _run_job(job=job)

    assert tracker_path.read_text() == "- not a tracker mapping\n"


@pytest.mark.xdist_group(name="orchestration")
def test_resolve_controller_config_returns_matching_controller(tmp_path):
    """Verifies that the controller whose identifier matches the job's source is returned from the config."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)
    tracker = _initialize_tracker(output_directory=output_directory, source_id="1")

    job = _build_registered_job(
        log_directory=log_directory, output_directory=output_directory, config_path=config_path, source_id="1"
    )

    controller_config = _resolve_controller_config(job=job, tracker=tracker)

    assert controller_config is not None
    assert str(controller_config.controller_id) == "1"
    assert controller_config.modules[0].module_type == 1
    assert controller_config.modules[0].module_id == 2
    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.SCHEDULED
