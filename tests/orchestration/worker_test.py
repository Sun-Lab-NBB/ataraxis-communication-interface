"""Contains tests for functions provided by the orchestration/worker.py module."""

from concurrent.futures import ProcessPoolExecutor

import polars as pl
import pytest
from tests.log_archives import (
    create_test_archive,
    write_extraction_config,
    make_kernel_data_payload,
    make_module_data_payload,
    make_kernel_state_payload,
    make_module_state_payload,
)
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    PARALLEL_PROCESSING_THRESHOLD,
    ProcessingStatus,
    ProcessingTracker,
)

from ataraxis_communication_interface.communication import SerialPrototypes
from ataraxis_communication_interface.orchestration import worker as worker_module
from ataraxis_communication_interface.orchestration.jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    resolve_kernel_path,
    resolve_module_path,
    resolve_tracker_path,
)
from ataraxis_communication_interface.orchestration.worker import (
    execute_job,
    run_extraction_job,
    resolve_controller_config,
)
from ataraxis_communication_interface.microcontroller.dataclasses import (
    ExtractionConfig,
    KernelExtractionConfig,
    ControllerExtractionConfig,
)
from ataraxis_communication_interface.microcontroller.extracted_data import ExtractedDataColumns

_SOURCE_ID: str = "1"
"""Stores the controller source identifier used by every synthetic log archive built by this module."""

_MODULE_TYPE: int = 1
"""Stores the type (family) code of the hardware module every synthetic archive logs messages for."""

_MODULE_ID: int = 2
"""Stores the identifier code of the hardware module every synthetic archive logs messages for."""

_MODULE_EVENT_CODES: tuple[int, ...] = (10, 20)
"""Stores the module event codes every extraction configuration built by this module requests."""

_KERNEL_EVENT_CODES: tuple[int, ...] = (5,)
"""Stores the kernel event codes every configuration exercising kernel extraction requests."""


class _CountingExecutor(ProcessPoolExecutor):
    """Wraps a real process pool with a counter that records how many batches were submitted to it."""

    def __init__(self, max_workers):
        super().__init__(max_workers=max_workers)
        self.submissions = 0

    def submit(self, fn, /, *args, **kwargs):
        """Records the submission before handing the work to the underlying pool."""
        self.submissions += 1
        return super().submit(fn, *args, **kwargs)


def _module_messages():
    """Creates the module state and data messages the synthetic archives of this module hold."""
    return [
        (1000, make_module_state_payload(module_type=_MODULE_TYPE, module_id=_MODULE_ID, command=1, event=10)),
        (
            2000,
            make_module_data_payload(
                module_type=_MODULE_TYPE,
                module_id=_MODULE_ID,
                command=2,
                event=20,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[42],
            ),
        ),
    ]


def _kernel_messages():
    """Creates the kernel state and data messages the synthetic archives exercising kernel extraction hold."""
    return [
        (3000, make_kernel_state_payload(command=1, event=5)),
        (
            4000,
            make_kernel_data_payload(
                command=2,
                event=5,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[7],
            ),
        ),
    ]


def _build_archive(log_directory, messages=None, source_id=_SOURCE_ID):
    """Creates one synthetic log archive for the requested controller source and returns the path it was written to."""
    log_directory.mkdir(parents=True, exist_ok=True)
    archive_path = log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    create_test_archive(
        archive_path=archive_path,
        source_id=int(source_id),
        messages=_module_messages() if messages is None else messages,
    )
    return archive_path


def _write_config(config_path, source_id=_SOURCE_ID, event_codes=_MODULE_EVENT_CODES, kernel_event_codes=None):
    """Writes the extraction configuration declaring one module, and optionally the kernel, for the target source."""
    return write_extraction_config(
        config_path=config_path,
        source_id=int(source_id),
        module_type=_MODULE_TYPE,
        module_id=_MODULE_ID,
        event_codes=event_codes,
        kernel_event_codes=kernel_event_codes,
    )


def _write_module_free_config(config_path, source_id=_SOURCE_ID, kernel_event_codes=None):
    """Writes an extraction configuration that declares no module, which the shared builder cannot express."""
    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=int(source_id),
                modules=(),
                kernel=None if kernel_event_codes is None else KernelExtractionConfig(event_codes=kernel_event_codes),
            )
        ]
    )
    config.to_yaml(file_path=config_path)
    return config_path


def _initialize_tracker(tracker_path, source_id=_SOURCE_ID):
    """Creates a processing tracker that already registers the extraction job of the target controller source."""
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.initialize_jobs(jobs=[(CONTROLLER_EXTRACTION_JOB_NAME, source_id)])
    return tracker


def _build_descriptor(log_directory, output_directory, config_path, source_id=_SOURCE_ID, core_weight=1):
    """Builds the descriptor of the extraction job reading the target source's archive under the log directory."""
    return JobDescriptor.for_archive(
        archive_path=log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
        output_directory=output_directory,
        config_path=config_path,
        tracker_path=resolve_tracker_path(output_directory=output_directory),
        source_id=source_id,
        log_directory=log_directory,
        core_weight=core_weight,
    )


def _module_path(output_directory, source_id=_SOURCE_ID):
    """Resolves the path of the module output file the shared synthetic module's messages are written to."""
    return resolve_module_path(
        output_directory=output_directory,
        source_id=source_id,
        module_type=_MODULE_TYPE,
        module_id=_MODULE_ID,
    )


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_writes_the_module_file_and_completes_the_job(tmp_path):
    """Verifies that execute_job writes the module file at the resolved path and completes the tracked job."""
    archive_path = _build_archive(log_directory=tmp_path / "logs")
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=_SOURCE_ID,
        job_id=job_id,
        workers=1,
        tracker=tracker,
        config_path=config_path,
        display_progress=False,
    )

    # The output file is written at the path the layout resolver names, and nowhere else.
    feather_path = _module_path(output_directory=output_directory)
    assert feather_path.is_file()
    assert feather_path.name == f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather"

    # The kernel file is only written when kernel extraction is configured, which it is not here.
    assert not resolve_kernel_path(output_directory=output_directory, source_id=_SOURCE_ID).exists()

    # The columns carry the names the extracted data columns enumeration declares, in its own storage order.
    dataframe = pl.read_ipc(source=feather_path)
    assert dataframe.columns == [str(column) for column in ExtractedDataColumns]
    assert dataframe.columns == ["timestamp_us", "command", "event", "dtype", "data"]
    assert dataframe[str(ExtractedDataColumns.EVENT)].to_list() == [10, 20]
    assert dataframe[str(ExtractedDataColumns.DTYPE)].to_list() == [None, "uint8"]
    assert dataframe[str(ExtractedDataColumns.DATA)].to_list() == [None, b"\x2a"]

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_writes_the_kernel_file_when_only_the_kernel_is_configured(tmp_path):
    """Verifies that execute_job writes the kernel file at the resolved path and no module file."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", messages=_module_messages() + _kernel_messages())
    config_path = _write_module_free_config(
        config_path=tmp_path / "config.yaml", kernel_event_codes=_KERNEL_EVENT_CODES
    )

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=_SOURCE_ID,
        job_id=job_id,
        workers=1,
        tracker=tracker,
        config_path=config_path,
        display_progress=False,
    )

    kernel_path = resolve_kernel_path(output_directory=output_directory, source_id=_SOURCE_ID)
    assert kernel_path.is_file()
    assert kernel_path.name == "controller_1_kernel.feather"
    assert not _module_path(output_directory=output_directory).exists()

    dataframe = pl.read_ipc(source=kernel_path)
    assert dataframe.columns == [str(column) for column in ExtractedDataColumns]
    assert dataframe[str(ExtractedDataColumns.EVENT)].to_list() == [5, 5]
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_writes_both_files_when_both_targets_are_configured(tmp_path):
    """Verifies that execute_job writes one module file and the kernel file when the config declares both."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", messages=_module_messages() + _kernel_messages())
    config_path = _write_config(config_path=tmp_path / "config.yaml", kernel_event_codes=_KERNEL_EVENT_CODES)

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=_SOURCE_ID,
        job_id=job_id,
        workers=1,
        tracker=tracker,
        config_path=config_path,
        display_progress=False,
    )

    assert len(pl.read_ipc(source=_module_path(output_directory=output_directory))) == 2
    assert len(pl.read_ipc(source=resolve_kernel_path(output_directory=output_directory, source_id=_SOURCE_ID))) == 2
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_writes_no_file_for_an_archive_without_messages(tmp_path):
    """Verifies that execute_job writes no output file when the archive holds no data message at all."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", messages=[])
    config_path = _write_config(config_path=tmp_path / "config.yaml", kernel_event_codes=_KERNEL_EVENT_CODES)

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=_SOURCE_ID,
        job_id=job_id,
        workers=1,
        tracker=tracker,
        config_path=config_path,
        display_progress=False,
    )

    # A target that produced no message contributes no file, so a silent recording completes with an empty directory.
    assert list(output_directory.glob(f"*{OutputLayout.FILE_SUFFIX}")) == []
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_records_the_failure_message_on_the_tracker(tmp_path):
    """Verifies that a failing extraction marks the job failed with the exception's message and re-raises."""
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    missing_archive = tmp_path / "nonexistent.npz"
    message = (
        f"Unable to extract microcontroller message data from the log file {missing_archive}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)) as error_info:
        execute_job(
            log_path=missing_archive,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
        )

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED

    # The tracker records the message of the exception that failed the job, rather than a generic failure note.
    assert tracker.get_job_info(job_id=job_id).error_message == str(error_info.value)

    # A failed extraction writes no output file.
    assert list(output_directory.glob(f"*{OutputLayout.FILE_SUFFIX}")) == []


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_records_a_configuration_error_on_the_tracker(tmp_path):
    """Verifies that a config declaring no entry for the job's source fails the job rather than escaping it."""
    archive_path = _build_archive(log_directory=tmp_path / "logs")

    # The configuration declares another controller entirely, so resolving this job's own entry fails.
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_id="2")

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    message = (
        f"Unable to execute the data extraction job for source '{_SOURCE_ID}'. The extraction config at "
        f"'{config_path}' declares no entry for that controller. Configured controller IDs: 2."
    )
    with pytest.raises(ValueError, match=error_format(message)) as error_info:
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
        )

    # The configuration is resolved inside the tracker's run_job() block, so a configuration error is recorded against
    # the job instead of leaving it in its initial state.
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job_id).error_message == str(error_info.value)


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_records_an_empty_module_filter_on_the_tracker(tmp_path):
    """Verifies that a module declaring empty event codes fails the job with the message naming that module."""
    archive_path = _build_archive(log_directory=tmp_path / "logs")
    config_path = _write_config(config_path=tmp_path / "config.yaml", event_codes=())

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    message = (
        f"Unable to execute the data extraction job for source '{_SOURCE_ID}'. Module with type code {_MODULE_TYPE} "
        f"and ID code {_MODULE_ID} has empty event_codes."
    )
    with pytest.raises(ValueError, match=error_format(message)) as error_info:
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
        )

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job_id).error_message == str(error_info.value)


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_records_an_empty_kernel_filter_on_the_tracker(tmp_path):
    """Verifies that a configured kernel declaring empty event codes fails the job."""
    archive_path = _build_archive(log_directory=tmp_path / "logs")
    config_path = _write_config(config_path=tmp_path / "config.yaml", kernel_event_codes=())

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    message = (
        f"Unable to execute the data extraction job for source '{_SOURCE_ID}'. Kernel extraction has empty event_codes."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
        )

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_records_a_targetless_configuration_on_the_tracker(tmp_path):
    """Verifies that a controller entry declaring neither a module nor the kernel fails the job."""
    archive_path = _build_archive(log_directory=tmp_path / "logs")
    config_path = _write_module_free_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    message = (
        f"Unable to execute the data extraction job for source '{_SOURCE_ID}'. The controller config has no modules "
        f"and no kernel extraction configured."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
        )

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_creates_the_output_directory(tmp_path):
    """Verifies that execute_job publishes its output files into an output directory that does not exist yet."""
    archive_path = _build_archive(log_directory=tmp_path / "logs")
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    tracker = _initialize_tracker(tracker_path=tmp_path / "tracker" / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    # The output directory is deliberately left uncreated, as the atomic publish creates it on demand.
    output_directory = tmp_path / "output"

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=_SOURCE_ID,
        job_id=job_id,
        workers=1,
        tracker=tracker,
        config_path=config_path,
        display_progress=False,
    )

    assert _module_path(output_directory=output_directory).is_file()
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_does_not_register_tracker_jobs(tmp_path):
    """Verifies that execute_job never registers its job on the tracker it is handed, leaving it to its caller."""
    archive_path = _build_archive(log_directory=tmp_path / "logs")
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output"
    output_directory.mkdir()

    # The tracker is deliberately left without any registered job, as registering the jobs is the caller's
    # responsibility.
    tracker_path = output_directory / OutputLayout.TRACKER_FILENAME
    tracker = ProcessingTracker(file_path=tracker_path)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    message = (
        f"Unable to mark the job with ID '{job_id}' as running using the processing tracker at '{tracker_path}'. The "
        f"requested job must be tracked by the instance, but the instance is not configured to track it. The instance "
        f"is currently configured to track jobs with IDs: ."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
        )

    assert ProcessingTracker(file_path=tracker_path).snapshot() == {}
    assert list(output_directory.glob(f"*{OutputLayout.FILE_SUFFIX}")) == []


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_reuses_the_provided_executor(tmp_path):
    """Verifies that execute_job submits its batch work to the caller's executor and leaves that executor usable."""
    messages = [
        (index * 10, make_module_state_payload(module_type=_MODULE_TYPE, module_id=_MODULE_ID, command=1, event=10))
        for index in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    archive_path = _build_archive(log_directory=tmp_path / "logs", messages=messages)
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    executor = _CountingExecutor(max_workers=2)
    try:
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=2,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
            executor=executor,
        )

        # The batch work reached the caller's pool instead of a pool the extraction opened for itself.
        assert executor.submissions > 0

        # The caller owns the pool, so the job must not shut it down. A pool closed by the job would instead raise
        # a RuntimeError when asked to accept more work.
        assert executor.submit(abs, -5).result() == 5
    finally:
        executor.shutdown(wait=True)

    assert len(pl.read_ipc(source=_module_path(output_directory=output_directory))) == PARALLEL_PROCESSING_THRESHOLD
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_leaves_the_executor_untouched_when_running_serially(tmp_path):
    """Verifies that a single-worker job processes its archive without submitting anything to the caller's pool."""
    messages = [
        (index * 10, make_module_state_payload(module_type=_MODULE_TYPE, module_id=_MODULE_ID, command=1, event=10))
        for index in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    archive_path = _build_archive(log_directory=tmp_path / "logs", messages=messages)
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output"
    tracker = _initialize_tracker(tracker_path=output_directory / OutputLayout.TRACKER_FILENAME)
    job_id = generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]

    executor = _CountingExecutor(max_workers=1)
    try:
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id=_SOURCE_ID,
            job_id=job_id,
            workers=1,
            tracker=tracker,
            config_path=config_path,
            display_progress=False,
            executor=executor,
        )

        assert executor.submissions == 0
    finally:
        executor.shutdown(wait=True)

    assert len(pl.read_ipc(source=_module_path(output_directory=output_directory))) == PARALLEL_PROCESSING_THRESHOLD
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_extraction_job_runs_the_job_from_its_descriptor(tmp_path):
    """Verifies that run_extraction_job runs the described job and records its outcome on the descriptor's tracker."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory)
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(log_directory=log_directory, output_directory=output_directory, config_path=config_path)

    # The tracker is aligned by the caller that prepared the job, exactly as the preparation stage does it.
    _initialize_tracker(tracker_path=job.tracker_path)

    run_extraction_job(job=job)

    feather_path = _module_path(output_directory=job.output_directory, source_id=job.source_id)
    assert feather_path.is_file()

    dataframe = pl.read_ipc(source=feather_path)
    assert dataframe.columns == [str(column) for column in ExtractedDataColumns]
    assert dataframe[str(ExtractedDataColumns.EVENT)].to_list() == [10, 20]

    # The status is read through a tracker instance the test opens itself, which is the same file the descriptor
    # names. The runner therefore recorded the outcome at the descriptor's own tracker path.
    assert job.job_id == generate_job_ids(source_ids=[_SOURCE_ID])[_SOURCE_ID]
    assert ProcessingTracker(file_path=job.tracker_path).get_job_status(job_id=job.job_id) == (
        ProcessingStatus.SUCCEEDED
    )


@pytest.mark.xdist_group(name="orchestration")
def test_run_extraction_job_records_a_failure_on_the_descriptor_tracker(tmp_path):
    """Verifies that a job whose archive is absent fails on the tracker the descriptor names and re-raises."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(log_directory=log_directory, output_directory=output_directory, config_path=config_path)
    _initialize_tracker(tracker_path=job.tracker_path)

    # The archive the descriptor names was never written, so the extraction raises inside the runner.
    assert not job.archive_path.exists()

    message = (
        f"Unable to extract microcontroller message data from the log file {job.archive_path}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)) as error_info:
        run_extraction_job(job=job)

    tracker = ProcessingTracker(file_path=job.tracker_path)
    assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
    assert tracker.get_job_info(job_id=job.job_id).error_message == str(error_info.value)


def test_run_extraction_job_forwards_every_descriptor_field(tmp_path, monkeypatch):
    """Verifies that run_extraction_job derives every execute_job argument from the descriptor alone."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory)
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(
        log_directory=log_directory,
        output_directory=output_directory,
        config_path=config_path,
        core_weight=4,
    )
    _initialize_tracker(tracker_path=job.tracker_path)

    calls = []

    def _record_call(**kwargs):
        """Records the arguments the runner derived from the descriptor instead of running the extraction."""
        calls.append(kwargs)

    monkeypatch.setattr(worker_module, "execute_job", _record_call)

    run_extraction_job(job=job)

    assert len(calls) == 1
    arguments = calls[0]
    assert arguments["log_path"] == job.archive_path
    assert arguments["output_directory"] == job.output_directory
    assert arguments["source_id"] == job.source_id
    assert arguments["job_id"] == job.job_id

    # The configuration a job reads its targets from travels on the descriptor, since a worker resolves it itself.
    assert arguments["config_path"] == job.config_path

    # The width the descriptor carries becomes the width of the extraction pool the job's body opens.
    assert arguments["workers"] == job.core_weight == 4

    # The tracker is opened by the runner from the descriptor's own path, because a tracker's file lock cannot cross
    # a process boundary.
    assert isinstance(arguments["tracker"], ProcessingTracker)
    assert arguments["tracker"].file_path == job.tracker_path

    # A pooled job has no console to draw on, and it never receives an outer pool to nest inside.
    assert arguments["display_progress"] is False
    assert "executor" not in arguments


@pytest.mark.xdist_group(name="orchestration")
def test_run_extraction_job_runs_inside_a_process_pool(tmp_path):
    """Verifies that the runner and its descriptor both pickle into a spawned worker and complete the job there."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory)
    config_path = _write_config(config_path=tmp_path / "config.yaml")

    output_directory = tmp_path / "output" / OutputLayout.DIRECTORY_NAME
    job = _build_descriptor(log_directory=log_directory, output_directory=output_directory, config_path=config_path)
    _initialize_tracker(tracker_path=job.tracker_path)

    executor = ProcessPoolExecutor(max_workers=1)
    try:
        assert executor.submit(run_extraction_job, job=job).result() is None
    finally:
        executor.shutdown(wait=True)

    feather_path = _module_path(output_directory=job.output_directory, source_id=job.source_id)
    assert pl.read_ipc(source=feather_path)[str(ExtractedDataColumns.EVENT)].to_list() == [10, 20]
    assert ProcessingTracker(file_path=job.tracker_path).get_job_status(job_id=job.job_id) == (
        ProcessingStatus.SUCCEEDED
    )


def test_resolve_controller_config_returns_the_requested_entry(tmp_path):
    """Verifies that resolve_controller_config returns the entry the config declares for the requested controller."""
    config_path = _write_config(config_path=tmp_path / "config.yaml", kernel_event_codes=_KERNEL_EVENT_CODES)

    controller_config = resolve_controller_config(config_path=config_path, source_id=_SOURCE_ID)

    # The source identifier is a string everywhere in the job layer, while the config stores it as an integer.
    assert controller_config.controller_id == int(_SOURCE_ID)
    assert controller_config.modules[0].module_type == _MODULE_TYPE
    assert controller_config.modules[0].module_id == _MODULE_ID
    assert tuple(controller_config.modules[0].event_codes) == _MODULE_EVENT_CODES
    assert tuple(controller_config.kernel.event_codes) == _KERNEL_EVENT_CODES


def test_resolve_controller_config_unconfigured_controller(tmp_path):
    """Verifies that resolve_controller_config raises ValueError naming every controller the config declares."""
    config_path = tmp_path / "config.yaml"
    ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=controller_id,
                modules=(),
                kernel=KernelExtractionConfig(event_codes=_KERNEL_EVENT_CODES),
            )
            for controller_id in (3, 2)
        ]
    ).to_yaml(file_path=config_path)

    # The configured identifiers are reported in sorted order, rather than in the order the config declares them.
    message = (
        f"Unable to execute the data extraction job for source '{_SOURCE_ID}'. The extraction config at "
        f"'{config_path}' declares no entry for that controller. Configured controller IDs: 2, 3."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        resolve_controller_config(config_path=config_path, source_id=_SOURCE_ID)


def test_resolve_controller_config_missing_config_file(tmp_path):
    """Verifies that resolve_controller_config raises FileNotFoundError when the configuration file is absent."""
    config_path = tmp_path / "missing_config.yaml"
    assert not config_path.exists()

    with pytest.raises(FileNotFoundError):
        resolve_controller_config(config_path=config_path, source_id=_SOURCE_ID)
