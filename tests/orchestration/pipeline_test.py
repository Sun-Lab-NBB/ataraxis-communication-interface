"""Contains tests for functions provided by the orchestration/pipeline.py module."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import pytest
from tests.log_archives import (
    create_test_archive,
    setup_test_environment,
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
from ataraxis_communication_interface.orchestration.jobs import (
    TRACKER_FILENAME,
    EXTRACTION_JOB_NAME,
    MICROCONTROLLER_DATA_DIRECTORY,
    generate_job_ids,
)
from ataraxis_communication_interface.orchestration.pipeline import (
    execute_job,
    _write_messages,
    _execute_sized_job,
    _resolve_event_filters,
    _resolve_controller_configs,
    run_log_processing_pipeline,
)
from ataraxis_communication_interface.microcontroller.dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    ModuleSourceData,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    ControllerExtractionConfig,
    write_microcontroller_manifest,
)
from ataraxis_communication_interface.microcontroller.log_processing import ExtractedMessages

_MODULE_TYPE = 1
"""The type (family) code of the hardware module every synthetic archive built in this module logs messages for."""

_MODULE_ID = 2
"""The identifier code of the hardware module every synthetic archive built in this module logs messages for."""

_MODULE_EVENT_CODES = (10, 20)
"""The module event codes every extraction configuration built in this module requests."""

_KERNEL_EVENT_CODES = (5,)
"""The kernel event codes every extraction configuration exercising kernel extraction requests."""


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


def _build_archive(log_directory, source_id, messages):
    """Creates the log archive of the requested source under the target directory and returns its path."""
    log_directory.mkdir(parents=True, exist_ok=True)
    archive_path = log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)
    return archive_path


def _write_manifest(log_directory, source_ids):
    """Writes a microcontroller manifest registering one single-module controller for each requested source."""
    for source_id in source_ids:
        write_microcontroller_manifest(
            log_directory=log_directory,
            controller_id=source_id,
            controller_name=f"controller_{source_id}",
            modules=(ModuleSourceData(module_type=_MODULE_TYPE, module_id=_MODULE_ID, name="test_module"),),
        )


def _write_config(config_path, source_ids, kernel_event_codes=None):
    """Writes an extraction configuration requesting the shared module and kernel filters for each source."""
    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=source_id,
                modules=(
                    ModuleExtractionConfig(
                        module_type=_MODULE_TYPE, module_id=_MODULE_ID, event_codes=_MODULE_EVENT_CODES
                    ),
                ),
                kernel=None if kernel_event_codes is None else KernelExtractionConfig(event_codes=kernel_event_codes),
            )
            for source_id in source_ids
        ]
    )
    config.to_yaml(file_path=config_path)
    return config_path


def _build_controller_config(modules=(), kernel_event_codes=None):
    """Creates a controller extraction configuration from the requested module and kernel filters."""
    return ControllerExtractionConfig(
        controller_id=1,
        modules=modules,
        kernel=None if kernel_event_codes is None else KernelExtractionConfig(event_codes=kernel_event_codes),
    )


def _module_config(event_codes=_MODULE_EVENT_CODES):
    """Creates the module extraction configuration tuple the shared synthetic module is extracted with."""
    return (ModuleExtractionConfig(module_type=_MODULE_TYPE, module_id=_MODULE_ID, event_codes=event_codes),)


def _initialize_tracker(tracker_path, source_id):
    """Creates a processing tracker that already registers the extraction job of the target source."""
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.initialize_jobs(jobs=[(EXTRACTION_JOB_NAME, source_id)])
    return tracker


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_module_extraction(tmp_path):
    """Verifies that execute_job writes one module feather file and completes the job when modules are requested."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=_module_messages())

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=_build_controller_config(modules=_module_config()),
        display_progress=False,
    )

    feather_path = output_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather"
    assert feather_path.is_file()
    assert not (output_directory / "controller_1_kernel.feather").exists()

    dataframe = pl.read_ipc(source=feather_path)
    assert dataframe.columns == ["timestamp_us", "command", "event", "dtype", "data"]
    assert dataframe["event"].to_list() == [10, 20]
    assert dataframe["dtype"].to_list() == [None, "uint8"]
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_kernel_extraction(tmp_path):
    """Verifies that execute_job writes only the kernel feather file when no modules are configured."""
    archive_path = _build_archive(
        log_directory=tmp_path / "logs", source_id=1, messages=_module_messages() + _kernel_messages()
    )

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=_build_controller_config(kernel_event_codes=_KERNEL_EVENT_CODES),
        display_progress=False,
    )

    kernel_path = output_directory / "controller_1_kernel.feather"
    assert kernel_path.is_file()
    assert not (output_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather").exists()

    dataframe = pl.read_ipc(source=kernel_path)
    assert dataframe["event"].to_list() == [5, 5]
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_module_and_kernel_extraction(tmp_path):
    """Verifies that execute_job writes both the module and the kernel feather files when both are configured."""
    archive_path = _build_archive(
        log_directory=tmp_path / "logs", source_id=1, messages=_module_messages() + _kernel_messages()
    )

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=_build_controller_config(modules=_module_config(), kernel_event_codes=_KERNEL_EVENT_CODES),
        display_progress=False,
    )

    module_path = output_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather"
    kernel_path = output_directory / "controller_1_kernel.feather"
    assert len(pl.read_ipc(source=module_path)) == 2
    assert len(pl.read_ipc(source=kernel_path)) == 2
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_empty_archive_writes_no_file(tmp_path):
    """Verifies that execute_job writes no feather file when the archive holds no data messages."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=[])

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=_build_controller_config(modules=_module_config(), kernel_event_codes=_KERNEL_EVENT_CODES),
        display_progress=False,
    )

    assert list(output_directory.glob("*.feather")) == []
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_empty_module_event_codes(tmp_path):
    """Verifies that execute_job raises ValueError when a configured module declares empty event codes."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=_module_messages())

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    message = (
        f"Unable to execute the data extraction job for source '1'. Module with type code {_MODULE_TYPE} and ID code "
        f"{_MODULE_ID} has empty event_codes."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=_build_controller_config(modules=_module_config(event_codes=())),
            display_progress=False,
        )

    # The configuration is validated before the tracker opens the job, so the job never leaves its initial state.
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_empty_kernel_event_codes(tmp_path):
    """Verifies that execute_job raises ValueError when the configured kernel extraction has empty event codes."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=_module_messages())

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    message = "Unable to execute the data extraction job for source '1'. Kernel extraction has empty event_codes."
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=_build_controller_config(modules=_module_config(), kernel_event_codes=()),
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_no_extraction_targets(tmp_path):
    """Verifies that execute_job raises ValueError when the controller config declares no extraction targets."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=_module_messages())

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    message = (
        "Unable to execute the data extraction job for source '1'. The controller config has no modules and no "
        "kernel extraction configured."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=_build_controller_config(),
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_invalid_archive_path(tmp_path):
    """Verifies that execute_job marks the job as failed and re-raises when the archive path does not resolve."""
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    missing_archive = tmp_path / "nonexistent.npz"
    message = (
        f"Unable to extract microcontroller message data from the log file {missing_archive}, as it does not exist "
        f"or does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=missing_archive,
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=_build_controller_config(modules=_module_config()),
            display_progress=False,
        )

    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_missing_output_directory(tmp_path):
    """Verifies that execute_job publishes its feather files into an output directory that does not exist yet."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=_module_messages())

    tracker_directory = tmp_path / "tracker"
    tracker_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=tracker_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    # The output directory is deliberately left uncreated, as the atomic feather publish creates it on demand.
    output_directory = tmp_path / "output"

    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=_build_controller_config(modules=_module_config()),
        display_progress=False,
    )

    assert (output_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather").is_file()
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_does_not_register_tracker_jobs(tmp_path):
    """Verifies that execute_job never registers its job on the tracker it is handed, leaving it to its caller."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=_module_messages())

    output_directory = tmp_path / "output"
    output_directory.mkdir()

    # The tracker is deliberately left without any registered job, as registering the jobs is the caller's
    # responsibility.
    tracker_path = output_directory / TRACKER_FILENAME
    tracker = ProcessingTracker(file_path=tracker_path)
    job_id = generate_job_ids(source_ids=["1"])["1"]

    message = (
        f"Unable to mark the job with ID '{job_id}' as running using the processing tracker at '{tracker_path}'. The "
        f"requested job must be tracked by the instance, but the instance is not configured to track it. The instance "
        f"is currently configured to track jobs with IDs: ."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=_build_controller_config(modules=_module_config()),
            display_progress=False,
        )

    assert ProcessingTracker(file_path=tracker_path).snapshot() == {}
    assert list(output_directory.glob("*.feather")) == []


@pytest.mark.xdist_group(name="orchestration")
def test_execute_job_external_executor(tmp_path):
    """Verifies that execute_job passes a caller-owned process pool through to the parallel extraction path."""
    messages = [
        (index * 10, make_module_state_payload(module_type=_MODULE_TYPE, module_id=_MODULE_ID, command=1, event=10))
        for index in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=messages)

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    with ProcessPoolExecutor(max_workers=2) as executor:
        execute_job(
            log_path=archive_path,
            output_directory=output_directory,
            source_id="1",
            job_id=job_id,
            workers=2,
            tracker=tracker,
            controller_config=_build_controller_config(modules=_module_config()),
            display_progress=False,
            executor=executor,
        )

    feather_path = output_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather"
    assert len(pl.read_ipc(source=feather_path)) == PARALLEL_PROCESSING_THRESHOLD
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


def test_resolve_event_filters_modules_and_kernel():
    """Verifies that _resolve_event_filters unpacks both the per-module and the kernel event code filters."""
    module_filters, kernel_event_codes = _resolve_event_filters(
        controller_config=_build_controller_config(modules=_module_config(), kernel_event_codes=_KERNEL_EVENT_CODES),
        source_id="1",
    )

    assert module_filters == {(_MODULE_TYPE, _MODULE_ID): frozenset(_MODULE_EVENT_CODES)}
    assert kernel_event_codes == frozenset(_KERNEL_EVENT_CODES)


def test_resolve_event_filters_modules_only():
    """Verifies that _resolve_event_filters returns a None kernel filter when kernel extraction is not configured."""
    module_filters, kernel_event_codes = _resolve_event_filters(
        controller_config=_build_controller_config(modules=_module_config()), source_id="1"
    )

    assert module_filters == {(_MODULE_TYPE, _MODULE_ID): frozenset(_MODULE_EVENT_CODES)}
    assert kernel_event_codes is None


def test_resolve_event_filters_kernel_only():
    """Verifies that _resolve_event_filters returns a None module filter when no modules are configured."""
    module_filters, kernel_event_codes = _resolve_event_filters(
        controller_config=_build_controller_config(kernel_event_codes=_KERNEL_EVENT_CODES), source_id="1"
    )

    assert module_filters is None
    assert kernel_event_codes == frozenset(_KERNEL_EVENT_CODES)


def test_write_messages(tmp_path):
    """Verifies that _write_messages serializes a columnar message block to a readable feather file."""
    messages = ExtractedMessages(
        timestamps=np.array([100, 200], dtype=np.uint64),
        commands=np.array([1, 2], dtype=np.uint8),
        events=np.array([10, 20], dtype=np.uint8),
        dtypes=(None, "uint8"),
        data_payloads=(None, np.uint8(42).tobytes()),
    )

    file_path = tmp_path / "messages.feather"
    _write_messages(messages=messages, file_path=file_path)

    dataframe = pl.read_ipc(source=file_path)
    assert dataframe.columns == ["timestamp_us", "command", "event", "dtype", "data"]
    assert dataframe["timestamp_us"].to_list() == [100, 200]
    assert dataframe["data"].to_list() == [None, np.uint8(42).tobytes()]


def test_resolve_controller_configs(tmp_path):
    """Verifies that _resolve_controller_configs keys every configured controller by its source identifier."""
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    controller_configs = _resolve_controller_configs(config=config_path, universe_ids=["1", "2"])

    assert sorted(controller_configs) == ["1", "2"]
    assert controller_configs["1"].controller_id == 1


def test_resolve_controller_configs_no_controllers(tmp_path):
    """Verifies that _resolve_controller_configs raises ValueError when the config declares no controllers."""
    config_path = tmp_path / "config.yaml"
    ExtractionConfig(controllers=[]).to_yaml(file_path=config_path)

    message = f"Unable to process logs using the extraction config at '{config_path}'. It declares no controllers."
    with pytest.raises(ValueError, match=error_format(message)):
        _resolve_controller_configs(config=config_path, universe_ids=["1"])


def test_resolve_controller_configs_unregistered_controller(tmp_path):
    """Verifies that _resolve_controller_configs raises ValueError for a controller the manifest does not register."""
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 3))

    message = (
        f"Unable to process logs using the extraction config at '{config_path}'. The following controller IDs are "
        f"not registered in the microcontroller manifest: 3. The corresponding log archives were not produced by "
        f"ataraxis-communication-interface. Registered IDs: ['1', '2']."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        _resolve_controller_configs(config=config_path, universe_ids=["1", "2"])


@pytest.mark.xdist_group(name="orchestration")
def test_execute_sized_job(tmp_path):
    """Verifies that _execute_sized_job sizes the job from its own archive and executes it at the resolved width."""
    archive_path = _build_archive(log_directory=tmp_path / "logs", source_id=1, messages=_module_messages())

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    tracker = _initialize_tracker(tracker_path=output_directory / TRACKER_FILENAME, source_id="1")
    job_id = generate_job_ids(source_ids=["1"])["1"]

    _execute_sized_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id="1",
        job_id=job_id,
        ceiling=4,
        tracker=tracker,
        controller_config=_build_controller_config(modules=_module_config()),
        display_progress=False,
    )

    feather_path = output_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather"
    assert len(pl.read_ipc(source=feather_path)) == 2
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode(tmp_path):
    """Verifies that local mode extracts every configured controller into the microcontroller data subdirectory."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)

    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        workers=1,
        display_progress=False,
    )

    data_directory = output_directory / MICROCONTROLLER_DATA_DIRECTORY
    assert data_directory.is_dir()
    assert (data_directory / TRACKER_FILENAME).is_file()

    feather_path = data_directory / "controller_1_module_1_2.feather"
    assert feather_path.is_file()
    assert len(pl.read_ipc(source=feather_path)) == 3

    tracker = ProcessingTracker(file_path=data_directory / TRACKER_FILENAME)
    assert tracker.get_job_status(job_id=generate_job_ids(source_ids=["1"])["1"]) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_kernel(tmp_path):
    """Verifies that local mode writes the kernel feather file under its canonical name when kernel data is present."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory, source_id=1, messages=_module_messages() + _kernel_messages())
    _write_manifest(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(
        config_path=tmp_path / "config.yaml", source_ids=(1,), kernel_event_codes=_KERNEL_EVENT_CODES
    )

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        workers=1,
        display_progress=False,
    )

    data_directory = output_directory / MICROCONTROLLER_DATA_DIRECTORY
    assert (data_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather").is_file()
    assert len(pl.read_ipc(source=data_directory / "controller_1_kernel.feather")) == 2


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_multiple_directories(tmp_path):
    """Verifies that local mode raises ValueError when the resolved log archives span multiple directories."""
    log_directory = tmp_path / "logs"
    first_directory = log_directory / "a"
    second_directory = log_directory / "b"
    _build_archive(log_directory=first_directory, source_id=1, messages=_module_messages())
    _build_archive(log_directory=second_directory, source_id=2, messages=_module_messages())

    # Writes the manifest at the search root, so both archives are registered under a single manifest.
    _write_manifest(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    parents = sorted(str(parent) for parent in (first_directory, second_directory))
    message = (
        f"Unable to process logs in '{log_directory}'. The requested log archives span multiple directories: "
        f"{parents}. Each DataLogger output directory must be processed independently."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config=config_path,
            workers=1,
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_remote_mode(tmp_path):
    """Verifies that remote mode executes only the job matching the requested job ID."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory, source_id=1, messages=_module_messages())
    _build_archive(log_directory=log_directory, source_id=2, messages=_module_messages())
    _write_manifest(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    output_directory = tmp_path / "output"
    job_ids = generate_job_ids(source_ids=["1", "2"])
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        job_id=job_ids["1"],
        workers=1,
        display_progress=False,
    )

    data_directory = output_directory / MICROCONTROLLER_DATA_DIRECTORY
    assert (data_directory / f"controller_1_module_{_MODULE_TYPE}_{_MODULE_ID}.feather").is_file()
    assert not (data_directory / f"controller_2_module_{_MODULE_TYPE}_{_MODULE_ID}.feather").exists()

    tracker = ProcessingTracker(file_path=data_directory / TRACKER_FILENAME)
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=job_ids["2"]) == ProcessingStatus.SCHEDULED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_remote_jobs_share_tracker(tmp_path):
    """Verifies that two independent remote jobs sharing one tracker both succeed without resetting each other."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory, source_id=1, messages=_module_messages())
    _build_archive(log_directory=log_directory, source_id=2, messages=_module_messages())
    _write_manifest(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    output_directory = tmp_path / "output"
    job_ids = generate_job_ids(source_ids=["1", "2"])

    # Dispatches each source as its own remote job against the same output directory, and therefore the same tracker.
    for source_id in ("1", "2"):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=output_directory,
            config=config_path,
            job_id=job_ids[source_id],
            workers=1,
            display_progress=False,
        )

    tracker = ProcessingTracker(file_path=output_directory / MICROCONTROLLER_DATA_DIRECTORY / TRACKER_FILENAME)
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=job_ids["2"]) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_invalid_job_id(tmp_path):
    """Verifies that remote mode raises ValueError when the requested job ID names no job in the manifest universe."""
    log_directory, config_path, output_directory = setup_test_environment(tmp_path=tmp_path)

    tracker_path = output_directory / MICROCONTROLLER_DATA_DIRECTORY / TRACKER_FILENAME
    known_ids = sorted(generate_job_ids(source_ids=["1"]).values())
    message = (
        f"Unable to resolve the job with ID 'invalid_job_id_value' against the job universe of the processing "
        f"tracker at '{tracker_path}'. The identifier must name a job the pipeline could produce, but the universe "
        f"holds only the jobs with IDs: {', '.join(known_ids)}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=output_directory,
            config=config_path,
            job_id="invalid_job_id_value",
            workers=1,
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_unconfigured_source(tmp_path):
    """Verifies that remote mode raises ValueError when the config declares no entry for the resolved source."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory, source_id=1, messages=_module_messages())
    _build_archive(log_directory=log_directory, source_id=2, messages=_module_messages())
    _write_manifest(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    job_id = generate_job_ids(source_ids=["2"])["2"]
    message = (
        f"Unable to execute the requested job with ID '{job_id}'. The extraction config at '{config_path}' declares "
        f"no entry for the controller with ID '2'. Configured controller IDs: ['1']."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config=config_path,
            job_id=job_id,
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_missing_config(tmp_path):
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when the config path does not resolve."""
    missing_config = tmp_path / "nonexistent.yaml"
    message = f"Unable to load the extraction config from '{missing_config}'. The path does not exist or is not a file."
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=tmp_path,
            output_directory=tmp_path / "output",
            config=missing_config,
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_directory_not_found(tmp_path):
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when the log directory does not exist."""
    config_path = write_extraction_config(config_path=tmp_path / "config.yaml", source_id=1)

    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to discover microcontroller data extraction jobs in '{missing_directory}'. The path does not exist "
        f"or is not a directory."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=missing_directory,
            output_directory=tmp_path / "output",
            config=config_path,
            workers=1,
            display_progress=False,
        )


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_no_manifest(tmp_path):
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when the log directory holds no manifest."""
    log_directory = tmp_path / "logs"
    _build_archive(log_directory=log_directory, source_id=1, messages=_module_messages())
    config_path = write_extraction_config(config_path=tmp_path / "config.yaml", source_id=1)

    message = (
        f"Unable to discover microcontroller data extraction jobs in '{log_directory}'. No "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} was found. A microcontroller manifest is required to identify which "
        f"log archives were produced by ataraxis-communication-interface."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config=config_path,
            workers=1,
            display_progress=False,
        )
