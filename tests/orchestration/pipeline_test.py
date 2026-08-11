"""Contains tests for the sequential processing pipeline provided by the orchestration/pipeline.py module."""

from typing import Any, NoReturn
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray
from tests.log_archives import (
    create_test_archive,
    write_extraction_config,
    create_kernel_data_payload,
    create_module_data_payload,
    create_kernel_state_payload,
    create_module_state_payload,
)
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingStatus, ProcessingTracker

from ataraxis_communication_interface.communication import SerialPrototypes
from ataraxis_communication_interface.orchestration import pipeline, discovery
from ataraxis_communication_interface.microcontroller import (
    ExtractionConfig,
    ModuleSourceData,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    ControllerExtractionConfig,
    write_microcontroller_manifest,
)
from ataraxis_communication_interface.orchestration.jobs import (
    OutputLayout,
    generate_job_ids,
    resolve_kernel_path,
    resolve_module_path,
    resolve_tracker_path,
    resolve_output_directory,
)
from ataraxis_communication_interface.orchestration.pipeline import run_log_processing_pipeline
from ataraxis_communication_interface.orchestration.allocation import resolve_core_budget

_MODULE_TYPE: int = 1
"""Stores the type (family) code of the hardware module every synthetic log archive built in this module logs
messages for."""

_MODULE_ID: int = 2
"""Stores the identifier code of the hardware module every synthetic log archive built in this module logs
messages for."""

_MODULE_EVENT_CODES: tuple[int, ...] = (10, 20)
"""Stores the module event codes every extraction configuration built in this module requests."""

_KERNEL_EVENT_CODES: tuple[int, ...] = (5,)
"""Stores the kernel event codes every extraction configuration exercising kernel extraction requests."""

_MESSAGE_COUNT: int = 2
"""Stores the number of messages each of the module and the kernel filters extracts from every synthetic archive."""


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_all_sources(tmp_path: Path) -> None:
    """Verifies that local mode processes every controller the extraction configuration declares."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(
        config_path=tmp_path / "config.yaml", source_ids=(1, 2), kernel_event_codes=_KERNEL_EVENT_CODES
    )

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        workers=1,
        display_progress=False,
    )

    # The pipeline materializes its own subdirectory and tracker under the nominated output root.
    data_directory = resolve_output_directory(output_directory=output_directory)
    assert data_directory.is_dir()
    assert data_directory.name == str(OutputLayout.DIRECTORY_NAME)
    assert resolve_tracker_path(output_directory=data_directory).is_file()

    tracker = _open_tracker(output_directory=output_directory)
    job_ids = generate_job_ids(source_ids=["1", "2"])
    for source_id in ("1", "2"):
        module_frame = pl.read_ipc(source=_module_path(output_directory=output_directory, source_id=source_id))
        kernel_frame = pl.read_ipc(source=_kernel_path(output_directory=output_directory, source_id=source_id))
        assert module_frame["event"].to_list() == list(_MODULE_EVENT_CODES)
        assert kernel_frame["event"].to_list() == [_KERNEL_EVENT_CODES[0]] * _MESSAGE_COUNT
        assert tracker.get_job_status(job_id=job_ids[source_id]) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_subset(tmp_path: Path) -> None:
    """Verifies that local mode processes only the explicitly requested subset of the configured controllers."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        source_ids=["1"],
        workers=1,
        display_progress=False,
    )

    assert _module_path(output_directory=output_directory, source_id="1").is_file()
    assert not _module_path(output_directory=output_directory, source_id="2").exists()

    # No kernel extraction is configured, so the run writes the module file alone.
    assert not _kernel_path(output_directory=output_directory, source_id="1").exists()

    # The unrequested controller stays off the tracker, as the alignment registers the prepared subset alone.
    tracker = _open_tracker(output_directory=output_directory)
    job_ids = generate_job_ids(source_ids=["1", "2"])
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert job_ids["2"] not in tracker.snapshot()


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_empty_source_ids(tmp_path: Path) -> None:
    """Verifies that an empty source ID sequence resolves through the config and processes every controller."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        source_ids=[],
        workers=1,
        display_progress=False,
    )

    tracker = _open_tracker(output_directory=output_directory)
    job_ids = generate_job_ids(source_ids=["1", "2"])
    for source_id in ("1", "2"):
        module_frame = pl.read_ipc(source=_module_path(output_directory=output_directory, source_id=source_id))
        assert len(module_frame) == _MESSAGE_COUNT
        assert tracker.get_job_status(job_id=job_ids[source_id]) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_local_mode_configured_controllers_only(tmp_path: Path) -> None:
    """Verifies that the extraction configuration bounds the work while the manifest still bounds the tracker."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1, 2))

    # The manifest registers both controllers, but the configuration declares only the first, so only it is processed.
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        workers=1,
        display_progress=False,
    )

    assert _module_path(output_directory=output_directory, source_id="1").is_file()
    assert not _module_path(output_directory=output_directory, source_id="2").exists()

    job_ids = generate_job_ids(source_ids=["1", "2"])
    assert sorted(_open_tracker(output_directory=output_directory).snapshot()) == [job_ids["1"]]


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_external_mode(tmp_path: Path) -> None:
    """Verifies that external mode executes only the single job the requested canonical job ID names."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1, 2))
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

    assert _module_path(output_directory=output_directory, source_id="1").is_file()
    assert not _module_path(output_directory=output_directory, source_id="2").exists()

    tracker = _open_tracker(output_directory=output_directory)
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert job_ids["2"] not in tracker.snapshot()


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_external_jobs_share_tracker(tmp_path: Path) -> None:
    """Verifies that two independent external jobs sharing one tracker both succeed without resetting each other."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    output_directory = tmp_path / "output"
    job_ids = generate_job_ids(source_ids=["1", "2"])

    # Dispatches each controller as its own external job against the same output directory, and therefore the same
    # tracker. Both invocations align that tracker against the full manifest universe, so neither discards the other's
    # recorded outcome as a foreign entry.
    for source_id in ("1", "2"):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=output_directory,
            config=config_path,
            job_id=job_ids[source_id],
            workers=1,
            display_progress=False,
        )

    tracker = _open_tracker(output_directory=output_directory)
    assert sorted(tracker.snapshot()) == sorted(job_ids.values())
    assert tracker.get_job_status(job_id=job_ids["1"]) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=job_ids["2"]) == ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="orchestration")
def test_run_log_processing_pipeline_unknown_job_id(tmp_path: Path) -> None:
    """Verifies that external mode raises ValueError when the requested job ID names no configured controller."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    message = (
        f"Unable to prepare the microcontroller data extraction job 'invalid_job_id_value' in '{log_directory}'. The "
        f"extraction config at '{config_path}' declares no controller with that job identifier. Configured controller "
        f"IDs: 1, 2."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config=config_path,
            job_id="invalid_job_id_value",
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_missing_config(tmp_path: Path) -> None:
    """Verifies that the pipeline reports the missing configuration file it requires before it resolves any job."""
    log_directory = tmp_path / "logs"
    _build_controller_logs(log_directory=log_directory, source_ids=(1,))

    missing_config = tmp_path / "nonexistent.yaml"
    message = f"Unable to load the extraction config from '{missing_config}'. The path does not exist or is not a file."
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config=missing_config,
            workers=1,
            display_progress=False,
        )

    # A failed resolution materializes nothing, as the output subdirectory is created only once the jobs resolve.
    assert not (tmp_path / "output").exists()


def test_run_log_processing_pipeline_missing_log_directory(tmp_path: Path) -> None:
    """Verifies that the pipeline reports the missing tree when the log directory does not exist."""
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to resolve microcontroller data extraction jobs in '{missing_directory}'. The path does not exist or "
        f"is not a directory."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=missing_directory,
            output_directory=tmp_path / "output",
            config=config_path,
            workers=1,
            display_progress=False,
        )

    assert not (tmp_path / "output").exists()


def test_run_log_processing_pipeline_resolves_no_job(tmp_path: Path) -> None:
    """Verifies that the pipeline fails loudly when the recording resolves no extraction job to run."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    # A tree holding archives and a valid configuration, but no manifest, owns no job this library processes.
    create_test_archive(archive_path=log_directory / "1.npz", source_id=1, messages=[])
    config_path = tmp_path / "config.yaml"
    write_extraction_config(config_path=config_path, source_id=1)

    message = (
        f"Unable to process microcontroller log archives in '{log_directory}'. The recording resolved no extraction "
        f"job. Its tree holds no microcontroller manifest, or the extraction config declares no controller whose log "
        f"archive resolves to exactly one file beneath it."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config=config_path,
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_reads_no_archive_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the pipeline dispatches its jobs without opening or sizing a single log archive."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir(parents=True)

    # Writes unreadable stand-ins for the log archives, so any read before dispatch raises instead of being missed.
    for source_id in (1, 2):
        _archive_path(log_directory=log_directory, source_id=source_id).write_bytes(b"not-a-valid-npz-archive")
        _write_manifest_entry(log_directory=log_directory, source_id=source_id)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    def _forbidden_footprint(**arguments: Any) -> NoReturn:
        message = f"The pipeline sized a job from its archive: {arguments}."
        raise AssertionError(message)

    monkeypatch.setattr(discovery, "resolve_archive_footprint", _forbidden_footprint)
    calls = _record_dispatches(monkeypatch=monkeypatch)

    output_directory = tmp_path / "output"
    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        workers=4,
        display_progress=False,
    )

    # Every configured controller reaches the runner, in ascending source identifier order.
    job_ids = generate_job_ids(source_ids=["1", "2"])
    assert [call["source_id"] for call in calls] == ["1", "2"]
    assert [call["job_id"] for call in calls] == [job_ids["1"], job_ids["2"]]
    assert [call["log_path"] for call in calls] == [
        _archive_path(log_directory=log_directory, source_id=source_id) for source_id in (1, 2)
    ]

    # Every job is dispatched at the requested ceiling, as narrowing that width is the extraction's own business, and
    # each one carries the configuration file its body reads its own extraction targets from.
    resolved_output = resolve_output_directory(output_directory=output_directory)
    assert {call["workers"] for call in calls} == {resolve_core_budget(requested_budget=4)}
    assert {call["output_directory"] for call in calls} == {resolved_output}
    assert {call["config_path"] for call in calls} == {config_path}
    assert {call["display_progress"] for call in calls} == {False}
    assert {call["tracker"].file_path for call in calls} == {resolve_tracker_path(output_directory=resolved_output)}

    # The preparation still materializes the output layout and registers both jobs on the shared tracker.
    assert sorted(_open_tracker(output_directory=output_directory).snapshot()) == sorted(job_ids.values())


def _archive_path(log_directory: Path, source_id: int) -> Path:
    """Resolves the path of the synthetic log archive of the target controller source."""
    return log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"


def _controller_messages() -> list[tuple[int, NDArray[np.uint8]]]:
    """Creates the module and the kernel messages every synthetic log archive built in this module holds."""
    return [
        (1000, create_module_state_payload(module_type=_MODULE_TYPE, module_id=_MODULE_ID, command=1, event=10)),
        (
            2000,
            create_module_data_payload(
                module_type=_MODULE_TYPE,
                module_id=_MODULE_ID,
                command=2,
                event=20,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[42],
            ),
        ),
        (3000, create_kernel_state_payload(command=1, event=5)),
        (
            4000,
            create_kernel_data_payload(
                command=2,
                event=5,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[7],
            ),
        ),
    ]


def _write_manifest_entry(log_directory: Path, source_id: int) -> None:
    """Appends the manifest entry of one single-module controller source to the target log directory's manifest."""
    write_microcontroller_manifest(
        log_directory=log_directory,
        controller_id=source_id,
        controller_name=f"controller_{source_id}",
        modules=(ModuleSourceData(module_type=_MODULE_TYPE, module_id=_MODULE_ID, name="test_module"),),
    )


def _build_controller_logs(log_directory: Path, source_ids: tuple[int, ...]) -> None:
    """Creates one synthetic log archive and one manifest entry for each of the requested controller sources."""
    log_directory.mkdir(parents=True, exist_ok=True)
    for source_id in source_ids:
        create_test_archive(
            archive_path=_archive_path(log_directory=log_directory, source_id=source_id),
            source_id=source_id,
            messages=_controller_messages(),
        )
        _write_manifest_entry(log_directory=log_directory, source_id=source_id)


def _write_config(
    config_path: Path, source_ids: tuple[int, ...], kernel_event_codes: tuple[int, ...] | None = None
) -> Path:
    """Writes the extraction configuration declaring the shared module and kernel filters of each controller."""
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


def _module_path(output_directory: Path, source_id: str) -> Path:
    """Resolves the module output file the pipeline writes for the target controller source."""
    return resolve_module_path(
        output_directory=resolve_output_directory(output_directory=output_directory),
        source_id=source_id,
        module_type=_MODULE_TYPE,
        module_id=_MODULE_ID,
    )


def _kernel_path(output_directory: Path, source_id: str) -> Path:
    """Resolves the kernel output file the pipeline writes for the target controller source."""
    return resolve_kernel_path(
        output_directory=resolve_output_directory(output_directory=output_directory), source_id=source_id
    )


def _open_tracker(output_directory: Path) -> ProcessingTracker:
    """Opens the processing tracker the pipeline aligned under the target output directory."""
    return ProcessingTracker(
        file_path=resolve_tracker_path(output_directory=resolve_output_directory(output_directory=output_directory))
    )


def _record_dispatches(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the single-job runner the pipeline dispatches with a recorder and returns the recorded calls."""
    calls = []

    def _record(**arguments: Any) -> None:
        calls.append(arguments)

    monkeypatch.setattr(pipeline, "execute_job", _record)
    return calls
