"""Contains tests for the classes and functions defined in the log_processing module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import pytest
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ataraxis_communication_interface.communication import SerialProtocols, SerialPrototypes
from ataraxis_communication_interface.microcontroller.dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    ModuleSourceData,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    MicroControllerManifest,
    MicroControllerSourceData,
    ControllerExtractionConfig,
)
from ataraxis_communication_interface.microcontroller.log_processing import (
    FEATHER_SUFFIX,
    TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX,
    KERNEL_FEATHER_INFIX,
    MODULE_FEATHER_INFIX,
    CONTROLLER_FEATHER_PREFIX,
    PARALLEL_PROCESSING_THRESHOLD,
    MICROCONTROLLER_DATA_DIRECTORY,
    execute_job,
    find_log_archive,
    generate_job_ids,
    _ColumnAccumulator,
    _ExtractedMessages,
    _ExtractedModuleData,
    _finalize_accumulator,
    _write_kernel_feather,
    _write_module_feather,
    resolve_recording_roots,
    _build_message_dataframe,
    _extract_unique_components,
    run_log_processing_pipeline,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _create_archive_entry(source_id: int, elapsed_us: int, payload: NDArray[np.uint8]) -> tuple[str, NDArray[np.uint8]]:
    """Creates a single archive entry (key, data) in the LogPackage format."""
    header = np.empty(9, dtype=np.uint8)
    header[0] = np.uint8(source_id)
    header[1:9] = np.frombuffer(np.uint64(elapsed_us).tobytes(), dtype=np.uint8)
    data = np.concatenate([header, payload.astype(np.uint8)])
    key = f"{source_id:03d}_{elapsed_us:020d}"
    return key, data


def _create_onset_entry(source_id: int, onset_us: int) -> tuple[str, NDArray[np.uint8]]:
    """Creates the onset timestamp entry for a log archive."""
    onset_payload = np.frombuffer(np.int64(onset_us).tobytes(), dtype=np.uint8)
    return _create_archive_entry(source_id=source_id, elapsed_us=0, payload=onset_payload)


def _create_test_archive(
    archive_path: Path,
    source_id: int,
    messages: list[tuple[int, NDArray[np.uint8]]],
    onset_us: int = 1_000_000_000_000,
) -> None:
    """Creates a test .npz log archive with the given messages."""
    entries: dict[str, NDArray[np.uint8]] = {}

    # Creates the onset entry.
    key, data = _create_onset_entry(source_id=source_id, onset_us=onset_us)
    entries[key] = data

    # Creates message entries.
    for elapsed_us, payload in messages:
        key, data = _create_archive_entry(source_id=source_id, elapsed_us=elapsed_us, payload=payload)
        entries[key] = data

    np.savez(str(archive_path), **entries)


def _make_module_state_payload(module_type: int, module_id: int, command: int, event: int) -> NDArray[np.uint8]:
    """Creates a MODULE_STATE message payload."""
    return np.array([SerialProtocols.MODULE_STATE, module_type, module_id, command, event], dtype=np.uint8)


def _make_module_data_payload(
    module_type: int, module_id: int, command: int, event: int, prototype_code: int, data_bytes: list[int]
) -> NDArray[np.uint8]:
    """Creates a MODULE_DATA message payload."""
    header = [SerialProtocols.MODULE_DATA, module_type, module_id, command, event, prototype_code]
    return np.array(header + data_bytes, dtype=np.uint8)


def _make_kernel_state_payload(command: int, event: int) -> NDArray[np.uint8]:
    """Creates a KERNEL_STATE message payload."""
    return np.array([SerialProtocols.KERNEL_STATE, command, event], dtype=np.uint8)


def _make_kernel_data_payload(
    command: int, event: int, prototype_code: int, data_bytes: list[int]
) -> NDArray[np.uint8]:
    """Creates a KERNEL_DATA message payload."""
    header = [SerialProtocols.KERNEL_DATA, command, event, prototype_code]
    return np.array(header + data_bytes, dtype=np.uint8)


def _setup_test_environment(
    tmp_path: Path,
    source_id: int = 1,
    module_type: int = 1,
    module_id: int = 2,
) -> tuple[Path, Path, Path]:
    """Creates a complete test environment with archive, manifest, and config."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Creates archive with module state and data messages.
    archive_path = log_dir / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (2000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (
            3000,
            _make_module_data_payload(
                module_type=module_type,
                module_id=module_id,
                command=2,
                event=20,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[42],
            ),
        ),
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    # Creates the manifest.
    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(
        MicroControllerSourceData(
            id=source_id,
            name="test_controller",
            modules=(ModuleSourceData(module_type=module_type, module_id=module_id, name="test_module"),),
        )
    )
    manifest.save(file_path=log_dir / MICROCONTROLLER_MANIFEST_FILENAME)

    # Creates the extraction config.
    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=source_id,
                modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10, 20)),),
                kernel=None,
            )
        ]
    )
    config_path = tmp_path / "config.yaml"
    config.save(file_path=config_path)

    return log_dir, config_path, output_dir


def test_log_processing_constants() -> None:
    """Verifies that module-level constants have expected values."""
    assert LOG_ARCHIVE_SUFFIX == "_log.npz"
    assert TRACKER_FILENAME == "microcontroller_processing_tracker.yaml"
    assert MICROCONTROLLER_DATA_DIRECTORY == "microcontroller_data"
    assert PARALLEL_PROCESSING_THRESHOLD == 2000
    assert CONTROLLER_FEATHER_PREFIX == "controller_"
    assert MODULE_FEATHER_INFIX == "_module_"
    assert KERNEL_FEATHER_INFIX == "_kernel"
    assert FEATHER_SUFFIX == ".feather"


def test_extracted_messages_count() -> None:
    """Verifies the count property of _ExtractedMessages."""
    messages = _ExtractedMessages(
        timestamps=np.array([100, 200, 300], dtype=np.uint64),
        commands=np.array([1, 2, 3], dtype=np.uint8),
        events=np.array([10, 20, 30], dtype=np.uint8),
        dtypes=(None, None, "uint8"),
        data_payloads=(None, None, b"\x01"),
    )

    assert messages.count == 3


def test_extracted_messages_count_empty() -> None:
    """Verifies the count property with no messages."""
    messages = _ExtractedMessages(
        timestamps=np.array([], dtype=np.uint64),
        commands=np.array([], dtype=np.uint8),
        events=np.array([], dtype=np.uint8),
        dtypes=(),
        data_payloads=(),
    )

    assert messages.count == 0


def test_finalize_accumulator() -> None:
    """Verifies that _finalize_accumulator converts lists to numpy arrays."""
    accumulator = _ColumnAccumulator(
        timestamps=[100, 200, 300],
        commands=[1, 2, 3],
        events=[10, 20, 30],
        dtypes=[None, "uint8", None],
        data_payloads=[None, b"\x01", None],
    )

    result = _finalize_accumulator(accumulator=accumulator)

    assert isinstance(result, _ExtractedMessages)
    assert result.count == 3
    assert result.timestamps.dtype == np.uint64
    assert result.commands.dtype == np.uint8
    assert result.events.dtype == np.uint8
    np.testing.assert_array_equal(result.timestamps, [100, 200, 300])
    np.testing.assert_array_equal(result.commands, [1, 2, 3])
    np.testing.assert_array_equal(result.events, [10, 20, 30])
    assert result.dtypes == (None, "uint8", None)
    assert result.data_payloads == (None, b"\x01", None)


def test_finalize_accumulator_empty() -> None:
    """Verifies that _finalize_accumulator handles empty accumulators."""
    accumulator = _ColumnAccumulator(timestamps=[], commands=[], events=[], dtypes=[], data_payloads=[])

    result = _finalize_accumulator(accumulator=accumulator)

    assert result.count == 0
    assert len(result.timestamps) == 0


def test_build_message_dataframe() -> None:
    """Verifies that _build_message_dataframe creates a correct polars DataFrame."""
    messages = _ExtractedMessages(
        timestamps=np.array([100, 200], dtype=np.uint64),
        commands=np.array([1, 2], dtype=np.uint8),
        events=np.array([10, 20], dtype=np.uint8),
        dtypes=("uint8", None),
        data_payloads=(b"\x42", None),
    )

    dataframe = _build_message_dataframe(messages=messages)

    assert isinstance(dataframe, pl.DataFrame)
    assert dataframe.shape == (2, 5)
    assert dataframe.columns == ["timestamp_us", "command", "event", "dtype", "data"]
    assert dataframe["timestamp_us"].dtype == pl.UInt64
    assert dataframe["command"].dtype == pl.UInt8
    assert dataframe["event"].dtype == pl.UInt8
    assert dataframe["dtype"].dtype == pl.String
    assert dataframe["data"].dtype == pl.Binary
    assert dataframe["timestamp_us"][0] == 100
    assert dataframe["command"][0] == 1
    assert dataframe["event"][0] == 10
    assert dataframe["dtype"][0] == "uint8"
    assert dataframe["data"][0] == b"\x42"
    assert dataframe["dtype"][1] is None
    assert dataframe["data"][1] is None


def test_build_message_dataframe_empty() -> None:
    """Verifies that _build_message_dataframe handles empty messages."""
    messages = _ExtractedMessages(
        timestamps=np.array([], dtype=np.uint64),
        commands=np.array([], dtype=np.uint8),
        events=np.array([], dtype=np.uint8),
        dtypes=(),
        data_payloads=(),
    )

    dataframe = _build_message_dataframe(messages=messages)

    assert dataframe.shape == (0, 5)


def test_write_module_feather(tmp_path: Path) -> None:
    """Verifies that _write_module_feather creates a valid feather file."""
    messages = _ExtractedMessages(
        timestamps=np.array([100, 200], dtype=np.uint64),
        commands=np.array([1, 2], dtype=np.uint8),
        events=np.array([10, 20], dtype=np.uint8),
        dtypes=("uint8", None),
        data_payloads=(b"\x42", None),
    )
    module_data = _ExtractedModuleData(module_type=1, module_id=2, messages=messages)

    _write_module_feather(module_data=module_data, source_id="1", output_directory=tmp_path)

    expected_filename = f"{CONTROLLER_FEATHER_PREFIX}1{MODULE_FEATHER_INFIX}1_2{FEATHER_SUFFIX}"
    feather_path = tmp_path / expected_filename
    assert feather_path.exists()

    dataframe = pl.read_ipc(feather_path)
    assert dataframe.shape == (2, 5)
    assert dataframe["timestamp_us"][0] == 100


def test_write_kernel_feather(tmp_path: Path) -> None:
    """Verifies that _write_kernel_feather creates a valid feather file."""
    kernel_data = _ExtractedMessages(
        timestamps=np.array([100, 200], dtype=np.uint64),
        commands=np.array([1, 2], dtype=np.uint8),
        events=np.array([5, 6], dtype=np.uint8),
        dtypes=(None, None),
        data_payloads=(None, None),
    )

    _write_kernel_feather(kernel_data=kernel_data, source_id="1", output_directory=tmp_path)

    expected_filename = f"{CONTROLLER_FEATHER_PREFIX}1{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}"
    feather_path = tmp_path / expected_filename
    assert feather_path.exists()

    dataframe = pl.read_ipc(feather_path)
    assert dataframe.shape == (2, 5)
    assert dataframe["event"][0] == 5


def test_find_log_archive(tmp_path: Path) -> None:
    """Verifies that find_log_archive locates a matching archive."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    _create_test_archive(archive_path=archive_path, source_id=1, messages=[])

    result = find_log_archive(log_directory=tmp_path, source_id="1")

    assert result == archive_path


def test_find_log_archive_nested(tmp_path: Path) -> None:
    """Verifies that find_log_archive searches recursively."""
    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)
    archive_path = nested / f"1{LOG_ARCHIVE_SUFFIX}"
    _create_test_archive(archive_path=archive_path, source_id=1, messages=[])

    result = find_log_archive(log_directory=tmp_path, source_id="1")

    assert result == archive_path


def test_find_log_archive_missing_directory(tmp_path: Path) -> None:
    """Verifies that find_log_archive raises FileNotFoundError for a missing directory."""
    nonexistent = tmp_path / "nonexistent"

    message = (
        f"Unable to find log archive for source '1' in '{nonexistent}'. The path does not exist or is not a directory."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        find_log_archive(log_directory=nonexistent, source_id="1")


def test_find_log_archive_not_found(tmp_path: Path) -> None:
    """Verifies that find_log_archive raises FileNotFoundError when no archive matches."""
    message = (
        f"Unable to find log archive for source '999' in '{tmp_path}'. "
        f"No file matching '999{LOG_ARCHIVE_SUFFIX}' was found."
    )
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        find_log_archive(log_directory=tmp_path, source_id="999")


def test_find_log_archive_multiple(tmp_path: Path) -> None:
    """Verifies that find_log_archive raises ValueError when multiple archives match."""
    first_subdirectory = tmp_path / "dir1"
    first_subdirectory.mkdir()
    second_subdirectory = tmp_path / "dir2"
    second_subdirectory.mkdir()

    _create_test_archive(archive_path=first_subdirectory / f"1{LOG_ARCHIVE_SUFFIX}", source_id=1, messages=[])
    _create_test_archive(archive_path=second_subdirectory / f"1{LOG_ARCHIVE_SUFFIX}", source_id=1, messages=[])

    message = f"Unable to find log archive for source '1' in '{tmp_path}'. Found 2 matching archives"
    with pytest.raises(ValueError, match=error_format(message)):
        find_log_archive(log_directory=tmp_path, source_id="1")


def test_extract_unique_components_basic() -> None:
    """Verifies basic unique component extraction."""
    paths = [Path("/data/day1/recording"), Path("/data/day2/recording")]

    result = _extract_unique_components(paths=paths)

    assert result == ("day1", "day2")


def test_extract_unique_components_single() -> None:
    """Verifies unique component extraction with a single path."""
    paths = [Path("/data/experiment/session1")]

    result = _extract_unique_components(paths=paths)

    # All components are unique when there's only one path; selects from the end.
    assert result == ("session1",)


def test_extract_unique_components_deep_nesting() -> None:
    """Verifies unique component extraction with deeply nested paths."""
    paths = [
        Path("/data/project/exp1/session/recording"),
        Path("/data/project/exp2/session/recording"),
    ]

    result = _extract_unique_components(paths=paths)

    assert result == ("exp1", "exp2")


def test_extract_unique_components_no_unique() -> None:
    """Verifies that _extract_unique_components raises RuntimeError when no unique component exists."""
    paths = [Path("/a/b/c"), Path("/a/b/c")]

    message = "Unable to extract a unique component from the given path"
    with pytest.raises(RuntimeError, match=error_format(message)):
        _extract_unique_components(paths=paths)


def test_resolve_recording_roots() -> None:
    """Verifies that resolve_recording_roots returns correct root paths."""
    paths = [Path("/data/day1/session/logs"), Path("/data/day2/session/logs")]

    result = resolve_recording_roots(paths=paths)

    assert len(result) == 2
    assert result[0] == Path("/data/day1")
    assert result[1] == Path("/data/day2")


def test_resolve_recording_roots_deduplicates() -> None:
    """Verifies that resolve_recording_roots deduplicates identical roots."""
    # Both paths share the same unique component path root.
    paths = [Path("/data/unique_session/logs"), Path("/data/unique_session/output")]

    result = resolve_recording_roots(paths=paths)

    # "logs" and "output" are unique components, both under unique_session.
    assert len(result) == 2


def test_generate_job_ids() -> None:
    """Verifies that generate_job_ids returns a mapping of source IDs to hex job IDs."""
    source_ids = ["1", "2", "3"]

    result = generate_job_ids(source_ids=source_ids)

    assert len(result) == 3
    assert set(result.keys()) == {"1", "2", "3"}
    # Validates that job IDs are hex strings.
    for job_id in result.values():
        assert isinstance(job_id, str)
        int(job_id, 16)


def test_generate_job_ids_deterministic() -> None:
    """Verifies that generate_job_ids produces deterministic results."""
    first_result = generate_job_ids(source_ids=["1", "2"])
    second_result = generate_job_ids(source_ids=["1", "2"])

    assert first_result == second_result


def test_execute_job_empty_event_codes(tmp_path: Path) -> None:
    """Verifies that execute_job raises ValueError when a module has empty event_codes."""
    config = ControllerExtractionConfig(
        controller_id=1,
        modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=()),),
        kernel=None,
    )
    tracker = ProcessingTracker(file_path=tmp_path / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", "1")])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier="1")

    message = (
        "Unable to execute the data extraction job for source '1'. Module with type code 1 and ID code 1 "
        "has empty event_codes."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=tmp_path / "fake.npz",
            output_directory=tmp_path,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=config,
        )


def test_execute_job_empty_kernel_events(tmp_path: Path) -> None:
    """Verifies that execute_job raises ValueError when kernel has empty event_codes."""
    config = ControllerExtractionConfig(
        controller_id=1,
        modules=(),
        kernel=KernelExtractionConfig(event_codes=()),
    )
    tracker = ProcessingTracker(file_path=tmp_path / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", "1")])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier="1")

    message = "Unable to execute the data extraction job for source '1'. Kernel extraction has empty event_codes."
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=tmp_path / "fake.npz",
            output_directory=tmp_path,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=config,
        )


def test_execute_job_no_targets(tmp_path: Path) -> None:
    """Verifies that execute_job raises ValueError when no extraction targets are configured."""
    config = ControllerExtractionConfig(controller_id=1, modules=(), kernel=None)
    tracker = ProcessingTracker(file_path=tmp_path / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", "1")])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier="1")

    message = (
        "Unable to execute the data extraction job for source '1'. The controller config has no modules "
        "and no kernel extraction configured."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=tmp_path / "fake.npz",
            output_directory=tmp_path,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=config,
        )


def test_execute_job_invalid_archive(tmp_path: Path) -> None:
    """Verifies that execute_job raises ValueError when the archive path is invalid."""
    config = ControllerExtractionConfig(
        controller_id=1,
        modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),),
        kernel=None,
    )
    tracker = ProcessingTracker(file_path=tmp_path / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", "1")])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier="1")

    fake_path = tmp_path / "nonexistent.npz"
    message = (
        f"Unable to execute the data extraction job for source '1'. The log archive '{fake_path}' does not "
        f"exist or is not a valid .npz file."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        execute_job(
            log_path=fake_path,
            output_directory=tmp_path,
            source_id="1",
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=config,
        )


def test_execute_job_sequential_module_only(tmp_path: Path) -> None:
    """Verifies sequential execute_job with module-only extraction."""
    source_id = 1
    module_type = 1
    module_id = 2

    # Creates archive with module messages.
    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (2000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (
            3000,
            _make_module_data_payload(
                module_type=module_type,
                module_id=module_id,
                command=2,
                event=20,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[42],
            ),
        ),
        # This message uses event=99 which is NOT in the filter and should be excluded.
        (4000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=99)),
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    # Sets up tracker.
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10, 20)),),
        kernel=None,
    )

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    # Verifies module feather output.
    expected_filename = (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}{MODULE_FEATHER_INFIX}{module_type}_{module_id}{FEATHER_SUFFIX}"
    )
    feather_path = output_dir / expected_filename
    assert feather_path.exists()

    dataframe = pl.read_ipc(feather_path)
    assert dataframe.shape[0] == 3  # 3 matching messages (event=10 x2, event=20 x1)


def test_execute_job_sequential_kernel_only(tmp_path: Path) -> None:
    """Verifies sequential execute_job with kernel-only extraction."""
    source_id = 1

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_kernel_state_payload(command=1, event=5)),
        (2000, _make_kernel_state_payload(command=2, event=6)),
        (
            3000,
            _make_kernel_data_payload(command=3, event=5, prototype_code=SerialPrototypes.ONE_UINT8, data_bytes=[99]),
        ),
        # Unmatched kernel event.
        (4000, _make_kernel_state_payload(command=1, event=99)),
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(),
        kernel=KernelExtractionConfig(event_codes=(5, 6)),
    )

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    expected_filename = f"{CONTROLLER_FEATHER_PREFIX}{source_id}{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}"
    feather_path = output_dir / expected_filename
    assert feather_path.exists()

    dataframe = pl.read_ipc(feather_path)
    assert dataframe.shape[0] == 3  # 3 matching messages (event=5 x2, event=6 x1)


def test_execute_job_empty_archive(tmp_path: Path) -> None:
    """Verifies that execute_job handles an empty archive (no messages, only onset)."""
    source_id = 1

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=[])

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),),
        kernel=None,
    )

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    # No feather file should exist for an empty archive.
    feather_files = list(output_dir.glob(f"*{FEATHER_SUFFIX}"))
    assert len(feather_files) == 0


def test_execute_job_module_and_kernel(tmp_path: Path) -> None:
    """Verifies execute_job with both module and kernel extraction."""
    source_id = 1
    module_type = 1
    module_id = 2

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (2000, _make_kernel_state_payload(command=1, event=5)),
        (3000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=2, event=10)),
        (4000, _make_kernel_state_payload(command=2, event=5)),
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10,)),),
        kernel=KernelExtractionConfig(event_codes=(5,)),
    )

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    # Both module and kernel feather files should exist.
    module_feather = output_dir / (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}{MODULE_FEATHER_INFIX}{module_type}_{module_id}{FEATHER_SUFFIX}"
    )
    kernel_feather = output_dir / (f"{CONTROLLER_FEATHER_PREFIX}{source_id}{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}")
    assert module_feather.exists()
    assert kernel_feather.exists()

    module_dataframe = pl.read_ipc(module_feather)
    kernel_dataframe = pl.read_ipc(kernel_feather)
    assert module_dataframe.shape[0] == 2
    assert kernel_dataframe.shape[0] == 2


def test_execute_job_no_matching_messages(tmp_path: Path) -> None:
    """Verifies execute_job when no messages match the extraction filter."""
    source_id = 1

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_module_state_payload(module_type=1, module_id=2, command=1, event=99)),  # Wrong event.
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=1, module_id=2, event_codes=(10,)),),
        kernel=None,
    )

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=1,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    # No feather files should exist since no messages matched.
    feather_files = list(output_dir.glob(f"*{FEATHER_SUFFIX}"))
    assert len(feather_files) == 0


def test_run_log_processing_pipeline_missing_directory(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises FileNotFoundError for a missing log directory."""
    nonexistent = tmp_path / "nonexistent"
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    message = f"Unable to process logs in '{nonexistent}'. The path does not exist or is not a directory."
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(log_directory=nonexistent, output_directory=tmp_path, config=config_path)


def test_run_log_processing_pipeline_missing_config(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises FileNotFoundError for a missing config."""
    config_path = tmp_path / "nonexistent_config.yaml"

    message = f"Unable to load extraction config from '{config_path}'. The path does not exist or is not a file."
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(log_directory=tmp_path, output_directory=tmp_path, config=config_path)


def test_run_log_processing_pipeline_no_manifest(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises FileNotFoundError when no manifest exists."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=1,
                modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),),
                kernel=None,
            )
        ]
    )
    config_path = tmp_path / "config.yaml"
    config.save(file_path=config_path)

    message = f"Unable to process logs in '{log_dir}'. No {MICROCONTROLLER_MANIFEST_FILENAME} was found."
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        run_log_processing_pipeline(log_directory=log_dir, output_directory=tmp_path / "output", config=config_path)


def test_run_log_processing_pipeline_unregistered_ids(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises ValueError for unregistered controller IDs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Creates manifest with controller ID=1 only.
    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(
        MicroControllerSourceData(id=1, name="ctrl", modules=(ModuleSourceData(module_type=1, module_id=1, name="m"),))
    )
    manifest.save(file_path=log_dir / MICROCONTROLLER_MANIFEST_FILENAME)

    # Config requests controller ID=99 which is not in the manifest.
    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=99,
                modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),),
                kernel=None,
            )
        ]
    )
    config_path = tmp_path / "config.yaml"
    config.save(file_path=config_path)

    message = f"Unable to process logs in '{log_dir}'. The following controller IDs are not registered"
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(log_directory=log_dir, output_directory=tmp_path / "output", config=config_path)


def test_run_log_processing_pipeline_local_mode(tmp_path: Path) -> None:
    """Verifies end-to-end local mode pipeline execution."""
    log_dir, config_path, output_dir = _setup_test_environment(tmp_path)

    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        config=config_path,
        workers=1,
        display_progress=False,
    )

    # Verifies output structure.
    data_dir = output_dir / MICROCONTROLLER_DATA_DIRECTORY
    assert data_dir.exists()

    # Verifies tracker was created.
    tracker_path = data_dir / TRACKER_FILENAME
    assert tracker_path.exists()

    # Verifies feather output exists.
    feather_files = list(data_dir.glob(f"*{FEATHER_SUFFIX}"))
    assert len(feather_files) == 1  # One module feather file


def test_run_log_processing_pipeline_remote_mode(tmp_path: Path) -> None:
    """Verifies end-to-end remote mode pipeline execution with a specific job_id."""
    log_dir, config_path, output_dir = _setup_test_environment(tmp_path)

    # Generates the expected job ID for source_id=1.
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier="1")

    # Pre-initializes the tracker so the remote mode can find the job.
    data_dir = output_dir / MICROCONTROLLER_DATA_DIRECTORY
    data_dir.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=data_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", "1")])

    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        config=config_path,
        job_id=job_id,
        workers=1,
        display_progress=False,
    )

    feather_files = list(data_dir.glob(f"*{FEATHER_SUFFIX}"))
    assert len(feather_files) == 1


def test_run_log_processing_pipeline_remote_jobs_share_tracker(tmp_path: Path) -> None:
    """Verifies that independent remote jobs sharing one tracker do not reset each other's state.

    Dispatches each configured controller as its own remote job against a single shared tracker. Running the
    second job must align the tracker against the configuration universe and leave the first job's completed
    state intact, rather than treating it as a foreign entry and resetting it.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    module_type, module_id = 1, 2
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (
            2000,
            _make_module_data_payload(
                module_type=module_type,
                module_id=module_id,
                command=2,
                event=20,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[42],
            ),
        ),
    ]

    manifest = MicroControllerManifest(controllers=[])
    controllers: list[ControllerExtractionConfig] = []
    for source_id in (1, 2):
        _create_test_archive(
            archive_path=log_dir / f"{source_id}{LOG_ARCHIVE_SUFFIX}", source_id=source_id, messages=messages
        )
        manifest.controllers.append(
            MicroControllerSourceData(
                id=source_id,
                name=f"ctrl_{source_id}",
                modules=(ModuleSourceData(module_type=module_type, module_id=module_id, name="m"),),
            )
        )
        controllers.append(
            ControllerExtractionConfig(
                controller_id=source_id,
                modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10, 20)),),
                kernel=None,
            )
        )
    manifest.save(file_path=log_dir / MICROCONTROLLER_MANIFEST_FILENAME)
    config_path = tmp_path / "config.yaml"
    ExtractionConfig(controllers=controllers).save(file_path=config_path)

    job_id_one = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier="1")
    job_id_two = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier="2")

    # Dispatches each controller as a separate remote job against the same output directory (shared tracker).
    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        config=config_path,
        job_id=job_id_one,
        workers=1,
        display_progress=False,
    )
    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        config=config_path,
        job_id=job_id_two,
        workers=1,
        display_progress=False,
    )

    # Both jobs must be recorded as succeeded; the second run must not have reset the first.
    tracker = ProcessingTracker(file_path=output_dir / MICROCONTROLLER_DATA_DIRECTORY / TRACKER_FILENAME)
    assert tracker.get_job_status(job_id=job_id_one) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=job_id_two) == ProcessingStatus.SUCCEEDED


def test_run_log_processing_pipeline_invalid_job_id(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises ValueError for an invalid job_id."""
    log_dir, config_path, output_dir = _setup_test_environment(tmp_path)

    message = "Unable to execute the requested job with ID 'invalid_id'"
    with pytest.raises(ValueError, match=error_format(message)):
        run_log_processing_pipeline(
            log_directory=log_dir,
            output_directory=output_dir,
            config=config_path,
            job_id="invalid_id",
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_multiple_directories(tmp_path: Path) -> None:
    """Verifies that run_log_processing_pipeline raises ValueError when archives span multiple directories."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # Creates archives in different subdirectories.
    first_subdirectory = log_dir / "dir1"
    first_subdirectory.mkdir()
    second_subdirectory = log_dir / "dir2"
    second_subdirectory.mkdir()

    _create_test_archive(archive_path=first_subdirectory / f"1{LOG_ARCHIVE_SUFFIX}", source_id=1, messages=[])
    _create_test_archive(archive_path=second_subdirectory / f"2{LOG_ARCHIVE_SUFFIX}", source_id=2, messages=[])

    # Creates manifest with both controllers.
    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(
        MicroControllerSourceData(
            id=1, name="ctrl_1", modules=(ModuleSourceData(module_type=1, module_id=1, name="m1"),)
        )
    )
    manifest.controllers.append(
        MicroControllerSourceData(
            id=2, name="ctrl_2", modules=(ModuleSourceData(module_type=2, module_id=1, name="m2"),)
        )
    )
    manifest.save(file_path=log_dir / MICROCONTROLLER_MANIFEST_FILENAME)

    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=1,
                modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),),
                kernel=None,
            ),
            ControllerExtractionConfig(
                controller_id=2,
                modules=(ModuleExtractionConfig(module_type=2, module_id=1, event_codes=(20,)),),
                kernel=None,
            ),
        ]
    )
    config_path = tmp_path / "config.yaml"
    config.save(file_path=config_path)

    with pytest.raises(ValueError, match="span multiple directories"):
        run_log_processing_pipeline(
            log_directory=log_dir,
            output_directory=tmp_path / "output",
            config=config_path,
            workers=1,
            display_progress=False,
        )


def test_run_log_processing_pipeline_with_kernel(tmp_path: Path) -> None:
    """Verifies pipeline with both module and kernel extraction configured."""
    source_id = 1
    module_type = 1
    module_id = 2
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "output"

    archive_path = log_dir / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (2000, _make_kernel_state_payload(command=1, event=5)),
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(
        MicroControllerSourceData(
            id=source_id,
            name="ctrl",
            modules=(ModuleSourceData(module_type=module_type, module_id=module_id, name="m"),),
        )
    )
    manifest.save(file_path=log_dir / MICROCONTROLLER_MANIFEST_FILENAME)

    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=source_id,
                modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10,)),),
                kernel=KernelExtractionConfig(event_codes=(5,)),
            )
        ]
    )
    config_path = tmp_path / "config.yaml"
    config.save(file_path=config_path)

    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        config=config_path,
        workers=1,
        display_progress=False,
    )

    data_dir = output_dir / MICROCONTROLLER_DATA_DIRECTORY
    feather_files = list(data_dir.glob(f"*{FEATHER_SUFFIX}"))
    assert len(feather_files) == 2  # Module + kernel feather files


def test_execute_job_parallel_processing(tmp_path: Path) -> None:
    """Verifies that execute_job uses parallel processing for archives with 2000+ messages."""
    source_id = 1
    module_type = 1
    module_id = 2

    # Creates an archive with PARALLEL_PROCESSING_THRESHOLD messages to trigger parallel path.
    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (i * 10, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10))
        for i in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10,)),),
        kernel=None,
    )

    # Uses 2 workers to trigger parallel processing path without the overhead of using all cores.
    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=2,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    expected_filename = (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}{MODULE_FEATHER_INFIX}{module_type}_{module_id}{FEATHER_SUFFIX}"
    )
    feather_path = output_dir / expected_filename
    assert feather_path.exists()

    dataframe = pl.read_ipc(feather_path)
    assert dataframe.shape[0] == PARALLEL_PROCESSING_THRESHOLD


def test_execute_job_parallel_with_progress(tmp_path: Path) -> None:
    """Verifies parallel execute_job with progress display enabled."""
    source_id = 1
    module_type = 1
    module_id = 2

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (i * 10, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10))
        for i in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10,)),),
        kernel=None,
    )

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=2,
        tracker=tracker,
        controller_config=config,
        display_progress=True,
    )

    expected_filename = (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}{MODULE_FEATHER_INFIX}{module_type}_{module_id}{FEATHER_SUFFIX}"
    )
    assert (output_dir / expected_filename).exists()


def test_execute_job_parallel_with_external_executor(tmp_path: Path) -> None:
    """Verifies parallel execute_job when an external executor is provided."""
    source_id = 1
    module_type = 1
    module_id = 2

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (i * 10, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10))
        for i in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10,)),),
        kernel=None,
    )

    # Provides an external executor to test the non-managed executor path.
    with ProcessPoolExecutor(max_workers=2) as external_executor:
        execute_job(
            log_path=archive_path,
            output_directory=output_dir,
            source_id=str(source_id),
            job_id=job_id,
            workers=2,
            tracker=tracker,
            controller_config=config,
            display_progress=False,
            executor=external_executor,
        )

    expected_filename = (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}{MODULE_FEATHER_INFIX}{module_type}_{module_id}{FEATHER_SUFFIX}"
    )
    assert (output_dir / expected_filename).exists()


def test_execute_job_parallel_kernel(tmp_path: Path) -> None:
    """Verifies parallel execute_job with kernel extraction to cover kernel combine path."""
    source_id = 1

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (i * 10, _make_kernel_state_payload(command=1, event=5)) for i in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(),
        kernel=KernelExtractionConfig(event_codes=(5,)),
    )

    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=2,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    expected_filename = f"{CONTROLLER_FEATHER_PREFIX}{source_id}{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}"
    feather_path = output_dir / expected_filename
    assert feather_path.exists()

    dataframe = pl.read_ipc(feather_path)
    assert dataframe.shape[0] == PARALLEL_PROCESSING_THRESHOLD


def test_execute_job_exception_handling(tmp_path: Path) -> None:
    """Verifies that execute_job records failure in tracker when an exception occurs during processing."""
    source_id = 1

    # Creates a corrupted .npz file that causes LogArchiveReader to fail during processing.
    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    # Creates an archive with a malformed onset entry (no valid int64 payload).
    entries: dict[str, NDArray[np.uint8]] = {}
    # Onset entry with truncated data (too short for int64 onset).
    onset_key = f"{source_id:03d}_00000000000000000000"
    onset_data = np.array([source_id, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)  # 9 bytes header, no payload
    entries[onset_key] = onset_data
    # Valid-looking message key.
    message_key = f"{source_id:03d}_00000000000000001000"
    message_header = np.empty(9, dtype=np.uint8)
    message_header[0] = np.uint8(source_id)
    message_header[1:9] = np.frombuffer(np.uint64(1000).tobytes(), dtype=np.uint8)
    message_payload = np.array([8, 1, 2, 3, 4], dtype=np.uint8)
    entries[message_key] = np.concatenate([message_header, message_payload])
    np.savez(str(archive_path), **entries)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=1, module_id=2, event_codes=(10,)),),
        kernel=None,
    )

    with pytest.raises(ValueError):
        execute_job(
            log_path=archive_path,
            output_directory=output_dir,
            source_id=str(source_id),
            job_id=job_id,
            workers=1,
            tracker=tracker,
            controller_config=config,
            display_progress=False,
        )

    # Verifies that the tracker recorded the failure.
    assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


def test_execute_job_parallel_auto_workers(tmp_path: Path) -> None:
    """Verifies parallel execute_job with workers=-1 to trigger automatic worker count resolution."""
    source_id = 1
    module_type = 1
    module_id = 2

    archive_path = tmp_path / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (i * 10, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10))
        for i in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    tracker = ProcessingTracker(file_path=output_dir / TRACKER_FILENAME)
    tracker.initialize_jobs(jobs=[("microcontroller_data_extraction", str(source_id))])
    job_id = ProcessingTracker.generate_job_id(job_name="microcontroller_data_extraction", specifier=str(source_id))

    config = ControllerExtractionConfig(
        controller_id=source_id,
        modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10,)),),
        kernel=None,
    )

    # Uses workers=-1 to trigger resolve_worker_count inside execute_job's parallel path.
    execute_job(
        log_path=archive_path,
        output_directory=output_dir,
        source_id=str(source_id),
        job_id=job_id,
        workers=-1,
        tracker=tracker,
        controller_config=config,
        display_progress=False,
    )

    expected_filename = (
        f"{CONTROLLER_FEATHER_PREFIX}{source_id}{MODULE_FEATHER_INFIX}{module_type}_{module_id}{FEATHER_SUFFIX}"
    )
    assert (output_dir / expected_filename).exists()


def test_run_log_processing_pipeline_local_mode_multi_worker(tmp_path: Path) -> None:
    """Verifies local mode pipeline with multiple workers to cover shared executor shutdown."""
    source_id = 1
    module_type = 1
    module_id = 2

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "output"

    archive_path = log_dir / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, _make_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
    ]
    _create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(
        MicroControllerSourceData(
            id=source_id,
            name="ctrl",
            modules=(ModuleSourceData(module_type=module_type, module_id=module_id, name="m"),),
        )
    )
    manifest.save(file_path=log_dir / MICROCONTROLLER_MANIFEST_FILENAME)

    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=source_id,
                modules=(ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=(10,)),),
                kernel=None,
            )
        ]
    )
    config_path = tmp_path / "config.yaml"
    config.save(file_path=config_path)

    # Uses workers=2 to create a shared executor (resolved_workers > 1), which covers the shutdown path.
    run_log_processing_pipeline(
        log_directory=log_dir,
        output_directory=output_dir,
        config=config_path,
        workers=2,
        display_progress=False,
    )

    data_dir = output_dir / MICROCONTROLLER_DATA_DIRECTORY
    feather_files = list(data_dir.glob(f"*{FEATHER_SUFFIX}"))
    assert len(feather_files) == 1
