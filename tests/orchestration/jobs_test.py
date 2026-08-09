"""Contains tests for the classes and functions provided by the orchestration/jobs.py module."""

from pathlib import Path

import pytest
from tests.log_archives import create_test_archive, make_module_state_payload
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingTracker

from ataraxis_communication_interface.orchestration.jobs import (
    FEATHER_SUFFIX,
    TRACKER_FILENAME,
    EXTRACTION_JOB_NAME,
    KERNEL_FEATHER_INFIX,
    MODULE_FEATHER_INFIX,
    CONTROLLER_FEATHER_PREFIX,
    MICROCONTROLLER_DATA_DIRECTORY,
    PendingJob,
    generate_job_ids,
    find_module_feathers,
    parse_module_feather_name,
    resolve_kernel_feather_path,
    resolve_module_feather_path,
    discover_microcontroller_jobs,
)
from ataraxis_communication_interface.microcontroller.dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData,
    MicroControllerManifest,
    write_microcontroller_manifest,
)


def _write_archive(directory, source_id):
    """Writes a synthetic log archive for the target source into the specified directory."""
    directory.mkdir(parents=True, exist_ok=True)
    create_test_archive(
        archive_path=directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
        source_id=source_id,
        messages=[(1000, make_module_state_payload(module_type=1, module_id=2, command=1, event=10))],
    )


def _write_manifest_entry(directory, source_id):
    """Appends a single-module controller entry for the target source to the manifest in the specified directory."""
    write_microcontroller_manifest(
        log_directory=directory,
        controller_id=source_id,
        controller_name=f"controller_{source_id}",
        modules=(ModuleSourceData(module_type=1, module_id=2, name="test_module"),),
    )


def test_constants():
    """Verifies that the module-level job identity constants have the expected values."""
    assert EXTRACTION_JOB_NAME == "microcontroller_data_extraction"
    assert TRACKER_FILENAME == "microcontroller_processing_tracker.yaml"
    assert MICROCONTROLLER_DATA_DIRECTORY == "microcontroller_data"
    assert CONTROLLER_FEATHER_PREFIX == "controller_"
    assert MODULE_FEATHER_INFIX == "_module_"
    assert KERNEL_FEATHER_INFIX == "_kernel"
    assert FEATHER_SUFFIX == ".feather"


def test_pending_job_creation():
    """Verifies that PendingJob stores every supplied field and applies the documented weight defaults."""
    job = PendingJob(
        log_directory="/logs",
        output_directory="/output",
        tracker_path="/output/microcontroller_processing_tracker.yaml",
        job_id="abc123",
        source_id="1",
        config_path="/config.yaml",
    )

    assert job.log_directory == "/logs"
    assert job.output_directory == "/output"
    assert job.tracker_path == "/output/microcontroller_processing_tracker.yaml"
    assert job.job_id == "abc123"
    assert job.source_id == "1"
    assert job.config_path == "/config.yaml"
    assert job.core_weight == 1
    assert job.memory_mb == 0
    assert job.archive_path is None


def test_pending_job_weight_overrides(tmp_path):
    """Verifies that PendingJob honors explicitly supplied core, memory, and archive overrides."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    job = PendingJob(
        log_directory=tmp_path,
        output_directory=tmp_path / "output",
        tracker_path=tmp_path / "output" / TRACKER_FILENAME,
        job_id="abc123",
        source_id="1",
        config_path=tmp_path / "config.yaml",
        core_weight=5,
        memory_mb=2048,
        archive_path=archive_path,
    )

    assert job.core_weight == 5
    assert job.memory_mb == 2048
    assert job.archive_path == archive_path


def test_pending_job_dispatch_key(tmp_path):
    """Verifies that dispatch_key pairs the stringified tracker path with the job identifier."""
    tracker_path = tmp_path / MICROCONTROLLER_DATA_DIRECTORY / TRACKER_FILENAME
    job = PendingJob(
        log_directory=tmp_path,
        output_directory=tmp_path / "output",
        tracker_path=tracker_path,
        job_id="deadbeef",
        source_id="7",
        config_path=tmp_path / "config.yaml",
    )

    assert job.dispatch_key == (str(tracker_path), "deadbeef")


def test_pending_job_dispatch_key_separates_trackers(tmp_path):
    """Verifies that dispatch_key distinguishes identical job identifiers stored under different trackers."""
    first = PendingJob(
        log_directory=tmp_path,
        output_directory=tmp_path / "first",
        tracker_path=tmp_path / "first" / TRACKER_FILENAME,
        job_id="shared",
        source_id="1",
        config_path=tmp_path / "config.yaml",
    )
    second = PendingJob(
        log_directory=tmp_path,
        output_directory=tmp_path / "second",
        tracker_path=tmp_path / "second" / TRACKER_FILENAME,
        job_id="shared",
        source_id="1",
        config_path=tmp_path / "config.yaml",
    )

    assert first.dispatch_key != second.dispatch_key


def test_generate_job_ids():
    """Verifies that generate_job_ids maps every requested source to its tracker-derived job identifier."""
    job_ids = generate_job_ids(source_ids=["1", "2", "10"])

    assert set(job_ids) == {"1", "2", "10"}
    for source_id in ("1", "2", "10"):
        expected_id = ProcessingTracker.generate_job_id(job_name=EXTRACTION_JOB_NAME, specifier=source_id)
        assert job_ids[source_id] == expected_id


def test_generate_job_ids_is_deterministic():
    """Verifies that generate_job_ids returns the same identifiers across repeated calls."""
    assert generate_job_ids(source_ids=["1", "2"]) == generate_job_ids(source_ids=["1", "2"])


def test_generate_job_ids_distinguishes_sources():
    """Verifies that generate_job_ids assigns a distinct identifier to every distinct source."""
    job_ids = generate_job_ids(source_ids=["1", "2", "3"])

    assert len(set(job_ids.values())) == 3


def test_generate_job_ids_empty_input():
    """Verifies that generate_job_ids returns an empty mapping when no sources are requested."""
    assert generate_job_ids(source_ids=[]) == {}


def test_resolve_module_feather_path(tmp_path):
    """Verifies that resolve_module_feather_path builds the feather path inside the requested directory."""
    path = resolve_module_feather_path(output_directory=tmp_path, source_id="1", module_type=2, module_id=3)

    assert path == tmp_path / "controller_1_module_2_3.feather"
    assert path.parent == tmp_path
    assert path.name == "controller_1_module_2_3.feather"


def test_resolve_module_feather_path_composition(tmp_path):
    """Verifies that resolve_module_feather_path composes the filename from the prefix, infix, and suffix constants."""
    path = resolve_module_feather_path(output_directory=tmp_path, source_id="42", module_type=7, module_id=9)

    assert path.name == f"{CONTROLLER_FEATHER_PREFIX}42{MODULE_FEATHER_INFIX}7_9{FEATHER_SUFFIX}"
    assert path.name.startswith(CONTROLLER_FEATHER_PREFIX)
    assert path.name.endswith(FEATHER_SUFFIX)


def test_resolve_kernel_feather_path(tmp_path):
    """Verifies that resolve_kernel_feather_path builds the kernel feather path inside the requested directory."""
    path = resolve_kernel_feather_path(output_directory=tmp_path, source_id="1")

    assert path == tmp_path / "controller_1_kernel.feather"
    assert path.parent == tmp_path
    assert path.name == "controller_1_kernel.feather"


def test_resolve_kernel_feather_path_composition(tmp_path):
    """Verifies that resolve_kernel_feather_path composes the filename from the prefix, infix, and suffix constants."""
    path = resolve_kernel_feather_path(output_directory=tmp_path, source_id="42")

    assert path.name == f"{CONTROLLER_FEATHER_PREFIX}42{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}"
    assert path.name.startswith(CONTROLLER_FEATHER_PREFIX)
    assert path.name.endswith(FEATHER_SUFFIX)


def test_discover_microcontroller_jobs_missing_directory(tmp_path):
    """Verifies that discover_microcontroller_jobs raises FileNotFoundError when the log directory does not exist."""
    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to discover microcontroller data extraction jobs in '{missing_directory}'. The path does not exist "
        f"or is not a directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        discover_microcontroller_jobs(log_directory=missing_directory)


def test_discover_microcontroller_jobs_not_a_directory(tmp_path):
    """Verifies that discover_microcontroller_jobs raises FileNotFoundError when the log path points to a file."""
    file_path = tmp_path / "logs.txt"
    file_path.write_text("not a directory")
    message = (
        f"Unable to discover microcontroller data extraction jobs in '{file_path}'. The path does not exist "
        f"or is not a directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        discover_microcontroller_jobs(log_directory=file_path)


def test_discover_microcontroller_jobs_missing_manifest(tmp_path):
    """Verifies that discover_microcontroller_jobs raises FileNotFoundError when the tree holds no manifest."""
    _write_archive(directory=tmp_path, source_id=1)
    message = (
        f"Unable to discover microcontroller data extraction jobs in '{tmp_path}'. No "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} was found. A microcontroller manifest is required to identify which "
        f"log archives were produced by ataraxis-communication-interface."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        discover_microcontroller_jobs(log_directory=tmp_path)


def test_discover_microcontroller_jobs_empty_manifest(tmp_path):
    """Verifies that discover_microcontroller_jobs raises ValueError when the manifest registers no controllers."""
    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    MicroControllerManifest(controllers=[]).to_yaml(file_path=manifest_path)
    message = (
        f"Unable to discover microcontroller data extraction jobs in '{tmp_path}'. The "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} at '{manifest_path}' contains no controller entries."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        discover_microcontroller_jobs(log_directory=tmp_path)


def test_discover_microcontroller_jobs_resolves_manifest_controllers(tmp_path):
    """Verifies that discover_microcontroller_jobs returns one sorted string-specified entry per controller."""
    for source_id in (1, 10, 2):
        _write_manifest_entry(directory=tmp_path, source_id=source_id)
        _write_archive(directory=tmp_path, source_id=source_id)

    universe, possible = discover_microcontroller_jobs(log_directory=tmp_path)

    assert universe == [(EXTRACTION_JOB_NAME, "1"), (EXTRACTION_JOB_NAME, "10"), (EXTRACTION_JOB_NAME, "2")]
    assert possible == universe
    assert all(isinstance(specifier, str) for _, specifier in universe)


def test_discover_microcontroller_jobs_deduplicates_repeated_controllers(tmp_path):
    """Verifies that discover_microcontroller_jobs collapses repeated entries for the same controller into one job."""
    _write_manifest_entry(directory=tmp_path, source_id=1)
    _write_manifest_entry(directory=tmp_path, source_id=1)
    _write_archive(directory=tmp_path, source_id=1)

    universe, possible = discover_microcontroller_jobs(log_directory=tmp_path)

    assert universe == [(EXTRACTION_JOB_NAME, "1")]
    assert possible == [(EXTRACTION_JOB_NAME, "1")]


def test_discover_microcontroller_jobs_finds_manifest_and_archives_in_subdirectories(tmp_path):
    """Verifies that discover_microcontroller_jobs searches the whole tree for the manifest and the log archives."""
    logger_directory = tmp_path / "logger"
    logger_directory.mkdir()
    _write_manifest_entry(directory=logger_directory, source_id=3)
    _write_archive(directory=logger_directory / "archives", source_id=3)

    universe, possible = discover_microcontroller_jobs(log_directory=tmp_path)

    assert universe == [(EXTRACTION_JOB_NAME, "3")]
    assert possible == [(EXTRACTION_JOB_NAME, "3")]


def test_discover_microcontroller_jobs_excludes_controller_without_archive(tmp_path):
    """Verifies that a controller registered without an archive stays in the universe but not in the possible set."""
    _write_manifest_entry(directory=tmp_path, source_id=1)
    _write_manifest_entry(directory=tmp_path, source_id=2)
    _write_archive(directory=tmp_path, source_id=1)

    universe, possible = discover_microcontroller_jobs(log_directory=tmp_path)

    assert universe == [(EXTRACTION_JOB_NAME, "1"), (EXTRACTION_JOB_NAME, "2")]
    assert possible == [(EXTRACTION_JOB_NAME, "1")]


def test_discover_microcontroller_jobs_excludes_ambiguous_controller(tmp_path):
    """Verifies that a controller whose archive name resolves to several files is excluded from the possible set."""
    _write_manifest_entry(directory=tmp_path, source_id=1)
    _write_manifest_entry(directory=tmp_path, source_id=2)
    _write_archive(directory=tmp_path / "logger_one", source_id=1)
    _write_archive(directory=tmp_path / "logger_one", source_id=2)
    _write_archive(directory=tmp_path / "logger_two", source_id=2)

    universe, possible = discover_microcontroller_jobs(log_directory=tmp_path)

    assert universe == [(EXTRACTION_JOB_NAME, "1"), (EXTRACTION_JOB_NAME, "2")]
    assert possible == [(EXTRACTION_JOB_NAME, "1")]


def test_find_module_feathers(tmp_path: Path) -> None:
    """Verifies that find_module_feathers discovers every module feather the directory holds, sorted by path."""
    second = resolve_module_feather_path(output_directory=tmp_path, source_id="2", module_type=3, module_id=4)
    first = resolve_module_feather_path(output_directory=tmp_path, source_id="1", module_type=1, module_id=2)
    for path in (second, first):
        path.touch()

    # The kernel feather and an unrelated file share the directory, so neither may appear in the module result.
    resolve_kernel_feather_path(output_directory=tmp_path, source_id="1").touch()
    tmp_path.joinpath("notes.txt").touch()

    assert find_module_feathers(data_directory=tmp_path) == [first, second]


def test_find_module_feathers_missing_directory(tmp_path: Path) -> None:
    """Verifies that find_module_feathers reports no feathers for a directory that does not exist."""
    assert find_module_feathers(data_directory=tmp_path / "absent") == []


def test_parse_module_feather_name_inverts_resolution(tmp_path: Path) -> None:
    """Verifies that parse_module_feather_name recovers the identity resolve_module_feather_path encoded."""
    feather_path = resolve_module_feather_path(output_directory=tmp_path, source_id="222", module_type=5, module_id=7)

    assert parse_module_feather_name(feather_path=feather_path) == (222, 5, 7)


@pytest.mark.parametrize(
    "filename",
    [
        "controller_1_module_2.feather",
        "controller_1_kernel.feather",
        "session_1_module_2_3.feather",
        "controller_1_widget_2_3.feather",
        "controller_one_module_2_3.feather",
        "controller_1_module_two_3.feather",
    ],
)
def test_parse_module_feather_name_rejects_foreign_names(tmp_path: Path, filename: str) -> None:
    """Verifies that parse_module_feather_name rejects a filename outside the module feather convention."""
    message = (
        f"Unable to parse the module feather filename '{filename}'. The filename does not follow the "
        f"'controller_{{source_id}}_module_{{module_type}}_{{module_id}}.feather' naming convention."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        parse_module_feather_name(feather_path=tmp_path / filename)
