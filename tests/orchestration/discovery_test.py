"""Contains tests for the classes and functions provided by the orchestration/discovery.py module."""

from typing import Any, NoReturn
from pathlib import Path

import pytest
from tests.log_archives import create_test_archive, create_module_state_payload
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingStatus, ProcessingTracker

from ataraxis_communication_interface.orchestration import discovery
from ataraxis_communication_interface.orchestration.jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    resolve_tracker_path,
    resolve_output_directory,
)
from ataraxis_communication_interface.orchestration.discovery import (
    JobSet,
    JobSource,
    JobUniverse,
    size_job,
    prepare_jobs,
    resolve_jobs,
)
from ataraxis_communication_interface.orchestration.allocation import (
    PARALLEL_EXTRACTION_THRESHOLD,
    CONTROLLER_EXTRACTION_JOB_CORES,
    resolve_job_workers,
    estimate_job_memory_mb,
)
from ataraxis_communication_interface.microcontroller.dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    ModuleSourceData,
    ModuleExtractionConfig,
    MicroControllerManifest,
    MicroControllerSourceData,
    ControllerExtractionConfig,
    write_microcontroller_manifest,
)

_ONSET_US: int = 1700000000000000
"""Stores the UTC epoch onset, in microseconds, written into every synthetic log archive built by this module."""

_MODULE_TYPE: int = 1
"""Stores the type (family) code of the only hardware module every synthetic controller manages."""

_MODULE_ID: int = 2
"""Stores the identifier code of the only hardware module every synthetic controller manages."""

_EVENT_CODES: tuple[int, ...] = (10, 20)
"""Stores the event codes every synthetic extraction configuration declares for its module."""

_WIDE_ARCHIVE_MESSAGES: int = PARALLEL_EXTRACTION_THRESHOLD
"""Stores the message count of the archive used to exercise the multi-core branch of the sizing model."""


def test_job_source_fields(tmp_path: Path) -> None:
    """Verifies that JobSource stores the source identifier, the manifest name, and the resolved archive path."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    source = JobSource(source_id="1", name="controller1", archive_path=archive_path)

    assert source.source_id == "1"
    assert source.name == "controller1"
    assert source.archive_path == archive_path

    # The record is frozen, so a resolved source never drifts after a consumer receives it.
    with pytest.raises(AttributeError):
        source.source_id = "2"


def test_resolve_jobs_resolves_manifest_sources(tmp_path: Path) -> None:
    """Verifies that resolve_jobs returns one sorted string-specified entry per source the manifest registers."""
    _build_recording(log_directory=tmp_path, source_ids=(1, 10, 2))

    universe = resolve_jobs(log_directory=tmp_path)

    assert isinstance(universe, JobUniverse)
    assert universe.log_directory == tmp_path
    assert universe.manifest_path == tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    assert universe.universe == (
        (CONTROLLER_EXTRACTION_JOB_NAME, "1"),
        (CONTROLLER_EXTRACTION_JOB_NAME, "10"),
        (CONTROLLER_EXTRACTION_JOB_NAME, "2"),
    )
    assert universe.possible == universe.universe
    assert all(isinstance(specifier, str) for _, specifier in universe.universe)
    assert [source.source_id for source in universe.sources] == ["1", "10", "2"]
    assert [source.name for source in universe.sources] == ["controller1", "controller10", "controller2"]


def test_resolve_jobs_archives_property(tmp_path: Path) -> None:
    """Verifies that the archives property keys every resolved archive by the source identifier that produced it."""
    _build_recording(log_directory=tmp_path, source_ids=(1, 2))
    _write_manifest_entry(log_directory=tmp_path, source_id=3)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.archives == {
        "1": tmp_path / f"1{LOG_ARCHIVE_SUFFIX}",
        "2": tmp_path / f"2{LOG_ARCHIVE_SUFFIX}",
    }
    # The source without an archive is registered, but it contributes no entry to the archive mapping.
    assert "3" not in universe.archives
    assert len(universe.sources) == 3


def test_resolve_jobs_deduplicates_repeated_sources(tmp_path: Path) -> None:
    """Verifies that resolve_jobs collapses repeated manifest entries for the same source into one job."""
    _write_manifest_entry(log_directory=tmp_path, source_id=1, name="controller1")
    _write_manifest_entry(log_directory=tmp_path, source_id=1, name="controller1_again")
    _build_archive(directory=tmp_path, source_id=1)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.universe == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"),)
    assert universe.possible == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"),)
    # The last entry written for a repeated source supplies its colloquial name.
    assert [source.name for source in universe.sources] == ["controller1_again"]


def test_resolve_jobs_finds_manifest_and_archives_in_subdirectories(tmp_path: Path) -> None:
    """Verifies that resolve_jobs searches the whole tree for the manifest and for the log archives."""
    logger_directory = tmp_path / "logger"
    _write_manifest_entry(log_directory=logger_directory, source_id=3)
    _build_archive(directory=logger_directory / "archives", source_id=3)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.manifest_path == logger_directory / MICROCONTROLLER_MANIFEST_FILENAME
    assert universe.universe == ((CONTROLLER_EXTRACTION_JOB_NAME, "3"),)
    assert universe.possible == ((CONTROLLER_EXTRACTION_JOB_NAME, "3"),)
    assert universe.archives == {"3": logger_directory / "archives" / f"3{LOG_ARCHIVE_SUFFIX}"}


def test_resolve_jobs_excludes_source_without_archive(tmp_path: Path) -> None:
    """Verifies that a source registered without an archive stays in the universe but not in the possible set."""
    _build_recording(log_directory=tmp_path, source_ids=(1,))
    _write_manifest_entry(log_directory=tmp_path, source_id=2)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.universe == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"), (CONTROLLER_EXTRACTION_JOB_NAME, "2"))
    assert universe.possible == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"),)
    assert universe.sources[1].archive_path is None


def test_resolve_jobs_excludes_ambiguous_source(tmp_path: Path) -> None:
    """Verifies that a source whose archive name resolves to several files is excluded from the possible set."""
    _write_manifest_entry(log_directory=tmp_path, source_id=1)
    _write_manifest_entry(log_directory=tmp_path, source_id=2)
    _build_archive(directory=tmp_path / "logger_one", source_id=1)
    _build_archive(directory=tmp_path / "logger_one", source_id=2)
    _build_archive(directory=tmp_path / "logger_two", source_id=2)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.universe == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"), (CONTROLLER_EXTRACTION_JOB_NAME, "2"))
    # An archive name matching several files spans several loggers, which is ambiguous rather than redundant.
    assert universe.possible == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"),)
    assert universe.sources[1].archive_path is None


def test_resolve_jobs_writes_nothing(tmp_path: Path) -> None:
    """Verifies that resolve_jobs leaves the log directory untouched, materializing no output and no tracker."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    before = _snapshot_tree(directory=tmp_path)

    universe = resolve_jobs(log_directory=log_directory)

    assert universe.possible == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"), (CONTROLLER_EXTRACTION_JOB_NAME, "2"))
    assert _snapshot_tree(directory=tmp_path) == before
    assert not (log_directory / OutputLayout.DIRECTORY_NAME).exists()
    assert not list(tmp_path.rglob(OutputLayout.TRACKER_FILENAME))


def test_resolve_jobs_missing_directory(tmp_path: Path) -> None:
    """Verifies that resolve_jobs reports the missing directory kind when the log directory does not exist."""
    missing_directory = tmp_path / "nonexistent"
    message = (
        f"Unable to resolve microcontroller data extraction jobs in '{missing_directory}'. The path does not exist or "
        f"is not a directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        resolve_jobs(log_directory=missing_directory)


def test_resolve_jobs_not_a_directory(tmp_path: Path) -> None:
    """Verifies that resolve_jobs reports the missing directory kind when the log path points to a file."""
    file_path = tmp_path / "logs.txt"
    file_path.write_text("not a directory")
    message = (
        f"Unable to resolve microcontroller data extraction jobs in '{file_path}'. The path does not exist or is not "
        f"a directory."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        resolve_jobs(log_directory=file_path)


def test_resolve_jobs_returns_an_empty_universe_when_the_tree_holds_no_manifest(tmp_path: Path) -> None:
    """Verifies that resolve_jobs reports a tree holding no microcontroller manifest as holding no jobs."""
    _build_archive(directory=tmp_path, source_id=1)

    universe = resolve_jobs(log_directory=tmp_path)

    assert universe.manifest_path is None
    assert universe.sources == ()
    assert universe.universe == ()
    assert universe.possible == ()
    assert universe.archives == {}


def test_resolve_jobs_empty_manifest(tmp_path: Path) -> None:
    """Verifies that resolve_jobs reports the empty manifest kind when the manifest registers no controllers."""
    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    MicroControllerManifest(controllers=[]).to_yaml(file_path=manifest_path)
    message = (
        f"Unable to resolve microcontroller data extraction jobs in '{tmp_path}'. The "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} at '{manifest_path}' contains no controller entries."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        resolve_jobs(log_directory=tmp_path)


def test_resolve_jobs_ambiguous_log_directory(tmp_path: Path) -> None:
    """Verifies that resolve_jobs rejects a tree holding several manifests."""
    _build_recording(log_directory=tmp_path / "recording_one", source_ids=(1,))
    _build_recording(log_directory=tmp_path / "recording_two", source_ids=(2,))

    message = (
        f"Unable to resolve microcontroller data extraction jobs in '{tmp_path}'. The directory tree holds 2 "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} files, which means it spans several recordings or several DataLogger "
        f"instances:"
    )

    with pytest.raises(ValueError, match=error_format(message)) as failure:
        resolve_jobs(log_directory=tmp_path)
    # The report names every manifest found, so the caller can split the tree into its individual recordings.
    assert str(tmp_path / "recording_one" / MICROCONTROLLER_MANIFEST_FILENAME) in str(failure.value)
    assert str(tmp_path / "recording_two" / MICROCONTROLLER_MANIFEST_FILENAME) in str(failure.value)


def test_prepare_jobs_creates_output_directory_and_tracker(tmp_path: Path) -> None:
    """Verifies that prepare_jobs materializes its own output subdirectory and the tracker recording every job."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    job_set = prepare_jobs(log_directory=log_directory, output_directory=output_root, config_path=config_path)

    assert isinstance(job_set, JobSet)
    assert job_set.log_directory == log_directory
    assert job_set.output_directory == resolve_output_directory(output_directory=output_root)
    assert job_set.output_directory == output_root / OutputLayout.DIRECTORY_NAME
    assert job_set.output_directory.is_dir()
    assert job_set.tracker_path == resolve_tracker_path(output_directory=job_set.output_directory)
    assert job_set.tracker_path.is_file()
    assert job_set.skipped_sources == ()


def test_prepare_jobs_builds_descriptors(tmp_path: Path) -> None:
    """Verifies that prepare_jobs builds one fully addressed descriptor per source, in source identifier order."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1, 10, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 10, 2))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_root,
        config_path=config_path,
    )

    assert [job.source_id for job in job_set.jobs] == ["1", "10", "2"]
    identifiers = generate_job_ids(source_ids=["1", "10", "2"])
    for job in job_set.jobs:
        assert job.log_directory == log_directory
        assert job.archive_path == log_directory / f"{job.source_id}{LOG_ARCHIVE_SUFFIX}"
        assert job.archive_path.is_file()
        assert job.output_directory == job_set.output_directory
        assert job.config_path == config_path
        assert job.tracker_path == job_set.tracker_path
        assert job.job_name == CONTROLLER_EXTRACTION_JOB_NAME
        assert job.job_id == identifiers[job.source_id]
        assert job.core_weight == CONTROLLER_EXTRACTION_JOB_CORES

    # Every descriptor addresses a distinct tracker entry, so no two jobs of one set collide during dispatch.
    assert len({job.dispatch_key for job in job_set.jobs}) == len(job_set.jobs)


def test_prepare_jobs_reads_no_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that prepare_jobs resolves every job without opening or sizing a single log archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    def _explode(**kwargs: Any) -> NoReturn:
        """Fails the call, standing in for the archive pass prepare_jobs must never perform."""
        message = f"prepare_jobs read an archive: {kwargs}."
        raise AssertionError(message)

    monkeypatch.setattr(discovery, "resolve_archive_footprint", _explode)

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
    )

    assert len(job_set.jobs) == 2
    # Every job carries the declared width, because resolving the shape its own archive earns belongs to the sizing
    # pass that opens it.
    assert {job.core_weight for job in job_set.jobs} == {CONTROLLER_EXTRACTION_JOB_CORES}


def test_prepare_jobs_accepts_unreadable_archive(tmp_path: Path) -> None:
    """Verifies that prepare_jobs prepares a job whose archive cannot be decoded, since it never decodes one."""
    log_directory = tmp_path / "logs"
    _write_manifest_entry(log_directory=log_directory, source_id=5)
    (log_directory / f"5{LOG_ARCHIVE_SUFFIX}").write_text("This is not a valid numpy archive.")
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(5,))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
    )

    assert [job.source_id for job in job_set.jobs] == ["5"]
    assert job_set.jobs[0].core_weight == CONTROLLER_EXTRACTION_JOB_CORES


def test_prepare_jobs_selects_requested_sources(tmp_path: Path) -> None:
    """Verifies that prepare_jobs prepares only the requested sources while keeping the universe complete."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2, 3))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2, 3))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
        source_ids=["3", "1"],
    )

    # The requested sources are prepared in ascending identifier order, whatever order the caller named them in.
    assert [job.source_id for job in job_set.jobs] == ["1", "3"]
    assert job_set.universe == (
        (CONTROLLER_EXTRACTION_JOB_NAME, "1"),
        (CONTROLLER_EXTRACTION_JOB_NAME, "2"),
        (CONTROLLER_EXTRACTION_JOB_NAME, "3"),
    )


def test_prepare_jobs_defaults_to_the_configured_controllers(tmp_path: Path) -> None:
    """Verifies that an unrequested preparation covers the controllers the configuration declares, and only those."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
    )

    # The configuration bounds the requested set the same way the manifest bounds the universe.
    assert [job.source_id for job in job_set.jobs] == ["1"]
    assert job_set.universe == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"), (CONTROLLER_EXTRACTION_JOB_NAME, "2"))
    assert job_set.skipped_sources == ()


def test_prepare_jobs_selects_single_job_by_id(tmp_path: Path) -> None:
    """Verifies that a requested job identifier selects one job and overrides any requested source identifiers."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))
    identifiers = generate_job_ids(source_ids=["1", "2"])

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
        source_ids=["1"],
        job_id=identifiers["2"],
    )

    assert [job.source_id for job in job_set.jobs] == ["2"]
    assert job_set.jobs[0].job_id == identifiers["2"]


def test_prepare_jobs_job_id_survives_missing_sibling_archive(tmp_path: Path) -> None:
    """Verifies that a job identifier resolves against the configuration even when a sibling holds no archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))
    identifiers = generate_job_ids(source_ids=["1", "2"])

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
        job_id=identifiers["1"],
    )

    assert [job.source_id for job in job_set.jobs] == ["1"]
    assert job_set.skipped_sources == ()


def test_prepare_jobs_unknown_job_id(tmp_path: Path) -> None:
    """Verifies that prepare_jobs reports the unknown job identifier kind for an identifier the config omits."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    message = (
        f"Unable to prepare the microcontroller data extraction job 'deadbeefdeadbeef' in '{log_directory}'. The "
        f"extraction config at '{config_path}' declares no controller with that job identifier. Configured "
        f"controller IDs: 1."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config_path=config_path,
            job_id="deadbeefdeadbeef",
        )


def test_prepare_jobs_missing_config(tmp_path: Path) -> None:
    """Verifies that prepare_jobs reports the missing configuration kind before it resolves any job."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = tmp_path / "config.yaml"

    message = f"Unable to load the extraction config from '{config_path}'. The path does not exist or is not a file."

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)


def test_prepare_jobs_config_declaring_no_controllers(tmp_path: Path) -> None:
    """Verifies that prepare_jobs reports the empty configuration kind when the config declares no controllers."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=())

    message = (
        f"Unable to prepare microcontroller data extraction jobs using the extraction config at '{config_path}'. It "
        f"declares no controllers."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)


def test_prepare_jobs_config_declaring_an_unregistered_controller(tmp_path: Path) -> None:
    """Verifies that prepare_jobs rejects a configuration declaring a controller the manifest does not register."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 7))

    message = (
        f"Unable to prepare microcontroller data extraction jobs using the extraction config at '{config_path}'. The "
        f"following controller IDs are not registered in the {MICROCONTROLLER_MANIFEST_FILENAME}: 7. The "
        f"corresponding log archives were not produced by ataraxis-communication-interface. Registered IDs: 1."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)


def test_prepare_jobs_unknown_source_under_strict_sourcing(tmp_path: Path) -> None:
    """Verifies that prepare_jobs reports the unconfigured source kind for a source the configuration omits."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 9))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    message = (
        f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The following requested "
        f"controller IDs are absent from the extraction config at '{config_path}': 9. Configured controller IDs: 1."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config_path=config_path,
            source_ids=["1", "9"],
        )


def test_prepare_jobs_missing_archive_under_strict_sourcing(tmp_path: Path) -> None:
    """Verifies that prepare_jobs reports the unresolved archive kind for a registered source holding no archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    message = (
        f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The log archives of the "
        f"following requested controller IDs are absent or resolve to more than one file: 2."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        prepare_jobs(
            log_directory=log_directory,
            output_directory=tmp_path / "output",
            config_path=config_path,
            source_ids=["2"],
        )


def test_prepare_jobs_ambiguous_archive_under_strict_sourcing(tmp_path: Path) -> None:
    """Verifies that prepare_jobs reports the unresolved archive kind when a source matches several archives."""
    log_directory = tmp_path / "logs"
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    _build_archive(directory=log_directory / "logger_one", source_id=2)
    _build_archive(directory=log_directory / "logger_two", source_id=2)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(2,))

    message = (
        f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The log archives of the "
        f"following requested controller IDs are absent or resolve to more than one file: 2."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)


def test_prepare_jobs_records_skipped_sources_without_strict_sourcing(tmp_path: Path) -> None:
    """Verifies that lenient sourcing records every unpreparable source with its reason."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    _write_manifest_entry(log_directory=log_directory, source_id=9)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
        source_ids=["1", "2", "9"],
        strict_sources=False,
    )

    assert [job.source_id for job in job_set.jobs] == ["1"]
    assert job_set.skipped_sources == (
        ("2", "The controller's log archive is absent or resolves to more than one file."),
        ("9", "The controller is absent from the extraction configuration."),
    )
    # The skipped sources stay in the universe the tracker is aligned against.
    assert (CONTROLLER_EXTRACTION_JOB_NAME, "2") in job_set.universe


def test_prepare_jobs_unregistered_controller_without_strict_sourcing(tmp_path: Path) -> None:
    """Verifies that lenient sourcing records a configured controller the manifest omits."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 7))

    # A project-wide configuration declares every controller the project uses, so a recording whose manifest registers
    # a subset of them prepares the sources it holds and reports the rest as skipped.
    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
        strict_sources=False,
    )

    assert [job.source_id for job in job_set.jobs] == ["1"]
    assert job_set.skipped_sources == (("7", "The controller is absent from the microcontroller manifest."),)
    # The unregistered controller contributes no entry to the universe the tracker is aligned against.
    assert job_set.universe == ((CONTROLLER_EXTRACTION_JOB_NAME, "1"),)


def test_prepare_jobs_split_logger_output(tmp_path: Path) -> None:
    """Verifies that prepare_jobs reports the split logger output kind when the archives span several directories."""
    log_directory = tmp_path / "logs"
    _write_manifest_entry(log_directory=log_directory, source_id=1)
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    _build_archive(directory=log_directory / "logger_one", source_id=1)
    _build_archive(directory=log_directory / "logger_two", source_id=2)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    message = (
        f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The resolved log archives sit "
        f"in 2 different directories:"
    )

    with pytest.raises(ValueError, match=error_format(message)) as failure:
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)
    assert str(log_directory / "logger_one") in str(failure.value)
    assert str(log_directory / "logger_two") in str(failure.value)


def test_prepare_jobs_guards_run_before_any_write(tmp_path: Path) -> None:
    """Verifies that a rejected preparation creates neither the output subdirectory nor the tracker."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _write_manifest_entry(log_directory=log_directory, source_id=1)
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    _build_archive(directory=log_directory / "logger_one", source_id=1)
    _build_archive(directory=log_directory / "logger_two", source_id=2)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    message = (
        f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The resolved log archives sit "
        f"in 2 different directories:"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=output_root, config_path=config_path)

    assert not output_root.exists()
    assert not list(tmp_path.rglob(OutputLayout.TRACKER_FILENAME))


def test_prepare_jobs_propagates_manifest_guards(tmp_path: Path) -> None:
    """Verifies that prepare_jobs surfaces the resolution guards of the universe it prepares from."""
    log_directory = tmp_path / "logs"
    _build_archive(directory=log_directory, source_id=1)
    _build_archive(directory=log_directory / "second", source_id=2)
    module = ModuleSourceData(module_type=_MODULE_TYPE, module_id=_MODULE_ID, name="test_module")
    MicroControllerManifest(controllers=[MicroControllerSourceData(id=1, name="one", modules=(module,))]).to_yaml(
        file_path=log_directory / MICROCONTROLLER_MANIFEST_FILENAME
    )
    MicroControllerManifest(controllers=[MicroControllerSourceData(id=2, name="two", modules=(module,))]).to_yaml(
        file_path=log_directory / "second" / MICROCONTROLLER_MANIFEST_FILENAME
    )
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    message = (
        f"Unable to resolve microcontroller data extraction jobs in '{log_directory}'. The directory tree holds 2 "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} files,"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)


def test_prepare_jobs_registers_prepared_jobs_on_the_tracker(tmp_path: Path) -> None:
    """Verifies that prepare_jobs registers every prepared job on the tracker as a scheduled job."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
        source_ids=["1"],
    )

    tracker = ProcessingTracker(file_path=job_set.tracker_path)
    identifiers = generate_job_ids(source_ids=["1", "2"])
    # Only the requested job is registered, since the universe governs foreign detection rather than registration.
    assert tracker.find_jobs() == {identifiers["1"]: (CONTROLLER_EXTRACTION_JOB_NAME, "1")}
    assert tracker.get_job_status(job_id=identifiers["1"]) == ProcessingStatus.SCHEDULED


def test_prepare_jobs_preserves_sibling_job_state(tmp_path: Path) -> None:
    """Verifies that preparing one source leaves the recorded outcome of a sibling source's job untouched."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))
    identifiers = generate_job_ids(source_ids=["1", "2"])

    first = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_root,
        config_path=config_path,
        source_ids=["1"],
    )
    tracker = ProcessingTracker(file_path=first.tracker_path)
    tracker.start_job(job_id=identifiers["1"])
    tracker.complete_job(job_id=identifiers["1"])

    second = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_root,
        config_path=config_path,
        source_ids=["2"],
    )

    assert second.tracker_path == first.tracker_path
    tracker = ProcessingTracker(file_path=second.tracker_path)
    assert tracker.get_job_status(job_id=identifiers["1"]) == ProcessingStatus.SUCCEEDED
    assert tracker.get_job_status(job_id=identifiers["2"]) == ProcessingStatus.SCHEDULED


def test_prepare_jobs_discards_out_of_universe_tracker_entries(tmp_path: Path) -> None:
    """Verifies that a tracker entry outside the manifest universe is discarded when the jobs are prepared."""
    log_directory = tmp_path / "logs"
    output_root = tmp_path / "output"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))

    # Registers a job for a source the manifest never registered, standing in for a stale tracker.
    resolved_output = resolve_output_directory(output_directory=output_root)
    resolved_output.mkdir(parents=True)
    tracker_path = resolve_tracker_path(output_directory=resolved_output)
    foreign_job = (CONTROLLER_EXTRACTION_JOB_NAME, "99")
    ProcessingTracker(file_path=tracker_path).align_jobs(jobs=[foreign_job], universe=[foreign_job])

    job_set = prepare_jobs(log_directory=log_directory, output_directory=output_root, config_path=config_path)

    tracker = ProcessingTracker(file_path=job_set.tracker_path)
    identifiers = generate_job_ids(source_ids=["1", "99"])
    assert identifiers["99"] not in tracker.find_jobs()
    assert tracker.find_jobs() == {identifiers["1"]: (CONTROLLER_EXTRACTION_JOB_NAME, "1")}


def test_prepare_jobs_returns_an_empty_set_when_lenient_sourcing_skips_every_source(tmp_path: Path) -> None:
    """Verifies that a lenient request preparing no job returns an empty job set."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
        source_ids=["2"],
        strict_sources=False,
    )

    assert job_set.jobs == ()
    assert [source_id for source_id, _ in job_set.skipped_sources] == ["2"]


def test_prepare_jobs_creates_no_output_directory_when_it_prepares_no_job(tmp_path: Path) -> None:
    """Verifies that a lenient request preparing no job leaves the caller's output path untouched."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    _write_manifest_entry(log_directory=log_directory, source_id=2)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))
    output_directory = tmp_path / "output"

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_directory,
        config_path=config_path,
        source_ids=["2"],
        strict_sources=False,
    )

    assert not resolve_output_directory(output_directory=output_directory).exists()
    assert not job_set.tracker_path.exists()


def test_size_job_applies_the_memory_model(tmp_path: Path) -> None:
    """Verifies that size_job reports the cores and the memory the allocation model resolves for the job's archive."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,), message_count=_WIDE_ARCHIVE_MESSAGES)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)

    sized_job, sizing, footprint = size_job(job=job_set.jobs[0])

    expected_cores = resolve_job_workers(footprint=footprint)
    assert sized_job.core_weight == expected_cores
    # An archive holding the threshold's worth of messages takes the parallel shape, which is the branch of the
    # memory model that charges one baseline and one reader per pool child on top of the job body's own pair.
    assert sized_job.core_weight == CONTROLLER_EXTRACTION_JOB_CORES
    assert sizing.cores == expected_cores
    assert sizing.memory_mb == estimate_job_memory_mb(footprint=footprint, cores=expected_cores)
    assert footprint.message_count == _WIDE_ARCHIVE_MESSAGES
    assert footprint.archive_bytes == job_set.jobs[0].archive_path.stat().st_size


def test_size_job_rejects_an_unreadable_archive(tmp_path: Path) -> None:
    """Verifies that size_job rejects an archive it cannot read rather than charging a baseline floor."""
    log_directory = tmp_path / "logs"
    _write_manifest_entry(log_directory=log_directory, source_id=1)
    (log_directory / f"1{LOG_ARCHIVE_SUFFIX}").write_text("This is not a valid numpy archive.")
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
    )
    message = (
        f"Unable to size the microcontroller data extraction job that reads the log archive "
        f"{job_set.jobs[0].archive_path}. The archive cannot be read, so the job reading it cannot run. Verify that "
        f"the path names a readable .npz log archive."
    )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        size_job(job=job_set.jobs[0])


def test_size_job_preserves_descriptor_identity(tmp_path: Path) -> None:
    """Verifies that size_job returns the supplied descriptor with its width replaced and every other field kept."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    job = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
    ).jobs[0]

    sized_job, _, _ = size_job(job=job)

    assert sized_job is not job
    # The preparation stamps the declared width on every descriptor, so an archive this small proves the sizing pass
    # replaces that width rather than passing it through.
    assert job.core_weight == CONTROLLER_EXTRACTION_JOB_CORES
    assert sized_job.core_weight == 1
    assert sized_job.dispatch_key == job.dispatch_key
    for field_name in (
        "log_directory",
        "archive_path",
        "output_directory",
        "config_path",
        "tracker_path",
        "job_name",
        "source_id",
    ):
        assert getattr(sized_job, field_name) == getattr(job, field_name)


def test_job_set_resolve_job(tmp_path: Path) -> None:
    """Verifies that JobSet.resolve_job returns the descriptor carrying the requested job identifier."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1, 2))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1, 2))
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)
    identifiers = generate_job_ids(source_ids=["1", "2"])

    for source_id in ("1", "2"):
        resolved = job_set.resolve_job(job_id=identifiers[source_id])

        assert resolved.job_id == identifiers[source_id]
        assert resolved.source_id == source_id
        assert resolved in job_set.jobs


def test_job_set_resolve_job_unknown_id(tmp_path: Path) -> None:
    """Verifies that JobSet.resolve_job reports the unknown job identifier kind and names the jobs it does hold."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)

    message = (
        f"Unable to resolve the microcontroller data extraction job 'deadbeefdeadbeef' in '{log_directory}'. The "
        f"prepared job set holds no job with that identifier. Held job IDs: {job_set.jobs[0].job_id}."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        job_set.resolve_job(job_id="deadbeefdeadbeef")


def test_job_set_resolve_job_empty_set(tmp_path: Path) -> None:
    """Verifies that JobSet.resolve_job reports the unknown job identifier kind when the set holds no job at all."""
    job_set = JobSet(
        log_directory=tmp_path,
        output_directory=tmp_path / OutputLayout.DIRECTORY_NAME,
        tracker_path=tmp_path / OutputLayout.DIRECTORY_NAME / OutputLayout.TRACKER_FILENAME,
        universe=((CONTROLLER_EXTRACTION_JOB_NAME, "1"),),
        jobs=(),
        skipped_sources=(("1", "The controller's log archive is absent or resolves to more than one file."),),
    )

    message = (
        f"Unable to resolve the microcontroller data extraction job 'deadbeefdeadbeef' in '{tmp_path}'. The prepared "
        f"job set holds no job with that identifier. Held job IDs: none."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        job_set.resolve_job(job_id="deadbeefdeadbeef")


def test_job_descriptor_from_mapping_missing_key(tmp_path: Path) -> None:
    """Verifies that JobDescriptor.from_mapping reports the malformed descriptor kind for an incomplete mapping."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)
    mapping = job_set.jobs[0].to_mapping()
    del mapping["archive_path"]

    message = (
        "Unable to read a microcontroller data extraction job descriptor from the supplied mapping. The following "
        "required keys are absent: archive_path."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=mapping)


def test_job_descriptor_from_mapping_unreadable_value(tmp_path: Path) -> None:
    """Verifies that JobDescriptor.from_mapping reports the malformed descriptor kind for an unreadable value."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    job_set = prepare_jobs(log_directory=log_directory, output_directory=tmp_path / "output", config_path=config_path)
    mapping = job_set.jobs[0].to_mapping()
    mapping["core_weight"] = "not an integer"

    message = (
        "Unable to read a microcontroller data extraction job descriptor from the supplied mapping. One of its values "
        "cannot be read as the type its field declares:"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=mapping)


def test_job_descriptor_round_trips_through_a_mapping(tmp_path: Path) -> None:
    """Verifies that a prepared descriptor survives the flat mapping the interface layer exchanges it through."""
    log_directory = tmp_path / "logs"
    _build_recording(log_directory=log_directory, source_ids=(1,))
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    job = prepare_jobs(
        log_directory=log_directory,
        output_directory=tmp_path / "output",
        config_path=config_path,
    ).jobs[0]

    assert JobDescriptor.from_mapping(mapping=job.to_mapping()) == job


def test_prepare_jobs_prepares_nothing_when_the_tree_holds_no_manifest(tmp_path: Path) -> None:
    """Verifies that a tree holding no manifest prepares no job whatever the configuration declares."""
    log_directory = tmp_path / "logs"
    _build_archive(directory=log_directory, source_id=1)
    config_path = _write_config(config_path=tmp_path / "config.yaml", source_ids=(1,))
    output_directory = tmp_path / "output"

    job_set = prepare_jobs(
        log_directory=log_directory,
        output_directory=output_directory,
        config_path=config_path,
    )

    assert job_set.jobs == ()
    assert job_set.universe == ()
    assert not resolve_output_directory(output_directory=output_directory).exists()


def _build_archive(directory: Path, source_id: int, message_count: int = 3) -> None:
    """Writes a synthetic log archive holding the requested number of module messages for the target source."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = create_module_state_payload(module_type=_MODULE_TYPE, module_id=_MODULE_ID, command=1, event=10)
    create_test_archive(
        archive_path=directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}",
        source_id=source_id,
        messages=[(elapsed_us, payload) for elapsed_us in range(1, message_count + 1)],
        onset_us=_ONSET_US,
    )


def _write_manifest_entry(log_directory: Path, source_id: int, name: str | None = None) -> None:
    """Registers a single-module controller entry for the target source in the manifest of the target directory,
    replacing any entry the manifest already holds for it."""
    log_directory.mkdir(parents=True, exist_ok=True)
    write_microcontroller_manifest(
        log_directory=log_directory,
        controller_id=source_id,
        controller_name=name if name is not None else f"controller{source_id}",
        modules=(ModuleSourceData(module_type=_MODULE_TYPE, module_id=_MODULE_ID, name="test_module"),),
    )


def _build_recording(log_directory: Path, source_ids: tuple[int, ...], message_count: int = 3) -> None:
    """Writes one manifest entry and one synthetic log archive for each of the requested controller sources."""
    for source_id in source_ids:
        _write_manifest_entry(log_directory=log_directory, source_id=source_id)
        _build_archive(directory=log_directory, source_id=source_id, message_count=message_count)


def _write_config(config_path: Path, source_ids: tuple[int, ...]) -> Path:
    """Writes an extraction configuration declaring one single-module controller for each requested source."""
    ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=source_id,
                modules=(
                    ModuleExtractionConfig(module_type=_MODULE_TYPE, module_id=_MODULE_ID, event_codes=_EVENT_CODES),
                ),
                kernel=None,
            )
            for source_id in source_ids
        ]
    ).to_yaml(file_path=config_path)
    return config_path


def _snapshot_tree(directory: Path) -> dict[Path, tuple[bool, int, int]]:
    """Captures the path, the directory flag, the size, and the modification time of every filesystem entry under
    the target directory."""
    return {
        path: (path.is_dir(), path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(directory.rglob("*"))
    }
