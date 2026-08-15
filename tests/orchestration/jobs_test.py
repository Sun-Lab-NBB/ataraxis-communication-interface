"""Contains tests for the classes and functions provided by the orchestration/jobs.py module."""

import pickle
from pathlib import Path
from dataclasses import fields, replace
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest
from tests.log_archives import create_test_archive, write_extraction_config, create_module_state_payload
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, ProcessingTracker

from ataraxis_communication_interface.orchestration.jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    JobSizing,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    find_kernel_paths,
    find_module_paths,
    parse_kernel_path,
    parse_module_path,
    resolve_kernel_path,
    resolve_module_path,
    resolve_tracker_path,
    resolve_output_directory,
)

_ONSET_US: int = 1700000000000000
"""Stores the UTC epoch onset, in microseconds, shared by every synthetic log archive built in this module."""

_SPAWN_TIMEOUT_SECONDS: int = 180
"""Stores the time the spawned worker round-trip test waits for its single submitted call before failing."""


def test_controller_extraction_job_name() -> None:
    """Verifies that the extraction job name keeps the value every persisted job identifier is derived from."""
    assert CONTROLLER_EXTRACTION_JOB_NAME == "microcontroller_data_extraction"


def test_output_layout_values() -> None:
    """Verifies that OutputLayout declares the filesystem names the extraction output layout is built from."""
    assert OutputLayout.DIRECTORY_NAME == "microcontroller_data"
    assert OutputLayout.TRACKER_FILENAME == "microcontroller_processing_tracker.yaml"
    assert OutputLayout.FILE_PREFIX == "controller_"
    assert OutputLayout.MODULE_INFIX == "_module_"
    assert OutputLayout.KERNEL_INFIX == "_kernel"
    assert OutputLayout.FILE_SUFFIX == ".feather"


def test_output_layout_members_interpolate_as_raw_strings() -> None:
    """Verifies that every OutputLayout member is a string that interpolates as its bare value."""
    for member in OutputLayout:
        assert isinstance(member, str)
        assert f"{member}" == member.value
        assert str(member) == member.value


def test_resolve_output_directory(tmp_path: Path) -> None:
    """Verifies that resolve_output_directory nests the library's own subdirectory under the nominated root."""
    resolved = resolve_output_directory(output_directory=tmp_path)

    assert resolved == tmp_path / "microcontroller_data"
    assert resolved.parent == tmp_path
    assert resolved.name == OutputLayout.DIRECTORY_NAME


def test_resolve_output_directory_creates_nothing(tmp_path: Path) -> None:
    """Verifies that resolve_output_directory only composes a path and leaves the filesystem untouched."""
    resolved = resolve_output_directory(output_directory=tmp_path / "missing")

    assert not resolved.exists()
    assert not resolved.parent.exists()


def test_resolve_tracker_path(tmp_path: Path) -> None:
    """Verifies that resolve_tracker_path names the tracker file inside the requested output directory."""
    tracker_path = resolve_tracker_path(output_directory=tmp_path)

    assert tracker_path == tmp_path / "microcontroller_processing_tracker.yaml"
    assert tracker_path.parent == tmp_path
    assert tracker_path.name == OutputLayout.TRACKER_FILENAME


def test_resolve_module_path(tmp_path: Path) -> None:
    """Verifies that resolve_module_path builds the module output path inside the requested directory."""
    path = resolve_module_path(output_directory=tmp_path, source_id="1", module_type=2, module_id=3)

    assert path == tmp_path / "controller_1_module_2_3.feather"
    assert path.parent == tmp_path
    assert path.name == "controller_1_module_2_3.feather"


def test_resolve_module_path_composition(tmp_path: Path) -> None:
    """Verifies that resolve_module_path composes the filename from the layout prefix, infix, and suffix."""
    path = resolve_module_path(output_directory=tmp_path, source_id="42", module_type=7, module_id=9)

    assert path.name == f"{OutputLayout.FILE_PREFIX}42{OutputLayout.MODULE_INFIX}7_9{OutputLayout.FILE_SUFFIX}"
    assert path.name.startswith(OutputLayout.FILE_PREFIX)
    assert path.name.endswith(OutputLayout.FILE_SUFFIX)


def test_resolve_module_path_separates_modules(tmp_path: Path) -> None:
    """Verifies that resolve_module_path gives every module of every source its own file in one directory."""
    first = resolve_module_path(output_directory=tmp_path, source_id="1", module_type=2, module_id=3)
    second = resolve_module_path(output_directory=tmp_path, source_id="1", module_type=2, module_id=4)
    third = resolve_module_path(output_directory=tmp_path, source_id="2", module_type=2, module_id=3)

    assert len({first, second, third}) == 3
    assert first.parent == second.parent == third.parent


def test_resolve_kernel_path(tmp_path: Path) -> None:
    """Verifies that resolve_kernel_path builds the kernel output path inside the requested directory."""
    path = resolve_kernel_path(output_directory=tmp_path, source_id="1")

    assert path == tmp_path / "controller_1_kernel.feather"
    assert path.parent == tmp_path
    assert path.name == "controller_1_kernel.feather"


def test_resolve_kernel_path_composition(tmp_path: Path) -> None:
    """Verifies that resolve_kernel_path composes the filename from the layout prefix, infix, and suffix."""
    path = resolve_kernel_path(output_directory=tmp_path, source_id="42")

    assert path.name == f"{OutputLayout.FILE_PREFIX}42{OutputLayout.KERNEL_INFIX}{OutputLayout.FILE_SUFFIX}"
    assert path.name.startswith(OutputLayout.FILE_PREFIX)
    assert path.name.endswith(OutputLayout.FILE_SUFFIX)


def test_resolve_kernel_path_separates_sources(tmp_path: Path) -> None:
    """Verifies that resolve_kernel_path gives every source its own kernel file in one directory."""
    first = resolve_kernel_path(output_directory=tmp_path, source_id="1")
    second = resolve_kernel_path(output_directory=tmp_path, source_id="2")

    assert first != second
    assert first.parent == second.parent


def test_kernel_and_module_paths_never_collide(tmp_path: Path) -> None:
    """Verifies that the kernel file of a source never shares a name with any of that source's module files."""
    kernel_path = resolve_kernel_path(output_directory=tmp_path, source_id="1")
    module_path = resolve_module_path(output_directory=tmp_path, source_id="1", module_type=1, module_id=1)

    assert kernel_path != module_path
    assert OutputLayout.MODULE_INFIX not in kernel_path.name
    assert OutputLayout.KERNEL_INFIX not in module_path.name


def test_output_layout_resolvers_compose(tmp_path: Path) -> None:
    """Verifies that the tracker and the output files of one batch sit inside the resolved output subdirectory."""
    resolved = resolve_output_directory(output_directory=tmp_path)
    tracker_path = resolve_tracker_path(output_directory=resolved)
    module_path = resolve_module_path(output_directory=resolved, source_id="1", module_type=2, module_id=3)
    kernel_path = resolve_kernel_path(output_directory=resolved, source_id="1")

    assert tracker_path.parent == resolved
    assert module_path.parent == resolved
    assert kernel_path.parent == resolved
    assert len({tracker_path, module_path, kernel_path}) == 3


def test_generate_job_ids() -> None:
    """Verifies that generate_job_ids maps every requested source to its tracker-derived job identifier."""
    job_ids = generate_job_ids(source_ids=["1", "2", "10"])

    assert set(job_ids) == {"1", "2", "10"}
    for source_id in ("1", "2", "10"):
        expected_id = ProcessingTracker.generate_job_id(job_name=CONTROLLER_EXTRACTION_JOB_NAME, specifier=source_id)
        assert job_ids[source_id] == expected_id


def test_generate_job_ids_is_deterministic() -> None:
    """Verifies that generate_job_ids returns the same identifiers across repeated calls."""
    assert generate_job_ids(source_ids=["1", "2"]) == generate_job_ids(source_ids=["1", "2"])


def test_generate_job_ids_distinguishes_sources() -> None:
    """Verifies that generate_job_ids assigns a distinct identifier to every distinct source."""
    job_ids = generate_job_ids(source_ids=["1", "2", "3"])

    assert len(set(job_ids.values())) == 3


def test_generate_job_ids_empty_input() -> None:
    """Verifies that generate_job_ids returns an empty mapping when no sources are requested."""
    assert generate_job_ids(source_ids=[]) == {}


def test_job_descriptor_stores_every_field(tmp_path: Path) -> None:
    """Verifies that JobDescriptor stores every supplied field verbatim."""
    job = JobDescriptor(
        log_directory=tmp_path / "logs",
        archive_path=tmp_path / "logs" / f"1{LOG_ARCHIVE_SUFFIX}",
        output_directory=tmp_path / "output",
        config_path=tmp_path / "config.yaml",
        tracker_path=tmp_path / "output" / "microcontroller_processing_tracker.yaml",
        job_name=CONTROLLER_EXTRACTION_JOB_NAME,
        job_id="abc123",
        source_id="1",
        core_weight=4,
    )

    assert job.log_directory == tmp_path / "logs"
    assert job.archive_path == tmp_path / "logs" / f"1{LOG_ARCHIVE_SUFFIX}"
    assert job.output_directory == tmp_path / "output"
    assert job.config_path == tmp_path / "config.yaml"
    assert job.tracker_path == tmp_path / "output" / "microcontroller_processing_tracker.yaml"
    assert job.job_name == CONTROLLER_EXTRACTION_JOB_NAME
    assert job.job_id == "abc123"
    assert job.source_id == "1"
    assert job.core_weight == 4


def test_job_descriptor_is_frozen(tmp_path: Path) -> None:
    """Verifies that a dispatched descriptor cannot drift after a scheduler receives it."""
    job = _build_descriptor(tmp_path=tmp_path)

    with pytest.raises(AttributeError):
        job.core_weight = 8


def test_for_archive_builds_descriptor(tmp_path: Path) -> None:
    """Verifies that for_archive fills every descriptor field from the archive a scheduler already resolved."""
    log_directory = tmp_path / "logs"
    archive_path = _create_archive(directory=log_directory, source_id=1)
    output_directory = resolve_output_directory(output_directory=tmp_path / "output")
    config_path = write_extraction_config(config_path=tmp_path / "config.yaml", source_id=1)
    tracker_path = resolve_tracker_path(output_directory=output_directory)

    job = JobDescriptor.for_archive(
        archive_path=archive_path,
        output_directory=output_directory,
        config_path=config_path,
        tracker_path=tracker_path,
        source_id="1",
        log_directory=log_directory,
        core_weight=8,
    )

    assert job.log_directory == log_directory
    assert job.archive_path == archive_path
    assert job.output_directory == output_directory
    assert job.config_path == config_path
    assert job.tracker_path == tracker_path
    assert job.job_name == CONTROLLER_EXTRACTION_JOB_NAME
    assert job.job_id == generate_job_ids(source_ids=["1"])["1"]
    assert job.source_id == "1"
    assert job.core_weight == 8


def test_for_archive_applies_defaults(tmp_path: Path) -> None:
    """Verifies that for_archive defaults the log directory to the archive's parent and the width to one core."""
    archive_path = _create_archive(directory=tmp_path / "logs", source_id=1)
    output_directory = resolve_output_directory(output_directory=tmp_path / "output")

    job = JobDescriptor.for_archive(
        archive_path=archive_path,
        output_directory=output_directory,
        config_path=write_extraction_config(config_path=tmp_path / "config.yaml", source_id=1),
        tracker_path=resolve_tracker_path(output_directory=output_directory),
        source_id="1",
    )

    assert job.log_directory == archive_path.parent
    assert job.core_weight == 1


def test_for_archive_matches_generated_job_ids(tmp_path: Path) -> None:
    """Verifies that a descriptor built for an archive addresses the same tracker entry a prepared job would."""
    output_directory = resolve_output_directory(output_directory=tmp_path / "output")
    tracker_path = resolve_tracker_path(output_directory=output_directory)
    config_path = write_extraction_config(config_path=tmp_path / "config.yaml", source_id=1)
    identifiers = generate_job_ids(source_ids=["1", "2", "10"])

    for source_id in ("1", "2", "10"):
        job = JobDescriptor.for_archive(
            archive_path=_create_archive(directory=tmp_path / "logs", source_id=int(source_id)),
            output_directory=output_directory,
            config_path=config_path,
            tracker_path=tracker_path,
            source_id=source_id,
        )
        assert job.job_id == identifiers[source_id]

    assert len({identifiers[source_id] for source_id in identifiers}) == 3


def test_dispatch_key(tmp_path: Path) -> None:
    """Verifies that dispatch_key pairs the stringified tracker path with the job identifier."""
    job = _build_descriptor(tmp_path=tmp_path)

    assert job.dispatch_key == (str(job.tracker_path), job.job_id)


def test_dispatch_key_separates_trackers(tmp_path: Path) -> None:
    """Verifies that dispatch_key distinguishes identical jobs recorded under different trackers."""
    first = _build_descriptor(tmp_path=tmp_path / "first")
    second = _build_descriptor(tmp_path=tmp_path / "second")

    assert first.job_id == second.job_id
    assert first.dispatch_key != second.dispatch_key


def test_dispatch_key_separates_sources(tmp_path: Path) -> None:
    """Verifies that dispatch_key distinguishes different sources recorded under one tracker."""
    first = _build_descriptor(tmp_path=tmp_path, source_id=1)
    second = _build_descriptor(tmp_path=tmp_path, source_id=2)

    assert first.tracker_path == second.tracker_path
    assert first.dispatch_key != second.dispatch_key


def test_dispatch_key_survives_resizing(tmp_path: Path) -> None:
    """Verifies that re-sizing a job's width leaves the key the batch addresses it by unchanged."""
    job = _build_descriptor(tmp_path=tmp_path, core_weight=1)
    resized = replace(job, core_weight=8)

    assert resized.core_weight == 8
    assert resized.dispatch_key == job.dispatch_key
    assert resized.archive_path == job.archive_path
    assert resized.config_path == job.config_path
    assert resized.source_id == job.source_id


def test_to_mapping(tmp_path: Path) -> None:
    """Verifies that to_mapping renders every descriptor field as a flat string or integer value."""
    job = _build_descriptor(tmp_path=tmp_path, core_weight=4)
    mapping = job.to_mapping()

    assert set(mapping) == {field.name for field in fields(JobDescriptor)}
    assert mapping["log_directory"] == str(job.log_directory)
    assert mapping["archive_path"] == str(job.archive_path)
    assert mapping["output_directory"] == str(job.output_directory)
    assert mapping["config_path"] == str(job.config_path)
    assert mapping["tracker_path"] == str(job.tracker_path)
    assert mapping["job_name"] == CONTROLLER_EXTRACTION_JOB_NAME
    assert mapping["job_id"] == job.job_id
    assert mapping["source_id"] == "1"
    assert mapping["core_weight"] == 4
    assert all(isinstance(value, (str, int)) for value in mapping.values())


def test_from_mapping_round_trip(tmp_path: Path) -> None:
    """Verifies that a descriptor rendered as a mapping reconstructs without loss."""
    job = _build_descriptor(tmp_path=tmp_path, core_weight=4)

    assert JobDescriptor.from_mapping(mapping=job.to_mapping()) == job


def test_from_mapping_reads_stringified_values(tmp_path: Path) -> None:
    """Verifies that from_mapping restores the declared field types from an all-string payload."""
    job = _build_descriptor(tmp_path=tmp_path, core_weight=4)
    payload = {key: str(value) for key, value in job.to_mapping().items()}

    restored = JobDescriptor.from_mapping(mapping=payload)

    assert restored == job
    assert restored.core_weight == 4
    assert isinstance(restored.core_weight, int)


def test_from_mapping_ignores_extra_keys(tmp_path: Path) -> None:
    """Verifies that from_mapping reads the fields it declares and tolerates unrelated payload keys."""
    job = _build_descriptor(tmp_path=tmp_path)
    payload = {**job.to_mapping(), "unrelated_key": "ignored"}

    assert JobDescriptor.from_mapping(mapping=payload) == job


def test_from_mapping_missing_key(tmp_path: Path) -> None:
    """Verifies that from_mapping raises the malformed descriptor error when a required key is absent."""
    job = _build_descriptor(tmp_path=tmp_path)
    payload = job.to_mapping()
    del payload["config_path"]

    field_names = ", ".join(field.name for field in fields(JobDescriptor))
    message = (
        f"Unable to read a microcontroller data extraction job descriptor from the supplied mapping. The following "
        f"required keys are absent: config_path. A descriptor mapping carries every key the descriptor writes: "
        f"{field_names}."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=payload)


def test_from_mapping_missing_key_reports_every_absent_key(tmp_path: Path) -> None:
    """Verifies that from_mapping names every absent key in sorted order."""
    job = _build_descriptor(tmp_path=tmp_path)
    payload = job.to_mapping()
    del payload["source_id"]
    del payload["job_name"]

    message = (
        "Unable to read a microcontroller data extraction job descriptor from the supplied mapping. The following "
        "required keys are absent: job_name, source_id."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=payload)


def test_from_mapping_empty_mapping() -> None:
    """Verifies that from_mapping rejects an empty payload."""
    message = (
        "Unable to read a microcontroller data extraction job descriptor from the supplied mapping. The following "
        "required keys are absent:"
    )

    with pytest.raises(ValueError, match=error_format(message)) as error:
        JobDescriptor.from_mapping(mapping={})
    for field in fields(JobDescriptor):
        assert field.name in _normalize(text=error.value)


def test_from_mapping_unreadable_value(tmp_path: Path) -> None:
    """Verifies that from_mapping raises the malformed descriptor error when a value has the wrong type."""
    job = _build_descriptor(tmp_path=tmp_path)
    payload = {**job.to_mapping(), "core_weight": "eight"}

    message = (
        "Unable to read a microcontroller data extraction job descriptor from the supplied mapping. One of its values "
        "cannot be read as the type its field declares:"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=payload)


def test_from_mapping_unreadable_path(tmp_path: Path) -> None:
    """Verifies that from_mapping rejects a payload whose path value is not a path-like object."""
    job = _build_descriptor(tmp_path=tmp_path)
    payload = {**job.to_mapping(), "config_path": None}

    message = (
        "Unable to read a microcontroller data extraction job descriptor from the supplied mapping. One of its values "
        "cannot be read as the type its field declares:"
    )

    with pytest.raises(ValueError, match=error_format(message)):
        JobDescriptor.from_mapping(mapping=payload)


def test_job_descriptor_pickle_round_trip(tmp_path: Path) -> None:
    """Verifies that a descriptor pickles and unpickles unchanged, as a pool submission requires."""
    job = _build_descriptor(tmp_path=tmp_path, core_weight=4)
    restored = pickle.loads(pickle.dumps(job))  # noqa: S301  # The payload is the descriptor built one line above.

    assert restored == job
    assert restored.dispatch_key == job.dispatch_key
    assert restored.archive_path == job.archive_path
    assert restored.config_path == job.config_path


@pytest.mark.xdist_group(name="orchestration")
def test_job_descriptor_crosses_spawned_worker(tmp_path: Path) -> None:
    """Verifies that a descriptor reaches a spawned worker intact and reconstructs from the mapping it returns."""
    job = _build_descriptor(tmp_path=tmp_path, core_weight=2)

    # Submits the descriptor's own rendering method, so the mapping under test is built inside the spawned child from
    # the copy that crossed the process boundary. The pool is kept to one worker, since a single call proves the round
    # trip.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        mapping = executor.submit(JobDescriptor.to_mapping, job).result(timeout=_SPAWN_TIMEOUT_SECONDS)

    assert mapping == job.to_mapping()
    assert JobDescriptor.from_mapping(mapping=mapping) == job


def test_job_sizing_stores_every_field() -> None:
    """Verifies that JobSizing stores the resources one sizing pass resolved."""
    sizing = JobSizing(cores=4, memory_mb=2048)

    assert sizing.cores == 4
    assert sizing.memory_mb == 2048


def test_job_sizing_is_frozen() -> None:
    """Verifies that the resolved sizing figures cannot drift after a scheduler receives them."""
    sizing = JobSizing(cores=4, memory_mb=2048)

    with pytest.raises(AttributeError):
        sizing.memory_mb = 10


def test_job_sizing_pickle_round_trip() -> None:
    """Verifies that a sizing record pickles unchanged, as a cross-process scheduler payload requires."""
    sizing = JobSizing(cores=4, memory_mb=2048)

    # The payload is the record built one line above, so the round trip deserializes nothing untrusted.
    assert pickle.loads(pickle.dumps(sizing)) == sizing  # noqa: S301


def test_find_module_paths(tmp_path: Path) -> None:
    """Verifies that find_module_paths discovers every module output file the directory holds, sorted by path."""
    second = resolve_module_path(output_directory=tmp_path, source_id="2", module_type=3, module_id=4)
    first = resolve_module_path(output_directory=tmp_path, source_id="1", module_type=1, module_id=2)
    for path in (second, first):
        path.touch()

    # The kernel file and an unrelated file share the directory, so neither may appear in the module result.
    resolve_kernel_path(output_directory=tmp_path, source_id="1").touch()
    tmp_path.joinpath("notes.txt").touch()

    assert find_module_paths(data_directory=tmp_path) == [first, second]


def test_find_module_paths_empty_directory(tmp_path: Path) -> None:
    """Verifies that find_module_paths reports no files for a directory an extraction job has not written to."""
    assert find_module_paths(data_directory=tmp_path) == []


def test_find_module_paths_missing_directory(tmp_path: Path) -> None:
    """Verifies that find_module_paths reports no files for a directory that does not exist."""
    assert find_module_paths(data_directory=tmp_path / "absent") == []


def test_find_module_paths_file_path(tmp_path: Path) -> None:
    """Verifies that find_module_paths reports no files when the nominated path is a file rather than a directory."""
    file_path = tmp_path / "data.feather"
    file_path.touch()

    assert find_module_paths(data_directory=file_path) == []


def test_parse_module_path_inverts_resolution(tmp_path: Path) -> None:
    """Verifies that parse_module_path recovers the identity resolve_module_path encoded."""
    module_path = resolve_module_path(output_directory=tmp_path, source_id="222", module_type=5, module_id=7)

    assert parse_module_path(file_path=module_path) == (222, 5, 7)


def test_parse_module_path_reads_multi_digit_codes(tmp_path: Path) -> None:
    """Verifies that parse_module_path reads every identity field whose code spans more than one digit."""
    module_path = resolve_module_path(output_directory=tmp_path, source_id="10", module_type=200, module_id=30)

    assert parse_module_path(file_path=module_path) == (10, 200, 30)


def test_find_module_paths_round_trips_through_parse_module_path(tmp_path: Path) -> None:
    """Verifies that every file find_module_paths discovers parses back into the identity that named it."""
    identities = ((1, 2, 3), (1, 4, 5), (10, 2, 3))
    for source_id, module_type, module_id in identities:
        resolve_module_path(
            output_directory=tmp_path,
            source_id=str(source_id),
            module_type=module_type,
            module_id=module_id,
        ).touch()

    # The kernel file of one of the sources shares the directory, so the discovery pass must not offer it for parsing.
    resolve_kernel_path(output_directory=tmp_path, source_id="1").touch()

    discovered = find_module_paths(data_directory=tmp_path)

    assert len(discovered) == len(identities)
    assert {parse_module_path(file_path=path) for path in discovered} == set(identities)


@pytest.mark.parametrize(
    "filename",
    [
        "controller_1_module_2.feather",
        "controller_1_module_2_3_4.feather",
        "controller_1_kernel.feather",
        "session_1_module_2_3.feather",
        "controller_1_widget_2_3.feather",
        "controller_one_module_2_3.feather",
        "controller_1_module_two_3.feather",
        "controller_1_module_2_three.feather",
    ],
)
def test_parse_module_path_rejects_foreign_names(tmp_path: Path, filename: str) -> None:
    """Verifies that parse_module_path rejects a filename outside the module output naming convention."""
    message = (
        f"Unable to parse the module output filename '{filename}'. The filename does not follow the "
        f"'controller_{{source_id}}_module_{{module_type}}_{{module_id}}.feather' naming convention."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        parse_module_path(file_path=tmp_path / filename)


def test_find_kernel_paths(tmp_path: Path) -> None:
    """Verifies that find_kernel_paths discovers every kernel output file the directory holds, sorted by path."""
    second = resolve_kernel_path(output_directory=tmp_path, source_id="2")
    first = resolve_kernel_path(output_directory=tmp_path, source_id="1")
    for path in (second, first):
        path.touch()

    # The module file and an unrelated file share the directory, so neither may appear in the kernel result.
    resolve_module_path(output_directory=tmp_path, source_id="1", module_type=3, module_id=4).touch()
    tmp_path.joinpath("notes.txt").touch()

    assert find_kernel_paths(data_directory=tmp_path) == [first, second]


def test_find_kernel_paths_empty_directory(tmp_path: Path) -> None:
    """Verifies that find_kernel_paths reports no files for a directory an extraction job has not written to."""
    assert find_kernel_paths(data_directory=tmp_path) == []


def test_find_kernel_paths_missing_directory(tmp_path: Path) -> None:
    """Verifies that find_kernel_paths reports no files for a directory that does not exist."""
    assert find_kernel_paths(data_directory=tmp_path / "absent") == []


def test_parse_kernel_path_inverts_resolution(tmp_path: Path) -> None:
    """Verifies that parse_kernel_path recovers the source identifier resolve_kernel_path encoded."""
    kernel_path = resolve_kernel_path(output_directory=tmp_path, source_id="222")

    assert parse_kernel_path(file_path=kernel_path) == 222


def test_find_kernel_paths_round_trips_through_parse_kernel_path(tmp_path: Path) -> None:
    """Verifies that every file find_kernel_paths discovers parses back into the source that named it."""
    for source_id in (1, 10, 222):
        resolve_kernel_path(output_directory=tmp_path, source_id=str(source_id)).touch()

    discovered = find_kernel_paths(data_directory=tmp_path)

    assert sorted(parse_kernel_path(file_path=path) for path in discovered) == [1, 10, 222]


@pytest.mark.parametrize(
    "filename",
    [
        "controller_1_module_2_3.feather",
        "controller_kernel.feather",
        "controller_1_kernel_2.feather",
        "controller_abc_kernel.feather",
        "session_1_kernel.feather",
        "controller_1_widget.feather",
    ],
)
def test_parse_kernel_path_rejects_foreign_names(tmp_path: Path, filename: str) -> None:
    """Verifies that parse_kernel_path rejects a filename outside the kernel output naming convention."""
    message = (
        f"Unable to parse the kernel output filename '{filename}'. The filename does not follow the "
        f"'controller_{{source_id}}_kernel.feather' naming convention."
    )

    with pytest.raises(ValueError, match=error_format(message)):
        parse_kernel_path(file_path=tmp_path / filename)


def _create_archive(directory: Path, source_id: int) -> Path:
    """Writes a synthetic microcontroller log archive for the target source into the requested directory."""
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    create_test_archive(
        archive_path=archive_path,
        source_id=source_id,
        messages=[
            (1000, create_module_state_payload(module_type=1, module_id=2, command=1, event=10)),
            (2000, create_module_state_payload(module_type=1, module_id=2, command=1, event=10)),
        ],
        onset_us=_ONSET_US,
    )
    return archive_path


def _build_descriptor(tmp_path: Path, source_id: int = 1, core_weight: int = 1) -> JobDescriptor:
    """Builds a descriptor addressing a real synthetic archive written under the requested temporary directory."""
    log_directory = tmp_path / "logs"
    archive_path = _create_archive(directory=log_directory, source_id=source_id)
    output_directory = resolve_output_directory(output_directory=tmp_path / "output")
    config_path = write_extraction_config(config_path=tmp_path / "config.yaml", source_id=source_id)

    return JobDescriptor.for_archive(
        archive_path=archive_path,
        output_directory=output_directory,
        config_path=config_path,
        tracker_path=resolve_tracker_path(output_directory=output_directory),
        source_id=str(source_id),
        log_directory=log_directory,
        core_weight=core_weight,
    )


def _normalize(text: object) -> str:
    """Collapses the line wrapping the console applies, so a message fragment matches the raised error's text."""
    return " ".join(str(text).split())
