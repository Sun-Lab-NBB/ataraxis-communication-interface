"""Contains tests for the classes and functions provided by the orchestration allocation.py module."""

import os
from pathlib import Path

import pytest
from tests.log_archives import (
    create_test_archive,
    make_kernel_data_payload,
    make_module_data_payload,
    make_kernel_state_payload,
    make_module_state_payload,
)
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, PARALLEL_PROCESSING_THRESHOLD

from ataraxis_communication_interface.communication import SerialPrototypes
from ataraxis_communication_interface.orchestration.allocation import (
    _RESERVED_CORES,
    _WORKER_MEMORY_MB,
    EXTRACTION_JOB_CORES,
    _SUBPROCESS_MEMORY_MB,
    _MEGABYTES_PER_GIGABYTE,
    _ARCHIVE_DIRECTORY_RATIO,
    _MINIMUM_MEMORY_BUDGET_MB,
    ArchiveFootprint,
    _apply_tolerance,
    _bytes_to_megabytes,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    _resolve_host_memory_mb,
    resolve_memory_budget_mb,
    resolve_archive_footprint,
)

_UNMODELED_FOOTPRINT: ArchiveFootprint = ArchiveFootprint(message_count=0, archive_bytes=0, modeled=False)
"""Stores the footprint resolve_archive_footprint returns for an archive it cannot read."""


def _modeled_footprint(message_count: int, archive_bytes: int) -> ArchiveFootprint:
    """Builds a modeled footprint carrying the requested message count and on-disk size."""
    return ArchiveFootprint(message_count=message_count, archive_bytes=archive_bytes, modeled=True)


def _expected_estimate(footprint: ArchiveFootprint, cores: int) -> int:
    """Recomputes the modeled memory estimate for the target footprint and core count."""
    per_reader = _bytes_to_megabytes(byte_count=footprint.archive_bytes * _ARCHIVE_DIRECTORY_RATIO)
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + cores * (per_reader + _SUBPROCESS_MEMORY_MB))


def test_archive_footprint_fields() -> None:
    """Verifies that ArchiveFootprint stores the message count, the archive size, and the modeled flag."""
    footprint = ArchiveFootprint(message_count=1500, archive_bytes=4096, modeled=True)

    assert footprint.message_count == 1500
    assert footprint.archive_bytes == 4096
    assert footprint.modeled

    # The dataclass is frozen, so the resolved figures cannot drift after a consumer receives them.
    with pytest.raises(AttributeError):
        footprint.message_count = 10  # type: ignore[misc]


def test_resolve_archive_footprint_models_real_archive(tmp_path: Path) -> None:
    """Verifies that resolve_archive_footprint models a readable log archive from its directory and file size."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    create_test_archive(
        archive_path=archive_path,
        source_id=1,
        messages=[
            (1000, make_module_state_payload(module_type=1, module_id=2, command=1, event=10)),
            (
                2000,
                make_module_data_payload(
                    module_type=1,
                    module_id=2,
                    command=2,
                    event=20,
                    prototype_code=SerialPrototypes.ONE_UINT8,
                    data_bytes=[42],
                ),
            ),
            (3000, make_kernel_state_payload(command=3, event=30)),
            (
                4000,
                make_kernel_data_payload(
                    command=4, event=40, prototype_code=SerialPrototypes.ONE_UINT8, data_bytes=[7]
                ),
            ),
        ],
    )

    footprint = resolve_archive_footprint(archive_path=archive_path)

    assert footprint.modeled
    # The onset message is excluded from the count, leaving the two module messages and the two kernel messages.
    assert footprint.message_count == 4
    assert footprint.archive_bytes == archive_path.stat().st_size
    assert footprint.archive_bytes > 0


def test_resolve_archive_footprint_falls_back_for_missing_archive(tmp_path: Path) -> None:
    """Verifies that resolve_archive_footprint returns an unmodeled footprint for an archive that does not exist."""
    footprint = resolve_archive_footprint(archive_path=tmp_path / f"1{LOG_ARCHIVE_SUFFIX}")

    assert footprint == _UNMODELED_FOOTPRINT
    assert not footprint.modeled
    assert footprint.message_count == 0
    assert footprint.archive_bytes == 0


def test_resolve_archive_footprint_falls_back_for_corrupt_archive(tmp_path: Path) -> None:
    """Verifies that resolve_archive_footprint returns an unmodeled footprint for an archive it cannot decode."""
    archive_path = tmp_path / f"2{LOG_ARCHIVE_SUFFIX}"
    archive_path.write_text("This is not a valid numpy archive.")

    footprint = resolve_archive_footprint(archive_path=archive_path)

    assert footprint == _UNMODELED_FOOTPRINT


@pytest.mark.parametrize("message_count", [0, 1, PARALLEL_PROCESSING_THRESHOLD - 1])
def test_resolve_job_workers_below_threshold(message_count: int) -> None:
    """Verifies that resolve_job_workers gives a single core to an archive below the parallel processing threshold."""
    footprint = _modeled_footprint(message_count=message_count, archive_bytes=1024)

    assert resolve_job_workers(footprint=footprint, ceiling=EXTRACTION_JOB_CORES * 4) == 1


def test_resolve_job_workers_caps_at_repaid_workers() -> None:
    """Verifies that resolve_job_workers holds a mid-size archive to the workers its message count repays."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 3, archive_bytes=64 * 1024 * 1024)

    workers = resolve_job_workers(footprint=footprint, ceiling=EXTRACTION_JOB_CORES * 4)

    # Three thresholds' worth of messages repay three workers, which is below the declared width of the stage.
    assert workers == footprint.message_count // PARALLEL_PROCESSING_THRESHOLD
    assert workers == 3
    assert workers < EXTRACTION_JOB_CORES


def test_resolve_job_workers_caps_at_declared_width() -> None:
    """Verifies that resolve_job_workers holds a large archive to the declared core allocation of the stage."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 100, archive_bytes=512 * 1024 * 1024)

    workers = resolve_job_workers(footprint=footprint, ceiling=EXTRACTION_JOB_CORES * 4)

    assert workers == EXTRACTION_JOB_CORES


def test_resolve_job_workers_caps_at_ceiling() -> None:
    """Verifies that resolve_job_workers never returns more cores than the ceiling that sized the job."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 100, archive_bytes=512 * 1024 * 1024)

    for ceiling in (1, 2, 3, EXTRACTION_JOB_CORES - 1):
        workers = resolve_job_workers(footprint=footprint, ceiling=ceiling)

        # A job resolved above its ceiling is never dispatchable within the budget that sized it, which stalls the
        # dispatcher, so the ceiling binds ahead of both the declared width and the repaid workers.
        assert workers <= ceiling
        assert workers == ceiling
        assert workers < EXTRACTION_JOB_CORES


def test_resolve_job_workers_floors_at_one() -> None:
    """Verifies that resolve_job_workers returns at least one core when the ceiling is non-positive."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 100, archive_bytes=512 * 1024 * 1024)

    assert resolve_job_workers(footprint=footprint, ceiling=0) == 1
    assert resolve_job_workers(footprint=footprint, ceiling=-5) == 1


def test_estimate_job_memory_mb_unmodeled_footprint() -> None:
    """Verifies that estimate_job_memory_mb falls back to the worker baseline floor for an unmodeled footprint."""
    baseline = _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)

    # The baseline carries the tolerance and rounds up to exactly one gigabyte, which is the floor every consumer
    # plans an unmodeled job around.
    assert baseline == _MEGABYTES_PER_GIGABYTE

    # The core count does not enter the estimate, since an unmodeled footprint carries no archive to split.
    for cores in (1, 4, EXTRACTION_JOB_CORES):
        estimate = estimate_job_memory_mb(footprint=_UNMODELED_FOOTPRINT, cores=cores)

        assert estimate == baseline
        assert estimate % _MEGABYTES_PER_GIGABYTE == 0
        assert estimate > 0


def test_estimate_job_memory_mb_composes_model_terms() -> None:
    """Verifies that estimate_job_memory_mb charges the job terms once and the reader allowance per core."""
    footprint = _modeled_footprint(message_count=10_000, archive_bytes=64 * 1024 * 1024)
    cores = 4

    estimate = estimate_job_memory_mb(footprint=footprint, cores=cores)

    assert estimate == _expected_estimate(footprint=footprint, cores=cores)
    assert estimate % _MEGABYTES_PER_GIGABYTE == 0
    assert estimate > _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)

    # A 64 MB archive holds a 257 MB directory per reader, so one core carries 384 + 457 MB and four carry
    # 384 + 1828 MB, which the tolerance rounds up to one and three gigabytes.
    assert estimate_job_memory_mb(footprint=footprint, cores=1) == _MEGABYTES_PER_GIGABYTE
    assert estimate == 3 * _MEGABYTES_PER_GIGABYTE


def test_estimate_job_memory_mb_scales_with_cores() -> None:
    """Verifies that estimate_job_memory_mb grows with the cores a job holds, as every core opens its own reader."""
    footprint = _modeled_footprint(message_count=1_000_000, archive_bytes=512 * 1024 * 1024)

    estimates = [estimate_job_memory_mb(footprint=footprint, cores=cores) for cores in (1, 2, 4, EXTRACTION_JOB_CORES)]

    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]
    assert all(estimate % _MEGABYTES_PER_GIGABYTE == 0 for estimate in estimates)


def test_estimate_job_memory_mb_scales_with_archive_bytes() -> None:
    """Verifies that estimate_job_memory_mb grows with the size of the archive the job reads."""
    estimates = [
        estimate_job_memory_mb(footprint=_modeled_footprint(message_count=1_000_000, archive_bytes=size), cores=4)
        for size in (1024, 64 * 1024 * 1024, 512 * 1024 * 1024)
    ]

    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]
    assert all(estimate % _MEGABYTES_PER_GIGABYTE == 0 for estimate in estimates)


def test_resolve_host_memory_mb() -> None:
    """Verifies that _resolve_host_memory_mb reports a positive physical memory figure for the host."""
    host_memory_mb = _resolve_host_memory_mb()

    assert isinstance(host_memory_mb, int)
    assert host_memory_mb > 0


def test_resolve_core_budget_honors_positive_request() -> None:
    """Verifies that resolve_core_budget honors a positive core request up to the logical core count of the host."""
    available_cores = os.cpu_count() or 1

    assert resolve_core_budget(requested_budget=1) == 1
    assert resolve_core_budget(requested_budget=available_cores) == available_cores

    # A request above the core count is capped rather than honored, since the reserved cores do not apply to it.
    assert resolve_core_budget(requested_budget=available_cores * 100) == available_cores


@pytest.mark.parametrize("requested_budget", [0, -1, -100])
def test_resolve_core_budget_auto_resolves(requested_budget: int) -> None:
    """Verifies that resolve_core_budget auto-resolves a non-positive request to at least one core."""
    available_cores = os.cpu_count() or 1

    budget = resolve_core_budget(requested_budget=requested_budget)

    assert budget >= 1
    assert budget <= available_cores
    assert budget == max(1, available_cores - _RESERVED_CORES)


@pytest.mark.parametrize("requested_budget_mb", [1, 1024, 4096, 1_000_000])
def test_resolve_memory_budget_mb_honors_positive_request(requested_budget_mb: int) -> None:
    """Verifies that resolve_memory_budget_mb returns a positive memory request verbatim."""
    assert resolve_memory_budget_mb(requested_budget_mb=requested_budget_mb) == requested_budget_mb


@pytest.mark.parametrize("requested_budget_mb", [0, -1, -4096])
def test_resolve_memory_budget_mb_auto_resolves(requested_budget_mb: int) -> None:
    """Verifies that resolve_memory_budget_mb auto-resolves a non-positive request to at least the floor."""
    budget_mb = resolve_memory_budget_mb(requested_budget_mb=requested_budget_mb)

    # The auto-resolved budget claims a share of the host memory, so it never exceeds what the host physically holds
    # and never drops below the floor that keeps a small host admitting one job at a time.
    assert budget_mb >= _MINIMUM_MEMORY_BUDGET_MB
    assert budget_mb <= max(_MINIMUM_MEMORY_BUDGET_MB, _resolve_host_memory_mb())


@pytest.mark.parametrize(
    "byte_count, expected",
    [
        (1, 1),
        (1024 * 1024, 2),
        (1024 * 1024 + 1, 2),
        (1.5 * 1024 * 1024, 2),
        (64 * 1024 * 1024, 65),
    ],
)
def test_bytes_to_megabytes_rounds_up(byte_count: float, expected: int) -> None:
    """Verifies that _bytes_to_megabytes converts a positive byte count into whole megabytes with one of headroom."""
    assert _bytes_to_megabytes(byte_count=byte_count) == expected


@pytest.mark.parametrize("byte_count", [0, 0.0, -1, -1024 * 1024, -0.5])
def test_bytes_to_megabytes_non_positive_byte_count(byte_count: float) -> None:
    """Verifies that _bytes_to_megabytes converts a zero or negative byte count to zero megabytes."""
    assert _bytes_to_megabytes(byte_count=byte_count) == 0


@pytest.mark.parametrize(
    "memory_mb, expected",
    [
        (0, 1024),
        (1, 1024),
        (_WORKER_MEMORY_MB, 1024),
        (1024, 2048),
        (2048, 3072),
    ],
)
def test_apply_tolerance(memory_mb: int, expected: int) -> None:
    """Verifies that _apply_tolerance carries the estimate margin and rounds the figure up to a whole gigabyte."""
    reportable = _apply_tolerance(memory_mb=memory_mb)

    assert reportable == expected
    assert reportable % _MEGABYTES_PER_GIGABYTE == 0
    assert reportable >= memory_mb
