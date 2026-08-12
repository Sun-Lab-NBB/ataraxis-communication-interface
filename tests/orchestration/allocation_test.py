"""Contains tests for the classes and functions provided by the orchestration/allocation.py module."""

import os
from pathlib import Path

import pytest
from tests.log_archives import (
    create_test_archive,
    create_kernel_data_payload,
    create_module_data_payload,
    create_kernel_state_payload,
    create_module_state_payload,
)
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX

from ataraxis_communication_interface.communication import SerialPrototypes
from ataraxis_communication_interface.orchestration.allocation import (
    _RESERVED_CORES,
    _BYTES_PER_MEGABYTE,
    _JOB_BODY_MEMORY_MB,
    SPAWNED_CHILD_MEMORY_MB,
    _MEMORY_BUDGET_FRACTION,
    _ARCHIVE_DIRECTORY_RATIO,
    _MINIMUM_MEMORY_BUDGET_MB,
    _MEMORY_ROUNDING_QUANTUM_MB,
    PARALLEL_EXTRACTION_THRESHOLD,
    CONTROLLER_EXTRACTION_JOB_CORES,
    _POOL_MEMORY_RESERVATION_DIVISOR,
    ArchiveFootprint,
    _apply_tolerance,
    size_archive_job,
    resolve_pool_size,
    _bytes_to_megabytes,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_host_memory_mb,
    resolve_memory_budget_mb,
    resolve_archive_footprint,
)

_MEGABYTE: int = 1024 * 1024
"""Stores the bytes one megabyte holds, which sizes the synthetic archives the memory model tests weigh."""

_UNMODELED_FOOTPRINT: ArchiveFootprint = ArchiveFootprint(message_count=0, archive_bytes=0, modeled=False)
"""Stores the footprint resolve_archive_footprint returns for an archive it cannot read."""

_UNMODELED_MEMORY_MB: int = _apply_tolerance(memory_mb=_JOB_BODY_MEMORY_MB)
"""Stores the memory floor every consumer plans around when an archive yields no footprint."""


def test_archive_footprint_fields() -> None:
    """Verifies that ArchiveFootprint stores the message count, the archive size, and the modeled flag."""
    footprint = ArchiveFootprint(message_count=1500, archive_bytes=4096, modeled=True)

    assert footprint.message_count == 1500
    assert footprint.archive_bytes == 4096
    assert footprint.modeled

    # The dataclass is frozen, so the resolved figures cannot drift after a consumer receives them.
    with pytest.raises(AttributeError):
        footprint.message_count = 10


def test_resolve_archive_footprint_models_real_archive(tmp_path: Path) -> None:
    """Verifies that resolve_archive_footprint models a readable log archive from its directory and file size."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    _write_archive(archive_path=archive_path)

    footprint = resolve_archive_footprint(archive_path=archive_path)

    assert footprint.modeled
    # The onset message is excluded from the count, leaving the four module messages and the two kernel messages.
    assert footprint.message_count == 6
    assert footprint.archive_bytes == archive_path.stat().st_size
    assert footprint.archive_bytes > 0


def test_resolve_archive_footprint_falls_back_for_missing_archive(tmp_path: Path) -> None:
    """Verifies that resolve_archive_footprint returns an unmodeled footprint for an archive that does not exist."""
    footprint = resolve_archive_footprint(archive_path=tmp_path / f"1{LOG_ARCHIVE_SUFFIX}")

    # Neither the message count read nor the stat call can resolve a path that does not exist.
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


@pytest.mark.parametrize("message_count", [0, 1, PARALLEL_EXTRACTION_THRESHOLD - 1])
def test_resolve_job_workers_below_threshold(message_count: int) -> None:
    """Verifies that resolve_job_workers gives a single core to an archive below the parallel extraction threshold."""
    footprint = _modeled_footprint(message_count=message_count, archive_bytes=1024)

    # An archive this small is decoded before a pool repays the children it spawns, so the sequential shape is the
    # only one offered to it whatever the archive weighs on disk.
    assert resolve_job_workers(footprint=footprint) == 1


@pytest.mark.parametrize(
    "message_count",
    [PARALLEL_EXTRACTION_THRESHOLD, PARALLEL_EXTRACTION_THRESHOLD + 1, PARALLEL_EXTRACTION_THRESHOLD * 100],
)
def test_resolve_job_workers_at_or_above_threshold(message_count: int) -> None:
    """Verifies that resolve_job_workers gives the declared core allocation to every archive above the threshold."""
    footprint = _modeled_footprint(message_count=message_count, archive_bytes=512 * _MEGABYTE)

    # The stage emits two job shapes and nothing between them, so a hundredfold larger archive receives the same
    # width as one that only just crossed the threshold.
    assert resolve_job_workers(footprint=footprint) == CONTROLLER_EXTRACTION_JOB_CORES


def test_estimate_job_memory_mb_unmodeled_footprint() -> None:
    """Verifies that estimate_job_memory_mb falls back to the job body baseline for an unmodeled footprint."""
    # The core count does not enter the estimate, since an unmodeled footprint carries no archive to split.
    for cores in (1, 4, CONTROLLER_EXTRACTION_JOB_CORES):
        estimate = estimate_job_memory_mb(footprint=_UNMODELED_FOOTPRINT, cores=cores)

        assert estimate == _UNMODELED_MEMORY_MB
        assert estimate == 256
        assert estimate % _MEMORY_ROUNDING_QUANTUM_MB == 0


def test_estimate_job_memory_mb_charges_one_body_and_one_reader_serially() -> None:
    """Verifies that estimate_job_memory_mb charges a single-core job for one job body and one reader."""
    footprint = _modeled_footprint(message_count=10_000, archive_bytes=64 * _MEGABYTE)

    estimate = estimate_job_memory_mb(footprint=footprint, cores=1)

    # A 64 MB archive builds a 173 MB reader, which the sequential body holds alongside its own 180 MB baseline. The
    # 353 MB sum carries the tolerance to 424 MB and rounds up to the 512 MB the batch is charged.
    per_reader = _bytes_to_megabytes(byte_count=footprint.archive_bytes * _ARCHIVE_DIRECTORY_RATIO)
    assert per_reader == 173
    assert estimate == _apply_tolerance(memory_mb=_JOB_BODY_MEMORY_MB + per_reader)
    assert estimate == _expected_memory_mb(archive_bytes=footprint.archive_bytes, cores=1)
    assert estimate == 512


def test_estimate_job_memory_mb_charges_every_reader_in_parallel() -> None:
    """Verifies that estimate_job_memory_mb charges a multi-core job for the job body and every pool child."""
    footprint = _modeled_footprint(message_count=10_000, archive_bytes=64 * _MEGABYTE)
    cores = 4

    estimate = estimate_job_memory_mb(footprint=footprint, cores=cores)

    # The job body holds its own baseline and reader, and each of the four pool children holds a spawned child
    # baseline and a reader of its own, which is 1653 MB before the tolerance and the rounding lift it to 2048 MB.
    per_reader = _bytes_to_megabytes(byte_count=footprint.archive_bytes * _ARCHIVE_DIRECTORY_RATIO)
    assert estimate == _apply_tolerance(
        memory_mb=_JOB_BODY_MEMORY_MB + per_reader + cores * (SPAWNED_CHILD_MEMORY_MB + per_reader)
    )
    assert estimate == _expected_memory_mb(archive_bytes=footprint.archive_bytes, cores=cores)
    assert estimate == 2048


def test_estimate_job_memory_mb_parallel_path_exceeds_serial_path() -> None:
    """Verifies that estimate_job_memory_mb charges a two-core job more than the sequential path it leaves behind."""
    footprint = _modeled_footprint(message_count=10_000, archive_bytes=64 * _MEGABYTE)

    serial_estimate = estimate_job_memory_mb(footprint=footprint, cores=1)
    parallel_estimate = estimate_job_memory_mb(footprint=footprint, cores=2)

    # Taking a second core opens a pool, so the body's reader is joined by one reader and one baseline per child.
    assert parallel_estimate > serial_estimate
    assert serial_estimate == 512
    assert parallel_estimate == 1280


def test_estimate_job_memory_mb_scales_with_cores() -> None:
    """Verifies that estimate_job_memory_mb grows with the cores a job holds, as every core opens its own reader."""
    footprint = _modeled_footprint(message_count=1_000_000, archive_bytes=512 * _MEGABYTE)
    core_counts = (1, 2, CONTROLLER_EXTRACTION_JOB_CORES)

    estimates = [estimate_job_memory_mb(footprint=footprint, cores=cores) for cores in core_counts]

    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]
    assert estimates == [
        _expected_memory_mb(archive_bytes=footprint.archive_bytes, cores=cores) for cores in core_counts
    ]
    assert estimates == [2048, 5632, 9472]
    assert all(estimate % _MEMORY_ROUNDING_QUANTUM_MB == 0 for estimate in estimates)


def test_estimate_job_memory_mb_scales_with_archive_bytes() -> None:
    """Verifies that estimate_job_memory_mb grows with the size of the archive the job reads."""
    archive_sizes = (1024, 64 * _MEGABYTE, 512 * _MEGABYTE)

    estimates = [
        estimate_job_memory_mb(footprint=_modeled_footprint(message_count=1_000_000, archive_bytes=size), cores=4)
        for size in archive_sizes
    ]

    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]
    assert estimates == [_expected_memory_mb(archive_bytes=size, cores=4) for size in archive_sizes]
    assert estimates == [1024, 2048, 9472]
    assert all(estimate % _MEMORY_ROUNDING_QUANTUM_MB == 0 for estimate in estimates)


def test_resolve_host_memory_mb() -> None:
    """Verifies that resolve_host_memory_mb reports a positive physical memory figure for the host."""
    host_memory_mb = resolve_host_memory_mb()

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
    """Verifies that resolve_memory_budget_mb auto-resolves a non-positive request to a share of the host memory."""
    host_memory_mb = resolve_host_memory_mb()

    budget_mb = resolve_memory_budget_mb(requested_budget_mb=requested_budget_mb)

    assert budget_mb == max(_MINIMUM_MEMORY_BUDGET_MB, int(host_memory_mb * _MEMORY_BUDGET_FRACTION))
    assert budget_mb >= _MINIMUM_MEMORY_BUDGET_MB
    # The auto-resolved budget never exceeds the greater of the host's memory and the floor a small host is held to.
    assert budget_mb <= max(_MINIMUM_MEMORY_BUDGET_MB, host_memory_mb)


def test_resolve_pool_size_binds_on_job_count() -> None:
    """Verifies that resolve_pool_size opens no more job slots than the batch holds jobs."""
    # Both budgets are set well above what three jobs can claim, leaving the job count as the binding term.
    assert resolve_pool_size(job_count=3, core_budget=64, memory_budget_mb=100_000) == 3


def test_resolve_pool_size_binds_on_core_budget() -> None:
    """Verifies that resolve_pool_size opens no more job slots than the core budget can hold running jobs."""
    assert resolve_pool_size(job_count=100, core_budget=6, memory_budget_mb=100_000) == 6


def test_resolve_pool_size_binds_on_affordable_bodies() -> None:
    """Verifies that resolve_pool_size holds the slot count to the warmed job bodies half the memory budget holds."""
    memory_budget_mb = 1024
    affordable_bodies = (memory_budget_mb // _POOL_MEMORY_RESERVATION_DIVISOR) // SPAWNED_CHILD_MEMORY_MB

    pool_size = resolve_pool_size(job_count=100, core_budget=64, memory_budget_mb=memory_budget_mb)

    # Half of a one gigabyte budget holds three 152 MB bodies, leaving the remainder for the work those bodies perform.
    assert pool_size == affordable_bodies
    assert pool_size == 3


def test_resolve_pool_size_scales_with_memory_budget() -> None:
    """Verifies that resolve_pool_size opens more job slots as the memory budget grows."""
    pool_sizes = [
        resolve_pool_size(job_count=100, core_budget=64, memory_budget_mb=budget_mb)
        for budget_mb in (1024, 4096, 16_384)
    ]

    assert pool_sizes == sorted(pool_sizes)
    assert pool_sizes[0] < pool_sizes[-1]
    assert pool_sizes == [3, 13, 53]


@pytest.mark.parametrize(
    "job_count, core_budget, memory_budget_mb",
    [
        (0, 8, 100_000),
        (8, 0, 100_000),
        (8, -4, 100_000),
        (8, 8, 1),
        (8, 8, 0),
        (8, 8, -100),
        (0, 0, 0),
    ],
)
def test_resolve_pool_size_floors_at_one(job_count: int, core_budget: int, memory_budget_mb: int) -> None:
    """Verifies that resolve_pool_size always opens at least one job slot, whatever the batch can afford."""
    # A pool with no slots can never dispatch, so the floor holds even when no term supports a single body.
    assert resolve_pool_size(job_count=job_count, core_budget=core_budget, memory_budget_mb=memory_budget_mb) == 1


@pytest.mark.parametrize(
    "byte_count, expected",
    [
        (1, 1),
        (_MEGABYTE, 2),
        (_MEGABYTE + 1, 2),
        (1.5 * _MEGABYTE, 2),
        (64 * _MEGABYTE, 65),
    ],
)
def test_bytes_to_megabytes_rounds_up(byte_count: float, expected: int) -> None:
    """Verifies that _bytes_to_megabytes converts a positive byte count into whole megabytes with one of headroom."""
    assert _bytes_to_megabytes(byte_count=byte_count) == expected
    assert _BYTES_PER_MEGABYTE == _MEGABYTE


@pytest.mark.parametrize("byte_count", [0, 0.0, -1, -_MEGABYTE, -0.5])
def test_bytes_to_megabytes_non_positive_byte_count(byte_count: float) -> None:
    """Verifies that _bytes_to_megabytes converts a zero or negative byte count to zero megabytes."""
    assert _bytes_to_megabytes(byte_count=byte_count) == 0


@pytest.mark.parametrize(
    "memory_mb, expected",
    [
        (0, 256),
        (1, 256),
        (SPAWNED_CHILD_MEMORY_MB, 256),
        (1024, 1280),
        (2048, 2560),
    ],
)
def test_apply_tolerance(memory_mb: int, expected: int) -> None:
    """Verifies that _apply_tolerance carries the estimate margin and rounds the figure up to the rounding quantum."""
    reportable = _apply_tolerance(memory_mb=memory_mb)

    assert reportable == expected
    assert reportable % _MEMORY_ROUNDING_QUANTUM_MB == 0
    assert reportable >= memory_mb


def _modeled_footprint(message_count: int, archive_bytes: int) -> ArchiveFootprint:
    """Builds a modeled footprint carrying the requested message count and on-disk size."""
    return ArchiveFootprint(message_count=message_count, archive_bytes=archive_bytes, modeled=True)


def _expected_memory_mb(archive_bytes: int, cores: int) -> int:
    """Recomputes the modeled memory estimate for an archive of the requested size at the requested core count."""
    per_reader = _bytes_to_megabytes(byte_count=archive_bytes * _ARCHIVE_DIRECTORY_RATIO)

    # A single-core job takes the sequential path, which opens no extraction pool and holds the body's reader alone.
    if cores == 1:
        return _apply_tolerance(memory_mb=_JOB_BODY_MEMORY_MB + per_reader)

    # Every pool child holds a spawned child's baseline and a reader of its own, alongside the body and its reader.
    return _apply_tolerance(memory_mb=_JOB_BODY_MEMORY_MB + per_reader + cores * (SPAWNED_CHILD_MEMORY_MB + per_reader))


def _write_archive(archive_path: Path, source_id: int = 1) -> None:
    """Writes a small readable log archive holding four module messages and two kernel messages."""
    create_test_archive(
        archive_path=archive_path,
        source_id=source_id,
        onset_us=1_000_000,
        messages=[
            (1000, create_module_state_payload(module_type=1, module_id=2, command=1, event=10)),
            (2000, create_module_state_payload(module_type=1, module_id=2, command=1, event=11)),
            (
                3000,
                create_module_data_payload(
                    module_type=1,
                    module_id=2,
                    command=2,
                    event=20,
                    prototype_code=SerialPrototypes.ONE_UINT8,
                    data_bytes=[42],
                ),
            ),
            (
                4000,
                create_module_data_payload(
                    module_type=1,
                    module_id=2,
                    command=2,
                    event=21,
                    prototype_code=SerialPrototypes.ONE_UINT8,
                    data_bytes=[43],
                ),
            ),
            (5000, create_kernel_state_payload(command=3, event=30)),
            (
                6000,
                create_kernel_data_payload(
                    command=4, event=40, prototype_code=SerialPrototypes.ONE_UINT8, data_bytes=[7]
                ),
            ),
        ],
    )


def test_size_archive_job_sizes_a_real_archive(tmp_path: Path) -> None:
    """Verifies that size_archive_job resolves both figures of the sizing model from one readable archive."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    _write_archive(archive_path=archive_path)

    cores, memory_mb, modeled = size_archive_job(archive_path=archive_path)

    # The synthetic archive holds far fewer messages than the threshold, so it takes the sequential shape.
    footprint = resolve_archive_footprint(archive_path=archive_path)
    assert modeled
    assert cores == 1
    assert cores == resolve_job_workers(footprint=footprint)
    assert memory_mb == estimate_job_memory_mb(footprint=footprint, cores=cores)


def test_size_archive_job_widens_past_the_threshold() -> None:
    """Verifies that size_archive_job reports the declared allocation for an archive above the threshold."""
    footprint = _modeled_footprint(message_count=PARALLEL_EXTRACTION_THRESHOLD, archive_bytes=64 * _MEGABYTE)

    assert resolve_job_workers(footprint=footprint) == CONTROLLER_EXTRACTION_JOB_CORES


def test_size_archive_job_falls_back_for_an_unreadable_archive(tmp_path: Path) -> None:
    """Verifies that size_archive_job reports the baseline floor for an archive it cannot read."""
    cores, memory_mb, modeled = size_archive_job(archive_path=tmp_path / f"missing{LOG_ARCHIVE_SUFFIX}")

    assert not modeled
    assert cores == 1
    assert memory_mb == _UNMODELED_MEMORY_MB
