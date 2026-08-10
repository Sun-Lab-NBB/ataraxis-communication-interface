"""Contains tests for the classes and functions provided by the orchestration/allocation.py module."""

import os

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
    _BYTES_PER_MEGABYTE,
    SPAWNED_CHILD_MEMORY_MB,
    _MEGABYTES_PER_GIGABYTE,
    _MEMORY_BUDGET_FRACTION,
    _ARCHIVE_DIRECTORY_RATIO,
    _MINIMUM_MEMORY_BUDGET_MB,
    CONTROLLER_EXTRACTION_JOB_CORES,
    _POOL_MEMORY_RESERVATION_DIVISOR,
    ArchiveFootprint,
    _apply_tolerance,
    resolve_pool_size,
    _bytes_to_megabytes,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_host_memory_mb,
    resolve_memory_budget_mb,
    resolve_archive_footprint,
    estimate_archive_job_memory_mb,
)

_MEGABYTE: int = 1024 * 1024
"""Stores the bytes one megabyte holds, which sizes the synthetic archives the memory model tests weigh."""

_UNMODELED_FOOTPRINT: ArchiveFootprint = ArchiveFootprint(message_count=0, archive_bytes=0, modeled=False)
"""Stores the footprint resolve_archive_footprint returns for an archive it cannot read."""

_UNMODELED_MEMORY_MB: int = _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB)
"""Stores the memory floor every consumer plans around when an archive yields no footprint."""


def _modeled_footprint(message_count: int, archive_bytes: int) -> ArchiveFootprint:
    """Builds a modeled footprint carrying the requested message count and on-disk size."""
    return ArchiveFootprint(message_count=message_count, archive_bytes=archive_bytes, modeled=True)


def _expected_memory_mb(archive_bytes: int, cores: int) -> int:
    """Recomputes the modeled memory estimate for an archive of the requested size at the requested core count."""
    per_reader = _bytes_to_megabytes(byte_count=archive_bytes * _ARCHIVE_DIRECTORY_RATIO)

    # A single-core job takes the sequential path, which opens no extraction pool and holds the body's reader alone.
    if cores == 1:
        return _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB + per_reader)

    # Every pool child holds a spawned child's baseline and a reader of its own, and the job body holds one of each.
    readers = cores + 1
    return _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB * readers + per_reader * readers)


def _write_archive(archive_path, source_id: int = 1) -> None:
    """Writes a small readable log archive holding four module messages and two kernel messages."""
    create_test_archive(
        archive_path=archive_path,
        source_id=source_id,
        onset_us=1_000_000,
        messages=[
            (1000, make_module_state_payload(module_type=1, module_id=2, command=1, event=10)),
            (2000, make_module_state_payload(module_type=1, module_id=2, command=1, event=11)),
            (
                3000,
                make_module_data_payload(
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
                make_module_data_payload(
                    module_type=1,
                    module_id=2,
                    command=2,
                    event=21,
                    prototype_code=SerialPrototypes.ONE_UINT8,
                    data_bytes=[43],
                ),
            ),
            (5000, make_kernel_state_payload(command=3, event=30)),
            (
                6000,
                make_kernel_data_payload(
                    command=4, event=40, prototype_code=SerialPrototypes.ONE_UINT8, data_bytes=[7]
                ),
            ),
        ],
    )


def test_archive_footprint_fields():
    """Verifies that ArchiveFootprint stores the message count, the archive size, and the modeled flag."""
    footprint = ArchiveFootprint(message_count=1500, archive_bytes=4096, modeled=True)

    assert footprint.message_count == 1500
    assert footprint.archive_bytes == 4096
    assert footprint.modeled

    # The dataclass is frozen, so the resolved figures cannot drift after a consumer receives them.
    with pytest.raises(AttributeError):
        footprint.message_count = 10


def test_resolve_archive_footprint_models_real_archive(tmp_path):
    """Verifies that resolve_archive_footprint models a readable log archive from its directory and file size."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    _write_archive(archive_path=archive_path)

    footprint = resolve_archive_footprint(archive_path=archive_path)

    assert footprint.modeled
    # The onset message is excluded from the count, leaving the four module messages and the two kernel messages.
    assert footprint.message_count == 6
    assert footprint.archive_bytes == archive_path.stat().st_size
    assert footprint.archive_bytes > 0


def test_resolve_archive_footprint_skips_message_count(tmp_path):
    """Verifies that resolve_archive_footprint reports a stat-only footprint when the message count is not requested."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    _write_archive(archive_path=archive_path)

    footprint = resolve_archive_footprint(archive_path=archive_path, read_message_count=False)

    # The archive is never opened, so the size is measured while the count stays at zero. The zero count still sizes
    # memory correctly and resolves the job to a single core.
    assert footprint.modeled
    assert footprint.message_count == 0
    assert footprint.archive_bytes == archive_path.stat().st_size
    assert resolve_job_workers(footprint=footprint, ceiling=CONTROLLER_EXTRACTION_JOB_CORES) == 1


@pytest.mark.parametrize("read_message_count", [True, False])
def test_resolve_archive_footprint_falls_back_for_missing_archive(tmp_path, read_message_count):
    """Verifies that resolve_archive_footprint returns an unmodeled footprint for an archive that does not exist."""
    footprint = resolve_archive_footprint(
        archive_path=tmp_path / f"1{LOG_ARCHIVE_SUFFIX}", read_message_count=read_message_count
    )

    # Neither the message count read nor the stat call can resolve a path that does not exist.
    assert footprint == _UNMODELED_FOOTPRINT
    assert not footprint.modeled
    assert footprint.message_count == 0
    assert footprint.archive_bytes == 0


def test_resolve_archive_footprint_falls_back_for_corrupt_archive(tmp_path):
    """Verifies that resolve_archive_footprint returns an unmodeled footprint for an archive it cannot decode."""
    archive_path = tmp_path / f"2{LOG_ARCHIVE_SUFFIX}"
    archive_path.write_text("This is not a valid numpy archive.")

    footprint = resolve_archive_footprint(archive_path=archive_path)

    assert footprint == _UNMODELED_FOOTPRINT


def test_resolve_archive_footprint_models_corrupt_archive_without_message_count(tmp_path):
    """Verifies that resolve_archive_footprint models a corrupt archive from its size when it does not open it."""
    archive_path = tmp_path / f"2{LOG_ARCHIVE_SUFFIX}"
    archive_path.write_text("This is not a valid numpy archive.")

    footprint = resolve_archive_footprint(archive_path=archive_path, read_message_count=False)

    # Skipping the message count skips the decode that would have rejected the file, so the sizing model sees a
    # readable file of the reported size. The stage that opens the archive is the one that reports the corruption.
    assert footprint.modeled
    assert footprint.message_count == 0
    assert footprint.archive_bytes == archive_path.stat().st_size


@pytest.mark.parametrize("message_count", [0, 1, PARALLEL_PROCESSING_THRESHOLD - 1])
def test_resolve_job_workers_below_threshold(message_count):
    """Verifies that resolve_job_workers gives a single core to an archive below the parallel processing threshold."""
    footprint = _modeled_footprint(message_count=message_count, archive_bytes=1024)

    assert resolve_job_workers(footprint=footprint, ceiling=CONTROLLER_EXTRACTION_JOB_CORES * 4) == 1


def test_resolve_job_workers_caps_at_repaid_workers():
    """Verifies that resolve_job_workers holds a mid-size archive to the workers its message count repays."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 3, archive_bytes=64 * _MEGABYTE)

    workers = resolve_job_workers(footprint=footprint, ceiling=CONTROLLER_EXTRACTION_JOB_CORES * 4)

    # Three thresholds' worth of messages repay three workers, which is below the declared width of the stage.
    assert workers == footprint.message_count // PARALLEL_PROCESSING_THRESHOLD
    assert workers == 3
    assert workers < CONTROLLER_EXTRACTION_JOB_CORES


def test_resolve_job_workers_caps_at_declared_width():
    """Verifies that resolve_job_workers holds a large archive to the declared core allocation of the stage."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 100, archive_bytes=512 * _MEGABYTE)

    workers = resolve_job_workers(footprint=footprint, ceiling=CONTROLLER_EXTRACTION_JOB_CORES * 4)

    assert workers == CONTROLLER_EXTRACTION_JOB_CORES


def test_resolve_job_workers_caps_at_ceiling():
    """Verifies that resolve_job_workers never returns more cores than the ceiling that sized the job."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 100, archive_bytes=512 * _MEGABYTE)

    for ceiling in (1, 2, 3, CONTROLLER_EXTRACTION_JOB_CORES - 1):
        workers = resolve_job_workers(footprint=footprint, ceiling=ceiling)

        # A job resolved above its ceiling is never dispatchable within the budget that sized it, which stalls the
        # dispatcher, so the ceiling binds ahead of both the declared width and the repaid workers.
        assert workers <= ceiling
        assert workers == ceiling
        assert workers < CONTROLLER_EXTRACTION_JOB_CORES


def test_resolve_job_workers_floors_at_one():
    """Verifies that resolve_job_workers returns at least one core when the ceiling is non-positive."""
    footprint = _modeled_footprint(message_count=PARALLEL_PROCESSING_THRESHOLD * 100, archive_bytes=512 * _MEGABYTE)

    assert resolve_job_workers(footprint=footprint, ceiling=0) == 1
    assert resolve_job_workers(footprint=footprint, ceiling=-5) == 1


def test_estimate_job_memory_mb_unmodeled_footprint():
    """Verifies that estimate_job_memory_mb falls back to the spawned child baseline for an unmodeled footprint."""
    # The core count does not enter the estimate, since an unmodeled footprint carries no archive to split.
    for cores in (1, 4, CONTROLLER_EXTRACTION_JOB_CORES):
        estimate = estimate_job_memory_mb(footprint=_UNMODELED_FOOTPRINT, cores=cores)

        assert estimate == _UNMODELED_MEMORY_MB
        assert estimate == _MEGABYTES_PER_GIGABYTE
        assert estimate % _MEGABYTES_PER_GIGABYTE == 0


def test_estimate_job_memory_mb_charges_one_body_and_one_reader_serially():
    """Verifies that estimate_job_memory_mb charges a single-core job for one spawned child and one reader."""
    footprint = _modeled_footprint(message_count=10_000, archive_bytes=64 * _MEGABYTE)

    estimate = estimate_job_memory_mb(footprint=footprint, cores=1)

    # A 64 MB archive builds a 257 MB reader, which the sequential body holds alongside its own 200 MB baseline. The
    # 457 MB sum carries the tolerance to 526 MB and rounds up to the single gigabyte the batch is charged.
    per_reader = _bytes_to_megabytes(byte_count=footprint.archive_bytes * _ARCHIVE_DIRECTORY_RATIO)
    assert per_reader == 257
    assert estimate == _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB + per_reader)
    assert estimate == _expected_memory_mb(archive_bytes=footprint.archive_bytes, cores=1)
    assert estimate == 1024


def test_estimate_job_memory_mb_charges_every_reader_in_parallel():
    """Verifies that estimate_job_memory_mb charges a multi-core job for the job body and every pool child."""
    footprint = _modeled_footprint(message_count=10_000, archive_bytes=64 * _MEGABYTE)
    cores = 4

    estimate = estimate_job_memory_mb(footprint=footprint, cores=cores)

    # Four pool children plus the job body hold five spawned child baselines and five readers, which is 2285 MB before
    # the tolerance carries it to 2628 MB and the rounding lifts it to three gigabytes.
    per_reader = _bytes_to_megabytes(byte_count=footprint.archive_bytes * _ARCHIVE_DIRECTORY_RATIO)
    readers = cores + 1
    assert estimate == _apply_tolerance(memory_mb=SPAWNED_CHILD_MEMORY_MB * readers + per_reader * readers)
    assert estimate == _expected_memory_mb(archive_bytes=footprint.archive_bytes, cores=cores)
    assert estimate == 3072


def test_estimate_job_memory_mb_parallel_path_exceeds_serial_path():
    """Verifies that estimate_job_memory_mb charges a two-core job more than the sequential path it leaves behind."""
    footprint = _modeled_footprint(message_count=10_000, archive_bytes=64 * _MEGABYTE)

    serial_estimate = estimate_job_memory_mb(footprint=footprint, cores=1)
    parallel_estimate = estimate_job_memory_mb(footprint=footprint, cores=2)

    # Taking a second core opens a pool, so the body's reader is joined by one reader and one baseline per child.
    assert parallel_estimate > serial_estimate
    assert serial_estimate == 1024
    assert parallel_estimate == 2048


def test_estimate_job_memory_mb_scales_with_cores():
    """Verifies that estimate_job_memory_mb grows with the cores a job holds, as every core opens its own reader."""
    footprint = _modeled_footprint(message_count=1_000_000, archive_bytes=512 * _MEGABYTE)
    core_counts = (1, 2, 4, CONTROLLER_EXTRACTION_JOB_CORES)

    estimates = [estimate_job_memory_mb(footprint=footprint, cores=cores) for cores in core_counts]

    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]
    assert estimates == [_expected_memory_mb(archive_bytes=footprint.archive_bytes, cores=c) for c in core_counts]
    assert estimates == [3072, 8192, 13312, 23552]
    assert all(estimate % _MEGABYTES_PER_GIGABYTE == 0 for estimate in estimates)


def test_estimate_job_memory_mb_scales_with_archive_bytes():
    """Verifies that estimate_job_memory_mb grows with the size of the archive the job reads."""
    archive_sizes = (1024, 64 * _MEGABYTE, 512 * _MEGABYTE)

    estimates = [
        estimate_job_memory_mb(footprint=_modeled_footprint(message_count=1_000_000, archive_bytes=size), cores=4)
        for size in archive_sizes
    ]

    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]
    assert estimates == [_expected_memory_mb(archive_bytes=size, cores=4) for size in archive_sizes]
    assert estimates == [2048, 3072, 13312]
    assert all(estimate % _MEGABYTES_PER_GIGABYTE == 0 for estimate in estimates)


def test_estimate_archive_job_memory_mb_models_real_archive(tmp_path):
    """Verifies that estimate_archive_job_memory_mb sizes a job from the archive on disk without opening it."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    _write_archive(archive_path=archive_path)
    archive_bytes = archive_path.stat().st_size

    memory_mb, modeled = estimate_archive_job_memory_mb(archive_path=archive_path, cores=4)

    assert modeled
    assert memory_mb == _expected_memory_mb(archive_bytes=archive_bytes, cores=4)
    # The estimate follows the stat-only footprint, whose message count stays at zero because the archive is not read.
    assert memory_mb == estimate_job_memory_mb(
        footprint=_modeled_footprint(message_count=0, archive_bytes=archive_bytes), cores=4
    )
    assert memory_mb > _UNMODELED_MEMORY_MB


def test_estimate_archive_job_memory_mb_scales_with_cores(tmp_path):
    """Verifies that estimate_archive_job_memory_mb charges more memory as the caller widens the job."""
    archive_path = tmp_path / f"1{LOG_ARCHIVE_SUFFIX}"
    _write_archive(archive_path=archive_path)

    estimates = [
        estimate_archive_job_memory_mb(archive_path=archive_path, cores=cores)[0]
        for cores in (1, 2, 4, CONTROLLER_EXTRACTION_JOB_CORES)
    ]

    assert estimates == sorted(estimates)
    assert estimates[0] < estimates[-1]
    assert all(estimate % _MEGABYTES_PER_GIGABYTE == 0 for estimate in estimates)


def test_estimate_archive_job_memory_mb_falls_back_for_missing_archive(tmp_path):
    """Verifies that estimate_archive_job_memory_mb reports the baseline floor for an archive that does not exist."""
    memory_mb, modeled = estimate_archive_job_memory_mb(archive_path=tmp_path / f"1{LOG_ARCHIVE_SUFFIX}", cores=4)

    # The flag is what tells a scheduler the figure is a floor to plan around rather than a measurement.
    assert not modeled
    assert memory_mb == _UNMODELED_MEMORY_MB
    assert memory_mb == _MEGABYTES_PER_GIGABYTE


def test_estimate_archive_job_memory_mb_models_undecodable_archive(tmp_path):
    """Verifies that estimate_archive_job_memory_mb sizes a present but undecodable archive from its size."""
    archive_path = tmp_path / f"2{LOG_ARCHIVE_SUFFIX}"
    archive_path.write_bytes(b"0" * (4 * _MEGABYTE))

    memory_mb, modeled = estimate_archive_job_memory_mb(archive_path=archive_path, cores=2)

    # The stat call succeeds, so the sizing model never learns the archive cannot be decoded. Sizing a doomed job is
    # harmless, since the extraction stage is the one that reports the failure.
    assert modeled
    assert memory_mb == _expected_memory_mb(archive_bytes=4 * _MEGABYTE, cores=2)


def test_resolve_host_memory_mb():
    """Verifies that resolve_host_memory_mb reports a positive physical memory figure for the host."""
    host_memory_mb = resolve_host_memory_mb()

    assert isinstance(host_memory_mb, int)
    assert host_memory_mb > 0


def test_resolve_core_budget_honors_positive_request():
    """Verifies that resolve_core_budget honors a positive core request up to the logical core count of the host."""
    available_cores = os.cpu_count() or 1

    assert resolve_core_budget(requested_budget=1) == 1
    assert resolve_core_budget(requested_budget=available_cores) == available_cores

    # A request above the core count is capped rather than honored, since the reserved cores do not apply to it.
    assert resolve_core_budget(requested_budget=available_cores * 100) == available_cores


@pytest.mark.parametrize("requested_budget", [0, -1, -100])
def test_resolve_core_budget_auto_resolves(requested_budget):
    """Verifies that resolve_core_budget auto-resolves a non-positive request to at least one core."""
    available_cores = os.cpu_count() or 1

    budget = resolve_core_budget(requested_budget=requested_budget)

    assert budget >= 1
    assert budget <= available_cores
    assert budget == max(1, available_cores - _RESERVED_CORES)


@pytest.mark.parametrize("requested_budget_mb", [1, 1024, 4096, 1_000_000])
def test_resolve_memory_budget_mb_honors_positive_request(requested_budget_mb):
    """Verifies that resolve_memory_budget_mb returns a positive memory request verbatim."""
    assert resolve_memory_budget_mb(requested_budget_mb=requested_budget_mb) == requested_budget_mb


@pytest.mark.parametrize("requested_budget_mb", [0, -1, -4096])
def test_resolve_memory_budget_mb_auto_resolves(requested_budget_mb):
    """Verifies that resolve_memory_budget_mb auto-resolves a non-positive request to a share of the host memory."""
    host_memory_mb = resolve_host_memory_mb()

    budget_mb = resolve_memory_budget_mb(requested_budget_mb=requested_budget_mb)

    assert budget_mb == max(_MINIMUM_MEMORY_BUDGET_MB, int(host_memory_mb * _MEMORY_BUDGET_FRACTION))
    assert budget_mb >= _MINIMUM_MEMORY_BUDGET_MB
    # The auto-resolved budget always leaves the host a share of its own memory.
    assert budget_mb <= max(_MINIMUM_MEMORY_BUDGET_MB, host_memory_mb)


def test_resolve_pool_size_binds_on_job_count():
    """Verifies that resolve_pool_size opens no more job slots than the batch holds jobs."""
    # Both budgets are set well above what three jobs can claim, leaving the job count as the binding term.
    assert resolve_pool_size(job_count=3, core_budget=64, memory_budget_mb=100_000) == 3


def test_resolve_pool_size_binds_on_core_budget():
    """Verifies that resolve_pool_size opens no more job slots than the core budget can hold running jobs."""
    assert resolve_pool_size(job_count=100, core_budget=6, memory_budget_mb=100_000) == 6


def test_resolve_pool_size_binds_on_affordable_bodies():
    """Verifies that resolve_pool_size holds the slot count to the warmed job bodies half the memory budget holds."""
    memory_budget_mb = 1024
    affordable_bodies = (memory_budget_mb // _POOL_MEMORY_RESERVATION_DIVISOR) // SPAWNED_CHILD_MEMORY_MB

    pool_size = resolve_pool_size(job_count=100, core_budget=64, memory_budget_mb=memory_budget_mb)

    # Half of a one gigabyte budget holds two 200 MB bodies, leaving the remainder for the work those bodies perform.
    assert pool_size == affordable_bodies
    assert pool_size == 2


def test_resolve_pool_size_scales_with_memory_budget():
    """Verifies that resolve_pool_size opens more job slots as the memory budget grows."""
    pool_sizes = [
        resolve_pool_size(job_count=100, core_budget=64, memory_budget_mb=budget_mb)
        for budget_mb in (1024, 4096, 16_384)
    ]

    assert pool_sizes == sorted(pool_sizes)
    assert pool_sizes[0] < pool_sizes[-1]
    assert pool_sizes == [2, 10, 40]


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
def test_resolve_pool_size_floors_at_one(job_count, core_budget, memory_budget_mb):
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
def test_bytes_to_megabytes_rounds_up(byte_count, expected):
    """Verifies that _bytes_to_megabytes converts a positive byte count into whole megabytes with one of headroom."""
    assert _bytes_to_megabytes(byte_count=byte_count) == expected
    assert _BYTES_PER_MEGABYTE == _MEGABYTE


@pytest.mark.parametrize("byte_count", [0, 0.0, -1, -_MEGABYTE, -0.5])
def test_bytes_to_megabytes_non_positive_byte_count(byte_count):
    """Verifies that _bytes_to_megabytes converts a zero or negative byte count to zero megabytes."""
    assert _bytes_to_megabytes(byte_count=byte_count) == 0


@pytest.mark.parametrize(
    "memory_mb, expected",
    [
        (0, 1024),
        (1, 1024),
        (SPAWNED_CHILD_MEMORY_MB, 1024),
        (1024, 2048),
        (2048, 3072),
    ],
)
def test_apply_tolerance(memory_mb, expected):
    """Verifies that _apply_tolerance carries the estimate margin and rounds the figure up to a whole gigabyte."""
    reportable = _apply_tolerance(memory_mb=memory_mb)

    assert reportable == expected
    assert reportable % _MEGABYTES_PER_GIGABYTE == 0
    assert reportable >= memory_mb
