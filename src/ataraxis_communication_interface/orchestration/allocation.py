"""Provides the declared core allocation of the data extraction stage and the archive-derived footprint model that
sizes each job's cores and memory from the input it will read.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

import psutil
from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import PARALLEL_PROCESSING_THRESHOLD, read_archive_message_count

if TYPE_CHECKING:
    from pathlib import Path

_RESERVED_CORES: int = 2
"""The number of CPU cores held back for host-system operations when a core budget auto-resolves. The value reaches
``resolve_worker_count``, which applies it only to a non-positive budget and honors an explicit budget up to the
logical core count.

Notes:
    The term is private to this module, because it describes how this library resolves its own batch budget rather
    than how a scheduler resolves the budget it dispatches this stage from. A consumer that runs its own batch holds
    the reservation it plans against, so exporting this one would offer it a second figure for the same decision.
"""

EXTRACTION_JOB_CORES: int = 8
"""The cores one microcontroller data extraction job occupies while it runs.

Notes:
    The stage splits its archive across a worker pool, and every worker opens the archive itself for each batch it
    receives, so the fixed cost per worker does not shrink as workers are added and the speedup flattens well before
    the core count. The value is an upper bound rather than a fixed width, since a job reading a small archive
    resolves below it.

    A scheduler that dispatches this stage reads the width from here, which keeps the figure with the library that
    owns the stage rather than with each consumer that runs it.
"""

_MEMORY_ESTIMATE_TOLERANCE: float = 1.15
"""The margin applied to every memory estimate before it is reported. It covers the working sets the model does not
enumerate and the variation between archives of the same size. The penalty for understating is asymmetric, since a
batch that overcommits its host swaps or is killed outright, so estimates round up.

Notes:
    This term and the four that follow it carry the values the platform's own estimators were calibrated to against
    measured peaks. A scheduler plans a batch that mixes this stage with the stages those estimators cover, so the
    figures have to be read on one scale, and a value that drifts from the calibrated one makes this stage's jobs
    admit against a different standard than everything queued beside them.
"""

_WORKER_MEMORY_MB: int = 384
"""The resident memory a job occupies before it reads any data, covering the interpreter and the package's import
graph. The term is charged once per job, while the per-core allowance is charged for every core the job holds."""

_SUBPROCESS_MEMORY_MB: int = 200
"""The resident memory each child of a job's worker pool occupies before it touches data. Processes are spawned
rather than forked, so every child re-imports the module its target function lives in."""

_ARCHIVE_DIRECTORY_RATIO: float = 4.0
"""The resident memory a log archive reader holds per byte of archive on disk. Reading an archive builds one
directory entry per logged message, which dominates the decoded payload itself."""

_MEMORY_BUDGET_FRACTION: float = 0.85
"""The share of the host's physical memory an auto-resolved batch budget claims, leaving the remainder to the host."""

_MINIMUM_MEMORY_BUDGET_MB: int = 1024
"""The floor an auto-resolved memory budget is held to, so a small host still admits one job at a time."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds, which every reportable estimate is rounded up to a multiple of."""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""


@dataclass(frozen=True, slots=True)
class ArchiveFootprint:
    """Describes the on-disk properties of one log archive that size the job reading it."""

    message_count: int
    """The number of data messages the archive holds."""
    archive_bytes: int
    """The size of the archive file on disk."""
    modeled: bool
    """Determines whether the figures were read from the archive rather than falling back to the job baseline."""


def resolve_archive_footprint(archive_path: Path) -> ArchiveFootprint:
    """Reads the properties of the target log archive that size the job reading it.

    Notes:
        Reads the archive's zip directory and its file metadata alone, so sizing an archive decodes no message and
        loads no payload. An archive that cannot be read yields an unmodeled footprint, which every consumer treats
        as a floor rather than as a measurement.

    Args:
        archive_path: The path to the .npz log archive to read.

    Returns:
        The footprint describing the archive.
    """
    try:
        message_count = read_archive_message_count(archive_path=archive_path)
        archive_bytes = archive_path.stat().st_size
    except Exception:
        return ArchiveFootprint(message_count=0, archive_bytes=0, modeled=False)

    return ArchiveFootprint(message_count=message_count, archive_bytes=archive_bytes, modeled=True)


def resolve_job_workers(footprint: ArchiveFootprint, ceiling: int) -> int:
    """Resolves the cores one extraction job receives, from the archive it reads and the cores its host can supply.

    Notes:
        An archive below the parallel processing threshold takes a single core, because the stage processes it
        sequentially whatever width it is given. Above that threshold the width is the declared allocation, narrowed
        to the cores the host supplies and to the workers the archive itself repays. A worker repays its fixed cost
        of opening the archive only once it holds a threshold's worth of messages, so an archive holding a few
        multiples of the threshold resolves below the declared width rather than at it.

        The result never passes a ceiling of one or more, so a job is always dispatchable within the budget that
        sized it.

    Args:
        footprint: The footprint of the archive this job reads.
        ceiling: The cores available to this job, which is the batch's core budget or a caller's explicit limit. A
            ceiling below one yields a single core, since every job occupies at least one.

    Returns:
        The cores this job receives, always at least one.
    """
    if footprint.message_count < PARALLEL_PROCESSING_THRESHOLD:
        return 1

    repaid_workers = footprint.message_count // PARALLEL_PROCESSING_THRESHOLD
    return max(1, min(EXTRACTION_JOB_CORES, ceiling, repaid_workers))


def estimate_job_memory_mb(footprint: ArchiveFootprint, cores: int) -> int:
    """Estimates the memory one extraction job holds at its allocated core count.

    Notes:
        The job splits its archive across a worker pool and every worker opens the archive itself, so the archive's
        message directory is held once per allocated core. An unmodeled footprint falls back to the job baseline,
        which is a floor to plan around rather than a measurement.

        This is the same model the platform applies to every stage that splits one log archive across a worker pool,
        so an extraction job and the runtime and camera-timestamp jobs beside it are sized on one scale.

    Args:
        footprint: The footprint of the archive this job reads.
        cores: The cores this job holds, which is how many readers it opens.

    Returns:
        The memory this job holds, in megabytes, carrying the estimate tolerance and rounded up to a whole gigabyte.
    """
    if not footprint.modeled:
        return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)

    per_reader = _bytes_to_megabytes(byte_count=footprint.archive_bytes * _ARCHIVE_DIRECTORY_RATIO)
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + cores * (per_reader + _SUBPROCESS_MEMORY_MB))


def resolve_core_budget(requested_budget: int) -> int:
    """Resolves the cores a batch may commit across all of its concurrently running jobs.

    Args:
        requested_budget: The cores the caller requested. A non-positive value auto-resolves to every available core
            minus the reserved host cores.

    Returns:
        The cores the batch may commit, always at least one.
    """
    return resolve_worker_count(requested_workers=requested_budget, reserved_cores=_RESERVED_CORES)


def resolve_memory_budget_mb(requested_budget_mb: int) -> int:
    """Resolves the memory a batch may commit across all of its concurrently running jobs.

    Args:
        requested_budget_mb: The memory the caller requested, in megabytes. A non-positive value auto-resolves to a
            share of the host's physical memory.

    Returns:
        The memory the batch may commit, in megabytes. A positive request is returned verbatim, while an
        auto-resolved budget is held to at least the auto-resolution floor.
    """
    if requested_budget_mb > 0:
        return requested_budget_mb

    return max(_MINIMUM_MEMORY_BUDGET_MB, int(_resolve_host_memory_mb() * _MEMORY_BUDGET_FRACTION))


def _resolve_host_memory_mb() -> int:
    """Reads the host's total physical memory.

    Returns:
        The host's total physical memory, in megabytes.
    """
    return int(psutil.virtual_memory().total / _BYTES_PER_MEGABYTE)


def _bytes_to_megabytes(byte_count: float) -> int:
    """Converts a byte count into whole megabytes, rounding up so an estimate never understates its demand.

    Args:
        byte_count: The number of bytes to convert.

    Returns:
        The equivalent size in megabytes.
    """
    return int(byte_count / _BYTES_PER_MEGABYTE) + 1 if byte_count > 0 else 0


def _apply_tolerance(memory_mb: int) -> int:
    """Applies the estimate tolerance to a modeled memory figure and rounds it up to a whole gigabyte.

    Notes:
        Rounding to a whole gigabyte here keeps the figure an admission decision weighs identical to the figure a
        caller reports, which is what lets a planned batch and a running one be compared directly.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes, carrying the tolerance.
    """
    reportable = int(memory_mb * _MEMORY_ESTIMATE_TOLERANCE) + 1
    return math.ceil(reportable / _MEGABYTES_PER_GIGABYTE) * _MEGABYTES_PER_GIGABYTE
