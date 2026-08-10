from pathlib import Path
from dataclasses import dataclass

_RESERVED_CORES: int
EXTRACTION_JOB_CORES: int
_MEMORY_ESTIMATE_TOLERANCE: float
_WORKER_MEMORY_MB: int
_SUBPROCESS_MEMORY_MB: int
_ARCHIVE_DIRECTORY_RATIO: float
_MEMORY_BUDGET_FRACTION: float
_MINIMUM_MEMORY_BUDGET_MB: int
_MEGABYTES_PER_GIGABYTE: int
_BYTES_PER_MEGABYTE: int

@dataclass(frozen=True, slots=True)
class ArchiveFootprint:
    message_count: int
    archive_bytes: int
    modeled: bool

def resolve_archive_footprint(archive_path: Path) -> ArchiveFootprint: ...
def resolve_job_workers(footprint: ArchiveFootprint, ceiling: int) -> int: ...
def estimate_job_memory_mb(footprint: ArchiveFootprint, cores: int) -> int: ...
def resolve_core_budget(requested_budget: int) -> int: ...
def resolve_memory_budget_mb(requested_budget_mb: int) -> int: ...
def _resolve_host_memory_mb() -> int: ...
def _bytes_to_megabytes(byte_count: float) -> int: ...
def _apply_tolerance(memory_mb: int) -> int: ...
