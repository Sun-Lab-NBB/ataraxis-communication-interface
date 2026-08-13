from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

from .jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME as CONTROLLER_EXTRACTION_JOB_NAME,
    JobSizing as JobSizing,
    JobDescriptor as JobDescriptor,
    generate_job_ids as generate_job_ids,
    resolve_tracker_path as resolve_tracker_path,
    resolve_output_directory as resolve_output_directory,
)
from .allocation import (
    CONTROLLER_EXTRACTION_JOB_CORES as CONTROLLER_EXTRACTION_JOB_CORES,
    resolve_job_workers as resolve_job_workers,
    estimate_job_memory_mb as estimate_job_memory_mb,
    resolve_archive_footprint as resolve_archive_footprint,
)
from ..microcontroller import (
    MICROCONTROLLER_MANIFEST_FILENAME as MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig as ExtractionConfig,
    MicroControllerManifest as MicroControllerManifest,
)

@dataclass(frozen=True, slots=True)
class JobSource:
    source_id: str
    name: str
    archive_path: Path | None

@dataclass(frozen=True, slots=True)
class JobUniverse:
    log_directory: Path
    manifest_path: Path | None
    sources: tuple[JobSource, ...]
    universe: tuple[tuple[str, str], ...]
    possible: tuple[tuple[str, str], ...]
    @property
    def archives(self) -> dict[str, Path]: ...

@dataclass(frozen=True, slots=True)
class JobSet:
    log_directory: Path
    output_directory: Path
    tracker_path: Path
    universe: tuple[tuple[str, str], ...]
    jobs: tuple[JobDescriptor, ...]
    skipped_sources: tuple[tuple[str, str], ...]
    def resolve_job(self, job_id: str) -> JobDescriptor: ...

def resolve_jobs(log_directory: Path) -> JobUniverse: ...
def prepare_jobs(
    log_directory: Path,
    output_directory: Path,
    config_path: Path,
    source_ids: Sequence[str] | None = None,
    job_id: str | None = None,
    *,
    strict_sources: bool = True,
) -> JobSet: ...
def size_job(job: JobDescriptor) -> tuple[JobDescriptor, JobSizing]: ...
def _resolve_configured_ids(config_path: Path, registered_ids: Sequence[str]) -> tuple[list[str], list[str]]: ...
