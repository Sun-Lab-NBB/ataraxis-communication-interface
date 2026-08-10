from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass
from collections.abc import Sequence

from ataraxis_data_structures import ProcessingTracker

from .jobs import PendingJob as PendingJob
from .pipeline import execute_job as execute_job
from .allocation import (
    ArchiveFootprint as ArchiveFootprint,
    resolve_job_workers as resolve_job_workers,
    estimate_job_memory_mb as estimate_job_memory_mb,
    resolve_archive_footprint as resolve_archive_footprint,
)
from ..microcontroller import (
    ExtractionConfig as ExtractionConfig,
    ControllerExtractionConfig as ControllerExtractionConfig,
)

_WORKER_THREAD_CEILING: int
_DISPATCH_POLL_SECONDS: int

@dataclass(slots=True)
class _ActiveJob:
    job: PendingJob
    thread: Thread

@dataclass(slots=True)
class JobExecutionState:
    all_jobs: dict[tuple[str, str], PendingJob] = field(default_factory=dict)
    pending_jobs: list[PendingJob] = field(default_factory=list)
    active_jobs: list[_ActiveJob] = field(default_factory=list)
    core_budget: int = ...
    memory_budget_mb: int = ...
    lock: Lock = field(default_factory=Lock)
    manager_thread: Thread | None = ...
    canceled: bool = ...

_execution_state: JobExecutionState | None

def get_execution_state() -> JobExecutionState | None: ...
def set_execution_state(state: JobExecutionState | None) -> None: ...
def size_pending_job(job: PendingJob, core_budget: int) -> ArchiveFootprint: ...
def job_execution_manager(state: JobExecutionState) -> None: ...
def group_jobs_by_tracker(state: JobExecutionState) -> dict[Path, list[PendingJob]]: ...
def _select_admissible_jobs(
    pending: Sequence[PendingJob], core_budget: int, memory_budget_mb: int, used_cores: int, used_memory_mb: int
) -> tuple[list[PendingJob], list[PendingJob]]: ...
def _run_job(job: PendingJob) -> None: ...
def _resolve_controller_config(job: PendingJob, tracker: ProcessingTracker) -> ControllerExtractionConfig | None: ...
def _record_failure(tracker: ProcessingTracker, job_id: str, error_message: str) -> None: ...
