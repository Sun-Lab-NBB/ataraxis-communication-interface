from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass

from ..microcontroller import (
    PARALLEL_PROCESSING_THRESHOLD as PARALLEL_PROCESSING_THRESHOLD,
    ExtractionConfig as ExtractionConfig,
    execute_job as execute_job,
    find_log_archive as find_log_archive,
)

_WORKER_SCALING_FACTOR: int
_WORKER_MULTIPLE: int

@dataclass(slots=True)
class PendingJob:
    log_directory: Path
    output_directory: Path
    tracker_path: Path
    job_id: str
    source_id: str
    config_path: Path
    @property
    def dispatch_key(self) -> tuple[str, str]: ...

@dataclass(slots=True)
class JobExecutionState:
    all_jobs: dict[tuple[str, str], PendingJob] = field(default_factory=dict)
    pending_queue: list[PendingJob] = field(default_factory=list)
    active_groups: list[_ActiveGroup] = field(default_factory=list)
    job_message_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    worker_budget: int = ...
    lock: Lock = field(default_factory=Lock)
    manager_thread: Thread | None = ...
    canceled: bool = ...

@dataclass(slots=True)
class _ActiveGroup:
    source_id: str
    jobs: list[PendingJob]
    workers: int
    thread: Thread

_execution_state: JobExecutionState | None

def get_execution_state() -> JobExecutionState | None: ...
def set_execution_state(state: JobExecutionState | None) -> None: ...
def job_execution_manager() -> None: ...
def _group_worker(jobs: list[PendingJob], workers: int, state: JobExecutionState) -> None: ...
def _compute_sqrt_minimum(message_count: int) -> int: ...
