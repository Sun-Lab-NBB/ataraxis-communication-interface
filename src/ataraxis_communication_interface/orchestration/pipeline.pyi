from pathlib import Path
from collections.abc import Sequence

from .worker import execute_job as execute_job
from .discovery import prepare_jobs as prepare_jobs
from .allocation import (
    resolve_job_workers as resolve_job_workers,
    resolve_archive_footprint as resolve_archive_footprint,
)

def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None = None,
    source_ids: Sequence[str] | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None: ...
