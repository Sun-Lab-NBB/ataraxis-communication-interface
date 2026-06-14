from typing import Any
from pathlib import Path

from .mcp_instance import (
    mcp as mcp,
    read_tracker_status as read_tracker_status,
)
from .mcp_execution import (
    PendingJob as PendingJob,
    JobExecutionState as JobExecutionState,
    get_execution_state as get_execution_state,
    set_execution_state as set_execution_state,
    job_execution_manager as job_execution_manager,
)
from ..microcontroller import (
    TRACKER_FILENAME as TRACKER_FILENAME,
    LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX,
    EXTRACTION_JOB_NAME as EXTRACTION_JOB_NAME,
    MICROCONTROLLER_DATA_DIRECTORY as MICROCONTROLLER_DATA_DIRECTORY,
    ExtractionConfig as ExtractionConfig,
    prepare_tracker as prepare_tracker,
    generate_job_ids as generate_job_ids,
)

_RESERVED_CORES: int

def prepare_log_processing_batch_tool(
    log_directories: list[str], source_ids: list[str], output_directories: list[str], config_path: str
) -> dict[str, Any]: ...
def execute_log_processing_jobs_tool(jobs: list[dict[str, str]], *, worker_budget: int = -1) -> dict[str, Any]: ...
def get_log_processing_status_tool() -> dict[str, Any]: ...
def get_log_processing_timing_tool() -> dict[str, Any]: ...
def cancel_log_processing_tool() -> dict[str, Any]: ...
def reset_log_processing_jobs_tool(tracker_path: str, source_ids: list[str] | None = None) -> dict[str, Any]: ...
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]: ...
def _derive_tracker_status(summary: dict[str, Any]) -> str: ...
def _group_jobs_by_tracker(state: JobExecutionState) -> dict[Path, list[PendingJob]]: ...
def _probe_archive_message_count(job: PendingJob) -> int: ...
