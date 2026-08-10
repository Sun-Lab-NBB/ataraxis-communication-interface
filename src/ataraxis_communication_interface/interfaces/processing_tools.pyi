from typing import Any

from .mcp_instance import (
    mcp as mcp,
    read_tracker_status as read_tracker_status,
)
from ..orchestration import (
    TRACKER_FILENAME as TRACKER_FILENAME,
    EXTRACTION_JOB_NAME as EXTRACTION_JOB_NAME,
    MICROCONTROLLER_DATA_DIRECTORY as MICROCONTROLLER_DATA_DIRECTORY,
    PendingJob as PendingJob,
    JobExecutionState as JobExecutionState,
    generate_job_ids as generate_job_ids,
    size_pending_job as size_pending_job,
    get_execution_state as get_execution_state,
    resolve_core_budget as resolve_core_budget,
    set_execution_state as set_execution_state,
    group_jobs_by_tracker as group_jobs_by_tracker,
    job_execution_manager as job_execution_manager,
    resolve_memory_budget_mb as resolve_memory_budget_mb,
    discover_microcontroller_jobs as discover_microcontroller_jobs,
)
from ..microcontroller import ExtractionConfig as ExtractionConfig

def prepare_log_processing_batch_tool(
    log_directories: list[str], source_ids: list[str], output_directories: list[str], config_path: str
) -> dict[str, Any]: ...
def execute_log_processing_jobs_tool(
    jobs: list[dict[str, str]], *, worker_budget: int = -1, memory_budget_mb: int = -1
) -> dict[str, Any]: ...
def get_log_processing_status_tool() -> dict[str, Any]: ...
def get_log_processing_timing_tool() -> dict[str, Any]: ...
def cancel_log_processing_tool() -> dict[str, Any]: ...
def reset_log_processing_jobs_tool(tracker_path: str, source_ids: list[str] | None = None) -> dict[str, Any]: ...
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]: ...
