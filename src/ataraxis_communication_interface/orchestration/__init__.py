"""Provides the orchestration layer: the job descriptors and manifest-derived job discovery, the declared core
allocation and archive-derived footprint model, the local batch execution engine, and the pipeline entry point.
"""

from .jobs import (
    FEATHER_SUFFIX,
    TRACKER_FILENAME,
    EXTRACTION_JOB_NAME,
    KERNEL_FEATHER_INFIX,
    MODULE_FEATHER_INFIX,
    CONTROLLER_FEATHER_PREFIX,
    MICROCONTROLLER_DATA_DIRECTORY,
    PendingJob,
    generate_job_ids,
    find_module_feathers,
    parse_module_feather_name,
    resolve_kernel_feather_path,
    resolve_module_feather_path,
    discover_microcontroller_jobs,
)
from .pipeline import execute_job, run_log_processing_pipeline
from .execution import (
    JobExecutionState,
    size_pending_job,
    get_execution_state,
    set_execution_state,
    group_jobs_by_tracker,
    job_execution_manager,
)
from .allocation import (
    EXTRACTION_JOB_CORES,
    ArchiveFootprint,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_memory_budget_mb,
    resolve_archive_footprint,
)

__all__ = [
    "CONTROLLER_FEATHER_PREFIX",
    "EXTRACTION_JOB_CORES",
    "EXTRACTION_JOB_NAME",
    "FEATHER_SUFFIX",
    "KERNEL_FEATHER_INFIX",
    "MICROCONTROLLER_DATA_DIRECTORY",
    "MODULE_FEATHER_INFIX",
    "TRACKER_FILENAME",
    "ArchiveFootprint",
    "JobExecutionState",
    "PendingJob",
    "discover_microcontroller_jobs",
    "estimate_job_memory_mb",
    "execute_job",
    "find_module_feathers",
    "generate_job_ids",
    "get_execution_state",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "parse_module_feather_name",
    "resolve_archive_footprint",
    "resolve_core_budget",
    "resolve_job_workers",
    "resolve_kernel_feather_path",
    "resolve_memory_budget_mb",
    "resolve_module_feather_path",
    "run_log_processing_pipeline",
    "set_execution_state",
    "size_pending_job",
]
