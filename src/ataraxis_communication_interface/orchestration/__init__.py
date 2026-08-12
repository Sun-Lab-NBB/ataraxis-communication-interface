"""Provides the orchestration layer: the job identity and output layout, the archive-derived sizing model, the
manifest-derived job resolution, the single-job runner, the shared-pool batch engine, and the sequential pipeline.
"""

from .jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    JobSizing,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    find_module_paths,
    parse_module_path,
    resolve_kernel_path,
    resolve_module_path,
    resolve_tracker_path,
    resolve_output_directory,
)
from .worker import execute_job, run_extraction_job, resolve_controller_config
from .pipeline import run_log_processing_pipeline
from .discovery import JobSet, JobSource, JobUniverse, size_job, prepare_jobs, resolve_jobs
from .execution import (
    JobExecutionState,
    get_execution_state,
    set_execution_state,
    group_jobs_by_tracker,
    job_execution_manager,
    start_execution_session,
)
from .allocation import (
    SPAWNED_CHILD_MEMORY_MB,
    PARALLEL_EXTRACTION_THRESHOLD,
    CONTROLLER_EXTRACTION_JOB_CORES,
    ArchiveFootprint,
    size_archive_job,
    resolve_pool_size,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_host_memory_mb,
    resolve_memory_budget_mb,
    resolve_archive_footprint,
)

__all__ = [
    "CONTROLLER_EXTRACTION_JOB_CORES",
    "CONTROLLER_EXTRACTION_JOB_NAME",
    "PARALLEL_EXTRACTION_THRESHOLD",
    "SPAWNED_CHILD_MEMORY_MB",
    "ArchiveFootprint",
    "JobDescriptor",
    "JobExecutionState",
    "JobSet",
    "JobSizing",
    "JobSource",
    "JobUniverse",
    "OutputLayout",
    "estimate_job_memory_mb",
    "execute_job",
    "find_module_paths",
    "generate_job_ids",
    "get_execution_state",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "parse_module_path",
    "prepare_jobs",
    "resolve_archive_footprint",
    "resolve_controller_config",
    "resolve_core_budget",
    "resolve_host_memory_mb",
    "resolve_job_workers",
    "resolve_jobs",
    "resolve_kernel_path",
    "resolve_memory_budget_mb",
    "resolve_module_path",
    "resolve_output_directory",
    "resolve_pool_size",
    "resolve_tracker_path",
    "run_extraction_job",
    "run_log_processing_pipeline",
    "set_execution_state",
    "size_archive_job",
    "size_job",
    "start_execution_session",
]
