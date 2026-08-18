"""Provides the orchestration layer: the job identity and output layout, the archive-derived sizing model, the
manifest-derived job resolution, the single-job runner, the shared-pool batch engine, and the sequential pipeline.
"""

from .jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    JobSizing,
    OutputLayout,
    JobDescriptor,
    generate_job_ids,
    find_kernel_paths,
    find_module_paths,
    parse_kernel_path,
    parse_module_path,
    resolve_kernel_path,
    resolve_module_path,
)
from .worker import execute_job
from .pipeline import run_log_processing_pipeline
from .discovery import JobSet, JobSource, JobUniverse, size_job, prepare_jobs, resolve_jobs
from .execution import (
    ActiveJob,
    JobExecutionState,
    session_is_active,
    get_execution_state,
    group_jobs_by_tracker,
    start_execution_session,
    finish_execution_session,
)
from .allocation import (
    CONTROLLER_EXTRACTION_JOB_CORES,
    ArchiveFootprint,
    size_archive_job,
    resolve_pool_size,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_memory_budget_mb,
)

__all__ = [
    "CONTROLLER_EXTRACTION_JOB_CORES",
    "CONTROLLER_EXTRACTION_JOB_NAME",
    "ActiveJob",
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
    "find_kernel_paths",
    "find_module_paths",
    "finish_execution_session",
    "generate_job_ids",
    "get_execution_state",
    "group_jobs_by_tracker",
    "parse_kernel_path",
    "parse_module_path",
    "prepare_jobs",
    "resolve_core_budget",
    "resolve_job_workers",
    "resolve_jobs",
    "resolve_kernel_path",
    "resolve_memory_budget_mb",
    "resolve_module_path",
    "resolve_pool_size",
    "run_log_processing_pipeline",
    "session_is_active",
    "size_archive_job",
    "size_job",
    "start_execution_session",
]
