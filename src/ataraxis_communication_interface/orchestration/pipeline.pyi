from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from ataraxis_data_structures import ProcessingTracker

from .jobs import (
    TRACKER_FILENAME as TRACKER_FILENAME,
    EXTRACTION_JOB_NAME as EXTRACTION_JOB_NAME,
    MICROCONTROLLER_DATA_DIRECTORY as MICROCONTROLLER_DATA_DIRECTORY,
    generate_job_ids as generate_job_ids,
    resolve_kernel_feather_path as resolve_kernel_feather_path,
    resolve_module_feather_path as resolve_module_feather_path,
    discover_microcontroller_jobs as discover_microcontroller_jobs,
)
from .allocation import (
    resolve_core_budget as resolve_core_budget,
    resolve_job_workers as resolve_job_workers,
    resolve_archive_footprint as resolve_archive_footprint,
)
from ..microcontroller import (
    ExtractionConfig as ExtractionConfig,
    ExtractedMessages as ExtractedMessages,
    ControllerExtractionConfig as ControllerExtractionConfig,
    build_message_dataframe as build_message_dataframe,
    extract_logged_microcontroller_data as extract_logged_microcontroller_data,
)

def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    controller_config: ControllerExtractionConfig,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> None: ...
def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None: ...
def _resolve_controller_configs(config: Path, universe_ids: list[str]) -> dict[str, ControllerExtractionConfig]: ...
def _resolve_event_filters(
    controller_config: ControllerExtractionConfig, source_id: str
) -> tuple[dict[tuple[int, int], frozenset[int]] | None, frozenset[int] | None]: ...
def _execute_sized_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    ceiling: int,
    tracker: ProcessingTracker,
    controller_config: ControllerExtractionConfig,
    *,
    display_progress: bool,
) -> None: ...
def _write_messages(messages: ExtractedMessages, file_path: Path) -> None: ...
