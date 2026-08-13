from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from ataraxis_data_structures import ProcessingTracker

from .jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME as CONTROLLER_EXTRACTION_JOB_NAME,
    JobDescriptor as JobDescriptor,
    resolve_kernel_path as resolve_kernel_path,
    resolve_module_path as resolve_module_path,
)
from ..microcontroller import (
    ExtractionConfig as ExtractionConfig,
    ExtractedMessages as ExtractedMessages,
    ControllerExtractionConfig as ControllerExtractionConfig,
    build_message_dataframe as build_message_dataframe,
    extract_logged_microcontroller_data as extract_logged_microcontroller_data,
)

def run_extraction_job(job: JobDescriptor) -> None: ...
def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    config_path: Path,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> None: ...
def resolve_controller_config(config_path: Path, source_id: str) -> ControllerExtractionConfig: ...
def _resolve_event_filters(
    controller_config: ControllerExtractionConfig, source_id: str
) -> tuple[dict[tuple[int, int], frozenset[int]] | None, frozenset[int] | None]: ...
def _write_messages(messages: ExtractedMessages, file_path: Path) -> None: ...
