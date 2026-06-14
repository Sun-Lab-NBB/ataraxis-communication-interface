from typing import Any

from .mcp_instance import (
    mcp as mcp,
    read_tracker_status as read_tracker_status,
)
from ..microcontroller import (
    FEATHER_SUFFIX as FEATHER_SUFFIX,
    TRACKER_FILENAME as TRACKER_FILENAME,
    KERNEL_FEATHER_INFIX as KERNEL_FEATHER_INFIX,
    MODULE_FEATHER_INFIX as MODULE_FEATHER_INFIX,
    MICROCONTROLLER_DATA_DIRECTORY as MICROCONTROLLER_DATA_DIRECTORY,
)

_MINIMUM_ROWS_FOR_INTERVALS: int

def verify_processing_output_tool(output_directory: str) -> dict[str, Any]: ...
def query_extracted_events_tool(feather_files: list[str], max_sample_rows: int = 10) -> dict[str, Any]: ...
def clean_log_processing_output_tool(output_directories: list[str]) -> dict[str, Any]: ...
def _clean_single_output(output_directory: str) -> dict[str, Any]: ...
def _analyze_single_event_feather(feather_file: str, max_sample_rows: int) -> dict[str, Any]: ...
