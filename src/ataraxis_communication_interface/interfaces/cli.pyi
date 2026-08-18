from typing import Literal
from pathlib import Path
from collections.abc import Callable as Callable

from ..communication import MQTTCommunication as MQTTCommunication
from ..orchestration import run_log_processing_pipeline as run_log_processing_pipeline
from ..microcontroller import (
    ExtractionConfig as ExtractionConfig,
    create_extraction_config as create_extraction_config,
    discover_microcontrollers as discover_microcontrollers,
)

_CONTEXT_SETTINGS: dict[str, int]

def _report_command_failure[**P](command: Callable[P, None]) -> Callable[P, None]: ...
def axci_cli() -> None: ...
@_report_command_failure
def identify(baudrate: int) -> None: ...
@_report_command_failure
def check_mqtt(host: str, port: int) -> None: ...
def config_group() -> None: ...
@_report_command_failure
def config_create(manifest_path: Path, output_path: Path) -> None: ...
@_report_command_failure
def config_show(config_path: Path) -> None: ...
@_report_command_failure
def process_log_archives(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None,
    specifier: tuple[str, ...],
    *,
    workers: int,
    no_progress: bool,
) -> None: ...
@_report_command_failure
def run_mcp_server(transport: Literal["stdio", "streamable-http"]) -> None: ...
