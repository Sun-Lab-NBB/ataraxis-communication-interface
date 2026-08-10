from typing import Any
from pathlib import Path

from .mcp_instance import mcp as mcp
from ..communication import MQTTCommunication as MQTTCommunication
from ..microcontroller import (
    MICROCONTROLLER_MANIFEST_FILENAME as MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData as ModuleSourceData,
    MicroControllerManifest as MicroControllerManifest,
    evaluate_port as evaluate_port,
    write_microcontroller_manifest as write_microcontroller_manifest,
)

_UNIDENTIFIED_CONTROLLER_ID: int
_WORKER_THREAD_CEILING: int

def list_microcontrollers_tool(baudrate: int = 115200) -> str: ...
def check_mqtt_broker_tool(host: str = "127.0.0.1", port: int = 1883) -> str: ...
def assemble_log_archives_tool(
    log_directory: str, *, remove_sources: bool = True, verify_integrity: bool = False
) -> dict[str, Any]: ...
def read_microcontroller_manifest_tool(manifest_path: str) -> dict[str, Any]: ...
def write_microcontroller_manifest_tool(
    log_directory: str, controller_id: int, controller_name: str, modules: list[dict[str, Any]]
) -> dict[str, Any]: ...
def discover_microcontroller_data_tool(root_directory: str) -> dict[str, Any]: ...
def _resolve_log_dir_roots(log_dir_paths: list[Path]) -> dict[Path, Path]: ...
