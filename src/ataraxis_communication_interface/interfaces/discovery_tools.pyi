from typing import Any
from pathlib import Path

from .responses import (
    page_fields as page_fields,
    project_item as project_item,
    resolve_page as resolve_page,
    item_breakdown as item_breakdown,
    reject_unknown as reject_unknown,
    resolve_detail_limit as resolve_detail_limit,
)
from .mcp_instance import mcp as mcp
from ..communication import MQTTCommunication as MQTTCommunication
from ..microcontroller import (
    MICROCONTROLLER_MANIFEST_FILENAME as MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData as ModuleSourceData,
    MicroControllerManifest as MicroControllerManifest,
    discover_microcontrollers as discover_microcontrollers,
    write_microcontroller_manifest as write_microcontroller_manifest,
)

_SOURCE_AXES: tuple[str, ...]
_SOURCE_SEMI_DETAIL_FIELDS: tuple[str, ...]
_SOURCE_DETAIL_FIELDS: tuple[str, ...]

def list_microcontrollers_tool(baudrate: int = 115200) -> str: ...
def check_mqtt_broker_tool(host: str = "127.0.0.1", port: int = 1883) -> str: ...
def assemble_log_archives_tool(
    log_directory: str, *, remove_sources: bool = True, verify_integrity: bool = False
) -> dict[str, Any]: ...
def read_microcontroller_manifest_tool(manifest_path: str) -> dict[str, Any]: ...
def write_microcontroller_manifest_tool(
    log_directory: str, controller_id: int, controller_name: str, modules: list[dict[str, Any]]
) -> dict[str, Any]: ...
def discover_microcontroller_data_tool(
    root_directory: str,
    source_ids: list[str] | None = None,
    name: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def _resolve_log_directory_roots(log_directory_paths: list[Path]) -> dict[Path, Path]: ...
