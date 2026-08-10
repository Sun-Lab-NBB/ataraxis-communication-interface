from typing import Any

from .mcp_instance import mcp as mcp
from ..microcontroller import (
    ExtractionConfig as ExtractionConfig,
    KernelExtractionConfig as KernelExtractionConfig,
    ModuleExtractionConfig as ModuleExtractionConfig,
    MicroControllerManifest as MicroControllerManifest,
    ControllerExtractionConfig as ControllerExtractionConfig,
)

def read_extraction_config_tool(config_path: str) -> dict[str, Any]: ...
def write_extraction_config_tool(config_path: str, controllers: list[dict[str, Any]]) -> dict[str, Any]: ...
def validate_extraction_config_tool(config_path: str, manifest_path: str | None = None) -> dict[str, Any]: ...
