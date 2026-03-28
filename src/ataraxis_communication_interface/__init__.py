"""Provides the centralized interface for exchanging commands and data between Arduino and Teensy microcontrollers
and host-computers.

See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface for more details.
API documentation: https://ataraxis-communication-interface-api.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner
"""

from .manifest import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData,
    MicroControllerManifest,
    MicroControllerSourceData,
    write_microcontroller_manifest,
)
from .mcp_server import (
    run_server,
    run_mcp_server,
)
from .communication import (
    ModuleData,
    ModuleState,
    MQTTCommunication,
    check_mqtt_connectivity,
)
from .log_processing import (
    ExtractedModuleData,
    ExtractedMessageData,
    extract_log_data,
    run_log_processing_pipeline,
)
from .microcontroller_interface import (
    ModuleInterface,
    MicroControllerInterface,
    print_microcontroller_ids,
)

__all__ = [
    "MICROCONTROLLER_MANIFEST_FILENAME",
    "ExtractedMessageData",
    "ExtractedModuleData",
    "MQTTCommunication",
    "MicroControllerInterface",
    "MicroControllerManifest",
    "MicroControllerSourceData",
    "ModuleData",
    "ModuleInterface",
    "ModuleSourceData",
    "ModuleState",
    "check_mqtt_connectivity",
    "extract_log_data",
    "print_microcontroller_ids",
    "run_log_processing_pipeline",
    "run_mcp_server",
    "run_server",
    "write_microcontroller_manifest",
]
