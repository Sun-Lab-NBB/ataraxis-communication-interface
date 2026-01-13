"""Provides the centralized interface for exchanging commands and data between Arduino and Teensy microcontrollers
and host-computers.

See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface for more details.
API documentation: https://ataraxis-communication-interface-api.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner
"""

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
from .microcontroller_interface import (
    ModuleInterface,
    ExtractedModuleData,
    ExtractedMessageData,
    MicroControllerInterface,
    print_microcontroller_ids,
    extract_logged_hardware_module_data,
)

__all__ = [
    "ExtractedMessageData",
    "ExtractedModuleData",
    "MQTTCommunication",
    "MicroControllerInterface",
    "ModuleData",
    "ModuleInterface",
    "ModuleState",
    "check_mqtt_connectivity",
    "extract_logged_hardware_module_data",
    "print_microcontroller_ids",
    "run_mcp_server",
    "run_server",
]
