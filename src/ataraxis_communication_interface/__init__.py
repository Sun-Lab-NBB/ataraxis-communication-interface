"""This library allows interfacing with custom hardware modules running on Arduino or Teensy microcontrollers and
exchanging data within and between PCs.

See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface for more details.
API documentation: https://ataraxis-communication-interface-api.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner
"""

from .communication import (
    ModuleData,
    ModuleState,
    ModuleParameters,
    MQTTCommunication,
    OneOffModuleCommand,
    DequeueModuleCommand,
    RepeatedModuleCommand,
)
from .microcontroller_interface import (
    ModuleInterface,
    ExtractedModuleData,
    MicroControllerInterface,
    extract_logged_hardware_module_data,
)

__all__ = [
    "DequeueModuleCommand",
    "ExtractedModuleData",
    "MQTTCommunication",
    "MicroControllerInterface",
    "ModuleData",
    "ModuleInterface",
    "ModuleParameters",
    "ModuleState",
    "OneOffModuleCommand",
    "RepeatedModuleCommand",
    "extract_logged_hardware_module_data",
]
