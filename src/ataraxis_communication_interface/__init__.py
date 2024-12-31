"""This library enables Python clients and Unity game engine to interface with custom hardware modules running on
Arduino or Teensy microcontrollers.

See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface for more details.
API documentation: https://ataraxis-communication-interface-api.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner
"""

from .communication import (
    ModuleData,
    ModuleState,
    ModuleParameters,
    UnityCommunication,
    OneOffModuleCommand,
    DequeueModuleCommand,
    RepeatedModuleCommand,
)
from .microcontroller_interface import (
    ModuleInterface,
    MicroControllerInterface,
)

__all__ = [
    "MicroControllerInterface",
    "ModuleInterface",
    "ModuleState",
    "ModuleData",
    "ModuleParameters",
    "RepeatedModuleCommand",
    "OneOffModuleCommand",
    "UnityCommunication",
    "DequeueModuleCommand",
]
