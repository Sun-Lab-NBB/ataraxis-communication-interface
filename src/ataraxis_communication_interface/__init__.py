"""This library enables interfacing with custom hardware modules running on Arduino or Teensy microcontrollers through
Python interface clients.

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
    "MQTTCommunication",
    "DequeueModuleCommand",
]
