"""This library provides classes and methods for interfacing with other project Ataraxis systems and platforms.

See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface for more details.
API documentation: https://ataraxis-communication-interface-api-docs.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner
"""

from .microcontroller_interface import MicroControllerInterface, ModuleInterface

__all__ = [
    "MicroControllerInterface",
    "ModuleInterface"
]
