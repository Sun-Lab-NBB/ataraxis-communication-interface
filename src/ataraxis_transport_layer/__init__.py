"""This library provides classes and methods that enable bidirectional communication between project Ataraxis systems.

See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer for more details.
API documentation: https://ataraxis-transport-layer-api-docs.netlify.app/
Author: Ivan Kondratyev (Inkaros)
"""

from .microcontroller import MicroControllerInterface
from .custom_interfaces import TTLInterface, LickInterface, BreakInterface, ValveInterface, EncoderInterface

__all__ = [
    "BreakInterface",
    "EncoderInterface",
    "LickInterface",
    "MicroControllerInterface",
    "TTLInterface",
    "ValveInterface",
]
