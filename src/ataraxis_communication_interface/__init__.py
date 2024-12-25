"""This library provides classes and methods that enable bidirectional communication between project Ataraxis systems.

See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer for more details.
API documentation: https://ataraxis-transport-layer-api-docs.netlify.app/
Author: Ivan Kondratyev (Inkaros)
"""

from .microcontroller_interface import MicroControllerInterface

__all__ = [
    "MicroControllerInterface",
]
