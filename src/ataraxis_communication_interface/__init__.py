"""Provides the centralized interface for exchanging commands and data between Arduino and Teensy microcontrollers and
host-computers.

See the `documentation <https://ataraxis-communication-interface-api.netlify.app/>`_ for the description of
available assets. See the `source code repository <https://github.com/Sun-Lab-NBB/ataraxis-communication-interface>`_
for more details.

Authors: Ivan Kondratyev (Inkaros), Jacob Groner (Jgroner11)
"""

from .dataclasses import (
    EXTRACTION_CONFIGURATION_FILENAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    ModuleSourceData,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    MicroControllerManifest,
    MicroControllerSourceData,
    ControllerExtractionConfig,
    create_extraction_config,
    write_microcontroller_manifest,
)
from .communication import (
    ModuleData,
    ModuleState,
    MQTTCommunication,
)
from .log_processing import run_log_processing_pipeline
from .microcontroller_interface import (
    ModuleInterface,
    MicroControllerInterface,
)

__all__ = [
    "EXTRACTION_CONFIGURATION_FILENAME",
    "MICROCONTROLLER_MANIFEST_FILENAME",
    "ControllerExtractionConfig",
    "ExtractionConfig",
    "KernelExtractionConfig",
    "MQTTCommunication",
    "MicroControllerInterface",
    "MicroControllerManifest",
    "MicroControllerSourceData",
    "ModuleData",
    "ModuleExtractionConfig",
    "ModuleInterface",
    "ModuleSourceData",
    "ModuleState",
    "create_extraction_config",
    "run_log_processing_pipeline",
    "write_microcontroller_manifest",
]
