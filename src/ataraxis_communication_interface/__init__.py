"""Provides the centralized interface for exchanging commands and data between Arduino and Teensy microcontrollers
and host-computers.

See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface for more details.
API documentation: https://ataraxis-communication-interface-api.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner
"""

from .dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    EXTRACTION_CONFIGURATION_FILENAME,
    ModuleSourceData,
    MicroControllerManifest,
    MicroControllerSourceData,
    ExtractionConfig,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    ControllerExtractionConfig,
    write_microcontroller_manifest,
    create_extraction_config,
)
from .communication import (
    ModuleData,
    ModuleState,
    MQTTCommunication,
)
from .log_processing import (
    FEATHER_SUFFIX,
    KERNEL_FEATHER_INFIX,
    MODULE_FEATHER_INFIX,
    CONTROLLER_FEATHER_PREFIX,
    ExtractedModuleData,
    ExtractedMessageData,
    extract_log_data,
    run_log_processing_pipeline,
)
from .microcontroller_interface import (
    ModuleInterface,
    MicroControllerInterface,
)

__all__ = [
    "CONTROLLER_FEATHER_PREFIX",
    "EXTRACTION_CONFIGURATION_FILENAME",
    "FEATHER_SUFFIX",
    "KERNEL_FEATHER_INFIX",
    "MICROCONTROLLER_MANIFEST_FILENAME",
    "MODULE_FEATHER_INFIX",
    "ControllerExtractionConfig",
    "ExtractedMessageData",
    "ExtractedModuleData",
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
    "extract_log_data",
    "run_log_processing_pipeline",
    "write_microcontroller_manifest",
]
