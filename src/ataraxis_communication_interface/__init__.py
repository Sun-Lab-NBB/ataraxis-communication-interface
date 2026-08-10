"""Provides the centralized interface for exchanging commands and data between Arduino and Teensy microcontrollers and
host-computers.

See the `documentation <https://ataraxis-communication-interface-api.netlify.app/>`_ for the description of
available assets. See the `source code repository <https://github.com/Sun-Lab-NBB/ataraxis-communication-interface>`_
for more details.

Authors: Ivan Kondratyev (Inkaros), Jacob Groner (Jgroner11)
"""

from .communication import (
    ModuleData,
    ModuleState,
    MQTTCommunication,
)
from .orchestration import run_log_processing_pipeline
from .microcontroller import (
    MAXIMUM_CUSTOM_STATUS_CODE,
    MINIMUM_CUSTOM_STATUS_CODE,
    EXTRACTION_CONFIGURATION_FILENAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleInterface,
    ExtractionConfig,
    ModuleSourceData,
    KernelStatusCodes,
    ModuleStatusCodes,
    KernelCommandCodes,
    TransportStatusCodes,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    MicroControllerManifest,
    CommunicationStatusCodes,
    MicroControllerInterface,
    MicroControllerSourceData,
    ControllerExtractionConfig,
    create_extraction_config,
    write_microcontroller_manifest,
)

__all__ = [
    "EXTRACTION_CONFIGURATION_FILENAME",
    "MAXIMUM_CUSTOM_STATUS_CODE",
    "MICROCONTROLLER_MANIFEST_FILENAME",
    "MINIMUM_CUSTOM_STATUS_CODE",
    "CommunicationStatusCodes",
    "ControllerExtractionConfig",
    "ExtractionConfig",
    "KernelCommandCodes",
    "KernelExtractionConfig",
    "KernelStatusCodes",
    "MQTTCommunication",
    "MicroControllerInterface",
    "MicroControllerManifest",
    "MicroControllerSourceData",
    "ModuleData",
    "ModuleExtractionConfig",
    "ModuleInterface",
    "ModuleSourceData",
    "ModuleState",
    "ModuleStatusCodes",
    "TransportStatusCodes",
    "create_extraction_config",
    "run_log_processing_pipeline",
    "write_microcontroller_manifest",
]
