"""Provides the microcontroller interface classes, the configuration dataclasses, and the log data extraction
algorithm.
"""

from .interface import ModuleInterface, MicroControllerInterface, evaluate_port
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
from .extracted_data import (
    ExtractedDataColumns,
    get_event_data,
    partition_events,
    get_event_timestamps,
    build_message_dataframe,
)
from .log_processing import (
    ExtractedMessages,
    ExtractedModuleData,
    ExtractedControllerData,
    extract_logged_microcontroller_data,
)

__all__ = [
    "EXTRACTION_CONFIGURATION_FILENAME",
    "MICROCONTROLLER_MANIFEST_FILENAME",
    "ControllerExtractionConfig",
    "ExtractedControllerData",
    "ExtractedDataColumns",
    "ExtractedMessages",
    "ExtractedModuleData",
    "ExtractionConfig",
    "KernelExtractionConfig",
    "MicroControllerInterface",
    "MicroControllerManifest",
    "MicroControllerSourceData",
    "ModuleExtractionConfig",
    "ModuleInterface",
    "ModuleSourceData",
    "build_message_dataframe",
    "create_extraction_config",
    "evaluate_port",
    "extract_logged_microcontroller_data",
    "get_event_data",
    "get_event_timestamps",
    "partition_events",
    "write_microcontroller_manifest",
]
