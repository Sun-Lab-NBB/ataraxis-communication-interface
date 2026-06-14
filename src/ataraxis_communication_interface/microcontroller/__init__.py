"""Provides the microcontroller interface classes, configuration dataclasses, and the log processing pipeline."""

from .interface import ModuleInterface, MicroControllerInterface
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
from .log_processing import run_log_processing_pipeline

__all__ = [
    "EXTRACTION_CONFIGURATION_FILENAME",
    "MICROCONTROLLER_MANIFEST_FILENAME",
    "ControllerExtractionConfig",
    "ExtractionConfig",
    "KernelExtractionConfig",
    "MicroControllerInterface",
    "MicroControllerManifest",
    "MicroControllerSourceData",
    "ModuleExtractionConfig",
    "ModuleInterface",
    "ModuleSourceData",
    "create_extraction_config",
    "run_log_processing_pipeline",
    "write_microcontroller_manifest",
]
