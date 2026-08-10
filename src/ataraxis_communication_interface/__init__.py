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
from .orchestration import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    CONTROLLER_EXTRACTION_JOB_CORES,
    JobSource,
    JobUniverse,
    execute_job,
    resolve_jobs,
    find_module_paths,
    parse_module_path,
    resolve_kernel_path,
    resolve_module_path,
    run_log_processing_pipeline,
    estimate_archive_job_memory_mb,
)
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
    "CONTROLLER_EXTRACTION_JOB_CORES",
    "CONTROLLER_EXTRACTION_JOB_NAME",
    "EXTRACTION_CONFIGURATION_FILENAME",
    "MAXIMUM_CUSTOM_STATUS_CODE",
    "MICROCONTROLLER_MANIFEST_FILENAME",
    "MINIMUM_CUSTOM_STATUS_CODE",
    "CommunicationStatusCodes",
    "ControllerExtractionConfig",
    "ExtractionConfig",
    "JobSource",
    "JobUniverse",
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
    "estimate_archive_job_memory_mb",
    "execute_job",
    "find_module_paths",
    "parse_module_path",
    "resolve_jobs",
    "resolve_kernel_path",
    "resolve_module_path",
    "run_log_processing_pipeline",
    "write_microcontroller_manifest",
]
