"""Provides configuration data classes for the microcontroller log data extraction pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import YamlConfig

from .manifest import MICROCONTROLLER_MANIFEST_FILENAME, MicroControllerManifest

if TYPE_CHECKING:
    from pathlib import Path


EXTRACTION_CONFIG_FILENAME: str = "extraction_config.yaml"
"""The default filename used for extraction configuration files."""


@dataclass(frozen=True, slots=True)
class ModuleExtractionConfig:
    """Defines extraction parameters for a single hardware module.

    Notes:
        Event codes must be globally unique within each module -- the same event code must not be reused with
        different semantics across commands. This invariant is enforced by the microcontroller firmware and enables
        extraction to filter by event code alone without requiring command code disambiguation.
    """

    module_type: int
    """The type (family) code of the hardware module."""
    module_id: int
    """The unique identifier code of the hardware module."""
    event_codes: tuple[int, ...]
    """The event codes to extract. Must not be empty. Each event code must be unique within this module."""


@dataclass(frozen=True, slots=True)
class KernelExtractionConfig:
    """Defines extraction parameters for kernel messages.

    Notes:
        Event codes must be globally unique within the kernel -- the same event code must not be reused with
        different semantics across commands.
    """

    event_codes: tuple[int, ...]
    """The kernel event codes to extract. Must not be empty. Each event code must be unique within the kernel."""


@dataclass(frozen=True, slots=True)
class ControllerExtractionConfig:
    """Defines extraction parameters for a single microcontroller source."""

    controller_id: int
    """The controller_id used by the MicroControllerInterface when logging."""
    modules: tuple[ModuleExtractionConfig, ...]
    """The hardware modules to extract data for."""
    kernel: KernelExtractionConfig | None
    """Kernel extraction settings, or None to skip kernel extraction for this controller."""


@dataclass
class ExtractionConfig(YamlConfig):
    """Defines the complete extraction configuration for microcontroller log processing.

    Specifies which controllers, modules, and events to extract from log archives. Processing requires a valid
    configuration file with non-empty event codes for every module and kernel entry. Use the CLI
    ``axci config create`` command or the ``create_extraction_config_tool`` MCP tool to generate a precursor
    configuration from an existing microcontroller manifest, then fill in the event codes before processing.
    """

    controllers: list[ControllerExtractionConfig]
    """The list of controller extraction configurations."""


def create_extraction_config(manifest_path: Path) -> ExtractionConfig:
    """Generates a precursor extraction configuration from a microcontroller manifest.

    Reads the manifest file and populates a ControllerExtractionConfig entry for each registered controller
    with placeholder empty event codes. The user must fill in the actual event codes for each module and kernel
    entry before the configuration is usable for processing.

    Args:
        manifest_path: The path to the microcontroller_manifest.yaml file.

    Returns:
        An ExtractionConfig instance with all controllers and modules populated but with empty event codes
        that must be filled in by the user.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the manifest contains no controller entries.
    """
    if not manifest_path.exists() or not manifest_path.is_file():
        message = (
            f"Unable to create extraction config from '{manifest_path}'. The path does not exist or is not a file."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest = MicroControllerManifest.from_yaml(file_path=manifest_path)

    if not manifest.controllers:
        message = (
            f"Unable to create extraction config from '{manifest_path}'. The "
            f"{MICROCONTROLLER_MANIFEST_FILENAME} contains no controller entries."
        )
        console.error(message=message, error=ValueError)

    controller_configs: list[ControllerExtractionConfig] = []
    for controller in manifest.controllers:
        # Creates placeholder module configs with empty event codes for the user to fill in.
        module_configs = tuple(
            ModuleExtractionConfig(
                module_type=source_module.module_type,
                module_id=source_module.module_id,
                event_codes=(),
            )
            for source_module in controller.modules
        )

        controller_configs.append(
            ControllerExtractionConfig(
                controller_id=controller.id,
                modules=module_configs,
                kernel=None,
            )
        )

    return ExtractionConfig(controllers=controller_configs)
