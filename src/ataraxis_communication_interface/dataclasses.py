"""Provides data classes, configuration structures, and helper functions for managing microcontroller log manifests and
extraction configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path


MICROCONTROLLER_MANIFEST_FILENAME: str = "microcontroller_manifest.yaml"
"""The filename used for microcontroller log manifest files within DataLogger output directories."""

EXTRACTION_CONFIGURATION_FILENAME: str = "extraction_configuration.yaml"
"""The default filename used for extraction configuration files."""


@dataclass(frozen=True, slots=True)
class ModuleSourceData:
    """Stores the identification data for a single hardware module registered in a log manifest."""

    module_type: int
    """The type (family) code of the hardware module."""
    module_id: int
    """The unique identifier code of the hardware module."""
    name: str
    """A colloquial human-readable name for the hardware module (e.g., 'encoder', 'lick_sensor')."""


@dataclass(frozen=True, slots=True)
class MicroControllerSourceData:
    """Stores the identification data for a single microcontroller registered in a log manifest.

    Each entry corresponds to one MicroControllerInterface instance that logs communication data to the same DataLogger
    output directory. The ``modules`` tuple enumerates all hardware module interfaces bound to this controller.
    """

    id: int
    """The controller_id used by the MicroControllerInterface instance when logging to the DataLogger."""
    name: str
    """A colloquial human-readable name for the microcontroller (e.g., 'actor_controller')."""
    modules: tuple[ModuleSourceData, ...]
    """The hardware modules managed by this microcontroller, identified by their type, id, and name."""


@dataclass
class MicroControllerManifest(YamlConfig):
    """Stores microcontroller source identification data for all MicroControllerInterface instances sharing a
    DataLogger.

    Each entry in the ``controllers`` list corresponds to one MicroControllerInterface instance that logs data to the
    same DataLogger output directory. The manifest file enables downstream tools to identify which log archives were
    produced by ataraxis-communication-interface and to associate controller IDs with human-readable names.
    """

    controllers: list[MicroControllerSourceData] = field(default_factory=list)
    """The list of microcontroller source entries registered in this manifest."""


@dataclass(frozen=True, slots=True)
class ModuleExtractionConfig:
    """Defines extraction parameters for a single hardware module.

    Notes:
        Event codes must be globally unique within each module -- the same event code must not be reused with
        different semantics across commands. This invariance is enforced by the microcontroller firmware and enables
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


def write_microcontroller_manifest(
    log_directory: Path,
    controller_id: int,
    controller_name: str,
    modules: tuple[ModuleSourceData, ...],
) -> None:
    """Writes or updates the microcontroller manifest file in the specified log directory.

    If the manifest file already exists (another MicroControllerInterface instance has already registered), reads the
    existing manifest, appends the new controller entry, and writes it back. Otherwise, creates a new manifest with a
    single entry.

    Args:
        log_directory: The path to the DataLogger output directory where the manifest file is stored.
        controller_id: The controller_id of the MicroControllerInterface instance to register.
        controller_name: The colloquial human-readable name for the microcontroller.
        modules: A tuple of ModuleSourceData instances describing the hardware modules managed by this controller.
    """
    manifest_path = log_directory / MICROCONTROLLER_MANIFEST_FILENAME

    # Reads the existing manifest if one has already been written by another MicroControllerInterface instance sharing
    # this DataLogger.
    manifest = (
        MicroControllerManifest.from_yaml(file_path=manifest_path)
        if manifest_path.exists()
        else MicroControllerManifest()
    )

    # Appends the new controller entry and writes the updated manifest back to disk.
    manifest.controllers.append(MicroControllerSourceData(id=controller_id, name=controller_name, modules=modules))
    manifest.to_yaml(file_path=manifest_path)


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
