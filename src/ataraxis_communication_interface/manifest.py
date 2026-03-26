"""Provides data classes and a helper function for managing microcontroller log manifest files."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path


MICROCONTROLLER_MANIFEST_FILENAME: str = "microcontroller_manifest.yaml"
"""The filename used for microcontroller log manifest files within DataLogger output directories."""


@dataclass(frozen=True, slots=True)
class ModuleSourceData:
    """Stores the identification data for a single hardware module registered in a log manifest."""

    type_id: int = 0
    """The combined type+id code (type << 8 | id) used by the module when communicating with the PC."""
    name: str = ""
    """A colloquial human-readable name for the hardware module (e.g., 'encoder', 'lick_sensor')."""


@dataclass(frozen=True, slots=True)
class MicroControllerSourceData:
    """Stores the identification data for a single microcontroller registered in a log manifest.

    Each entry corresponds to one MicroControllerInterface instance that logs communication data to the same DataLogger
    output directory. The ``modules`` tuple enumerates all hardware module interfaces bound to this controller.
    """

    id: int = 0
    """The controller_id used by the MicroControllerInterface instance when logging to the DataLogger."""
    name: str = ""
    """A colloquial human-readable name for the microcontroller (e.g., 'actor_controller')."""
    modules: tuple[ModuleSourceData, ...] = ()
    """The hardware modules managed by this microcontroller, identified by their combined type+id codes and names."""


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
