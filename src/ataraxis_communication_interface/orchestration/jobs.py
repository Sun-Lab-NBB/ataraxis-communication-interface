"""Provides the job identity constants, the output layout names and resolvers, and the descriptor and sizing records
every consumer that schedules microcontroller data extraction exchanges.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import fields, dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingTracker

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CONTROLLER_EXTRACTION_JOB_NAME: str = "microcontroller_data_extraction"
"""The job name under which microcontroller data extraction is registered in a ProcessingTracker.

Notes:
    The value is hashed into every persisted job identifier, so changing the string invalidates every identifier a
    tracker already holds and every identifier a scheduler derived independently.
"""

_MODULE_FILE_FIELD_COUNT: int = 5
"""The underscore-separated fields a module output filename holds, which is the prefix, the source identifier, the
module marker, the module type, and the module identifier."""

_KERNEL_FILE_FIELD_COUNT: int = 3
"""The underscore-separated fields a kernel output filename holds, which is the prefix, the source identifier, and the
kernel marker."""


class OutputLayout(StrEnum):
    """Defines the filesystem names an extraction job writes its tracker and its output files under."""

    DIRECTORY_NAME = "microcontroller_data"
    """The subdirectory created under a caller's output path for the tracker and the extracted files."""
    TRACKER_FILENAME = "microcontroller_processing_tracker.yaml"
    """The processing tracker file recording the outcome of every job writing to one directory."""
    FILE_PREFIX = "controller_"
    """The prefix of every output file an extraction job writes."""
    MODULE_INFIX = "_module_"
    """The infix separating the controller source identifier from the module type and identifier codes."""
    KERNEL_INFIX = "_kernel"
    """The infix marking an output file as holding kernel messages."""
    FILE_SUFFIX = ".feather"
    """The filename suffix of every output (Arrow IPC) file an extraction job writes."""


@dataclass(frozen=True, slots=True)
class JobDescriptor:
    """Describes one microcontroller data extraction job, addressed by the single log archive it reads.

    Notes:
        Every field is a path, a string, or an integer, so an instance pickles into a spawned worker and crosses a
        scheduler boundary or a tool payload unchanged.
    """

    log_directory: Path
    """The path to the DataLogger output directory whose tree holds the log archive."""
    archive_path: Path
    """The path to the .npz log archive this job reads."""
    output_directory: Path
    """The path to the directory this job writes its output files into."""
    config_path: Path
    """The path to the ExtractionConfig .yaml file naming the events this job extracts."""
    tracker_path: Path
    """The path to the ProcessingTracker file that records this job's outcome."""
    job_name: str
    """The tracker job name this job is registered under."""
    job_id: str
    """The unique hexadecimal identifier of this job in its tracker."""
    source_id: str
    """The identifier of the controller source whose archive this job reads."""
    core_weight: int
    """The cores this job occupies while it runs, which is the width of the extraction pool its body opens once the
    job holds more than one core."""

    @classmethod
    def for_archive(
        cls,
        archive_path: Path,
        output_directory: Path,
        config_path: Path,
        tracker_path: Path,
        source_id: str,
        log_directory: Path | None = None,
        core_weight: int = 1,
    ) -> JobDescriptor:
        """Builds a descriptor for one archive an external scheduler has already resolved.

        Notes:
            Derives the job identifier as this library's own preparation does, so one built here addresses the same
            tracker entry.

        Args:
            archive_path: The path to the .npz log archive the job reads.
            output_directory: The path to the directory the job writes its output files into.
            config_path: The path to the ExtractionConfig .yaml file naming the events the job extracts.
            tracker_path: The path to the ProcessingTracker file that records the job's outcome.
            source_id: The identifier of the controller source whose archive the job reads.
            log_directory: The path to the DataLogger output directory holding the archive. Leaving this unset uses
                the archive's own parent directory.
            core_weight: The cores the job occupies while it runs.

        Returns:
            The built descriptor.
        """
        return cls(
            log_directory=log_directory if log_directory is not None else archive_path.parent,
            archive_path=archive_path,
            output_directory=output_directory,
            config_path=config_path,
            tracker_path=tracker_path,
            job_name=CONTROLLER_EXTRACTION_JOB_NAME,
            job_id=ProcessingTracker.generate_job_id(job_name=CONTROLLER_EXTRACTION_JOB_NAME, specifier=source_id),
            source_id=source_id,
            core_weight=core_weight,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> JobDescriptor:
        """Reconstructs a descriptor from the mapping a caller received across a tool boundary.

        Args:
            mapping: The mapping to read, carrying every field name to_mapping writes.

        Returns:
            The reconstructed descriptor.

        Raises:
            ValueError: If a required key is absent, or if a value cannot be read as the type its field declares.
        """
        field_names = tuple(field.name for field in fields(cls))
        missing_keys = [name for name in field_names if name not in mapping]

        if missing_keys:
            message = (
                f"Unable to read a microcontroller data extraction job descriptor from the supplied mapping. The "
                f"following required keys are absent: {', '.join(sorted(missing_keys))}. A descriptor mapping "
                f"carries every key the descriptor writes: {', '.join(field_names)}."
            )
            console.error(message=message, error=ValueError)

        try:
            return cls(
                log_directory=Path(mapping["log_directory"]),
                archive_path=Path(mapping["archive_path"]),
                output_directory=Path(mapping["output_directory"]),
                config_path=Path(mapping["config_path"]),
                tracker_path=Path(mapping["tracker_path"]),
                job_name=str(mapping["job_name"]),
                job_id=str(mapping["job_id"]),
                source_id=str(mapping["source_id"]),
                core_weight=int(mapping["core_weight"]),
            )
        except (TypeError, ValueError) as error:
            message = (
                f"Unable to read a microcontroller data extraction job descriptor from the supplied mapping. One of "
                f"its values cannot be read as the type its field declares: {error}."
            )
            console.error(message=message, error=ValueError)

            # Satisfies ruff RET503. console.error() is NoReturn, so this line never executes.
            raise  # pragma: no cover

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the tracker path and job identifier pair that identifies this job across the batch."""
        return str(self.tracker_path), self.job_id

    def to_mapping(self) -> dict[str, str | int]:
        """Renders this descriptor as the flat mapping the interface layer exchanges.

        Notes:
            Every value is a string or an integer, so the mapping reconstructs through from_mapping without loss.

        Returns:
            The descriptor's fields keyed by their field names, with every path rendered as a string.
        """
        return {
            "log_directory": str(self.log_directory),
            "archive_path": str(self.archive_path),
            "output_directory": str(self.output_directory),
            "config_path": str(self.config_path),
            "tracker_path": str(self.tracker_path),
            "job_name": self.job_name,
            "job_id": self.job_id,
            "source_id": self.source_id,
            "core_weight": self.core_weight,
        }


@dataclass(frozen=True, slots=True)
class JobSizing:
    """Describes the resources one job receives, as one sizing pass resolved them."""

    cores: int
    """The CPU cores the job occupies while it runs, which is the width of the extraction pool its body opens once it
    holds more than one core."""
    memory_mb: int
    """The memory the job occupies at its peak, in megabytes."""


def generate_job_ids(source_ids: Sequence[str]) -> dict[str, str]:
    """Generates the processing job identifier of every requested controller source.

    Args:
        source_ids: The controller source identifiers to generate job identifiers for.

    Returns:
        The generated hexadecimal job identifier of each source, keyed by that source identifier.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=CONTROLLER_EXTRACTION_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }


def resolve_output_directory(output_directory: Path) -> Path:
    """Resolves the subdirectory the extraction output and its tracker are written into.

    Args:
        output_directory: The root output directory the caller nominated.

    Returns:
        The path to the library's own subdirectory under the nominated root.
    """
    return output_directory / OutputLayout.DIRECTORY_NAME


def resolve_tracker_path(output_directory: Path) -> Path:
    """Resolves the path of the processing tracker recording the outcome of every job writing to a directory.

    Args:
        output_directory: The directory the extraction jobs write their output into.

    Returns:
        The path to the tracker file.
    """
    return output_directory / OutputLayout.TRACKER_FILENAME


def resolve_module_path(output_directory: Path, source_id: str, module_type: int, module_id: int) -> Path:
    """Resolves the path of the file holding the target module's extracted messages.

    Args:
        output_directory: The directory the extraction job writes its output into.
        source_id: The identifier of the controller source that manages the module.
        module_type: The type (family) code of the hardware module.
        module_id: The unique identifier code of the hardware module.

    Returns:
        The path to the module's message file.
    """
    filename = (
        f"{OutputLayout.FILE_PREFIX}{source_id}{OutputLayout.MODULE_INFIX}{module_type}_{module_id}"
        f"{OutputLayout.FILE_SUFFIX}"
    )
    return output_directory / filename


def resolve_kernel_path(output_directory: Path, source_id: str) -> Path:
    """Resolves the path of the file holding the target source's extracted kernel messages.

    Args:
        output_directory: The directory the extraction job writes its output into.
        source_id: The identifier of the controller source whose kernel messages the file holds.

    Returns:
        The path to the source's kernel message file.
    """
    filename = f"{OutputLayout.FILE_PREFIX}{source_id}{OutputLayout.KERNEL_INFIX}{OutputLayout.FILE_SUFFIX}"
    return output_directory / filename


def find_module_paths(data_directory: Path) -> list[Path]:
    """Discovers every module output file an extraction job wrote into the target directory.

    Args:
        data_directory: The directory the extraction jobs write their output into.

    Returns:
        The paths to every module output file the directory holds, sorted by path, and an empty list when the
        directory does not exist.
    """
    if not data_directory.is_dir():
        return []

    pattern = f"{OutputLayout.FILE_PREFIX}*{OutputLayout.MODULE_INFIX}*{OutputLayout.FILE_SUFFIX}"
    return sorted(data_directory.glob(pattern))


def find_kernel_paths(data_directory: Path) -> list[Path]:
    """Discovers every kernel output file an extraction job wrote into the target directory.

    Args:
        data_directory: The directory the extraction jobs write their output into.

    Returns:
        The paths to every kernel output file the directory holds, sorted by path, and an empty list when the
        directory does not exist.
    """
    if not data_directory.is_dir():
        return []

    pattern = f"{OutputLayout.FILE_PREFIX}*{OutputLayout.KERNEL_INFIX}{OutputLayout.FILE_SUFFIX}"
    return sorted(data_directory.glob(pattern))


def parse_module_path(file_path: Path) -> tuple[int, int, int]:
    """Reads the controller source, the module type, and the module identifier a module output filename encodes.

    Notes:
        Inverts resolve_module_path().

    Args:
        file_path: The path to the module output file to read the identity of.

    Returns:
        The controller source identifier, the module type code, and the module identifier code, in that order.

    Raises:
        ValueError: If the filename does not follow the module output naming convention, or if any of its three
            identity fields is not an integer.
    """
    parts = file_path.stem.split("_")

    if (
        len(parts) != _MODULE_FILE_FIELD_COUNT
        or f"{parts[0]}_" != OutputLayout.FILE_PREFIX
        or f"_{parts[2]}_" != OutputLayout.MODULE_INFIX
        or not all(field.isdigit() for field in (parts[1], parts[3], parts[4]))
    ):
        message = (
            f"Unable to parse the module output filename '{file_path.name}'. The filename does not follow the "
            f"'{OutputLayout.FILE_PREFIX}{{source_id}}{OutputLayout.MODULE_INFIX}{{module_type}}_{{module_id}}"
            f"{OutputLayout.FILE_SUFFIX}' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1]), int(parts[3]), int(parts[4])


def parse_kernel_path(file_path: Path) -> int:
    """Reads the controller source a kernel output filename encodes.

    Notes:
        Inverts resolve_kernel_path().

    Args:
        file_path: The path to the kernel output file to read the identity of.

    Returns:
        The controller source identifier.

    Raises:
        ValueError: If the filename does not follow the kernel output naming convention, or if its source identifier
            is not an integer.
    """
    parts = file_path.stem.split("_")

    if (
        len(parts) != _KERNEL_FILE_FIELD_COUNT
        or f"{parts[0]}_" != OutputLayout.FILE_PREFIX
        or f"_{parts[2]}" != OutputLayout.KERNEL_INFIX
        or not parts[1].isdigit()
    ):
        message = (
            f"Unable to parse the kernel output filename '{file_path.name}'. The filename does not follow the "
            f"'{OutputLayout.FILE_PREFIX}{{source_id}}{OutputLayout.KERNEL_INFIX}{OutputLayout.FILE_SUFFIX}' "
            f"naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1])
