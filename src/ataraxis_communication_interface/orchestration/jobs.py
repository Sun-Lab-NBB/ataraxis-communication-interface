"""Provides the job identity constants, the batch job descriptor, and the manifest-derived job discovery shared by
every consumer that schedules microcontroller data extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    ProcessingTracker,
    index_marker_files,
    discover_marker_files,
)

from ..microcontroller import MICROCONTROLLER_MANIFEST_FILENAME, MicroControllerManifest

if TYPE_CHECKING:
    from pathlib import Path

EXTRACTION_JOB_NAME: str = "microcontroller_data_extraction"
"""The job name under which microcontroller data extraction is registered in a ProcessingTracker.

Notes:
    The value is hashed into every persisted job identifier, so changing the string invalidates every identifier a
    tracker already holds and every identifier a scheduler derived independently.
"""

TRACKER_FILENAME: str = "microcontroller_processing_tracker.yaml"
"""The name of the processing tracker file the pipeline places in its output directory."""

MICROCONTROLLER_DATA_DIRECTORY: str = "microcontroller_data"
"""The name of the subdirectory the pipeline creates under its output path for tracker and feather files."""

CONTROLLER_FEATHER_PREFIX: str = "controller_"
"""The prefix of every feather file an extraction job writes."""

MODULE_FEATHER_INFIX: str = "_module_"
"""The infix separating the controller source identifier from the module type and identifier codes."""

KERNEL_FEATHER_INFIX: str = "_kernel"
"""The infix identifying the feather file holding kernel messages."""

FEATHER_SUFFIX: str = ".feather"
"""The filename suffix of every feather (Arrow IPC) file an extraction job writes."""

_MODULE_FEATHER_FIELD_COUNT: int = 5
"""The underscore-separated fields a module feather filename carries, which are the controller prefix, the source
identifier, the module infix, the module type code, and the module identifier code."""


@dataclass(slots=True)
class PendingJob:
    """Describes a single data extraction job queued for batch execution.

    Notes:
        The core and memory weights are resolved from the job's own archive before dispatch, so admission weighs each
        job against the budgets at the size that archive actually demands.
    """

    log_directory: Path
    """The path to the DataLogger output directory whose tree holds the log archive."""
    output_directory: Path
    """The path to the directory the extracted feather files are written to."""
    tracker_path: Path
    """The path to the ProcessingTracker file that records this job's outcome."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    source_id: str
    """The identifier of the controller source whose archive this job reads."""
    config_path: Path
    """The path to the ExtractionConfig .yaml file naming the events this job extracts."""
    core_weight: int = 1
    """The cores this job occupies while it runs."""
    memory_mb: int = 0
    """The memory this job occupies while it runs, estimated from the archive it reads."""
    archive_path: Path | None = None
    """The path to the archive this job reads, resolved while the job is sized, or None when it did not resolve."""

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite tracker path and job identifier pair that identifies this job across the batch."""
        return str(self.tracker_path), self.job_id


def discover_microcontroller_jobs(log_directory: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the data extraction job universe and the subset backed by an archive on disk.

    Notes:
        The universe is a manifest fingerprint rather than an invocation fingerprint, so every invocation aligns a
        tracker against the same set and no invocation resets the jobs it did not request. The microcontroller manifest
        also gates the discovery, which keeps archives written by other libraries out of the resolved set.

    Args:
        log_directory: The root directory whose tree is searched for the microcontroller manifest and the log archives.

    Returns:
        The full job universe the manifest defines and the subset whose archives resolve to exactly one file, each as
        a list of job name and source identifier pairs.

    Raises:
        FileNotFoundError: If the log directory does not exist, is not a directory, or holds no microcontroller
            manifest.
        OSError: If any directory beneath the log directory cannot be read.
        ValueError: If the microcontroller manifest registers no controllers.
    """
    if not log_directory.is_dir():
        message = (
            f"Unable to discover microcontroller data extraction jobs in '{log_directory}'. The path does not exist "
            f"or is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    candidates = discover_marker_files(directory=log_directory, marker_name=MICROCONTROLLER_MANIFEST_FILENAME)
    if not candidates:
        message = (
            f"Unable to discover microcontroller data extraction jobs in '{log_directory}'. No "
            f"{MICROCONTROLLER_MANIFEST_FILENAME} was found. A microcontroller manifest is required to identify which "
            f"log archives were produced by ataraxis-communication-interface."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest = MicroControllerManifest.from_yaml(file_path=candidates[0])
    source_ids = sorted({str(controller.id) for controller in manifest.controllers})

    if not source_ids:
        message = (
            f"Unable to discover microcontroller data extraction jobs in '{log_directory}'. The "
            f"{MICROCONTROLLER_MANIFEST_FILENAME} at '{candidates[0]}' contains no controller entries."
        )
        console.error(message=message, error=ValueError)

    universe = [(EXTRACTION_JOB_NAME, source_id) for source_id in source_ids]

    # Indexes every source's archive in one pass, since the archive names are known once the manifest resolves. A
    # source whose name resolves to several archives spans several loggers, which is ambiguous rather than redundant,
    # so it is left out of the possible set alongside the sources holding no archive at all.
    archives = index_marker_files(
        directory=log_directory,
        marker_names=[f"{source_id}{LOG_ARCHIVE_SUFFIX}" for source_id in source_ids],
    )
    possible = [
        (EXTRACTION_JOB_NAME, source_id)
        for source_id in source_ids
        if len(archives[f"{source_id}{LOG_ARCHIVE_SUFFIX}"]) == 1
    ]

    return universe, possible


def generate_job_ids(source_ids: list[str]) -> dict[str, str]:
    """Generates the processing job identifier of every requested controller source.

    Args:
        source_ids: The controller source identifiers to generate job identifiers for.

    Returns:
        The generated hexadecimal job identifier of each source, keyed by that source identifier.
    """
    return {
        source_id: ProcessingTracker.generate_job_id(job_name=EXTRACTION_JOB_NAME, specifier=source_id)
        for source_id in source_ids
    }


def resolve_module_feather_path(output_directory: Path, source_id: str, module_type: int, module_id: int) -> Path:
    """Resolves the path of the feather file holding the target module's extracted messages.

    Args:
        output_directory: The directory the extraction job writes its output into.
        source_id: The identifier of the controller source that manages the module.
        module_type: The type (family) code of the hardware module.
        module_id: The unique identifier code of the hardware module.

    Returns:
        The path to the module's message feather file.
    """
    filename = f"{CONTROLLER_FEATHER_PREFIX}{source_id}{MODULE_FEATHER_INFIX}{module_type}_{module_id}{FEATHER_SUFFIX}"
    return output_directory / filename


def find_module_feathers(data_directory: Path) -> list[Path]:
    """Discovers every module feather file an extraction job wrote into the target directory.

    Args:
        data_directory: The directory the extraction jobs write their output into.

    Returns:
        The paths to every module feather file the directory holds, sorted by path, and an empty list when the
        directory does not exist.
    """
    if not data_directory.is_dir():
        return []
    return sorted(data_directory.glob(f"{CONTROLLER_FEATHER_PREFIX}*{MODULE_FEATHER_INFIX}*{FEATHER_SUFFIX}"))


def parse_module_feather_name(feather_path: Path) -> tuple[int, int, int]:
    """Reads the controller source, the module type, and the module identifier a module feather filename encodes.

    Notes:
        Inverts resolve_module_feather_path(), so a consumer recovers the identity of the module a feather holds
        from the file alone rather than from the manifest that named it.

    Args:
        feather_path: The path to the module feather file to read the identity of.

    Returns:
        The controller source identifier, the module type code, and the module identifier code, in that order.

    Raises:
        ValueError: If the filename does not follow the module feather naming convention, or if any of its three
            identity fields is not an integer.
    """
    parts = feather_path.stem.split("_")

    if (
        len(parts) != _MODULE_FEATHER_FIELD_COUNT
        or f"{parts[0]}_" != CONTROLLER_FEATHER_PREFIX
        or f"_{parts[2]}_" != MODULE_FEATHER_INFIX
        or not all(field.isdigit() for field in (parts[1], parts[3], parts[4]))
    ):
        message = (
            f"Unable to parse the module feather filename '{feather_path.name}'. The filename does not follow the "
            f"'{CONTROLLER_FEATHER_PREFIX}{{source_id}}{MODULE_FEATHER_INFIX}{{module_type}}_{{module_id}}"
            f"{FEATHER_SUFFIX}' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1]), int(parts[3]), int(parts[4])


def resolve_kernel_feather_path(output_directory: Path, source_id: str) -> Path:
    """Resolves the path of the feather file holding the target source's extracted kernel messages.

    Args:
        output_directory: The directory the extraction job writes its output into.
        source_id: The identifier of the controller source whose kernel messages the file holds.

    Returns:
        The path to the source's kernel message feather file.
    """
    return output_directory / f"{CONTROLLER_FEATHER_PREFIX}{source_id}{KERNEL_FEATHER_INFIX}{FEATHER_SUFFIX}"
