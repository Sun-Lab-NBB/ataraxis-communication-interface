"""Provides the manifest-derived job resolution every consumer shares, the preparation that turns a resolved universe
into dispatchable job descriptors, and the archive pass that sizes one prepared job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace, dataclass

from ataraxis_base_utilities import console
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    ProcessingTracker,
    index_marker_files,
    discover_marker_files,
)

from .jobs import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    JobSizing,
    JobDescriptor,
    generate_job_ids,
    resolve_tracker_path,
    resolve_output_directory,
)
from .allocation import (
    CONTROLLER_EXTRACTION_JOB_CORES,
    resolve_core_budget,
    resolve_job_workers,
    estimate_job_memory_mb,
    resolve_archive_footprint,
)
from ..microcontroller import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    MicroControllerManifest,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class JobSource:
    """Describes one controller source the manifest registers and the log archive it produced."""

    source_id: str
    """The identifier of the source, as it appears in every job specifier and every archive filename."""
    name: str
    """The colloquial name the manifest records for the source."""
    archive_path: Path | None
    """The path to the source's log archive, or None when the tree holds no single archive for it."""


@dataclass(frozen=True, slots=True)
class JobUniverse:
    """Describes every extraction job one log directory's manifest defines and the subset its archives back."""

    log_directory: Path
    """The root directory the resolution searched."""
    manifest_path: Path | None
    """The path to the single microcontroller manifest the directory holds, or None when the tree holds none."""
    sources: tuple[JobSource, ...]
    """Every source the manifest registers, in ascending source identifier order."""
    universe: tuple[tuple[str, str], ...]
    """Every job the manifest defines, as job name and source identifier pairs.

    Notes:
        This is a manifest fingerprint rather than an invocation fingerprint, so every invocation aligns a tracker
        against the same set and no invocation resets the jobs it did not request.
    """
    possible: tuple[tuple[str, str], ...]
    """The subset of the universe whose archive resolved to exactly one file under the log directory."""

    @property
    def archives(self) -> dict[str, Path]:
        """Returns the resolved archive of each source that has one, keyed by that source identifier."""
        return {source.source_id: source.archive_path for source in self.sources if source.archive_path is not None}


@dataclass(frozen=True, slots=True)
class JobSet:
    """Describes the dispatchable extraction jobs one invocation prepared for one log directory."""

    log_directory: Path
    """The root directory holding the manifest and the log archives."""
    output_directory: Path
    """The subdirectory the preparation resolves, created once at least one job is prepared, which holds the tracker
    and every output file."""
    tracker_path: Path
    """The path to the ProcessingTracker file recording every job in this set."""
    universe: tuple[tuple[str, str], ...]
    """Every job the manifest defines, which is the set the tracker is aligned against."""
    jobs: tuple[JobDescriptor, ...]
    """Every dispatchable job this set holds, in ascending source identifier order."""
    skipped_sources: tuple[tuple[str, str], ...]
    """Each source that yielded no job, paired with the reason. Always empty under strict sourcing, where a source
    that cannot be prepared raises instead."""
    core_ceiling: int
    """The cores any single job of this set may receive."""

    def resolve_job(self, job_id: str) -> JobDescriptor:
        """Returns the descriptor of the requested job.

        Args:
            job_id: The hexadecimal identifier of the job to resolve.

        Returns:
            The descriptor of the requested job.

        Raises:
            ValueError: If no job in this set carries the requested identifier.
        """
        matches = {descriptor.job_id: descriptor for descriptor in self.jobs}

        if job_id in matches:
            return matches[job_id]

        held_ids = ", ".join(sorted(matches)) or "none"
        message = (
            f"Unable to resolve the microcontroller data extraction job '{job_id}' in '{self.log_directory}'. The "
            f"prepared job set holds no job with that identifier. Held job IDs: {held_ids}."
        )
        console.error(message=message, error=ValueError)

        # Satisfies ruff RET503. console.error() is NoReturn, so this line never executes.
        raise ValueError(message)  # pragma: no cover


def resolve_jobs(log_directory: Path) -> JobUniverse:
    """Resolves the extraction job universe of one log directory and the subset its archives back.

    Notes:
        Reads the manifest and indexes the archive filenames, decoding no message and writing nothing, so a caller
        enumerates a directory's jobs without launching or materializing anything. Two tree walks serve any number
        of sources, one for the manifest and one indexing every archive name the manifest implies.

        One recording writes one MicroControllerInterface set to one DataLogger, so a tree holding several manifests
        spans several recordings and is rejected rather than resolved against the first manifest found. A tree
        holding no manifest holds no microcontroller jobs, and yields an empty universe rather than an error.

    Args:
        log_directory: The root directory whose tree is searched for the manifest and the log archives.

    Returns:
        The resolved job universe.

    Raises:
        FileNotFoundError: If the log directory does not exist or is not a directory.
        ValueError: If the tree holds more than one microcontroller manifest, or if a manifest registers no sources.
        OSError: If any directory beneath the log directory cannot be read.
    """
    if not log_directory.is_dir():
        message = (
            f"Unable to resolve microcontroller data extraction jobs in '{log_directory}'. The path does not exist "
            f"or is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    candidates = discover_marker_files(directory=log_directory, marker_name=MICROCONTROLLER_MANIFEST_FILENAME)

    # A tree holding no microcontroller manifest holds no extraction jobs, which is an answer rather than a failure.
    # A caller walking many recordings reads the empty universe and moves on, while a caller that asked for work to
    # be done raises on the empty result itself.
    if not candidates:
        return JobUniverse(
            log_directory=log_directory,
            manifest_path=None,
            sources=(),
            universe=(),
            possible=(),
        )

    if len(candidates) > 1:
        message = (
            f"Unable to resolve microcontroller data extraction jobs in '{log_directory}'. The directory tree holds "
            f"{len(candidates)} {MICROCONTROLLER_MANIFEST_FILENAME} files, which means it spans several recordings "
            f"or several DataLogger instances: {[str(candidate) for candidate in candidates]}. One recording writes "
            f"one MicroControllerInterface set to one logger, so exactly one manifest is supported per invocation. "
            f"Pass the individual DataLogger output directory of each recording instead."
        )
        console.error(message=message, error=ValueError)

    manifest_path = candidates[0]
    manifest = MicroControllerManifest.from_yaml(file_path=manifest_path)
    entries = {str(controller.id): controller.name for controller in manifest.controllers}

    if not entries:
        message = (
            f"Unable to resolve microcontroller data extraction jobs in '{log_directory}'. The "
            f"{MICROCONTROLLER_MANIFEST_FILENAME} at '{manifest_path}' contains no controller entries."
        )
        console.error(message=message, error=ValueError)

    source_ids = sorted(entries)

    # Indexes every source's archive in one pass, since the archive names are known once the manifest resolves. A
    # source whose name resolves to several archives spans several loggers, which is ambiguous rather than redundant,
    # so it is left unresolved alongside the sources holding no archive at all.
    archives = index_marker_files(
        directory=log_directory,
        marker_names=[f"{source_id}{LOG_ARCHIVE_SUFFIX}" for source_id in source_ids],
    )
    matches = {source_id: archives[f"{source_id}{LOG_ARCHIVE_SUFFIX}"] for source_id in source_ids}

    sources = tuple(
        JobSource(
            source_id=source_id,
            name=entries[source_id],
            archive_path=matches[source_id][0] if len(matches[source_id]) == 1 else None,
        )
        for source_id in source_ids
    )

    return JobUniverse(
        log_directory=log_directory,
        manifest_path=manifest_path,
        sources=sources,
        universe=tuple((CONTROLLER_EXTRACTION_JOB_NAME, source_id) for source_id in source_ids),
        possible=tuple(
            (CONTROLLER_EXTRACTION_JOB_NAME, source.source_id) for source in sources if source.archive_path is not None
        ),
    )


def prepare_jobs(
    log_directory: Path,
    output_directory: Path,
    config_path: Path,
    source_ids: Sequence[str] | None = None,
    job_id: str | None = None,
    *,
    core_ceiling: int = -1,
    strict_sources: bool = True,
) -> JobSet:
    """Resolves and registers the microcontroller data extraction jobs of one log directory.

    Notes:
        Materializes the output subdirectory and aligns the tracker against the manifest universe once at least one
        job is prepared, which is every write this call performs outside a job's own output. The prepared job list
        lives in the returned set rather than on disk.

        Reads no archive. Every job carries the core ceiling as its width, and the extraction reads an archive below
        the parallel processing threshold sequentially whatever width it is given, so a caller that only runs jobs pays
        nothing to prepare them.

        A tree holding no manifest prepares no job whatever the configuration declares, matching the empty universe
        the resolution reports for it.

        The extraction configuration declares which controllers this stage processes, so it bounds the requested set
        the same way the manifest bounds the universe.

    Args:
        log_directory: The root directory whose tree holds the manifest and the log archives.
        output_directory: The root output directory. The library's own subdirectory is created under it.
        config_path: The path to the ExtractionConfig .yaml file naming the controllers and events to extract.
        source_ids: The sources to prepare jobs for, or None to prepare every controller the configuration declares.
            The argument is ignored when a job identifier selects the work.
        job_id: The hexadecimal identifier of the single job to prepare. Leaving this unset prepares every requested
            source.
        core_ceiling: The cores any single job may receive. A non-positive value resolves the ceiling from the host.
        strict_sources: Determines whether a source that cannot be prepared stops the call. When set, a requested
            source the manifest or the configuration does not register, or one whose archive does not resolve to
            exactly one file, raises. When unset, such a source is recorded in the returned set's skipped sources.

    Returns:
        The prepared job set.

    Raises:
        FileNotFoundError: If the log directory or the configuration file does not exist, or if a requested source's
            archive is absent under strict sourcing.
        ValueError: If the tree holds more than one manifest, if a manifest registers no sources, if the configuration
            declares no controllers, or if a job identifier matches no configured controller. Also raised if a
            requested source is absent from the microcontroller manifest or from the extraction configuration under
            strict sourcing, or if the resolved archives span several directories.
        OSError: If any directory beneath the log directory cannot be read.
        TimeoutError: If the tracker's lock cannot be acquired.
    """
    if not config_path.is_file():
        message = (
            f"Unable to load the extraction config from '{config_path}'. The path does not exist or is not a file."
        )
        console.error(message=message, error=FileNotFoundError)

    universe = resolve_jobs(log_directory=log_directory)
    registered_ids = [source.source_id for source in universe.sources]
    archives = universe.archives

    resolved_output = resolve_output_directory(output_directory=output_directory)
    tracker_path = resolve_tracker_path(output_directory=resolved_output)

    # The host budget bounds what the whole recording may claim, and the declared job width bounds what any one job
    # repays, so a job is dispatched at the smaller of the two.
    ceiling = min(resolve_core_budget(requested_budget=core_ceiling), CONTROLLER_EXTRACTION_JOB_CORES)

    # A tree holding no manifest holds no job this library owns, whatever the configuration declares. Reporting the
    # empty set here keeps the answer the resolution already gave, since weighing the configuration against an
    # unregistered universe would attribute the absent manifest to the controllers the caller asked for.
    if universe.manifest_path is None:
        return JobSet(
            log_directory=log_directory,
            output_directory=resolved_output,
            tracker_path=tracker_path,
            universe=(),
            jobs=(),
            skipped_sources=(),
            core_ceiling=ceiling,
        )

    configured_ids, unregistered_ids = _resolve_configured_ids(config_path=config_path, registered_ids=registered_ids)

    if job_id is not None:
        # An identifier names one job, so the requested set is the source whose identifier matches it. Resolving it
        # against the configuration rather than against the archives on disk keeps a missing sibling archive from
        # hiding the job that was actually named.
        identifiers = generate_job_ids(source_ids=configured_ids)
        matched = [source_id for source_id, candidate in identifiers.items() if candidate == job_id]

        if not matched:
            message = (
                f"Unable to prepare the microcontroller data extraction job '{job_id}' in '{log_directory}'. The "
                f"extraction config at '{config_path}' declares no controller with that job identifier. Configured "
                f"controller IDs: {', '.join(configured_ids)}."
            )
            console.error(message=message, error=ValueError)

        requested_ids = matched
    else:
        requested_ids = sorted(source_ids) if source_ids else configured_ids

    # A controller the configuration declares but the manifest does not register bears on this call only when it is
    # requested, so a shared configuration spanning several recordings still prepares the sources each one holds.
    unregistered_requests = sorted(set(requested_ids) & set(unregistered_ids))
    unconfigured_ids = [source_id for source_id in requested_ids if source_id not in configured_ids]
    skipped: list[tuple[str, str]] = []

    if unregistered_requests:
        message = (
            f"Unable to prepare microcontroller data extraction jobs using the extraction config at "
            f"'{config_path}'. The following controller IDs are not registered in the "
            f"{MICROCONTROLLER_MANIFEST_FILENAME}: {', '.join(unregistered_requests)}. The corresponding log "
            f"archives were not produced by ataraxis-communication-interface. Registered IDs: "
            f"{', '.join(registered_ids)}."
        )
        if strict_sources:
            console.error(message=message, error=ValueError)
        skipped.extend(
            (source_id, "The controller is absent from the microcontroller manifest.")
            for source_id in unregistered_requests
        )

    if unconfigured_ids:
        message = (
            f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The following requested "
            f"controller IDs are absent from the extraction config at '{config_path}': "
            f"{', '.join(unconfigured_ids)}. Configured controller IDs: {', '.join(configured_ids)}."
        )
        if strict_sources:
            console.error(message=message, error=ValueError)
        skipped.extend(
            (source_id, "The controller is absent from the extraction configuration.") for source_id in unconfigured_ids
        )

    unresolved_ids = [
        source_id
        for source_id in requested_ids
        if source_id in configured_ids and source_id not in unregistered_ids and source_id not in archives
    ]

    if unresolved_ids:
        message = (
            f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The log archives of the "
            f"following requested controller IDs are absent or resolve to more than one file: "
            f"{', '.join(unresolved_ids)}."
        )
        if strict_sources:
            console.error(message=message, error=FileNotFoundError)
        skipped.extend(
            (source_id, "The controller's log archive is absent or resolves to more than one file.")
            for source_id in unresolved_ids
        )

    prepared_ids = [source_id for source_id in requested_ids if source_id in archives and source_id in configured_ids]
    parent_directories = {archives[source_id].parent for source_id in prepared_ids}

    if len(parent_directories) > 1:
        message = (
            f"Unable to prepare microcontroller data extraction jobs in '{log_directory}'. The resolved log archives "
            f"sit in {len(parent_directories)} different directories: "
            f"{sorted(str(parent) for parent in parent_directories)}. Archives in separate directories were written "
            f"by separate DataLogger instances, and one recording writes one logger, so this tree holds more than "
            f"one recording. Each DataLogger output directory must be prepared and processed on its own invocation."
        )
        console.error(message=message, error=ValueError)

    # Creates the output layout only once a job is going to be written into it, so a lenient request that prepared
    # nothing leaves the caller's output path as it found it.
    if prepared_ids:
        resolved_output.mkdir(parents=True, exist_ok=True)

    jobs = tuple(
        JobDescriptor.for_archive(
            archive_path=archives[source_id],
            output_directory=resolved_output,
            config_path=config_path,
            tracker_path=tracker_path,
            source_id=source_id,
            log_directory=log_directory,
            core_weight=ceiling,
        )
        for source_id in prepared_ids
    )

    # Aligns the tracker with the prepared subset while detecting foreign entries against the full manifest universe,
    # so an invocation covering part of a recording leaves its siblings' recorded outcomes untouched. A lenient
    # request that prepared nothing registers nothing, since a tracker names at least one job and the caller receives
    # the reasons through the skipped sources instead.
    if prepared_ids:
        ProcessingTracker(file_path=tracker_path).align_jobs(
            jobs=[(CONTROLLER_EXTRACTION_JOB_NAME, source_id) for source_id in prepared_ids],
            universe=list(universe.universe),
        )

    return JobSet(
        log_directory=log_directory,
        output_directory=resolved_output,
        tracker_path=tracker_path,
        universe=universe.universe,
        jobs=jobs,
        skipped_sources=tuple(sorted(skipped)),
        core_ceiling=ceiling,
    )


def size_job(job: JobDescriptor, core_ceiling: int = -1) -> tuple[JobDescriptor, JobSizing]:
    """Sizes one prepared job from the archive it reads.

    Notes:
        Reads the archive's zip directory and its file metadata alone, decoding no message.

    Args:
        job: The prepared job to size.
        core_ceiling: The cores this job may receive, which bounds the width its archive resolves to. A non-positive
            value resolves the ceiling from the host.

    Returns:
        The job carrying its resolved width, and the figures the sizing produced.
    """
    ceiling = resolve_core_budget(requested_budget=core_ceiling) if core_ceiling < 1 else core_ceiling
    footprint = resolve_archive_footprint(archive_path=job.archive_path)
    core_weight = resolve_job_workers(footprint=footprint, ceiling=ceiling)

    return (
        replace(job, core_weight=core_weight),
        JobSizing(
            memory_mb=estimate_job_memory_mb(footprint=footprint, cores=core_weight),
            message_count=footprint.message_count,
            archive_bytes=footprint.archive_bytes,
            modeled=footprint.modeled,
        ),
    )


def _resolve_configured_ids(config_path: Path, registered_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Reads the extraction configuration and resolves its controllers against the manifest job universe.

    Args:
        config_path: The path to the extraction configuration .yaml file.
        registered_ids: The controller source identifiers the microcontroller manifest registers.

    Returns:
        The controller source identifiers the configuration declares, and the subset of them the manifest does not
        register, both in ascending order.

    Raises:
        ValueError: If the configuration declares no controllers.
    """
    resolved_config = ExtractionConfig.from_yaml(file_path=config_path)
    configured_ids = sorted({str(controller.controller_id) for controller in resolved_config.controllers})

    if not configured_ids:
        message = (
            f"Unable to prepare microcontroller data extraction jobs using the extraction config at "
            f"'{config_path}'. It declares no controllers."
        )
        console.error(message=message, error=ValueError)

    return configured_ids, sorted(set(configured_ids) - set(registered_ids))
