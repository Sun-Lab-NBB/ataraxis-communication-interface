"""Provides the extraction algorithm that reads hardware module and kernel message data from MicroControllerInterface
log archives and the columnar structures it returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from ataraxis_base_utilities import console, resolve_worker_count
from ataraxis_data_structures import (
    PARALLEL_PROCESSING_THRESHOLD,
    LogArchiveReader,
    limit_worker_threads,
    initialize_worker_threads,
)

from ..communication import SerialProtocols, SerialPrototypes

if TYPE_CHECKING:
    from pathlib import Path
    from multiprocessing.context import SpawnContext

    from numpy.typing import NDArray

_BATCH_MULTIPLIER: int = 4
"""The number of message batches created per worker process during parallel processing. Splitting each worker's share
into four batches improves load distribution across workers."""

_WORKER_THREAD_CEILING: int = 1
"""The number of threads each worker of a self-owned pool pins its numeric backends to."""

_MULTIPROCESSING_CONTEXT: SpawnContext = get_context("spawn")
"""The spawn-based multiprocessing context used to create the process pool that extracts message data, ensuring
identical cross-platform behavior on all supported platforms."""


@dataclass(frozen=True, slots=True)
class ExtractedMessages:
    """Stores the data parsed from a set of incoming messages received by the PC from the microcontroller during
    runtime, in columnar form. All arrays share the same length, with each index position corresponding to a single
    message.
    """

    timestamps: NDArray[np.uint64]
    """Microseconds elapsed since the UTC epoch onset when each message was received by the PC."""
    commands: NDArray[np.uint8]
    """The command code that the module or kernel was executing when it sent each message."""
    events: NDArray[np.uint8]
    """The event code of each message."""
    dtypes: tuple[str | None, ...]
    """The numpy dtype string for the data payload of each message (e.g., ``'float32'``, ``'uint16'``), or None for
    state-only messages that carry no data and for data messages whose prototype code this library does not recognize.
    Combined with the corresponding ``data_payloads`` entry, the stored dtype string allows reconstructing the
    original numpy array from the payload bytes without any library dependency."""
    data_payloads: tuple[bytes | None, ...]
    """The serialized binary payload of each message, or None for state-only messages and for data messages whose
    prototype code this library does not recognize. Each entry is the raw byte representation of the numpy data array,
    decodable via the corresponding ``dtypes`` entry."""

    @property
    def count(self) -> int:
        """Returns the number of messages stored in this columnar block."""
        return len(self.timestamps)


@dataclass(frozen=True, slots=True)
class ExtractedModuleData:
    """Stores the data extracted from all messages sent to the PC by a hardware module instance during runtime that
    matched the caller's event code filter, in columnar form.
    """

    module_type: int
    """The type (family) code of the hardware module instance."""
    module_id: int
    """The unique identifier code of the hardware module instance."""
    messages: ExtractedMessages
    """Columnar storage for all extracted messages from this module."""


@dataclass(frozen=True, slots=True)
class ExtractedControllerData:
    """Stores every message one extraction pass over a microcontroller log archive resolved."""

    modules: tuple[ExtractedModuleData, ...]
    """The data of each hardware module that produced at least one matching message."""
    kernel: ExtractedMessages
    """Columnar storage for the extracted kernel messages, empty when kernel extraction was not configured."""


@dataclass(slots=True)
class _ColumnAccumulator:
    """Accumulates message data in parallel lists during batch extraction."""

    timestamps: list[int]
    """Microseconds elapsed since the UTC epoch onset when each message was received by the PC."""
    commands: list[int]
    """The command code for each message."""
    events: list[int]
    """The event code for each message."""
    dtypes: list[str | None]
    """The numpy dtype string for each message's data payload, or None for state-only messages and for data messages
    whose prototype code this library does not recognize."""
    data_payloads: list[bytes | None]
    """The serialized binary payload of each message, or None for state-only messages and for data messages whose
    prototype code this library does not recognize."""


type _BatchResult = tuple[
    dict[tuple[int, int], _ColumnAccumulator],
    _ColumnAccumulator,
]
"""Describes the return type of _process_message_batch: a module data dictionary mapping (type, id) tuples to
column accumulators, and a column accumulator for kernel messages."""


def extract_logged_microcontroller_data(
    log_path: Path,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
    kernel_event_codes: frozenset[int] | None,
    workers: int = -1,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> ExtractedControllerData:
    """Extracts the hardware module and kernel message data from the target .npz log archive.

    Reads the archive that the assemble_log_archives() function of ataraxis-data-structures builds from a
    MicroControllerInterface instance's DataLogger output and returns every incoming message whose event code the
    caller's filters admit.

    Notes:
        Works exclusively with the incoming messages the microcontroller sent to the PC. Each module is filtered
        against its own event code set, which prevents off-target extraction across modules that reuse an event code
        with different semantics.

        An archive holding fewer messages than the parallel processing threshold is read sequentially whatever
        worker count it is given, since the worker startup and the message transfer cost more than the parallel
        decode saves.

    Args:
        log_path: The path to the .npz log archive to process.
        module_filters: The event codes to extract for each module, keyed by the module type and identifier pair, or
            None to skip module extraction.
        kernel_event_codes: The event codes to extract for kernel messages, or None to skip kernel extraction.
        workers: The number of parallel worker processes (CPU cores) to use for processing. Setting this to a value
            below 1 auto-resolves the count to every available CPU core minus the cores reserved for the host system.
            Setting this to a value of 1 conducts the processing sequentially.
        display_progress: Determines whether to display a progress bar during parallel batch processing.
        executor: When provided, parallel batch work is submitted to this pool instead of a newly created one, and
            the caller owns the pool's lifecycle. Its worker count must match the workers value used for batch
            generation, and the caller is responsible for the worker thread limit its processes inherit.

    Returns:
        The extracted module and kernel messages in columnar form.

    Raises:
        ValueError: If the target path does not exist, does not have a .npz suffix, or does not point to a file. Also
            raised if the archive carries no onset timestamp message, and if a data message carries a data payload of
            a different size than its prototype code declares.
    """
    # Validates the archive path. LogArchiveReader checks existence, but not the .npz suffix or file type.
    if not log_path.exists() or log_path.suffix != ".npz" or not log_path.is_file():
        message = (
            f"Unable to extract microcontroller message data from the log file {log_path}, as it does not exist or "
            f"does not point to a valid .npz archive."
        )
        console.error(message=message, error=ValueError)

    # Creates a reader for the target archive. The reader handles onset timestamp discovery and message key management.
    reader = LogArchiveReader(archive_path=log_path)

    # An archive holding no data messages yields empty columns for every requested target.
    if not reader.message_count:
        return _finalize_batch(
            module_accumulators=_create_accumulators(module_filters=module_filters),
            kernel_accumulator=_create_accumulator(),
            module_filters=module_filters,
        )

    onset_us = reader.onset_timestamp_us

    # Processes small archives sequentially to avoid the unnecessary overhead of setting up the multiprocessing
    # runtime. Also applies when the caller explicitly requests a single worker process.
    if workers == 1 or reader.message_count < PARALLEL_PROCESSING_THRESHOLD:
        single_batch = reader.get_batches(workers=1, batch_multiplier=1)
        module_accumulators, kernel_accumulator = _process_message_batch(
            log_path=log_path,
            file_names=single_batch[0],
            onset_us=onset_us,
            module_filters=module_filters,
            kernel_event_codes=kernel_event_codes,
        )
        return _finalize_batch(
            module_accumulators=module_accumulators,
            kernel_accumulator=kernel_accumulator,
            module_filters=module_filters,
        )

    # Resolves the number of workers if not already resolved by the caller. External executors are pre-sized, so
    # the caller provides a positive workers value that matches the executor's pool size.
    if workers < 1:
        workers = resolve_worker_count(requested_workers=workers)

    batches = reader.get_batches(workers=workers, batch_multiplier=_BATCH_MULTIPLIER)

    results = _run_extraction_batches(
        log_path=log_path,
        batches=batches,
        onset_us=onset_us,
        module_filters=module_filters,
        kernel_event_codes=kernel_event_codes,
        workers=workers,
        display_progress=display_progress,
        executor=executor,
    )

    # Combines the columnar accumulators of every batch, maintaining chronological ordering. Each batch is released as
    # soon as its messages are copied out, so the batch columns and the combined columns are never both fully resident.
    # The list is reversed once and drained from the end, which keeps the chronological order while making each removal
    # a constant-time operation.
    combined_modules = _create_accumulators(module_filters=module_filters)
    combined_kernel = _create_accumulator()

    results.reverse()
    while results:
        batch_modules, batch_kernel = results.pop()
        for module_key, accumulator in batch_modules.items():
            _extend_accumulator(target=combined_modules[module_key], source=accumulator)
        _extend_accumulator(target=combined_kernel, source=batch_kernel)

    return _finalize_batch(
        module_accumulators=combined_modules,
        kernel_accumulator=combined_kernel,
        module_filters=module_filters,
    )


def _run_extraction_batches(
    log_path: Path,
    batches: list[list[str]],
    onset_us: np.uint64,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
    kernel_event_codes: frozenset[int] | None,
    workers: int,
    *,
    display_progress: bool,
    executor: ProcessPoolExecutor | None,
) -> list[_BatchResult]:
    """Processes every message batch of one archive across a worker pool.

    Notes:
        A pool this function owns pins its workers from both sides. The environment limit reaches the backends that
        size their pool while importing, and the initializer reaches the backends that read their width the first
        time they are asked to do work. A caller-supplied pool is left as the caller configured it.

    Args:
        log_path: The path to the .npz log archive to process.
        batches: The message keys of each batch, as the archive reader grouped them.
        onset_us: The onset of the data acquisition, in microseconds elapsed since UTC epoch onset.
        module_filters: The event codes to extract for each module, keyed by the module type and identifier pair, or
            None to skip module extraction.
        kernel_event_codes: The event codes to extract for kernel messages, or None to skip kernel extraction.
        workers: The number of worker processes to size a self-owned pool at.
        display_progress: Determines whether to display a progress bar while the batches are processed.
        executor: When provided, the batches are submitted to this pool instead of a newly created one.

    Returns:
        The accumulators each batch produced, in the order the batches were generated.
    """
    with ExitStack() as pool_scope:
        if executor is not None:
            active_executor = executor
        else:
            pool_scope.enter_context(limit_worker_threads(thread_count=_WORKER_THREAD_CEILING))
            active_executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=_MULTIPROCESSING_CONTEXT,
                initializer=initialize_worker_threads,
                initargs=(_WORKER_THREAD_CEILING,),
            )
            pool_scope.callback(active_executor.shutdown, wait=True)

        future_to_index = {
            active_executor.submit(
                _process_message_batch,
                log_path=log_path,
                file_names=batch_keys,
                onset_us=onset_us,
                module_filters=module_filters,
                kernel_event_codes=kernel_event_codes,
            ): index
            for index, batch_keys in enumerate(batches)
        }

        # Collects results while maintaining message order.
        results: list[_BatchResult | None] = [None] * len(batches)

        if display_progress:
            with console.progress(
                total=len(batches), description="Extracting microcontroller log data", unit="batch"
            ) as progress_bar:
                for future in as_completed(future_to_index):
                    results[future_to_index[future]] = future.result()
                    progress_bar.update(1)
        else:
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

    return [batch_result for batch_result in results if batch_result is not None]


def _process_message_batch(
    log_path: Path,
    file_names: list[str],
    onset_us: np.uint64,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
    kernel_event_codes: frozenset[int] | None,
) -> _BatchResult:
    """Processes a batch of messages from a MicroControllerInterface log archive, extracting both hardware module
    and kernel messages in a single pass into columnar accumulators.

    Each message is routed to module or kernel accumulators based on its protocol code, then filtered by per-module
    event codes. Each module is filtered against its own event code set to prevent off-target extraction across
    modules. Data payloads are converted to bytes immediately to avoid storing intermediate numpy objects.

    Args:
        log_path: The path to the processed .npz log file.
        file_names: The names of the individual message .npy files stored in the target archive.
        onset_us: The onset of the data acquisition, in microseconds elapsed since UTC epoch onset.
        module_filters: A mapping from (module_type, module_id) tuples to their per-module event code frozensets,
            or None to skip module extraction.
        kernel_event_codes: The event codes to extract for kernel messages, or None to skip kernel extraction.

    Returns:
        A tuple of (module_accumulators, kernel_accumulator). module_accumulators maps module (type, id) tuples
        to column accumulators. kernel_accumulator stores kernel messages in chronological order.

    Raises:
        ValueError: If a data message carries a data payload of a different size than its prototype code declares.
    """
    # Pre-creates columnar accumulators for each requested module and the kernel.
    extract_modules = module_filters is not None
    extract_kernel = kernel_event_codes is not None

    module_data = _create_accumulators(module_filters=module_filters)
    kernel_accumulator = _create_accumulator()

    # Pairs each module's event codes with its accumulator, so a message resolves both through one lookup instead of
    # probing the filter mapping and the accumulator mapping with the same key.
    module_targets: dict[tuple[int, int], tuple[frozenset[int], _ColumnAccumulator]] = {
        module: (event_codes, module_data[module]) for module, event_codes in (module_filters or {}).items()
    }

    # Pre-creates protocol sets outside the per-message loop to avoid re-creating them on every iteration.
    module_protocols = frozenset({SerialProtocols.MODULE_DATA, SerialProtocols.MODULE_STATE})
    kernel_protocols = frozenset({SerialProtocols.KERNEL_DATA, SerialProtocols.KERNEL_STATE})

    # Uses LogArchiveReader to iterate over the batch messages. Passing the pre-discovered onset_us avoids redundant
    # onset scanning in each worker process.
    reader = LogArchiveReader(archive_path=log_path, onset_us=onset_us)
    for log_message in reader.iter_messages(keys=file_names):
        payload = log_message.payload
        protocol = payload[0]

        # Routes module messages (MODULE_DATA / MODULE_STATE) through the extraction pipeline.
        if extract_modules and protocol in module_protocols:
            # Reads the header in one pass, since indexing the resulting bytes object yields Python integers while
            # indexing the payload array allocates a NumPy scalar per field. A MODULE_STATE payload carries only the
            # first five of these bytes, and the sixth is read solely under MODULE_DATA.
            header = payload[:6].tobytes()

            # Looks up the per-module event codes and accumulator in a single dict access. Returns None if the module
            # is not requested, combining module membership and event filter retrieval into one O(1) operation.
            current_module = (header[1], header[2])
            target = module_targets.get(current_module)
            if target is None:
                continue
            module_events, accumulator = target

            # Filters against this specific module's event codes, preventing off-target extraction.
            event_code = header[4]
            if event_code not in module_events:
                continue

            # Resolves the numpy dtype string and extracts the raw data bytes for MODULE_DATA messages. Uses the
            # pre-built dtype lookup to avoid per-message prototype object allocation.
            dtype_str: str | None = None
            data_payload: bytes | None = None
            if header[0] == SerialProtocols.MODULE_DATA:
                prototype_code = header[5]
                dtype_str = SerialPrototypes.get_dtype_for_code(code=prototype_code)
                if dtype_str is not None:
                    data_payload = payload[6:].tobytes()
                    _validate_payload_size(
                        prototype_code=prototype_code,
                        data_payload=data_payload,
                        module=current_module,
                        log_path=log_path,
                    )

            # Appends directly to the module's columnar accumulator.
            accumulator.timestamps.append(int(log_message.timestamp_us))
            accumulator.commands.append(header[3])
            accumulator.events.append(event_code)
            accumulator.dtypes.append(dtype_str)
            accumulator.data_payloads.append(data_payload)

        # Routes kernel messages (KERNEL_DATA / KERNEL_STATE) through the extraction pipeline.
        elif extract_kernel and protocol in kernel_protocols:
            # Reads the header in one pass on the same terms as the module header above. A KERNEL_STATE payload
            # carries only the first three of these bytes, and the fourth is read solely under KERNEL_DATA.
            header = payload[:4].tobytes()

            # Extracts only messages with requested event codes.
            event_code = header[2]
            if event_code not in kernel_event_codes:  # type: ignore[operator]  # narrowed by the extract_kernel flag.
                continue

            # Resolves the numpy dtype string and extracts the raw data bytes for KERNEL_DATA messages.
            dtype_str = None
            data_payload = None
            if header[0] == SerialProtocols.KERNEL_DATA:
                prototype_code = header[3]
                dtype_str = SerialPrototypes.get_dtype_for_code(code=prototype_code)
                if dtype_str is not None:
                    data_payload = payload[4:].tobytes()
                    _validate_payload_size(
                        prototype_code=prototype_code,
                        data_payload=data_payload,
                        module=None,
                        log_path=log_path,
                    )

            # Appends directly to the kernel's columnar accumulator.
            kernel_accumulator.timestamps.append(int(log_message.timestamp_us))
            kernel_accumulator.commands.append(header[1])
            kernel_accumulator.events.append(event_code)
            kernel_accumulator.dtypes.append(dtype_str)
            kernel_accumulator.data_payloads.append(data_payload)

    return module_data, kernel_accumulator


def _validate_payload_size(
    prototype_code: int,
    data_payload: bytes,
    module: tuple[int, int] | None,
    log_path: Path,
) -> None:
    """Verifies that a data message carries the payload width its prototype code declares.

    Notes:
        The prototype code declares both the dtype and the width of the data object that follows it, so a payload of
        any other width cannot be decoded through the dtype stored alongside it in the extracted feather.

    Args:
        prototype_code: The prototype code the message declares.
        data_payload: The raw payload bytes the message carries.
        module: The type and identifier codes of the hardware module that sent the message, or None when the kernel
            sent it. Used to attribute the failure.
        log_path: The path to the archive holding the message, used to attribute the failure.

    Raises:
        ValueError: If the payload width disagrees with the width the prototype code declares.
    """
    declared_size = SerialPrototypes.get_byte_size_for_code(code=prototype_code)
    if declared_size is not None and len(data_payload) != declared_size:
        # The label is built here rather than at the call site, so the ordinary path formats no string per message.
        source_label = "the kernel" if module is None else f"the module {module[0]} {module[1]}"
        message = (
            f"Unable to extract the data message logged by {source_label} to '{log_path}'. The message declares the "
            f"prototype code {prototype_code}, whose data object occupies {declared_size} bytes, but it carries a "
            f"{len(data_payload)}-byte data payload."
        )
        console.error(message=message, error=ValueError)


def _create_accumulator() -> _ColumnAccumulator:
    """Creates an empty column accumulator.

    Returns:
        A column accumulator holding no messages.
    """
    return _ColumnAccumulator(timestamps=[], commands=[], events=[], dtypes=[], data_payloads=[])


def _create_accumulators(
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
) -> dict[tuple[int, int], _ColumnAccumulator]:
    """Creates one empty column accumulator per requested module.

    Args:
        module_filters: The event codes to extract for each module, keyed by the module type and identifier pair, or
            None when module extraction is not requested.

    Returns:
        An empty column accumulator for each requested module, keyed by that module's type and identifier pair.
    """
    return {module: _create_accumulator() for module in (module_filters or {})}


def _extend_accumulator(target: _ColumnAccumulator, source: _ColumnAccumulator) -> None:
    """Appends every message a source accumulator holds to the target accumulator.

    Args:
        target: The accumulator the messages are appended to.
        source: The accumulator the messages are read from.
    """
    target.timestamps.extend(source.timestamps)
    target.commands.extend(source.commands)
    target.events.extend(source.events)
    target.dtypes.extend(source.dtypes)
    target.data_payloads.extend(source.data_payloads)


def _finalize_accumulator(accumulator: _ColumnAccumulator) -> ExtractedMessages:
    """Converts a growable column accumulator into a finalized ExtractedMessages instance with numpy arrays.

    Args:
        accumulator: The column accumulator to finalize.

    Returns:
        An ExtractedMessages instance with numpy arrays built from the accumulator's lists.
    """
    return ExtractedMessages(
        timestamps=np.array(accumulator.timestamps, dtype=np.uint64),
        commands=np.array(accumulator.commands, dtype=np.uint8),
        events=np.array(accumulator.events, dtype=np.uint8),
        dtypes=tuple(accumulator.dtypes),
        data_payloads=tuple(accumulator.data_payloads),
    )


def _finalize_batch(
    module_accumulators: dict[tuple[int, int], _ColumnAccumulator],
    kernel_accumulator: _ColumnAccumulator,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
) -> ExtractedControllerData:
    """Converts the accumulators of one extraction pass into the finalized controller data.

    Notes:
        Reports a module only when it produced at least one matching message, so a configured module that stayed
        silent for the whole recording contributes no entry and therefore no output file.

    Args:
        module_accumulators: The column accumulator of each requested module, keyed by the module type and
            identifier pair. The mapping is emptied as the accumulators are converted.
        kernel_accumulator: The column accumulator holding the extracted kernel messages.
        module_filters: The event codes requested for each module, whose key order the reported modules follow.

    Returns:
        The finalized controller data.
    """
    # Removes each accumulator as it is converted, so its columns are released while the remaining modules are still
    # being converted rather than all at once when the caller releases the mapping.
    modules: list[ExtractedModuleData] = []
    for module in module_filters or {}:
        accumulator = module_accumulators.pop(module)
        if accumulator.timestamps:
            modules.append(
                ExtractedModuleData(
                    module_type=module[0],
                    module_id=module[1],
                    messages=_finalize_accumulator(accumulator=accumulator),
                )
            )

    return ExtractedControllerData(modules=tuple(modules), kernel=_finalize_accumulator(accumulator=kernel_accumulator))
