from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import SpawnContext

import numpy as np
from numpy.typing import NDArray as NDArray

from ..communication import (
    SerialProtocols as SerialProtocols,
    SerialPrototypes as SerialPrototypes,
)

_BATCH_MULTIPLIER: int
_WORKER_THREAD_CEILING: int
_MULTIPROCESSING_CONTEXT: SpawnContext

@dataclass(frozen=True, slots=True)
class ExtractedMessages:
    timestamps: NDArray[np.uint64]
    commands: NDArray[np.uint8]
    events: NDArray[np.uint8]
    dtypes: tuple[str | None, ...]
    data_payloads: tuple[bytes | None, ...]
    @property
    def count(self) -> int: ...

@dataclass(frozen=True, slots=True)
class ExtractedModuleData:
    module_type: int
    module_id: int
    messages: ExtractedMessages

@dataclass(frozen=True, slots=True)
class ExtractedControllerData:
    modules: tuple[ExtractedModuleData, ...]
    kernel: ExtractedMessages

@dataclass(slots=True)
class _ColumnAccumulator:
    timestamps: list[int]
    commands: list[int]
    events: list[int]
    dtypes: list[str | None]
    data_payloads: list[bytes | None]

type _BatchResult = tuple[dict[tuple[int, int], _ColumnAccumulator], _ColumnAccumulator]

def extract_logged_microcontroller_data(
    log_path: Path,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
    kernel_event_codes: frozenset[int] | None,
    workers: int = -1,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> ExtractedControllerData: ...
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
) -> list[_BatchResult]: ...
def _process_message_batch(
    log_path: Path,
    file_names: list[str],
    onset_us: np.uint64,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
    kernel_event_codes: frozenset[int] | None,
) -> _BatchResult: ...
def _validate_payload_size(prototype_code: int, data_payload: bytes, source_label: str, log_path: Path) -> None: ...
def _create_accumulator() -> _ColumnAccumulator: ...
def _create_accumulators(
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
) -> dict[tuple[int, int], _ColumnAccumulator]: ...
def _extend_accumulator(target: _ColumnAccumulator, source: _ColumnAccumulator) -> None: ...
def _finalize_accumulator(accumulator: _ColumnAccumulator) -> ExtractedMessages: ...
def _finalize_batch(
    module_accumulators: dict[tuple[int, int], _ColumnAccumulator],
    kernel_accumulator: _ColumnAccumulator,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
) -> ExtractedControllerData: ...
