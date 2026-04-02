from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from numpy.typing import NDArray as NDArray
from ataraxis_data_structures import ProcessingTracker

from .dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME as MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig as ExtractionConfig,
    MicroControllerManifest as MicroControllerManifest,
    ControllerExtractionConfig as ControllerExtractionConfig,
)
from .communication import (
    SerialProtocols as SerialProtocols,
    SerialPrototypes as SerialPrototypes,
)

LOG_ARCHIVE_SUFFIX: str
TRACKER_FILENAME: str
MICROCONTROLLER_DATA_DIRECTORY: str
PARALLEL_PROCESSING_THRESHOLD: int
CONTROLLER_FEATHER_PREFIX: str
MODULE_FEATHER_INFIX: str
KERNEL_FEATHER_INFIX: str
FEATHER_SUFFIX: str
_EXTRACTION_JOB_NAME: str

@dataclass(slots=True)
class _ExtractedMessages:
    timestamps: NDArray[np.uint64]
    commands: NDArray[np.uint8]
    events: NDArray[np.uint8]
    dtypes: tuple[str | None, ...]
    data_payloads: tuple[bytes | None, ...]
    @property
    def count(self) -> int: ...

@dataclass(slots=True)
class _ExtractedModuleData:
    module_type: int
    module_id: int
    messages: _ExtractedMessages

@dataclass(slots=True)
class _ColumnAccumulator:
    timestamps: list[int]
    commands: list[int]
    events: list[int]
    dtypes: list[str | None]
    data_payloads: list[bytes | None]

type _BatchResult = tuple[dict[tuple[int, int], _ColumnAccumulator], _ColumnAccumulator]

def run_log_processing_pipeline(
    log_directory: Path,
    output_directory: Path,
    config: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = True,
) -> None: ...
def execute_job(
    log_path: Path,
    output_directory: Path,
    source_id: str,
    job_id: str,
    workers: int,
    tracker: ProcessingTracker,
    controller_config: ControllerExtractionConfig,
    *,
    display_progress: bool = True,
    executor: ProcessPoolExecutor | None = None,
) -> None: ...
def resolve_recording_roots(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]: ...
def find_log_archive(log_directory: Path, source_id: str) -> Path: ...
def initialize_processing_tracker(output_directory: Path, source_ids: list[str]) -> dict[str, str]: ...
def _process_message_batch(
    log_path: Path,
    file_names: list[str],
    onset_us: np.uint64,
    module_filters: dict[tuple[int, int], frozenset[int]] | None,
    kernel_event_codes: frozenset[int] | None,
) -> _BatchResult: ...
def _extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]: ...
def _generate_job_ids(source_ids: list[str]) -> dict[str, str]: ...
def _finalize_accumulator(accumulator: _ColumnAccumulator) -> _ExtractedMessages: ...
def _build_message_dataframe(messages: _ExtractedMessages) -> pl.DataFrame: ...
def _write_module_feather(module_data: _ExtractedModuleData, source_id: str, output_directory: Path) -> None: ...
def _write_kernel_feather(kernel_data: _ExtractedMessages, source_id: str, output_directory: Path) -> None: ...
