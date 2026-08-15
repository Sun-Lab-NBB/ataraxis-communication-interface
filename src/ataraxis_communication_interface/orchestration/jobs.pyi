from enum import StrEnum
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

CONTROLLER_EXTRACTION_JOB_NAME: str
_MODULE_FILE_FIELD_COUNT: int
_KERNEL_FILE_FIELD_COUNT: int

class OutputLayout(StrEnum):
    DIRECTORY_NAME = "microcontroller_data"
    TRACKER_FILENAME = "microcontroller_processing_tracker.yaml"
    FILE_PREFIX = "controller_"
    MODULE_INFIX = "_module_"
    KERNEL_INFIX = "_kernel"
    FILE_SUFFIX = ".feather"

@dataclass(frozen=True, slots=True)
class JobDescriptor:
    log_directory: Path
    archive_path: Path
    output_directory: Path
    config_path: Path
    tracker_path: Path
    job_name: str
    job_id: str
    source_id: str
    core_weight: int
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
    ) -> JobDescriptor: ...
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> JobDescriptor: ...
    @property
    def dispatch_key(self) -> tuple[str, str]: ...
    def to_mapping(self) -> dict[str, str | int]: ...

@dataclass(frozen=True, slots=True)
class JobSizing:
    cores: int
    memory_mb: int

def generate_job_ids(source_ids: Sequence[str]) -> dict[str, str]: ...
def resolve_output_directory(output_directory: Path) -> Path: ...
def resolve_tracker_path(output_directory: Path) -> Path: ...
def resolve_module_path(output_directory: Path, source_id: str, module_type: int, module_id: int) -> Path: ...
def resolve_kernel_path(output_directory: Path, source_id: str) -> Path: ...
def find_module_paths(data_directory: Path) -> list[Path]: ...
def find_kernel_paths(data_directory: Path) -> list[Path]: ...
def parse_module_path(file_path: Path) -> tuple[int, int, int]: ...
def parse_kernel_path(file_path: Path) -> int: ...
