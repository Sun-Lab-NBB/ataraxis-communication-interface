from pathlib import Path
from dataclasses import dataclass

from ..microcontroller import (
    MICROCONTROLLER_MANIFEST_FILENAME as MICROCONTROLLER_MANIFEST_FILENAME,
    MicroControllerManifest as MicroControllerManifest,
)

EXTRACTION_JOB_NAME: str
TRACKER_FILENAME: str
MICROCONTROLLER_DATA_DIRECTORY: str
CONTROLLER_FEATHER_PREFIX: str
MODULE_FEATHER_INFIX: str
KERNEL_FEATHER_INFIX: str
FEATHER_SUFFIX: str
_MODULE_FEATHER_FIELD_COUNT: int

@dataclass(slots=True)
class PendingJob:
    log_directory: Path
    output_directory: Path
    tracker_path: Path
    job_id: str
    source_id: str
    config_path: Path
    core_weight: int = ...
    memory_mb: int = ...
    archive_path: Path | None = ...
    @property
    def dispatch_key(self) -> tuple[str, str]: ...

def discover_microcontroller_jobs(log_directory: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]: ...
def generate_job_ids(source_ids: list[str]) -> dict[str, str]: ...
def resolve_module_feather_path(output_directory: Path, source_id: str, module_type: int, module_id: int) -> Path: ...
def find_module_feathers(data_directory: Path) -> list[Path]: ...
def parse_module_feather_name(feather_path: Path) -> tuple[int, int, int]: ...
def resolve_kernel_feather_path(output_directory: Path, source_id: str) -> Path: ...
