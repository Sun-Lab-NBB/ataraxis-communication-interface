from pathlib import Path
from dataclasses import dataclass

from ataraxis_data_structures import YamlConfig

MICROCONTROLLER_MANIFEST_FILENAME: str
EXTRACTION_CONFIGURATION_FILENAME: str
_MANIFEST_LOCK_TIMEOUT: float

def write_microcontroller_manifest(
    log_directory: Path, controller_id: int, controller_name: str, modules: tuple[ModuleSourceData, ...]
) -> None: ...
def create_extraction_config(manifest_path: Path) -> ExtractionConfig: ...

@dataclass(frozen=True, slots=True)
class ModuleSourceData:
    module_type: int
    module_id: int
    name: str

@dataclass(frozen=True, slots=True)
class MicroControllerSourceData:
    id: int
    name: str
    modules: tuple[ModuleSourceData, ...]

@dataclass
class MicroControllerManifest(YamlConfig):
    controllers: list[MicroControllerSourceData]
    def __post_init__(self) -> None: ...

@dataclass(frozen=True, slots=True)
class ModuleExtractionConfig:
    module_type: int
    module_id: int
    event_codes: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class KernelExtractionConfig:
    event_codes: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class ControllerExtractionConfig:
    controller_id: int
    modules: tuple[ModuleExtractionConfig, ...]
    kernel: KernelExtractionConfig | None

@dataclass
class ExtractionConfig(YamlConfig):
    controllers: list[ControllerExtractionConfig]
    def __post_init__(self) -> None: ...
