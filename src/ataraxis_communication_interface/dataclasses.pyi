from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_data_structures import YamlConfig

MICROCONTROLLER_MANIFEST_FILENAME: str
EXTRACTION_CONFIGURATION_FILENAME: str

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
    controllers: list[MicroControllerSourceData] = field(default_factory=list)
    def save(self, file_path: Path) -> None: ...
    @classmethod
    def load(cls, file_path: Path) -> MicroControllerManifest: ...

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
    def save(self, file_path: Path) -> None: ...
    @classmethod
    def load(cls, file_path: Path) -> ExtractionConfig: ...
