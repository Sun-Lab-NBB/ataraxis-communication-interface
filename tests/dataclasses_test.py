"""Contains tests for the classes and functions defined in the dataclasses module."""

from pathlib import Path

import pytest
from ataraxis_base_utilities import error_format

from ataraxis_communication_interface.dataclasses import (
    EXTRACTION_CONFIGURATION_FILENAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    ModuleSourceData,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    MicroControllerManifest,
    MicroControllerSourceData,
    ControllerExtractionConfig,
    create_extraction_config,
    write_microcontroller_manifest,
)


def test_constants() -> None:
    """Verifies that module-level constants have the expected values."""
    assert MICROCONTROLLER_MANIFEST_FILENAME == "microcontroller_manifest.yaml"
    assert EXTRACTION_CONFIGURATION_FILENAME == "extraction_configuration.yaml"


def test_module_source_data() -> None:
    """Verifies ModuleSourceData initialization and field access."""
    module = ModuleSourceData(module_type=1, module_id=2, name="encoder")

    assert module.module_type == 1
    assert module.module_id == 2
    assert module.name == "encoder"


def test_module_source_data_frozen() -> None:
    """Verifies that ModuleSourceData instances are immutable."""
    module = ModuleSourceData(module_type=1, module_id=2, name="encoder")

    with pytest.raises(AttributeError):
        module.module_type = 3  # type: ignore[misc]


def test_microcontroller_source_data() -> None:
    """Verifies MicroControllerSourceData initialization and field access."""
    modules = (
        ModuleSourceData(module_type=1, module_id=1, name="encoder"),
        ModuleSourceData(module_type=2, module_id=1, name="lick_sensor"),
    )
    controller = MicroControllerSourceData(id=10, name="actor_controller", modules=modules)

    assert controller.id == 10
    assert controller.name == "actor_controller"
    assert len(controller.modules) == 2
    assert controller.modules[0].name == "encoder"
    assert controller.modules[1].name == "lick_sensor"


def test_microcontroller_source_data_frozen() -> None:
    """Verifies that MicroControllerSourceData instances are immutable."""
    controller = MicroControllerSourceData(id=10, name="ctrl", modules=())

    with pytest.raises(AttributeError):
        controller.id = 20  # type: ignore[misc]


def test_module_extraction_config() -> None:
    """Verifies ModuleExtractionConfig initialization and field access."""
    config = ModuleExtractionConfig(module_type=1, module_id=2, event_codes=(10, 20, 30))

    assert config.module_type == 1
    assert config.module_id == 2
    assert config.event_codes == (10, 20, 30)


def test_kernel_extraction_config() -> None:
    """Verifies KernelExtractionConfig initialization and field access."""
    config = KernelExtractionConfig(event_codes=(1, 2, 3))

    assert config.event_codes == (1, 2, 3)


def test_controller_extraction_config() -> None:
    """Verifies ControllerExtractionConfig initialization with modules and kernel."""
    modules = (ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),)
    kernel = KernelExtractionConfig(event_codes=(5,))
    config = ControllerExtractionConfig(controller_id=10, modules=modules, kernel=kernel)

    assert config.controller_id == 10
    assert len(config.modules) == 1
    assert config.kernel is not None
    assert config.kernel.event_codes == (5,)


def test_controller_extraction_config_no_kernel() -> None:
    """Verifies ControllerExtractionConfig initialization with kernel set to None."""
    config = ControllerExtractionConfig(
        controller_id=10,
        modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),),
        kernel=None,
    )

    assert config.kernel is None


def test_microcontroller_manifest_empty() -> None:
    """Verifies that an empty MicroControllerManifest can be created."""
    manifest = MicroControllerManifest()

    assert manifest.controllers == []


def test_microcontroller_manifest_save_load_roundtrip(tmp_path: Path) -> None:
    """Verifies that a MicroControllerManifest can be saved and loaded with data intact."""
    modules = (ModuleSourceData(module_type=1, module_id=1, name="encoder"),)
    controller = MicroControllerSourceData(id=10, name="actor_controller", modules=modules)

    manifest = MicroControllerManifest()
    manifest.controllers.append(controller)

    file_path = tmp_path / "manifest.yaml"
    manifest.save(file_path=file_path)

    assert file_path.exists()

    loaded = MicroControllerManifest.load(file_path=file_path)

    assert len(loaded.controllers) == 1
    assert loaded.controllers[0].id == 10
    assert loaded.controllers[0].name == "actor_controller"
    assert len(loaded.controllers[0].modules) == 1
    assert loaded.controllers[0].modules[0].module_type == 1
    assert loaded.controllers[0].modules[0].module_id == 1
    assert loaded.controllers[0].modules[0].name == "encoder"


def test_microcontroller_manifest_multiple_controllers(tmp_path: Path) -> None:
    """Verifies that a manifest with multiple controllers roundtrips correctly."""
    controller_1 = MicroControllerSourceData(
        id=1, name="ctrl_1", modules=(ModuleSourceData(module_type=1, module_id=1, name="m1"),)
    )
    controller_2 = MicroControllerSourceData(
        id=2, name="ctrl_2", modules=(ModuleSourceData(module_type=2, module_id=1, name="m2"),)
    )

    manifest = MicroControllerManifest()
    manifest.controllers.extend([controller_1, controller_2])

    file_path = tmp_path / "manifest.yaml"
    manifest.save(file_path=file_path)

    loaded = MicroControllerManifest.load(file_path=file_path)

    assert len(loaded.controllers) == 2
    assert loaded.controllers[0].id == 1
    assert loaded.controllers[1].id == 2


def test_extraction_config_save_load_roundtrip(tmp_path: Path) -> None:
    """Verifies that an ExtractionConfig can be saved and loaded with data intact."""
    modules = (ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10, 20)),)
    kernel = KernelExtractionConfig(event_codes=(5, 6))
    controller = ControllerExtractionConfig(controller_id=10, modules=modules, kernel=kernel)

    config = ExtractionConfig(controllers=[controller])

    file_path = tmp_path / "config.yaml"
    config.save(file_path=file_path)

    assert file_path.exists()

    loaded = ExtractionConfig.load(file_path=file_path)

    assert len(loaded.controllers) == 1
    assert loaded.controllers[0].controller_id == 10
    assert loaded.controllers[0].modules[0].event_codes == (10, 20)
    assert loaded.controllers[0].kernel is not None
    assert loaded.controllers[0].kernel.event_codes == (5, 6)


def test_extraction_config_no_kernel_roundtrip(tmp_path: Path) -> None:
    """Verifies that an ExtractionConfig with kernel=None roundtrips correctly."""
    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=1,
                modules=(ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10,)),),
                kernel=None,
            )
        ]
    )

    file_path = tmp_path / "config.yaml"
    config.save(file_path=file_path)
    loaded = ExtractionConfig.load(file_path=file_path)

    assert loaded.controllers[0].kernel is None


def test_write_microcontroller_manifest_new(tmp_path: Path) -> None:
    """Verifies that write_microcontroller_manifest creates a new manifest file."""
    modules = (ModuleSourceData(module_type=1, module_id=1, name="encoder"),)
    write_microcontroller_manifest(
        log_directory=tmp_path, controller_id=10, controller_name="actor_ctrl", modules=modules
    )

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    assert manifest_path.exists()

    loaded = MicroControllerManifest.load(file_path=manifest_path)
    assert len(loaded.controllers) == 1
    assert loaded.controllers[0].id == 10
    assert loaded.controllers[0].name == "actor_ctrl"


def test_write_microcontroller_manifest_append(tmp_path: Path) -> None:
    """Verifies that write_microcontroller_manifest appends to an existing manifest."""
    modules_1 = (ModuleSourceData(module_type=1, module_id=1, name="encoder"),)
    modules_2 = (ModuleSourceData(module_type=2, module_id=1, name="lick_sensor"),)

    write_microcontroller_manifest(
        log_directory=tmp_path, controller_id=10, controller_name="ctrl_1", modules=modules_1
    )
    write_microcontroller_manifest(
        log_directory=tmp_path, controller_id=20, controller_name="ctrl_2", modules=modules_2
    )

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    loaded = MicroControllerManifest.load(file_path=manifest_path)

    assert len(loaded.controllers) == 2
    assert loaded.controllers[0].id == 10
    assert loaded.controllers[1].id == 20


def test_create_extraction_config(tmp_path: Path) -> None:
    """Verifies that create_extraction_config generates a valid config from a manifest."""
    modules = (
        ModuleSourceData(module_type=1, module_id=1, name="encoder"),
        ModuleSourceData(module_type=2, module_id=1, name="lick_sensor"),
    )
    manifest = MicroControllerManifest()
    manifest.controllers.append(MicroControllerSourceData(id=10, name="ctrl", modules=modules))

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    manifest.save(file_path=manifest_path)

    config = create_extraction_config(manifest_path=manifest_path)

    assert len(config.controllers) == 1
    assert config.controllers[0].controller_id == 10
    assert len(config.controllers[0].modules) == 2
    assert config.controllers[0].modules[0].module_type == 1
    assert config.controllers[0].modules[0].event_codes == ()
    assert config.controllers[0].modules[1].module_type == 2
    assert config.controllers[0].kernel is None


def test_create_extraction_config_multiple_controllers(tmp_path: Path) -> None:
    """Verifies that create_extraction_config handles multiple controllers."""
    manifest = MicroControllerManifest()
    manifest.controllers.append(
        MicroControllerSourceData(
            id=1, name="ctrl_1", modules=(ModuleSourceData(module_type=1, module_id=1, name="m1"),)
        )
    )
    manifest.controllers.append(
        MicroControllerSourceData(
            id=2, name="ctrl_2", modules=(ModuleSourceData(module_type=2, module_id=1, name="m2"),)
        )
    )

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    manifest.save(file_path=manifest_path)

    config = create_extraction_config(manifest_path=manifest_path)

    assert len(config.controllers) == 2
    assert config.controllers[0].controller_id == 1
    assert config.controllers[1].controller_id == 2


def test_create_extraction_config_missing_file(tmp_path: Path) -> None:
    """Verifies that create_extraction_config raises FileNotFoundError for a missing file."""
    nonexistent = tmp_path / "nonexistent.yaml"

    message = f"Unable to create extraction config from '{nonexistent}'. The path does not exist or is not a file."
    with pytest.raises(FileNotFoundError, match=error_format(message)):
        create_extraction_config(manifest_path=nonexistent)


def test_create_extraction_config_empty_manifest(tmp_path: Path) -> None:
    """Verifies that create_extraction_config raises ValueError for an empty manifest."""
    manifest = MicroControllerManifest()
    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    manifest.save(file_path=manifest_path)

    message = (
        f"Unable to create extraction config from '{manifest_path}'. The "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} contains no controller entries."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        create_extraction_config(manifest_path=manifest_path)
