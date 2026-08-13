"""Contains tests for the classes and functions provided by the dataclasses.py module."""

from typing import Any
from pathlib import Path
import threading
import multiprocessing

import pytest
from filelock import Timeout, FileLock
from ataraxis_base_utilities import error_format

from ataraxis_communication_interface.microcontroller import dataclasses
from ataraxis_communication_interface.microcontroller.dataclasses import (
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

_CONCURRENCY_TIMEOUT: int = 30
"""Stores the time, in seconds, the concurrency tests wait for a helper process or a lock before giving up."""


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
    controller = MicroControllerSourceData(id=10, name="controller", modules=())

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
    manifest = MicroControllerManifest(controllers=[])

    assert manifest.controllers == []


def test_microcontroller_manifest_save_load_roundtrip(tmp_path: Path) -> None:
    """Verifies that a MicroControllerManifest can be saved and loaded with data intact."""
    modules = (ModuleSourceData(module_type=1, module_id=1, name="encoder"),)
    controller = MicroControllerSourceData(id=10, name="actor_controller", modules=modules)

    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(controller)

    file_path = tmp_path / "manifest.yaml"
    manifest.to_yaml(file_path=file_path)

    assert file_path.exists()

    loaded = MicroControllerManifest.from_yaml(file_path=file_path)

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
        id=1, name="controller_1", modules=(ModuleSourceData(module_type=1, module_id=1, name="module_1"),)
    )
    controller_2 = MicroControllerSourceData(
        id=2, name="controller_2", modules=(ModuleSourceData(module_type=2, module_id=1, name="module_2"),)
    )

    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.extend([controller_1, controller_2])

    file_path = tmp_path / "manifest.yaml"
    manifest.to_yaml(file_path=file_path)

    loaded = MicroControllerManifest.from_yaml(file_path=file_path)

    assert len(loaded.controllers) == 2
    assert loaded.controllers[0].id == 1
    assert loaded.controllers[1].id == 2


def test_microcontroller_manifest_non_list_controllers() -> None:
    """Verifies that MicroControllerManifest rejects a 'controllers' field that does not store a list."""
    message = (
        "Unable to initialize the MicroControllerManifest instance. The 'controllers' field must store a list of "
        "MicroControllerSourceData instances, but got NoneType."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        MicroControllerManifest(controllers=None)  # type: ignore[arg-type]


def test_microcontroller_manifest_null_controllers_yaml(tmp_path: Path) -> None:
    """Verifies that loading a manifest whose 'controllers' key carries no value raises a ValueError."""
    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    manifest_path.write_text("controllers:\n", encoding="utf-8")

    message = (
        "Unable to initialize the MicroControllerManifest instance. The 'controllers' field must store a list of "
        "MicroControllerSourceData instances, but got NoneType."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        MicroControllerManifest.from_yaml(file_path=manifest_path)


def test_extraction_config_empty() -> None:
    """Verifies that an empty ExtractionConfig can be created."""
    config = ExtractionConfig(controllers=[])

    assert config.controllers == []


def test_extraction_config_save_load_roundtrip(tmp_path: Path) -> None:
    """Verifies that an ExtractionConfig can be saved and loaded with data intact."""
    modules = (ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(10, 20)),)
    kernel = KernelExtractionConfig(event_codes=(5, 6))
    controller = ControllerExtractionConfig(controller_id=10, modules=modules, kernel=kernel)

    config = ExtractionConfig(controllers=[controller])

    file_path = tmp_path / "config.yaml"
    config.to_yaml(file_path=file_path)

    assert file_path.exists()

    loaded = ExtractionConfig.from_yaml(file_path=file_path)

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
    config.to_yaml(file_path=file_path)
    loaded = ExtractionConfig.from_yaml(file_path=file_path)

    assert loaded.controllers[0].kernel is None


def test_extraction_config_non_list_controllers() -> None:
    """Verifies that ExtractionConfig rejects a 'controllers' field that does not store a list."""
    message = (
        "Unable to initialize the ExtractionConfig instance. The 'controllers' field must store a list of "
        "ControllerExtractionConfig instances, but got NoneType."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        ExtractionConfig(controllers=None)  # type: ignore[arg-type]


def test_extraction_config_null_controllers_yaml(tmp_path: Path) -> None:
    """Verifies that loading a config whose 'controllers' key carries no value raises a ValueError."""
    config_path = tmp_path / EXTRACTION_CONFIGURATION_FILENAME
    config_path.write_text("controllers:\n", encoding="utf-8")

    message = (
        "Unable to initialize the ExtractionConfig instance. The 'controllers' field must store a list of "
        "ControllerExtractionConfig instances, but got NoneType."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        ExtractionConfig.from_yaml(file_path=config_path)


def test_write_microcontroller_manifest_new(tmp_path: Path) -> None:
    """Verifies that write_microcontroller_manifest creates a new manifest file."""
    modules = (ModuleSourceData(module_type=1, module_id=1, name="encoder"),)
    write_microcontroller_manifest(
        log_directory=tmp_path, controller_id=10, controller_name="actor_controller", modules=modules
    )

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    assert manifest_path.exists()

    loaded = MicroControllerManifest.from_yaml(file_path=manifest_path)
    assert len(loaded.controllers) == 1
    assert loaded.controllers[0].id == 10
    assert loaded.controllers[0].name == "actor_controller"


def test_write_microcontroller_manifest_append(tmp_path: Path) -> None:
    """Verifies that write_microcontroller_manifest appends a controller the manifest does not already carry."""
    modules_1 = (ModuleSourceData(module_type=1, module_id=1, name="encoder"),)
    modules_2 = (ModuleSourceData(module_type=2, module_id=1, name="lick_sensor"),)

    write_microcontroller_manifest(
        log_directory=tmp_path, controller_id=10, controller_name="controller_1", modules=modules_1
    )
    write_microcontroller_manifest(
        log_directory=tmp_path, controller_id=20, controller_name="controller_2", modules=modules_2
    )

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    loaded = MicroControllerManifest.from_yaml(file_path=manifest_path)

    assert len(loaded.controllers) == 2
    assert loaded.controllers[0].id == 10
    assert loaded.controllers[1].id == 20


def test_write_microcontroller_manifest_replaces_a_repeated_controller(tmp_path: Path) -> None:
    """Verifies that re-registering a controller id replaces its entry instead of adding a second one."""
    write_microcontroller_manifest(
        log_directory=tmp_path,
        controller_id=10,
        controller_name="controller_1",
        modules=(ModuleSourceData(module_type=1, module_id=1, name="encoder"),),
    )
    write_microcontroller_manifest(
        log_directory=tmp_path,
        controller_id=10,
        controller_name="controller_2",
        modules=(ModuleSourceData(module_type=2, module_id=1, name="lick_sensor"),),
    )

    loaded = MicroControllerManifest.from_yaml(file_path=tmp_path / MICROCONTROLLER_MANIFEST_FILENAME)

    assert len(loaded.controllers) == 1
    assert loaded.controllers[0].name == "controller_2"
    assert loaded.controllers[0].modules[0].name == "lick_sensor"


@pytest.mark.xdist_group(name="orchestration")
def test_write_microcontroller_manifest_serializes_concurrent_processes(tmp_path: Path) -> None:
    """Verifies that controllers registered from separate processes all survive in the manifest."""
    # The file lock serializes writers across process boundaries. Each writer reads the manifest, adds its own
    # entry, and writes the result back, so a writer that slips past the lock overwrites the entries written between
    # its own read and its own write. The barrier releases every writer into that sequence at once.
    controller_ids = (10, 20, 30, 40, 50, 60)
    barrier = multiprocessing.Barrier(parties=len(controller_ids))
    processes = [
        multiprocessing.Process(target=_register_controller, args=(tmp_path, controller_id, barrier))
        for controller_id in controller_ids
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=_CONCURRENCY_TIMEOUT)

    assert [process.exitcode for process in processes] == [0] * len(controller_ids)

    loaded = MicroControllerManifest.from_yaml(file_path=tmp_path / MICROCONTROLLER_MANIFEST_FILENAME)
    assert sorted(controller.id for controller in loaded.controllers) == sorted(controller_ids)


def test_write_microcontroller_manifest_serializes_concurrent_threads(tmp_path: Path) -> None:
    """Verifies that controllers registered from separate threads of one process all survive in the manifest."""
    # The threads that concurrent MCP tool calls run on reach this function as well, so the file lock has to serialize
    # them too.
    controller_ids = (10, 20, 30, 40, 50, 60)
    barrier = threading.Barrier(parties=len(controller_ids))
    failures: list[Exception] = []

    def register(controller_id: int) -> None:
        """Registers one controller and records the error a failed write raises, which a thread otherwise swallows."""
        try:
            _register_controller(log_directory=tmp_path, controller_id=controller_id, barrier=barrier)
        except Exception as error:
            failures.append(error)

    threads = [threading.Thread(target=register, args=(controller_id,)) for controller_id in controller_ids]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_CONCURRENCY_TIMEOUT)

    assert failures == []

    loaded = MicroControllerManifest.from_yaml(file_path=tmp_path / MICROCONTROLLER_MANIFEST_FILENAME)
    assert sorted(controller.id for controller in loaded.controllers) == sorted(controller_ids)


def test_write_microcontroller_manifest_times_out_on_a_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a manifest write aborts when another writer holds the lock past the timeout."""
    monkeypatch.setattr(dataclasses, "_MANIFEST_LOCK_TIMEOUT", 0.1)
    holder = FileLock(lock_file=str(tmp_path / f"{MICROCONTROLLER_MANIFEST_FILENAME}.lock"))

    with holder.acquire(timeout=_CONCURRENCY_TIMEOUT), pytest.raises(Timeout):
        write_microcontroller_manifest(
            log_directory=tmp_path,
            controller_id=10,
            controller_name="controller",
            modules=(ModuleSourceData(module_type=1, module_id=1, name="encoder"),),
        )


def test_create_extraction_config(tmp_path: Path) -> None:
    """Verifies that create_extraction_config generates a valid config from a manifest."""
    modules = (
        ModuleSourceData(module_type=1, module_id=1, name="encoder"),
        ModuleSourceData(module_type=2, module_id=1, name="lick_sensor"),
    )
    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(MicroControllerSourceData(id=10, name="controller", modules=modules))

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    manifest.to_yaml(file_path=manifest_path)

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
    manifest = MicroControllerManifest(controllers=[])
    manifest.controllers.append(
        MicroControllerSourceData(
            id=1, name="controller_1", modules=(ModuleSourceData(module_type=1, module_id=1, name="module_1"),)
        )
    )
    manifest.controllers.append(
        MicroControllerSourceData(
            id=2, name="controller_2", modules=(ModuleSourceData(module_type=2, module_id=1, name="module_2"),)
        )
    )

    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    manifest.to_yaml(file_path=manifest_path)

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
    manifest = MicroControllerManifest(controllers=[])
    manifest_path = tmp_path / MICROCONTROLLER_MANIFEST_FILENAME
    manifest.to_yaml(file_path=manifest_path)

    message = (
        f"Unable to create extraction config from '{manifest_path}'. The "
        f"{MICROCONTROLLER_MANIFEST_FILENAME} contains no controller entries."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        create_extraction_config(manifest_path=manifest_path)


def _register_controller(log_directory: Path, controller_id: int, barrier: Any) -> None:
    """Registers one controller in the shared manifest once every sibling worker has reached the barrier."""
    barrier.wait(timeout=_CONCURRENCY_TIMEOUT)
    write_microcontroller_manifest(
        log_directory=log_directory,
        controller_id=controller_id,
        controller_name=f"controller_{controller_id}",
        modules=(ModuleSourceData(module_type=1, module_id=1, name="encoder"),),
    )
