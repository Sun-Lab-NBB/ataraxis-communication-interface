"""Provides synthetic log archive, manifest, and extraction configuration builders shared by the orchestration and
extraction test modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX

from ataraxis_communication_interface.communication import SerialProtocols, SerialPrototypes
from ataraxis_communication_interface.microcontroller.dataclasses import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    ModuleSourceData,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    MicroControllerManifest,
    MicroControllerSourceData,
    ControllerExtractionConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

DEFAULT_ONSET_US: int = 1_000_000_000_000


def create_test_archive(
    archive_path: Path,
    source_id: int,
    messages: list[tuple[int, NDArray[np.uint8]]],
    onset_us: int = DEFAULT_ONSET_US,
) -> None:
    """Creates a test .npz log archive with the given messages."""
    entries: dict[str, NDArray[np.uint8]] = {}

    key, data = _create_onset_entry(source_id=source_id, onset_us=onset_us)
    entries[key] = data

    for elapsed_us, payload in messages:
        key, data = _create_archive_entry(source_id=source_id, elapsed_us=elapsed_us, payload=payload)
        entries[key] = data

    np.savez(str(archive_path), **entries)


def create_module_state_payload(module_type: int, module_id: int, command: int, event: int) -> NDArray[np.uint8]:
    """Creates a MODULE_STATE message payload."""
    return np.array([SerialProtocols.MODULE_STATE, module_type, module_id, command, event], dtype=np.uint8)


def create_module_data_payload(
    module_type: int, module_id: int, command: int, event: int, prototype_code: int, data_bytes: list[int]
) -> NDArray[np.uint8]:
    """Creates a MODULE_DATA message payload."""
    header = [SerialProtocols.MODULE_DATA, module_type, module_id, command, event, prototype_code]
    return np.array(header + data_bytes, dtype=np.uint8)


def create_kernel_state_payload(command: int, event: int) -> NDArray[np.uint8]:
    """Creates a KERNEL_STATE message payload."""
    return np.array([SerialProtocols.KERNEL_STATE, command, event], dtype=np.uint8)


def create_kernel_data_payload(
    command: int, event: int, prototype_code: int, data_bytes: list[int]
) -> NDArray[np.uint8]:
    """Creates a KERNEL_DATA message payload."""
    header = [SerialProtocols.KERNEL_DATA, command, event, prototype_code]
    return np.array(header + data_bytes, dtype=np.uint8)


def write_extraction_config(
    config_path: Path,
    source_id: int,
    module_type: int = 1,
    module_id: int = 2,
    event_codes: tuple[int, ...] = (10, 20),
    kernel_event_codes: tuple[int, ...] | None = None,
) -> Path:
    """Writes an extraction configuration registering one controller with one module."""
    config = ExtractionConfig(
        controllers=[
            ControllerExtractionConfig(
                controller_id=source_id,
                modules=(
                    ModuleExtractionConfig(module_type=module_type, module_id=module_id, event_codes=event_codes),
                ),
                kernel=None if kernel_event_codes is None else KernelExtractionConfig(event_codes=kernel_event_codes),
            )
        ]
    )
    config.to_yaml(file_path=config_path)
    return config_path


def setup_test_environment(
    tmp_path: Path,
    source_id: int = 1,
    module_type: int = 1,
    module_id: int = 2,
) -> tuple[Path, Path, Path]:
    """Creates a complete test environment with an archive, a manifest, and an extraction configuration."""
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    output_directory = tmp_path / "output"
    output_directory.mkdir()

    archive_path = log_directory / f"{source_id}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        (1000, create_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (2000, create_module_state_payload(module_type=module_type, module_id=module_id, command=1, event=10)),
        (
            3000,
            create_module_data_payload(
                module_type=module_type,
                module_id=module_id,
                command=2,
                event=20,
                prototype_code=SerialPrototypes.ONE_UINT8,
                data_bytes=[42],
            ),
        ),
    ]
    create_test_archive(archive_path=archive_path, source_id=source_id, messages=messages)

    _write_manifest(log_directory=log_directory, source_id=source_id, module_type=module_type, module_id=module_id)

    config_path = tmp_path / "config.yaml"
    write_extraction_config(config_path=config_path, source_id=source_id, module_type=module_type, module_id=module_id)

    return log_directory, config_path, output_directory


def _create_archive_entry(source_id: int, elapsed_us: int, payload: NDArray[np.uint8]) -> tuple[str, NDArray[np.uint8]]:
    """Creates a single archive entry (key, data) in the LogPackage format."""
    header = np.empty(9, dtype=np.uint8)
    header[0] = np.uint8(source_id)
    header[1:9] = np.frombuffer(np.uint64(elapsed_us).tobytes(), dtype=np.uint8)
    data = np.concatenate([header, payload.astype(np.uint8)])
    key = f"{source_id:03d}_{elapsed_us:020d}"
    return key, data


def _create_onset_entry(source_id: int, onset_us: int) -> tuple[str, NDArray[np.uint8]]:
    """Creates the onset timestamp entry for a log archive."""
    onset_payload = np.frombuffer(np.uint64(onset_us).tobytes(), dtype=np.uint8)
    return _create_archive_entry(source_id=source_id, elapsed_us=0, payload=onset_payload)


def _write_manifest(log_directory: Path, source_id: int, module_type: int = 1, module_id: int = 2) -> Path:
    """Writes a microcontroller manifest registering one controller with one module."""
    manifest = MicroControllerManifest(
        controllers=[
            MicroControllerSourceData(
                id=source_id,
                name="test_controller",
                modules=(ModuleSourceData(module_type=module_type, module_id=module_id, name="test_module"),),
            )
        ]
    )
    manifest_path = log_directory / MICROCONTROLLER_MANIFEST_FILENAME
    manifest.to_yaml(file_path=manifest_path)
    return manifest_path
