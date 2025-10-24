import abc
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any
from pathlib import Path
from threading import Thread
from dataclasses import dataclass
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from multiprocessing.managers import SyncManager

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_data_structures import DataLogger, SharedMemoryArray

from .communication import (
    KernelData as KernelData,
    ModuleData as ModuleData,
    KernelState as KernelState,
    ModuleState as ModuleState,
    KernelCommand as KernelCommand,
    ReceptionCode as ReceptionCode,
    SerialProtocols as SerialProtocols,
    ModuleParameters as ModuleParameters,
    SerialPrototypes as SerialPrototypes,
    OneOffModuleCommand as OneOffModuleCommand,
    SerialCommunication as SerialCommunication,
    DequeueModuleCommand as DequeueModuleCommand,
    ModuleIdentification as ModuleIdentification,
    RepeatedModuleCommand as RepeatedModuleCommand,
    ControllerIdentification as ControllerIdentification,
)

_MAXIMUM_BYTE_VALUE: int
_ZERO_BYTE: Incomplete
_ZERO_BOOL: Incomplete
_ZERO_LONG: Incomplete

class _RuntimeParameters(IntEnum):
    KEEPALIVE_RETURN_CODE = 255
    SERVICE_CODE_THRESHOLD = 50
    PROCESS_INITIALIZATION_TIMEOUT = 30
    MICROCONTROLLER_ID_TIMEOUT = 2000
    MAXIMUM_COMMUNICATION_ATTEMPTS = 3
    PROCESS_TERMINATION_TIMEOUT = 60
    WATCHDOG_INTERVAL = 20
    PARALLEL_PROCESSING_THRESHOLD = 2000
    MINIMUM_MODULE_DATA_SIZE = 5

class _KernelStatusCodes(IntEnum):
    MODULE_SETUP_ERROR = 2
    RECEPTION_ERROR = 3
    TRANSMISSION_ERROR = 4
    INVALID_MESSAGE_PROTOCOL = 5
    MODULE_PARAMETERS_ERROR = 7
    COMMAND_NOT_RECOGNIZED = 8
    TARGET_MODULE_NOT_FOUND = 9
    KEEPALIVE_TIMEOUT = 10

class _ModuleStatusCodes(IntEnum):
    TRANSMISSION_ERROR = 1
    COMMAND_COMPLETE = 2
    COMMAND_NOT_RECOGNIZED = 3

class ModuleInterface(ABC, metaclass=abc.ABCMeta):
    _module_type: np.uint8
    _module_id: np.uint8
    _type_id: np.uint16
    _data_codes: set[np.uint8]
    _error_codes: set[np.uint8]
    _input_queue: MPQueue | None
    _dequeue_command: Incomplete
    _create_command_message: Incomplete
    _create_parameters_message: Incomplete
    def __init__(
        self,
        module_type: np.uint8,
        module_id: np.uint8,
        error_codes: set[np.uint8] | None = None,
        data_codes: set[np.uint8] | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @abstractmethod
    def initialize_remote_assets(self) -> None: ...
    @abstractmethod
    def terminate_remote_assets(self) -> None: ...
    @abstractmethod
    def process_received_data(self, message: ModuleData | ModuleState) -> None: ...
    def _create_command_message_implementation(
        self, command: np.uint8, noblock: np.bool_, cycle_delay: np.uint32
    ) -> OneOffModuleCommand | RepeatedModuleCommand: ...
    def _create_parameters_message_implementation(
        self, parameter_data: tuple[np.number[Any] | np.bool_, ...]
    ) -> ModuleParameters: ...
    def send_command(self, command: np.uint8, noblock: np.bool, repetition_delay: np.uint32 = ...) -> None: ...
    def send_parameters(
        self, parameter_data: tuple[np.unsignedinteger[Any] | np.signedinteger[Any] | np.bool | np.floating[Any], ...]
    ) -> None: ...
    def reset_command_queue(self) -> None: ...
    def set_input_queue(self, input_queue: MPQueue) -> None: ...
    @property
    def module_type(self) -> np.uint8: ...
    @property
    def module_id(self) -> np.uint8: ...
    @property
    def type_id(self) -> np.uint16: ...
    @property
    def data_codes(self) -> set[np.uint8]: ...
    @property
    def error_codes(self) -> set[np.uint8]: ...

class MicroControllerInterface:
    _reset_command: Incomplete
    _started: bool
    _mp_manager: SyncManager
    _controller_id: np.uint8
    _port: str
    _baudrate: int
    _buffer_size: int
    _modules: tuple[ModuleInterface, ...]
    _logger_queue: MPQueue
    _log_directory: Path
    _input_queue: MPQueue
    _terminator_array: None | SharedMemoryArray
    _communication_process: None | Process
    _watchdog_thread: None | Thread
    _keepalive_interval: Incomplete
    def __init__(
        self,
        controller_id: np.uint8,
        data_logger: DataLogger,
        module_interfaces: tuple[ModuleInterface, ...],
        buffer_size: int,
        port: str,
        baudrate: int = 115200,
        keepalive_interval: int = 0,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __del__(self) -> None: ...
    def reset_controller(self) -> None: ...
    def _watchdog(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @staticmethod
    def _verify_microcontroller_communication(
        serial_communication: SerialCommunication,
        timeout_timer: PrecisionTimer,
        controller_id: np.uint8,
        module_interfaces: tuple[ModuleInterface, ...],
        terminator_array: SharedMemoryArray,
    ) -> None: ...
    @staticmethod
    def _parse_kernel_data(controller_id: np.uint8, in_data: KernelState | KernelData) -> None: ...
    @staticmethod
    def _parse_service_module_data(controller_id: np.uint8, in_data: ModuleState | ModuleData) -> None: ...
    @staticmethod
    def _runtime_cycle(
        controller_id: np.uint8,
        module_interfaces: tuple[ModuleInterface, ...],
        input_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        port: str,
        baudrate: int,
        buffer_size: int,
        keepalive_interval: int,
    ) -> None: ...

@dataclass()
class ExtractedMessageData:
    timestamp: np.uint64
    command: np.uint8
    data: None | np.number | NDArray[np.number]

@dataclass()
class ExtractedModuleData:
    module_type: int
    module_id: int
    event_data: dict[np.uint8, tuple[ExtractedMessageData, ...]]

def _process_module_message_batch(
    log_path: Path, file_names: list[str], onset_us: np.uint64, module_type_id: tuple[tuple[int, int], ...]
) -> dict[tuple[int, int], dict[np.uint8, list[ExtractedMessageData]]]: ...
def extract_logged_hardware_module_data(
    log_path: Path, module_type_id: tuple[tuple[int, int], ...], n_workers: int = -1
) -> tuple[ExtractedModuleData, ...]: ...
