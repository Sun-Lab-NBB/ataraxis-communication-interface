from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer

from .messages import (
    KernelData as KernelData,
    ModuleData as ModuleData,
    KernelState as KernelState,
    ModuleState as ModuleState,
    KernelCommand as KernelCommand,
    ReceptionCode as ReceptionCode,
    ModuleParameters as ModuleParameters,
    OneOffModuleCommand as OneOffModuleCommand,
    DequeueModuleCommand as DequeueModuleCommand,
    ModuleIdentification as ModuleIdentification,
    RepeatedModuleCommand as RepeatedModuleCommand,
    ControllerIdentification as ControllerIdentification,
)
from .protocols import (
    SerialProtocols as SerialProtocols,
    SerialPrototypes as SerialPrototypes,
)

_PROTOCOL_MODULE_DATA: np.uint8
_PROTOCOL_KERNEL_DATA: np.uint8
_PROTOCOL_MODULE_STATE: np.uint8
_PROTOCOL_KERNEL_STATE: np.uint8
_PROTOCOL_RECEPTION_CODE: np.uint8
_PROTOCOL_CONTROLLER_IDENTIFICATION: np.uint8
_PROTOCOL_MODULE_IDENTIFICATION: np.uint8

class SerialCommunication:
    _transport_layer: Incomplete
    _module_data: Incomplete
    _kernel_data: Incomplete
    _module_state: Incomplete
    _kernel_state: Incomplete
    _controller_identification: Incomplete
    _module_identification: Incomplete
    _reception_code: Incomplete
    _timestamp_timer: PrecisionTimer
    _source_id: np.uint8
    _logger_queue: MPQueue
    _usb_port: str
    def __init__(
        self,
        controller_id: np.uint8,
        microcontroller_serial_buffer_size: int,
        port: str,
        logger_queue: MPQueue,
        baudrate: int = 115200,
        *,
        test_mode: bool = False,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def send_message(
        self,
        message: RepeatedModuleCommand | OneOffModuleCommand | DequeueModuleCommand | KernelCommand | ModuleParameters,
    ) -> None: ...
    def receive_message(
        self,
    ) -> (
        ModuleData
        | ModuleState
        | KernelData
        | KernelState
        | ControllerIdentification
        | ModuleIdentification
        | ReceptionCode
        | None
    ): ...
    def _log_data(self, timestamp: int, data: NDArray[np.uint8]) -> None: ...
