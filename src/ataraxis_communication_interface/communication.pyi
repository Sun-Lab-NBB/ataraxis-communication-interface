from enum import IntEnum
from queue import Queue
from typing import Any
from dataclasses import field, dataclass
from collections.abc import Callable as Callable
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
import paho.mqtt.client as mqtt

_ZERO_BYTE: Incomplete
_ZERO_SHORT: Incomplete
_ZERO_LONG: Incomplete
_TRUE: Incomplete

class SerialProtocols(IntEnum):
    UNDEFINED = 0
    REPEATED_MODULE_COMMAND = 1
    ONE_OFF_MODULE_COMMAND = 2
    DEQUEUE_MODULE_COMMAND = 3
    KERNEL_COMMAND = 4
    MODULE_PARAMETERS = 5
    MODULE_DATA = 6
    KERNEL_DATA = 7
    MODULE_STATE = 8
    KERNEL_STATE = 9
    RECEPTION_CODE = 10
    CONTROLLER_IDENTIFICATION = 11
    MODULE_IDENTIFICATION = 12
    def as_uint8(self) -> np.uint8: ...

type PrototypeType = (
    np.bool_
    | np.uint8
    | np.int8
    | np.uint16
    | np.int16
    | np.uint32
    | np.int32
    | np.uint64
    | np.int64
    | np.float32
    | np.float64
    | NDArray[np.bool_]
    | NDArray[np.uint8]
    | NDArray[np.int8]
    | NDArray[np.uint16]
    | NDArray[np.int16]
    | NDArray[np.uint32]
    | NDArray[np.int32]
    | NDArray[np.uint64]
    | NDArray[np.int64]
    | NDArray[np.float32]
    | NDArray[np.float64]
)

_PROTOTYPE_FACTORIES: dict[int, Callable[[], PrototypeType]]

class SerialPrototypes(IntEnum):
    ONE_BOOL = 1
    ONE_UINT8 = 2
    ONE_INT8 = 3
    TWO_BOOLS = 4
    TWO_UINT8S = 5
    TWO_INT8S = 6
    ONE_UINT16 = 7
    ONE_INT16 = 8
    THREE_BOOLS = 9
    THREE_UINT8S = 10
    THREE_INT8S = 11
    FOUR_BOOLS = 12
    FOUR_UINT8S = 13
    FOUR_INT8S = 14
    TWO_UINT16S = 15
    TWO_INT16S = 16
    ONE_UINT32 = 17
    ONE_INT32 = 18
    ONE_FLOAT32 = 19
    FIVE_BOOLS = 20
    FIVE_UINT8S = 21
    FIVE_INT8S = 22
    SIX_BOOLS = 23
    SIX_UINT8S = 24
    SIX_INT8S = 25
    THREE_UINT16S = 26
    THREE_INT16S = 27
    SEVEN_BOOLS = 28
    SEVEN_UINT8S = 29
    SEVEN_INT8S = 30
    EIGHT_BOOLS = 31
    EIGHT_UINT8S = 32
    EIGHT_INT8S = 33
    FOUR_UINT16S = 34
    FOUR_INT16S = 35
    TWO_UINT32S = 36
    TWO_INT32S = 37
    TWO_FLOAT32S = 38
    ONE_UINT64 = 39
    ONE_INT64 = 40
    ONE_FLOAT64 = 41
    NINE_BOOLS = 42
    NINE_UINT8S = 43
    NINE_INT8S = 44
    TEN_BOOLS = 45
    TEN_UINT8S = 46
    TEN_INT8S = 47
    FIVE_UINT16S = 48
    FIVE_INT16S = 49
    ELEVEN_BOOLS = 50
    ELEVEN_UINT8S = 51
    ELEVEN_INT8S = 52
    TWELVE_BOOLS = 53
    TWELVE_UINT8S = 54
    TWELVE_INT8S = 55
    SIX_UINT16S = 56
    SIX_INT16S = 57
    THREE_UINT32S = 58
    THREE_INT32S = 59
    THREE_FLOAT32S = 60
    THIRTEEN_BOOLS = 61
    THIRTEEN_UINT8S = 62
    THIRTEEN_INT8S = 63
    FOURTEEN_BOOLS = 64
    FOURTEEN_UINT8S = 65
    FOURTEEN_INT8S = 66
    SEVEN_UINT16S = 67
    SEVEN_INT16S = 68
    FIFTEEN_BOOLS = 69
    FIFTEEN_UINT8S = 70
    FIFTEEN_INT8S = 71
    EIGHT_UINT16S = 72
    EIGHT_INT16S = 73
    FOUR_UINT32S = 74
    FOUR_INT32S = 75
    FOUR_FLOAT32S = 76
    TWO_UINT64S = 77
    TWO_INT64S = 78
    TWO_FLOAT64S = 79
    NINE_UINT16S = 80
    NINE_INT16S = 81
    TEN_UINT16S = 82
    TEN_INT16S = 83
    FIVE_UINT32S = 84
    FIVE_INT32S = 85
    FIVE_FLOAT32S = 86
    ELEVEN_UINT16S = 87
    ELEVEN_INT16S = 88
    TWELVE_UINT16S = 89
    TWELVE_INT16S = 90
    SIX_UINT32S = 91
    SIX_INT32S = 92
    SIX_FLOAT32S = 93
    THREE_UINT64S = 94
    THREE_INT64S = 95
    THREE_FLOAT64S = 96
    THIRTEEN_UINT16S = 97
    THIRTEEN_INT16S = 98
    FOURTEEN_UINT16S = 99
    FOURTEEN_INT16S = 100
    SEVEN_UINT32S = 101
    SEVEN_INT32S = 102
    SEVEN_FLOAT32S = 103
    FIFTEEN_UINT16S = 104
    FIFTEEN_INT16S = 105
    EIGHT_UINT32S = 106
    EIGHT_INT32S = 107
    EIGHT_FLOAT32S = 108
    FOUR_UINT64S = 109
    FOUR_INT64S = 110
    FOUR_FLOAT64S = 111
    NINE_UINT32S = 112
    NINE_INT32S = 113
    NINE_FLOAT32S = 114
    TEN_UINT32S = 115
    TEN_INT32S = 116
    TEN_FLOAT32S = 117
    FIVE_UINT64S = 118
    FIVE_INT64S = 119
    FIVE_FLOAT64S = 120
    ELEVEN_UINT32S = 121
    ELEVEN_INT32S = 122
    ELEVEN_FLOAT32S = 123
    TWELVE_UINT32S = 124
    TWELVE_INT32S = 125
    TWELVE_FLOAT32S = 126
    SIX_UINT64S = 127
    SIX_INT64S = 128
    SIX_FLOAT64S = 129
    THIRTEEN_UINT32S = 130
    THIRTEEN_INT32S = 131
    THIRTEEN_FLOAT32S = 132
    FOURTEEN_UINT32S = 133
    FOURTEEN_INT32S = 134
    FOURTEEN_FLOAT32S = 135
    SEVEN_UINT64S = 136
    SEVEN_INT64S = 137
    SEVEN_FLOAT64S = 138
    FIFTEEN_UINT32S = 139
    FIFTEEN_INT32S = 140
    FIFTEEN_FLOAT32S = 141
    EIGHT_UINT64S = 142
    EIGHT_INT64S = 143
    EIGHT_FLOAT64S = 144
    NINE_UINT64S = 145
    NINE_INT64S = 146
    NINE_FLOAT64S = 147
    TEN_UINT64S = 148
    TEN_INT64S = 149
    TEN_FLOAT64S = 150
    ELEVEN_UINT64S = 151
    ELEVEN_INT64S = 152
    ELEVEN_FLOAT64S = 153
    TWELVE_UINT64S = 154
    TWELVE_INT64S = 155
    TWELVE_FLOAT64S = 156
    THIRTEEN_UINT64S = 157
    THIRTEEN_INT64S = 158
    THIRTEEN_FLOAT64S = 159
    FOURTEEN_UINT64S = 160
    FOURTEEN_INT64S = 161
    FOURTEEN_FLOAT64S = 162
    FIFTEEN_UINT64S = 163
    FIFTEEN_INT64S = 164
    FIFTEEN_FLOAT64S = 165
    def as_uint8(self) -> np.uint8: ...
    def get_prototype(self) -> PrototypeType: ...
    @classmethod
    def get_prototype_for_code(cls, code: np.uint8) -> PrototypeType | None: ...

@dataclass(frozen=True)
class RepeatedModuleCommand:
    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
    cycle_delay: np.uint32 = ...
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.REPEATED_MODULE_COMMAND.as_uint8())
    def __post_init__(self) -> None: ...
    def __repr__(self) -> str: ...

@dataclass(frozen=True)
class OneOffModuleCommand:
    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.ONE_OFF_MODULE_COMMAND.as_uint8())
    def __post_init__(self) -> None: ...
    def __repr__(self) -> str: ...

@dataclass(frozen=True)
class DequeueModuleCommand:
    module_type: np.uint8
    module_id: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.DEQUEUE_MODULE_COMMAND.as_uint8())
    def __post_init__(self) -> None: ...
    def __repr__(self) -> str: ...

@dataclass(frozen=True)
class KernelCommand:
    command: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.KERNEL_COMMAND.as_uint8())
    def __post_init__(self) -> None: ...
    def __repr__(self) -> str: ...

@dataclass(frozen=True)
class ModuleParameters:
    module_type: np.uint8
    module_id: np.uint8
    parameter_data: tuple[np.number[Any] | np.bool, ...]
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    parameters_size: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.MODULE_PARAMETERS.as_uint8())
    def __post_init__(self) -> None: ...
    def __repr__(self) -> str: ...

@dataclass
class ModuleData:
    message: NDArray[np.uint8] = field(default_factory=Incomplete)
    data_object: np.number[Any] | NDArray[Any] = ...
    def __repr__(self) -> str: ...
    @property
    def module_type(self) -> np.uint8: ...
    @property
    def module_id(self) -> np.uint8: ...
    @property
    def command(self) -> np.uint8: ...
    @property
    def event(self) -> np.uint8: ...
    @property
    def prototype_code(self) -> np.uint8: ...

@dataclass
class KernelData:
    message: NDArray[np.uint8] = field(default_factory=Incomplete)
    data_object: np.number[Any] | NDArray[Any] = ...
    def __repr__(self) -> str: ...
    @property
    def command(self) -> np.uint8: ...
    @property
    def event(self) -> np.uint8: ...
    @property
    def prototype_code(self) -> np.uint8: ...

@dataclass
class ModuleState:
    message: NDArray[np.uint8] = field(default_factory=Incomplete)
    def __repr__(self) -> str: ...
    @property
    def module_type(self) -> np.uint8: ...
    @property
    def module_id(self) -> np.uint8: ...
    @property
    def command(self) -> np.uint8: ...
    @property
    def event(self) -> np.uint8: ...

@dataclass
class KernelState:
    message: NDArray[np.uint8] = field(default_factory=Incomplete)
    def __repr__(self) -> str: ...
    @property
    def command(self) -> np.uint8: ...
    @property
    def event(self) -> np.uint8: ...

@dataclass
class ReceptionCode:
    message: NDArray[np.uint8] = field(default_factory=Incomplete)
    def __repr__(self) -> str: ...
    @property
    def reception_code(self) -> np.uint8: ...

@dataclass
class ControllerIdentification:
    message: NDArray[np.uint8] = field(default_factory=Incomplete)
    def __repr__(self) -> str: ...
    @property
    def controller_id(self) -> np.uint8: ...

@dataclass
class ModuleIdentification:
    module_type_id: np.uint16 = ...
    def __repr__(self) -> str: ...

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

class MQTTCommunication:
    _ip: str
    _port: int
    _connected: bool
    _monitored_topics: tuple[str, ...]
    _output_queue: Queue
    _client: mqtt.Client
    def __init__(
        self, ip: str = "127.0.0.1", port: int = 1883, monitored_topics: None | tuple[str, ...] = None
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __del__(self) -> None: ...
    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None: ...
    def connect(self) -> None: ...
    def send_data(self, topic: str, payload: str | bytes | bytearray | float | None = None) -> None: ...
    @property
    def has_data(self) -> bool: ...
    def get_data(self) -> tuple[str, bytes | bytearray] | None: ...
    def disconnect(self) -> None: ...
