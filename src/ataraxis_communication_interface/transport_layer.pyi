from enum import IntEnum
from typing import Any

import numpy as np
from serial import Serial
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from serial.tools.list_ports_common import ListPortInfo as ListPortInfo

from .helper_modules import (
    SerialMock as SerialMock,
    CRCProcessor as CRCProcessor,
    COBSProcessor as COBSProcessor,
    CRCStatusCode as CRCStatusCode,
    COBSStatusCode as COBSStatusCode,
    _CRCProcessor as _CRCProcessor,
    _COBSProcessor as _COBSProcessor,
)

_ZERO: Incomplete
_POLYNOMIAL: Incomplete
_EMPTY_ARRAY: Incomplete
_MAXIMUM_BYTE_VALUE: int
_MAXIMUM_PAYLOAD_SIZE: int

class PacketParsingStatus(IntEnum):
    PACKET_SIZE_UNKNOWN = 0
    PACKET_PARSED = 1
    NOT_ENOUGH_PACKET_BYTES = 2
    NOT_ENOUGH_CRC_BYTES = 3
    NO_BYTES_TO_READ = 101
    NO_START_BYTE_FOUND = 102
    PAYLOAD_SIZE_MISMATCH = 103
    DELIMITER_FOUND_TOO_EARLY = 104
    DELIMITER_NOT_FOUND = 105

class DataManipulationCodes(IntEnum):
    INSUFFICIENT_BUFFER_SPACE_ERROR = 0
    MULTIDIMENSIONAL_ARRAY_ERROR = -1
    EMPTY_ARRAY_ERROR = -2

def list_available_ports() -> tuple[ListPortInfo, ...]: ...
def print_available_ports() -> None: ...

class TransportLayer:
    _accepted_numpy_scalars: tuple[
        type[np.uint8],
        type[np.uint16],
        type[np.uint32],
        type[np.uint64],
        type[np.int8],
        type[np.int16],
        type[np.int32],
        type[np.int64],
        type[np.float32],
        type[np.float64],
        type[np.bool],
    ]
    _opened: bool
    _port: SerialMock | Serial
    _crc_processor: Incomplete
    _cobs_processor: Incomplete
    _timer: Incomplete
    _start_byte: np.uint8
    _delimiter_byte: np.uint8
    _timeout: int
    _allow_start_byte_errors: bool
    _postamble_size: np.uint8
    _max_tx_payload_size: np.uint8
    _max_rx_payload_size: np.uint8
    _min_rx_payload_size: np.uint8
    _transmission_buffer: NDArray[np.uint8]
    _reception_buffer: NDArray[np.uint8]
    _minimum_packet_size: int
    _bytes_in_transmission_buffer: int
    _bytes_in_reception_buffer: int
    _leftover_bytes: bytes
    def __init__(
        self,
        port: str,
        microcontroller_serial_buffer_size: int,
        baudrate: int,
        polynomial: np.uint8 | np.uint16 | np.uint32 = ...,
        initial_crc_value: np.uint8 | np.uint16 | np.uint32 = ...,
        final_crc_xor_value: np.uint8 | np.uint16 | np.uint32 = ...,
        maximum_transmitted_payload_size: int = 0,
        minimum_received_payload_size: int = 1,
        start_byte: int = 129,
        delimiter_byte: int = 0,
        timeout: int = 20000,
        *,
        test_mode: bool = False,
        allow_start_byte_errors: bool = False,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def available(self) -> bool: ...
    @property
    def transmission_buffer(self) -> NDArray[np.uint8]: ...
    @property
    def reception_buffer(self) -> NDArray[np.uint8]: ...
    @property
    def bytes_in_transmission_buffer(self) -> int: ...
    @property
    def bytes_in_reception_buffer(self) -> int: ...
    def reset_transmission_buffer(self) -> None: ...
    def reset_reception_buffer(self) -> None: ...
    def write_data(self, data_object: Any, start_index: int | None = None) -> int: ...
    @staticmethod
    def _write_scalar_data(target_buffer: NDArray[np.uint8], scalar_object: Any, start_index: int) -> int: ...
    @staticmethod
    def _write_array_data(target_buffer: NDArray[np.uint8], array_object: NDArray[Any], start_index: int) -> int: ...
    def read_data(self, data_object: Any, start_index: int = 0) -> tuple[Any, int]: ...
    @staticmethod
    def _read_array_data(
        source_buffer: NDArray[np.uint8], array_object: NDArray[Any], start_index: int, payload_size: int
    ) -> tuple[NDArray[Any], int]: ...
    def send_data(self) -> bool: ...
    @staticmethod
    def _construct_packet(
        payload_buffer: NDArray[np.uint8],
        cobs_processor: _COBSProcessor,
        crc_processor: _CRCProcessor,
        payload_size: int,
        delimiter_byte: np.uint8,
        start_byte: np.uint8,
    ) -> NDArray[np.uint8]: ...
    def receive_data(self) -> bool: ...
    def _receive_packet(self) -> bool: ...
    def _bytes_available(self, required_bytes_count: int = 1, timeout: int = 0) -> bool: ...
    @staticmethod
    def _parse_packet(
        unparsed_bytes: NDArray[np.uint8],
        start_byte: np.uint8,
        delimiter_byte: np.uint8,
        max_payload_size: np.uint8,
        min_payload_size: np.uint8,
        postamble_size: np.uint8,
        allow_start_byte_errors: bool,
        start_found: bool = False,
        parsed_byte_count: int = 0,
        parsed_bytes: NDArray[np.uint8] = ...,
    ) -> tuple[int, int, NDArray[np.uint8], NDArray[np.uint8]]: ...
    @staticmethod
    def _validate_packet(
        reception_buffer: NDArray[np.uint8],
        packet_size: int,
        cobs_processor: _COBSProcessor,
        crc_processor: _CRCProcessor,
        delimiter_byte: np.uint8,
        postamble_size: np.uint8,
    ) -> int: ...
