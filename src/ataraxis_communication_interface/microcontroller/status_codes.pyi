from enum import IntEnum

import numpy as np

from ..communication import (
    KernelData as KernelData,
    ModuleData as ModuleData,
    KernelState as KernelState,
    ModuleState as ModuleState,
    SerialProtocols as SerialProtocols,
)

MINIMUM_CUSTOM_STATUS_CODE: int
MAXIMUM_CUSTOM_STATUS_CODE: int
_UNRECOGNIZED_CODE_NAME: str

class KernelCommandCodes(IntEnum):
    STANDBY = 0
    RECEIVE_DATA = 1
    RESET_CONTROLLER = 2
    IDENTIFY_CONTROLLER = 3
    IDENTIFY_MODULES = 4
    KEEPALIVE = 5

class KernelStatusCodes(IntEnum):
    STANDBY = 0
    SETUP_COMPLETE = 1
    MODULE_SETUP_ERROR = 2
    RECEPTION_ERROR = 3
    TRANSMISSION_ERROR = 4
    INVALID_MESSAGE_PROTOCOL = 5
    MODULE_PARAMETERS_SET = 6
    MODULE_PARAMETERS_ERROR = 7
    COMMAND_NOT_RECOGNIZED = 8
    TARGET_MODULE_NOT_FOUND = 9
    KEEPALIVE_TIMEOUT = 10

class ModuleStatusCodes(IntEnum):
    STANDBY = 0
    TRANSMISSION_ERROR = 1
    COMMAND_COMPLETED = 2
    COMMAND_NOT_RECOGNIZED = 3

class CommunicationStatusCodes(IntEnum):
    STANDBY = 51
    RECEPTION_ERROR = 52
    PARSING_ERROR = 53
    PACKING_ERROR = 54
    MESSAGE_SENT = 55
    MESSAGE_RECEIVED = 56
    INVALID_PROTOCOL = 57
    NO_BYTES_TO_RECEIVE = 58
    PARAMETER_MISMATCH = 59
    PARAMETERS_EXTRACTED = 60
    EXTRACTION_FORBIDDEN = 61
    TRANSMISSION_ERROR = 62

class TransportStatusCodes(IntEnum):
    STANDBY = 11
    DECODING_FAILED = 12
    PACKET_SENT = 13
    PAYLOAD_SIZE_BYTE_NOT_FOUND = 14
    INVALID_PAYLOAD_SIZE = 15
    PACKET_TIMEOUT_ERROR = 16
    NO_BYTES_TO_PARSE = 17
    PACKET_PARSED = 18
    CRC_CHECK_FAILED = 19
    PACKET_RECEIVED = 20
    WRITE_OBJECT_BUFFER_ERROR = 21
    OBJECT_WRITTEN_TO_BUFFER = 22
    READ_OBJECT_BUFFER_ERROR = 23
    OBJECT_READ_FROM_BUFFER = 24
    DELIMITER_NOT_FOUND_ERROR = 25
    DELIMITER_FOUND_TOO_EARLY_ERROR = 26
    POSTAMBLE_TIMEOUT_ERROR = 27
    EMPTY_PAYLOAD_ERROR = 28
    PACKET_PARTIALLY_SENT = 29

_KERNEL_ERROR_DESCRIPTIONS: dict[int, str]
_MODULE_ERROR_DESCRIPTIONS: dict[int, str]
_KERNEL_STATUS_VALUES: frozenset[int]
_MODULE_STATUS_VALUES: frozenset[int]
_COMMUNICATION_STATUS_MEANINGS: dict[int, str]
_TRANSPORT_STATUS_MEANINGS: dict[int, str]

def describe_kernel_event(
    message: KernelData | KernelState, controller_id: np.uint8, controller_name: str
) -> str | None: ...
def describe_module_event(
    message: ModuleData | ModuleState, controller_id: np.uint8, controller_name: str, module_name: str
) -> str | None: ...
def describe_custom_module_error(
    message: ModuleData | ModuleState, controller_id: np.uint8, controller_name: str, module_name: str, description: str
) -> str: ...
def _format_code(code: int, code_type: type[IntEnum]) -> str: ...
def _format_module_context(
    message: ModuleData | ModuleState, controller_id: np.uint8, controller_name: str, module_name: str
) -> str: ...
def _describe_unrecognized_code(code: int, code_type: type[IntEnum]) -> str: ...
def _describe_kernel_payload(message: KernelData | KernelState, event: int) -> str: ...
def _describe_serial_failure(message: KernelData | ModuleData | ModuleState) -> str: ...
def _format_serial_status(layer: str, code: int, code_type: type[IntEnum], meanings: dict[int, str]) -> str: ...
def _resolve_payload_values(message: KernelData | ModuleData, count: int) -> tuple[int, ...] | None: ...
def _join_sentences(*sentences: str) -> str: ...
