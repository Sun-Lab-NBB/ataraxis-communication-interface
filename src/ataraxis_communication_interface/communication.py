"""This module provides the SerialCommunication and MQTTCommunication classes that enable communication between
the PC, Arduino / Teensy microcontrollers running Ataraxis software and the other infrastructure with MQTT bindings.

SerialCommunication supports the PC-MicroController communication over USB / UART interface, while MQTTCommunication
supports the communication over the MQTT protocol (virtual / real TCP sockets). MQTTCommunication can be used to
transmit data locally (within the same PC) or remotely via a network connection.

Additionally, this module exposes message and helper structures used to serialize and deserialize the transmitted data.
"""

from enum import IntEnum
from queue import Queue
from typing import Any, Dict, Union
from dataclasses import field, dataclass
from collections.abc import Callable
from multiprocessing import Queue as MPQueue

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
import paho.mqtt.client as mqtt
from ataraxis_base_utilities import console
from ataraxis_data_structures import LogPackage
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_transport_layer_pc import TransportLayer


class SerialProtocols(IntEnum):
    """Stores the protocol codes used in data transmission between the PC and the microcontroller over the serial port.

    Each sent and received message starts with the specific protocol code from this enumeration that instructs the
    receiver on how to process the rest of the data payload. The codes available through this class have to match the
    contents of the kProtocols Enumeration available from the ataraxis-micro-controller library
    (axmc_communication_assets namespace).

    Notes:
        The values available through this enumeration should be read through their 'as_uint8' property to enforce the
        type expected by other classes from ths library.
    """

    UNDEFINED = 0
    """Not a valid protocol code. This is used to initialize the Communication class of the microcontroller."""

    REPEATED_MODULE_COMMAND = 1
    """Protocol for sending Module-addressed commands that should be repeated (executed recurrently)."""

    ONE_OFF_MODULE_COMMAND = 2
    """Protocol for sending Module-addressed commands that should not be repeated (executed only once)."""

    DEQUEUE_MODULE_COMMAND = 3
    """Protocol for sending Module-addressed commands that remove all queued commands (including recurrent commands)."""

    KERNEL_COMMAND = 4
    """Protocol for sending Kernel-addressed commands. All Kernel commands are always non-repeatable (one-shot)."""

    MODULE_PARAMETERS = 5
    """Protocol for sending Module-addressed parameters. This relies on transmitting arbitrary sized parameter objects 
    likely to be unique for each module type (family)."""

    KERNEL_PARAMETERS = 6
    """Protocol for sending Kernel-addressed parameters. The parameters transmitted via these messages will be used to 
    overwrite the global parameters shared by the Kernel and all Modules of the microcontroller (global runtime 
    parameters)."""

    MODULE_DATA = 7
    """Protocol for receiving Module-sent data or error messages that include an arbitrary data object in addition to 
    event state-code."""

    KERNEL_DATA = 8
    """Protocol for receiving Kernel-sent data or error messages that include an arbitrary data object in addition to 
    event state-code."""

    MODULE_STATE = 9
    """Protocol for receiving Module-sent data or error messages that do not include additional data objects."""

    KERNEL_STATE = 10
    """Protocol for receiving Kernel-sent data or error messages that do not include additional data objects."""

    RECEPTION_CODE = 11
    """Protocol used to ensure that the microcontroller has received a previously sent command or parameter message. 
    Specifically, when an outgoing message includes a reception_code, this code is transmitted back to the PC using 
    this service protocol to acknowledge message reception. Currently, this protocol is only intended for testing 
    purposes, as at this time the Communication class does not explicitly ensure message delivery."""

    CONTROLLER_IDENTIFICATION = 12
    """Protocol used to identify the controller connected to a particular USB port. This service protocol is used by 
    the controller that receives the 'IdentifyController' Kernel-addressed command and replies with it's ID code. 
    This protocol is automatically used by the MicrocontrollerInterface class during initialization and should not be 
    used manually."""

    MODULE_IDENTIFICATION = 13
    """Protocol used to identify all hardware module instances managed by the connected microcontroller. This service 
    protocol is used by the controller that receives the 'IdentifyModules' Kernel-addressed command and sequentially 
    transmits the combined type_id uint16 code for each managed module instance. This protocol is automatically used
    by the MicrocontrollerInterface class during initialization and should not be used manually."""

    def as_uint8(self) -> np.uint8:
        """Convert the enum value to numpy.uint8 type.

        Returns:
            np.uint8: The enum value as a numpy unsigned 8-bit integer.
        """
        return np.uint8(self.value)


# Type alias for supported numpy types
NumericType = Union[
    np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32, np.float64
]
PrototypeType = Union[
    NumericType,
    NDArray[np.bool_],
    NDArray[np.uint8],
    NDArray[np.int8],
    NDArray[np.uint16],
    NDArray[np.int16],
    NDArray[np.uint32],
    NDArray[np.int32],
    NDArray[np.uint64],
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float64],
]


# Defines prototype factories for all supported types
_PROTOTYPE_FACTORIES: Dict[int, Callable[[], PrototypeType]] = {
    # 1 byte total
    1: lambda: np.bool_(0),  # kOneBool
    2: lambda: np.uint8(0),  # kOneUint8
    3: lambda: np.int8(0),  # kOneInt8
    # 2 bytes total
    4: lambda: np.zeros(2, dtype=np.bool_),  # kTwoBools
    5: lambda: np.zeros(2, dtype=np.uint8),  # kTwoUint8s
    6: lambda: np.zeros(2, dtype=np.int8),  # kTwoInt8s
    7: lambda: np.uint16(0),  # kOneUint16
    8: lambda: np.int16(0),  # kOneInt16
    # 3 bytes total
    9: lambda: np.zeros(3, dtype=np.bool_),  # kThreeBools
    10: lambda: np.zeros(3, dtype=np.uint8),  # kThreeUint8s
    11: lambda: np.zeros(3, dtype=np.int8),  # kThreeInt8s
    # 4 bytes total
    12: lambda: np.zeros(4, dtype=np.bool_),  # kFourBools
    13: lambda: np.zeros(4, dtype=np.uint8),  # kFourUint8s
    14: lambda: np.zeros(4, dtype=np.int8),  # kFourInt8s
    15: lambda: np.zeros(2, dtype=np.uint16),  # kTwoUint16s
    16: lambda: np.zeros(2, dtype=np.int16),  # kTwoInt16s
    17: lambda: np.uint32(0),  # kOneUint32
    18: lambda: np.int32(0),  # kOneInt32
    19: lambda: np.float32(0),  # kOneFloat32
    # 5 bytes total
    20: lambda: np.zeros(5, dtype=np.bool_),  # kFiveBools
    21: lambda: np.zeros(5, dtype=np.uint8),  # kFiveUint8s
    22: lambda: np.zeros(5, dtype=np.int8),  # kFiveInt8s
    # 6 bytes total
    23: lambda: np.zeros(6, dtype=np.bool_),  # kSixBools
    24: lambda: np.zeros(6, dtype=np.uint8),  # kSixUint8s
    25: lambda: np.zeros(6, dtype=np.int8),  # kSixInt8s
    26: lambda: np.zeros(3, dtype=np.uint16),  # kThreeUint16s
    27: lambda: np.zeros(3, dtype=np.int16),  # kThreeInt16s
    # 7 bytes total
    28: lambda: np.zeros(7, dtype=np.bool_),  # kSevenBools
    29: lambda: np.zeros(7, dtype=np.uint8),  # kSevenUint8s
    30: lambda: np.zeros(7, dtype=np.int8),  # kSevenInt8s
    # 8 bytes total
    31: lambda: np.zeros(8, dtype=np.bool_),  # kEightBools
    32: lambda: np.zeros(8, dtype=np.uint8),  # kEightUint8s
    33: lambda: np.zeros(8, dtype=np.int8),  # kEightInt8s
    34: lambda: np.zeros(4, dtype=np.uint16),  # kFourUint16s
    35: lambda: np.zeros(4, dtype=np.int16),  # kFourInt16s
    36: lambda: np.zeros(2, dtype=np.uint32),  # kTwoUint32s
    37: lambda: np.zeros(2, dtype=np.int32),  # kTwoInt32s
    38: lambda: np.zeros(2, dtype=np.float32),  # kTwoFloat32s
    39: lambda: np.uint64(0),  # kOneUint64
    40: lambda: np.int64(0),  # kOneInt64
    41: lambda: np.float64(0),  # kOneFloat64
    # 9 bytes total
    42: lambda: np.zeros(9, dtype=np.bool_),  # kNineBools
    43: lambda: np.zeros(9, dtype=np.uint8),  # kNineUint8s
    44: lambda: np.zeros(9, dtype=np.int8),  # kNineInt8s
    # 10 bytes total
    45: lambda: np.zeros(10, dtype=np.bool_),  # kTenBools
    46: lambda: np.zeros(10, dtype=np.uint8),  # kTenUint8s
    47: lambda: np.zeros(10, dtype=np.int8),  # kTenInt8s
    48: lambda: np.zeros(5, dtype=np.uint16),  # kFiveUint16s
    49: lambda: np.zeros(5, dtype=np.int16),  # kFiveInt16s
    # 11 bytes total
    50: lambda: np.zeros(11, dtype=np.bool_),  # kElevenBools
    51: lambda: np.zeros(11, dtype=np.uint8),  # kElevenUint8s
    52: lambda: np.zeros(11, dtype=np.int8),  # kElevenInt8s
    # 12 bytes total
    53: lambda: np.zeros(12, dtype=np.bool_),  # kTwelveBools
    54: lambda: np.zeros(12, dtype=np.uint8),  # kTwelveUint8s
    55: lambda: np.zeros(12, dtype=np.int8),  # kTwelveInt8s
    56: lambda: np.zeros(6, dtype=np.uint16),  # kSixUint16s
    57: lambda: np.zeros(6, dtype=np.int16),  # kSixInt16s
    58: lambda: np.zeros(3, dtype=np.uint32),  # kThreeUint32s
    59: lambda: np.zeros(3, dtype=np.int32),  # kThreeInt32s
    60: lambda: np.zeros(3, dtype=np.float32),  # kThreeFloat32s
    # 13 bytes total
    61: lambda: np.zeros(13, dtype=np.bool_),  # kThirteenBools
    62: lambda: np.zeros(13, dtype=np.uint8),  # kThirteenUint8s
    63: lambda: np.zeros(13, dtype=np.int8),  # kThirteenInt8s
    # 14 bytes total
    64: lambda: np.zeros(14, dtype=np.bool_),  # kFourteenBools
    65: lambda: np.zeros(14, dtype=np.uint8),  # kFourteenUint8s
    66: lambda: np.zeros(14, dtype=np.int8),  # kFourteenInt8s
    67: lambda: np.zeros(7, dtype=np.uint16),  # kSevenUint16s
    68: lambda: np.zeros(7, dtype=np.int16),  # kSevenInt16s
    # 15 bytes total
    69: lambda: np.zeros(15, dtype=np.bool_),  # kFifteenBools
    70: lambda: np.zeros(15, dtype=np.uint8),  # kFifteenUint8s
    71: lambda: np.zeros(15, dtype=np.int8),  # kFifteenInt8s
    # 16 bytes total
    72: lambda: np.zeros(8, dtype=np.uint16),  # kEightUint16s
    73: lambda: np.zeros(8, dtype=np.int16),  # kEightInt16s
    74: lambda: np.zeros(4, dtype=np.uint32),  # kFourUint32s
    75: lambda: np.zeros(4, dtype=np.int32),  # kFourInt32s
    76: lambda: np.zeros(4, dtype=np.float32),  # kFourFloat32s
    77: lambda: np.zeros(2, dtype=np.uint64),  # kTwoUint64s
    78: lambda: np.zeros(2, dtype=np.int64),  # kTwoInt64s
    79: lambda: np.zeros(2, dtype=np.float64),  # kTwoFloat64s
    # 18 bytes total
    80: lambda: np.zeros(9, dtype=np.uint16),  # kNineUint16s
    81: lambda: np.zeros(9, dtype=np.int16),  # kNineInt16s
    # 20 bytes total
    82: lambda: np.zeros(10, dtype=np.uint16),  # kTenUint16s
    83: lambda: np.zeros(10, dtype=np.int16),  # kTenInt16s
    84: lambda: np.zeros(5, dtype=np.uint32),  # kFiveUint32s
    85: lambda: np.zeros(5, dtype=np.int32),  # kFiveInt32s
    86: lambda: np.zeros(5, dtype=np.float32),  # kFiveFloat32s
    # 22 bytes total
    87: lambda: np.zeros(11, dtype=np.uint16),  # kElevenUint16s
    88: lambda: np.zeros(11, dtype=np.int16),  # kElevenInt16s
    # 24 bytes total
    89: lambda: np.zeros(12, dtype=np.uint16),  # kTwelveUint16s
    90: lambda: np.zeros(12, dtype=np.int16),  # kTwelveInt16s
    91: lambda: np.zeros(6, dtype=np.uint32),  # kSixUint32s
    92: lambda: np.zeros(6, dtype=np.int32),  # kSixInt32s
    93: lambda: np.zeros(6, dtype=np.float32),  # kSixFloat32s
    94: lambda: np.zeros(3, dtype=np.uint64),  # kThreeUint64s
    95: lambda: np.zeros(3, dtype=np.int64),  # kThreeInt64s
    96: lambda: np.zeros(3, dtype=np.float64),  # kThreeFloat64s
    # 26 bytes total
    97: lambda: np.zeros(13, dtype=np.uint16),  # kThirteenUint16s
    98: lambda: np.zeros(13, dtype=np.int16),  # kThirteenInt16s
    # 28 bytes total
    99: lambda: np.zeros(14, dtype=np.uint16),  # kFourteenUint16s
    100: lambda: np.zeros(14, dtype=np.int16),  # kFourteenInt16s
    101: lambda: np.zeros(7, dtype=np.uint32),  # kSevenUint32s
    102: lambda: np.zeros(7, dtype=np.int32),  # kSevenInt32s
    103: lambda: np.zeros(7, dtype=np.float32),  # kSevenFloat32s
    # 30 bytes total
    104: lambda: np.zeros(15, dtype=np.uint16),  # kFifteenUint16s
    105: lambda: np.zeros(15, dtype=np.int16),  # kFifteenInt16s
    # 32 bytes total
    106: lambda: np.zeros(8, dtype=np.uint32),  # kEightUint32s
    107: lambda: np.zeros(8, dtype=np.int32),  # kEightInt32s
    108: lambda: np.zeros(8, dtype=np.float32),  # kEightFloat32s
    109: lambda: np.zeros(4, dtype=np.uint64),  # kFourUint64s
    110: lambda: np.zeros(4, dtype=np.int64),  # kFourInt64s
    111: lambda: np.zeros(4, dtype=np.float64),  # kFourFloat64s
    # 36 bytes total
    112: lambda: np.zeros(9, dtype=np.uint32),  # kNineUint32s
    113: lambda: np.zeros(9, dtype=np.int32),  # kNineInt32s
    114: lambda: np.zeros(9, dtype=np.float32),  # kNineFloat32s
    # 40 bytes total
    115: lambda: np.zeros(10, dtype=np.uint32),  # kTenUint32s
    116: lambda: np.zeros(10, dtype=np.int32),  # kTenInt32s
    117: lambda: np.zeros(10, dtype=np.float32),  # kTenFloat32s
    118: lambda: np.zeros(5, dtype=np.uint64),  # kFiveUint64s
    119: lambda: np.zeros(5, dtype=np.int64),  # kFiveInt64s
    120: lambda: np.zeros(5, dtype=np.float64),  # kFiveFloat64s
    # 44 bytes total
    121: lambda: np.zeros(11, dtype=np.uint32),  # kElevenUint32s
    122: lambda: np.zeros(11, dtype=np.int32),  # kElevenInt32s
    123: lambda: np.zeros(11, dtype=np.float32),  # kElevenFloat32s
    # 48 bytes total
    124: lambda: np.zeros(12, dtype=np.uint32),  # kTwelveUint32s
    125: lambda: np.zeros(12, dtype=np.int32),  # kTwelveInt32s
    126: lambda: np.zeros(12, dtype=np.float32),  # kTwelveFloat32s
    127: lambda: np.zeros(6, dtype=np.uint64),  # kSixUint64s
    128: lambda: np.zeros(6, dtype=np.int64),  # kSixInt64s
    129: lambda: np.zeros(6, dtype=np.float64),  # kSixFloat64s
    # 52 bytes total
    130: lambda: np.zeros(13, dtype=np.uint32),  # kThirteenUint32s
    131: lambda: np.zeros(13, dtype=np.int32),  # kThirteenInt32s
    132: lambda: np.zeros(13, dtype=np.float32),  # kThirteenFloat32s
    # 56 bytes total
    133: lambda: np.zeros(14, dtype=np.uint32),  # kFourteenUint32s
    134: lambda: np.zeros(14, dtype=np.int32),  # kFourteenInt32s
    135: lambda: np.zeros(14, dtype=np.float32),  # kFourteenFloat32s
    136: lambda: np.zeros(7, dtype=np.uint64),  # kSevenUint64s
    137: lambda: np.zeros(7, dtype=np.int64),  # kSevenInt64s
    138: lambda: np.zeros(7, dtype=np.float64),  # kSevenFloat64s
    # 60 bytes total
    139: lambda: np.zeros(15, dtype=np.uint32),  # kFifteenUint32s
    140: lambda: np.zeros(15, dtype=np.int32),  # kFifteenInt32s
    141: lambda: np.zeros(15, dtype=np.float32),  # kFifteenFloat32s
    # 64 bytes total
    142: lambda: np.zeros(8, dtype=np.uint64),  # kEightUint64s
    143: lambda: np.zeros(8, dtype=np.int64),  # kEightInt64s
    144: lambda: np.zeros(8, dtype=np.float64),  # kEightFloat64s
    # 72 bytes total
    145: lambda: np.zeros(9, dtype=np.uint64),  # kNineUint64s
    146: lambda: np.zeros(9, dtype=np.int64),  # kNineInt64s
    147: lambda: np.zeros(9, dtype=np.float64),  # kNineFloat64s
    # 80 bytes total
    148: lambda: np.zeros(10, dtype=np.uint64),  # kTenUint64s
    149: lambda: np.zeros(10, dtype=np.int64),  # kTenInt64s
    150: lambda: np.zeros(10, dtype=np.float64),  # kTenFloat64s
    # 88 bytes total
    151: lambda: np.zeros(11, dtype=np.uint64),  # kElevenUint64s
    152: lambda: np.zeros(11, dtype=np.int64),  # kElevenInt64s
    153: lambda: np.zeros(11, dtype=np.float64),  # kElevenFloat64s
    # 96 bytes total
    154: lambda: np.zeros(12, dtype=np.uint64),  # kTwelveUint64s
    155: lambda: np.zeros(12, dtype=np.int64),  # kTwelveInt64s
    156: lambda: np.zeros(12, dtype=np.float64),  # kTwelveFloat64s
    # 104 bytes total
    157: lambda: np.zeros(13, dtype=np.uint64),  # kThirteenUint64s
    158: lambda: np.zeros(13, dtype=np.int64),  # kThirteenInt64s
    159: lambda: np.zeros(13, dtype=np.float64),  # kThirteenFloat64s
    # 112 bytes total
    160: lambda: np.zeros(14, dtype=np.uint64),  # kFourteenUint64s
    161: lambda: np.zeros(14, dtype=np.int64),  # kFourteenInt64s
    162: lambda: np.zeros(14, dtype=np.float64),  # kFourteenFloat64s
    # 120 bytes total
    163: lambda: np.zeros(15, dtype=np.uint64),  # kFifteenUint64s
    164: lambda: np.zeros(15, dtype=np.int64),  # kFifteenInt64s
    165: lambda: np.zeros(15, dtype=np.float64),  # kFifteenFloat64s
}


class SerialPrototypes(IntEnum):
    """Stores the prototype codes used in data transmission between the PC and the microcontroller over the serial port.

    Prototype codes are used by Data messages (Kernel and Module) to inform the receiver about the structure (prototype)
    that can be used to deserialize the included data object. Transmitting these codes with the message ensures that
    the receiver has the necessary information to decode the data without doing any additional processing. In turn,
    this allows optimizing the reception procedure to efficiently decode the data objects.

    Notes:
        While the use of 8-bit (byte) value limits the number of mapped prototypes to 255 (256 if 0 is made a valid
        value), this number should be enough to support many unique runtime configurations.
    """

    # 1 byte total
    ONE_BOOL = 1
    """1 8-bit boolean"""
    ONE_UINT8 = 2
    """1 unsigned 8-bit integer"""
    ONE_INT8 = 3
    """1 signed 8-bit integer"""

    # 2 bytes total
    TWO_BOOLS = 4
    """An array of 2 8-bit booleans"""
    TWO_UINT8S = 5
    """An array of 2 unsigned 8-bit integers"""
    TWO_INT8S = 6
    """An array of 2 signed 8-bit integers"""
    ONE_UINT16 = 7
    """1 unsigned 16-bit integer"""
    ONE_INT16 = 8
    """1 signed 16-bit integer"""

    # 3 bytes total
    THREE_BOOLS = 9
    """An array of 3 8-bit booleans"""
    THREE_UINT8S = 10
    """An array of 3 unsigned 8-bit integers"""
    THREE_INT8S = 11
    """An array of 3 signed 8-bit integers"""

    # 4 bytes total
    FOUR_BOOLS = 12
    """An array of 4 8-bit booleans"""
    FOUR_UINT8S = 13
    """An array of 4 unsigned 8-bit integers"""
    FOUR_INT8S = 14
    """An array of 4 signed 8-bit integers"""
    TWO_UINT16S = 15
    """An array of 2 unsigned 16-bit integers"""
    TWO_INT16S = 16
    """An array of 2 signed 16-bit integers"""
    ONE_UINT32 = 17
    """1 unsigned 32-bit integer"""
    ONE_INT32 = 18
    """1 signed 32-bit integer"""
    ONE_FLOAT32 = 19
    """1 single-precision 32-bit floating-point number"""

    # 5 bytes total
    FIVE_BOOLS = 20
    """An array of 5 8-bit booleans"""
    FIVE_UINT8S = 21
    """An array of 5 unsigned 8-bit integers"""
    FIVE_INT8S = 22
    """An array of 5 signed 8-bit integers"""

    # 6 bytes total
    SIX_BOOLS = 23
    """An array of 6 8-bit booleans"""
    SIX_UINT8S = 24
    """An array of 6 unsigned 8-bit integers"""
    SIX_INT8S = 25
    """An array of 6 signed 8-bit integers"""
    THREE_UINT16S = 26
    """An array of 3 unsigned 16-bit integers"""
    THREE_INT16S = 27
    """An array of 3 signed 16-bit integers"""

    # 7 bytes total
    SEVEN_BOOLS = 28
    """An array of 7 8-bit booleans"""
    SEVEN_UINT8S = 29
    """An array of 7 unsigned 8-bit integers"""
    SEVEN_INT8S = 30
    """An array of 7 signed 8-bit integers"""

    # 8 bytes total
    EIGHT_BOOLS = 31
    """An array of 8 8-bit booleans"""
    EIGHT_UINT8S = 32
    """An array of 8 unsigned 8-bit integers"""
    EIGHT_INT8S = 33
    """An array of 8 signed 8-bit integers"""
    FOUR_UINT16S = 34
    """An array of 4 unsigned 16-bit integers"""
    FOUR_INT16S = 35
    """An array of 4 signed 16-bit integers"""
    TWO_UINT32S = 36
    """An array of 2 unsigned 32-bit integers"""
    TWO_INT32S = 37
    """An array of 2 signed 32-bit integers"""
    TWO_FLOAT32S = 38
    """An array of 2 single-precision 32-bit floating-point numbers"""
    ONE_UINT64 = 39
    """1 unsigned 64-bit integer"""
    ONE_INT64 = 40
    """1 signed 64-bit integer"""
    ONE_FLOAT64 = 41
    """1 double-precision 64-bit floating-point number"""

    # 9 bytes total
    NINE_BOOLS = 42
    """An array of 9 8-bit booleans"""
    NINE_UINT8S = 43
    """An array of 9 unsigned 8-bit integers"""
    NINE_INT8S = 44
    """An array of 9 signed 8-bit integers"""

    # 10 bytes total
    TEN_BOOLS = 45
    """An array of 10 8-bit booleans"""
    TEN_UINT8S = 46
    """An array of 10 unsigned 8-bit integers"""
    TEN_INT8S = 47
    """An array of 10 signed 8-bit integers"""
    FIVE_UINT16S = 48
    """An array of 5 unsigned 16-bit integers"""
    FIVE_INT16S = 49
    """An array of 5 signed 16-bit integers"""

    # 11 bytes total
    ELEVEN_BOOLS = 50
    """An array of 11 8-bit booleans"""
    ELEVEN_UINT8S = 51
    """An array of 11 unsigned 8-bit integers"""
    ELEVEN_INT8S = 52
    """An array of 11 signed 8-bit integers"""

    # 12 bytes total
    TWELVE_BOOLS = 53
    """An array of 12 8-bit booleans"""
    TWELVE_UINT8S = 54
    """An array of 12 unsigned 8-bit integers"""
    TWELVE_INT8S = 55
    """An array of 12 signed 8-bit integers"""
    SIX_UINT16S = 56
    """An array of 6 unsigned 16-bit integers"""
    SIX_INT16S = 57
    """An array of 6 signed 16-bit integers"""
    THREE_UINT32S = 58
    """An array of 3 unsigned 32-bit integers"""
    THREE_INT32S = 59
    """An array of 3 signed 32-bit integers"""
    THREE_FLOAT32S = 60
    """An array of 3 single-precision 32-bit floating-point numbers"""

    # 13 bytes total
    THIRTEEN_BOOLS = 61
    """An array of 13 8-bit booleans"""
    THIRTEEN_UINT8S = 62
    """An array of 13 unsigned 8-bit integers"""
    THIRTEEN_INT8S = 63
    """An array of 13 signed 8-bit integers"""

    # 14 bytes total
    FOURTEEN_BOOLS = 64
    """An array of 14 8-bit booleans"""
    FOURTEEN_UINT8S = 65
    """An array of 14 unsigned 8-bit integers"""
    FOURTEEN_INT8S = 66
    """An array of 14 signed 8-bit integers"""
    SEVEN_UINT16S = 67
    """An array of 7 unsigned 16-bit integers"""
    SEVEN_INT16S = 68
    """An array of 7 signed 16-bit integers"""

    # 15 bytes total
    FIFTEEN_BOOLS = 69
    """An array of 15 8-bit booleans"""
    FIFTEEN_UINT8S = 70
    """An array of 15 unsigned 8-bit integers"""
    FIFTEEN_INT8S = 71
    """An array of 15 signed 8-bit integers"""

    # 16 bytes total
    EIGHT_UINT16S = 72
    """An array of 8 unsigned 16-bit integers"""
    EIGHT_INT16S = 73
    """An array of 8 signed 16-bit integers"""
    FOUR_UINT32S = 74
    """An array of 4 unsigned 32-bit integers"""
    FOUR_INT32S = 75
    """An array of 4 signed 32-bit integers"""
    FOUR_FLOAT32S = 76
    """An array of 4 single-precision 32-bit floating-point numbers"""
    TWO_UINT64S = 77
    """An array of 2 unsigned 64-bit integers"""
    TWO_INT64S = 78
    """An array of 2 signed 64-bit integers"""
    TWO_FLOAT64S = 79
    """An array of 2 double-precision 64-bit floating-point numbers"""

    # 18 bytes total
    NINE_UINT16S = 80
    """An array of 9 unsigned 16-bit integers"""
    NINE_INT16S = 81
    """An array of 9 signed 16-bit integers"""

    # 20 bytes total
    TEN_UINT16S = 82
    """An array of 10 unsigned 16-bit integers"""
    TEN_INT16S = 83
    """An array of 10 signed 16-bit integers"""
    FIVE_UINT32S = 84
    """An array of 5 unsigned 32-bit integers"""
    FIVE_INT32S = 85
    """An array of 5 signed 32-bit integers"""
    FIVE_FLOAT32S = 86
    """An array of 5 single-precision 32-bit floating-point numbers"""

    # 22 bytes total
    ELEVEN_UINT16S = 87
    """An array of 11 unsigned 16-bit integers"""
    ELEVEN_INT16S = 88
    """An array of 11 signed 16-bit integers"""

    # 24 bytes total
    TWELVE_UINT16S = 89
    """An array of 12 unsigned 16-bit integers"""
    TWELVE_INT16S = 90
    """An array of 12 signed 16-bit integers"""
    SIX_UINT32S = 91
    """An array of 6 unsigned 32-bit integers"""
    SIX_INT32S = 92
    """An array of 6 signed 32-bit integers"""
    SIX_FLOAT32S = 93
    """An array of 6 single-precision 32-bit floating-point numbers"""
    THREE_UINT64S = 94
    """An array of 3 unsigned 64-bit integers"""
    THREE_INT64S = 95
    """An array of 3 signed 64-bit integers"""
    THREE_FLOAT64S = 96
    """An array of 3 double-precision 64-bit floating-point numbers"""

    # 26 bytes total
    THIRTEEN_UINT16S = 97
    """An array of 13 unsigned 16-bit integers"""
    THIRTEEN_INT16S = 98
    """An array of 13 signed 16-bit integers"""

    # 28 bytes total
    FOURTEEN_UINT16S = 99
    """An array of 14 unsigned 16-bit integers"""
    FOURTEEN_INT16S = 100
    """An array of 14 signed 16-bit integers"""
    SEVEN_UINT32S = 101
    """An array of 7 unsigned 32-bit integers"""
    SEVEN_INT32S = 102
    """An array of 7 signed 32-bit integers"""
    SEVEN_FLOAT32S = 103
    """An array of 7 single-precision 32-bit floating-point numbers"""

    # 30 bytes total
    FIFTEEN_UINT16S = 104
    """An array of 15 unsigned 16-bit integers"""
    FIFTEEN_INT16S = 105
    """An array of 15 signed 16-bit integers"""

    # 32 bytes total
    EIGHT_UINT32S = 106
    """An array of 8 unsigned 32-bit integers"""
    EIGHT_INT32S = 107
    """An array of 8 signed 32-bit integers"""
    EIGHT_FLOAT32S = 108
    """An array of 8 single-precision 32-bit floating-point numbers"""
    FOUR_UINT64S = 109
    """An array of 4 unsigned 64-bit integers"""
    FOUR_INT64S = 110
    """An array of 4 signed 64-bit integers"""
    FOUR_FLOAT64S = 111
    """An array of 4 double-precision 64-bit floating-point numbers"""

    # 36 bytes total
    NINE_UINT32S = 112
    """An array of 9 unsigned 32-bit integers"""
    NINE_INT32S = 113
    """An array of 9 signed 32-bit integers"""
    NINE_FLOAT32S = 114
    """An array of 9 single-precision 32-bit floating-point numbers"""

    # 40 bytes total
    TEN_UINT32S = 115
    """An array of 10 unsigned 32-bit integers"""
    TEN_INT32S = 116
    """An array of 10 signed 32-bit integers"""
    TEN_FLOAT32S = 117
    """An array of 10 single-precision 32-bit floating-point numbers"""
    FIVE_UINT64S = 118
    """An array of 5 unsigned 64-bit integers"""
    FIVE_INT64S = 119
    """An array of 5 signed 64-bit integers"""
    FIVE_FLOAT64S = 120
    """An array of 5 double-precision 64-bit floating-point numbers"""

    # 44 bytes total
    ELEVEN_UINT32S = 121
    """An array of 11 unsigned 32-bit integers"""
    ELEVEN_INT32S = 122
    """An array of 11 signed 32-bit integers"""
    ELEVEN_FLOAT32S = 123
    """An array of 11 single-precision 32-bit floating-point numbers"""

    # 48 bytes total
    TWELVE_UINT32S = 124
    """An array of 12 unsigned 32-bit integers"""
    TWELVE_INT32S = 125
    """An array of 12 signed 32-bit integers"""
    TWELVE_FLOAT32S = 126
    """An array of 12 single-precision 32-bit floating-point numbers"""
    SIX_UINT64S = 127
    """An array of 6 unsigned 64-bit integers"""
    SIX_INT64S = 128
    """An array of 6 signed 64-bit integers"""
    SIX_FLOAT64S = 129
    """An array of 6 double-precision 64-bit floating-point numbers"""

    # 52 bytes total
    THIRTEEN_UINT32S = 130
    """An array of 13 unsigned 32-bit integers"""
    THIRTEEN_INT32S = 131
    """An array of 13 signed 32-bit integers"""
    THIRTEEN_FLOAT32S = 132
    """An array of 13 single-precision 32-bit floating-point numbers"""

    # 56 bytes total
    FOURTEEN_UINT32S = 133
    """An array of 14 unsigned 32-bit integers"""
    FOURTEEN_INT32S = 134
    """An array of 14 signed 32-bit integers"""
    FOURTEEN_FLOAT32S = 135
    """An array of 14 single-precision 32-bit floating-point numbers"""
    SEVEN_UINT64S = 136
    """An array of 7 unsigned 64-bit integers"""
    SEVEN_INT64S = 137
    """An array of 7 signed 64-bit integers"""
    SEVEN_FLOAT64S = 138
    """An array of 7 double-precision 64-bit floating-point numbers"""

    # 60 bytes total
    FIFTEEN_UINT32S = 139
    """An array of 15 unsigned 32-bit integers"""
    FIFTEEN_INT32S = 140
    """An array of 15 signed 32-bit integers"""
    FIFTEEN_FLOAT32S = 141
    """An array of 15 single-precision 32-bit floating-point numbers"""

    # 64 bytes total
    EIGHT_UINT64S = 142
    """An array of 8 unsigned 64-bit integers"""
    EIGHT_INT64S = 143
    """An array of 8 signed 64-bit integers"""
    EIGHT_FLOAT64S = 144
    """An array of 8 double-precision 64-bit floating-point numbers"""

    # 72 bytes total
    NINE_UINT64S = 145
    """An array of 9 unsigned 64-bit integers"""
    NINE_INT64S = 146
    """An array of 9 signed 64-bit integers"""
    NINE_FLOAT64S = 147
    """An array of 9 double-precision 64-bit floating-point numbers"""

    # 80 bytes total
    TEN_UINT64S = 148
    """An array of 10 unsigned 64-bit integers"""
    TEN_INT64S = 149
    """An array of 10 signed 64-bit integers"""
    TEN_FLOAT64S = 150
    """An array of 10 double-precision 64-bit floating-point numbers"""

    # 88 bytes total
    ELEVEN_UINT64S = 151
    """An array of 11 unsigned 64-bit integers"""
    ELEVEN_INT64S = 152
    """An array of 11 signed 64-bit integers"""
    ELEVEN_FLOAT64S = 153
    """An array of 11 double-precision 64-bit floating-point numbers"""

    # 96 bytes total
    TWELVE_UINT64S = 154
    """An array of 12 unsigned 64-bit integers"""
    TWELVE_INT64S = 155
    """An array of 12 signed 64-bit integers"""
    TWELVE_FLOAT64S = 156
    """An array of 12 double-precision 64-bit floating-point numbers"""

    # 104 bytes total
    THIRTEEN_UINT64S = 157
    """An array of 13 unsigned 64-bit integers"""
    THIRTEEN_INT64S = 158
    """An array of 13 signed 64-bit integers"""
    THIRTEEN_FLOAT64S = 159
    """An array of 13 double-precision 64-bit floating-point numbers"""

    # 112 bytes total
    FOURTEEN_UINT64S = 160
    """An array of 14 unsigned 64-bit integers"""
    FOURTEEN_INT64S = 161
    """An array of 14 signed 64-bit integers"""
    FOURTEEN_FLOAT64S = 162
    """An array of 14 double-precision 64-bit floating-point numbers"""

    # 120 bytes total
    FIFTEEN_UINT64S = 163
    """An array of 15 unsigned 64-bit integers"""
    FIFTEEN_INT64S = 164
    """An array of 15 signed 64-bit integers"""
    FIFTEEN_FLOAT64S = 165
    """An array of 15 double-precision 64-bit floating-point numbers"""

    def as_uint8(self) -> np.uint8:
        """Converts the enum value to numpy.uint8 type.

        Returns:
            The enum value as a numpy unsigned 8-bit integer.
        """
        return np.uint8(self.value)

    def get_prototype(self) -> PrototypeType:
        """Returns the prototype object associated with this prototype enum value.

        The prototype object returned by this method can be passed to the reading method of the TransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this enum class.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type.
        """
        return _PROTOTYPE_FACTORIES[self.value]()

    @classmethod
    def get_prototype_for_code(cls, code: np.uint8) -> PrototypeType | None:
        """Returns the prototype object associated with the input prototype code.

        The prototype object returned by this method can be passed to the reading method of the TransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this enum.

        Args:
            code: The prototype byte-code to retrieve the prototype for.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type. If the input code is not one of
            the supported codes, returns None to indicate a matching error.
        """
        try:
            enum_value = cls(int(code))
            return enum_value.get_prototype()
        except ValueError:
            return None


@dataclass(frozen=True)
class RepeatedModuleCommand:
    """Instructs the addressed Module to repeatedly (recurrently) run the specified command."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module within the broader module-family."""
    command: np.uint8
    """The code of the command to execute. Valid command codes are in the range between 1 and 255."""
    return_code: np.uint8 = np.uint8(0)
    """
    When this attribute is set to a value other than 0, the microcontroller will send this code back to the PC upon 
    successfully receiving and decoding the command.
    """
    noblock: np.bool_ = np.bool(True)
    """
    Determines whether the command runs in blocking or non-blocking mode. If set to False, the controller
    will block in-place for any sensor- or time-waiting loops during command execution. Otherwise, the
    controller will run other commands while waiting for the block to complete.
    """
    cycle_delay: np.uint32 = np.uint32(0)
    """The period of time, in microseconds, to delay before repeating (cycling) the command."""
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.REPEATED_MODULE_COMMAND.as_uint8())
    """Stores the protocol code used by this type of messages."""

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(10, dtype=np.uint8)
        packed[0:6] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
            self.command,
            self.noblock,
        ]
        packed[6:10] = np.frombuffer(self.cycle_delay.tobytes(), dtype=np.uint8)
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""
        message = (
            f"RepeatedModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock}, cycle_delay={self.cycle_delay} us)."
        )
        return message


@dataclass(frozen=True)
class OneOffModuleCommand:
    """Instructs the addressed Module to run the specified command exactly once (non-recurrently)."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module within the broader module-family."""
    command: np.uint8
    """The code of the command to execute. Valid command codes are in the range between 1 and 255."""
    return_code: np.uint8 = np.uint8(0)
    """When this attribute is set to a value other than 0, the microcontroller will send this code back
    to the PC upon successfully receiving and decoding the command."""
    noblock: np.bool_ = np.bool(True)
    """Determines whether the command runs in blocking or non-blocking mode. If set to False, the controller
    will block in-place for any sensor- or time-waiting loops during command execution. Otherwise, the
    controller will run other commands while waiting for the block to complete."""
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.ONE_OFF_MODULE_COMMAND.as_uint8())
    """Stores the protocol code used by this type of messages."""

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(6, dtype=np.uint8)
        packed[0:6] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
            self.command,
            self.noblock,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the OneOffModuleCommand object."""
        message = (
            f"OneOffModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock})."
        )
        return message


@dataclass(frozen=True)
class DequeueModuleCommand:
    """Instructs the addressed Module to clear (empty) its command queue.

    Note, clearing the command queue does not terminate already executing commands, but it prevents recurrent commands
    from running again.
    """

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module within the broader module-family."""
    return_code: np.uint8 = np.uint8(0)
    """When this attribute is set to a value other than 0, the microcontroller will send this code back
    to the PC upon successfully receiving and decoding the command."""
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.DEQUEUE_MODULE_COMMAND.as_uint8())
    """Stores the protocol code used by this type of messages."""

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(4, dtype=np.uint8)
        packed[0:4] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""
        message = (
            f"kDequeueModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code})."
        )
        return message


@dataclass(frozen=True)
class KernelCommand:
    """Instructs the Kernel to run the specified command exactly once.

    Currently, the Kernel only supports blocking one-off commands.
    """

    command: np.uint8
    """The code of the command to execute. Valid command codes are in the range between 1 and 255."""
    return_code: np.uint8 = np.uint8(0)
    """When this attribute is set to a value other than 0, the microcontroller will send this code back
    to the PC upon successfully receiving and decoding the command."""
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.KERNEL_COMMAND.as_uint8())
    """Stores the protocol code used by this type of messages."""

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(3, dtype=np.uint8)
        packed[0:3] = [
            self.protocol_code,
            self.return_code,
            self.command,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelCommand object."""
        message = (
            f"KernelCommand(protocol_code={self.protocol_code}, command={self.command}, "
            f"return_code={self.return_code})."
        )
        return message


@dataclass(frozen=True)
class ModuleParameters:
    """Instructs the addressed Module to overwrite its custom parameters object with the included object data."""

    module_type: np.uint8
    """The type (family) code of the module to which the parameters are addressed."""
    module_id: np.uint8
    """The ID of the specific module within the broader module-family."""
    parameter_data: tuple[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool, ...]
    """A tuple of parameter values to send. Each value will be serialized into bytes and sequentially
    packed into the data object included with the message. Each parameter value has to use a scalar numpy type."""
    return_code: np.uint8 = np.uint8(0)
    """When this attribute is set to a value other than 0, the microcontroller will send this code back
    to the PC upon successfully receiving and decoding the command."""
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores serialized message data."""
    parameters_size: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the total size of serialized parameters in bytes."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.MODULE_PARAMETERS.as_uint8())
    """Stores the protocol code used by this type of messages."""

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        # Converts scalar parameter values to byte arrays (serializes them)
        byte_parameters = [np.frombuffer(np.array([param]), dtype=np.uint8).copy() for param in self.parameter_data]

        # Calculates the total size of serialized parameters in bytes and adds it to the parameters_size attribute
        parameters_size = np.uint8(sum(param.size for param in byte_parameters))
        object.__setattr__(self, "parameters_size", parameters_size)

        # Pre-allocates the full array with exact size (header and parameters object)
        packed_data = np.empty(4 + parameters_size, dtype=np.uint8)

        # Packs the header data into the precreated array
        packed_data[0:4] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
        ]

        # Loops over and sequentially appends parameter data to the array.
        current_position = 4
        for param_bytes in byte_parameters:
            param_size = param_bytes.size
            packed_data[current_position : current_position + param_size] = param_bytes
            current_position += param_size

        # Writes the constructed packed data object to the packed_data attribute
        object.__setattr__(self, "packed_data", packed_data)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleParameters object."""
        message = (
            f"ModuleParameters(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code}, "
            f"parameter_object_size={self.parameters_size} bytes)."
        )
        return message


@dataclass(frozen=True)
class KernelParameters:
    """Instructs the Kernel to update the microcontroller-wide parameters with the values included in the message.

    These parameters are shared by the Kernel and all custom Modules, and the exact parameter layout is hardcoded. This
    is in contrast to Module parameters, that differ between module types.
    """

    action_lock: np.bool
    """Determines whether the controller allows non-ttl modules to change output pin states.
    When True, all hardware-connected pins are blocked from changing states. This has no effect on sensor and
    TTL pins."""
    ttl_lock: np.bool
    """Same as action_lock, but specifically controls output TTL (Transistor-to-Transistor Logic) pin
    activity. This has no effect on sensor and non-ttl hardware-connected pins."""
    return_code: np.uint8 = np.uint8(0)
    """When this attribute is set to a value other than 0, the microcontroller will send this code back
    to the PC upon successfully receiving and decoding the command."""
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores serialized message data."""
    parameters_size: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the total size of serialized parameters in bytes."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.KERNEL_PARAMETERS.as_uint8())
    """Stores the protocol code used by this type of messages."""

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed_data = np.empty(4, dtype=np.uint8)
        packed_data[0:4] = [self.protocol_code, self.return_code, self.action_lock, self.ttl_lock]

        object.__setattr__(
            self, "parameters_size", packed_data.nbytes - 2
        )  # -2 to account for protocol and return code
        object.__setattr__(self, "packed_data", packed_data)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelParameters object."""
        message = (
            f"KernelParameters(protocol_code={self.protocol_code}, return_code={self.return_code}, "
            f"parameter_object_size={self.parameters_size} bytes)."
        )
        return message


class ModuleData:
    """Communicates the event state-code of the sender Module and includes an additional data object.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        module_type: The type (family) code of the module that sent the message.
        module_id: The ID of the specific module within the broader module-family.
        command: The code of the command the module was executing when it sent the message.
        event: The code of the event that prompted sending the message.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes enumeration.
        _transport_layer: Stores the reference to the TransportLayer class.
    """

    def __init__(self, transport_layer: TransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = SerialProtocols.MODULE_DATA.as_uint8()

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.module_type: np.uint8 = np.uint8(0)
        self.module_id: np.uint8 = np.uint8(0)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)
        self.data_object: np.unsignedinteger[Any] | NDArray[Any] = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the static header data from the extracted message
        self.module_type = self.message[1]
        self.module_id = self.message[2]
        self.command = self.message[3]
        self.event = self.message[4]

        # Parses the prototype code and uses it to retrieve the prototype object from the prototypes dataclass instance
        prototype = SerialPrototypes.get_prototype_for_code(code=self.message[5])

        # If prototype retrieval fails, raises ValueError
        if prototype is None:
            message = (
                f"Invalid prototype code {self.message[5]} encountered when extracting the data object from "
                f"the received ModuleData message sent my module {self.module_id} of type {self.module_type}. All "
                f"data prototype codes have to be available from the SerialPrototypes class to be resolved."
            )
            console.error(message, ValueError)
        else:
            # Otherwise, uses the retrieved prototype to parse the data object
            self.data_object, _ = self._transport_layer.read_data(prototype, start_index=6)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleData object."""
        message = (
            f"ModuleData(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, event={self.event}, "
            f"data_object={self.data_object})."
        )
        return message


class KernelData:
    """Communicates the event state-code of the Kernel and includes an additional data object.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        command: The code of the command the Kernel was executing when it sent the message.
        event: The code of the event that prompted sending the message.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes enumeration.
        _transport_layer: Stores the reference to the TransportLayer class.
    """

    def __init__(self, transport_layer: TransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = SerialProtocols.KERNEL_DATA.as_uint8()

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)
        self.data_object: np.unsignedinteger[Any] | NDArray[Any] = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the static header data from the extracted message
        self.command = self.message[1]
        self.event = self.message[2]

        # Parses the prototype code and uses it to retrieve the prototype object from the prototypes dataclass instance
        prototype = SerialPrototypes.get_prototype_for_code(code=self.message[3])

        # If the prototype retrieval fails, raises ValueError.
        if prototype is None:
            message = (
                f"Invalid prototype code {self.message[3]} encountered when extracting the data object from "
                f"the received KernelData message. All data prototype codes have to be available from the "
                f"SerialPrototypes class to be resolved."
            )
            console.error(message, ValueError)

        else:
            # Otherwise, uses the retrieved prototype to parse the data object
            self.data_object, _ = self._transport_layer.read_data(prototype, start_index=4)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelData object."""
        message = (
            f"KernelData(protocol_code={self.protocol_code}, command={self.command}, event={self.event}, "
            f"data_object={self.data_object})."
        )
        return message


class ModuleState:
    """Communicates the event state-code of the sender Module.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        module_type: The type (family) code of the module that sent the message.
        module_id: The ID of the specific module within the broader module-family.
        command: The code of the command the module was executing when it sent the message.
        event: The code of the event that prompted sending the message.
    """

    def __init__(self, transport_layer: TransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = SerialProtocols.MODULE_STATE.as_uint8()

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.module_type: np.uint8 = np.uint8(0)
        self.module_id: np.uint8 = np.uint8(0)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.module_type = self.message[1]
        self.module_id = self.message[2]
        self.command = self.message[3]
        self.event = self.message[4]

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleState object."""
        message = (
            f"ModuleState(module_type={self.module_type}, module_id={self.module_id}, command={self.command}, "
            f"event={self.event})."
        )
        return message


class KernelState:
    """Communicates the event state-code of the Kernel.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        command: The code of the command the Kernel was executing when it sent the message.
        event: The code of the event that prompted sending the message.
    """

    def __init__(self, transport_layer: TransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = SerialProtocols.KERNEL_STATE.as_uint8()

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.command = self.message[1]
        self.event = self.message[2]

    def __repr__(self) -> str:
        """Returns a string representation of the KernelState object."""
        message = f"KernelState(command={self.command}, event={self.event})."
        return message


class ReceptionCode:
    """Returns the reception_code originally received from the PC to indicate that the message with that code was
    received and parsed.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        reception_code: The reception code originally sent as part of the outgoing Command or Parameters messages.
    """

    def __init__(self, transport_layer: TransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = SerialProtocols.RECEPTION_CODE.as_uint8()

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.reception_code: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.reception_code = self.message[1]

    def __repr__(self) -> str:
        """Returns a string representation of the ReceptionCode object."""
        message = f"ReceptionCode(reception_code={self.reception_code})."
        return message


class ControllerIdentification:
    """Identifies the connected microcontroller by communicating its unique byte id-code.

    For the ID codes to be unique, they have to be manually assigned to the Kernel class of each concurrently
    used microcontroller.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its
    lifetime to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        controller_id: The unique ID of the microcontroller. This ID is hardcoded in the microcontroller firmware
            and helps track which AXMC firmware is running on the given controller.

    """

    def __init__(self, transport_layer: TransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = SerialProtocols.CONTROLLER_IDENTIFICATION.as_uint8()

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.controller_id: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.controller_id = self.message[1]

    def __repr__(self) -> str:
        """Returns a string representation of the ControllerIdentification object."""
        message = f"ControllerIdentification(controller_id={self.controller_id})."
        return message


class ModuleIdentification:
    """Identifies a hardware module instance by communicating its combined type + id 16-bit code.

    It is expected that each hardware module instance will have a unique combination of type (family) code and instance
    (ID) code. The user assigns both type and ID codes at their discretion when writing the main .cpp file for each
    microcontroller.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its
    lifetime to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        module_type_id: The unique uint16 code that results from combining the type and ID codes of the module instance.

    """

    def __init__(self, transport_layer: TransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = SerialProtocols.MODULE_IDENTIFICATION.as_uint8()

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.module_type_id: np.uint16 = np.uint16(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data. Ignores the protocol code stored under index 0 and reads the 2-byte combined code
        # value
        self.module_type_id, _ = self._transport_layer.read_data(data_object=np.uint16(0), start_index=1)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleIdentification object."""
        message = f"ModuleIdentification(module_type_id={self.module_type_id})."
        return message


class SerialCommunication:
    """Wraps a TransportLayer class instance and exposes methods that allow communicating with a microcontroller
    running ataraxis-micro-controller library using the USB or UART protocol.

    This class is built on top of the TransportLayer, designed to provide the microcontroller communication
    interface (API) for other Ataraxis libraries. This class is not designed to be instantiated directly and
    should instead be used through the MicroControllerInterface class available through this library!

    Notes:
        This class is explicitly designed to use the same parameters as the Communication class used by the
        microcontroller. Do not modify this class unless you know what you are doing.

        Due to the use of many non-pickleable classes, this class cannot be piped to a remote process and has to be
        initialized by the remote process directly.

        This class is designed to integrate with DataLogger class available from the ataraxis_data_structures library.
        The DataLogger is used to write all incoming and outgoing messages to disk as serialized message payloads.

    Args:
        source_id: The ID code to identify the source of the logged messages. This is used by the DataLogger to
            distinguish between log sources (classes that sent data to be logged) and, therefore, has to be unique for
            all Ataraxis classes that use DataLogger and are active at the same time.
        microcontroller_serial_buffer_size: The size, in bytes, of the buffer used by the target microcontroller's
            Serial buffer. Usually, this information is available from the microcontroller's manufacturer
            (UART / USB controller specification).
        usb_port: The name of the USB port to use for communication to, e.g.: 'COM3' or '/dev/ttyUSB0'. This has to be
            the port to which the target microcontroller is connected. Use the list_available_ports() function available
            from this library to get the list of discoverable serial port names.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
            This queue is used to buffer and pipe data to be logged to the logger cores.
        baudrate: The baudrate to use for the communication over the UART protocol. Should match the value used by
            the microcontrollers that only support UART protocol. This is ignored for microcontrollers that use the
            USB protocol.
        test_mode: This parameter is only used during testing. When True, it initializes the underlying TransportLayer
            class in the test configuration. Make sure this is set to False during production runtime.

    Attributes:
        _transport_layer: The TransportLayer instance that handles the communication.
        _module_data: Received ModuleData messages are unpacked into this structure.
        _kernel_data: Received KernelData messages are unpacked into this structure.
        _module_state: Received ModuleState messages are unpacked into this structure.
        _kernel_state: Received KernelState messages are unpacked into this structure.
        _controller_identification: Received ControllerIdentification messages are unpacked into this structure.
        _module_identification: Received ModuleIdentification messages are unpacked into this structure.
        _reception_code: Received ReceptionCode messages are unpacked into this structure.
        _timestamp_timer: The PrecisionTimer instance used to stamp incoming and outgoing data as it is logged.
        _source_id: Stores the unique integer-code that identifies the class instance in data logs.
        _logger_queue: Stores the multiprocessing Queue that buffers and pipes the data to the Logger process(es).
        _usb_port: Stores the ID of the USB port used for communication.
    """

    def __init__(
        self,
        source_id: np.uint8,
        microcontroller_serial_buffer_size: int,
        usb_port: str,
        logger_queue: MPQueue,  # type: ignore
        baudrate: int = 115200,
        *,
        test_mode: bool = False,
    ) -> None:
        # Initializes the TransportLayer to mostly match a similar specialization carried out by the microcontroller
        # Communication class. This doubles up as an input argument check, as the class will raise an error if any
        # input argument is not valid.
        self._transport_layer = TransportLayer(
            port=usb_port,
            baudrate=baudrate,
            polynomial=np.uint16(0x1021),
            initial_crc_value=np.uint16(0xFFFF),
            final_crc_xor_value=np.uint16(0x0000),
            microcontroller_serial_buffer_size=microcontroller_serial_buffer_size,
            minimum_received_payload_size=2,  # Protocol (1) and uint8_t Service Message byte-code (1), 2 bytes total
            start_byte=129,
            delimiter_byte=0,
            timeout=20000,
            test_mode=test_mode,
        )

        # Pre-initializes the structures used to parse and store received message data.
        self._module_data = ModuleData(self._transport_layer)
        self._kernel_data = KernelData(self._transport_layer)
        self._module_state = ModuleState(self._transport_layer)
        self._kernel_state = KernelState(self._transport_layer)
        self._controller_identification = ControllerIdentification(self._transport_layer)
        self._module_identification = ModuleIdentification(self._transport_layer)
        self._reception_code = ReceptionCode(self._transport_layer)

        # Initializes the trackers used to id-stamp data sent to the logger via the logger_queue.
        self._timestamp_timer: PrecisionTimer = PrecisionTimer("us")
        self._source_id: np.uint8 = source_id  # uint8 type is used to enforce byte-range
        self._logger_queue = logger_queue

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all later delta time
        # readouts. The time is returned as an array of bytes.
        onset: NDArray[np.uint8] = get_timestamp(as_bytes=True)  # type: ignore
        self._timestamp_timer.reset()  # Immediately resets the timer to make it as close as possible to the onset time

        # Logs the onset timestamp. All further timestamps will be treated as integer time deltas (in microseconds)
        # relative to the onset timestamp.
        package = LogPackage(self._source_id, np.uint64(0), onset)  # Packages the id, timestamp, and data.
        self._logger_queue.put(package)
        self._usb_port = usb_port

    def __repr__(self) -> str:
        """Returns a string representation of the SerialCommunication object."""
        return f"SerialCommunication(usb_port={self._usb_port}, source_id={self._source_id})."

    def send_message(
        self,
        message: (
            RepeatedModuleCommand
            | OneOffModuleCommand
            | DequeueModuleCommand
            | KernelCommand
            | KernelParameters
            | ModuleParameters
        ),
    ) -> None:
        """Serializes the input command or parameters message and sends it to the connected microcontroller.

        This method relies on every valid outgoing message structure exposing a packed_data attribute, that contains
        the serialized payload data to be sent. Functionally, this method is a wrapper around the
        TransportLayer's write_data() and send_data() methods.

        Args:
            message: The command or parameters message to send to the microcontroller.
        """
        # Writes the pre-packaged data into the transmission buffer. Mypy flags packed_data as potentially None, but for
        # valid messages packed_data cannot be None, so this is a false positive.
        self._transport_layer.write_data(data_object=message.packed_data)

        # Constructs and sends the data message to the connected system.
        self._transport_layer.send_data()
        stamp = self._timestamp_timer.elapsed  # Stamps transmission time.

        # Logs the transmitted message data
        self._log_data(stamp, message.packed_data)  # type: ignore

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
    ):
        """Receives the incoming message from the connected microcontroller and parses into the appropriate structure.

        This method uses the protocol code, assumed to be stored in the first variable of each received payload, to
        determine how to parse the data. It then parses into a precreated message structure stored in class attributes.

        Notes:
            To optimize overall runtime speed, this class creates message structures for all supported messages at
            initialization and overwrites the appropriate message attribute with the data extracted from each received
            message payload. This method than returns the reference to the overwritten class attribute. Therefore,
            it is advised to copy or finish working with the structure returned by this method before receiving another
            message. Otherwise, it is possible that the received message will be used to overwrite the data of the
            previously referenced structure, leading to the loss of unprocessed / unsaved data.

        Returns:
            A reference the parsed message structure instance stored in class attributes, or None, if no message was
            received. Note, None return does not indicate an error, but rather indicates that the microcontroller did
            not send any data.

        Raises:
            ValueError: If the received message uses an invalid (unrecognized) message protocol code.

        """
        # Attempts to receive the data message. If there is no data to receive, returns None. This is a non-error,
        # no-message return case.
        if not self._transport_layer.receive_data():
            return None

        stamp = self._timestamp_timer.elapsed  # Otherwise, stamps message reception time.

        # If the data was received, first reads the protocol code, expected to be found as the first value of every
        # incoming payload. The protocol is a byte-value, so uses np.uint8 prototype.
        protocol, _ = self._transport_layer.read_data(np.uint8(0), start_index=0)

        # Uses the extracted protocol value to determine the type of the received message and process the received data.
        # All supported message structure classes expose an API method that allows them to process and parse the message
        # payload.
        if protocol == SerialProtocols.MODULE_DATA.as_uint8():
            self._module_data.update_message_data()
            self._log_data(stamp, self._module_data.message)
            return self._module_data

        if protocol == SerialProtocols.KERNEL_DATA.as_uint8():
            self._kernel_data.update_message_data()
            self._log_data(stamp, self._kernel_data.message)
            return self._kernel_data

        if protocol == SerialProtocols.MODULE_STATE.as_uint8():
            self._module_state.update_message_data()
            self._log_data(stamp, self._module_state.message)
            return self._module_state

        if protocol == SerialProtocols.KERNEL_STATE.as_uint8():
            self._kernel_state.update_message_data()
            self._log_data(stamp, self._kernel_state.message)
            return self._kernel_state

        if protocol == SerialProtocols.RECEPTION_CODE.as_uint8():
            self._reception_code.update_message_data()
            self._log_data(stamp, self._reception_code.message)
            return self._reception_code

        if protocol == SerialProtocols.CONTROLLER_IDENTIFICATION.as_uint8():
            self._controller_identification.update_message_data()
            self._log_data(stamp, self._controller_identification.message)
            return self._controller_identification

        if protocol == SerialProtocols.MODULE_IDENTIFICATION.as_uint8():
            self._module_identification.update_message_data()
            self._log_data(stamp, self._module_identification.message)
            return self._module_identification

        # If the protocol code is not resolved by any conditional above, it is not valid. Terminates runtime with a
        # ValueError
        message = (
            f"Invalid protocol code {protocol} encountered when attempting to parse a message received from the "
            f"microcontroller. All incoming messages have to use one of the valid incoming message protocol codes "
            f"available from the SerialProtocols enumeration."
        )
        console.error(message, error=ValueError)
        # Fallback to appease mypy
        raise ValueError(message)  # pragma: no cover

    def _log_data(self, timestamp: int, data: NDArray[np.uint8]) -> None:
        """Packages and sends the input data to teh DataLogger instance that writes it to disk.

        Args:
            timestamp: The value of the timestamp timer 'elapsed' property that communicates the number of elapsed
                microseconds relative to the 'onset' timestamp.
            data: The byte-serialized message payload that was sent or received.
        """
        # Packages the data to be logged into the appropriate tuple format (with ID variables)
        package = LogPackage(self._source_id, np.uint64(timestamp), data)

        # Sends the data to the logger
        self._logger_queue.put(package)


class MQTTCommunication:
    """Wraps an MQTT client and exposes methods for bidirectionally communicating with other clients connected to the
    same MQTT broker.

    This class leverages MQTT protocol on Python side and to establish bidirectional communication between the Python
    process running this class and other MQTT clients. Primarily, the class is intended to be used together with
    SerialCommunication class to transfer data between microcontrollers and the rest of the infrastructure used during
    runtime. Usually, both communication classes will be managed by the same process (core) that handles the necessary
    transformations to bridge MQTT and Serial communication protocols used by this library. This class is not designed
    to be instantiated directly and should instead be used through the MicroControllerInterface class available through
    this library!

    Notes:
        MQTT protocol requires a broker that facilitates communication, which this class does NOT provide. Make sure
        your infrastructure includes a working MQTT broker before using this class. See https://mqtt.org/ for more
        details.

    Args:
        ip: The IP address of the MQTT broker that facilitates the communication.
        port: The socket port used by the MQTT broker that facilitates the communication.
        monitored_topics: The list of MQTT topics which the class instance should subscribe to and monitor for incoming
            messages.

    Attributes:
        _ip: Stores the IP address of the MQTT broker.
        _port: Stores the port used by the broker's TCP socket.
        _connected: Tracks whether the class instance is currently connected to the MQTT broker.
        _monitored_topics: Stores the topics the class should monitor for incoming messages sent by other MQTT clients.
        _output_queue: A multithreading queue used to buffer incoming messages received from other MQTT clients before
            their data is requested via class methods.
        _client: Stores the initialized mqtt client instance that carries out the communication.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 1883,
        monitored_topics: None | tuple[str, ...] = None,
    ) -> None:
        self._ip: str = ip
        self._port: int = port
        self._connected = False
        self._monitored_topics: tuple[str, ...] = monitored_topics if monitored_topics is not None else tuple()

        # Initializes the queue to buffer incoming data. The queue may not be used if the class is not configured to
        # receive any data, but this is a fairly minor inefficiency.
        self._output_queue: Queue = Queue()  # type: ignore

        # Initializes the MQTT client. Note, it needs to be connected before it can send and receive messages!
        # noinspection PyArgumentList,PyUnresolvedReferences
        self._client: mqtt.Client = mqtt.Client(
            protocol=mqtt.MQTTv5,
            transport="tcp",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,  # type: ignore
        )

    def __repr__(self) -> str:
        """Returns a string representation of the MQTTCommunication object."""
        return (
            f"MQTTCommunication(broker_ip={self._ip}, socket_port={self._port}, connected={self._connected}, "
            f"subscribed_topics={self._monitored_topics}"
        )

    def __del__(self) -> None:
        """Ensures proper resource release when the class instance is garbage-collected."""
        self.disconnect()

    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:  # pragma: no cover
        """The callback function used to receive data from MQTT broker.

        When passed to the client, this function will be called each time a new message is received. This function
        will then record the message topic and payload and put them into the output_queue for the data to be consumed
        by external processes.

        Args:
            _client: The MQTT client that received the message. Currently not used.
            _userdata: Custom user-defined data. Currently not used.
            message: The received MQTT message.
        """
        # Whenever a message is received, it is buffered via the local queue object.
        self._output_queue.put_nowait((message.topic, message.payload))

    def connect(self) -> None:
        """Connects to the MQTT broker and subscribes to the requested input topics.

        This method has to be called to initialize communication, both for incoming and outgoing messages. Any message
        sent to the MQTT broker from other clients before this method is called may not reach this class.

        Notes:
            If this class instance subscribes (listens) to any topics, it will start a perpetually active thread
            with a listener callback to monitor incoming traffic.

        Raises:
            RuntimeError: If the MQTT broker cannot be connected to using the provided IP and Port.
        """
        # Guards against re-connecting an already connected client.
        if self._connected:
            return

        # Connects to the broker
        try:
            result = self._client.connect(self._ip, self._port)
            if result != mqtt.MQTT_ERR_SUCCESS:
                # If the result is not the expected code, raises an exception
                raise Exception  # pragma: no cover
        # The exception can also be raised by connect() method raising an exception internally.
        except Exception:
            message = (
                f"Unable to connect MQTTCommunication class instance to the MQTT broker. Failed to connect to MQTT "
                f"broker at {self._ip}:{self._port}. This likely indicates that the broker is not running or that "
                f"there is an issue with the provided IP and socket port."
            )
            console.error(message, error=RuntimeError)

        # If the class is configured to connect to any topics, enables the connection callback and starts the monitoring
        # thread.
        if len(self._monitored_topics) != 0:
            # Adds the callback function and starts the monitoring loop.
            self._client.on_message = self._on_message
            self._client.loop_start()

        # Subscribes to necessary topics with qos of 0. Note, this assumes that the communication is happening over
        # a virtual TCP socket and, therefore, does not need qos.
        for topic in self._monitored_topics:
            self._client.subscribe(topic=topic, qos=0)

        # Sets the connected flag
        self._connected = True

    def send_data(self, topic: str, payload: str | bytes | bytearray | float | None = None) -> None:
        """Publishes the input payload to the specified MQTT topic.

        This method should be used for sending data to MQTT via one of the input topics. This method does not verify
        the validity of the input topic or payload data.

        Args:
            topic: The MQTT topic to publish the data to.
            payload: The data to be published. When set to None, an empty message will be sent, which is often used as
                a boolean trigger.

        Raises:
            RuntimeError: If the instance is not connected to the MQTT broker.
        """
        if not self._connected:
            message = (
                f"Cannot send data to the MQTT broker at {self._ip}:{self._port} via the MQTTCommunication instance. "
                f"The MQTTCommunication instance is not connected to the MQTT broker, call connect() method before "
                f"sending data."
            )
            console.error(message=message, error=RuntimeError)
        self._client.publish(topic=topic, payload=payload, qos=0)

    @property
    def has_data(self) -> bool:
        """Returns True if the instance received messages from other MQTT clients and can output received data via the
        get_dataq() method.
        """
        if not self._output_queue.empty():
            return True
        return False

    def get_data(self) -> tuple[str, bytes | bytearray] | None:
        """Extracts and returns the first available message stored inside the instance buffer queue.

        Returns:
            A two-element tuple. The first element is a string that communicates the MQTT topic of the received message.
            The second element is the payload of the message, which is a bytes or bytearray object. If no buffered
            objects are stored in the queue (queue is empty), returns None.

        Raises:
            RuntimeError: If the instance is not connected to the MQTT broker.
        """
        if not self._connected:
            message = (
                f"Cannot get data from the MQTT broker at {self._ip}:{self._port} via the MQTTCommunication instance. "
                f"The MQTTCommunication instance is not connected to the MQTT broker, call connect() method before "
                f"sending data."
            )
            console.error(message=message, error=RuntimeError)

        if not self.has_data:
            return None

        data: tuple[str, bytes | bytearray] = self._output_queue.get_nowait()
        return data

    def disconnect(self) -> None:
        """Disconnects the client from the MQTT broker."""
        # Prevents running the rest of the code if the client was not connected.
        if not self._connected:
            return

        # Stops the listener thread if the client was subscribed to receive topic data.
        if len(self._monitored_topics) != 0:
            self._client.loop_stop()

        # Disconnects from the client.
        self._client.disconnect()

        # Sets the connection flag
        self._connected = False
