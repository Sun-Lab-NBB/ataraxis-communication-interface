"""This module provides the SerialCommunication and MQTTCommunication classes that jointly support the communication
between and within host-machines (PCs) and Arduino / Teensy microcontrollers.
"""

from enum import IntEnum
from queue import Queue
from typing import Any
from dataclasses import field, dataclass
from collections.abc import Callable
from multiprocessing import Queue as MPQueue

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer, TimerPrecisions, TimestampFormats
import paho.mqtt.client as mqtt
from ataraxis_base_utilities import console
from ataraxis_data_structures import LogPackage
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_transport_layer_pc import TransportLayer

# Defines constants frequently used in this module
_ZERO_BYTE = np.uint8(0)
_ZERO_SHORT = np.uint16(0)
_ZERO_LONG = np.uint32(0)
_TRUE = np.bool(True)


class SerialProtocols(IntEnum):
    """Defines the protocol codes used to specify incoming and outgoing message layouts during PC-microcontroller
    communication.

    Notes:
        The elements in this enumeration should be accessed through their 'as_uint8' property to enforce
        the type expected by other classes from this library.
    """

    UNDEFINED = 0
    """Not a valid protocol code. Used to initialize the SerialCommunication class."""

    REPEATED_MODULE_COMMAND = 1
    """Used by Module-addressed commands that should be repeated (executed recurrently)."""

    ONE_OFF_MODULE_COMMAND = 2
    """Used by Module-addressed commands that should not be repeated (executed only once)."""

    DEQUEUE_MODULE_COMMAND = 3
    """Used by Module-addressed commands that remove all queued commands, including recurrent commands."""

    KERNEL_COMMAND = 4
    """Used by Kernel-addressed commands. All Kernel commands are always non-repeatable (one-shot)."""

    MODULE_PARAMETERS = 5
    """Used by Module-addressed parameter messages."""

    MODULE_DATA = 6
    """Used by Module data or error messages that include an arbitrary data object in addition to the event state-code.
    """

    KERNEL_DATA = 7
    """Used by Kernel data or error messages that include an arbitrary data object in addition to event state-code."""

    MODULE_STATE = 8
    """Used by Module data or error messages that only include the state-code."""

    KERNEL_STATE = 9
    """Used by Kernel data or error messages that only include the state-code."""

    RECEPTION_CODE = 10
    """Used to acknowledge the reception of command and parameter messages from the PC."""

    CONTROLLER_IDENTIFICATION = 11
    """Used to identify the host-microcontroller to the PC."""

    MODULE_IDENTIFICATION = 12
    """Used to identify the hardware module instances managed by the microcontroller's Kernel instance to the PC."""

    def as_uint8(self) -> np.uint8:
        """Returns the specified enumeration element as a numpy uint8 type."""
        return np.uint8(self.value)


# Type alias for supported numpy types
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


# Defines prototype factories for all supported types
_PROTOTYPE_FACTORIES: dict[int, Callable[[], PrototypeType]] = {
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
    """Defines the prototype codes used during data transmission to specify the layout of additional data objects
    transmitted by KernelData and ModuleData messages.
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
        """Returns the specified the enumeration value as a numpy uint8 type."""
        return np.uint8(self.value)

    # noinspection PyTypeHints
    def get_prototype(self) -> PrototypeType:
        """Returns the prototype object associated with the prototype enumeration value."""
        return _PROTOTYPE_FACTORIES[self.value]()

    # noinspection PyTypeHints
    @classmethod
    def get_prototype_for_code(cls, code: np.uint8) -> PrototypeType | None:
        """Returns the prototype object associated with the input prototype code.

        Args:
            code: The prototype byte-code for which to retrieve the prototype object.

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
    """Instructs the addressed Module instance to run the specified command repeatedly (recurrently)."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module instance within the broader module family."""
    command: np.uint8
    """The code of the command to execute."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
    noblock: np.bool_ = _TRUE
    """Determines whether to allow concurrent execution of other commands while waiting for the requested command to 
    complete."""
    cycle_delay: np.uint32 = _ZERO_LONG
    """The delay, in microseconds, before repeating (cycling) the command."""
    # noinspection PyTypeHints
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.REPEATED_MODULE_COMMAND.as_uint8())
    """Stores the message protocol code."""

    def __post_init__(self) -> None:
        """Serializes the instance's data."""
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
        """Returns the string representation of the instance."""
        return (
            f"RepeatedModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock}, cycle_delay={self.cycle_delay} us)."
        )


@dataclass(frozen=True)
class OneOffModuleCommand:
    """Instructs the addressed Module instance to run the specified command exactly once (non-recurrently)."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module instance within the broader module family."""
    command: np.uint8
    """The code of the command to execute."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
    noblock: np.bool_ = _TRUE
    """Determines whether to allow concurrent execution of other commands while waiting for the requested command to 
    complete."""
    # noinspection PyTypeHints
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.ONE_OFF_MODULE_COMMAND.as_uint8())
    """Stores the message protocol code."""

    def __post_init__(self) -> None:
        """Serializes the instance's data."""
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
        """Returns the string representation of the instance."""
        return (
            f"OneOffModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock})."
        )


@dataclass(frozen=True)
class DequeueModuleCommand:
    """Instructs the addressed Module instance to clear (empty) its command queue."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module instance within the broader module family."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
    # noinspection PyTypeHints
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.DEQUEUE_MODULE_COMMAND.as_uint8())
    """Stores the message protocol code."""

    def __post_init__(self) -> None:
        """Serializes the instance's data."""
        packed = np.empty(4, dtype=np.uint8)
        packed[0:4] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return (
            f"DequeueModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code})."
        )


@dataclass(frozen=True)
class KernelCommand:
    """Instructs the Kernel to run the specified command exactly once."""

    command: np.uint8
    """The code of the command to execute."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
    # noinspection PyTypeHints
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the serialized message data."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.KERNEL_COMMAND.as_uint8())
    """Stores the message protocol code."""

    def __post_init__(self) -> None:
        """Serializes the instance's data."""
        packed = np.empty(3, dtype=np.uint8)
        packed[0:3] = [
            self.protocol_code,
            self.return_code,
            self.command,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return (
            f"KernelCommand(protocol_code={self.protocol_code}, command={self.command}, "
            f"return_code={self.return_code})."
        )


@dataclass(frozen=True)
class ModuleParameters:
    """Instructs the addressed Module instance to update its parameters with the included data."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module instance within the broader module family."""
    # noinspection PyTypeHints
    parameter_data: tuple[np.number[Any] | np.bool, ...]
    """A tuple of parameter values to send. The values inside the tuple must match the type and format of the values 
    used in the addressed module's parameter structure on the microcontroller."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
    # noinspection PyTypeHints
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the serialized message data."""
    # noinspection PyTypeHints
    parameters_size: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the total size of the serialized parameters in bytes."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.MODULE_PARAMETERS.as_uint8())
    """Stores the message protocol code."""

    def __post_init__(self) -> None:
        """Serializes the instance's data."""
        # Converts scalar parameter values to byte arrays (serializes them)
        byte_parameters = [np.frombuffer(np.array([param]), dtype=np.uint8).copy() for param in self.parameter_data]

        # Calculates the total size of serialized parameters in bytes and adds it to the parameters_size attribute
        parameters_size = np.uint8(sum(param.size for param in byte_parameters))
        object.__setattr__(self, "parameters_size", parameters_size)

        # Pre-allocates the full array with the exact size (header and parameters object)
        packed_data = np.empty(4 + parameters_size, dtype=np.uint8)

        # Packs the header data into the pre-created array
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
        """Returns the string representation of the instance."""
        return (
            f"ModuleParameters(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code}, "
            f"parameter_object_size={self.parameters_size} bytes)."
        )


@dataclass
class ModuleData:
    """Communicates that the Module has encountered a notable event and includes an additional data object."""

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=5, dtype=np.uint8))
    """The parsed message header data."""
    data_object: np.number[Any] | NDArray[Any] = _ZERO_BYTE
    """The parsed data object transmitted with the message."""

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return (
            f"ModuleData(module_type={self.message[0]}, module_id={self.message[1]}, command={self.message[2]}, "
            f"event={self.message[3]}, data_object={self.data_object})."
        )

    @property
    def module_type(self) -> np.uint8:
        """Returns the type (family) code of the module that sent the message."""
        return np.uint8(self.message[0])

    @property
    def module_id(self) -> np.uint8:
        """Returns the unique identifier code of the module instance that sent the message."""
        return np.uint8(self.message[1])

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the module that sent the message."""
        return np.uint8(self.message[2])

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        return np.uint8(self.message[3])

    @property
    def prototype_code(self) -> np.uint8:
        """Returns the code that specifies the type of the data object transmitted with the message."""
        return np.uint8(self.message[4])


@dataclass
class KernelData:
    """Communicates that the Kernel has encountered a notable event and includes an additional data object."""

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=3, dtype=np.uint8))
    """The parsed message header data."""
    data_object: np.number[Any] | NDArray[Any] = _ZERO_BYTE
    """The parsed data object transmitted with the message."""

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return f"KernelData(command={self.message[0]}, event={self.message[1]}, data_object={self.data_object})."

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the Kernel when it sent the message."""
        return np.uint8(self.message[0])

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        return np.uint8(self.message[1])

    @property
    def prototype_code(self) -> np.uint8:
        """Returns the code that specifies the type of the data object transmitted with the message."""
        return np.uint8(self.message[2])


@dataclass
class ModuleState:
    """Communicates that the Module has encountered a notable event."""

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=4, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return (
            f"ModuleState(module_type={self.message[0]}, module_id={self.message[1]}, "
            f"command={self.message[2]}, event={self.message[3]})."
        )

    @property
    def module_type(self) -> np.uint8:
        """Returns the type (family) code of the module that sent the message."""
        return np.uint8(self.message[0])

    @property
    def module_id(self) -> np.uint8:
        """Returns the ID of the specific module instance within the broader module family."""
        return np.uint8(self.message[1])

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the module that sent the message."""
        return np.uint8(self.message[2])

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        return np.uint8(self.message[3])


@dataclass
class KernelState:
    """Communicates that the Kernel has encountered a notable event."""

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=2, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return f"KernelState(command={self.message[0]}, event={self.message[1]})."

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the Kernel when it sent the message."""
        return np.uint8(self.message[0])

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        return np.uint8(self.message[1])


@dataclass
class ReceptionCode:
    """Communicates the reception code originally received with the message sent by the PC to indicate that the message
    was received and parsed by the microcontroller.
    """

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=1, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return f"ReceptionCode(reception_code={self.message[0]})."

    @property
    def reception_code(self) -> np.uint8:
        """Returns the reception code originally sent as part of the outgoing Command or Parameters message."""
        return np.uint8(self.message[0])


@dataclass
class ControllerIdentification:
    """Communicates the unique identifier code of the microcontroller."""

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=1, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return f"ControllerIdentification(controller_id={self.message[0]})."

    @property
    def controller_id(self) -> np.uint8:
        """Returns the unique identifier of the microcontroller."""
        return np.uint8(self.message[0])


@dataclass
class ModuleIdentification:
    """Identifies a hardware module instance by communicating its combined type and id code."""

    module_type_id: np.uint16 = _ZERO_SHORT
    """The unique uint16 code that results from combining the type and ID codes of the module instance."""

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return f"ModuleIdentification(module_type_id={self.module_type_id})."


class SerialCommunication:
    """Provides methods for bidirectionally communicating with a microcontroller running the ataraxis-micro-controller
    library over the USB or UART serial interface.

    Notes:
        This class is explicitly designed to be used by other library assets and should not be used directly by end
        users. An instance of this class is initialized and managed by the MicrocontrollerInterface class.

    Args:
        controller_id: The identifier code of the microcontroller to communicate with.
        microcontroller_serial_buffer_size: The size, in bytes, of the buffer used by the communicated microcontroller's
            serial communication interface. Usually, this information is available from the microcontroller's
            manufacturer (UART / USB controller specification).
        port: The name of the serial port to connect to, e.g.: 'COM3' or '/dev/ttyUSB0'.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger instance used to pipe the data to be
            logged to the logger process.
        baudrate: The baudrate to use for communication if the microcontroller uses the UART interface. Must match
            the value used by the microcontroller. This parameter is ignored when using the USB interface.
        test_mode: Determines whether the instance uses a pySerial (real) or a StreamMock (mocked) communication
            interface. This flag is used during testing and should be disabled for all production runtimes.

    Attributes:
        _transport_layer: The TransportLayer instance that handles the communication.
        _module_data: Stores the data of the last received ModuleData message.
        _kernel_data: Stores the data of the last received KernelData message.
        _module_state: Stores the data of the last received ModuleState message.
        _kernel_state: Stores the data of the last received KernelState message.
        _controller_identification: Stores the data of the last received ControllerIdentification message.
        _module_identification: Stores the data of the last received ModuleIdentification message.
        _reception_code: Stores the data of the last received ReceptionCode message.
        _timestamp_timer: Stores the PrecisionTimer instance used to timestamp incoming and outgoing data as it is
            being saved (logged) to disk.
        _source_id: Stores the unique identifier of the microcontroller with which the instance communicates at runtime.
        _logger_queue: Stores the multiprocessing Queue that buffers and pipes the data to the DataLogger process(es).
        _usb_port: Stores the ID of the USB port used for communication.
    """

    def __init__(
        self,
        controller_id: np.uint8,
        microcontroller_serial_buffer_size: int,
        port: str,
        logger_queue: MPQueue,  # type: ignore[type-arg]
        baudrate: int = 115200,
        *,
        test_mode: bool = False,
    ) -> None:
        # Initializes the TransportLayer to mostly match a similar specialization carried out by the microcontroller
        # Communication class.
        self._transport_layer = TransportLayer(
            port=port,
            baudrate=baudrate,
            polynomial=np.uint16(0x1021),
            initial_crc_value=np.uint16(0xFFFF),
            final_crc_xor_value=np.uint16(0x0000),
            microcontroller_serial_buffer_size=microcontroller_serial_buffer_size,
            test_mode=test_mode,
        )

        # Pre-initializes the structures used to store the received message data.
        self._module_data = ModuleData()
        self._kernel_data = KernelData()
        self._module_state = ModuleState()
        self._kernel_state = KernelState()
        self._controller_identification = ControllerIdentification()
        self._module_identification = ModuleIdentification()
        self._reception_code = ReceptionCode()

        # Initializes the trackers used to timestamp the data sent to the logger via the logger_queue.
        self._timestamp_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MICROSECOND)
        self._source_id: np.uint8 = controller_id  # uint8 type is used to enforce byte-range
        self._logger_queue: MPQueue = logger_queue  # type: ignore[type-arg]

        # Constructs a timezone-aware stamp using the UTC time. This creates a reference point for all later delta time
        # readouts.
        onset: NDArray[np.uint8] = get_timestamp(output_format=TimestampFormats.BYTES)  # type: ignore[assignment]
        self._timestamp_timer.reset()  # Immediately resets the timer to make it as close as possible to the onset time

        # Logs the onset timestamp. All further timestamps are treated as integer time deltas (in microseconds)
        # relative to the onset timestamp.
        package = LogPackage(source_id=self._source_id, acquisition_time=np.uint64(0), serialized_data=onset)
        self._logger_queue.put(package)
        self._usb_port: str = port

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return f"SerialCommunication(usb_port={self._usb_port}, controller_id={self._source_id})."

    def send_message(
        self,
        message: (
            RepeatedModuleCommand | OneOffModuleCommand | DequeueModuleCommand | KernelCommand | ModuleParameters
        ),
    ) -> None:
        """Serializes the input message and sends it to the connected microcontroller.

        Args:
            message: The message to send to the microcontroller.
        """
        # Writes the pre-packaged data into the transmission buffer.
        self._transport_layer.write_data(data_object=message.packed_data)

        # Constructs and sends the data message to the connected system.
        self._transport_layer.send_data()

        # Logs the transmitted data to disk
        self._log_data(self._timestamp_timer.elapsed, message.packed_data)  # type: ignore[arg-type]

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
        """Receives a message sent by the microcontroller and parses its contents into the appropriate instance
        attribute.

        Notes:
            Each call to this method overwrites the previously received message data stored in the instance's
            attributes. It is advised to finish working with the received message data before receiving another message.

        Returns:
            A reference to the parsed message data stored as an instance's attribute, or None, if no message was
            received.

        Raises:
            ValueError: If the received message uses an invalid (unrecognized) message protocol code. If the received
                data message uses an unsupported data object prototype code.

        """
        # Attempts to receive the data message. If there is no data to receive, returns None. This is a non-error,
        # no-message return case.
        if not self._transport_layer.receive_data():
            return None

        # Timestamps and logs the serialized message data to disk before further processing.
        self._log_data(
            self._timestamp_timer.elapsed,
            self._transport_layer.reception_buffer[: self._transport_layer.bytes_in_reception_buffer],
        )

        # Reads the message protocol code, expected to be found as the first value of every incoming payload. This
        # code determines how to parse the message's payload
        protocol = self._transport_layer.read_data(data_object=np.uint8(0))

        # Uses the extracted protocol code to determine the type of the received message and process the received data.
        if protocol == SerialProtocols.MODULE_DATA.as_uint8():
            # Parses the static header data from the extracted message
            self._module_data.message = self._transport_layer.read_data(data_object=self._module_data.message)

            # Resolves the prototype code and uses it to retrieve the prototype object from the prototypes dataclass
            # instance
            prototype = SerialPrototypes.get_prototype_for_code(code=self._module_data.prototype_code)

            # If prototype retrieval fails, raises ValueError
            if prototype is None:
                message = (
                    f"Invalid prototype code {self._module_data.prototype_code} encountered when extracting the data "
                    f"object from the received ModuleData message sent my module {self._module_data.module_id} of type "
                    f"{self._module_data.module_type}. All messages must use one of the valid prototype "
                    f"codes available from the SerialPrototypes enumeration."
                )
                console.error(message, ValueError)
            else:
                # Otherwise, uses the retrieved prototype to parse the data object
                self._module_data.data_object = self._transport_layer.read_data(data_object=prototype)

            return self._module_data

        if protocol == SerialProtocols.KERNEL_DATA.as_uint8():
            # Parses the static header data from the extracted message
            self._kernel_data.message = self._transport_layer.read_data(data_object=self._kernel_data.message)

            # Resolves the prototype code and uses it to retrieve the prototype object from the prototypes dataclass
            # instance
            prototype = SerialPrototypes.get_prototype_for_code(code=self._kernel_data.prototype_code)

            # If the prototype retrieval fails, raises ValueError.
            if prototype is None:
                message = (
                    f"Invalid prototype code {self._kernel_data.prototype_code} encountered when extracting the data "
                    f"object from the received KernelData message. All messages must use one of the valid prototype "
                    f"codes available from the SerialPrototypes enumeration."
                )
                console.error(message, ValueError)

            else:
                # Otherwise, uses the retrieved prototype to parse the data object
                self._kernel_data.data_object = self._transport_layer.read_data(data_object=prototype)

            return self._kernel_data

        if protocol == SerialProtocols.MODULE_STATE.as_uint8():
            self._module_state.message = self._transport_layer.read_data(data_object=self._module_state.message)
            return self._module_state

        if protocol == SerialProtocols.KERNEL_STATE.as_uint8():
            self._kernel_state.message = self._transport_layer.read_data(data_object=self._kernel_state.message)
            return self._kernel_state

        if protocol == SerialProtocols.RECEPTION_CODE.as_uint8():
            self._reception_code.message = self._transport_layer.read_data(data_object=self._reception_code.message)
            return self._reception_code

        if protocol == SerialProtocols.CONTROLLER_IDENTIFICATION.as_uint8():
            self._controller_identification.message = self._transport_layer.read_data(
                data_object=self._controller_identification.message
            )
            return self._controller_identification

        if protocol == SerialProtocols.MODULE_IDENTIFICATION.as_uint8():
            # Since the entire message payload is the uint16 type-id value, read the value directly into the
            # module_type_id attribute.
            self._module_identification.module_type_id = self._transport_layer.read_data(
                data_object=self._module_identification.module_type_id
            )
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
        """Packages and sends the input data to the DataLogger instance that writes it to disk.

        Args:
            timestamp: The value of the timestamp timer's 'elapsed' property that communicates the number of elapsed
                microseconds relative to the 'onset' timestamp at the time of data acquisition.
            data: The serialized message payload to be logged.
        """
        # Packages the data to be logged into the appropriate tuple format (with ID variables)
        package = LogPackage(self._source_id, np.uint64(timestamp), data)

        # Sends the data to the logger
        self._logger_queue.put(package)


class MQTTCommunication:
    """Provides methods for bidirectionally communicating with other clients connected to the same MQTT broker using the
    MQTT protocol over the TCP interface.

    Notes:
        Primarily, the class is intended to be used alongside the SerialCommunication class to transfer the data between
        microcontrollers and the rest of the runtime infrastructure.

        The MQTT protocol requires a broker that facilitates the communication, which has to be available to this class
        at initialization. See https://mqtt.org/ for more details.

    Args:
        ip: The IP address of the MQTT broker.
        port: The socket port used by the MQTT broker.
        monitored_topics: The list of MQTT topics to monitor for incoming messages.

    Attributes:
        _ip: Stores the IP address of the MQTT broker.
        _port: Stores the port used by the broker's TCP socket.
        _connected: Tracks whether the class instance is currently connected to the MQTT broker.
        _monitored_topics: Stores the topics monitored by the instance for incoming messages.
        _output_queue: Buffers incoming messages received from other MQTT clients before their data is accessed via
            class methods.
        _client: The initialized MQTT client instance that carries out the communication.
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
        self._monitored_topics: tuple[str, ...] = monitored_topics if monitored_topics is not None else ()

        # Initializes the queue to buffer incoming data. The queue may not be used if the class is not configured to
        # receive any data, but this is a fairly minor inefficiency.
        self._output_queue: Queue = Queue()  # type: ignore[type-arg]

        # Initializes the MQTT client. Note, it needs to be connected before it can send and receive messages!
        # noinspection PyArgumentList,PyUnresolvedReferences
        self._client: mqtt.Client = mqtt.Client(  # type: ignore[call-arg]
            protocol=mqtt.MQTTv5,
            transport="tcp",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,  # type: ignore[attr-defined]
        )

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
        return (
            f"MQTTCommunication(broker_ip={self._ip}, socket_port={self._port}, connected={self._connected}, "
            f"subscribed_topics={self._monitored_topics}"
        )

    def __del__(self) -> None:
        """Ensures that the instance disconnects from the broker before being garbage-collected."""
        self.disconnect()

    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:  # pragma: no cover
        """The custom callback method used to receive data from the MQTT broker.

        This method records the topic and payload of each received message and puts them into the output_queue.

        Args:
            _client: The MQTT client that received the message. Currently not used.
            _userdata: Custom user-defined data. Currently not used.
            message: The received MQTT message.
        """
        # Whenever a message is received, it is buffered via the local queue object.
        self._output_queue.put_nowait((message.topic, message.payload))

    def connect(self) -> None:
        """Connects to the MQTT broker and subscribes to the requested list of monitored topics.

        Notes:
            This method has to be called after class initialization to start the communication process. Any message
            sent to the MQTT broker from other clients before this method is called may not reach this instance.

            If this instance is configured to subscribe (listen) to any topics, it starts a perpetually active thread
            with a listener callback to monitor the incoming traffic.

        Raises:
            ConnectionError: If the MQTT broker cannot be connected using the provided IP and Port.
        """
        # Guards against re-connecting an already connected client.
        if self._connected:
            return

        # Connects to the broker
        try:
            result = self._client.connect(self._ip, self._port)
        except TimeoutError:  # Fixed a minor regression in the newest MQTT version that raises Python errors
            result = mqtt.MQTT_ERR_NO_CONN
        if result != mqtt.MQTT_ERR_SUCCESS:
            # If the result is not the expected code, raises an exception
            message = (
                f"Unable to connect MQTTCommunication class instance to the MQTT broker. Failed to connect to MQTT "
                f"broker at {self._ip}:{self._port}. This likely indicates that the broker is not running or that "
                f"there is an issue with the provided IP and socket port."
            )
            console.error(message, error=ConnectionError)

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

        Args:
            topic: The MQTT topic to publish the data to.
            payload: The data to be published. Setting this to None sends an empty message.

        Raises:
            ConnectionError: If the instance is not connected to the MQTT broker.
        """
        if not self._connected:
            message = (
                f"Cannot send data to the MQTT broker at {self._ip}:{self._port} via the MQTTCommunication instance. "
                f"The MQTTCommunication instance is not connected to the MQTT broker, call connect() method before "
                f"sending data."
            )
            console.error(message=message, error=ConnectionError)
        self._client.publish(topic=topic, payload=payload, qos=0)

    @property
    def has_data(self) -> bool:
        """Returns True if the instance's get_data() method can be used to retrieve a message received from another
        MQTT client.
        """
        return bool(not self._output_queue.empty())

    def get_data(self) -> tuple[str, bytes | bytearray] | None:
        """Extracts and returns the first available message stored inside the instance's buffer queue.

        Returns:
            A two-element tuple if there is data to retrieve. The first element is the MQTT topic of the received
            message. The second element is the payload of the message. If there is no data to retrieve, returns None.

        Raises:
            ConnectionError: If the instance is not connected to the MQTT broker.
        """
        if not self._connected:
            message = (
                f"Cannot get data from the MQTT broker at {self._ip}:{self._port} via the MQTTCommunication instance. "
                f"The MQTTCommunication instance is not connected to the MQTT broker, call connect() method before "
                f"sending data."
            )
            console.error(message=message, error=ConnectionError)

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
