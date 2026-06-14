"""Provides the SerialProtocols and SerialPrototypes enumerations that define the message protocol and data prototype
codes used during PC-microcontroller serial communication.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


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
    # Extended prototypes (codes 166-252)
    # bool extended (16 bytes to 248 bytes)
    166: lambda: np.zeros(16, dtype=np.bool_),  # kSixteenBools
    167: lambda: np.zeros(24, dtype=np.bool_),  # kTwentyFourBools
    168: lambda: np.zeros(32, dtype=np.bool_),  # kThirtyTwoBools
    169: lambda: np.zeros(40, dtype=np.bool_),  # kFortyBools
    170: lambda: np.zeros(48, dtype=np.bool_),  # kFortyEightBools
    171: lambda: np.zeros(52, dtype=np.bool_),  # kFiftyTwoBools
    172: lambda: np.zeros(248, dtype=np.bool_),  # kTwoHundredFortyEightBools
    # uint8_t extended (16 bytes to 248 bytes)
    173: lambda: np.zeros(16, dtype=np.uint8),  # kSixteenUint8s
    174: lambda: np.zeros(18, dtype=np.uint8),  # kEighteenUint8s
    175: lambda: np.zeros(20, dtype=np.uint8),  # kTwentyUint8s
    176: lambda: np.zeros(22, dtype=np.uint8),  # kTwentyTwoUint8s
    177: lambda: np.zeros(24, dtype=np.uint8),  # kTwentyFourUint8s
    178: lambda: np.zeros(28, dtype=np.uint8),  # kTwentyEightUint8s
    179: lambda: np.zeros(32, dtype=np.uint8),  # kThirtyTwoUint8s
    180: lambda: np.zeros(36, dtype=np.uint8),  # kThirtySixUint8s
    181: lambda: np.zeros(40, dtype=np.uint8),  # kFortyUint8s
    182: lambda: np.zeros(44, dtype=np.uint8),  # kFortyFourUint8s
    183: lambda: np.zeros(48, dtype=np.uint8),  # kFortyEightUint8s
    184: lambda: np.zeros(52, dtype=np.uint8),  # kFiftyTwoUint8s
    185: lambda: np.zeros(64, dtype=np.uint8),  # kSixtyFourUint8s
    186: lambda: np.zeros(96, dtype=np.uint8),  # kNinetySixUint8s
    187: lambda: np.zeros(128, dtype=np.uint8),  # kOneHundredTwentyEightUint8s
    188: lambda: np.zeros(192, dtype=np.uint8),  # kOneHundredNinetyTwoUint8s
    189: lambda: np.zeros(244, dtype=np.uint8),  # kTwoHundredFortyFourUint8s
    190: lambda: np.zeros(248, dtype=np.uint8),  # kTwoHundredFortyEightUint8s
    # int8_t extended (16 bytes to 248 bytes)
    191: lambda: np.zeros(16, dtype=np.int8),  # kSixteenInt8s
    192: lambda: np.zeros(24, dtype=np.int8),  # kTwentyFourInt8s
    193: lambda: np.zeros(32, dtype=np.int8),  # kThirtyTwoInt8s
    194: lambda: np.zeros(40, dtype=np.int8),  # kFortyInt8s
    195: lambda: np.zeros(48, dtype=np.int8),  # kFortyEightInt8s
    196: lambda: np.zeros(52, dtype=np.int8),  # kFiftyTwoInt8s
    197: lambda: np.zeros(92, dtype=np.int8),  # kNinetyTwoInt8s
    198: lambda: np.zeros(132, dtype=np.int8),  # kOneHundredThirtyTwoInt8s
    199: lambda: np.zeros(172, dtype=np.int8),  # kOneHundredSeventyTwoInt8s
    200: lambda: np.zeros(212, dtype=np.int8),  # kTwoHundredTwelveInt8s
    201: lambda: np.zeros(244, dtype=np.int8),  # kTwoHundredFortyFourInt8s
    202: lambda: np.zeros(248, dtype=np.int8),  # kTwoHundredFortyEightInt8s
    # uint16_t extended (32 bytes to 248 bytes)
    203: lambda: np.zeros(16, dtype=np.uint16),  # kSixteenUint16s
    204: lambda: np.zeros(20, dtype=np.uint16),  # kTwentyUint16s
    205: lambda: np.zeros(24, dtype=np.uint16),  # kTwentyFourUint16s
    206: lambda: np.zeros(26, dtype=np.uint16),  # kTwentySixUint16s
    207: lambda: np.zeros(32, dtype=np.uint16),  # kThirtyTwoUint16s
    208: lambda: np.zeros(48, dtype=np.uint16),  # kFortyEightUint16s
    209: lambda: np.zeros(64, dtype=np.uint16),  # kSixtyFourUint16s
    210: lambda: np.zeros(96, dtype=np.uint16),  # kNinetySixUint16s
    211: lambda: np.zeros(122, dtype=np.uint16),  # kOneHundredTwentyTwoUint16s
    212: lambda: np.zeros(124, dtype=np.uint16),  # kOneHundredTwentyFourUint16s
    # int16_t extended (32 bytes to 248 bytes)
    213: lambda: np.zeros(16, dtype=np.int16),  # kSixteenInt16s
    214: lambda: np.zeros(20, dtype=np.int16),  # kTwentyInt16s
    215: lambda: np.zeros(24, dtype=np.int16),  # kTwentyFourInt16s
    216: lambda: np.zeros(26, dtype=np.int16),  # kTwentySixInt16s
    217: lambda: np.zeros(32, dtype=np.int16),  # kThirtyTwoInt16s
    218: lambda: np.zeros(48, dtype=np.int16),  # kFortyEightInt16s
    219: lambda: np.zeros(64, dtype=np.int16),  # kSixtyFourInt16s
    220: lambda: np.zeros(96, dtype=np.int16),  # kNinetySixInt16s
    221: lambda: np.zeros(122, dtype=np.int16),  # kOneHundredTwentyTwoInt16s
    222: lambda: np.zeros(124, dtype=np.int16),  # kOneHundredTwentyFourInt16s
    # uint32_t extended (64 bytes to 248 bytes)
    223: lambda: np.zeros(16, dtype=np.uint32),  # kSixteenUint32s
    224: lambda: np.zeros(20, dtype=np.uint32),  # kTwentyUint32s
    225: lambda: np.zeros(24, dtype=np.uint32),  # kTwentyFourUint32s
    226: lambda: np.zeros(32, dtype=np.uint32),  # kThirtyTwoUint32s
    227: lambda: np.zeros(48, dtype=np.uint32),  # kFortyEightUint32s
    228: lambda: np.zeros(62, dtype=np.uint32),  # kSixtyTwoUint32s
    # int32_t extended (64 bytes to 248 bytes)
    229: lambda: np.zeros(16, dtype=np.int32),  # kSixteenInt32s
    230: lambda: np.zeros(20, dtype=np.int32),  # kTwentyInt32s
    231: lambda: np.zeros(24, dtype=np.int32),  # kTwentyFourInt32s
    232: lambda: np.zeros(32, dtype=np.int32),  # kThirtyTwoInt32s
    233: lambda: np.zeros(48, dtype=np.int32),  # kFortyEightInt32s
    234: lambda: np.zeros(62, dtype=np.int32),  # kSixtyTwoInt32s
    # float extended (64 bytes to 248 bytes)
    235: lambda: np.zeros(16, dtype=np.float32),  # kSixteenFloat32s
    236: lambda: np.zeros(20, dtype=np.float32),  # kTwentyFloat32s
    237: lambda: np.zeros(24, dtype=np.float32),  # kTwentyFourFloat32s
    238: lambda: np.zeros(32, dtype=np.float32),  # kThirtyTwoFloat32s
    239: lambda: np.zeros(48, dtype=np.float32),  # kFortyEightFloat32s
    240: lambda: np.zeros(62, dtype=np.float32),  # kSixtyTwoFloat32s
    # uint64_t extended (128 bytes to 248 bytes)
    241: lambda: np.zeros(16, dtype=np.uint64),  # kSixteenUint64s
    242: lambda: np.zeros(20, dtype=np.uint64),  # kTwentyUint64s
    243: lambda: np.zeros(24, dtype=np.uint64),  # kTwentyFourUint64s
    244: lambda: np.zeros(31, dtype=np.uint64),  # kThirtyOneUint64s
    # int64_t extended (128 bytes to 248 bytes)
    245: lambda: np.zeros(16, dtype=np.int64),  # kSixteenInt64s
    246: lambda: np.zeros(20, dtype=np.int64),  # kTwentyInt64s
    247: lambda: np.zeros(24, dtype=np.int64),  # kTwentyFourInt64s
    248: lambda: np.zeros(31, dtype=np.int64),  # kThirtyOneInt64s
    # double extended (128 bytes to 248 bytes)
    249: lambda: np.zeros(16, dtype=np.float64),  # kSixteenFloat64s
    250: lambda: np.zeros(20, dtype=np.float64),  # kTwentyFloat64s
    251: lambda: np.zeros(24, dtype=np.float64),  # kTwentyFourFloat64s
    252: lambda: np.zeros(31, dtype=np.float64),  # kThirtyOneFloat64s
}
"""Maps prototype integer codes to factory callables that produce the corresponding numpy prototype objects."""

_PROTOTYPE_DTYPE_STRINGS: dict[int, str] = {
    code: str(factory().dtype) for code, factory in _PROTOTYPE_FACTORIES.items()
}
"""Maps prototype integer codes to their numpy dtype strings (e.g., ``'float32'``, ``'uint16'``). Built once at module
load by calling each factory and caching the dtype, avoiding per-message object allocation during log processing."""


class SerialProtocols(IntEnum):
    """Defines the protocol codes used to specify incoming and outgoing message layouts during PC-microcontroller
    communication.

    Notes:
        The elements in this enumeration should be accessed through their 'as_uint8' method to enforce
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


class SerialPrototypes(IntEnum):
    """Defines the prototype codes used during data transmission to specify the layout of additional data objects
    transmitted by KernelData and ModuleData messages.
    """

    # 1 byte total
    ONE_BOOL = 1
    """1 8-bit boolean."""
    ONE_UINT8 = 2
    """1 unsigned 8-bit integer."""
    ONE_INT8 = 3
    """1 signed 8-bit integer."""

    # 2 bytes total
    TWO_BOOLS = 4
    """An array of 2 8-bit booleans."""
    TWO_UINT8S = 5
    """An array of 2 unsigned 8-bit integers."""
    TWO_INT8S = 6
    """An array of 2 signed 8-bit integers."""
    ONE_UINT16 = 7
    """1 unsigned 16-bit integer."""
    ONE_INT16 = 8
    """1 signed 16-bit integer."""

    # 3 bytes total
    THREE_BOOLS = 9
    """An array of 3 8-bit booleans."""
    THREE_UINT8S = 10
    """An array of 3 unsigned 8-bit integers."""
    THREE_INT8S = 11
    """An array of 3 signed 8-bit integers."""

    # 4 bytes total
    FOUR_BOOLS = 12
    """An array of 4 8-bit booleans."""
    FOUR_UINT8S = 13
    """An array of 4 unsigned 8-bit integers."""
    FOUR_INT8S = 14
    """An array of 4 signed 8-bit integers."""
    TWO_UINT16S = 15
    """An array of 2 unsigned 16-bit integers."""
    TWO_INT16S = 16
    """An array of 2 signed 16-bit integers."""
    ONE_UINT32 = 17
    """1 unsigned 32-bit integer."""
    ONE_INT32 = 18
    """1 signed 32-bit integer."""
    ONE_FLOAT32 = 19
    """1 single-precision 32-bit floating-point number."""

    # 5 bytes total
    FIVE_BOOLS = 20
    """An array of 5 8-bit booleans."""
    FIVE_UINT8S = 21
    """An array of 5 unsigned 8-bit integers."""
    FIVE_INT8S = 22
    """An array of 5 signed 8-bit integers."""

    # 6 bytes total
    SIX_BOOLS = 23
    """An array of 6 8-bit booleans."""
    SIX_UINT8S = 24
    """An array of 6 unsigned 8-bit integers."""
    SIX_INT8S = 25
    """An array of 6 signed 8-bit integers."""
    THREE_UINT16S = 26
    """An array of 3 unsigned 16-bit integers."""
    THREE_INT16S = 27
    """An array of 3 signed 16-bit integers."""

    # 7 bytes total
    SEVEN_BOOLS = 28
    """An array of 7 8-bit booleans."""
    SEVEN_UINT8S = 29
    """An array of 7 unsigned 8-bit integers."""
    SEVEN_INT8S = 30
    """An array of 7 signed 8-bit integers."""

    # 8 bytes total
    EIGHT_BOOLS = 31
    """An array of 8 8-bit booleans."""
    EIGHT_UINT8S = 32
    """An array of 8 unsigned 8-bit integers."""
    EIGHT_INT8S = 33
    """An array of 8 signed 8-bit integers."""
    FOUR_UINT16S = 34
    """An array of 4 unsigned 16-bit integers."""
    FOUR_INT16S = 35
    """An array of 4 signed 16-bit integers."""
    TWO_UINT32S = 36
    """An array of 2 unsigned 32-bit integers."""
    TWO_INT32S = 37
    """An array of 2 signed 32-bit integers."""
    TWO_FLOAT32S = 38
    """An array of 2 single-precision 32-bit floating-point numbers."""
    ONE_UINT64 = 39
    """1 unsigned 64-bit integer."""
    ONE_INT64 = 40
    """1 signed 64-bit integer."""
    ONE_FLOAT64 = 41
    """1 double-precision 64-bit floating-point number."""

    # 9 bytes total
    NINE_BOOLS = 42
    """An array of 9 8-bit booleans."""
    NINE_UINT8S = 43
    """An array of 9 unsigned 8-bit integers."""
    NINE_INT8S = 44
    """An array of 9 signed 8-bit integers."""

    # 10 bytes total
    TEN_BOOLS = 45
    """An array of 10 8-bit booleans."""
    TEN_UINT8S = 46
    """An array of 10 unsigned 8-bit integers."""
    TEN_INT8S = 47
    """An array of 10 signed 8-bit integers."""
    FIVE_UINT16S = 48
    """An array of 5 unsigned 16-bit integers."""
    FIVE_INT16S = 49
    """An array of 5 signed 16-bit integers."""

    # 11 bytes total
    ELEVEN_BOOLS = 50
    """An array of 11 8-bit booleans."""
    ELEVEN_UINT8S = 51
    """An array of 11 unsigned 8-bit integers."""
    ELEVEN_INT8S = 52
    """An array of 11 signed 8-bit integers."""

    # 12 bytes total
    TWELVE_BOOLS = 53
    """An array of 12 8-bit booleans."""
    TWELVE_UINT8S = 54
    """An array of 12 unsigned 8-bit integers."""
    TWELVE_INT8S = 55
    """An array of 12 signed 8-bit integers."""
    SIX_UINT16S = 56
    """An array of 6 unsigned 16-bit integers."""
    SIX_INT16S = 57
    """An array of 6 signed 16-bit integers."""
    THREE_UINT32S = 58
    """An array of 3 unsigned 32-bit integers."""
    THREE_INT32S = 59
    """An array of 3 signed 32-bit integers."""
    THREE_FLOAT32S = 60
    """An array of 3 single-precision 32-bit floating-point numbers."""

    # 13 bytes total
    THIRTEEN_BOOLS = 61
    """An array of 13 8-bit booleans."""
    THIRTEEN_UINT8S = 62
    """An array of 13 unsigned 8-bit integers."""
    THIRTEEN_INT8S = 63
    """An array of 13 signed 8-bit integers."""

    # 14 bytes total
    FOURTEEN_BOOLS = 64
    """An array of 14 8-bit booleans."""
    FOURTEEN_UINT8S = 65
    """An array of 14 unsigned 8-bit integers."""
    FOURTEEN_INT8S = 66
    """An array of 14 signed 8-bit integers."""
    SEVEN_UINT16S = 67
    """An array of 7 unsigned 16-bit integers."""
    SEVEN_INT16S = 68
    """An array of 7 signed 16-bit integers."""

    # 15 bytes total
    FIFTEEN_BOOLS = 69
    """An array of 15 8-bit booleans."""
    FIFTEEN_UINT8S = 70
    """An array of 15 unsigned 8-bit integers."""
    FIFTEEN_INT8S = 71
    """An array of 15 signed 8-bit integers."""

    # 16 bytes total
    EIGHT_UINT16S = 72
    """An array of 8 unsigned 16-bit integers."""
    EIGHT_INT16S = 73
    """An array of 8 signed 16-bit integers."""
    FOUR_UINT32S = 74
    """An array of 4 unsigned 32-bit integers."""
    FOUR_INT32S = 75
    """An array of 4 signed 32-bit integers."""
    FOUR_FLOAT32S = 76
    """An array of 4 single-precision 32-bit floating-point numbers."""
    TWO_UINT64S = 77
    """An array of 2 unsigned 64-bit integers."""
    TWO_INT64S = 78
    """An array of 2 signed 64-bit integers."""
    TWO_FLOAT64S = 79
    """An array of 2 double-precision 64-bit floating-point numbers."""

    # 18 bytes total
    NINE_UINT16S = 80
    """An array of 9 unsigned 16-bit integers."""
    NINE_INT16S = 81
    """An array of 9 signed 16-bit integers."""

    # 20 bytes total
    TEN_UINT16S = 82
    """An array of 10 unsigned 16-bit integers."""
    TEN_INT16S = 83
    """An array of 10 signed 16-bit integers."""
    FIVE_UINT32S = 84
    """An array of 5 unsigned 32-bit integers."""
    FIVE_INT32S = 85
    """An array of 5 signed 32-bit integers."""
    FIVE_FLOAT32S = 86
    """An array of 5 single-precision 32-bit floating-point numbers."""

    # 22 bytes total
    ELEVEN_UINT16S = 87
    """An array of 11 unsigned 16-bit integers."""
    ELEVEN_INT16S = 88
    """An array of 11 signed 16-bit integers."""

    # 24 bytes total
    TWELVE_UINT16S = 89
    """An array of 12 unsigned 16-bit integers."""
    TWELVE_INT16S = 90
    """An array of 12 signed 16-bit integers."""
    SIX_UINT32S = 91
    """An array of 6 unsigned 32-bit integers."""
    SIX_INT32S = 92
    """An array of 6 signed 32-bit integers."""
    SIX_FLOAT32S = 93
    """An array of 6 single-precision 32-bit floating-point numbers."""
    THREE_UINT64S = 94
    """An array of 3 unsigned 64-bit integers."""
    THREE_INT64S = 95
    """An array of 3 signed 64-bit integers."""
    THREE_FLOAT64S = 96
    """An array of 3 double-precision 64-bit floating-point numbers."""

    # 26 bytes total
    THIRTEEN_UINT16S = 97
    """An array of 13 unsigned 16-bit integers."""
    THIRTEEN_INT16S = 98
    """An array of 13 signed 16-bit integers."""

    # 28 bytes total
    FOURTEEN_UINT16S = 99
    """An array of 14 unsigned 16-bit integers."""
    FOURTEEN_INT16S = 100
    """An array of 14 signed 16-bit integers."""
    SEVEN_UINT32S = 101
    """An array of 7 unsigned 32-bit integers."""
    SEVEN_INT32S = 102
    """An array of 7 signed 32-bit integers."""
    SEVEN_FLOAT32S = 103
    """An array of 7 single-precision 32-bit floating-point numbers."""

    # 30 bytes total
    FIFTEEN_UINT16S = 104
    """An array of 15 unsigned 16-bit integers."""
    FIFTEEN_INT16S = 105
    """An array of 15 signed 16-bit integers."""

    # 32 bytes total
    EIGHT_UINT32S = 106
    """An array of 8 unsigned 32-bit integers."""
    EIGHT_INT32S = 107
    """An array of 8 signed 32-bit integers."""
    EIGHT_FLOAT32S = 108
    """An array of 8 single-precision 32-bit floating-point numbers."""
    FOUR_UINT64S = 109
    """An array of 4 unsigned 64-bit integers."""
    FOUR_INT64S = 110
    """An array of 4 signed 64-bit integers."""
    FOUR_FLOAT64S = 111
    """An array of 4 double-precision 64-bit floating-point numbers."""

    # 36 bytes total
    NINE_UINT32S = 112
    """An array of 9 unsigned 32-bit integers."""
    NINE_INT32S = 113
    """An array of 9 signed 32-bit integers."""
    NINE_FLOAT32S = 114
    """An array of 9 single-precision 32-bit floating-point numbers."""

    # 40 bytes total
    TEN_UINT32S = 115
    """An array of 10 unsigned 32-bit integers."""
    TEN_INT32S = 116
    """An array of 10 signed 32-bit integers."""
    TEN_FLOAT32S = 117
    """An array of 10 single-precision 32-bit floating-point numbers."""
    FIVE_UINT64S = 118
    """An array of 5 unsigned 64-bit integers."""
    FIVE_INT64S = 119
    """An array of 5 signed 64-bit integers."""
    FIVE_FLOAT64S = 120
    """An array of 5 double-precision 64-bit floating-point numbers."""

    # 44 bytes total
    ELEVEN_UINT32S = 121
    """An array of 11 unsigned 32-bit integers."""
    ELEVEN_INT32S = 122
    """An array of 11 signed 32-bit integers."""
    ELEVEN_FLOAT32S = 123
    """An array of 11 single-precision 32-bit floating-point numbers."""

    # 48 bytes total
    TWELVE_UINT32S = 124
    """An array of 12 unsigned 32-bit integers."""
    TWELVE_INT32S = 125
    """An array of 12 signed 32-bit integers."""
    TWELVE_FLOAT32S = 126
    """An array of 12 single-precision 32-bit floating-point numbers."""
    SIX_UINT64S = 127
    """An array of 6 unsigned 64-bit integers."""
    SIX_INT64S = 128
    """An array of 6 signed 64-bit integers."""
    SIX_FLOAT64S = 129
    """An array of 6 double-precision 64-bit floating-point numbers."""

    # 52 bytes total
    THIRTEEN_UINT32S = 130
    """An array of 13 unsigned 32-bit integers."""
    THIRTEEN_INT32S = 131
    """An array of 13 signed 32-bit integers."""
    THIRTEEN_FLOAT32S = 132
    """An array of 13 single-precision 32-bit floating-point numbers."""

    # 56 bytes total
    FOURTEEN_UINT32S = 133
    """An array of 14 unsigned 32-bit integers."""
    FOURTEEN_INT32S = 134
    """An array of 14 signed 32-bit integers."""
    FOURTEEN_FLOAT32S = 135
    """An array of 14 single-precision 32-bit floating-point numbers."""
    SEVEN_UINT64S = 136
    """An array of 7 unsigned 64-bit integers."""
    SEVEN_INT64S = 137
    """An array of 7 signed 64-bit integers."""
    SEVEN_FLOAT64S = 138
    """An array of 7 double-precision 64-bit floating-point numbers."""

    # 60 bytes total
    FIFTEEN_UINT32S = 139
    """An array of 15 unsigned 32-bit integers."""
    FIFTEEN_INT32S = 140
    """An array of 15 signed 32-bit integers."""
    FIFTEEN_FLOAT32S = 141
    """An array of 15 single-precision 32-bit floating-point numbers."""

    # 64 bytes total
    EIGHT_UINT64S = 142
    """An array of 8 unsigned 64-bit integers."""
    EIGHT_INT64S = 143
    """An array of 8 signed 64-bit integers."""
    EIGHT_FLOAT64S = 144
    """An array of 8 double-precision 64-bit floating-point numbers."""

    # 72 bytes total
    NINE_UINT64S = 145
    """An array of 9 unsigned 64-bit integers."""
    NINE_INT64S = 146
    """An array of 9 signed 64-bit integers."""
    NINE_FLOAT64S = 147
    """An array of 9 double-precision 64-bit floating-point numbers."""

    # 80 bytes total
    TEN_UINT64S = 148
    """An array of 10 unsigned 64-bit integers."""
    TEN_INT64S = 149
    """An array of 10 signed 64-bit integers."""
    TEN_FLOAT64S = 150
    """An array of 10 double-precision 64-bit floating-point numbers."""

    # 88 bytes total
    ELEVEN_UINT64S = 151
    """An array of 11 unsigned 64-bit integers."""
    ELEVEN_INT64S = 152
    """An array of 11 signed 64-bit integers."""
    ELEVEN_FLOAT64S = 153
    """An array of 11 double-precision 64-bit floating-point numbers."""

    # 96 bytes total
    TWELVE_UINT64S = 154
    """An array of 12 unsigned 64-bit integers."""
    TWELVE_INT64S = 155
    """An array of 12 signed 64-bit integers."""
    TWELVE_FLOAT64S = 156
    """An array of 12 double-precision 64-bit floating-point numbers."""

    # 104 bytes total
    THIRTEEN_UINT64S = 157
    """An array of 13 unsigned 64-bit integers."""
    THIRTEEN_INT64S = 158
    """An array of 13 signed 64-bit integers."""
    THIRTEEN_FLOAT64S = 159
    """An array of 13 double-precision 64-bit floating-point numbers."""

    # 112 bytes total
    FOURTEEN_UINT64S = 160
    """An array of 14 unsigned 64-bit integers."""
    FOURTEEN_INT64S = 161
    """An array of 14 signed 64-bit integers."""
    FOURTEEN_FLOAT64S = 162
    """An array of 14 double-precision 64-bit floating-point numbers."""

    # 120 bytes total
    FIFTEEN_UINT64S = 163
    """An array of 15 unsigned 64-bit integers."""
    FIFTEEN_INT64S = 164
    """An array of 15 signed 64-bit integers."""
    FIFTEEN_FLOAT64S = 165
    """An array of 15 double-precision 64-bit floating-point numbers."""

    # Extended prototypes (codes 166-252)

    # bool extended (16 bytes to 248 bytes)
    SIXTEEN_BOOLS = 166
    """An array of 16 8-bit booleans."""
    TWENTY_FOUR_BOOLS = 167
    """An array of 24 8-bit booleans."""
    THIRTY_TWO_BOOLS = 168
    """An array of 32 8-bit booleans."""
    FORTY_BOOLS = 169
    """An array of 40 8-bit booleans."""
    FORTY_EIGHT_BOOLS = 170
    """An array of 48 8-bit booleans."""
    FIFTY_TWO_BOOLS = 171
    """An array of 52 8-bit booleans."""
    TWO_HUNDRED_FORTY_EIGHT_BOOLS = 172
    """An array of 248 8-bit booleans."""

    # uint8_t extended (16 bytes to 248 bytes)
    SIXTEEN_UINT8S = 173
    """An array of 16 unsigned 8-bit integers."""
    EIGHTEEN_UINT8S = 174
    """An array of 18 unsigned 8-bit integers."""
    TWENTY_UINT8S = 175
    """An array of 20 unsigned 8-bit integers."""
    TWENTY_TWO_UINT8S = 176
    """An array of 22 unsigned 8-bit integers."""
    TWENTY_FOUR_UINT8S = 177
    """An array of 24 unsigned 8-bit integers."""
    TWENTY_EIGHT_UINT8S = 178
    """An array of 28 unsigned 8-bit integers."""
    THIRTY_TWO_UINT8S = 179
    """An array of 32 unsigned 8-bit integers."""
    THIRTY_SIX_UINT8S = 180
    """An array of 36 unsigned 8-bit integers."""
    FORTY_UINT8S = 181
    """An array of 40 unsigned 8-bit integers."""
    FORTY_FOUR_UINT8S = 182
    """An array of 44 unsigned 8-bit integers."""
    FORTY_EIGHT_UINT8S = 183
    """An array of 48 unsigned 8-bit integers."""
    FIFTY_TWO_UINT8S = 184
    """An array of 52 unsigned 8-bit integers."""
    SIXTY_FOUR_UINT8S = 185
    """An array of 64 unsigned 8-bit integers."""
    NINETY_SIX_UINT8S = 186
    """An array of 96 unsigned 8-bit integers."""
    ONE_HUNDRED_TWENTY_EIGHT_UINT8S = 187
    """An array of 128 unsigned 8-bit integers."""
    ONE_HUNDRED_NINETY_TWO_UINT8S = 188
    """An array of 192 unsigned 8-bit integers."""
    TWO_HUNDRED_FORTY_FOUR_UINT8S = 189
    """An array of 244 unsigned 8-bit integers."""
    TWO_HUNDRED_FORTY_EIGHT_UINT8S = 190
    """An array of 248 unsigned 8-bit integers."""

    # int8_t extended (16 bytes to 248 bytes)
    SIXTEEN_INT8S = 191
    """An array of 16 signed 8-bit integers."""
    TWENTY_FOUR_INT8S = 192
    """An array of 24 signed 8-bit integers."""
    THIRTY_TWO_INT8S = 193
    """An array of 32 signed 8-bit integers."""
    FORTY_INT8S = 194
    """An array of 40 signed 8-bit integers."""
    FORTY_EIGHT_INT8S = 195
    """An array of 48 signed 8-bit integers."""
    FIFTY_TWO_INT8S = 196
    """An array of 52 signed 8-bit integers."""
    NINETY_TWO_INT8S = 197
    """An array of 92 signed 8-bit integers."""
    ONE_HUNDRED_THIRTY_TWO_INT8S = 198
    """An array of 132 signed 8-bit integers."""
    ONE_HUNDRED_SEVENTY_TWO_INT8S = 199
    """An array of 172 signed 8-bit integers."""
    TWO_HUNDRED_TWELVE_INT8S = 200
    """An array of 212 signed 8-bit integers."""
    TWO_HUNDRED_FORTY_FOUR_INT8S = 201
    """An array of 244 signed 8-bit integers."""
    TWO_HUNDRED_FORTY_EIGHT_INT8S = 202
    """An array of 248 signed 8-bit integers."""

    # uint16_t extended (32 bytes to 248 bytes)
    SIXTEEN_UINT16S = 203
    """An array of 16 unsigned 16-bit integers."""
    TWENTY_UINT16S = 204
    """An array of 20 unsigned 16-bit integers."""
    TWENTY_FOUR_UINT16S = 205
    """An array of 24 unsigned 16-bit integers."""
    TWENTY_SIX_UINT16S = 206
    """An array of 26 unsigned 16-bit integers."""
    THIRTY_TWO_UINT16S = 207
    """An array of 32 unsigned 16-bit integers."""
    FORTY_EIGHT_UINT16S = 208
    """An array of 48 unsigned 16-bit integers."""
    SIXTY_FOUR_UINT16S = 209
    """An array of 64 unsigned 16-bit integers."""
    NINETY_SIX_UINT16S = 210
    """An array of 96 unsigned 16-bit integers."""
    ONE_HUNDRED_TWENTY_TWO_UINT16S = 211
    """An array of 122 unsigned 16-bit integers."""
    ONE_HUNDRED_TWENTY_FOUR_UINT16S = 212
    """An array of 124 unsigned 16-bit integers."""

    # int16_t extended (32 bytes to 248 bytes)
    SIXTEEN_INT16S = 213
    """An array of 16 signed 16-bit integers."""
    TWENTY_INT16S = 214
    """An array of 20 signed 16-bit integers."""
    TWENTY_FOUR_INT16S = 215
    """An array of 24 signed 16-bit integers."""
    TWENTY_SIX_INT16S = 216
    """An array of 26 signed 16-bit integers."""
    THIRTY_TWO_INT16S = 217
    """An array of 32 signed 16-bit integers."""
    FORTY_EIGHT_INT16S = 218
    """An array of 48 signed 16-bit integers."""
    SIXTY_FOUR_INT16S = 219
    """An array of 64 signed 16-bit integers."""
    NINETY_SIX_INT16S = 220
    """An array of 96 signed 16-bit integers."""
    ONE_HUNDRED_TWENTY_TWO_INT16S = 221
    """An array of 122 signed 16-bit integers."""
    ONE_HUNDRED_TWENTY_FOUR_INT16S = 222
    """An array of 124 signed 16-bit integers."""

    # uint32_t extended (64 bytes to 248 bytes)
    SIXTEEN_UINT32S = 223
    """An array of 16 unsigned 32-bit integers."""
    TWENTY_UINT32S = 224
    """An array of 20 unsigned 32-bit integers."""
    TWENTY_FOUR_UINT32S = 225
    """An array of 24 unsigned 32-bit integers."""
    THIRTY_TWO_UINT32S = 226
    """An array of 32 unsigned 32-bit integers."""
    FORTY_EIGHT_UINT32S = 227
    """An array of 48 unsigned 32-bit integers."""
    SIXTY_TWO_UINT32S = 228
    """An array of 62 unsigned 32-bit integers."""

    # int32_t extended (64 bytes to 248 bytes)
    SIXTEEN_INT32S = 229
    """An array of 16 signed 32-bit integers."""
    TWENTY_INT32S = 230
    """An array of 20 signed 32-bit integers."""
    TWENTY_FOUR_INT32S = 231
    """An array of 24 signed 32-bit integers."""
    THIRTY_TWO_INT32S = 232
    """An array of 32 signed 32-bit integers."""
    FORTY_EIGHT_INT32S = 233
    """An array of 48 signed 32-bit integers."""
    SIXTY_TWO_INT32S = 234
    """An array of 62 signed 32-bit integers."""

    # float extended (64 bytes to 248 bytes)
    SIXTEEN_FLOAT32S = 235
    """An array of 16 single-precision 32-bit floating-point numbers."""
    TWENTY_FLOAT32S = 236
    """An array of 20 single-precision 32-bit floating-point numbers."""
    TWENTY_FOUR_FLOAT32S = 237
    """An array of 24 single-precision 32-bit floating-point numbers."""
    THIRTY_TWO_FLOAT32S = 238
    """An array of 32 single-precision 32-bit floating-point numbers."""
    FORTY_EIGHT_FLOAT32S = 239
    """An array of 48 single-precision 32-bit floating-point numbers."""
    SIXTY_TWO_FLOAT32S = 240
    """An array of 62 single-precision 32-bit floating-point numbers."""

    # uint64_t extended (128 bytes to 248 bytes)
    SIXTEEN_UINT64S = 241
    """An array of 16 unsigned 64-bit integers."""
    TWENTY_UINT64S = 242
    """An array of 20 unsigned 64-bit integers."""
    TWENTY_FOUR_UINT64S = 243
    """An array of 24 unsigned 64-bit integers."""
    THIRTY_ONE_UINT64S = 244
    """An array of 31 unsigned 64-bit integers."""

    # int64_t extended (128 bytes to 248 bytes)
    SIXTEEN_INT64S = 245
    """An array of 16 signed 64-bit integers."""
    TWENTY_INT64S = 246
    """An array of 20 signed 64-bit integers."""
    TWENTY_FOUR_INT64S = 247
    """An array of 24 signed 64-bit integers."""
    THIRTY_ONE_INT64S = 248
    """An array of 31 signed 64-bit integers."""

    # double extended (128 bytes to 248 bytes)
    SIXTEEN_FLOAT64S = 249
    """An array of 16 double-precision 64-bit floating-point numbers."""
    TWENTY_FLOAT64S = 250
    """An array of 20 double-precision 64-bit floating-point numbers."""
    TWENTY_FOUR_FLOAT64S = 251
    """An array of 24 double-precision 64-bit floating-point numbers."""
    THIRTY_ONE_FLOAT64S = 252
    """An array of 31 double-precision 64-bit floating-point numbers."""

    def as_uint8(self) -> np.uint8:
        """Returns the enumeration value as a numpy uint8 type."""
        return np.uint8(self.value)

    # noinspection PyTypeHints
    def get_prototype(self) -> PrototypeType:
        """Returns the prototype object associated with the prototype enumeration value."""
        return _PROTOTYPE_FACTORIES[self.value]()

    # noinspection PyTypeHints
    @staticmethod
    def get_prototype_for_code(code: np.uint8) -> PrototypeType | None:
        """Returns the prototype object associated with the input prototype code.

        Args:
            code: The prototype code for which to retrieve the prototype object. The code is converted to an integer
                before lookup.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type. If the input code is not one of
            the supported codes, returns None to indicate a matching error.
        """
        factory = _PROTOTYPE_FACTORIES.get(int(code))
        if factory is None:
            return None
        return factory()

    @staticmethod
    def get_dtype_for_code(code: int) -> str | None:
        """Returns the numpy dtype string associated with the input prototype code.

        Uses a pre-built lookup table to avoid instantiating a prototype object, making this suitable for hot paths
        where only the dtype string is needed (e.g., log processing serialization).

        Args:
            code: The prototype integer code for which to retrieve the dtype string.

        Returns:
            The numpy dtype string (e.g., ``'float32'``, ``'uint16'``), or None if the code is not recognized.
        """
        return _PROTOTYPE_DTYPE_STRINGS.get(code)
