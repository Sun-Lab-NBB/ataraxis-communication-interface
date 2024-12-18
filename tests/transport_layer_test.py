"""This file contains the test functions that verify the functionality and error-handling of all
SerialTransportLayer class methods. Special care is taken to fully test the 4 major methods: write_data(),
read_data(), send_data(), and receive_data(). You can also use this file if you need more examples on how to use
class methods.
"""

from typing import Any
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from ataraxis_base_utilities import error_format

from ataraxis_transport_layer.transport_layer import SerialTransportLayer


@dataclass
class SampleDataClass:
    """A simple dataclass used to test 'structure' serialization capability of the SerialTransportLayer class. Has
     to use numpy arrays and scalars as field types to support serialization.

    Attributes:
        uint_value: Any numpy unsigned integer scalar value. Used to test the ability to serialize scalar dataclass
            fields.
        uint_array: Any numpy array value. Used to test the ability to serialize numpy array dataclass fields.
    """

    uint_value: np.unsignedinteger
    uint_array: np.ndarray


@pytest.fixture()
def protocol() -> SerialTransportLayer:
    """Returns a SerialTransportLayer instance with test mode enabled.

    This asset is used to streamline STL initialization for testing purposes.
    """
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        test_mode=True,
    )

    return protocol


def test_init_and_repr(protocol) -> None:
    """Verifies the functionality of SerialTransportLayer __repr__ method.

    Also, indirectly verifies the __init__() method through the use of protocol fixture.
    """
    representation_string = (
        f"SerialTransportLayer(port & baudrate=MOCKED, polynomial={protocol._crc_processor.polynomial}, "
        f"start_byte={protocol._start_byte}, delimiter_byte={protocol._delimiter_byte}, timeout={protocol._timeout} "
        f"us, maximum_tx_payload_size = {protocol._max_tx_payload_size}, "
        f"maximum_rx_payload_size={protocol._max_rx_payload_size})"
    )
    assert repr(protocol) == representation_string


def test_init_errors() -> None:
    """Verifies the error handling of the SerialTransportLayer __init__() method.

    Avoids checking arguments that are given to helper modules, such as the input polynomial. Assumes that helper
    modules have been tested before testing the SerialTransportLayer class and are known to properly handle invalid
    initialization arguments.
    """
    # Invalid port argument
    port = None
    message = (
        f"Unable to initialize SerialTransportLayer class. Expected a string value for 'port' argument, but "
        f"encountered {port} of type {type(port).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        SerialTransportLayer(
            port=port,
        )

    # Invalid baudrate argument
    baudrate = -9600
    message = (
        f"Unable to initialize SerialTransportLayer class. Expected a positive integer value for 'baudrate' "
        f"argument, but encountered {baudrate} of type {type(baudrate).__name__}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        SerialTransportLayer(port="COM7", baudrate=baudrate)

    # Invalid start_byte argument
    start_byte = 300
    message = (
        f"Unable to initialize SerialTransportLayer class. Expected an integer value between 0 and 255 for "
        f"'start_byte' argument, but encountered {start_byte} of type {type(start_byte).__name__}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        SerialTransportLayer(port="COM7", start_byte=start_byte)

    # Invalid delimiter_byte argument
    delimiter_byte = 300
    message = (
        f"Unable to initialize SerialTransportLayer class. Expected an integer value between 0 and 255 for "
        f"'delimiter_byte' argument, but encountered {delimiter_byte} of type {type(delimiter_byte).__name__}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        SerialTransportLayer(port="COM7", delimiter_byte=delimiter_byte)

    # Invalid timeout argument
    timeout = -5000
    message = (
        f"Unable to initialize SerialTransportLayer class. Expected an integer value of 0 or above for "
        f"'timeout' argument, but encountered {timeout} of type {type(timeout).__name__}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        SerialTransportLayer(port="COM7", timeout=timeout)

    # Delimiter and Start byte are the same error
    message = (
        "Unable to initialize SerialTransportLayer class. The 'start_byte' and 'delimiter_byte' cannot be the same."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        SerialTransportLayer(port="COM7", start_byte=129, delimiter_byte=129)


@pytest.mark.parametrize(
    "data, expected_buffer",
    [
        # Case 1: Unsigned scalars
        ((np.uint8(10), np.uint16(451), np.uint32(123456)), np.array([10, 195, 1, 64, 226, 1, 0], dtype=np.uint8)),
        # Case 2: Signed scalars
        (
            (np.int8(-10), np.int16(-451), np.int32(-123456)),
            np.array([246, 61, 254, 192, 29, 254, 255], dtype=np.uint8),
        ),
        # Case 3: Float scalar
        ((np.float32(312.142),), np.array([45, 18, 156, 67], dtype=np.uint8)),
        # Case 4: Boolean scalar
        ((np.bool_(True),), np.array([1], dtype=np.uint8)),
        # Case 5: 64-bit arrays
        (
            (
                np.array([1, 2, 3, 4, 5], dtype=np.uint64),
                np.array([-1, -2, -3, -4, -5], dtype=np.int64),
                np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
            ),
            np.array(
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,  # First array (uint64)
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,  # Second array (int64)
                    254,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    253,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    252,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    251,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    255,
                    154,
                    153,
                    153,
                    153,
                    153,
                    153,
                    241,
                    63,  # Third array (float64)
                    154,
                    153,
                    153,
                    153,
                    153,
                    153,
                    1,
                    64,
                    102,
                    102,
                    102,
                    102,
                    102,
                    102,
                    10,
                    64,
                    154,
                    153,
                    153,
                    153,
                    153,
                    153,
                    17,
                    64,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    22,
                    64,
                ],
                dtype=np.uint8,
            ),
        ),
        # Case 6: Sample Data Class
        (
            (SampleDataClass(uint_value=np.uint8(50), uint_array=np.array([1, 2, 3], np.uint8)),),
            np.array([50, 1, 2, 3], dtype=np.uint8),
        ),
    ],
)
def test_data_transmission_cycle(protocol, data: tuple[Any, ...], expected_buffer: NDArray[Any]) -> None:
    """Verifies the functioning of SerialTransportLayer write_data(), send_data(), receive_data() and read_data()
    methods.

    This test suite cycles the data through the four major methods used by the class to carry out bidirectional serial
    communication. Additionally, it tests all supported data types using mark parametrization to efficiently allocate
    test cases to different cores.

    Args:
        protocol: The SerialTransportLayer instance to test.
        data: Tuple containing test data of various types.
        expected_buffer: Expected buffer state after writing the data
    """

    # Step 1: Writes all data items to the transmission buffer
    current_index = 0
    for item in data:
        end_index = protocol.write_data(item, start_index=current_index)
        current_index = end_index

    # Verifies buffer state after writing the data
    assert np.array_equal(protocol.transmission_buffer[:current_index], expected_buffer)
    assert protocol.bytes_in_transmission_buffer == len(expected_buffer)

    # Step 2: Sends the data to the SerialMock's transmission buffer
    previous_buffer = protocol.transmission_buffer  # Saves the current transmission buffer state for Step 3
    assert protocol.send_data()
    assert protocol.bytes_in_transmission_buffer == 0  # Buffer should be reset after sending

    # Step 3: Simulates reception via copying the data inside the Serial Mock (tx -> rx)
    assert not protocol.available  # No data should be available until the assignment operation below.
    protocol._port.rx_buffer = protocol._port.tx_buffer
    assert protocol.available  # Should be True since data is now available for reception
    assert protocol.receive_data()
    assert protocol.bytes_in_reception_buffer == len(expected_buffer)
    # Verifies that received serialized payload matches the transmitted one
    assert np.array_equal(previous_buffer[:current_index], protocol.reception_buffer[:current_index])

    # Step 4: Reads and verifies each data item
    current_index = 0
    for item in data:
        # Creates the appropriate zero-initialized prototype for reading
        if isinstance(item, np.ndarray):
            prototype = np.zeros_like(item)
        elif isinstance(item, SampleDataClass):
            prototype = SampleDataClass(uint_value=np.uint8(0), uint_array=np.zeros_like(item.uint_array))
        else:
            prototype = type(item)(0)

        # Reads the data
        received_item, end_index = protocol.read_data(prototype, start_index=current_index)
        current_index = end_index

        # Verifies the received data matches the original
        if isinstance(item, np.ndarray):
            assert np.array_equal(received_item, item)
        elif isinstance(item, SampleDataClass):
            assert received_item.uint_value == item.uint_value
            assert np.array_equal(received_item.uint_array, item.uint_array)
        else:
            assert received_item == item

    # Clean up
    protocol.reset_transmission_buffer()
    protocol.reset_reception_buffer()
    protocol._port.tx_buffer = b""
    protocol._port.rx_buffer = b""


def test_receive_bytes_available(protocol) -> None:
    """Verifies the functionality of the SerialTransportLayer _bytes_available() private method not tested by other
    test cases."""

    # _bytes_available() is designed to receive data broken into chunks. This functionality is hard to test indirectly
    # without using two instances of TransportLayer class. Instead, this method directly verifies that functionality
    # by simulating receiving the data in chunks.

    # Reuses the payload from the test_receive_data_errors()
    test_payload: NDArray[np.uint8] = np.array([1, 2, 3, 4, 0, 0, 7, 8, 9, 10], dtype=np.uint8)
    preamble = np.array([129, 10], dtype=np.uint8)
    packet = protocol._cobs_processor.encode_payload(payload=test_payload, delimiter=np.uint8(0))
    checksum = protocol._crc_processor.calculate_crc_checksum(packet)
    checksum = protocol._crc_processor.convert_checksum_to_bytes(checksum)
    test_data = np.concatenate((preamble, packet, checksum), dtype=np.uint8, axis=0)

    # Breaks the packet into 2 chunks
    chunk_1 = test_data[:8]
    chunk_2 = test_data[8:16]

    # Verifies that TransportLayer correctly combines data 'leftover' from previous data reception with new data that
    # became available before the most recent read_data call().
    protocol._leftover_bytes = chunk_1.tobytes()
    protocol._port.rx_buffer = chunk_2.tobytes()
    protocol.receive_data()

    # Verifies that TransportLayer can receive the data entirely from 'leftover' bytes.
    protocol._leftover_bytes = test_data.tobytes()
    protocol.receive_data()


def test_read_data_errors(protocol) -> None:
    """Verifies the error handling behavior of SerialTransportLayer read_data() method"""

    # Sets the received bytes tracker to 5. The instance interprets this as meaning that it has 5 bytes available for
    # reading inside the reception buffer. This is necessary to trigger the error cases below.
    protocol._bytes_in_reception_buffer = 5

    # Unsupported prototype
    unsupported_data_object = "unsupported_type"
    message = (
        f"Failed to read the data from the reception buffer. Encountered an unsupported input data_object "
        f"type ({type(unsupported_data_object).__name__}). At this time, only the following numpy scalar or array "
        f"types are supported: {protocol._accepted_numpy_scalars}. Alternatively, a dataclass with all attributes "
        f"set to supported numpy scalar or array types is also supported."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.read_data(data_object=unsupported_data_object)

    # Empty NdArray prototype
    empty_array = np.empty(0, dtype=np.uint8)
    message = (
        "Failed to read the data from the reception buffer. Encountered an empty (size 0) numpy array as "
        "input data_object. Reading empty arrays is not supported."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.read_data(empty_array)

    # Multidimensional NdArray input.
    multidimensional_array = np.empty([2, 2], dtype=np.uint8)
    message = (
        f"Failed to read the data from the reception buffer. Encountered a multidimensional numpy array with "
        f"{multidimensional_array.ndim} dimensions as input data_object. At this time, only "
        f"one-dimensional (flat) arrays are supported."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.read_data(multidimensional_array)

    # Prototype needs more data than available for reading
    large_array = np.empty(shape=300, dtype=np.uint8)
    message = (
        f"Failed to read the data from the reception buffer. The reception buffer does not have enough "
        f"bytes available to fully fill the object starting at the index {0}. Specifically, given "
        f"the object size of {large_array.nbytes} bytes, the required payload size is {0 + large_array.nbytes} bytes, "
        f"but the available size is {protocol.bytes_in_reception_buffer} bytes."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.read_data(large_array)


def test_write_data_errors(protocol) -> None:
    """Verifies the error handling behavior of SerialTransportLayer write_data() method"""

    # Invalid data type
    invalid_data = None
    message = (
        f"Failed to write the data to the transmission buffer. Encountered an unsupported input data_object "
        f"type ({type(invalid_data).__name__}). At this time, only the following numpy scalar or array "
        f"types are supported: {protocol._accepted_numpy_scalars}. Alternatively, a dataclass with all attributes "
        f"set to supported numpy scalar or array types is also supported."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.write_data(invalid_data)

    # Empty NdArray input. Also tests encountering an error when serializing a data-class instance by assigning an
    # empty array to a data-class attribute.
    message = (
        "Failed to write the data to the transmission buffer. Encountered an empty (size 0) numpy array as input "
        "data_object. Writing empty arrays is not supported."
    )
    empty_array: NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
    test_dataclass = SampleDataClass(uint_array=empty_array, uint_value=np.uint8(5))
    with pytest.raises(
        ValueError,
        match=error_format(message),
    ):
        # noinspection PyTypeChecker
        protocol.write_data(test_dataclass)

    # Multidimensional NdArray input.
    message = (
        "Failed to write the data to the transmission buffer. Encountered a multidimensional numpy array with 2 "
        "dimensions as input data_object. At this time, only one-dimensional (flat) arrays are supported."
    )
    invalid_array: np.ndarray = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.write_data(invalid_array)

    # An object whose size exceeds the available transmission buffer space.
    large_data = np.empty(300, dtype=np.uint8)
    # 1 below is due to a partial success of the dataclass writing (uint8 scalar is written). During real-world usage,
    # the class would raise an exception and terminate the runtime, so this discrepancy is not an issue.
    message = (
        f"Failed to write the data to the transmission buffer. The transmission buffer does not have enough "
        f"space to write the data starting at the index {1}. Specifically, given the data size of "
        f"{large_data.nbytes} bytes, the required buffer size is {1+large_data.nbytes} bytes, but the available "
        f"size is {protocol._transmission_buffer.size} bytes."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.write_data(large_data)


def test_receive_data_errors(protocol):
    """Verifies the error handling behavior of SerialTransportLayer class receive_data() method."""

    # Generates a test payload and uses SerialTransportLayer internal methods to encode, checksum and assemble the
    # data packet around the payload. This simulates the steps typically taken as part of the send_data() method
    # runtime.
    test_payload: NDArray[np.uint8] = np.array([1, 2, 3, 4, 0, 0, 7, 8, 9, 10], dtype=np.uint8)
    preamble = np.array([129, 10], dtype=np.uint8)
    packet = protocol._cobs_processor.encode_payload(payload=test_payload, delimiter=np.uint8(0))
    checksum = protocol._crc_processor.calculate_crc_checksum(packet)
    checksum = protocol._crc_processor.convert_checksum_to_bytes(checksum)
    test_data = np.concatenate((preamble, packet, checksum), axis=0)

    # Also generates a buffer that does not have a start byte to test errors associated with handling communication
    # line noise
    empty_buffer = np.random.randint(low=0, high=128, dtype=np.uint8, size=20)  # Will never be 129 (START byte value)

    # CODE 101. A buffer without a start byte is interpreted as a noise-filled buffer. Since start-byte-associated
    # errors are disabled, the receive_data() method should return False, but should not raise an error.
    protocol._port.rx_buffer = empty_buffer.tobytes()
    assert not protocol.receive_data()

    # CODE 102. Receiving a message with start byte errors turned on and without start byte raises a RuntimeError
    protocol._allow_start_byte_errors = True
    protocol._port.rx_buffer = empty_buffer.tobytes()  # Since rx_buffer is consumed each call, it needs to be reset
    message = (
        "Failed to parse the incoming serial packet data. Unable to find the start_byte "
        "(129) value among the bytes stored inside the serial buffer."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol.receive_data()

    # CODE 0. Packet size byte not received in time.
    empty_buffer[-1] = 129  # Sets the last byte of the empty_buffer to stat byte value.
    protocol._port.rx_buffer = empty_buffer.tobytes()
    message = (
        f"Failed to parse the size of the incoming serial packet. The packet size byte was not received in "
        f"time ({protocol._timeout} microseconds), following the reception of the START byte."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol.receive_data()

    # Cleans up and resets the test buffer
    protocol._leftover_bytes = bytes()  # Clears leftover bytes to prevent it from accumulating unprocessed bytes.
    empty_buffer[-1] = 129

    # CODE 2. Packet reception stalls while waiting for additional payload bytes.
    test_data[1] = 110  # Sets packet size to a number that exceeds the number of available bytes
    test_data[13] = 1  # Replaces the original delimiter byte to avoid Delimiter Byte Found Too Early error
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        "Failed to parse the incoming serial packet data. The byte number 15 out of 114 "
        "was not received in time (20000 microseconds), following the reception of the previous byte. "
        "Packet reception staled."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol.receive_data()

    # Cleans up and resets the test buffer
    protocol._leftover_bytes = bytes()
    # Does not reset the packet size, as the test below also modifies this value
    test_data[13] = 0

    # CODE 103. The received message contains an invalid payload_size value (second value of the packet)
    test_data[1] = 255  # Replaces the packet size with an invalid value
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to parse the incoming serial packet data. The parsed size of the COBS-encoded payload "
        f"(255), is outside the expected boundaries ({protocol._min_rx_payload_size} to "
        f"{protocol._max_rx_payload_size}). This likely indicates a mismatch in the transmission parameters between "
        f"this system and the Microcontroller."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol.receive_data()

    # Cleans up and resets the test buffer
    protocol._leftover_bytes = bytes()
    test_data[1] = 10

    # CODE 104. Delimiter byte value found before reaching the end of the encoded packet.
    test_data[-4] = 0  # Inserts the delimiter 1 position before the actual delimiter position
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to parse the incoming serial packet data. Delimiter byte value ({protocol._delimiter_byte}) "
        f"encountered at payload byte number {11}, instead of the expected byte number "
        f"{12}. This likely indicates packet corruption or "
        f"mismatch in the transmission parameters between this system and the Microcontroller."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol.receive_data()

    # Cleans up and resets the test buffer
    protocol._leftover_bytes = bytes()
    test_data[-4] = 10  # This was the initial value at index -4

    # CODE 105. Delimiter byte not found at the end of the encoded packet.
    test_data[-3] = 10  # Overrides the delimiter
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to parse the incoming serial packet data. Delimiter byte value ({protocol._delimiter_byte}) "
        f"expected as the last encoded packet byte ({12}), but instead encountered {10}. This likely indicates packet "
        f"corruption or mismatch in the transmission parameters between this system and the Microcontroller."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol.receive_data()

    # Cleans up and resets the test buffer
    protocol._leftover_bytes = bytes()
    test_data[-3] = 0  # Restores the delimiter

    # CRC Checksum verification error.
    # Translates the real and invalid checksums into hexadecimals used in error messages
    expected_checksum = hex(protocol._crc_processor.convert_bytes_to_checksum(test_data[-2:].copy()))  # to Hexadecimal
    received_checksum = hex(protocol._crc_processor.convert_bytes_to_checksum(np.array([0x00, 0x00], dtype=np.uint8)))

    # Replaces the checksum in the test_data packet with an invalid checksum
    test_data[-2:] = np.array([0x00, 0x00], dtype=np.uint8)  # Fake checksum
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to verify the received serial packet's integrity. The checksum value transmitted with the packet "
        f"{received_checksum} did not match the expected value based on the packet data {expected_checksum}. This "
        f"indicates the packet was corrupted during transmission or reception."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        protocol.receive_data()

    # Cleans up and resets the test buffer
    protocol._leftover_bytes = bytes()

    # COBS verification error.
    # For this test, creates a special test payload by introducing an error after COBS-encoding the payload, but
    # before generating the CRC checksum. This simulates for a very rare case where the packet corruption is so major
    # the CRC fails to detect the corruption. However, the corruption can break COBS-encoding, which COBS will detect.
    packet = protocol._cobs_processor.encode_payload(payload=test_payload, delimiter=np.uint8(0))
    packet[5] = 2  # Replaces one of the COBS_encoded values with a different value, introducing a COBS error
    checksum = protocol._crc_processor.calculate_crc_checksum(packet)
    checksum = protocol._crc_processor.convert_checksum_to_bytes(checksum)
    test_data = np.concatenate((preamble, packet, checksum), axis=0)

    # Checks the COBS error
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to decode payload using COBS scheme. The decoder did not find the unencoded delimiter "
        f"at the end of the packet. This is either because the end-value is not an unencoded delimiter or "
        f"because the decoding does not end at the final index of the packet. Packet is likely "
        f"corrupted. CODE: {18}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        protocol.receive_data()


def test_send_data_errors(protocol):
    """Verifies the error handling behavior of SerialTransportLayer class send_data() method."""

    # Tests calling send_data() with an empty transmission_buffer.
    message = (
        f"Failed to encode the payload using COBS scheme. The size of the input payload "
        f"({0}) is too small. A minimum size of 1 elements (bytes) is required. CODE: 12."
    )
    with pytest.raises(
        ValueError,
        match=error_format(message),
    ):
        protocol.send_data()
