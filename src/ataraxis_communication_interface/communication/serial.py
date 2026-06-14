"""Provides the SerialCommunication class that handles bidirectional serial communication with a microcontroller running
the ataraxis-micro-controller library over the USB or UART interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions, TimestampFormats, get_timestamp
from ataraxis_base_utilities import console
from ataraxis_data_structures import LogPackage
from ataraxis_transport_layer_pc import TransportLayer

from .messages import (
    KernelData,
    ModuleData,
    KernelState,
    ModuleState,
    KernelCommand,
    ReceptionCode,
    ModuleParameters,
    OneOffModuleCommand,
    DequeueModuleCommand,
    ModuleIdentification,
    RepeatedModuleCommand,
    ControllerIdentification,
)
from .protocols import SerialProtocols, SerialPrototypes

if TYPE_CHECKING:
    from multiprocessing import Queue as MPQueue

    from numpy.typing import NDArray

_PROTOCOL_MODULE_DATA: np.uint8 = SerialProtocols.MODULE_DATA.as_uint8()
"""Cached uint8 value for the MODULE_DATA protocol code, used in receive_message dispatch."""

_PROTOCOL_KERNEL_DATA: np.uint8 = SerialProtocols.KERNEL_DATA.as_uint8()
"""Cached uint8 value for the KERNEL_DATA protocol code, used in receive_message dispatch."""

_PROTOCOL_MODULE_STATE: np.uint8 = SerialProtocols.MODULE_STATE.as_uint8()
"""Cached uint8 value for the MODULE_STATE protocol code, used in receive_message dispatch."""

_PROTOCOL_KERNEL_STATE: np.uint8 = SerialProtocols.KERNEL_STATE.as_uint8()
"""Cached uint8 value for the KERNEL_STATE protocol code, used in receive_message dispatch."""

_PROTOCOL_RECEPTION_CODE: np.uint8 = SerialProtocols.RECEPTION_CODE.as_uint8()
"""Cached uint8 value for the RECEPTION_CODE protocol code, used in receive_message dispatch."""

_PROTOCOL_CONTROLLER_IDENTIFICATION: np.uint8 = SerialProtocols.CONTROLLER_IDENTIFICATION.as_uint8()
"""Cached uint8 value for the CONTROLLER_IDENTIFICATION protocol code, used in receive_message dispatch."""

_PROTOCOL_MODULE_IDENTIFICATION: np.uint8 = SerialProtocols.MODULE_IDENTIFICATION.as_uint8()
"""Cached uint8 value for the MODULE_IDENTIFICATION protocol code, used in receive_message dispatch."""


class SerialCommunication:
    """Provides methods for bidirectionally communicating with a microcontroller running the ataraxis-micro-controller
    library over the USB or UART serial interface.

    Notes:
        This class is explicitly designed to be used by other library assets and should not be used directly by end
        users. An instance of this class is initialized and managed by the MicroControllerInterface class.

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
        _usb_port: Stores the name of the serial port (USB or UART) used for communication.
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
        """Returns a string representation of the SerialCommunication instance."""
        return f"SerialCommunication(usb_port={self._usb_port}, controller_id={self._source_id})"

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

        # Logs the transmitted data to disk.
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
            A reference to the parsed message data stored as an instance's attribute, which is one of ModuleData,
            ModuleState, KernelData, KernelState, ControllerIdentification, ModuleIdentification, or ReceptionCode, or
            None if no message was received.

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
        if protocol == _PROTOCOL_MODULE_DATA:
            # Parses the static header data from the extracted message
            self._module_data.message = self._transport_layer.read_data(data_object=self._module_data.message)

            # Resolves the prototype code and uses it to retrieve the prototype object from the prototypes dataclass
            # instance
            prototype = SerialPrototypes.get_prototype_for_code(code=self._module_data.prototype_code)

            # If prototype retrieval fails, raises ValueError
            if prototype is None:
                message = (
                    f"Invalid prototype code {self._module_data.prototype_code} encountered when extracting the data "
                    f"object from the received ModuleData message sent by module {self._module_data.module_id} of type "
                    f"{self._module_data.module_type}. All messages must use one of the valid prototype "
                    f"codes available from the SerialPrototypes enumeration."
                )
                console.error(message=message, error=ValueError)

            # Uses the retrieved prototype to parse the data object.
            self._module_data.data_object = self._transport_layer.read_data(data_object=prototype)

            return self._module_data

        if protocol == _PROTOCOL_KERNEL_DATA:
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
                console.error(message=message, error=ValueError)

            # Uses the retrieved prototype to parse the data object.
            self._kernel_data.data_object = self._transport_layer.read_data(data_object=prototype)

            return self._kernel_data

        if protocol == _PROTOCOL_MODULE_STATE:
            self._module_state.message = self._transport_layer.read_data(data_object=self._module_state.message)
            return self._module_state

        if protocol == _PROTOCOL_KERNEL_STATE:
            self._kernel_state.message = self._transport_layer.read_data(data_object=self._kernel_state.message)
            return self._kernel_state

        if protocol == _PROTOCOL_RECEPTION_CODE:
            self._reception_code.message = self._transport_layer.read_data(data_object=self._reception_code.message)
            return self._reception_code

        if protocol == _PROTOCOL_CONTROLLER_IDENTIFICATION:
            self._controller_identification.message = self._transport_layer.read_data(
                data_object=self._controller_identification.message
            )
            return self._controller_identification

        if protocol == _PROTOCOL_MODULE_IDENTIFICATION:
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
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        # noinspection PyUnreachableCode
        raise ValueError(message)  # pragma: no cover

    def _log_data(self, timestamp: int, data: NDArray[np.uint8]) -> None:
        """Packages and sends the input data to the DataLogger instance that writes it to disk.

        Args:
            timestamp: The value of the timestamp timer's 'elapsed' property that communicates the number of elapsed
                microseconds relative to the 'onset' timestamp at the time of data acquisition.
            data: The serialized message payload to be logged.
        """
        # Packages the data to be logged into the appropriate tuple format (with ID variables).
        package = LogPackage(source_id=self._source_id, acquisition_time=np.uint64(timestamp), serialized_data=data)

        # Sends the data to the logger.
        self._logger_queue.put(package)
