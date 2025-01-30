from enum import IntEnum
from typing import Any
from dataclasses import dataclass
from collections.abc import Callable as Callable
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray
import paho.mqtt.client as mqtt
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

    UNDEFINED: int
    REPEATED_MODULE_COMMAND: int
    ONE_OFF_MODULE_COMMAND: int
    DEQUEUE_MODULE_COMMAND: int
    KERNEL_COMMAND: int
    MODULE_PARAMETERS: int
    KERNEL_PARAMETERS: int
    MODULE_DATA: int
    KERNEL_DATA: int
    MODULE_STATE: int
    KERNEL_STATE: int
    RECEPTION_CODE: int
    CONTROLLER_IDENTIFICATION: int
    MODULE_IDENTIFICATION: int
    def as_uint8(self) -> np.uint8:
        """Convert the enum value to numpy.uint8 type.

        Returns:
            np.uint8: The enum value as a numpy unsigned 8-bit integer.
        """

NumericType: Incomplete
PrototypeType: Incomplete
_PROTOTYPE_FACTORIES: dict[int, Callable[[], PrototypeType]]

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

    ONE_BOOL: int
    ONE_UINT8: int
    ONE_INT8: int
    TWO_BOOLS: int
    TWO_UINT8S: int
    TWO_INT8S: int
    ONE_UINT16: int
    ONE_INT16: int
    THREE_BOOLS: int
    THREE_UINT8S: int
    THREE_INT8S: int
    FOUR_BOOLS: int
    FOUR_UINT8S: int
    FOUR_INT8S: int
    TWO_UINT16S: int
    TWO_INT16S: int
    ONE_UINT32: int
    ONE_INT32: int
    ONE_FLOAT32: int
    FIVE_BOOLS: int
    FIVE_UINT8S: int
    FIVE_INT8S: int
    SIX_BOOLS: int
    SIX_UINT8S: int
    SIX_INT8S: int
    THREE_UINT16S: int
    THREE_INT16S: int
    SEVEN_BOOLS: int
    SEVEN_UINT8S: int
    SEVEN_INT8S: int
    EIGHT_BOOLS: int
    EIGHT_UINT8S: int
    EIGHT_INT8S: int
    FOUR_UINT16S: int
    FOUR_INT16S: int
    TWO_UINT32S: int
    TWO_INT32S: int
    TWO_FLOAT32S: int
    ONE_UINT64: int
    ONE_INT64: int
    ONE_FLOAT64: int
    NINE_BOOLS: int
    NINE_UINT8S: int
    NINE_INT8S: int
    TEN_BOOLS: int
    TEN_UINT8S: int
    TEN_INT8S: int
    FIVE_UINT16S: int
    FIVE_INT16S: int
    ELEVEN_BOOLS: int
    ELEVEN_UINT8S: int
    ELEVEN_INT8S: int
    TWELVE_BOOLS: int
    TWELVE_UINT8S: int
    TWELVE_INT8S: int
    SIX_UINT16S: int
    SIX_INT16S: int
    THREE_UINT32S: int
    THREE_INT32S: int
    THREE_FLOAT32S: int
    THIRTEEN_BOOLS: int
    THIRTEEN_UINT8S: int
    THIRTEEN_INT8S: int
    FOURTEEN_BOOLS: int
    FOURTEEN_UINT8S: int
    FOURTEEN_INT8S: int
    SEVEN_UINT16S: int
    SEVEN_INT16S: int
    FIFTEEN_BOOLS: int
    FIFTEEN_UINT8S: int
    FIFTEEN_INT8S: int
    EIGHT_UINT16S: int
    EIGHT_INT16S: int
    FOUR_UINT32S: int
    FOUR_INT32S: int
    FOUR_FLOAT32S: int
    TWO_UINT64S: int
    TWO_INT64S: int
    TWO_FLOAT64S: int
    NINE_UINT16S: int
    NINE_INT16S: int
    TEN_UINT16S: int
    TEN_INT16S: int
    FIVE_UINT32S: int
    FIVE_INT32S: int
    FIVE_FLOAT32S: int
    ELEVEN_UINT16S: int
    ELEVEN_INT16S: int
    TWELVE_UINT16S: int
    TWELVE_INT16S: int
    SIX_UINT32S: int
    SIX_INT32S: int
    SIX_FLOAT32S: int
    THREE_UINT64S: int
    THREE_INT64S: int
    THREE_FLOAT64S: int
    THIRTEEN_UINT16S: int
    THIRTEEN_INT16S: int
    FOURTEEN_UINT16S: int
    FOURTEEN_INT16S: int
    SEVEN_UINT32S: int
    SEVEN_INT32S: int
    SEVEN_FLOAT32S: int
    FIFTEEN_UINT16S: int
    FIFTEEN_INT16S: int
    EIGHT_UINT32S: int
    EIGHT_INT32S: int
    EIGHT_FLOAT32S: int
    FOUR_UINT64S: int
    FOUR_INT64S: int
    FOUR_FLOAT64S: int
    NINE_UINT32S: int
    NINE_INT32S: int
    NINE_FLOAT32S: int
    TEN_UINT32S: int
    TEN_INT32S: int
    TEN_FLOAT32S: int
    FIVE_UINT64S: int
    FIVE_INT64S: int
    FIVE_FLOAT64S: int
    ELEVEN_UINT32S: int
    ELEVEN_INT32S: int
    ELEVEN_FLOAT32S: int
    TWELVE_UINT32S: int
    TWELVE_INT32S: int
    TWELVE_FLOAT32S: int
    SIX_UINT64S: int
    SIX_INT64S: int
    SIX_FLOAT64S: int
    THIRTEEN_UINT32S: int
    THIRTEEN_INT32S: int
    THIRTEEN_FLOAT32S: int
    FOURTEEN_UINT32S: int
    FOURTEEN_INT32S: int
    FOURTEEN_FLOAT32S: int
    SEVEN_UINT64S: int
    SEVEN_INT64S: int
    SEVEN_FLOAT64S: int
    FIFTEEN_UINT32S: int
    FIFTEEN_INT32S: int
    FIFTEEN_FLOAT32S: int
    EIGHT_UINT64S: int
    EIGHT_INT64S: int
    EIGHT_FLOAT64S: int
    NINE_UINT64S: int
    NINE_INT64S: int
    NINE_FLOAT64S: int
    TEN_UINT64S: int
    TEN_INT64S: int
    TEN_FLOAT64S: int
    ELEVEN_UINT64S: int
    ELEVEN_INT64S: int
    ELEVEN_FLOAT64S: int
    TWELVE_UINT64S: int
    TWELVE_INT64S: int
    TWELVE_FLOAT64S: int
    THIRTEEN_UINT64S: int
    THIRTEEN_INT64S: int
    THIRTEEN_FLOAT64S: int
    FOURTEEN_UINT64S: int
    FOURTEEN_INT64S: int
    FOURTEEN_FLOAT64S: int
    FIFTEEN_UINT64S: int
    FIFTEEN_INT64S: int
    FIFTEEN_FLOAT64S: int
    def as_uint8(self) -> np.uint8:
        """Converts the enum value to numpy.uint8 type.

        Returns:
            The enum value as a numpy unsigned 8-bit integer.
        """
    def get_prototype(self) -> PrototypeType:
        """Returns the prototype object associated with this prototype enum value.

        The prototype object returned by this method can be passed to the reading method of the TransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this enum class.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type.
        """
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

@dataclass(frozen=True)
class RepeatedModuleCommand:
    """Instructs the addressed Module to repeatedly (recurrently) run the specified command."""

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
    cycle_delay: np.uint32 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""

@dataclass(frozen=True)
class OneOffModuleCommand:
    """Instructs the addressed Module to run the specified command exactly once (non-recurrently)."""

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the OneOffModuleCommand object."""

@dataclass(frozen=True)
class DequeueModuleCommand:
    """Instructs the addressed Module to clear (empty) its command queue.

    Note, clearing the command queue does not terminate already executing commands, but it prevents recurrent commands
    from running again.
    """

    module_type: np.uint8
    module_id: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""

@dataclass(frozen=True)
class KernelCommand:
    """Instructs the Kernel to run the specified command exactly once.

    Currently, the Kernel only supports blocking one-off commands.
    """

    command: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelCommand object."""

@dataclass(frozen=True)
class ModuleParameters:
    """Instructs the addressed Module to overwrite its custom parameters object with the included object data."""

    module_type: np.uint8
    module_id: np.uint8
    parameter_data: tuple[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool, ...]
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleParameters object."""

@dataclass(frozen=True)
class KernelParameters:
    """Instructs the Kernel to update the microcontroller-wide parameters with the values included in the message.

    These parameters are shared by the Kernel and all custom Modules, and the exact parameter layout is hardcoded. This
    is in contrast to Module parameters, that differ between module types.
    """

    action_lock: np.bool
    ttl_lock: np.bool
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelParameters object."""

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

    protocol_code: Incomplete
    message: Incomplete
    module_type: Incomplete
    module_id: Incomplete
    command: Incomplete
    event: Incomplete
    data_object: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: TransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleData object."""

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

    protocol_code: Incomplete
    message: Incomplete
    command: Incomplete
    event: Incomplete
    data_object: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: TransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the KernelData object."""

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

    protocol_code: Incomplete
    message: Incomplete
    module_type: Incomplete
    module_id: Incomplete
    command: Incomplete
    event: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: TransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        """
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleState object."""

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

    protocol_code: Incomplete
    message: Incomplete
    command: Incomplete
    event: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: TransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the KernelState object."""

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

    protocol_code: Incomplete
    message: Incomplete
    reception_code: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: TransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the ReceptionCode object."""

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

    protocol_code: Incomplete
    message: Incomplete
    controller_id: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: TransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the ControllerIdentification object."""

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

    protocol_code: Incomplete
    message: Incomplete
    module_type_id: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: TransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleIdentification object."""

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

    _transport_layer: Incomplete
    _module_data: Incomplete
    _kernel_data: Incomplete
    _module_state: Incomplete
    _kernel_state: Incomplete
    _controller_identification: Incomplete
    _module_identification: Incomplete
    _reception_code: Incomplete
    _timestamp_timer: Incomplete
    _source_id: Incomplete
    _logger_queue: Incomplete
    _usb_port: Incomplete
    def __init__(
        self,
        source_id: np.uint8,
        microcontroller_serial_buffer_size: int,
        usb_port: str,
        logger_queue: MPQueue,
        baudrate: int = 115200,
        *,
        test_mode: bool = False,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the SerialCommunication object."""
    def send_message(
        self,
        message: RepeatedModuleCommand
        | OneOffModuleCommand
        | DequeueModuleCommand
        | KernelCommand
        | KernelParameters
        | ModuleParameters,
    ) -> None:
        """Serializes the input command or parameters message and sends it to the connected microcontroller.

        This method relies on every valid outgoing message structure exposing a packed_data attribute, that contains
        the serialized payload data to be sent. Functionally, this method is a wrapper around the
        TransportLayer's write_data() and send_data() methods.

        Args:
            message: The command or parameters message to send to the microcontroller.
        """
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
    def _log_data(self, timestamp: int, data: NDArray[np.uint8]) -> None:
        """Packages and sends the input data to teh DataLogger instance that writes it to disk.

        Args:
            timestamp: The value of the timestamp timer 'elapsed' property that communicates the number of elapsed
                microseconds relative to the 'onset' timestamp.
            data: The byte-serialized message payload that was sent or received.
        """

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

    _ip: Incomplete
    _port: Incomplete
    _connected: bool
    _monitored_topics: Incomplete
    _output_queue: Incomplete
    _client: Incomplete
    def __init__(
        self, ip: str = "127.0.0.1", port: int = 1883, monitored_topics: None | tuple[str, ...] = None
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the MQTTCommunication object."""
    def __del__(self) -> None:
        """Ensures proper resource release when the class instance is garbage-collected."""
    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:
        """The callback function used to receive data from MQTT broker.

        When passed to the client, this function will be called each time a new message is received. This function
        will then record the message topic and payload and put them into the output_queue for the data to be consumed
        by external processes.

        Args:
            _client: The MQTT client that received the message. Currently not used.
            _userdata: Custom user-defined data. Currently not used.
            message: The received MQTT message.
        """
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
    @property
    def has_data(self) -> bool:
        """Returns True if the instance received messages from other MQTT clients and can output received data via the
        get_dataq() method.
        """
    def get_data(self) -> tuple[str, bytes | bytearray] | None:
        """Extracts and returns the first available message stored inside the instance buffer queue.

        Returns:
            A two-element tuple. The first element is a string that communicates the MQTT topic of the received message.
            The second element is the payload of the message, which is a bytes or bytearray object. If no buffered
            objects are stored in the queue (queue is empty), returns None.

        Raises:
            RuntimeError: If the instance is not connected to the MQTT broker.
        """
    def disconnect(self) -> None:
        """Disconnects the client from the MQTT broker."""
