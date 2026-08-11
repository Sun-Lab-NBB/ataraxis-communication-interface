"""Provides the command and data message classes used to construct outgoing messages and parse incoming messages
exchanged between host-machines (PCs) and Arduino / Teensy microcontrollers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import field, dataclass

import numpy as np
from ataraxis_base_utilities import console

from .protocols import PrototypeType, SerialProtocols

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MAXIMUM_PARAMETERS_SIZE: int = 250
"""The maximum combined size, in bytes, of the parameter values transmitted by a single ModuleParameters message.

COBS encodes at most 254 payload bytes behind a single overhead byte, which caps every payload the TransportLayer
transmits at 254 bytes. The four-byte ModuleParameters header travels inside that payload, leaving 250 bytes for the
serialized parameter values. This is the protocol ceiling, which a microcontroller reaches only when its serial buffer
is large enough. Boards with smaller buffers cap the payload lower, and the TransportLayer reports that bound.
"""

_ZERO_BYTE: np.uint8 = np.uint8(0)
"""Default zero value for uint8 fields used across message dataclasses."""

_ZERO_SHORT: np.uint16 = np.uint16(0)
"""Default zero value for uint16 fields used across message dataclasses."""

_ZERO_LONG: np.uint32 = np.uint32(0)
"""Default zero value for uint32 fields used across message dataclasses."""

_TRUE: np.bool_ = np.bool_(True)  # noqa: FBT003 - the positional argument is a numpy scalar value, not a flag.
"""Default boolean true value for noblock fields used across command dataclasses."""


@dataclass(frozen=True, slots=True)
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
        """Returns a string representation of the RepeatedModuleCommand instance."""
        return (
            f"RepeatedModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock}, cycle_delay={self.cycle_delay} us)"
        )


@dataclass(frozen=True, slots=True)
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
        """Returns a string representation of the OneOffModuleCommand instance."""
        return (
            f"OneOffModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock})"
        )


@dataclass(frozen=True, slots=True)
class DequeueModuleCommand:
    """Instructs the addressed Module instance to clear (empty) its command queue."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module instance within the broader module family."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
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
        """Returns a string representation of the DequeueModuleCommand instance."""
        return (
            f"DequeueModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code})"
        )


@dataclass(frozen=True, slots=True)
class KernelCommand:
    """Instructs the Kernel to run the specified command exactly once."""

    command: np.uint8
    """The code of the command to execute."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
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
        """Returns a string representation of the KernelCommand instance."""
        return (
            f"KernelCommand(protocol_code={self.protocol_code}, command={self.command}, return_code={self.return_code})"
        )


@dataclass(frozen=True, slots=True)
class ModuleParameters:
    """Instructs the addressed Module instance to update its parameters with the included data."""

    module_type: np.uint8
    """The type (family) code of the module to which the command is addressed."""
    module_id: np.uint8
    """The ID of the specific module instance within the broader module family."""
    parameter_data: tuple[np.number[Any] | np.bool_, ...]
    """A tuple of parameter values to send. Each value must be a numpy scalar or numpy boolean (e.g., np.uint8,
    np.float32), as serialization relies on the numpy itemsize and tobytes() interface. The values must match the type
    and order of the addressed module's parameter structure on the microcontroller."""
    return_code: np.uint8 = _ZERO_BYTE
    """The code to use for acknowledging the reception of the message, if set to a non-zero value."""
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    """Stores the serialized message data."""
    parameters_size: int | None = field(init=False, default=None)
    """Stores the total size of the serialized parameters in bytes."""
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.MODULE_PARAMETERS.as_uint8())
    """Stores the message protocol code."""

    def __post_init__(self) -> None:
        """Serializes the instance's data.

        Raises:
            ValueError: If the combined size of the serialized parameter values exceeds the ceiling the COBS framing
                imposes on a single payload. Microcontrollers with small serial buffers cap the size below that
                ceiling, and the TransportLayer reports the lower bound when the message reaches it.
        """
        # Calculates the total size of serialized parameters in bytes directly from item sizes. The sum accumulates in
        # Python integer arithmetic, as a uint8 accumulator silently wraps at the very sizes the guard below rejects.
        parameters_size = sum(parameter.itemsize for parameter in self.parameter_data)
        if parameters_size > _MAXIMUM_PARAMETERS_SIZE:
            message = (
                f"Unable to serialize the parameters message addressed to the module with ID {self.module_id} and "
                f"type {self.module_type}. The combined size of the serialized parameter values must be at most "
                f"{_MAXIMUM_PARAMETERS_SIZE} bytes, but got {parameters_size} bytes."
            )
            console.error(message=message, error=ValueError)
        object.__setattr__(self, "parameters_size", parameters_size)

        packed_data = np.empty(4 + parameters_size, dtype=np.uint8)

        packed_data[0:4] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
        ]

        current_position = 4
        for parameter in self.parameter_data:
            parameter_bytes = np.frombuffer(parameter.tobytes(), dtype=np.uint8)
            parameter_size = parameter_bytes.size
            packed_data[current_position : current_position + parameter_size] = parameter_bytes
            current_position += parameter_size

        object.__setattr__(self, "packed_data", packed_data)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleParameters instance."""
        return (
            f"ModuleParameters(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code}, "
            f"parameter_object_size={self.parameters_size} bytes)"
        )


@dataclass(slots=True)
class ModuleData:
    """Communicates that the Module has encountered a notable event and includes an additional data object.

    Notes:
        Event codes are unique within each module. The same code always carries the same semantic meaning regardless of
        the command that was executing when the message was sent.
    """

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=5, dtype=np.uint8))
    """The parsed message header data."""
    data_object: PrototypeType = _ZERO_BYTE
    """The parsed data object transmitted with the message."""

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleData instance."""
        return (
            f"ModuleData(module_type={self.message[0]}, module_id={self.message[1]}, command={self.message[2]}, "
            f"event={self.message[3]}, data_object={self.data_object})"
        )

    @property
    def module_type(self) -> np.uint8:
        """Returns the type (family) code of the module that sent the message."""
        value: np.uint8 = self.message[0]
        return value

    @property
    def module_id(self) -> np.uint8:
        """Returns the unique identifier code of the module instance that sent the message."""
        value: np.uint8 = self.message[1]
        return value

    @property
    def type_id(self) -> np.uint16:
        """Returns the unique 16-bit unsigned integer value that results from combining the bits of the type-code and
        the id-code of the module instance that sent the message.
        """
        header = self.message
        value: np.uint16 = np.uint16((int(header[0]) << 8) | int(header[1]))
        return value

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the module that sent the message."""
        value: np.uint8 = self.message[2]
        return value

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        value: np.uint8 = self.message[3]
        return value

    @property
    def prototype_code(self) -> np.uint8:
        """Returns the code that specifies the type of the data object transmitted with the message."""
        value: np.uint8 = self.message[4]
        return value


@dataclass(slots=True)
class KernelData:
    """Communicates that the Kernel has encountered a notable event and includes an additional data object.

    Notes:
        Event codes are unique within the kernel. The same code always carries the same semantic meaning regardless of
        the command that was executing when the message was sent.
    """

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=3, dtype=np.uint8))
    """The parsed message header data."""
    data_object: PrototypeType = _ZERO_BYTE
    """The parsed data object transmitted with the message."""

    def __repr__(self) -> str:
        """Returns a string representation of the KernelData instance."""
        return f"KernelData(command={self.message[0]}, event={self.message[1]}, data_object={self.data_object})"

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the Kernel when it sent the message."""
        value: np.uint8 = self.message[0]
        return value

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        value: np.uint8 = self.message[1]
        return value

    @property
    def prototype_code(self) -> np.uint8:
        """Returns the code that specifies the type of the data object transmitted with the message."""
        value: np.uint8 = self.message[2]
        return value


@dataclass(slots=True)
class ModuleState:
    """Communicates that the Module has encountered a notable event.

    Notes:
        Event codes are unique within each module. The same code always carries the same semantic meaning regardless of
        the command that was executing when the message was sent.
    """

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=4, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleState instance."""
        return (
            f"ModuleState(module_type={self.message[0]}, module_id={self.message[1]}, "
            f"command={self.message[2]}, event={self.message[3]})"
        )

    @property
    def module_type(self) -> np.uint8:
        """Returns the type (family) code of the module that sent the message."""
        value: np.uint8 = self.message[0]
        return value

    @property
    def module_id(self) -> np.uint8:
        """Returns the ID of the specific module instance within the broader module family."""
        value: np.uint8 = self.message[1]
        return value

    @property
    def type_id(self) -> np.uint16:
        """Returns the unique 16-bit unsigned integer value that results from combining the bits of the type-code and
        the id-code of the module instance that sent the message.
        """
        header = self.message
        value: np.uint16 = np.uint16((int(header[0]) << 8) | int(header[1]))
        return value

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the module that sent the message."""
        value: np.uint8 = self.message[2]
        return value

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        value: np.uint8 = self.message[3]
        return value


@dataclass(slots=True)
class KernelState:
    """Communicates that the Kernel has encountered a notable event.

    Notes:
        Event codes are unique within the kernel. The same code always carries the same semantic meaning regardless of
        the command that was executing when the message was sent.
    """

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=2, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns a string representation of the KernelState instance."""
        return f"KernelState(command={self.message[0]}, event={self.message[1]})"

    @property
    def command(self) -> np.uint8:
        """Returns the code of the command executed by the Kernel when it sent the message."""
        value: np.uint8 = self.message[0]
        return value

    @property
    def event(self) -> np.uint8:
        """Returns the code of the event that prompted sending the message."""
        value: np.uint8 = self.message[1]
        return value


@dataclass(slots=True)
class ReceptionCode:
    """Communicates the reception code originally received with the message sent by the PC to indicate that the message
    was received and parsed by the microcontroller.
    """

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=1, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns a string representation of the ReceptionCode instance."""
        return f"ReceptionCode(reception_code={self.message[0]})"

    @property
    def reception_code(self) -> np.uint8:
        """Returns the reception code originally sent as part of the outgoing Command or Parameters message."""
        value: np.uint8 = self.message[0]
        return value


@dataclass(slots=True)
class ControllerIdentification:
    """Communicates the unique identifier code of the microcontroller."""

    message: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=1, dtype=np.uint8))
    """The parsed message header data."""

    def __repr__(self) -> str:
        """Returns a string representation of the ControllerIdentification instance."""
        return f"ControllerIdentification(controller_id={self.message[0]})"

    @property
    def controller_id(self) -> np.uint8:
        """Returns the unique identifier of the microcontroller."""
        value: np.uint8 = self.message[0]
        return value


@dataclass(slots=True)
class ModuleIdentification:
    """Identifies a hardware module instance by communicating its combined type and id code.

    Notes:
        The entire message payload is the combined type and ID value, so the class stores that value directly.
    """

    module_type_id: np.uint16 = _ZERO_SHORT
    """The unique uint16 code that results from combining the type and ID codes of the module instance."""

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleIdentification instance."""
        return f"ModuleIdentification(module_type_id={self.module_type_id})"
