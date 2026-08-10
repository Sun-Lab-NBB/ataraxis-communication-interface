"""Provides the status code enumerations that mirror the ataraxis-micro-controller firmware and the functions that
translate incoming Kernel and Module messages into readable error descriptions.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from ..communication import (
    KernelData,
    ModuleData,
    KernelState,
    ModuleState,
    SerialProtocols,
)

MINIMUM_CUSTOM_STATUS_CODE: int = 51
"""The lowest event code a custom hardware module is allowed to use. Every code below this bound is reserved for the
service status codes the base Module class of the ataraxis-micro-controller library defines, and this library resolves
those codes itself instead of routing them to the module interface."""

MAXIMUM_CUSTOM_STATUS_CODE: int = 250
"""The highest event code a custom hardware module is allowed to use."""

_UNRECOGNIZED_CODE_NAME: str = "UNRECOGNIZED"
"""The name used in place of a firmware enumeration name when a received code matches no known enumeration member."""


class KernelCommandCodes(IntEnum):
    """Defines the command codes the microcontroller's Kernel class accepts from the PC."""

    STANDBY = 0
    """The placeholder code the Kernel holds while it executes no command."""
    RECEIVE_DATA = 1
    """Checks for and receives PC-sent data. The Kernel issues this command to itself."""
    RESET_CONTROLLER = 2
    """Resets the software and hardware state of every asset the Kernel manages."""
    IDENTIFY_CONTROLLER = 3
    """Requests the microcontroller to report its identifier code."""
    IDENTIFY_MODULES = 4
    """Requests the microcontroller to report the combined type and id code of every hardware module it manages."""
    KEEPALIVE = 5
    """Resets the keepalive watchdog timer, which starts a new keepalive cycle."""


class KernelStatusCodes(IntEnum):
    """Defines the status codes the microcontroller's Kernel class uses to report its runtime state to the PC."""

    STANDBY = 0
    """The reserved placeholder code the Kernel never transmits."""
    SETUP_COMPLETE = 1
    """The Setup() method runtime succeeded."""
    MODULE_SETUP_ERROR = 2
    """The Setup() method runtime failed because a hardware module rejected its setup sequence."""
    RECEPTION_ERROR = 3
    """A communication error occurred while receiving data from the PC."""
    TRANSMISSION_ERROR = 4
    """A communication error occurred while sending data to the PC."""
    INVALID_MESSAGE_PROTOCOL = 5
    """A message using an unsupported protocol was received."""
    MODULE_PARAMETERS_SET = 6
    """The received parameters were applied to the addressed hardware module."""
    MODULE_PARAMETERS_ERROR = 7
    """The received parameters could not be applied to the addressed hardware module."""
    COMMAND_NOT_RECOGNIZED = 8
    """An unsupported Kernel command was received."""
    TARGET_MODULE_NOT_FOUND = 9
    """No hardware module matches the combined type and id code the message addressed."""
    KEEPALIVE_TIMEOUT = 10
    """No keepalive message arrived within the timeout window the Kernel derives from the keepalive interval."""


class ModuleStatusCodes(IntEnum):
    """Defines the service status codes the shared methods of the microcontroller's base Module class report to the PC.

    Notes:
        The firmware reserves the codes 0 through 50 for this enumeration, so a custom hardware module assigns its own
        event codes from the range that MINIMUM_CUSTOM_STATUS_CODE and MAXIMUM_CUSTOM_STATUS_CODE bound.
    """

    STANDBY = 0
    """The reserved placeholder code the base Module class never transmits."""
    TRANSMISSION_ERROR = 1
    """An error occurred while sending data to the PC."""
    COMMAND_COMPLETED = 2
    """The active command completed and was removed from the command queue."""
    COMMAND_NOT_RECOGNIZED = 3
    """The RunActiveCommand() method does not implement the requested command."""


class CommunicationStatusCodes(IntEnum):
    """Defines the status codes the microcontroller's Communication class reports for its data manipulations.

    Notes:
        These codes never arrive as message event codes. The firmware attaches the most recent one as the first byte
        of the two-byte payload that accompanies every reception and transmission error message.
    """

    STANDBY = 51
    """The value the Communication class holds before it completes any operation."""
    RECEPTION_ERROR = 52
    """The Communication class encountered an error while receiving a message."""
    PARSING_ERROR = 53
    """The Communication class encountered an error while reading a received message."""
    PACKING_ERROR = 54
    """The Communication class encountered an error while writing a message to the payload."""
    MESSAGE_SENT = 55
    """The Communication class sent a message."""
    MESSAGE_RECEIVED = 56
    """The Communication class received a message."""
    INVALID_PROTOCOL = 57
    """The message protocol code is not valid for the direction the message travelled."""
    NO_BYTES_TO_RECEIVE = 58
    """The Communication class did not receive enough bytes to process a message."""
    PARAMETER_MISMATCH = 59
    """The size of the received parameter structure does not match the expected size."""
    PARAMETERS_EXTRACTED = 60
    """The Communication class extracted the parameter data."""
    EXTRACTION_FORBIDDEN = 61
    """Parameter extraction was attempted on a message other than a ModuleParameters message."""
    TRANSMISSION_ERROR = 62
    """The communication interface accepted only a part of the transmitted message."""


class TransportStatusCodes(IntEnum):
    """Defines the status codes the microcontroller's TransportLayer class reports for its packet operations.

    Notes:
        These codes come from the ataraxis-transport-layer-mc library that runs on the microcontroller, and they
        occupy a value range and carry meanings distinct from the TransportLayerStatus codes that the PC-side
        ataraxis-transport-layer-pc library reports. The firmware attaches the most recent one as the second byte of
        the two-byte payload that accompanies every reception and transmission error message.
    """

    STANDBY = 11
    """The value the TransportLayer class holds before it completes any operation."""
    DECODING_FAILED = 12
    """The payload could not be decoded from the received packet."""
    PACKET_SENT = 13
    """The packet was transmitted."""
    PAYLOAD_SIZE_BYTE_NOT_FOUND = 14
    """The payload size byte was not found in the incoming stream."""
    INVALID_PAYLOAD_SIZE = 15
    """The received payload size is not valid."""
    PACKET_TIMEOUT_ERROR = 16
    """Packet parsing failed because the incoming byte stream stalled."""
    NO_BYTES_TO_PARSE = 17
    """No parseable packet was found in the reception buffer."""
    PACKET_PARSED = 18
    """The packet was parsed."""
    CRC_CHECK_FAILED = 19
    """The CRC check failed, which indicates that the incoming packet is corrupted."""
    PACKET_RECEIVED = 20
    """The packet was received."""
    WRITE_OBJECT_BUFFER_ERROR = 21
    """The payload region of the buffer does not have enough space to write the object."""
    OBJECT_WRITTEN_TO_BUFFER = 22
    """The object was written to the buffer."""
    READ_OBJECT_BUFFER_ERROR = 23
    """The payload region of the buffer does not hold enough bytes to read the object from."""
    OBJECT_READ_FROM_BUFFER = 24
    """The object was read from the buffer."""
    DELIMITER_NOT_FOUND_ERROR = 25
    """The delimiter byte was not found at the end of the packet."""
    DELIMITER_FOUND_TOO_EARLY_ERROR = 26
    """The delimiter byte was found before the end of the packet."""
    POSTAMBLE_TIMEOUT_ERROR = 27
    """The postamble was not received within the reception timeout."""
    EMPTY_PAYLOAD_ERROR = 28
    """The packet could not be sent because the staged payload is empty."""
    PACKET_PARTIALLY_SENT = 29
    """The communication interface accepted only a part of the packet."""


_KERNEL_ERROR_DESCRIPTIONS: dict[int, str] = {
    KernelStatusCodes.STANDBY: "Reserved placeholder code that the Kernel never transmits.",
    KernelStatusCodes.MODULE_SETUP_ERROR: (
        "A hardware module failed its setup sequence. The controller runs no commands until its firmware is "
        "re-uploaded."
    ),
    KernelStatusCodes.RECEPTION_ERROR: "Reception failed.",
    KernelStatusCodes.TRANSMISSION_ERROR: "Transmission failed.",
    KernelStatusCodes.INVALID_MESSAGE_PROTOCOL: (
        "The received message declared a protocol code the microcontroller does not accept."
    ),
    KernelStatusCodes.MODULE_PARAMETERS_ERROR: ("The addressed hardware module rejected the PC-sent parameter block."),
    KernelStatusCodes.COMMAND_NOT_RECOGNIZED: "The Kernel does not implement the received command code.",
    KernelStatusCodes.TARGET_MODULE_NOT_FOUND: "No hardware module matches the addressed type and id codes.",
    KernelStatusCodes.KEEPALIVE_TIMEOUT: (
        "No keepalive command arrived within the timeout window. The Kernel performed an emergency reset, returning "
        "all managed hardware to its default state."
    ),
}
"""Maps each Kernel status code that reports a fault to the condition it reports. Membership in this table defines
which Kernel codes interrupt the runtime, so the codes the Kernel uses to report ordinary progress are absent."""

_MODULE_ERROR_DESCRIPTIONS: dict[int, str] = {
    ModuleStatusCodes.STANDBY: "Reserved placeholder code that the base Module class never transmits.",
    ModuleStatusCodes.TRANSMISSION_ERROR: "Transmission failed.",
    ModuleStatusCodes.COMMAND_NOT_RECOGNIZED: "The module does not implement the received command code.",
}
"""Maps each service status code that reports a fault to the condition it reports. Membership in this table defines
which service codes interrupt the runtime, so the code that reports command completion is absent."""

_KERNEL_STATUS_VALUES: frozenset[int] = frozenset(code.value for code in KernelStatusCodes)
"""The value of every status code the Kernel defines. Every received Kernel message tests its event code against this
set, which resolves through one hash probe, because testing against the enumeration itself walks a list of values
behind a metaclass call."""

_MODULE_STATUS_VALUES: frozenset[int] = frozenset(code.value for code in ModuleStatusCodes)
"""The value of every service status code the base Module class defines. Serves the same per-message membership test
for module messages that the Kernel set above serves for Kernel messages."""

_COMMUNICATION_STATUS_MEANINGS: dict[int, str] = {
    CommunicationStatusCodes.STANDBY: "no operation completed since the last reset",
    CommunicationStatusCodes.RECEPTION_ERROR: "could not read a complete message from the serial stream",
    CommunicationStatusCodes.PARSING_ERROR: "message payload did not match the layout its protocol code declares",
    CommunicationStatusCodes.PACKING_ERROR: "outgoing message did not fit the transmission payload",
    CommunicationStatusCodes.MESSAGE_SENT: "last recorded operation was a completed transmission",
    CommunicationStatusCodes.MESSAGE_RECEIVED: "last recorded operation was a completed reception",
    CommunicationStatusCodes.INVALID_PROTOCOL: "protocol code not valid for the message's direction",
    CommunicationStatusCodes.NO_BYTES_TO_RECEIVE: "no complete message in the serial stream",
    CommunicationStatusCodes.PARAMETER_MISMATCH: (
        "received parameter block size differs from the module's parameter structure size"
    ),
    CommunicationStatusCodes.PARAMETERS_EXTRACTED: "last recorded operation was a completed parameter extraction",
    CommunicationStatusCodes.EXTRACTION_FORBIDDEN: (
        "parameter extraction attempted on a message carrying no parameter block"
    ),
    CommunicationStatusCodes.TRANSMISSION_ERROR: "serial interface accepted only part of the outgoing message",
}
"""Maps each Communication class status code to the condition it reports."""

_TRANSPORT_STATUS_MEANINGS: dict[int, str] = {
    TransportStatusCodes.STANDBY: "no operation completed since the last reset",
    TransportStatusCodes.DECODING_FAILED: "COBS decoding failed, packet bytes likely corrupted in transit",
    TransportStatusCodes.PACKET_SENT: "last recorded operation was a transmitted packet",
    TransportStatusCodes.PAYLOAD_SIZE_BYTE_NOT_FOUND: "packet start byte found, payload size byte never arrived",
    TransportStatusCodes.INVALID_PAYLOAD_SIZE: "declared payload size outside the accepted range",
    TransportStatusCodes.PACKET_TIMEOUT_ERROR: "packet bytes stopped arriving before the packet completed",
    TransportStatusCodes.NO_BYTES_TO_PARSE: "no packet start byte in the reception buffer",
    TransportStatusCodes.PACKET_PARSED: "last recorded operation was a parsed packet",
    TransportStatusCodes.CRC_CHECK_FAILED: "CRC mismatch, packet bytes likely corrupted in transit",
    TransportStatusCodes.PACKET_RECEIVED: "last recorded operation was a received packet",
    TransportStatusCodes.WRITE_OBJECT_BUFFER_ERROR: "transmission buffer payload region too small for the object",
    TransportStatusCodes.OBJECT_WRITTEN_TO_BUFFER: "last recorded operation was a completed buffer write",
    TransportStatusCodes.READ_OBJECT_BUFFER_ERROR: (
        "reception buffer payload region holds fewer bytes than the object requires"
    ),
    TransportStatusCodes.OBJECT_READ_FROM_BUFFER: "last recorded operation was a completed buffer read",
    TransportStatusCodes.DELIMITER_NOT_FOUND_ERROR: (
        "payload delimiter byte absent, packet bytes likely corrupted in transit"
    ),
    TransportStatusCodes.DELIMITER_FOUND_TOO_EARLY_ERROR: (
        "payload delimiter byte appeared early, packet bytes likely corrupted in transit"
    ),
    TransportStatusCodes.POSTAMBLE_TIMEOUT_ERROR: "CRC postamble did not arrive within the reception timeout",
    TransportStatusCodes.EMPTY_PAYLOAD_ERROR: "send attempted with an empty staged payload",
    TransportStatusCodes.PACKET_PARTIALLY_SENT: "serial interface accepted only part of the outgoing packet",
}
"""Maps each microcontroller-side TransportLayer status code to the condition it reports."""


def describe_kernel_event(
    message: KernelData | KernelState,
    controller_id: np.uint8,
    controller_name: str,
) -> str | None:
    """Builds the description of the fault the input Kernel message reports.

    Args:
        message: The Kernel message received from the microcontroller.
        controller_id: The identifier code of the microcontroller that sent the message.
        controller_name: The human-readable name of the microcontroller that sent the message.

    Returns:
        The description of the reported fault, or None when the message reports an ordinary Kernel state.
    """
    # The Kernel reports ordinary progress far more often than it reports a fault, and this runs inside the
    # communication loop, so the table lookup that rules out a fault precedes every string the description needs.
    event = int(message.event)
    description = _KERNEL_ERROR_DESCRIPTIONS.get(event)
    if description is None and event in _KERNEL_STATUS_VALUES:
        return None

    context = (
        f"Microcontroller {controller_id} ('{controller_name}') Kernel status "
        f"{_format_code(code=event, code_type=KernelStatusCodes)} during Kernel command "
        f"{_format_code(code=int(message.command), code_type=KernelCommandCodes)}."
    )
    if description is None:
        return _join_sentences(context, _describe_unrecognized_code(code=event, code_type=KernelStatusCodes))

    return _join_sentences(context, description, _describe_kernel_payload(message=message, event=event))


def describe_module_event(
    message: ModuleData | ModuleState,
    controller_id: np.uint8,
    controller_name: str,
    module_name: str,
) -> str | None:
    """Builds the description of the fault the input service Module message reports.

    Notes:
        This function resolves the service status codes the base Module class of the firmware defines, which occupy
        the event code range below MINIMUM_CUSTOM_STATUS_CODE. Codes at or above that bound belong to the custom
        hardware module and are resolved by describe_custom_module_error() instead.

    Args:
        message: The Module message received from the microcontroller.
        controller_id: The identifier code of the microcontroller that manages the reporting hardware module.
        controller_name: The human-readable name of the microcontroller that manages the reporting hardware module.
        module_name: The human-readable name of the hardware module that sent the message.

    Returns:
        The description of the reported fault, or None when the message reports an ordinary module state.
    """
    # Every completed command reaches this function, and it runs inside the communication loop, so the table lookup
    # that rules out a fault precedes every string the description needs.
    event = int(message.event)
    description = _MODULE_ERROR_DESCRIPTIONS.get(event)
    if description is None and event in _MODULE_STATUS_VALUES:
        return None

    source = _format_module_context(
        message=message,
        controller_id=controller_id,
        controller_name=controller_name,
        module_name=module_name,
    )
    context = (
        f"{source} service status {_format_code(code=event, code_type=ModuleStatusCodes)} during command "
        f"{message.command}."
    )
    if description is None:
        return _join_sentences(context, _describe_unrecognized_code(code=event, code_type=ModuleStatusCodes))

    detail = _describe_serial_failure(message=message) if event == ModuleStatusCodes.TRANSMISSION_ERROR else ""
    return _join_sentences(context, description, detail)


def describe_custom_module_error(
    message: ModuleData | ModuleState,
    controller_id: np.uint8,
    controller_name: str,
    module_name: str,
    description: str,
) -> str:
    """Builds the description of the custom error the input Module message reports.

    Args:
        message: The Module message received from the microcontroller.
        controller_id: The identifier code of the microcontroller that manages the reporting hardware module.
        controller_name: The human-readable name of the microcontroller that manages the reporting hardware module.
        module_name: The human-readable name of the hardware module that sent the message.
        description: The explanation the module interface registered for the message's event code.

    Returns:
        The description of the reported error.
    """
    source = _format_module_context(
        message=message,
        controller_id=controller_id,
        controller_name=controller_name,
        module_name=module_name,
    )
    context = f"{source} error code {message.event} during command {message.command}."
    payload = f"Data object: {message.data_object}." if isinstance(message, ModuleData) else ""
    return _join_sentences(context, description, payload)


def _format_code(code: int, code_type: type[IntEnum]) -> str:
    """Returns the firmware enumeration name of the input status code alongside its numeric value."""
    if code not in code_type:
        return f"{_UNRECOGNIZED_CODE_NAME} (code {code})"
    return f"{code_type(code).name} (code {code})"


def _format_module_context(
    message: ModuleData | ModuleState,
    controller_id: np.uint8,
    controller_name: str,
    module_name: str,
) -> str:
    """Returns the clause that identifies the hardware module that sent the input message."""
    return (
        f"Hardware module '{module_name}' (type {message.module_type}, id {message.module_id}) on microcontroller "
        f"{controller_id} ('{controller_name}')"
    )


def _describe_unrecognized_code(code: int, code_type: type[IntEnum]) -> str:
    """Returns the clause that reports a status code matching no member of the input firmware enumeration."""
    known_codes = sorted(member.value for member in code_type)
    return f"Code {code} is outside the {known_codes[0]} through {known_codes[-1]} range this library resolves."


def _describe_kernel_payload(message: KernelData | KernelState, event: int) -> str:
    """Returns the clause that interprets the data object accompanying the input Kernel fault message."""
    if not isinstance(message, KernelData):
        return ""

    if event in (KernelStatusCodes.RECEPTION_ERROR, KernelStatusCodes.TRANSMISSION_ERROR):
        return _describe_serial_failure(message=message)

    if event in (
        KernelStatusCodes.MODULE_SETUP_ERROR,
        KernelStatusCodes.MODULE_PARAMETERS_ERROR,
        KernelStatusCodes.TARGET_MODULE_NOT_FOUND,
    ):
        module_codes = _resolve_payload_values(message=message, count=2)
        if module_codes is None:
            return ""
        return f"Hardware module: type {module_codes[0]}, id {module_codes[1]}."

    if event == KernelStatusCodes.INVALID_MESSAGE_PROTOCOL:
        protocol = _resolve_payload_values(message=message, count=1)
        if protocol is None:
            return ""
        return f"Rejected protocol: {_format_code(code=protocol[0], code_type=SerialProtocols)}."

    if event == KernelStatusCodes.KEEPALIVE_TIMEOUT:
        timeout = _resolve_payload_values(message=message, count=1)
        if timeout is None:
            return ""
        # The firmware sends the timeout it derives rather than the interval it was configured with, and the two
        # differ by the factor of two the Kernel applies, so the message states which of the values it carries.
        return f"Timeout window: {timeout[0]} milliseconds, twice the firmware's configured keepalive interval."

    return ""


def _describe_serial_failure(message: KernelData | ModuleData | ModuleState) -> str:
    """Returns the clause that interprets the Communication and TransportLayer status pair carried by the input
    message.
    """
    if not isinstance(message, (KernelData, ModuleData)):
        return ""

    statuses = _resolve_payload_values(message=message, count=2)
    if statuses is None:
        return ""

    return _join_sentences(
        _format_serial_status(
            layer="Communication",
            code=statuses[0],
            code_type=CommunicationStatusCodes,
            meanings=_COMMUNICATION_STATUS_MEANINGS,
        ),
        _format_serial_status(
            layer="TransportLayer",
            code=statuses[1],
            code_type=TransportStatusCodes,
            meanings=_TRANSPORT_STATUS_MEANINGS,
        ),
    )


def _format_serial_status(layer: str, code: int, code_type: type[IntEnum], meanings: dict[int, str]) -> str:
    """Returns the clause that reports one serial layer's status code alongside the condition it reports."""
    meaning = meanings.get(code)
    if meaning is None:
        return f"{layer} {_format_code(code=code, code_type=code_type)}."
    return f"{layer} {_format_code(code=code, code_type=code_type)}: {meaning}."


def _resolve_payload_values(message: KernelData | ModuleData, count: int) -> tuple[int, ...] | None:
    """Returns the data object of the input message as integers, or None when the object does not hold exactly the
    requested number of values.
    """
    payload = np.atleast_1d(message.data_object)
    if payload.size != count:
        return None
    return tuple(int(value) for value in payload)


def _join_sentences(*sentences: str) -> str:
    """Returns the input sentences joined into a single message, with the empty ones left out."""
    return " ".join(sentence for sentence in sentences if sentence)
