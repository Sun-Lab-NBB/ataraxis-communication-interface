"""Contains tests for the enumerations and functions provided by the status_codes.py module."""

import numpy as np
import pytest

from ataraxis_communication_interface.communication import (
    KernelData,
    ModuleData,
    KernelState,
    ModuleState,
)
from ataraxis_communication_interface.microcontroller.status_codes import (
    MAXIMUM_CUSTOM_STATUS_CODE,
    MINIMUM_CUSTOM_STATUS_CODE,
    _KERNEL_ERROR_DESCRIPTIONS,
    _MODULE_ERROR_DESCRIPTIONS,
    _TRANSPORT_STATUS_MEANINGS,
    _COMMUNICATION_STATUS_MEANINGS,
    KernelStatusCodes,
    ModuleStatusCodes,
    KernelCommandCodes,
    TransportStatusCodes,
    CommunicationStatusCodes,
    describe_kernel_event,
    describe_module_event,
    describe_custom_module_error,
)

_CONTROLLER_ID = np.uint8(3)
_CONTROLLER_NAME = "actor_controller"
_MODULE_NAME = "water_valve"


def _kernel_data(command: int, event: int, payload: np.number | np.ndarray) -> KernelData:
    """Builds a KernelData message carrying the requested command, event, and data object."""
    return KernelData(message=np.array([command, event, 0], dtype=np.uint8), data_object=payload)


def _kernel_state(command: int, event: int) -> KernelState:
    """Builds a KernelState message carrying the requested command and event."""
    return KernelState(message=np.array([command, event], dtype=np.uint8))


def _module_data(command: int, event: int, payload: np.number | np.ndarray) -> ModuleData:
    """Builds a ModuleData message carrying the requested command, event, and data object."""
    return ModuleData(message=np.array([5, 1, command, event, 0], dtype=np.uint8), data_object=payload)


def _module_state(command: int, event: int) -> ModuleState:
    """Builds a ModuleState message carrying the requested command and event."""
    return ModuleState(message=np.array([5, 1, command, event], dtype=np.uint8))


def _describe_kernel(message: KernelData | KernelState) -> str | None:
    """Describes the input Kernel message using the shared controller identity."""
    return describe_kernel_event(message=message, controller_id=_CONTROLLER_ID, controller_name=_CONTROLLER_NAME)


def _describe_module(message: ModuleData | ModuleState) -> str | None:
    """Describes the input Module message using the shared controller and module identities."""
    return describe_module_event(
        message=message,
        controller_id=_CONTROLLER_ID,
        controller_name=_CONTROLLER_NAME,
        module_name=_MODULE_NAME,
    )


def test_kernel_status_codes_mirror_firmware() -> None:
    """Verifies that the Kernel status enumeration reproduces the firmware's kKernelStatusCodes values."""
    assert {member.name: member.value for member in KernelStatusCodes} == {
        "STANDBY": 0,
        "SETUP_COMPLETE": 1,
        "MODULE_SETUP_ERROR": 2,
        "RECEPTION_ERROR": 3,
        "TRANSMISSION_ERROR": 4,
        "INVALID_MESSAGE_PROTOCOL": 5,
        "MODULE_PARAMETERS_SET": 6,
        "MODULE_PARAMETERS_ERROR": 7,
        "COMMAND_NOT_RECOGNIZED": 8,
        "TARGET_MODULE_NOT_FOUND": 9,
        "KEEPALIVE_TIMEOUT": 10,
    }


def test_kernel_command_codes_mirror_firmware() -> None:
    """Verifies that the Kernel command enumeration reproduces the firmware's kKernelCommands values."""
    assert {member.name: member.value for member in KernelCommandCodes} == {
        "STANDBY": 0,
        "RECEIVE_DATA": 1,
        "RESET_CONTROLLER": 2,
        "IDENTIFY_CONTROLLER": 3,
        "IDENTIFY_MODULES": 4,
        "KEEPALIVE": 5,
    }


def test_module_status_codes_mirror_firmware() -> None:
    """Verifies that the service status enumeration reproduces the firmware's kCoreStatusCodes values."""
    assert {member.name: member.value for member in ModuleStatusCodes} == {
        "STANDBY": 0,
        "TRANSMISSION_ERROR": 1,
        "COMMAND_COMPLETED": 2,
        "COMMAND_NOT_RECOGNIZED": 3,
    }


def test_communication_status_codes_mirror_firmware() -> None:
    """Verifies that the Communication status enumeration spans the firmware's reserved 51 through 62 range."""
    assert [member.value for member in CommunicationStatusCodes] == list(range(51, 63))
    assert CommunicationStatusCodes.PARSING_ERROR == 53
    assert CommunicationStatusCodes.TRANSMISSION_ERROR == 62


def test_transport_status_codes_mirror_firmware() -> None:
    """Verifies that the TransportLayer status enumeration spans the firmware's reserved 11 through 29 range."""
    assert [member.value for member in TransportStatusCodes] == list(range(11, 30))
    assert TransportStatusCodes.CRC_CHECK_FAILED == 19
    assert TransportStatusCodes.PACKET_PARTIALLY_SENT == 29


def test_custom_status_code_range_abuts_the_service_range() -> None:
    """Verifies that the custom event code range starts directly above the firmware's reserved service range."""
    assert MINIMUM_CUSTOM_STATUS_CODE == 51
    assert MAXIMUM_CUSTOM_STATUS_CODE == 250
    assert max(member.value for member in ModuleStatusCodes) < MINIMUM_CUSTOM_STATUS_CODE


def test_description_tables_cover_every_fault_code() -> None:
    """Verifies that the Kernel and module tables hold exactly the codes that report a fault, while the two
    serial layer tables hold their whole enumeration."""
    assert set(_KERNEL_ERROR_DESCRIPTIONS) == {
        KernelStatusCodes.STANDBY,
        KernelStatusCodes.MODULE_SETUP_ERROR,
        KernelStatusCodes.RECEPTION_ERROR,
        KernelStatusCodes.TRANSMISSION_ERROR,
        KernelStatusCodes.INVALID_MESSAGE_PROTOCOL,
        KernelStatusCodes.MODULE_PARAMETERS_ERROR,
        KernelStatusCodes.COMMAND_NOT_RECOGNIZED,
        KernelStatusCodes.TARGET_MODULE_NOT_FOUND,
        KernelStatusCodes.KEEPALIVE_TIMEOUT,
    }
    assert set(_MODULE_ERROR_DESCRIPTIONS) == {
        ModuleStatusCodes.STANDBY,
        ModuleStatusCodes.TRANSMISSION_ERROR,
        ModuleStatusCodes.COMMAND_NOT_RECOGNIZED,
    }
    # Any code the firmware may report as the last status of either serial layer can accompany a fault message, so
    # both nested tables cover their whole enumeration rather than the fault subset.
    assert set(_COMMUNICATION_STATUS_MEANINGS) == set(CommunicationStatusCodes)
    assert set(_TRANSPORT_STATUS_MEANINGS) == set(TransportStatusCodes)


def test_every_description_carries_prose() -> None:
    """Verifies that every description table entry holds non-empty text."""
    for description in (
        *_KERNEL_ERROR_DESCRIPTIONS.values(),
        *_MODULE_ERROR_DESCRIPTIONS.values(),
        *_COMMUNICATION_STATUS_MEANINGS.values(),
        *_TRANSPORT_STATUS_MEANINGS.values(),
    ):
        assert description


def test_descriptions_prescribe_no_response() -> None:
    """Verifies that no description offers a remedy, as the reported code cannot establish a cause."""
    for description in (
        *_KERNEL_ERROR_DESCRIPTIONS.values(),
        *_MODULE_ERROR_DESCRIPTIONS.values(),
        *_COMMUNICATION_STATUS_MEANINGS.values(),
        *_TRANSPORT_STATUS_MEANINGS.values(),
    ):
        assert "Recommended action" not in description
        assert "Confirm that" not in description
        assert "Inspect the" not in description


@pytest.mark.parametrize("event", [KernelStatusCodes.SETUP_COMPLETE, KernelStatusCodes.MODULE_PARAMETERS_SET])
def test_non_fault_kernel_events_produce_no_description(event: KernelStatusCodes) -> None:
    """Verifies that the Kernel codes reporting ordinary progress produce no description."""
    assert _describe_kernel(_kernel_state(command=2, event=event)) is None


def test_non_fault_module_event_produces_no_description() -> None:
    """Verifies that the service code reporting command completion produces no description."""
    assert _describe_module(_module_state(command=4, event=ModuleStatusCodes.COMMAND_COMPLETED)) is None


def test_every_kernel_fault_code_produces_a_description() -> None:
    """Verifies that every Kernel fault code resolves to a description naming the code and its response."""
    for code in _KERNEL_ERROR_DESCRIPTIONS:
        description = _describe_kernel(_kernel_data(command=1, event=code, payload=np.zeros(2, dtype=np.uint8)))
        assert description is not None
        assert KernelStatusCodes(code).name in description
        assert f"code {code}" in description
        assert _KERNEL_ERROR_DESCRIPTIONS[code] in description


def test_every_module_fault_code_produces_a_description() -> None:
    """Verifies that every service fault code resolves to a description naming the module and the code."""
    for code in _MODULE_ERROR_DESCRIPTIONS:
        description = _describe_module(_module_data(command=4, event=code, payload=np.zeros(2, dtype=np.uint8)))
        assert description is not None
        assert ModuleStatusCodes(code).name in description
        assert _MODULE_NAME in description
        assert _MODULE_ERROR_DESCRIPTIONS[code] in description


def test_reception_error_names_both_serial_layer_codes() -> None:
    """Verifies that a reception fault resolves both bytes of its payload into named serial layer statuses."""
    payload = np.array([CommunicationStatusCodes.PARSING_ERROR, TransportStatusCodes.CRC_CHECK_FAILED], dtype=np.uint8)
    description = _describe_kernel(_kernel_data(command=1, event=KernelStatusCodes.RECEPTION_ERROR, payload=payload))

    assert description is not None
    assert "PARSING_ERROR (code 53)" in description
    assert "CRC_CHECK_FAILED (code 19)" in description
    assert _COMMUNICATION_STATUS_MEANINGS[CommunicationStatusCodes.PARSING_ERROR] in description
    assert _TRANSPORT_STATUS_MEANINGS[TransportStatusCodes.CRC_CHECK_FAILED] in description


def test_module_transmission_error_names_both_serial_layer_codes() -> None:
    """Verifies that a module transmission fault resolves both bytes of its payload into named serial layer statuses."""
    payload = np.array(
        [CommunicationStatusCodes.TRANSMISSION_ERROR, TransportStatusCodes.PACKET_PARTIALLY_SENT], dtype=np.uint8
    )
    description = _describe_module(_module_data(command=4, event=ModuleStatusCodes.TRANSMISSION_ERROR, payload=payload))

    assert description is not None
    assert "TRANSMISSION_ERROR (code 62)" in description
    assert "PACKET_PARTIALLY_SENT (code 29)" in description


def test_module_transmission_error_drops_the_serial_clause_for_state_messages() -> None:
    """Verifies that a module transmission fault delivered as a state message reports no serial layer codes."""
    description = _describe_module(_module_state(command=4, event=ModuleStatusCodes.TRANSMISSION_ERROR))

    assert description is not None
    assert "TRANSMISSION_ERROR (code 1)" in description
    assert "Communication " not in description


def test_module_addressed_faults_name_the_reported_module() -> None:
    """Verifies that the Kernel faults carrying a module type and id pair report both codes."""
    payload = np.array([7, 2], dtype=np.uint8)
    for code in (
        KernelStatusCodes.MODULE_SETUP_ERROR,
        KernelStatusCodes.MODULE_PARAMETERS_ERROR,
        KernelStatusCodes.TARGET_MODULE_NOT_FOUND,
    ):
        description = _describe_kernel(_kernel_data(command=2, event=code, payload=payload))
        assert description is not None
        assert "Hardware module: type 7, id 2." in description


def test_module_addressed_faults_drop_the_module_clause_for_narrow_payloads() -> None:
    """Verifies that a Kernel fault whose payload omits one of the two module codes reports neither code."""
    message = _kernel_data(command=2, event=KernelStatusCodes.MODULE_SETUP_ERROR, payload=np.uint8(7))
    description = _describe_kernel(message)

    assert description is not None
    assert "MODULE_SETUP_ERROR (code 2)" in description
    assert "Hardware module:" not in description


def test_invalid_protocol_fault_names_the_rejected_protocol() -> None:
    """Verifies that an invalid protocol fault resolves the rejected code to its protocol name."""
    message = _kernel_data(command=1, event=KernelStatusCodes.INVALID_MESSAGE_PROTOCOL, payload=np.uint8(4))
    description = _describe_kernel(message)

    assert description is not None
    assert "KERNEL_COMMAND (code 4)" in description


def test_keepalive_timeout_fault_reports_the_derived_timeout() -> None:
    """Verifies that a keepalive timeout fault reports the timeout value the payload carries."""
    message = _kernel_data(command=5, event=KernelStatusCodes.KEEPALIVE_TIMEOUT, payload=np.uint32(1000))
    description = _describe_kernel(message)

    assert description is not None
    assert "1000 milliseconds" in description


def test_unrecognized_kernel_code_reports_a_version_mismatch() -> None:
    """Verifies that a Kernel code outside the known range resolves to a version mismatch description."""
    description = _describe_kernel(_kernel_state(command=2, event=42))

    assert description is not None
    assert "UNRECOGNIZED (code 42)" in description
    assert "Code 42 is outside the 0 through 10 range this library resolves." in description


def test_unrecognized_service_code_reports_a_version_mismatch() -> None:
    """Verifies that a service code outside the known range resolves to a version mismatch description."""
    description = _describe_module(_module_state(command=4, event=20))

    assert description is not None
    assert "UNRECOGNIZED (code 20)" in description
    assert "Code 20 is outside the 0 through 3 range this library resolves." in description


def test_faults_arriving_without_a_payload_still_describe_the_code() -> None:
    """Verifies that a fault delivered as a state message describes the code without a payload clause."""
    description = _describe_kernel(_kernel_state(command=1, event=KernelStatusCodes.RECEPTION_ERROR))

    assert description is not None
    assert "RECEPTION_ERROR (code 3)" in description
    assert "Communication " not in description


def test_faults_carrying_a_malformed_payload_still_describe_the_code() -> None:
    """Verifies that a payload of an unexpected width is dropped rather than raising during description assembly."""
    message = _kernel_data(command=1, event=KernelStatusCodes.RECEPTION_ERROR, payload=np.uint8(53))
    description = _describe_kernel(message)

    assert description is not None
    assert "RECEPTION_ERROR (code 3)" in description
    assert "Communication " not in description


def test_custom_module_error_surfaces_the_registered_explanation() -> None:
    """Verifies that a custom module error carries the interface-supplied explanation and the message payload."""
    explanation = "The valve is not configured to emit audible tones."
    description = describe_custom_module_error(
        message=_module_data(command=4, event=56, payload=np.uint16(12)),
        controller_id=_CONTROLLER_ID,
        controller_name=_CONTROLLER_NAME,
        module_name=_MODULE_NAME,
        description=explanation,
    )

    assert explanation in description
    assert "error code 56" in description
    assert "command 4" in description
    assert _MODULE_NAME in description
    assert "Data object: 12." in description


def test_custom_module_error_omits_the_payload_clause_for_state_messages() -> None:
    """Verifies that a custom module error delivered as a state message reports no data object."""
    explanation = "The pin mode does not permit the requested command."
    description = describe_custom_module_error(
        message=_module_state(command=3, event=53),
        controller_id=_CONTROLLER_ID,
        controller_name=_CONTROLLER_NAME,
        module_name=_MODULE_NAME,
        description=explanation,
    )

    assert explanation in description
    assert "Data object" not in description
