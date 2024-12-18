"""Contains tests for the classes and methods defined in the communications module."""

import time
import multiprocessing

multiprocessing.set_start_method("spawn")

import numpy as np
import pytest
import paho.mqtt.client as mqtt
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import DataLogger

from ataraxis_transport_layer import SerialTransportLayer
from ataraxis_transport_layer.communication import (
    KernelData,
    ModuleData,
    KernelState,
    ModuleState,
    KernelCommand,
    ReceptionCode,
    Identification,
    SerialProtocols,
    KernelParameters,
    ModuleParameters,
    SerialPrototypes,
    UnityCommunication,
    OneOffModuleCommand,
    SerialCommunication,
    DequeueModuleCommand,
    RepeatedModuleCommand,
)


@pytest.fixture
def transport_layer() -> SerialTransportLayer:
    """Creates a transport layer instance in test mode."""
    return SerialTransportLayer(port="TEST", test_mode=True)


@pytest.fixture
def logger_queue(tmp_path) -> multiprocessing.Queue:
    """Creates a DataLogger instance and returns its input queue."""
    logger = DataLogger(output_directory=tmp_path)
    return logger.input_queue


def test_serial_protocols_members() -> None:
    """Verifies that SerialProtocols enum has correct values."""
    assert SerialProtocols.UNDEFINED.value == 0
    assert SerialProtocols.REPEATED_MODULE_COMMAND.value == 1
    assert SerialProtocols.ONE_OFF_MODULE_COMMAND.value == 2
    assert SerialProtocols.DEQUEUE_MODULE_COMMAND.value == 3
    assert SerialProtocols.KERNEL_COMMAND.value == 4
    assert SerialProtocols.MODULE_PARAMETERS.value == 5
    assert SerialProtocols.KERNEL_PARAMETERS.value == 6
    assert SerialProtocols.MODULE_DATA.value == 7
    assert SerialProtocols.KERNEL_DATA.value == 8
    assert SerialProtocols.MODULE_STATE.value == 9
    assert SerialProtocols.KERNEL_STATE.value == 10
    assert SerialProtocols.RECEPTION_CODE.value == 11
    assert SerialProtocols.IDENTIFICATION.value == 12


@pytest.mark.parametrize(
    "protocol,expected_value",
    [
        (SerialProtocols.UNDEFINED, 0),
        (SerialProtocols.REPEATED_MODULE_COMMAND, 1),
        (SerialProtocols.MODULE_DATA, 7),
        (SerialProtocols.IDENTIFICATION, 12),
    ],
)
def test_serial_protocols_as_uint8(protocol, expected_value) -> None:
    """Verifies the functioning of the SerialProtocols enum as_uint8() method."""
    result = protocol.as_uint8()
    assert isinstance(result, np.uint8)
    assert result == expected_value


def test_serial_protocols_comparison() -> None:
    """Verifies SerialProtocols enum comparison operations."""
    assert SerialProtocols.UNDEFINED < SerialProtocols.REPEATED_MODULE_COMMAND
    assert SerialProtocols.IDENTIFICATION > SerialProtocols.RECEPTION_CODE
    assert SerialProtocols.MODULE_DATA == SerialProtocols.MODULE_DATA
    assert SerialProtocols.MODULE_DATA != SerialProtocols.KERNEL_DATA
    assert SerialProtocols.UNDEFINED == 0


def test_serial_prototypes_members() -> None:
    """Verifies that SerialPrototypes enum has correct values."""
    assert SerialPrototypes.ONE_UNSIGNED_BYTE.value == 1
    assert SerialPrototypes.TWO_UNSIGNED_BYTES.value == 2
    assert SerialPrototypes.THREE_UNSIGNED_BYTES.value == 3
    assert SerialPrototypes.FOUR_UNSIGNED_BYTES.value == 4
    assert SerialPrototypes.ONE_UNSIGNED_LONG.value == 5
    assert SerialPrototypes.ONE_UNSIGNED_SHORT.value == 6


@pytest.mark.parametrize(
    "prototype,expected_value",
    [
        (SerialPrototypes.ONE_UNSIGNED_BYTE, 1),
        (SerialPrototypes.TWO_UNSIGNED_BYTES, 2),
        (SerialPrototypes.ONE_UNSIGNED_LONG, 5),
        (SerialPrototypes.ONE_UNSIGNED_SHORT, 6),
    ],
)
def test_serial_prototypes_as_uint8(prototype, expected_value) -> None:
    """Verifies the functioning of the SerialPrototypes enum as_uint8() method."""
    result = prototype.as_uint8()
    assert isinstance(result, np.uint8)
    assert result == expected_value


@pytest.mark.parametrize(
    "prototype,expected_type,expected_shape,expected_dtype",
    [
        (SerialPrototypes.ONE_UNSIGNED_BYTE, np.uint8, None, None),
        (SerialPrototypes.TWO_UNSIGNED_BYTES, np.ndarray, (2,), np.uint8),
        (SerialPrototypes.THREE_UNSIGNED_BYTES, np.ndarray, (3,), np.uint8),
        (SerialPrototypes.FOUR_UNSIGNED_BYTES, np.ndarray, (4,), np.uint8),
        (SerialPrototypes.ONE_UNSIGNED_LONG, np.uint32, None, None),
        (SerialPrototypes.ONE_UNSIGNED_SHORT, np.uint16, None, None),
    ],
)
def test_serial_prototypes_get_prototype(prototype, expected_type, expected_shape, expected_dtype) -> None:
    """Verifies the functioning of the SerialPrototypes enum get_prototype() method."""
    result = prototype.get_prototype()
    assert isinstance(result, expected_type)

    if expected_shape is not None:
        assert result.shape == expected_shape
        assert result.dtype == expected_dtype


@pytest.mark.parametrize(
    "code,expected_result",
    [
        (np.uint8(1), np.uint8(0)),  # ONE_UNSIGNED_BYTE
        (np.uint8(2), np.zeros(2, dtype=np.uint8)),  # TWO_UNSIGNED_BYTES
        (np.uint8(5), np.uint32(0)),  # ONE_UNSIGNED_LONG
        (np.uint8(6), np.uint16(0)),  # ONE_UNSIGNED_SHORT
        (np.uint8(255), None),  # Invalid code
    ],
)
def test_serial_prototypes_get_prototype_for_code(code, expected_result) -> None:
    """Verifies the functioning of the SerialPrototypes enum get_prototype_for_code() method."""
    # noinspection PyTypeChecker
    result = SerialPrototypes.get_prototype_for_code(code)

    if expected_result is None:
        assert result is None
    else:
        assert isinstance(result, type(expected_result))
        if isinstance(result, np.ndarray):
            assert np.array_equal(result, expected_result)
        else:
            assert result == expected_result


def test_serial_prototypes_comparison() -> None:
    """Verifies SerialPrototypes enum comparison operations."""
    assert SerialPrototypes.ONE_UNSIGNED_BYTE < SerialPrototypes.TWO_UNSIGNED_BYTES
    assert SerialPrototypes.ONE_UNSIGNED_SHORT > SerialPrototypes.ONE_UNSIGNED_BYTE
    assert SerialPrototypes.THREE_UNSIGNED_BYTES == SerialPrototypes.THREE_UNSIGNED_BYTES
    assert SerialPrototypes.THREE_UNSIGNED_BYTES != SerialPrototypes.FOUR_UNSIGNED_BYTES
    assert SerialPrototypes.ONE_UNSIGNED_BYTE == 1


def test_repeated_module_command() -> None:
    """Verifies RepeatedModuleCommand initialization and data packing."""
    cmd = RepeatedModuleCommand(
        module_type=np.uint8(1),
        module_id=np.uint8(2),
        command=np.uint8(3),
        return_code=np.uint8(4),
        noblock=np.bool_(False),
        cycle_delay=np.uint32(1000),
    )

    # Test attributes
    assert cmd.module_type == 1
    assert cmd.module_id == 2
    assert cmd.command == 3
    assert cmd.return_code == 4
    assert not cmd.noblock
    assert cmd.cycle_delay == 1000
    assert cmd.protocol_code == SerialProtocols.REPEATED_MODULE_COMMAND.as_uint8()

    # Test packed data
    assert isinstance(cmd.packed_data, np.ndarray)
    assert cmd.packed_data.dtype == np.uint8
    assert cmd.packed_data.size == 10
    assert np.array_equal(cmd.packed_data[0:6], [cmd.protocol_code, 1, 2, 4, 3, False])

    # Test repr
    expected_repr = (
        f"RepeatedModuleCommand(protocol_code={cmd.protocol_code}, module_type=1, "
        f"module_id=2, command=3, return_code=4, noblock=False, cycle_delay=1000 us)."
    )
    assert repr(cmd) == expected_repr


def test_one_off_module_command() -> None:
    """Verifies OneOffModuleCommand initialization and data packing."""
    cmd = OneOffModuleCommand(
        module_type=np.uint8(1),
        module_id=np.uint8(2),
        command=np.uint8(3),
        return_code=np.uint8(4),
        noblock=np.bool_(False),
    )

    # Test attributes
    assert cmd.module_type == 1
    assert cmd.module_id == 2
    assert cmd.command == 3
    assert cmd.return_code == 4
    assert not cmd.noblock
    assert cmd.protocol_code == SerialProtocols.ONE_OFF_MODULE_COMMAND.as_uint8()

    # Test packed data
    assert isinstance(cmd.packed_data, np.ndarray)
    assert cmd.packed_data.dtype == np.uint8
    assert cmd.packed_data.size == 6
    assert np.array_equal(cmd.packed_data, [cmd.protocol_code, 1, 2, 4, 3, False])

    # Test repr
    expected_repr = (
        f"OneOffModuleCommand(protocol_code={cmd.protocol_code}, module_type=1, "
        f"module_id=2, command=3, return_code=4, noblock=False)."
    )
    assert repr(cmd) == expected_repr


def test_dequeue_module_command() -> None:
    """Verifies DequeueModuleCommand initialization and data packing."""
    cmd = DequeueModuleCommand(module_type=np.uint8(1), module_id=np.uint8(2), return_code=np.uint8(3))

    # Test attributes
    assert cmd.module_type == 1
    assert cmd.module_id == 2
    assert cmd.return_code == 3
    assert cmd.protocol_code == SerialProtocols.DEQUEUE_MODULE_COMMAND.as_uint8()

    # Test packed data
    assert isinstance(cmd.packed_data, np.ndarray)
    assert cmd.packed_data.dtype == np.uint8
    assert cmd.packed_data.size == 4
    assert np.array_equal(cmd.packed_data, [cmd.protocol_code, 1, 2, 3])

    # Test repr
    expected_repr = (
        f"kDequeueModuleCommand(protocol_code={cmd.protocol_code}, module_type=1, " f"module_id=2, return_code=3)."
    )
    assert repr(cmd) == expected_repr


def test_kernel_command() -> None:
    """Verifies KernelCommand initialization and data packing."""
    cmd = KernelCommand(command=np.uint8(1), return_code=np.uint8(2))

    # Test attributes
    assert cmd.command == 1
    assert cmd.return_code == 2
    assert cmd.protocol_code == SerialProtocols.KERNEL_COMMAND.as_uint8()

    # Test packed data
    assert isinstance(cmd.packed_data, np.ndarray)
    assert cmd.packed_data.dtype == np.uint8
    assert cmd.packed_data.size == 3
    assert np.array_equal(cmd.packed_data, [cmd.protocol_code, 2, 1])

    # Test repr
    expected_repr = f"KernelCommand(protocol_code={cmd.protocol_code}, command=1, " f"return_code=2)."
    assert repr(cmd) == expected_repr


def test_module_parameters() -> None:
    """Verifies ModuleParameters initialization and data packing."""
    params = ModuleParameters(
        module_type=np.uint8(1),
        module_id=np.uint8(2),
        parameter_data=(np.uint8(3), np.uint16(4), np.float32(5.0)),
        return_code=np.uint8(6),
    )

    # Test attributes
    assert params.module_type == 1
    assert params.module_id == 2
    assert params.return_code == 6
    assert params.protocol_code == SerialProtocols.MODULE_PARAMETERS.as_uint8()

    # Test packed data
    assert isinstance(params.packed_data, np.ndarray)
    assert params.packed_data.dtype == np.uint8
    assert params.packed_data.size > 4  # Header size is 4 bytes
    assert np.array_equal(params.packed_data[0:4], [params.protocol_code, 1, 2, 6])

    # Test repr
    expected_repr = (
        f"ModuleParameters(protocol_code={params.protocol_code}, module_type=1, "
        f"module_id=2, return_code=6, parameter_object_size={params.parameters_size} bytes)."
    )
    assert repr(params) == expected_repr


def test_kernel_parameters() -> None:
    """Verifies KernelParameters initialization and data packing."""
    params = KernelParameters(action_lock=np.bool_(True), ttl_lock=np.bool_(False), return_code=np.uint8(1))

    # Test attributes
    assert params.action_lock
    assert not params.ttl_lock
    assert params.return_code == 1
    assert params.protocol_code == SerialProtocols.KERNEL_PARAMETERS.as_uint8()

    # Test packed data
    assert isinstance(params.packed_data, np.ndarray)
    assert params.packed_data.dtype == np.uint8
    assert params.packed_data.size == 4
    assert np.array_equal(params.packed_data, [params.protocol_code, 1, True, False])

    # Test repr
    expected_repr = (
        f"KernelParameters(protocol_code={params.protocol_code}, return_code=1, "
        f"parameter_object_size={params.parameters_size} bytes)."
    )
    assert repr(params) == expected_repr


@pytest.mark.parametrize(
    "command_class,kwargs,expected_size",
    [
        (RepeatedModuleCommand, {"module_type": np.uint8(1), "module_id": np.uint8(2), "command": np.uint8(3)}, 10),
        (OneOffModuleCommand, {"module_type": np.uint8(1), "module_id": np.uint8(2), "command": np.uint8(3)}, 6),
        (DequeueModuleCommand, {"module_type": np.uint8(1), "module_id": np.uint8(2)}, 4),
        (KernelCommand, {"command": np.uint8(1)}, 3),
    ],
)
def test_command_packed_data_sizes(command_class, kwargs, expected_size) -> None:
    """Verifies that all command classes pack data to the expected size."""
    # noinspection PyArgumentList
    cmd = command_class(**kwargs)
    assert cmd.packed_data.size == expected_size


@pytest.mark.parametrize(
    "parameter_class,kwargs",
    [
        (
            ModuleParameters,
            {"module_type": np.uint8(1), "module_id": np.uint8(2), "parameter_data": (np.uint8(3), np.uint16(4))},
        ),
        (KernelParameters, {"action_lock": np.bool_(True), "ttl_lock": np.bool_(False)}),
    ],
)
def test_parameters_packed_data_validation(parameter_class, kwargs) -> None:
    """Verifies that parameter classes correctly pack their data."""
    # noinspection PyArgumentList
    params = parameter_class(**kwargs)
    assert params.packed_data is not None
    assert params.parameters_size is not None
    assert isinstance(params.packed_data, np.ndarray)
    assert params.packed_data.dtype == np.uint8


def test_module_data_init(transport_layer) -> None:
    """Verifies ModuleData initialization."""
    data = ModuleData(transport_layer)

    assert data.protocol_code == SerialProtocols.MODULE_DATA.as_uint8()
    assert isinstance(data.message, np.ndarray)
    assert data.module_type == 0
    assert data.module_id == 0
    assert data.command == 0
    assert data.event == 0
    assert isinstance(data.data_object, np.uint8)
    assert data._transport_layer == transport_layer


def test_module_data_update(transport_layer) -> None:
    """Verifies the functioning of ModuleData update_message_data() method."""
    data = ModuleData(transport_layer)

    # Setup mock message in transport layer's reception buffer
    message = np.array([7, 1, 2, 3, 4, 1, 42], dtype=np.uint8)  # 1 is prototype for ONE_UNSIGNED_BYTE
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    data.update_message_data()

    assert data.module_type == 1
    assert data.module_id == 2
    assert data.command == 3
    assert data.event == 4
    assert data.data_object == 42
    assert np.array_equal(data.message, message)


def test_module_data_update_error(transport_layer) -> None:
    """Verifies the error handling of ModuleData update_message_data() method."""
    data = ModuleData(transport_layer)

    # Setup mock message with invalid prototype (255)
    message = np.array([7, 1, 2, 3, 4, 255, 42], dtype=np.uint8)
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    expected_error = (
        "Invalid prototype code 255 encountered when extracting the data object from "
        "the received ModuleData message sent my module 2 of type 1. All "
        "data prototype codes have to be available from the SerialPrototypes class to be resolved."
    )

    with pytest.raises(ValueError, match=error_format(expected_error)):
        data.update_message_data()


def test_kernel_data_init(transport_layer) -> None:
    """Verifies KernelData initialization."""
    data = KernelData(transport_layer)

    assert data.protocol_code == SerialProtocols.KERNEL_DATA.as_uint8()
    assert isinstance(data.message, np.ndarray)
    assert data.command == 0
    assert data.event == 0
    assert isinstance(data.data_object, np.uint8)
    assert data._transport_layer == transport_layer


def test_kernel_data_update(transport_layer) -> None:
    """Verifies the functioning of KernelData update_message_data() method."""
    data = KernelData(transport_layer)

    # Setup mock message in transport layer's reception buffer
    message = np.array([8, 1, 2, 1, 42], dtype=np.uint8)  # 1 is prototype for ONE_UNSIGNED_BYTE
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    data.update_message_data()

    assert data.command == 1
    assert data.event == 2
    assert data.data_object == 42
    assert np.array_equal(data.message, message)


def test_kernel_data_invalid_prototype(transport_layer) -> None:
    """Verifies the error handling of KernelData update_message_data() method."""
    data = KernelData(transport_layer)

    # Setup mock message with invalid prototype (255)
    message = np.array([8, 1, 2, 255, 42], dtype=np.uint8)
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    expected_error = (
        "Invalid prototype code 255 encountered when extracting the data object from "
        "the received KernelData message. All data prototype codes have to be available from the "
        "SerialPrototypes class to be resolved."
    )

    with pytest.raises(ValueError, match=error_format(expected_error)):
        data.update_message_data()


def test_module_state_init(transport_layer) -> None:
    """Verifies ModuleState initialization."""
    state = ModuleState(transport_layer)

    assert state.protocol_code == SerialProtocols.MODULE_STATE.as_uint8()
    assert isinstance(state.message, np.ndarray)
    assert state.module_type == 0
    assert state.module_id == 0
    assert state.command == 0
    assert state.event == 0
    assert state._transport_layer == transport_layer


def test_module_state_update(transport_layer) -> None:
    """Verifies the functioning of ModuleState update_message_data() method."""
    state = ModuleState(transport_layer)

    # Setup mock message in transport layer's reception buffer
    message = np.array([9, 1, 2, 3, 4], dtype=np.uint8)
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    state.update_message_data()

    assert state.module_type == 1
    assert state.module_id == 2
    assert state.command == 3
    assert state.event == 4
    assert np.array_equal(state.message, message)


def test_kernel_state_init(transport_layer) -> None:
    """Verifies KernelState initialization."""
    state = KernelState(transport_layer)

    assert state.protocol_code == SerialProtocols.KERNEL_STATE.as_uint8()
    assert isinstance(state.message, np.ndarray)
    assert state.command == 0
    assert state.event == 0
    assert state._transport_layer == transport_layer


def test_kernel_state_update(transport_layer) -> None:
    """Verifies the functioning of KernelState update_message_data() method."""
    state = KernelState(transport_layer)

    # Setup mock message in transport layer's reception buffer
    message = np.array([10, 1, 2], dtype=np.uint8)
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    state.update_message_data()

    assert state.command == 1
    assert state.event == 2
    assert np.array_equal(state.message, message)


def test_reception_code_init(transport_layer) -> None:
    """Verifies ReceptionCode initialization."""
    code = ReceptionCode(transport_layer)

    assert code.protocol_code == SerialProtocols.RECEPTION_CODE.as_uint8()
    assert isinstance(code.message, np.ndarray)
    assert code.reception_code == 0
    assert code._transport_layer == transport_layer


def test_reception_code_update(transport_layer) -> None:
    """Verifies the functioning of ReceptionCode update_message_data() method."""
    code = ReceptionCode(transport_layer)

    # Setup mock message in transport layer's reception buffer
    message = np.array([11, 42], dtype=np.uint8)
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    code.update_message_data()

    assert code.reception_code == 42
    assert np.array_equal(code.message, message)


def test_identification_init(transport_layer) -> None:
    """Verifies Identification initialization."""
    ident = Identification(transport_layer)

    assert ident.protocol_code == SerialProtocols.IDENTIFICATION.as_uint8()
    assert isinstance(ident.message, np.ndarray)
    assert ident.controller_id == 0
    assert ident._transport_layer == transport_layer


def test_identification_update(transport_layer) -> None:
    """Verifies the functioning of Identification update_message_data() method."""
    ident = Identification(transport_layer)

    # Setup mock message in transport layer's reception buffer
    message = np.array([12, 42], dtype=np.uint8)
    transport_layer._reception_buffer[: len(message)] = message
    transport_layer._bytes_in_reception_buffer = len(message)

    ident.update_message_data()

    assert ident.controller_id == 42
    assert np.array_equal(ident.message, message)


@pytest.mark.parametrize(
    "message_class,expected_repr,init_data,message_data",
    [
        (
            ModuleData,
            "ModuleData(protocol_code=7, module_type=1, module_id=2, command=3, event=4, data_object=42).",
            {},
            np.array([7, 1, 2, 3, 4, 1, 42], dtype=np.uint8),
        ),
        (
            KernelData,
            "KernelData(protocol_code=8, command=1, event=2, data_object=42).",
            {},
            np.array([8, 1, 2, 1, 42], dtype=np.uint8),
        ),
        (
            ModuleState,
            "ModuleState(module_type=1, module_id=2, command=3, event=4).",
            {},
            np.array([9, 1, 2, 3, 4], dtype=np.uint8),
        ),
        (KernelState, "KernelState(command=1, event=2).", {}, np.array([10, 1, 2], dtype=np.uint8)),
        (ReceptionCode, "ReceptionCode(reception_code=42).", {}, np.array([11, 42], dtype=np.uint8)),
        (Identification, "Identification(controller_id=42).", {}, np.array([12, 42], dtype=np.uint8)),
    ],
)
def test_message_repr(transport_layer, message_class, expected_repr, init_data, message_data) -> None:
    """Verifies string representation of message classes."""
    # noinspection PyArgumentList
    message = message_class(transport_layer, **init_data)

    # Setup mock data and update message
    transport_layer._reception_buffer[: len(message_data)] = message_data
    transport_layer._bytes_in_reception_buffer = len(message_data)
    message.update_message_data()

    assert repr(message) == expected_repr


def test_serial_communication_init_and_repr(logger_queue) -> None:
    """Verifies SerialCommunication initialization and string representation."""
    comm = SerialCommunication(usb_port="TEST", logger_queue=logger_queue, source_id=np.uint8(1), test_mode=True)

    # Test initialization
    assert comm._transport_layer is not None
    assert isinstance(comm._module_data, ModuleData)
    assert isinstance(comm._kernel_data, KernelData)
    assert isinstance(comm._module_state, ModuleState)
    assert isinstance(comm._kernel_state, KernelState)
    assert isinstance(comm._identification, Identification)
    assert isinstance(comm._reception_code, ReceptionCode)
    assert comm._source_id == 1
    assert comm._usb_port == "TEST"
    assert not comm._verbose

    # Test string representation
    expected_repr = "SerialCommunication(usb_port=TEST, source_id=1, verbose=False)."
    assert repr(comm) == expected_repr


def test_serial_communication_send_message(logger_queue) -> None:
    """Verifies the functionality of the SerialCommunication send_message() method."""
    comm = SerialCommunication(
        usb_port="TEST", logger_queue=logger_queue, source_id=np.uint8(1), verbose=True, test_mode=True
    )

    # Creates the test message
    message = KernelCommand(command=np.uint8(1))

    # Sends the message
    # noinspection PyTypeChecker
    comm.send_message(message)

    # Verifies data was written to transport layer
    assert comm._transport_layer._transmission_buffer[:3].tobytes() == message.packed_data.tobytes()


@pytest.mark.parametrize(
    "message_data,expected_type,expected_values",
    [
        # ModuleData message (protocol code 7)
        (
            np.array([7, 1, 2, 3, 4, 1, 42], dtype=np.uint8),
            ModuleData,
            {
                "module_type": 1,
                "module_id": 2,
                "command": 3,
                "event": 4,
                "data_object": 42,
            },
        ),
        # KernelData message (protocol code 8)
        (
            np.array([8, 1, 2, 1, 42], dtype=np.uint8),
            KernelData,
            {
                "command": 1,
                "event": 2,
                "data_object": 42,
            },
        ),
        # ModuleState message (protocol code 9)
        (
            np.array([9, 1, 2, 3, 4], dtype=np.uint8),
            ModuleState,
            {
                "module_type": 1,
                "module_id": 2,
                "command": 3,
                "event": 4,
            },
        ),
        # KernelState message (protocol code 10)
        (
            np.array([10, 1, 2], dtype=np.uint8),
            KernelState,
            {
                "command": 1,
                "event": 2,
            },
        ),
        # ReceptionCode message (protocol code 11)
        (
            np.array([11, 42], dtype=np.uint8),
            ReceptionCode,
            {
                "reception_code": 42,
            },
        ),
        # Identification message (protocol code 12)
        (
            np.array([12, 42], dtype=np.uint8),
            Identification,
            {
                "controller_id": 42,
            },
        ),
    ],
)
def test_serial_communication_receive_message(logger_queue, message_data, expected_type, expected_values) -> None:
    """Verifies the functioning of SerialCommunication receive_message(0 method."""
    # Initialize communication
    comm = SerialCommunication(
        usb_port="TEST", logger_queue=logger_queue, source_id=np.uint8(1), verbose=True, test_mode=True
    )

    # First verifies that the method returns None when there is no data to receive.
    assert comm.receive_message() is None

    # Next, transforms the tested payload into the message format that can be received via the TransportLayer. This is
    # done by first 'sending' it and then using the 'sent' (well-formatted) data for the reception test.
    comm._transport_layer.write_data(message_data)
    comm._transport_layer.send_data()
    comm._transport_layer._port.rx_buffer = comm._transport_layer._port.tx_buffer

    # Receives and verifies the received data
    received = comm.receive_message()
    assert isinstance(received, expected_type)
    for attr, value in expected_values.items():
        assert getattr(received, attr) == value

    # Verifies message array matches original data
    assert np.array_equal(received.message, message_data)


def test_serial_communication_receive_message_error(logger_queue) -> None:
    """Verifies the error handling of the SerialCommunication receive_data() method."""
    comm = SerialCommunication(usb_port="TEST", logger_queue=logger_queue, source_id=np.uint8(1), test_mode=True)

    # Test receiving the message with invalid protocol code
    message_data = np.array([255, 1, 2], dtype=np.uint8)  # Invalid protocol code

    # First 'sends' the message to the SerialMock class, which COBS-encodes and CRC-stamps the message
    comm._transport_layer.write_data(message_data)
    comm._transport_layer.send_data()

    # Next, transfers the message from the tx_buffer to the rx_buffer. The message then can be 'received' and it now
    # has the correct format to pass TransportLayer verification steps that ensure message integrity.
    comm._transport_layer._port.rx_buffer = comm._transport_layer._port.tx_buffer

    # Ensures that a message with an invalid protocol raises a ValueError
    message = (
        f"Invalid protocol code {255} encountered when attempting to parse a message received from the "
        f"microcontroller. All incoming messages have to use one of the valid incoming message protocol codes "
        f"available from the SerialProtocols enumeration."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        comm.receive_message()


# Test configuration
BROKER_IP = "127.0.0.1"
BROKER_PORT = 1883
TEST_TOPICS = ("test/topic1", "test/topic2")


def broker_available() -> bool:
    """Checks if MQTT broker is available at test IP/port.

    This fixture should be used with pytest.mark.skipif to skip tests when broker is not available.
    """
    try:
        UnityCommunication(ip=BROKER_IP, port=BROKER_PORT)
        return True
    except Exception:
        return False


def test_unity_communication_init_and_repr() -> None:
    """Verifies the successful initialization of UnityCommunication __init__() method and the __repr__() method."""
    # Skips the test if the test MQTT broker is not available
    if not broker_available():
        pytest.skip(f"Skipping this test as it requires an MQTT broker at ip {BROKER_IP} and port {BROKER_PORT}.")

    comm = UnityCommunication(ip=BROKER_IP, port=BROKER_PORT, monitored_topics=TEST_TOPICS)

    # Verifies class representation string
    expected_repr = (
        f"UnityCommunication(broker_ip={BROKER_IP}, socket_port={BROKER_PORT}, connected={False}, "
        f"subscribed_topics={TEST_TOPICS}"
    )
    assert repr(comm) == expected_repr


def test_unity_communication_init_error() -> None:
    """Verifies the error handling behavior of UnityCommunication __init__() method."""
    message = (
        f"Unable to initialize UnityCommunication class instance. Failed to connect to MQTT broker at "
        f"{BROKER_IP}:{1880}. This likely indicates that the broker is not running or that there is an "
        f"issue with the provided IP and socket port."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        # Invalid port
        UnityCommunication(ip=BROKER_IP, port=1880, monitored_topics=TEST_TOPICS)


@pytest.mark.xdist_group(name="group1")
def test_unity_communication_send_receive() -> None:
    """Verifies bidirectional communication between UnityCommunication and simulated Unity client."""
    # Skips the test if the test MQTT broker is not available
    if not broker_available():
        pytest.skip(f"Skipping this test as it requires an MQTT broker at ip {BROKER_IP} and port {BROKER_PORT}.")

    unity_comm = UnityCommunication(ip=BROKER_IP, port=BROKER_PORT, monitored_topics=TEST_TOPICS)
    unity_comm.connect()
    unity_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # type: ignore
    unity_client.connect(BROKER_IP, BROKER_PORT)
    unity_client.loop_start()

    # Stores received messages in this list
    received_messages = []

    # Creates Unity client receiver function
    def on_message(_client, _userdata, message):
        received_messages.append((message.topic, message.payload))

    unity_client.on_message = on_message

    # Subscribes the Unity client to the test topic
    test_topic = "test/output"
    unity_client.subscribe(test_topic)
    time.sleep(0.1)  # Allow subscription to establish

    # Tests sending data from UnityCommunication to Unity
    test_data = [
        ("test message", str),
        (b"binary data", bytes),
        ("3.14", str),
    ]

    for data, data_type in test_data:
        unity_comm.send_data(test_topic, data)
        time.sleep(0.1)  # Allows the message to be received

        # Verifies Unity client received the message
        assert len(received_messages) > 0
        topic, payload = received_messages[-1]
        assert topic == test_topic
        if data_type == str:
            assert payload.decode() == data
        else:
            assert payload == data

    # Tests sending data from Unity to UnityCommunication
    for topic in TEST_TOPICS:
        test_message = f"Unity message for {topic}"
        unity_client.publish(topic, test_message)
        time.sleep(0.1)  # Allows the message to be received

        # Verifies UnityCommunication received the message
        assert unity_comm.has_data
        received = unity_comm.get_data()
        assert received is not None
        received_topic, received_payload = received
        assert received_topic == topic
        assert received_payload.decode() == test_message


def test_unity_communication_send_receive_errors() -> None:
    """Verifies the error handling behavior of UnityCommunication send_data() and get_data() methods."""

    # Both methods raise RuntimeErrors if they are called when the class is not connected to the MQTT broker.
    unity_comm = UnityCommunication(ip=BROKER_IP, port=BROKER_PORT, monitored_topics=TEST_TOPICS)
    message = (
        f"Cannot send data to the MQTT broker at {BROKER_IP}:{BROKER_PORT} via the UnityCommunication instance. "
        f"The UnityCommunication instance is not connected to the MQTT broker, call connect() method before "
        f"sending data."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        unity_comm.send_data("test/ topic1")
    message = (
        f"Cannot get data from the MQTT broker at {BROKER_IP}:{BROKER_PORT} via the UnityCommunication instance. "
        f"The UnityCommunication instance is not connected to the MQTT broker, call connect() method before "
        f"sending data."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        unity_comm.get_data()


@pytest.mark.xdist_group(name="group1")
def test_unity_communication_queue_management() -> None:
    """Verifies that UnityCommunication message queue properly handles multiple messages."""
    # Skips the test if the test MQTT broker is not available
    if not broker_available():
        pytest.skip(f"Skipping this test as it requires an MQTT broker at ip {BROKER_IP} and port {BROKER_PORT}.")

    unity_comm = UnityCommunication(ip=BROKER_IP, port=BROKER_PORT, monitored_topics=TEST_TOPICS)
    unity_comm.connect()
    unity_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # type: ignore
    unity_client.connect(BROKER_IP, BROKER_PORT)
    unity_client.loop_start()

    # Sends multiple messages from Unity
    messages = [
        (TEST_TOPICS[0], "message1"),
        (TEST_TOPICS[0], "message2"),
        (TEST_TOPICS[1], "message3"),
    ]

    for topic, msg in messages:
        unity_client.publish(topic, msg)
        time.sleep(0.1)  # Allows the message to be received

    # Verifies all messages are received in order
    received_messages = []
    while unity_comm.has_data:
        data = unity_comm.get_data()
        assert data is not None
        received_messages.append((data[0], data[1].decode()))

    assert received_messages == messages


@pytest.mark.xdist_group(name="group1")
def test_unity_communication_reconnection() -> None:
    """Verifies UnityCommunication disconnecting and reconnecting while maintaining subscriptions."""
    # Skips the test if the test MQTT broker is not available
    if not broker_available():
        pytest.skip(f"Skipping this test as it requires an MQTT broker at ip {BROKER_IP} and port {BROKER_PORT}.")

    unity_comm = UnityCommunication(ip=BROKER_IP, port=BROKER_PORT, monitored_topics=TEST_TOPICS)
    unity_comm.connect()
    unity_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # type: ignore
    unity_client.connect(BROKER_IP, BROKER_PORT)
    unity_client.loop_start()

    # Sends the initial message
    test_message = "before disconnect"
    unity_client.publish(TEST_TOPICS[0], test_message)
    time.sleep(0.1)

    # Verifies that the message was received
    assert unity_comm.has_data
    data = unity_comm.get_data()
    assert data is not None
    assert data[1].decode() == test_message

    # Disconnects and reconnects. Also verifies that calling each method the second time has no effect
    unity_comm.disconnect()
    unity_comm.disconnect()
    unity_comm.connect()
    unity_comm.connect()
    time.sleep(0.1)  # Allows reconnection to be established

    # Sends the new message
    test_message = "after reconnect"
    unity_client.publish(TEST_TOPICS[0], test_message)
    time.sleep(0.1)

    # Verifies the message was received after reconnection
    assert unity_comm.has_data
    data = unity_comm.get_data()
    assert data is not None
    assert data[1].decode() == test_message

    # Verifies that if there is no data to receive, get_data returns None
    assert unity_comm.get_data() is None


@pytest.mark.xdist_group(name="group1")
def test_unity_communication_large_message() -> None:
    """Verifies UnityCommunication handling of larger messages."""
    # Skips the test if the test MQTT broker is not available
    if not broker_available():
        pytest.skip(f"Skipping this test as it requires an MQTT broker at ip {BROKER_IP} and port {BROKER_PORT}.")

    unity_comm = UnityCommunication(ip=BROKER_IP, port=BROKER_PORT, monitored_topics=TEST_TOPICS)
    unity_comm.connect()
    unity_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # type: ignore
    unity_client.connect(BROKER_IP, BROKER_PORT)
    unity_client.loop_start()

    # Creates a large test message (100KB)
    large_message = b"x" * 100000

    # Sends from UnityCommunication to Unity
    test_topic = "test/large"
    unity_client.subscribe(test_topic)
    time.sleep(0.1)

    received_large_message = None

    def on_message(_client, _userdata, message):
        nonlocal received_large_message
        received_large_message = message.payload

    unity_client.on_message = on_message

    unity_comm.send_data(test_topic, large_message)
    time.sleep(0.2)  # Waits a bit longer for the larger message

    assert received_large_message == large_message

    # Sends from Unity to UnityCommunication
    unity_client.publish(TEST_TOPICS[0], large_message)
    time.sleep(0.2)

    assert unity_comm.has_data
    data = unity_comm.get_data()
    assert data is not None
    assert data[1] == large_message
