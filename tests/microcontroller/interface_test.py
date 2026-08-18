"""Contains tests for the classes and functions provided by the microcontroller/interface.py module."""

from __future__ import annotations

import os
from queue import Queue
import pickle
from typing import TYPE_CHECKING, Any, Self
import itertools
from threading import Thread
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from concurrent.futures import Future

import numpy as np
import pytest
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import DataLogger, SharedMemoryArray
from serial.tools.list_ports_common import ListPortInfo

from ataraxis_communication_interface.communication import (
    ModuleData,
    ModuleState,
    KernelCommand,
    ModuleParameters,
    OneOffModuleCommand,
    SerialCommunication,
    DequeueModuleCommand,
    RepeatedModuleCommand,
)
from ataraxis_communication_interface.microcontroller import interface
from ataraxis_communication_interface.microcontroller.interface import (
    ModuleInterface,
    MicroControllerInterface,
    evaluate_port,
    discover_microcontrollers,
)
from ataraxis_communication_interface.microcontroller.status_codes import (
    MAXIMUM_CUSTOM_STATUS_CODE,
    MINIMUM_CUSTOM_STATUS_CODE,
    KernelStatusCodes,
    ModuleStatusCodes,
    KernelCommandCodes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Generator

    from numpy.typing import NDArray

_CONTROLLER_ID: np.uint8 = np.uint8(1)
"""Stores the identifier code of the microcontroller the tests below interface with."""

_CONTROLLER_NAME: str = "test_controller"
"""Stores the human-readable name of the microcontroller the tests below interface with."""

_MODULE_TYPE: np.uint8 = np.uint8(1)
"""Stores the type code of the hardware module the tests below interface with."""

_MODULE_ID: np.uint8 = np.uint8(2)
"""Stores the identifier code of the hardware module the tests below interface with."""

_MODULE_NAME: str = "test_module"
"""Stores the human-readable name of the hardware module the tests below interface with."""

_BUFFER_SIZE: int = 300
"""Stores the serial buffer size the tests below advertise for the interfaced microcontroller."""

_DATA_CODE: np.uint8 = np.uint8(60)
"""Stores the custom event code the test module interface registers for online data processing."""

_ERROR_CODE: np.uint8 = np.uint8(61)
"""Stores the custom event code the test module interface registers as a runtime error."""

_ERROR_EXPLANATION: str = "The test module reported an unrecoverable hardware fault."
"""Stores the explanation the test module interface registers for its custom error code."""

_ONE_UINT8_PROTOTYPE: int = 2
"""Stores the SerialPrototypes code for a single uint8 value, which every data message below carries."""

_JOIN_TIMEOUT: int = 30
"""Stores the time, in seconds, the tests below wait for a helper process or thread to terminate before
giving up."""

_IDENTIFICATION_TIMEOUT: int = 20
"""Stores the shortened identification timeout, in milliseconds, the verification tests wait out in full."""

_CYCLE_DELAY: int = 3
"""Stores the time, in milliseconds, the scripted terminator array spends on each communication cycle it grants."""

_ARRAY_COUNTER: itertools.count[int] = itertools.count()
"""Numbers the shared memory buffers the tests create, keeping every buffer name unique within the test process."""


# Stays above the tests: test_microcontroller_interface_initialization_invalid_module_interfaces evaluates its
# parametrize argvalue at import, so a lower class raises NameError at collection.
class _RecordingModule(ModuleInterface):
    """Records every remote asset and data processing call the communication cycle makes on the interface.

    Notes:
        Each processed message is recorded as the codes it carries rather than as the message object, because the
        SerialCommunication instance reuses one object per message type and overwrites it on every reception.

    Args:
        module_type: The code that identifies the type of the interfaced module.
        module_id: The code that identifies the specific interfaced module instance.
        name: The human-readable name of the interfaced module.
        error_codes: The mapping of the module's error codes to their explanations.
        data_codes: The set of the module's event codes that require online processing.

    Attributes:
        initializations: Counts the calls the communication cycle made to initialize_remote_assets().
        terminations: Counts the calls the communication cycle made to terminate_remote_assets().
        processed: Stores the command and event code pair of every message routed to process_received_data().
    """

    def __init__(
        self,
        module_type: np.uint8 = _MODULE_TYPE,
        module_id: np.uint8 = _MODULE_ID,
        name: str = _MODULE_NAME,
        error_codes: dict[np.uint8, str] | None = None,
        data_codes: set[np.uint8] | None = None,
    ) -> None:
        super().__init__(
            module_type=module_type,
            module_id=module_id,
            name=name,
            error_codes=error_codes,
            data_codes=data_codes,
        )
        self.initializations = 0
        self.terminations = 0
        self.processed: list[tuple[int, int]] = []

    def initialize_remote_assets(self) -> None:
        """Records that the communication cycle requested the interface's remote assets."""
        self.initializations += 1

    def terminate_remote_assets(self) -> None:
        """Records that the communication cycle released the interface's remote assets."""
        self.terminations += 1

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Records the command and event codes of the message routed to this interface."""
        self.processed.append((int(message.command), int(message.event)))


@pytest.fixture
def logger(tmp_path_factory: pytest.TempPathFactory) -> DataLogger:
    """Creates a DataLogger instance whose output directory receives the microcontroller manifest."""
    temporary_directory = tmp_path_factory.mktemp("logger_data")
    return DataLogger(output_directory=temporary_directory, instance_name="test_logger")


@pytest.fixture
def logger_queue() -> MPQueue:  # type: ignore[type-arg]
    """Creates the multiprocessing queue the SerialCommunication instances log their message data to."""
    return MPQueue()


@pytest.fixture
def terminator_array() -> Generator[SharedMemoryArray, None, None]:
    """Creates the shared memory array that carries the communication process runtime flags."""
    array = SharedMemoryArray.create_array(
        name=f"aci_test_{os.getpid()}_{next(_ARRAY_COUNTER)}",
        prototype=np.zeros(shape=2, dtype=np.uint8),
        exists_ok=True,
    )
    yield array
    array.destroy()


@pytest.fixture
def terminated_process() -> Process:
    """Creates a communication process stand-in that has already run to completion."""
    process = Process(target=int, daemon=True)
    process.start()
    process.join(timeout=_JOIN_TIMEOUT)
    return process


@pytest.fixture
def terminated_thread() -> Thread:
    """Creates a watchdog thread stand-in that has already run to completion."""
    thread = Thread(target=int, daemon=True)
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT)
    return thread


@pytest.fixture
def short_identification_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shortens the microcontroller identification timeout, which the verification stage always waits out in full."""
    monkeypatch.setattr(interface._RuntimeParameters.MICROCONTROLLER_ID_TIMEOUT, "_value_", _IDENTIFICATION_TIMEOUT)


@pytest.fixture
def initialized_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reports a completed configuration verification, which the dedicated verification tests exercise in full."""

    def verify(terminator_array: Any, **_kwargs: Any) -> None:
        """Records the completed initialization the way the verified configuration check does."""
        terminator_array[1] = 1

    monkeypatch.setattr(MicroControllerInterface, "_verify_microcontroller_communication", verify)


def test_module_interface_initialization() -> None:
    """Verifies the ModuleInterface initialization and the values it derives from the input codes."""
    module = _RecordingModule(data_codes={_DATA_CODE}, error_codes={_ERROR_CODE: _ERROR_EXPLANATION})

    assert module.module_type == _MODULE_TYPE
    assert module.module_id == _MODULE_ID
    assert module.type_id == np.uint16((int(_MODULE_TYPE) << 8) | int(_MODULE_ID))
    assert module.name == _MODULE_NAME
    assert module.data_codes == {_DATA_CODE}
    assert module.error_codes == {_ERROR_CODE: _ERROR_EXPLANATION}


def test_module_interface_initialization_without_optional_codes() -> None:
    """Verifies that a ModuleInterface built without event codes stores empty code collections."""
    module = _RecordingModule()

    assert module.data_codes == set()
    assert module.error_codes == {}


def test_module_interface_type_id_is_position_aware() -> None:
    """Verifies that inverting the type and id codes of a module produces a different combined code."""
    assert _RecordingModule(module_type=np.uint8(4), module_id=np.uint8(5)).type_id != (
        _RecordingModule(module_type=np.uint8(5), module_id=np.uint8(4)).type_id
    )


@pytest.mark.parametrize("module_type", [0, 1, np.uint16(1), np.uint8(0)])
def test_module_interface_initialization_invalid_module_type(module_type: Any) -> None:
    """Verifies that ModuleInterface rejects a module type code that is not a numpy uint8 inside the valid byte
    range."""
    message = (
        f"Unable to initialize the ModuleInterface instance for module {_MODULE_ID} of type {module_type}. "
        f"Expected an unsigned integer value between 1 and 255 for 'module_type' argument, but encountered "
        f"{module_type} of type {type(module_type).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        _RecordingModule(module_type=module_type)


@pytest.mark.parametrize("module_id", [0, 2, np.uint16(2), np.uint8(0)])
def test_module_interface_initialization_invalid_module_id(module_id: Any) -> None:
    """Verifies that ModuleInterface rejects a module id code that is not a numpy uint8 inside the valid byte
    range."""
    message = (
        f"Unable to initialize the ModuleInterface instance for module {module_id} of type {_MODULE_TYPE}. "
        f"Expected an unsigned integer value between 1 and 255 for 'module_id' argument, but encountered "
        f"{module_id} of type {type(module_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        _RecordingModule(module_id=module_id)


@pytest.mark.parametrize(
    "error_codes",
    [
        [_ERROR_CODE],
        {_ERROR_CODE: 5},
        {_ERROR_CODE: ""},
        {61: _ERROR_EXPLANATION},
    ],
)
def test_module_interface_initialization_invalid_error_codes(error_codes: Any) -> None:
    """Verifies that ModuleInterface rejects an error code mapping that is not uint8 codes to non-empty strings."""
    message = (
        f"Unable to initialize the ModuleInterface instance for module {_MODULE_ID} of type {_MODULE_TYPE}. "
        f"Expected a dictionary mapping numpy uint8 values to non-empty strings or None for 'error_codes' "
        f"argument, but encountered {error_codes} of type {type(error_codes).__name__} and / or at least one "
        f"invalid key or value."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        _RecordingModule(error_codes=error_codes)


@pytest.mark.parametrize("data_codes", [[_DATA_CODE], {60}, (_DATA_CODE,)])
def test_module_interface_initialization_invalid_data_codes(data_codes: Any) -> None:
    """Verifies that ModuleInterface rejects a data code collection that is not a set of uint8 values."""
    message = (
        f"Unable to initialize the ModuleInterface instance for module {_MODULE_ID} of type {_MODULE_TYPE}. "
        f"Expected a set of numpy uint8 values or None for 'data_codes' argument, but encountered "
        f"{data_codes} of type {type(data_codes).__name__} and / or at least one non-uint8 item."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        _RecordingModule(data_codes=data_codes)


@pytest.mark.parametrize("code", [np.uint8(MINIMUM_CUSTOM_STATUS_CODE - 1), np.uint8(MAXIMUM_CUSTOM_STATUS_CODE + 1)])
def test_module_interface_initialization_error_code_outside_custom_range(code: np.uint8) -> None:
    """Verifies that ModuleInterface rejects an error code outside the range reserved for custom modules."""
    message = (
        f"Unable to initialize the ModuleInterface instance for module {_MODULE_ID} of type {_MODULE_TYPE}. "
        f"Expected every code in the 'error_codes' argument to fall between "
        f"{MINIMUM_CUSTOM_STATUS_CODE} and {MAXIMUM_CUSTOM_STATUS_CODE}, which is the event code range "
        f"the microcontroller reserves for custom hardware modules, but encountered {[int(code)]}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        _RecordingModule(error_codes={code: _ERROR_EXPLANATION})


@pytest.mark.parametrize("code", [np.uint8(MINIMUM_CUSTOM_STATUS_CODE - 1), np.uint8(MAXIMUM_CUSTOM_STATUS_CODE + 1)])
def test_module_interface_initialization_data_code_outside_custom_range(code: np.uint8) -> None:
    """Verifies that ModuleInterface rejects a data code outside the range reserved for custom modules."""
    message = (
        f"Unable to initialize the ModuleInterface instance for module {_MODULE_ID} of type {_MODULE_TYPE}. "
        f"Expected every code in the 'data_codes' argument to fall between "
        f"{MINIMUM_CUSTOM_STATUS_CODE} and {MAXIMUM_CUSTOM_STATUS_CODE}, which is the event code range "
        f"the microcontroller reserves for custom hardware modules, but encountered {[int(code)]}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        _RecordingModule(data_codes={code})


@pytest.mark.parametrize("name", ["", 5, None])
def test_module_interface_initialization_invalid_name(name: Any) -> None:
    """Verifies that ModuleInterface rejects a name that is not a non-empty string."""
    message = (
        f"Unable to initialize the ModuleInterface instance for module {_MODULE_ID} of type {_MODULE_TYPE}. "
        f"Expected a non-empty string for the 'name' argument, but encountered {name!r} of type "
        f"{type(name).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        _RecordingModule(name=name)


def test_module_interface_pickling_excludes_the_message_caches() -> None:
    """Verifies that pickling a ModuleInterface drops the caches that do not support it."""
    module = _RecordingModule(data_codes={_DATA_CODE})

    restored = pickle.loads(pickle.dumps(module))  # noqa: S301  # The payload is the interface built one line above.

    assert restored.module_type == _MODULE_TYPE
    assert restored.module_id == _MODULE_ID
    assert restored.data_codes == {_DATA_CODE}
    assert restored._create_command_message is None
    assert restored._create_parameters_message is None


def test_module_interface_send_command_without_repetition() -> None:
    """Verifies that send_command() queues a one-off command when it is given no repetition delay."""
    module = _RecordingModule()
    input_queue: Queue[Any] = Queue()
    module.set_input_queue(input_queue=input_queue)

    module.send_command(command=np.uint8(1), noblock=np.bool_(False))

    command = input_queue.get_nowait()
    assert isinstance(command, OneOffModuleCommand)
    assert command.module_type == _MODULE_TYPE
    assert command.module_id == _MODULE_ID
    assert command.command == 1


def test_module_interface_send_command_with_repetition() -> None:
    """Verifies that send_command() queues a repeated command when it is given a repetition delay."""
    module = _RecordingModule()
    input_queue: Queue[Any] = Queue()
    module.set_input_queue(input_queue=input_queue)

    module.send_command(command=np.uint8(1), noblock=np.bool_(True), repetition_delay=np.uint32(1000))

    command = input_queue.get_nowait()
    assert isinstance(command, RepeatedModuleCommand)
    assert command.cycle_delay == 1000


def test_module_interface_send_command_reuses_the_cached_message() -> None:
    """Verifies that repeating a command reuses the message object cached for its parameters."""
    module = _RecordingModule()
    input_queue: Queue[Any] = Queue()
    module.set_input_queue(input_queue=input_queue)

    module.send_command(command=np.uint8(1), noblock=np.bool_(False))
    module.send_command(command=np.uint8(1), noblock=np.bool_(False))

    assert input_queue.get_nowait() is input_queue.get_nowait()


def test_module_interface_send_command_before_binding() -> None:
    """Verifies that send_command() refuses to build a message before the interface is bound to a queue."""
    module = _RecordingModule()
    message = (
        f"Unable to send the command message to the module {_MODULE_ID} of type "
        f"{_MODULE_TYPE}. Use the module interface instance to initialize the MicroControllerInterface "
        f"instance to enable constructing and sending messages to the microcontroller. Note: at this time only "
        f"the main runtime process can construct and send messages to the microcontroller."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        module.send_command(command=np.uint8(1), noblock=np.bool_(False))


def test_module_interface_send_parameters() -> None:
    """Verifies that send_parameters() queues the parameter message built from the input tuple."""
    module = _RecordingModule()
    input_queue: Queue[Any] = Queue()
    module.set_input_queue(input_queue=input_queue)

    module.send_parameters(parameter_data=(np.uint8(1), np.float32(2.5)))

    parameters = input_queue.get_nowait()
    assert isinstance(parameters, ModuleParameters)
    assert parameters.module_type == _MODULE_TYPE
    assert parameters.module_id == _MODULE_ID


def test_module_interface_send_parameters_before_binding() -> None:
    """Verifies that send_parameters() refuses to build a message before the interface is bound to a queue."""
    module = _RecordingModule()
    message = (
        f"Unable to send the runtime parameters update message to the module {_MODULE_ID} of type "
        f"{_MODULE_TYPE}. Use the module interface instance to initialize the MicroControllerInterface "
        f"instance to enable constructing and sending messages to the microcontroller. Note: at this time only "
        f"the main runtime process can construct and send messages to the microcontroller."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        module.send_parameters(parameter_data=(np.uint8(1),))


def test_module_interface_reset_command_queue() -> None:
    """Verifies that reset_command_queue() queues the interface's pre-built dequeue command."""
    module = _RecordingModule()
    input_queue: Queue[Any] = Queue()
    module.set_input_queue(input_queue=input_queue)

    module.reset_command_queue()

    assert isinstance(input_queue.get_nowait(), DequeueModuleCommand)


def test_module_interface_reset_command_queue_before_binding() -> None:
    """Verifies that reset_command_queue() refuses to send before the interface is bound to a queue."""
    module = _RecordingModule()
    message = (
        f"Unable to send the dequeue command message to the module {_MODULE_ID} of type "
        f"{_MODULE_TYPE}. Use the module interface instance to initialize and start the "
        f"MicroControllerInterface instance to enable constructing and sending messages to the microcontroller."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        module.reset_command_queue()


def test_microcontroller_interface_initialization(logger: DataLogger) -> None:
    """Verifies the MicroControllerInterface initialization and the manifest entry it writes."""
    module = _RecordingModule()
    controller = MicroControllerInterface(
        controller_id=_CONTROLLER_ID,
        data_logger=logger,
        module_interfaces=(module,),
        buffer_size=_BUFFER_SIZE,
        port="TEST",
        name=_CONTROLLER_NAME,
    )

    assert controller.controller_id == _CONTROLLER_ID
    assert controller.name == _CONTROLLER_NAME
    assert controller.modules == (module,)
    assert module._input_queue is not None
    assert (logger.output_directory / "microcontroller_manifest.yaml").exists()


def test_microcontroller_interface_reset_controller(logger: DataLogger) -> None:
    """Verifies that reset_controller() queues the pre-packaged Kernel reset command."""
    controller = MicroControllerInterface(
        controller_id=_CONTROLLER_ID,
        data_logger=logger,
        module_interfaces=(_RecordingModule(),),
        buffer_size=_BUFFER_SIZE,
        port="TEST",
        name=_CONTROLLER_NAME,
    )

    controller.reset_controller()

    command = controller._input_queue.get()
    assert isinstance(command, KernelCommand)
    assert command.command == KernelCommandCodes.RESET_CONTROLLER.value


@pytest.mark.parametrize("controller_id", [0, 1, np.uint16(1), np.uint8(0)])
def test_microcontroller_interface_initialization_invalid_controller_id(logger: DataLogger, controller_id: Any) -> None:
    """Verifies that MicroControllerInterface rejects a controller id that is not a numpy uint8 inside the valid
    byte range."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance. Expected an unsigned integer value "
        f"between 1 and 255 for the 'controller_id' argument, but encountered {controller_id} of type "
        f"{type(controller_id).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=controller_id,
            data_logger=logger,
            module_interfaces=(_RecordingModule(),),
            buffer_size=_BUFFER_SIZE,
            port="TEST",
            name=_CONTROLLER_NAME,
        )


@pytest.mark.parametrize("module_interfaces", [(), [_RecordingModule()], None])
def test_microcontroller_interface_initialization_invalid_module_interfaces(
    logger: DataLogger, module_interfaces: Any
) -> None:
    """Verifies that MicroControllerInterface rejects a module interface collection that is not a non-empty tuple."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
        f"{_CONTROLLER_ID}. Expected a non-empty tuple of ModuleInterface instances for "
        f"'module_interfaces' argument, but encountered {module_interfaces} of type "
        f"{type(module_interfaces).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=_CONTROLLER_ID,
            data_logger=logger,
            module_interfaces=module_interfaces,
            buffer_size=_BUFFER_SIZE,
            port="TEST",
            name=_CONTROLLER_NAME,
        )


def test_microcontroller_interface_initialization_invalid_module_interface_item(logger: DataLogger) -> None:
    """Verifies that MicroControllerInterface rejects a module interface tuple holding a foreign object."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
        f"{_CONTROLLER_ID}. All items in 'module_interfaces' tuple must be ModuleInterface instances."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=_CONTROLLER_ID,
            data_logger=logger,
            module_interfaces=(_RecordingModule(), "module"),
            buffer_size=_BUFFER_SIZE,
            port="TEST",
            name=_CONTROLLER_NAME,
        )


def test_microcontroller_interface_initialization_invalid_data_logger() -> None:
    """Verifies that MicroControllerInterface rejects a data logger that is not a DataLogger instance."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
        f"{_CONTROLLER_ID}. Expected an initialized DataLogger instance for 'data_logger' argument, but "
        f"encountered None of type NoneType."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=_CONTROLLER_ID,
            data_logger=None,
            module_interfaces=(_RecordingModule(),),
            buffer_size=_BUFFER_SIZE,
            port="TEST",
            name=_CONTROLLER_NAME,
        )


@pytest.mark.parametrize("buffer_size", [8, 300.0, None])
def test_microcontroller_interface_initialization_invalid_buffer_size(logger: DataLogger, buffer_size: Any) -> None:
    """Verifies that MicroControllerInterface rejects a serial buffer size that is not an integer at or above the
    transport layer's minimum."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
        f"{_CONTROLLER_ID}. Expected an integer value of at least {interface._MINIMUM_SERIAL_BUFFER_SIZE} for the "
        f"'buffer_size' argument, but encountered {buffer_size} of type {type(buffer_size).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=_CONTROLLER_ID,
            data_logger=logger,
            module_interfaces=(_RecordingModule(),),
            buffer_size=buffer_size,
            port="TEST",
            name=_CONTROLLER_NAME,
        )


@pytest.mark.parametrize("keepalive_interval", [-1, 1.5, None])
def test_microcontroller_interface_initialization_invalid_keepalive_interval(
    logger: DataLogger, keepalive_interval: Any
) -> None:
    """Verifies that MicroControllerInterface rejects a negative or non-integer keepalive interval."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
        f"{_CONTROLLER_ID}. Expected a non-negative integer value for the 'keepalive_interval' argument, but "
        f"encountered {keepalive_interval} of type {type(keepalive_interval).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=_CONTROLLER_ID,
            data_logger=logger,
            module_interfaces=(_RecordingModule(),),
            buffer_size=_BUFFER_SIZE,
            port="TEST",
            name=_CONTROLLER_NAME,
            keepalive_interval=keepalive_interval,
        )


@pytest.mark.parametrize("name", ["", 5, None])
def test_microcontroller_interface_initialization_invalid_name(logger: DataLogger, name: Any) -> None:
    """Verifies that MicroControllerInterface rejects a name that is not a non-empty string."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
        f"{_CONTROLLER_ID}. Expected a non-empty string for the 'name' argument, but encountered {name!r} of "
        f"type {type(name).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=_CONTROLLER_ID,
            data_logger=logger,
            module_interfaces=(_RecordingModule(),),
            buffer_size=_BUFFER_SIZE,
            port="TEST",
            name=name,
        )


def test_microcontroller_interface_initialization_duplicate_module_codes(logger: DataLogger) -> None:
    """Verifies that MicroControllerInterface rejects two module interfaces sharing a type and id combination."""
    message = (
        f"Unable to initialize the MicroControllerInterface instance for the microcontroller with "
        f"id {_CONTROLLER_ID}. Encountered two module interface instances with the same type-code "
        f"({_MODULE_TYPE}) and id-code ({_MODULE_ID}), which is not allowed. Each type and id "
        f"combination can only be used by a single module interface instance."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        MicroControllerInterface(
            controller_id=_CONTROLLER_ID,
            data_logger=logger,
            module_interfaces=(_RecordingModule(), _RecordingModule(name="duplicate_module")),
            buffer_size=_BUFFER_SIZE,
            port="TEST",
            name=_CONTROLLER_NAME,
        )


def test_parse_kernel_data_ignores_ordinary_states() -> None:
    """Verifies that parsing an ordinary Kernel state message raises no error."""
    MicroControllerInterface._parse_kernel_data(
        controller_id=_CONTROLLER_ID,
        controller_name=_CONTROLLER_NAME,
        incoming_data=interface.KernelState(
            message=np.array([KernelCommandCodes.IDENTIFY_CONTROLLER, KernelStatusCodes.SETUP_COMPLETE], dtype=np.uint8)
        ),
    )


def test_parse_kernel_data_raises_for_faults() -> None:
    """Verifies that parsing a Kernel fault message raises a runtime error naming the reported status."""
    incoming_data = interface.KernelState(
        message=np.array(
            [KernelCommandCodes.IDENTIFY_CONTROLLER, KernelStatusCodes.COMMAND_NOT_RECOGNIZED], dtype=np.uint8
        )
    )
    with pytest.raises(RuntimeError, match="COMMAND_NOT_RECOGNIZED"):
        MicroControllerInterface._parse_kernel_data(
            controller_id=_CONTROLLER_ID,
            controller_name=_CONTROLLER_NAME,
            incoming_data=incoming_data,
        )


def test_parse_service_module_data_ignores_ordinary_states() -> None:
    """Verifies that parsing an ordinary service module state message raises no error."""
    MicroControllerInterface._parse_service_module_data(
        controller_id=_CONTROLLER_ID,
        controller_name=_CONTROLLER_NAME,
        module_name=_MODULE_NAME,
        incoming_data=ModuleState(
            message=np.array([_MODULE_TYPE, _MODULE_ID, 1, ModuleStatusCodes.COMMAND_COMPLETED], dtype=np.uint8)
        ),
    )


def test_parse_service_module_data_raises_for_faults() -> None:
    """Verifies that parsing a service module fault message raises a runtime error naming the module."""
    incoming_data = ModuleState(
        message=np.array([_MODULE_TYPE, _MODULE_ID, 1, ModuleStatusCodes.COMMAND_NOT_RECOGNIZED], dtype=np.uint8)
    )
    with pytest.raises(RuntimeError, match=_MODULE_NAME):
        MicroControllerInterface._parse_service_module_data(
            controller_id=_CONTROLLER_ID,
            controller_name=_CONTROLLER_NAME,
            module_name=_MODULE_NAME,
            incoming_data=incoming_data,
        )


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_accepts_a_matching_configuration(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a microcontroller reporting the expected identity and modules completes the verification."""
    communication = _verify_against(
        payloads=[
            _build_controller_identification(controller_id=int(_CONTROLLER_ID)),
            _build_module_identification(module_type=int(_MODULE_TYPE), module_id=int(_MODULE_ID)),
        ],
        modules=(_RecordingModule(),),
        terminator_array=terminator_array,
        logger_queue=logger_queue,
    )

    assert terminator_array[1] == 1
    assert [message.command for message in communication.transmitted] == [
        KernelCommandCodes.IDENTIFY_CONTROLLER.value,
        KernelCommandCodes.IDENTIFY_MODULES.value,
    ]


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_accepts_extra_hardware_modules(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a microcontroller managing more modules than the interfaces cover passes the verification."""
    _verify_against(
        payloads=[
            _build_controller_identification(controller_id=int(_CONTROLLER_ID)),
            _build_module_identification(module_type=int(_MODULE_TYPE), module_id=int(_MODULE_ID)),
            _build_module_identification(module_type=7, module_id=9),
        ],
        modules=(_RecordingModule(),),
        terminator_array=terminator_array,
        logger_queue=logger_queue,
    )

    assert terminator_array[1] == 1


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_unresponsive_controller(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a microcontroller that never identifies itself aborts the verification."""
    message = (
        f"Unable to initialize the communication with the microcontroller {_CONTROLLER_ID}. The "
        f"microcontroller did not respond to the identification request after "
        f"{interface._RuntimeParameters.MAXIMUM_COMMUNICATION_ATTEMPTS.value} attempts."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _verify_against(
            payloads=[],
            modules=(_RecordingModule(),),
            terminator_array=terminator_array,
            logger_queue=logger_queue,
        )


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_mismatched_controller_id(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a microcontroller reporting an unexpected identity aborts the verification."""
    message = (
        f"Unable to initialize the communication with the microcontroller {_CONTROLLER_ID}. Expected "
        f"{_CONTROLLER_ID} in response to the controller identification request, but "
        f"received a non-matching id 9."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        _verify_against(
            payloads=[_build_controller_identification(controller_id=9)],
            modules=(_RecordingModule(),),
            terminator_array=terminator_array,
            logger_queue=logger_queue,
        )


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_unreported_modules(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a microcontroller that reports no hardware modules aborts the verification."""
    message = (
        f"Unable to initialize the communication with the microcontroller {_CONTROLLER_ID}. The "
        f"microcontroller did not respond to the module identification request."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        _verify_against(
            payloads=[_build_controller_identification(controller_id=int(_CONTROLLER_ID))],
            modules=(_RecordingModule(),),
            terminator_array=terminator_array,
            logger_queue=logger_queue,
        )


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_missing_hardware_modules(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a microcontroller managing fewer modules than the interfaces expect aborts the verification."""
    message = (
        f"Unable to initialize the communication with the microcontroller {_CONTROLLER_ID}. The microcontroller "
        f"does not manage all of the hardware modules expected by the interfaces passed to the "
        f"MicroControllerInterface instance."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        _verify_against(
            payloads=[
                _build_controller_identification(controller_id=int(_CONTROLLER_ID)),
                _build_module_identification(module_type=int(_MODULE_TYPE), module_id=int(_MODULE_ID)),
            ],
            modules=(_RecordingModule(), _RecordingModule(module_id=np.uint8(3), name="second_module")),
            terminator_array=terminator_array,
            logger_queue=logger_queue,
        )


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_duplicate_hardware_modules(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a microcontroller managing two modules with one type and id pair aborts the verification."""
    message = (
        f"Unable to initialize the communication with the microcontroller {_CONTROLLER_ID}. The microcontroller "
        f"contains multiple module instances with the same type + id code combination. All modules must use "
        f"a unique combination of type and id codes."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        _verify_against(
            payloads=[
                _build_controller_identification(controller_id=int(_CONTROLLER_ID)),
                _build_module_identification(module_type=int(_MODULE_TYPE), module_id=int(_MODULE_ID)),
                _build_module_identification(module_type=int(_MODULE_TYPE), module_id=int(_MODULE_ID)),
            ],
            modules=(_RecordingModule(),),
            terminator_array=terminator_array,
            logger_queue=logger_queue,
        )


@pytest.mark.usefixtures("short_identification_timeout")
def test_verify_microcontroller_communication_unmatched_module_interface(
    terminator_array: SharedMemoryArray, logger_queue: MPQueue
) -> None:
    """Verifies that a module interface without a matching hardware module aborts the verification."""
    message = (
        f"Unable to initialize the communication with the microcontroller {_CONTROLLER_ID}. "
        f"The interface instance for the module with type {_MODULE_TYPE} and id "
        f"{_MODULE_ID} codes does not have a matching hardware module instance managed by the "
        f"microcontroller."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        _verify_against(
            payloads=[
                _build_controller_identification(controller_id=int(_CONTROLLER_ID)),
                _build_module_identification(module_type=7, module_id=9),
            ],
            modules=(_RecordingModule(),),
            terminator_array=terminator_array,
            logger_queue=logger_queue,
        )


@pytest.mark.usefixtures("short_identification_timeout")
def test_runtime_cycle_verifies_the_configuration_before_the_communication_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that the communication cycle runs the configuration verification before it exchanges any data."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_controller_identification(controller_id=int(_CONTROLLER_ID)),
            _build_module_identification(module_type=int(_MODULE_TYPE), module_id=int(_MODULE_ID)),
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    module = _RecordingModule(data_codes={_DATA_CODE})
    array = _ScriptedTerminatorArray()

    _run_cycle(modules=(module,), terminator_array=array)

    assert array.initialized
    assert array.connections == 1
    assert array.disconnections == 1
    assert module.initializations == 1
    assert module.terminations == 1


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_transmits_the_queued_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the communication cycle transmits every message queued for the microcontroller."""
    factory = _StagedCommunicationFactory()
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    input_queue: Queue[Any] = Queue()
    command = KernelCommand(command=np.uint8(KernelCommandCodes.RESET_CONTROLLER.value), return_code=np.uint8(0))
    input_queue.put(command)

    _run_cycle(
        modules=(_RecordingModule(),), terminator_array=_ScriptedTerminatorArray(cycles=1), input_queue=input_queue
    )

    assert factory.instance is not None
    assert factory.instance.transmitted == [command]


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_records_the_keepalive_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a reception code carrying the keepalive return code answers the outstanding keepalive."""
    factory = _StagedCommunicationFactory(
        payloads=[_build_reception_code(code=interface._RuntimeParameters.KEEPALIVE_RETURN_CODE.value)]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    array = _ScriptedTerminatorArray(cycles=2, cycle_delay=_CYCLE_DELAY)

    _run_cycle(modules=(_RecordingModule(),), terminator_array=array, keepalive_interval=1)

    assert factory.instance is not None
    assert [int(message.command) for message in factory.instance.transmitted] == [
        KernelCommandCodes.KEEPALIVE.value,
        KernelCommandCodes.KEEPALIVE.value,
    ]


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_raises_when_the_keepalive_goes_unanswered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unanswered keepalive message resets the microcontroller and aborts the runtime."""
    factory = _StagedCommunicationFactory()
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    array = _ScriptedTerminatorArray(cycles=2, cycle_delay=_CYCLE_DELAY)
    message = (
        f"Unable to maintain the communication with the microcontroller {_CONTROLLER_ID}. The "
        f"microcontroller did not respond to the keepalive message within the expected interval "
        f"of 1 milliseconds."
    )

    with pytest.raises(RuntimeError, match=error_format(message)):
        _run_cycle(modules=(_RecordingModule(),), terminator_array=array, keepalive_interval=1)

    assert factory.instance is not None
    assert [int(message.command) for message in factory.instance.transmitted] == [
        KernelCommandCodes.KEEPALIVE.value,
        KernelCommandCodes.RESET_CONTROLLER.value,
    ]


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_raises_for_kernel_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a Kernel message reporting a fault aborts the communication runtime."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_kernel_state(
                command=KernelCommandCodes.RESET_CONTROLLER.value, event=KernelStatusCodes.COMMAND_NOT_RECOGNIZED.value
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)

    with pytest.raises(RuntimeError, match="COMMAND_NOT_RECOGNIZED"):
        _run_cycle(modules=(_RecordingModule(),), terminator_array=_ScriptedTerminatorArray(cycles=1))


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_ignores_ordinary_kernel_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a Kernel message reporting an ordinary state leaves the communication runtime running."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_kernel_state(
                command=KernelCommandCodes.IDENTIFY_CONTROLLER.value, event=KernelStatusCodes.SETUP_COMPLETE.value
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    module = _RecordingModule()

    _run_cycle(modules=(module,), terminator_array=_ScriptedTerminatorArray(cycles=1))

    assert module.terminations == 1


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_raises_for_service_module_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a module message carrying a service fault code aborts the communication runtime."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_module_state(
                module_type=int(_MODULE_TYPE),
                module_id=int(_MODULE_ID),
                command=1,
                event=ModuleStatusCodes.COMMAND_NOT_RECOGNIZED.value,
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)

    with pytest.raises(RuntimeError, match=_MODULE_NAME):
        _run_cycle(modules=(_RecordingModule(),), terminator_array=_ScriptedTerminatorArray(cycles=1))


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_names_unregistered_senders_of_service_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a service fault from a module no interface claims names the module as unregistered."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_module_state(
                module_type=7, module_id=9, command=1, event=ModuleStatusCodes.COMMAND_NOT_RECOGNIZED.value
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)

    with pytest.raises(RuntimeError, match=interface._UNKNOWN_MODULE_NAME):
        _run_cycle(modules=(_RecordingModule(),), terminator_array=_ScriptedTerminatorArray(cycles=1))


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_skips_modules_without_registered_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a custom event addressed to an interface registering no codes reaches no processing method."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_module_data(
                module_type=int(_MODULE_TYPE),
                module_id=int(_MODULE_ID),
                command=1,
                event=int(_DATA_CODE),
                value=42,
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    module = _RecordingModule()

    _run_cycle(modules=(module,), terminator_array=_ScriptedTerminatorArray(cycles=1))

    assert module.processed == []


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_ignores_unregistered_custom_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a custom event matching neither a data nor an error code reaches no processing method."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_module_data(
                module_type=int(_MODULE_TYPE),
                module_id=int(_MODULE_ID),
                command=1,
                event=MINIMUM_CUSTOM_STATUS_CODE,
                value=42,
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    module = _RecordingModule(data_codes={_DATA_CODE})

    _run_cycle(modules=(module,), terminator_array=_ScriptedTerminatorArray(cycles=1))

    assert module.processed == []


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_processes_registered_data_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a custom event matching a registered data code reaches the interface's processing method."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_module_data(
                module_type=int(_MODULE_TYPE),
                module_id=int(_MODULE_ID),
                command=3,
                event=int(_DATA_CODE),
                value=42,
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    module = _RecordingModule(data_codes={_DATA_CODE})

    _run_cycle(modules=(module,), terminator_array=_ScriptedTerminatorArray(cycles=1))

    assert module.processed == [(3, int(_DATA_CODE))]


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_raises_for_registered_error_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a custom event matching a registered error code aborts the runtime with its explanation."""
    factory = _StagedCommunicationFactory(
        payloads=[
            _build_module_data(
                module_type=int(_MODULE_TYPE),
                module_id=int(_MODULE_ID),
                command=3,
                event=int(_ERROR_CODE),
                value=42,
            )
        ]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    module = _RecordingModule(error_codes={_ERROR_CODE: _ERROR_EXPLANATION})

    with pytest.raises(RuntimeError, match=error_format(_ERROR_EXPLANATION)):
        _run_cycle(modules=(module,), terminator_array=_ScriptedTerminatorArray(cycles=1))

    assert module.processed == []


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_reports_asset_initialization_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that an interface failing to claim its assets aborts the runtime and releases the claimed assets."""
    factory = _StagedCommunicationFactory()
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    healthy = _FailingModule(module_id=np.uint8(3), name="healthy_module")
    failing = _FailingModule(name="failing_module", fail_initialization=True)

    with pytest.raises(RuntimeError, match="failed to claim its remote assets"):
        _run_cycle(modules=(healthy, failing), terminator_array=_ScriptedTerminatorArray())

    # The interface that raised never reached the tracker, so the shutdown releases the assets of its sibling alone.
    assert healthy.terminations == 1
    assert failing.terminations == 0
    assert "failed to claim its remote assets" in capsys.readouterr().err


@pytest.mark.usefixtures("initialized_cycle")
def test_runtime_cycle_reports_asset_termination_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that an interface failing to release its assets still releases the assets of the interfaces after it."""
    factory = _StagedCommunicationFactory()
    monkeypatch.setattr(interface, "SerialCommunication", factory)
    failing = _FailingModule(name="failing_module", fail_termination=True)
    healthy = _FailingModule(module_id=np.uint8(3), name="healthy_module")

    _run_cycle(modules=(failing, healthy), terminator_array=_ScriptedTerminatorArray())

    assert healthy.terminations == 1
    assert "Unable to terminate the remote assets of the failing_module module interface" in capsys.readouterr().err


def test_stop_releases_the_runtime_assets(
    logger: DataLogger,
    terminator_array: SharedMemoryArray,
    terminated_process: Process,
    terminated_thread: Thread,
) -> None:
    """Verifies that stopping a running interface signals the shutdown and releases the shared memory buffer."""
    controller = MicroControllerInterface(
        controller_id=_CONTROLLER_ID,
        data_logger=logger,
        module_interfaces=(_RecordingModule(),),
        buffer_size=_BUFFER_SIZE,
        port="TEST",
        name=_CONTROLLER_NAME,
    )
    _prepare_shutdown(
        controller=controller,
        terminator_array=terminator_array,
        process=terminated_process,
        watchdog_thread=terminated_thread,
    )
    observed_flags = _observe_shutdown_flag(process=terminated_process, terminator_array=terminator_array)

    controller.stop()

    assert not controller._started
    assert not terminator_array.is_connected
    # The shutdown signal reaches the communication process before the interface waits for that process to terminate,
    # which is the only ordering that lets the process drain its command queue and exit on its own.
    assert observed_flags == [1]
    assert isinstance(controller._input_queue.get(), KernelCommand)


def test_stop_ignores_an_interface_that_never_started(logger: DataLogger) -> None:
    """Verifies that stopping an interface whose communication process never started releases nothing."""
    controller = MicroControllerInterface(
        controller_id=_CONTROLLER_ID,
        data_logger=logger,
        module_interfaces=(_RecordingModule(),),
        buffer_size=_BUFFER_SIZE,
        port="TEST",
        name=_CONTROLLER_NAME,
    )

    controller.stop()

    assert controller._terminator_array is None


def test_watchdog_reports_a_prematurely_terminated_process(
    logger: DataLogger, terminator_array: SharedMemoryArray, terminated_process: Process
) -> None:
    """Verifies that the watchdog releases the runtime assets and reports a communication process that died."""
    controller = MicroControllerInterface(
        controller_id=_CONTROLLER_ID,
        data_logger=logger,
        module_interfaces=(_RecordingModule(),),
        buffer_size=_BUFFER_SIZE,
        port="TEST",
        name=_CONTROLLER_NAME,
    )
    _prepare_shutdown(controller=controller, terminator_array=terminator_array, process=terminated_process)
    observed_flags = _observe_shutdown_flag(process=terminated_process, terminator_array=terminator_array)
    message = (
        f"Unable to maintain the communication process of the MicroControllerInterface with id "
        f"{_CONTROLLER_ID}. The process has been prematurely shut down, which likely indicates that "
        f"it has encountered a runtime error that terminated it."
    )

    with pytest.raises(RuntimeError, match=error_format(message)):
        controller._watchdog()

    assert observed_flags == [1]

    assert not controller._started
    assert not terminator_array.is_connected


def test_watchdog_yields_the_shutdown_to_a_concurrent_stop(
    logger: DataLogger, terminator_array: SharedMemoryArray, terminated_process: Process
) -> None:
    """Verifies that the watchdog leaves the teardown alone once another caller claims the shutdown."""
    controller = MicroControllerInterface(
        controller_id=_CONTROLLER_ID,
        data_logger=logger,
        module_interfaces=(_RecordingModule(),),
        buffer_size=_BUFFER_SIZE,
        port="TEST",
        name=_CONTROLLER_NAME,
    )
    _prepare_shutdown(controller=controller, terminator_array=terminator_array, process=terminated_process)

    def is_alive() -> bool:
        """Reports a dead process after clearing the flag a concurrent stop() clears when it claims the shutdown."""
        controller._started = False
        return False

    terminated_process.is_alive = is_alive  # type: ignore[method-assign]

    controller._watchdog()

    # The claimant owns the teardown, so the watchdog leaves the buffer for that caller to release.
    assert terminator_array.is_connected


def test_watchdog_waits_out_the_communication_process_startup(logger: DataLogger) -> None:
    """Verifies that the watchdog completes its cycle without error while the interface reports an unstarted
    runtime."""
    controller = MicroControllerInterface(
        controller_id=_CONTROLLER_ID,
        data_logger=logger,
        module_interfaces=(_RecordingModule(),),
        buffer_size=_BUFFER_SIZE,
        port="TEST",
        name=_CONTROLLER_NAME,
    )
    controller._terminator_array = _ScriptedTerminatorArray(cycles=1)  # type: ignore[assignment]

    controller._watchdog()

    assert not controller._started


def test_evaluate_port_reports_the_microcontroller_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that evaluating a port connected to a microcontroller reports the identifier it answers with."""
    factory = _StagedCommunicationFactory(
        payloads=[_build_controller_identification(controller_id=int(_CONTROLLER_ID))]
    )
    monkeypatch.setattr(interface, "SerialCommunication", factory)

    assert evaluate_port(port="TEST") == (int(_CONTROLLER_ID), None)


@pytest.mark.usefixtures("short_identification_timeout")
def test_evaluate_port_reports_an_unresponsive_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that evaluating a port that never answers the identification request reports no microcontroller."""
    monkeypatch.setattr(interface, "SerialCommunication", _StagedCommunicationFactory())

    assert evaluate_port(port="TEST") == (-1, None)


def test_evaluate_port_reports_a_connection_failure() -> None:
    """Verifies that evaluating a port that cannot be opened reports the failure."""
    identifier, error = evaluate_port(port="/dev/aci_absent_port")

    assert identifier == -1
    assert error is not None


def test_discover_microcontrollers_skips_the_ports_without_a_product_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that discovery evaluates no port when every enumerated port lacks a product identifier."""

    def forbidden_pool(**_kwargs: Any) -> None:
        message = "Discovery spawned an evaluation pool for the ports it reports no microcontroller for."
        raise AssertionError(message)

    monkeypatch.setattr(interface, "list_available_ports", lambda: (_build_port(device="/dev/ttyS0", pid=None),))
    monkeypatch.setattr(interface, "ProcessPoolExecutor", forbidden_pool)

    assert discover_microcontrollers() == ()


def test_discover_microcontrollers_reports_every_evaluated_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that discovery reports one entry per evaluable port, in the order the host enumerates the ports."""
    evaluations = {
        "/dev/ttyACM0": (int(_CONTROLLER_ID), None),
        "/dev/ttyACM1": (-1, None),
        "/dev/ttyACM2": (-1, "OSError: port busy"),
    }
    ports = tuple(_build_port(device=device, pid=index) for index, device in enumerate(evaluations))

    def evaluate(port: str, baudrate: int) -> tuple[int, str | None]:
        assert baudrate == 9600
        return evaluations[port]

    pools: list[_InlineExecutor] = []
    monkeypatch.setattr(interface, "list_available_ports", lambda: ports)
    monkeypatch.setattr(interface, "evaluate_port", evaluate)
    monkeypatch.setattr(interface, "ProcessPoolExecutor", _build_inline_pool(pools=pools))

    discovered = discover_microcontrollers(baudrate=9600)

    # The pool answers in whatever order the ports respond in, so the reported order is checked against the enumeration
    # order every caller that renumbers the ports depends on.
    assert [controller.port for controller in discovered] == list(evaluations)
    assert [controller.description for controller in discovered] == ["port 0", "port 1", "port 2"]
    # The evaluator's sentinel reaches the caller as an absent identifier rather than as a negative code.
    assert [controller.controller_id for controller in discovered] == [int(_CONTROLLER_ID), None, None]
    assert [controller.error_message for controller in discovered] == [None, None, "OSError: port busy"]


def test_discover_microcontrollers_pins_every_evaluation_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that discovery hands the pool the thread pinning each port evaluation worker starts with."""
    ports = (_build_port(device="/dev/ttyACM0", pid=0),)
    pools: list[_InlineExecutor] = []

    monkeypatch.setattr(interface, "list_available_ports", lambda: ports)
    monkeypatch.setattr(interface, "evaluate_port", lambda **_kwargs: (-1, None))
    monkeypatch.setattr(interface, "ProcessPoolExecutor", _build_inline_pool(pools=pools))

    discover_microcontrollers()

    assert len(pools) == 1
    # A worker waits on one serial port, so the pool it belongs to pins the numeric backends of every worker to a
    # single thread.
    assert pools[0].initializer is interface.initialize_worker_threads
    assert pools[0].initargs == (interface._WORKER_THREAD_CEILING,)
    # The pool never outgrows the port count, which is what keeps a host with many ports from spawning a process per
    # port.
    assert pools[0].max_workers == 1


def _build_port(device: str, pid: int | None) -> ListPortInfo:
    """Builds the serial port record the host enumeration hands to discovery."""
    port = ListPortInfo(device=device, skip_link_detection=True)
    port.pid = pid
    port.description = f"port {pid}"
    return port


def _build_inline_pool(pools: list[_InlineExecutor]) -> Callable[..., _InlineExecutor]:
    """Builds the process pool factory that records every pool discovery creates."""

    def build(max_workers: int, initializer: Callable[[int], None], initargs: tuple[int, ...]) -> _InlineExecutor:
        """Creates the recorded pool."""
        pool = _InlineExecutor(max_workers=max_workers, initializer=initializer, initargs=initargs)
        pools.append(pool)
        return pool

    return build


class _InlineExecutor:
    """Evaluates every submitted port in the calling process, standing in for the pool discovery fans out over.

    Notes:
        The initializer is recorded rather than called, as running the thread pinning of a spawned worker inside the
        test process would reconfigure the numeric backends of the whole test session.

    Args:
        max_workers: The worker count discovery sized the pool to.
        initializer: The callable discovery pins each spawned worker with.
        initargs: The arguments discovery passes to the initializer.

    Attributes:
        max_workers: The worker count discovery sized the pool to.
        initializer: The callable discovery pins each spawned worker with.
        initargs: The arguments discovery passes to the initializer.
    """

    def __init__(self, max_workers: int, initializer: Callable[[int], None], initargs: tuple[int, ...]) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self) -> Self:
        """Returns the pool to the discovery context."""
        return self

    def __exit__(self, *_details: object) -> bool:
        """Releases the pool without suppressing the exceptions raised inside the discovery context."""
        return False

    def submit(self, function: Callable[..., Any], /, **kwargs: Any) -> Future[Any]:
        """Evaluates the submitted port immediately and reports the result through a resolved future."""
        future: Future[Any] = Future()
        future.set_result(function(**kwargs))
        return future


class _FailingModule(_RecordingModule):
    """Raises from the remote asset methods the communication cycle calls, reporting the requested failures.

    Args:
        module_type: The code that identifies the type of the interfaced module.
        module_id: The code that identifies the specific interfaced module instance.
        name: The human-readable name of the interfaced module.
        fail_initialization: Determines whether initialize_remote_assets() raises.
        fail_termination: Determines whether terminate_remote_assets() raises.

    Attributes:
        fail_initialization: Determines whether initialize_remote_assets() raises.
        fail_termination: Determines whether terminate_remote_assets() raises.
    """

    def __init__(
        self,
        module_type: np.uint8 = _MODULE_TYPE,
        module_id: np.uint8 = _MODULE_ID,
        name: str = _MODULE_NAME,
        *,
        fail_initialization: bool = False,
        fail_termination: bool = False,
    ) -> None:
        super().__init__(module_type=module_type, module_id=module_id, name=name)
        self.fail_initialization = fail_initialization
        self.fail_termination = fail_termination

    def initialize_remote_assets(self) -> None:
        """Raises if the interface is configured to fail its remote asset initialization."""
        super().initialize_remote_assets()
        if self.fail_initialization:
            message = f"The {self.name} module interface failed to claim its remote assets."
            raise RuntimeError(message)

    def terminate_remote_assets(self) -> None:
        """Raises if the interface is configured to fail its remote asset termination."""
        super().terminate_remote_assets()
        if self.fail_termination:
            message = f"The {self.name} module interface failed to release its remote assets."
            raise RuntimeError(message)


class _RecordingSerialCommunication(SerialCommunication):
    """Records every message transmitted through the instance while sending it through the real serial stack.

    Notes:
        The instance drives the mocked serial port the TransportLayer builds under test mode, so each recorded
        message is also encoded and framed the way a transmission to a real microcontroller would be.

    Attributes:
        transmitted: Stores every message object handed to send_message(), in transmission order.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(test_mode=True, **kwargs)
        self.transmitted: list[Any] = []

    def send_message(self, message: Any) -> None:
        """Records the outgoing message before transmitting it."""
        self.transmitted.append(message)
        super().send_message(message=message)


class _StagedCommunicationFactory:
    """Builds the communication cycle's SerialCommunication instance with the microcontroller's responses staged.

    Notes:
        The factory replaces the module-level SerialCommunication name the communication cycle constructs its
        instance from, and forwards every argument the cycle supplies. Only the mocked port and the recording of the
        transmitted messages separate the constructed instance from the one a production runtime builds.

    Args:
        payloads: The message payloads the microcontroller answers with, in the order it sends them.

    Attributes:
        payloads: The message payloads the microcontroller answers with, in the order it sends them.
        instance: Stores the constructed instance once the communication cycle asks for it.
    """

    def __init__(self, payloads: Sequence[NDArray[np.uint8]] = ()) -> None:
        self.payloads = payloads
        self.instance: _RecordingSerialCommunication | None = None

    def __call__(self, **kwargs: Any) -> _RecordingSerialCommunication:
        """Constructs the instance the communication cycle communicates through."""
        self.instance = _RecordingSerialCommunication(**kwargs)
        _stage_incoming_messages(communication=self.instance, payloads=self.payloads)
        return self.instance


class _ScriptedTerminatorArray:
    """Stands in for the shared memory array the communication process reads its runtime flags from.

    Notes:
        Reading the shutdown flag reports a running runtime for the requested number of communication cycles and a
        requested shutdown afterwards. That ends the cycle at a point each test chooses.

        Each granted cycle spends the configured delay before it reports, which advances the keepalive timer the
        communication cycle reads by the same interval a cycle of a production runtime would.

    Args:
        cycles: The number of communication cycles to grant before reporting the shutdown request.
        cycle_delay: The time, in milliseconds, each granted cycle takes.

    Attributes:
        _cycles: Counts the communication cycles still granted before the shutdown request is reported.
        _timer: The timer that spends the configured delay on each granted communication cycle.
        _cycle_delay: The time, in milliseconds, each granted cycle takes.
        connections: Counts the calls the communication cycle made to connect().
        disconnections: Counts the calls the communication cycle made to disconnect().
        initialized: Determines whether the verification stage reported a completed initialization.
    """

    def __init__(self, cycles: int = 0, cycle_delay: int = 0) -> None:
        self._cycles = cycles
        self._timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
        self._cycle_delay = cycle_delay
        self.connections = 0
        self.disconnections = 0
        self.initialized = False

    def __getitem__(self, index: int) -> int:
        """Returns the runtime flag stored under the requested index."""
        if index != 0:
            return int(self.initialized)
        if self._cycles <= 0:
            return 1
        self._cycles -= 1
        if self._cycle_delay:
            self._timer.delay(delay=self._cycle_delay, allow_sleep=True, block=False)
        return 0

    def __setitem__(self, index: int, value: int) -> None:
        """Records the completed initialization the communication cycle reports, and drops every other index."""
        if index == 1:
            self.initialized = bool(value)

    def connect(self) -> None:
        """Records that the communication cycle connected to the array."""
        self.connections += 1

    def disconnect(self) -> None:
        """Records that the communication cycle disconnected from the array."""
        self.disconnections += 1


def _stage_incoming_messages(communication: SerialCommunication, payloads: Sequence[NDArray[np.uint8]]) -> None:
    """Encodes each payload the way the microcontroller would and stages the result as data to receive."""
    port = communication._transport_layer._port
    for payload in payloads:
        communication._transport_layer.write_data(data_object=payload)
        communication._transport_layer.send_data()
    port.rx_buffer = port.tx_buffer
    port.tx_buffer = b""


def _build_controller_identification(controller_id: int) -> NDArray[np.uint8]:
    """Builds the payload of a ControllerIdentification message reporting the requested controller."""
    return np.array([11, controller_id], dtype=np.uint8)


def _build_module_identification(module_type: int, module_id: int) -> NDArray[np.uint8]:
    """Builds the payload of a ModuleIdentification message reporting the requested hardware module."""
    type_id = np.uint16((module_type << 8) | module_id)
    return np.concatenate((np.array([12], dtype=np.uint8), np.frombuffer(type_id.tobytes(), dtype=np.uint8)))


def _build_module_state(module_type: int, module_id: int, command: int, event: int) -> NDArray[np.uint8]:
    """Builds the payload of a ModuleState message reporting the requested command and event."""
    return np.array([8, module_type, module_id, command, event], dtype=np.uint8)


def _build_module_data(module_type: int, module_id: int, command: int, event: int, value: int) -> NDArray[np.uint8]:
    """Builds the payload of a ModuleData message reporting the requested command, event, and data object."""
    return np.array([6, module_type, module_id, command, event, _ONE_UINT8_PROTOTYPE, value], dtype=np.uint8)


def _build_kernel_state(command: int, event: int) -> NDArray[np.uint8]:
    """Builds the payload of a KernelState message reporting the requested command and event."""
    return np.array([9, command, event], dtype=np.uint8)


def _build_reception_code(code: int) -> NDArray[np.uint8]:
    """Builds the payload of a ReceptionCode message reporting the requested reception code."""
    return np.array([10, code], dtype=np.uint8)


def _run_cycle(
    modules: tuple[ModuleInterface, ...],
    terminator_array: Any,
    input_queue: Any = None,
    keepalive_interval: int = 0,
) -> None:
    """Runs the communication cycle against the supplied runtime assets."""
    MicroControllerInterface._runtime_cycle(
        controller_id=_CONTROLLER_ID,
        controller_name=_CONTROLLER_NAME,
        module_interfaces=modules,
        input_queue=input_queue if input_queue is not None else Queue(),
        logger_queue=MPQueue(),
        terminator_array=terminator_array,
        port="TEST",
        baudrate=115200,
        buffer_size=_BUFFER_SIZE,
        keepalive_interval=keepalive_interval,
    )


def _verify_against(
    payloads: Sequence[NDArray[np.uint8]],
    modules: tuple[ModuleInterface, ...],
    terminator_array: SharedMemoryArray,
    logger_queue: MPQueue,  # type: ignore[type-arg]
    controller_id: np.uint8 = _CONTROLLER_ID,
) -> _RecordingSerialCommunication:
    """Runs the configuration verification against a microcontroller staged to answer with the input payloads."""
    communication = _RecordingSerialCommunication(
        controller_id=controller_id,
        microcontroller_serial_buffer_size=_BUFFER_SIZE,
        port="TEST",
        logger_queue=logger_queue,
    )
    _stage_incoming_messages(communication=communication, payloads=payloads)
    MicroControllerInterface._verify_microcontroller_communication(
        serial_communication=communication,
        timeout_timer=PrecisionTimer(precision=TimerPrecisions.MILLISECOND),
        controller_id=controller_id,
        module_interfaces=modules,
        terminator_array=terminator_array,
    )
    return communication


def _prepare_shutdown(
    controller: MicroControllerInterface,
    terminator_array: SharedMemoryArray,
    process: Process,
    watchdog_thread: Thread | None = None,
) -> None:
    """Places the interface in the state the start() method leaves behind once the communication process is running."""
    controller._terminator_array = terminator_array
    controller._communication_process = process
    controller._watchdog_thread = watchdog_thread
    controller._started = True


def _observe_shutdown_flag(process: Process, terminator_array: SharedMemoryArray) -> list[int]:
    """Records the shutdown flag the communication process observes at the moment the interface waits on it.

    Notes:
        The teardown destroys the shared memory buffer before it returns, so the flag is unreadable by the time the
        caller regains control. Reading it from the join the teardown performs captures both the value and its
        ordering against the wait.
    """
    observed: list[int] = []

    def join(timeout: float | None = None) -> None:
        """Records the shutdown flag."""
        observed.append(int(terminator_array[0]))

    process.join = join  # type: ignore[method-assign]
    return observed
