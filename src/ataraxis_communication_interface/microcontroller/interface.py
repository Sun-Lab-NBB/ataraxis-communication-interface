"""Provides the ModuleInterface and MicroControllerInterface classes that aggregate the methods allowing Python PC
clients to bidirectionally interface with custom hardware modules managed by Arduino or Teensy microcontrollers. It also
provides the evaluate_port() function, which determines whether a serial port is connected to an Ataraxis
microcontroller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import sys
from enum import IntEnum
from typing import TYPE_CHECKING, Any
from functools import partial, lru_cache
from threading import Lock, Thread
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import console
from ataraxis_data_structures import DataLogger, SharedMemoryArray

from .dataclasses import ModuleSourceData, write_microcontroller_manifest
from .status_codes import (
    MAXIMUM_CUSTOM_STATUS_CODE,
    MINIMUM_CUSTOM_STATUS_CODE,
    KernelCommandCodes,
    describe_kernel_event,
    describe_module_event,
    describe_custom_module_error,
)
from ..communication import (
    KernelData,
    ModuleData,
    KernelState,
    ModuleState,
    KernelCommand,
    ReceptionCode,
    ModuleParameters,
    OneOffModuleCommand,
    SerialCommunication,
    DequeueModuleCommand,
    ModuleIdentification,
    RepeatedModuleCommand,
    ControllerIdentification,
)

if TYPE_CHECKING:
    from pathlib import Path
    from multiprocessing.managers import SyncManager

_MAXIMUM_BYTE_VALUE: int = 255
"""The maximum valid uint8 byte value, used to validate that byte-codes fall within the allowed range."""
_MINIMUM_SERIAL_BUFFER_SIZE: int = 9
"""The smallest microcontroller serial buffer size the TransportLayer accepts, which is the width of the packet
preamble and postamble that leaves room for a single payload byte. Checking the value here attributes an invalid
buffer size to the constructor call rather than to the spawned communication process that builds the TransportLayer."""
_ZERO_BYTE: np.uint8 = np.uint8(0)
"""The uint8 zero value, used as the default return code for command and parameter messages."""
_ZERO_LONG: np.uint32 = np.uint32(0)
"""The uint32 zero value, used as the default command repetition delay."""
_UNKNOWN_MODULE_NAME: str = "unregistered module"
"""The name used to identify the sender of a message that no managed module interface claims, which happens when the
microcontroller manages more hardware modules than the interfaces passed to the MicroControllerInterface."""


class _RuntimeParameters(IntEnum):
    """Defines hardcoded runtime parameter constants used throughout this module."""

    DEFAULT_RETURN_CODE = 0
    """The default return code used by Kernel command messages."""
    KEEPALIVE_RETURN_CODE = 255
    """The return code used in keepalive command messages."""
    PROCESS_INITIALIZATION_TIMEOUT = 30
    """The maximum period of time, in seconds, that the MicroControllerInterface class can take to fully initialize the 
    communication process."""
    MICROCONTROLLER_ID_TIMEOUT = 2000
    """The maximum period of time, in milliseconds, that the MicroControllerInterface class can take to request and 
    receive a single microcontroller or hardware module ID message during communication process initialization."""
    MAXIMUM_COMMUNICATION_ATTEMPTS = 3
    """The maximum number of microcontroller ID information request attempts the communication process can carry out 
    during the initialization before raising an error."""
    PROCESS_TERMINATION_TIMEOUT = 60
    """The maximum period of time, in seconds, to wait for the communication process to terminate gracefully. The
    caller stops waiting once this period elapses, which prevents being stuck in a graceful process termination
    loop."""
    WATCHDOG_INTERVAL = 20
    """The frequency, in milliseconds, at which the MicroControllerInterface's watchdog thread checks the state 
    of the remote communication process."""


class ModuleInterface(ABC):
    """Provides the API used by other library components to interface with the custom hardware module controlled by
    the companion Arduino or Teensy microcontroller.

    Any class that inherits from this class gains the API used by the MicroControllerInterface class to bind the
    interface to the managed hardware module instance running on the companion microcontroller. Also, inheriting from
    this class provides the user-facing API for sending commands and parameters to the managed hardware module.

    Notes:
        Every custom hardware module interface has to inherit from this base class. When inheriting from this class,
        initialize the superclass by calling the 'super().__init__()' during the subclass initialization.

        All data received from or sent to the microcontroller is automatically logged to disk. Only provide additional
        data and error codes if the interface must carry out 'online' error detection or data processing.

        Some attributes of this (base) class are assigned by the managing MicroControllerInterface during its
        initialization. Each module interface that inherits from the base ModuleInterface class has to be bound to an
        initialized MicroControllerInterface instance to be fully functional.

        Use the utility methods inherited from the base ModuleInterface to send command and parameter messages to the
        managed hardware module instance. These methods are heavily optimized for runtime efficiency and performance.

    Args:
        module_type: The code that identifies the type (family) of the interfaced module.
        module_id: The code that identifies the specific interfaced module instance.
        name: A colloquial human-readable name for this hardware module (e.g., 'encoder', 'lick_sensor'). Written
            to the microcontroller manifest file alongside the module's type and id codes to identify this module.
        error_codes: An optional mapping of the codes used by the module to communicate runtime errors to the
            explanations surfaced when those errors arrive. Receiving a message with an event-code from this mapping
            raises a RuntimeError that carries the matching explanation and aborts the runtime. Every code must fall
            within the custom event code range the microcontroller reserves for hardware modules, as the library
            resolves the codes below that range itself.
        data_codes: An optional set of codes used by the module to communicate data messages that required online
            processing. Received messages with an event-code from this set are passed to the interface instance's
            process_received_data() method for further processing. Every code must fall within the custom event code
            range the microcontroller reserves for hardware modules.

    Attributes:
        _module_type: Stores the id-code of the managed hardware module's type (family).
        _module_id: Stores the specific instance ID of the managed hardware module.
        _type_id: Stores the type and id codes combined into a single uint16 value.
        _data_codes: Stores all message event-codes that require additional processing.
        _error_codes: Maps each message error-code that warrants runtime interruption to its explanation.
        _name: Stores the human-readable name of this module instance.
        _input_queue: The multiprocessing queue used to send command and parameter messages to the microcontroller
            communication process.
        _dequeue_command: Stores the instance's DequeueModuleCommand object.
        _create_command_message: LRU-cached method (maxsize=32) for creating command message objects. Caches up to 32
            unique command message configurations to avoid redundant object creation and serialization during repeated
            command operations.
        _create_parameters_message: LRU-cached method (maxsize=16) for creating parameter message objects. Caches up
            to 16 unique parameter configurations to avoid redundant object creation and serialization when repeatedly
            sending the same parameter presets.

    Raises:
        TypeError: If input arguments are not of the expected type.
        ValueError: If any error or data code falls outside the custom event code range.
    """

    def __init__(
        self,
        module_type: np.uint8,
        module_id: np.uint8,
        name: str,
        error_codes: dict[np.uint8, str] | None = None,
        data_codes: set[np.uint8] | None = None,
    ) -> None:
        # Ensures that input byte-codes use valid value ranges.
        if not isinstance(module_type, np.uint8) or not 1 <= module_type <= _MAXIMUM_BYTE_VALUE:
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected an unsigned integer value between 1 and 255 for 'module_type' argument, but encountered "
                f"{module_type} of type {type(module_type).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(module_id, np.uint8) or not 1 <= module_id <= _MAXIMUM_BYTE_VALUE:
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected an unsigned integer value between 1 and 255 for 'module_id' argument, but encountered "
                f"{module_id} of type {type(module_id).__name__}."
            )
            console.error(message=message, error=TypeError)
        if (error_codes is not None and not isinstance(error_codes, dict)) or (
            isinstance(error_codes, dict)
            and not all(
                isinstance(code, np.uint8) and isinstance(explanation, str) and explanation
                for code, explanation in error_codes.items()
            )
        ):
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected a dictionary mapping numpy uint8 values to non-empty strings or None for 'error_codes' "
                f"argument, but encountered {error_codes} of type {type(error_codes).__name__} and / or at least one "
                f"invalid key or value."
            )
            console.error(message=message, error=TypeError)
        if (data_codes is not None and not isinstance(data_codes, set)) or (
            isinstance(data_codes, set) and not all(isinstance(code, np.uint8) for code in data_codes)
        ):
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected a set of numpy uint8 values or None for 'data_codes' argument, but encountered "
                f"{data_codes} of type {type(data_codes).__name__} and / or at least one non-uint8 item."
            )
            console.error(message=message, error=TypeError)

        # The runtime resolves every event code below the custom range through the library's own service code handling,
        # so a code declared here from that range would never reach the interface that declared it.
        for argument_name, codes in (("error_codes", error_codes or {}), ("data_codes", data_codes or set())):
            invalid_codes = sorted(
                int(code) for code in codes if not MINIMUM_CUSTOM_STATUS_CODE <= code <= MAXIMUM_CUSTOM_STATUS_CODE
            )
            if invalid_codes:
                message = (
                    f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                    f"Expected every code in the {argument_name!r} argument to fall between "
                    f"{MINIMUM_CUSTOM_STATUS_CODE} and {MAXIMUM_CUSTOM_STATUS_CODE}, which is the event code range "
                    f"the microcontroller reserves for custom hardware modules, but encountered {invalid_codes}."
                )
                console.error(message=message, error=ValueError)
        if not isinstance(name, str) or not name:
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected a non-empty string for the 'name' argument, but encountered {name!r} of type "
                f"{type(name).__name__}."
            )
            console.error(message=message, error=TypeError)

        self._module_type: np.uint8 = module_type
        self._module_id: np.uint8 = module_id
        self._name: str = name

        # Combines type and ID codes into a 16-bit value. This is used to ensure every module instance has a unique
        # ID + Type combination. This method is position-aware, so inverse type-id pairs are coded as different
        # values e.g.: 4-5 != 5-4
        self._type_id: np.uint16 = np.uint16(
            (self._module_type.astype(np.uint16) << 8) | self._module_id.astype(np.uint16)
        )

        # Resolves message codes that require additional (custom) processing.
        self._data_codes: set[np.uint8] = data_codes if data_codes is not None else set()

        # Adds error-handling support. This allows raising errors when the module sends a message with an error code
        # from the microcontroller to the PC, and surfaces the registered explanation alongside the raised error.
        self._error_codes: dict[np.uint8, str] = error_codes if error_codes is not None else {}

        # This attribute is initialized to a placeholder value. The actual value is assigned by the
        # MicroControllerInterface class that manages this ModuleInterface. During MicroControllerInterface
        # initialization, it updates the attribute for all managed interfaces via referencing.
        self._input_queue: MPQueue | None = None  # type: ignore[type-arg]  # MPQueue is not generic.

        #  Pre-creates the Dequeue command object, as it does not change throughout runtime.
        self._dequeue_command = DequeueModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=_ZERO_BYTE,
        )

        # Binds LRU caches for command and parameter message creation. The LRU cache is used to optimize runtime
        # performance.
        self._create_command_message = lru_cache(maxsize=32)(self._create_command_message_implementation)
        self._create_parameters_message = lru_cache(maxsize=16)(self._create_parameters_message_implementation)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleInterface instance."""
        return (
            f"ModuleInterface(module_type={self._module_type}, module_id={self._module_id}, "
            f"name='{self._name}', combined_type_id={self._type_id}, data_codes={sorted(self._data_codes)}, "
            f"error_codes={sorted(self._error_codes)})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Excludes LRU cache wrappers when pickling the instance."""
        # Since LRU caches are only used in the main thread and cannot be pickled, the easiest way to ensure each
        # interface functions as expected on Windows (requires pickling to spawn the communication process) is to
        # exclude them during pickling.
        state = self.__dict__.copy()
        state["_create_command_message"] = None
        state["_create_parameters_message"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restores the instance without LRU caches during unpickling."""
        self.__dict__.update(state)

    @abstractmethod
    def initialize_remote_assets(self) -> None:
        """Initializes the interface instance assets used in the remote microcontroller communication process.

        This method is called during the initial setup sequence of the remote microcontroller communication process,
        before the PC-microcontroller communication cycle.

        Notes:
            This method should instantiate all interface assets that do not support pickling, such as PrecisionTimer
            or SharedMemoryArray instances. All assets initialized by this method must be destroyed by the
            terminate_remote_assets() method.
        """
        raise NotImplementedError

    @abstractmethod
    def terminate_remote_assets(self) -> None:
        """Terminates the interface instance assets used in the remote microcontroller communication process.

        This method is the opposite of the initialize_remote_assets() method. It is called as part of the remote
        communication process shutdown routine to ensure any resources claimed by the interface are properly
        released before the communication process terminates.
        """
        raise NotImplementedError

    @abstractmethod
    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes the input message.

        This method is called during the communication cycle's runtime when the interface instance receives a message
        from the microcontroller that uses an event code provided at class initialization as 'data_codes' argument.

        Notes:
            This method should implement the custom online data-processing logic associated with each message whose
            event code is specified in the 'data_codes' argument.

            All incoming message data is automatically cached (saved) to disk at runtime, so this method should NOT be
            used for data saving purposes.

            The data processing logic implemented via this method should be optimized for runtime speed, as processing
            the data hogs the communication process, reducing its throughput.

        Args:
            message: The ModuleState or ModuleData instance that stores the message data received from the interfaced
                hardware module instance.
        """
        raise NotImplementedError

    def send_command(self, command: np.uint8, *, noblock: np.bool_, repetition_delay: np.uint32 = _ZERO_LONG) -> None:
        """Packages the input command data into the appropriate message structure and sends it to the managed hardware
        module.

        Notes:
            This method caches up to 32 unique command messages in the instance-specific LRU cache to speed up sending
            previously created command messages.

        Args:
            command: The id-code of the command to execute.
            noblock: Determines whether the microcontroller managing the hardware module is allowed to concurrently
                execute other commands while executing the requested command.
            repetition_delay: The time, in microseconds, to wait before repeating the command. If set to 0, the command
                is only executed once.
        """
        # Prevents interfacing with the microcontroller until the communication is initialized.
        if self._input_queue is None or self._create_command_message is None:
            message = (
                f"Unable to send the command message to the module {self._module_id} of type "
                f"{self._module_type}. Use the module interface instance to initialize the MicroControllerInterface "
                f"instance to enable constructing and sending messages to the microcontroller. Note: at this time only "
                f"the main runtime process can construct and send messages to the microcontroller."
            )
            console.error(message=message, error=RuntimeError)

        command_message = self._create_command_message(command=command, noblock=noblock, cycle_delay=repetition_delay)
        self._input_queue.put(command_message)

    def send_parameters(
        self, parameter_data: tuple[np.unsignedinteger[Any] | np.signedinteger[Any] | np.bool_ | np.floating[Any], ...]
    ) -> None:
        """Packages the input parameter tuple into the appropriate message structure and sends it to the managed
        hardware module.

        Notes:
            This method caches up to 16 unique parameter messages in the instance-specific LRU cache to speed up sending
            previously created parameter messages.

        Args:
            parameter_data: A tuple that contains the values for the PC-addressable parameters of the target hardware
                module. Note, the parameters must appear in the same order and use the same data-types as the module's
                parameter structure on the microcontroller.
        """
        # Prevents interfacing with the microcontroller until the communication is initialized.
        if self._input_queue is None or self._create_parameters_message is None:
            message = (
                f"Unable to send the runtime parameters update message to the module {self._module_id} of type "
                f"{self._module_type}. Use the module interface instance to initialize the MicroControllerInterface "
                f"instance to enable constructing and sending messages to the microcontroller. Note: at this time only "
                f"the main runtime process can construct and send messages to the microcontroller."
            )
            console.error(message=message, error=RuntimeError)

        # Creates or queries the command message object from the instance-specific LRU cache and submits it to the
        # microcontroller.
        self._input_queue.put(self._create_parameters_message(parameter_data=parameter_data))

    def reset_command_queue(self) -> None:
        """Instructs the microcontroller to clear the managed hardware module's command queue."""
        # Prevents interfacing with the microcontroller until the communication is initialized.
        if self._input_queue is None:
            message = (
                f"Unable to send the dequeue command message to the module {self._module_id} of type "
                f"{self._module_type}. Use the module interface instance to initialize and start the "
                f"MicroControllerInterface instance to enable constructing and sending messages to the microcontroller."
            )
            console.error(message=message, error=RuntimeError)

        self._input_queue.put(self._dequeue_command)

    def set_input_queue(self, input_queue: MPQueue) -> None:  # type: ignore[type-arg]  # MPQueue is not generic.
        """Overwrites the '_input_queue' instance attribute with the reference to the provided Queue object."""
        self._input_queue = input_queue

    @property
    def module_type(self) -> np.uint8:
        """Returns the id-code of the type (family) of modules managed by this interface instance."""
        return self._module_type

    @property
    def module_id(self) -> np.uint8:
        """Returns the id-code of the specific module instance managed by this interface instance."""
        return self._module_id

    @property
    def type_id(self) -> np.uint16:
        """Returns the unique 16-bit unsigned integer value that results from combining the bits of the type-code and
        the id-code of the managed module instance.
        """
        return self._type_id

    @property
    def data_codes(self) -> set[np.uint8]:
        """Returns the set of message event-codes that require online processing during runtime."""
        return self._data_codes

    @property
    def error_codes(self) -> dict[np.uint8, str]:
        """Returns the mapping of message event codes that trigger runtime errors to their explanations."""
        return self._error_codes

    @property
    def name(self) -> str:
        """Returns the human-readable name of this module interface instance."""
        return self._name

    def _create_command_message_implementation(
        self,
        command: np.uint8,
        *,
        noblock: np.bool_,
        cycle_delay: np.uint32,
    ) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Creates the command message object using the input parameters.

        This worker method is passed to the LRU cache wrapper to prevent recreating repeatedly used command objects at
        runtime.

        Args:
            command: The id-code of the command to execute.
            noblock: Determines whether the microcontroller managing the hardware module is allowed to concurrently
                execute other commands while executing the requested command.
            cycle_delay: The time, in microseconds, to wait before repeating the command.
        """
        if cycle_delay == _ZERO_LONG:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=_ZERO_BYTE,
                command=command,
                noblock=noblock,
            )
        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=_ZERO_BYTE,
            command=command,
            noblock=noblock,
            cycle_delay=cycle_delay,
        )

    def _create_parameters_message_implementation(
        self,
        parameter_data: tuple[np.number[Any] | np.bool_, ...],
    ) -> ModuleParameters:
        """Creates the parameter message object using the input parameters.

        This worker method is passed to the LRU cache wrapper to prevent recreating repeatedly used parameter objects at
        runtime.

        Args:
            parameter_data: A tuple that contains the values for the PC-addressable parameters of the target hardware
                module.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=_ZERO_BYTE,
            parameter_data=parameter_data,
        )


class MicroControllerInterface:
    """Interfaces with the hardware module instances managed by the Arduino or Teensy microcontroller running the
    ataraxis-micro-controller library.

    This class binds each hardware module managed by the microcontroller to its user-facing interface implemented via
    this library. It abstracts all necessary steps to bidirectionally communicate with the microcontroller and log the
    incoming and outgoing message data to disk.

    Notes:
        An instance of this class has to be instantiated for each microcontroller active at the same time.

        Initializing this class does not automatically start the communication. Call the start() method of an
        initialized class instance to start the communication with the microcontroller.

        Initializing MicroControllerInterface also completes the configuration of all ModuleInterface instances passed
        to the instance during initialization.

    Args:
        controller_id: The unique identifier code of the managed microcontroller. This value is used to identify the
            microcontroller in all output streams (e.g., log files and terminal messages).
        data_logger: An initialized DataLogger instance used to log all incoming and outgoing messages handled by
            this MicroControllerInterface instance.
        module_interfaces: The custom hardware module interfaces for the hardware module instance managed by the
            microcontroller. Note, each module instance requires a unique interface instance.
        buffer_size: The size, in bytes, of the buffer used by the microcontroller's serial communication interface.
            Usually, this information is available from the microcontroller's manufacturer (UART / USB controller
            specification). Must be at least 9 bytes. The value bounds the size of the payloads the PC transmits,
            while reception is bounded by the 254-byte ceiling the COBS encoding imposes.
        port: The name of the serial port to connect to, e.g.: 'COM3' or '/dev/ttyUSB0'. Use the 'axci id' CLI
            command to discover the available microcontrollers and their respective communication port names.
        name: A colloquial human-readable name for this microcontroller (e.g., 'actor_controller'). Written to the
            microcontroller manifest file alongside the controller_id to identify this controller.
        baudrate: The baudrate to use for communication if the microcontroller uses the UART interface. Must match
            the value used by the microcontroller. This parameter is ignored when using the USB interface.
        keepalive_interval: The interval, in milliseconds, at which to send the keepalive messages to the
            microcontroller. Setting this argument to 0 disables keepalive messaging functionality.

    Attributes:
        _started: Tracks whether the communication process has been started.
        _shutdown_lock: Stores the lock that serializes the shutdown sequence between stop() and the watchdog thread,
            so exactly one of the two retires the instance.
        _controller_id: Stores the id of the managed microcontroller.
        _name: Stores the human-readable name of this microcontroller instance.
        _port: Stores the serial port used for microcontroller communication.
        _baudrate: Stores the baudrate used during communication over the UART serial interface.
        _buffer_size: Stores the microcontroller's serial buffer size, in bytes.
        _modules: Stores ModuleInterface instances managed by this MicroControllerInterface.
        _logger_queue: The Multiprocessing Queue object used to pipe log data to the DataLogger core(s).
        _log_directory: Stores the output directory used by the DataLogger to save temporary log entries and the final
            .npz log archive.
        _multiprocessing_manager: The multiprocessing Manager used to initialize and manage the Queue instance that
            pipes command and parameter messages to the communication process.
        _input_queue: The multiprocessing Queue used to pipe the data to be sent to the microcontroller to
            the remote communication process.
        _terminator_array: Stores the SharedMemoryArray instance used to control the remote communication process.
        _communication_process: Stores the Process instance that runs the communication cycle.
        _watchdog_thread: Stores the thread used to monitor the runtime status of the remote communication process.
        _reset_command: Stores the pre-packaged Kernel-addressed command that resets the managed microcontroller to the
            default state.
        _keepalive_interval: Stores the keepalive interval in milliseconds.

    Raises:
        TypeError: If any of the input arguments are not of the expected type.
        ValueError: If two ModuleInterface instances share the same combined module type-code and id-code.
    """

    # Pre-packages the user-addressable Kernel reset command into a class attribute. Since the command is known and
    # fixed at class initialization, it only needs to be defined once.
    _reset_command = KernelCommand(
        command=np.uint8(KernelCommandCodes.RESET_CONTROLLER.value),
        return_code=np.uint8(_RuntimeParameters.DEFAULT_RETURN_CODE.value),
    )

    def __init__(
        self,
        controller_id: np.uint8,
        data_logger: DataLogger,
        module_interfaces: tuple[ModuleInterface, ...],
        buffer_size: int,
        port: str,
        name: str,
        baudrate: int = 115200,
        keepalive_interval: int = 0,
    ) -> None:
        # Initializes the started tracker first to avoid issues during __del__ runtime if the class is not able to
        # initialize.
        self._started: bool = False

        # Serializes the shutdown sequence. stop() and the watchdog thread both clear the started flag and then
        # release the terminator array, and the flag alone cannot separate them, since each side reads it several
        # statements before it touches the array. The array's own lock does not cover this, as it guards element
        # access while disconnect() and destroy() take no lock at all.
        self._shutdown_lock: Lock = Lock()

        self._multiprocessing_manager: SyncManager = Manager()  # The manager is terminated by the __del__ method.

        # Ensures that input arguments have valid types. Only checks the arguments that are not verified by downstream
        # classes.
        if not isinstance(controller_id, np.uint8) or not 1 <= controller_id <= _MAXIMUM_BYTE_VALUE:
            message = (
                f"Unable to initialize the MicroControllerInterface instance. Expected an unsigned integer value "
                f"between 1 and 255 for the 'controller_id' argument, but encountered {controller_id} of type "
                f"{type(controller_id).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(module_interfaces, tuple) or not module_interfaces:
            message = (
                f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
                f"{controller_id}. Expected a non-empty tuple of ModuleInterface instances for "
                f"'module_interfaces' argument, but encountered {module_interfaces} of type "
                f"{type(module_interfaces).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not all(isinstance(module, ModuleInterface) for module in module_interfaces):
            message = (
                f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
                f"{controller_id}. All items in 'module_interfaces' tuple must be ModuleInterface instances."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(data_logger, DataLogger):
            message = (
                f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
                f"{controller_id}. Expected an initialized DataLogger instance for 'data_logger' argument, but "
                f"encountered {data_logger} of type {type(data_logger).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(buffer_size, int) or buffer_size < _MINIMUM_SERIAL_BUFFER_SIZE:
            message = (
                f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
                f"{controller_id}. Expected an integer value of at least {_MINIMUM_SERIAL_BUFFER_SIZE} for the "
                f"'buffer_size' argument, but encountered {buffer_size} of type {type(buffer_size).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(keepalive_interval, int) or keepalive_interval < 0:
            message = (
                f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
                f"{controller_id}. Expected a non-negative integer value for the 'keepalive_interval' argument, but "
                f"encountered {keepalive_interval} of type {type(keepalive_interval).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(name, str) or not name:
            message = (
                f"Unable to initialize the MicroControllerInterface instance for the microcontroller with id "
                f"{controller_id}. Expected a non-empty string for the 'name' argument, but encountered {name!r} of "
                f"type {type(name).__name__}."
            )
            console.error(message=message, error=TypeError)

        self._controller_id: np.uint8 = controller_id
        self._name: str = name

        # SerialCommunication parameters. This is used to initialize the communication in the remote process.
        self._port: str = port
        self._baudrate: int = baudrate
        self._buffer_size: int = buffer_size

        self._modules: tuple[ModuleInterface, ...] = tuple(module_interfaces)

        # Reads the queue and the output directory off the logger instance rather than re-deriving them, so the
        # interface and the logger always resolve to the same destination.
        self._logger_queue: MPQueue = data_logger.input_queue  # type: ignore[type-arg]  # MPQueue is not generic.
        self._log_directory: Path = data_logger.output_directory

        # Sets up the assets used to deploy the communication runtime on a separate core and bidirectionally transfer
        # data between the communication process and the main process managing the overall runtime. The suppression
        # below is required because MPQueue is not generic and because the manager returns a queue proxy rather than a
        # multiprocessing.Queue.
        self._input_queue: MPQueue = self._multiprocessing_manager.Queue()  # type: ignore[assignment, type-arg]
        self._terminator_array: SharedMemoryArray | None = None
        self._communication_process: Process | None = None
        self._watchdog_thread: Thread | None = None

        # Initializes class attributes used to track the current microcontroller configuration and communication
        # runtime parameters.
        self._keepalive_interval = keepalive_interval

        # Verifies that all input ModuleInterface instances have a unique type+id combination and configures each
        # module to use the input queue instantiated above to submit command and parameter messages to the
        # microcontroller.
        processed_type_ids: set[np.uint16] = set()  # This is used to ensure each instance has a unique type+id pair.
        for module in self._modules:
            # If the module's combined type + id code is already inside the processed_types_id set, this means another
            # module with the same exact type and ID combination has already been processed.
            if module.type_id in processed_type_ids:
                message = (
                    f"Unable to initialize the MicroControllerInterface instance for the microcontroller with "
                    f"id {controller_id}. Encountered two module interface instances with the same type-code "
                    f"({module.module_type}) and id-code ({module.module_id}), which is not allowed. Each type and id "
                    f"combination can only be used by a single module interface instance."
                )
                console.error(message=message, error=ValueError)

            processed_type_ids.add(module.type_id)

            # Overwrites the attributes for each processed ModuleInterface with valid data. This effectively binds some
            # data and functionality realized through the main interface to each module interface.
            module.set_input_queue(input_queue=self._input_queue)

        # Writes the controller and its modules to the manifest file in the DataLogger output directory. This enables
        # downstream log processing tools to identify which archives were produced by this library.
        module_sources = tuple(
            ModuleSourceData(module_type=int(module.module_type), module_id=int(module.module_id), name=module.name)
            for module in self._modules
        )
        write_microcontroller_manifest(
            log_directory=self._log_directory,
            controller_id=int(self._controller_id),
            controller_name=self._name,
            modules=module_sources,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the MicroControllerInterface instance."""
        return (
            f"MicroControllerInterface(controller_id={self._controller_id}, name='{self._name}', "
            f"usb_port={self._port}, baudrate={self._baudrate}, started={self._started}, "
            f"keepalive_interval={self._keepalive_interval})"
        )

    def __del__(self) -> None:
        """Ensures that all resources are properly released when the instance is garbage-collected."""
        self.stop()
        self._multiprocessing_manager.shutdown()

    def reset_controller(self) -> None:
        """Resets the managed microcontroller to use the default hardware and software parameters."""
        self._input_queue.put(self._reset_command)

    @property
    def controller_id(self) -> np.uint8:
        """Returns the unique identifier code of the managed microcontroller."""
        return self._controller_id

    @property
    def name(self) -> str:
        """Returns the human-readable name of this microcontroller interface instance."""
        return self._name

    @property
    def modules(self) -> tuple[ModuleInterface, ...]:
        """Returns the tuple of ModuleInterface instances managed by this MicroControllerInterface."""
        return self._modules

    def start(self) -> None:  # pragma: no cover - spawns the communication process against a live serial port.
        """Starts the instance's communication process and begins interfacing with the microcontroller.

        Notes:
            As part of this method runtime, the interface verifies the target microcontroller's configuration to
            ensure it matches the interface's configuration.

        Raises:
            RuntimeError: If the instance fails to initialize the communication process.
        """
        # Prevents restarting an already running communication process
        if self._started:
            return

        # This timer is used to forcibly terminate processes that stall at initialization.
        initialization_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

        # Instantiates the shared memory array used to control the runtime of the communication Process.
        # Index 0 = terminator, index 1 = initialization status tracker
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._controller_id}_terminator_array",  # Uses class id with an additional specifier
            prototype=np.zeros(shape=2, dtype=np.uint8),
            exists_ok=True,  # Automatically recreates the buffer if it already exists
        )

        # Binds runtime arguments to the communication cycle function before passing it to the Process instance.
        runtime_cycle_with_args = partial(
            self._runtime_cycle,
            controller_id=self._controller_id,
            controller_name=self._name,
            module_interfaces=self._modules,
            input_queue=self._input_queue,
            logger_queue=self._logger_queue,
            terminator_array=self._terminator_array,
            port=self._port,
            baudrate=self._baudrate,
            buffer_size=self._buffer_size,
            keepalive_interval=self._keepalive_interval,
        )

        # Sets up the communication process. This process continuously cycles through the communication loop until
        # terminated, enabling bidirectional communication with the controller.
        self._communication_process = Process(
            target=runtime_cycle_with_args,
            daemon=True,
        )
        self._communication_process.start()

        initialization_timer.reset()
        # Blocks until the microcontroller has finished all initialization steps or encounters an initialization error.
        while self._terminator_array[1] != 1:
            if (
                not self._communication_process.is_alive()
                or initialization_timer.elapsed > _RuntimeParameters.PROCESS_INITIALIZATION_TIMEOUT.value
            ):
                # Ensures proper resource cleanup before terminating the process runtime, if this error is triggered:
                self._terminator_array[0] = 1

                # Waits for at most _RuntimeParameters.PROCESS_TERMINATION_TIMEOUT seconds and gives up on the join
                # once that period elapses, to prevent deadlocks.
                self._communication_process.join(timeout=_RuntimeParameters.PROCESS_TERMINATION_TIMEOUT.value)

                # Disconnects from the shared memory array and destroys its shared buffer.
                self._terminator_array.disconnect()
                self._terminator_array.destroy()

                message = (
                    f"Unable to start the MicroControllerInterface with id {self._controller_id}. The microcontroller "
                    f"communication process has unexpectedly shut down or stalled for more than "
                    f"{_RuntimeParameters.PROCESS_INITIALIZATION_TIMEOUT.value} seconds during initialization. "
                    f"This likely indicates a problem with the SerialCommunication instance managed by this process."
                )
                console.error(error=RuntimeError, message=message)

        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

        # Issues the global reset command. This ensures that the controller always starts with 'default' parameters for
        # Teensy microcontroller boards that do not reset upon communication interface connection cycling.
        self.reset_controller()

        self._started = True

    def stop(self) -> None:
        """Stops the instance's communication process and releases all reserved resources.

        Notes:
            The shutdown is claimed under a lock the watchdog thread takes as well, so exactly one of the two performs
            the teardown. The lock is released before this method joins that thread, since the watchdog acquires the
            same lock and holding it across the join would leave each side waiting on the other.
        """
        # Claims the shutdown. A watchdog that reached its own teardown first leaves the flag clear, which takes the
        # early return below and keeps this method away from the terminator array that thread has already released.
        with self._shutdown_lock:
            if not self._started or self._terminator_array is None:
                return

            # Resets the microcontroller to ensure all hardware is set to default states that are assumed to be safe.
            self.reset_controller()

            # This inactivates the watchdog thread monitoring, ensuring it does not err when the processes are
            # terminated.
            self._started = False

            # Emits the process shutdown signal.
            self._terminator_array[0] = 1

        # Joins the communication process and the watchdog thread before releasing the shared memory array below, as
        # both read the array during their own shutdown and would fault on a buffer destroyed under them.
        if self._communication_process is not None:
            self._communication_process.join(timeout=_RuntimeParameters.PROCESS_TERMINATION_TIMEOUT.value)

        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=_RuntimeParameters.PROCESS_TERMINATION_TIMEOUT.value)

        # Disconnects from the shared memory array and destroys its shared buffer.
        self._terminator_array.disconnect()
        self._terminator_array.destroy()

    def _watchdog(self) -> None:
        """Monitors the communication process to ensure it remains alive during runtime.

        Raises RuntimeErrors if it detects that the communication process has prematurely shut down. Verifies the
        process state in 20-millisecond cycles and releases the GIL between state verifications.

        Notes:
            If the method detects that the communication process has terminated prematurely, it carries out the
            necessary resource cleanup steps before raising the error. The watchdog runs in a daemon thread, so the
            error surfaces as a traceback on stderr while the main process continues.
        """
        timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # The watchdog function runs until the global shutdown signal is emitted.
        while self._terminator_array is not None and not self._terminator_array[0]:
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay(delay=_RuntimeParameters.WATCHDOG_INTERVAL.value, allow_sleep=True, block=False)

            # Only monitors the Process state after the communication is initialized via the start() method.
            if not self._started:
                continue

            if self._communication_process is not None and not self._communication_process.is_alive():
                # Claims the shutdown under the lock stop() takes as well, so exactly one of the two retires the
                # instance. A concurrent stop() can claim it between the liveness check above and this acquisition,
                # and the claimant owns every step below. Re-reading the flag here keeps this thread from tearing the
                # instance down a second time and reporting a shutdown that already happened.
                with self._shutdown_lock:
                    if not self._started or self._terminator_array is None:
                        return

                    # Prevents the __del__ method from running stop(), as the code below terminates all assets.
                    self._started = False

                    # Activates the shutdown flag.
                    self._terminator_array[0] = 1

                    # The process should already be terminated, but there are no downsides to making sure it is dead.
                    self._communication_process.join(timeout=_RuntimeParameters.PROCESS_TERMINATION_TIMEOUT.value)

                    # Disconnects from the shared memory array and destroys its shared buffer.
                    self._terminator_array.disconnect()
                    self._terminator_array.destroy()

                # Raises outside the lock, since a traceback unwinding with the lock held would leave a concurrent
                # stop() waiting on a lock this thread never releases.
                message = (
                    f"The communication process of the MicroControllerInterface with id {self._controller_id} has been "
                    f"prematurely shut down. This likely indicates that the process has encountered a runtime error "
                    f"that terminated the process."
                )
                console.error(message=message, error=RuntimeError)

    @staticmethod
    def _verify_microcontroller_communication(
        serial_communication: SerialCommunication,
        timeout_timer: PrecisionTimer,
        controller_id: np.uint8,
        module_interfaces: tuple[ModuleInterface, ...],
        terminator_array: SharedMemoryArray,
    ) -> None:
        """Verifies that the managed microcontroller and the interface instance have a matching configuration.

        Args:
            serial_communication: The SerialCommunication instance used to communicate with the microcontroller.
            timeout_timer: The PrecisionTimer instance used to prevent verification from stalling.
            controller_id: The expected ID code of the microcontroller.
            module_interfaces: The interface instances for all managed hardware modules.
            terminator_array: The SharedMemoryArray instance used to control the runtime of the communication process.

        Raises:
            RuntimeError: If the method is unable to communicate with the microcontroller.
            ValueError: If the microcontroller and the interface instance do not have matching configurations.
        """
        # Constructs Kernel-addressed commands used to verify that the interface and the microcontroller have matching
        # configurations.
        identify_controller_command = KernelCommand(
            command=np.uint8(KernelCommandCodes.IDENTIFY_CONTROLLER.value),
            return_code=np.uint8(_RuntimeParameters.DEFAULT_RETURN_CODE.value),
        )
        identify_modules_command = KernelCommand(
            command=np.uint8(KernelCommandCodes.IDENTIFY_MODULES.value),
            return_code=np.uint8(_RuntimeParameters.DEFAULT_RETURN_CODE.value),
        )

        # Blocks until the microcontroller responds with its identification code.
        attempt = 0
        response = None
        while attempt < _RuntimeParameters.MAXIMUM_COMMUNICATION_ATTEMPTS.value and not isinstance(
            response, ControllerIdentification
        ):
            # Sends microcontroller identification command. This command requests the microcontroller to return its
            # id code.
            serial_communication.send_message(message=identify_controller_command)
            attempt += 1

            # Waits for response with timeout
            timeout_timer.reset()
            while timeout_timer.elapsed < _RuntimeParameters.MICROCONTROLLER_ID_TIMEOUT.value:
                response = serial_communication.receive_message()
                if isinstance(response, ControllerIdentification):
                    break

        # If the microcontroller did not respond to the identification request, raises an error.
        if not isinstance(response, ControllerIdentification):
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. The "
                f"microcontroller did not respond to the identification request after "
                f"{_RuntimeParameters.MAXIMUM_COMMUNICATION_ATTEMPTS.value} attempts."
            )
            console.error(message=message, error=RuntimeError)

        # If a response is received, but the ID contained in the received message does not match the expected ID,
        # raises an error
        if response.controller_id != controller_id:
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. Expected "
                f"{controller_id} in response to the controller identification request, but "
                f"received a non-matching id {response.controller_id}."
            )
            console.error(message=message, error=ValueError)

        # Verifies that the microcontroller manages the hardware module instances expected by the hardware
        # module interfaces
        serial_communication.send_message(message=identify_modules_command)
        timeout_timer.reset()
        module_type_ids = []
        while timeout_timer.elapsed < _RuntimeParameters.MICROCONTROLLER_ID_TIMEOUT.value:
            # Receives the message. If the message is a module type+id code, adds it to the storage list
            response = serial_communication.receive_message()
            if isinstance(response, ModuleIdentification):
                module_type_ids.append(response.module_type_id)

                # Keeps the loop running as long as messages keep coming in within expected intervals.
                timeout_timer.reset()

        # If no response was received from the microcontroller, raises an error
        if not module_type_ids:
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. The "
                f"microcontroller did not respond to the module identification request."
            )
            console.error(message=message, error=RuntimeError)

        # The microcontroller may have more modules than the number of managed interfaces, but it can never have fewer
        # modules than interfaces.
        if len(module_type_ids) < len(module_interfaces):
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. The microcontroller "
                f"does not manage all of the hardware modules expected by the interfaces passed to the "
                f"MicroControllerInterface instance."
            )
            console.error(message=message, error=ValueError)

        # Ensures that all type_id codes are unique on the microcontroller.
        if len(module_type_ids) != len(set(module_type_ids)):
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. The microcontroller "
                f"contains multiple module instances with the same type + id code combination. All modules must use "
                f"a unique combination of type and id codes."
            )
            console.error(message=message, error=ValueError)

        # Ensures that each module interface has a matching hardware module on the microcontroller
        for module in module_interfaces:
            if module.type_id not in module_type_ids:
                message = (
                    f"Unable to initialize the communication with the microcontroller {controller_id}. "
                    f"The interface instance for the module with type {module.module_type} and id "
                    f"{module.module_id} codes does not have a matching hardware module instance managed by the "
                    f"microcontroller."
                )
                console.error(message=message, error=ValueError)

        # Reports that the communication class has been successfully initialized.
        terminator_array[1] = 1

    @staticmethod
    def _parse_kernel_data(
        controller_id: np.uint8,
        controller_name: str,
        incoming_data: KernelState | KernelData,
    ) -> None:
        """Parses incoming KernelState and KernelData messages and, if necessary, raises runtime errors.

        Args:
            controller_id: The ID of the interfaced microcontroller.
            controller_name: The human-readable name of the interfaced microcontroller.
            incoming_data: The KernelState or KernelData message to be parsed.

        Raises:
            RuntimeError: If the incoming Kernel message carries a status code that reports a fault.
        """
        description = describe_kernel_event(
            message=incoming_data,
            controller_id=controller_id,
            controller_name=controller_name,
        )
        if description is not None:
            console.error(message=description, error=RuntimeError)

    @staticmethod
    def _parse_service_module_data(
        controller_id: np.uint8,
        controller_name: str,
        module_name: str,
        incoming_data: ModuleState | ModuleData,
    ) -> None:
        """Parses incoming service ModuleState and ModuleData messages and, if necessary, raises runtime errors.

        Notes:
            Service messages use the event code range the microcontroller reserves for the base Module class, which
            spans every code below MINIMUM_CUSTOM_STATUS_CODE.

        Args:
            controller_id: The ID of the interfaced microcontroller.
            controller_name: The human-readable name of the interfaced microcontroller.
            module_name: The human-readable name of the hardware module that sent the message.
            incoming_data: The ModuleState or ModuleData message to be parsed.

        Raises:
            RuntimeError: If the incoming Module message carries a service status code that reports a fault.
        """
        description = describe_module_event(
            message=incoming_data,
            controller_id=controller_id,
            controller_name=controller_name,
            module_name=module_name,
        )
        if description is not None:
            console.error(message=description, error=RuntimeError)

    @staticmethod
    def _runtime_cycle(
        controller_id: np.uint8,
        controller_name: str,
        module_interfaces: tuple[ModuleInterface, ...],
        input_queue: MPQueue,  # type: ignore[type-arg]  # MPQueue is not generic.
        logger_queue: MPQueue,  # type: ignore[type-arg]  # MPQueue is not generic.
        terminator_array: SharedMemoryArray,
        port: str,
        baudrate: int,
        buffer_size: int,
        keepalive_interval: int,
    ) -> None:
        """Aggregates the logic for bidirectionally communicating with the interfaced microcontroller during runtime.

        This method is designed to run in a remote Process. It encapsulates the steps for sending and receiving the
        data from the connected microcontroller.

        Args:
            controller_id: The unique identifier of the interfaced microcontroller.
            controller_name: The human-readable name of the interfaced microcontroller, used to identify it in the
                error messages this method raises.
            module_interfaces: The custom hardware module interfaces for the hardware module instance managed by the
                microcontroller.
            input_queue: The multiprocessing queue used to issue commands to the microcontroller.
            logger_queue: The multiprocessing queue used to buffer and pipe received and outgoing messages to be
                logged (saved) to disk to the logger process.
            terminator_array: The shared memory array used to control the communication process runtime.
            port: The serial port to use for communicating with the microcontroller.
            baudrate: The baudrate to use when communicating with microcontrollers using the UART serial interface.
            buffer_size: The size of the microcontroller's serial buffer.
            keepalive_interval: The interval (in milliseconds) at which to send the keepalive messages to the
                microcontroller.
        """
        # Constructs Kernel-addressed command used to verify that the microcontroller-PC communication is active during
        # runtime. This is used to detect communication issues and problems with the microcontroller during runtime.
        keepalive_command = KernelCommand(
            command=np.uint8(KernelCommandCodes.KEEPALIVE.value),
            return_code=np.uint8(_RuntimeParameters.KEEPALIVE_RETURN_CODE.value),
        )

        # Initializes the timer used during initialization to abort stale initialization attempts and during runtime to
        # support the keepalive functionality.
        timeout_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # Connects to the terminator array. This is done early, as the terminator_array is used to track the
        # initialization and runtime status of the process.
        terminator_array.connect()

        # Pre-creates the assets used to optimize the communication runtime cycling. These assets are filled below to
        # support efficient interaction between the SerialCommunication instance and the module interface instances.
        processing_map: dict[np.uint16, ModuleInterface] = {}

        # Every managed module contributes its name, as service messages identify their sender before the processing
        # map is consulted and are emitted by modules that declare no data or error codes.
        module_names: dict[np.uint16, str] = {module.type_id: module.name for module in module_interfaces}

        # Records the interfaces whose remote assets are live, so the shutdown below terminates the assets that exist
        # and leaves an interface that raised during its own initialization alone.
        initialized_modules: list[ModuleInterface] = []

        # The initialization steps below run under the same guard as the communication loop, as each of them can raise
        # after some interfaces have already claimed their remote assets.
        try:
            for module in module_interfaces:
                # For each module, initializes the assets that need to be configured / created inside the remote
                # Process.
                module.initialize_remote_assets()
                initialized_modules.append(module)

                # If the interface is configured to process incoming data or raise runtime errors, maps its type+id
                # combined code to the interface instance. This is used to quickly find the module interface instance
                # addressed by incoming data, so that it can handle the data or error message.
                if module.data_codes or module.error_codes:
                    processing_map[module.type_id] = module

            # Initializes the serial communication class and connects to the target microcontroller.
            serial_communication = SerialCommunication(
                port=port,
                controller_id=controller_id,
                logger_queue=logger_queue,
                baudrate=baudrate,
                microcontroller_serial_buffer_size=buffer_size,
            )

            # Verifies that the microcontroller and the interface instance are configured correctly to support the
            # runtime.
            MicroControllerInterface._verify_microcontroller_communication(
                serial_communication=serial_communication,
                module_interfaces=module_interfaces,
                controller_id=controller_id,
                timeout_timer=timeout_timer,
                terminator_array=terminator_array,
            )

            # Tracks whether the microcontroller has responded to the last keepalive command sent from the PC.
            keepalive_response_received = True  # Must be initialized to True.

            # Initializes the main communication loop. This loop runs until the exit conditions are encountered.
            # The exit conditions for the loop require the first variable in the terminator_array to be set to True
            # and the main input queue of the interface to be empty. This ensures that all queued commands issued from
            # the central process are fully carried out before the communication is terminated.
            timeout_timer.reset()
            while not terminator_array[0] or not input_queue.empty():
                # Main data sending loop. The method sequentially retrieves the queued messages and sends them to the
                # microcontroller.
                while not input_queue.empty():
                    # Transmits the data to the microcontroller. Expects that the queue always yields valid messages.
                    serial_communication.send_message(message=input_queue.get())

                # Keepalive messaging. Sends a keepalive message every keepalive_interval milliseconds to ensure that
                # the microcontroller-PC communication is functional. Each time a keepalive message is sent, the
                # keepalive response tracker and the timer are reset to ensure that the microcontroller responds before
                # the next keepalive cycle iteration.
                if 0 < keepalive_interval <= timeout_timer.elapsed:
                    # If the microcontroller does not respond to the keepalive message, it is likely that the
                    # communication is broken or that the microcontroller has encountered a fatal runtime error.
                    if not keepalive_response_received:
                        # While this is unlikely to succeed, instructs the microcontroller to reset itself before
                        # ending the runtime.
                        serial_communication.send_message(message=MicroControllerInterface._reset_command)
                        message = (
                            f"Communication with the microcontroller {controller_id} is interrupted. The "
                            f"microcontroller did not respond to the keepalive message within the expected interval "
                            f"of {keepalive_interval} milliseconds."
                        )
                        console.error(message=message, error=RuntimeError)

                    # Otherwise, sends another keepalive message and resets the response tracker and the timeout timer.
                    serial_communication.send_message(message=keepalive_command)
                    keepalive_response_received = False
                    timeout_timer.reset()

                incoming_data = serial_communication.receive_message()

                # If no data is available advances to the next cycle iteration
                if incoming_data is None:
                    continue

                # Currently, the only explicitly supported type of reception feedback messaging is the keepalive
                # communication cycle. All keepalive messages use response code 255.
                if (
                    isinstance(incoming_data, ReceptionCode)
                    and incoming_data.reception_code == _RuntimeParameters.KEEPALIVE_RETURN_CODE.value
                ):
                    keepalive_response_received = True  # Indicates that the response code was received

                # Converts valid KernelData and State messages into errors. This is used to raise runtime errors when
                # an appropriate error message is transmitted from the microcontroller. This clause does not evaluate
                # non-error codes.
                elif isinstance(incoming_data, (KernelData, KernelState)):
                    MicroControllerInterface._parse_kernel_data(
                        incoming_data=incoming_data,
                        controller_id=controller_id,
                        controller_name=controller_name,
                    )

                # Handles Module-addressed messages. The event codes below the custom range are reserved for the base
                # Module class of the firmware and are resolved by this library. The codes at or above that bound are
                # assigned by module developers to communicate states and errors.
                elif isinstance(incoming_data, (ModuleState, ModuleData)):
                    # Reads the combined type and id code of the incoming data. This is used to find the specific
                    # ModuleInterface to which the message is addressed and, if necessary, invoke interface-specific
                    # additional processing methods.
                    target_type_id: np.uint16 = incoming_data.type_id

                    # Binds the event code once, as the branches below test it against three separate collections.
                    event = incoming_data.event

                    if event < MINIMUM_CUSTOM_STATUS_CODE:
                        MicroControllerInterface._parse_service_module_data(
                            incoming_data=incoming_data,
                            controller_id=controller_id,
                            controller_name=controller_name,
                            module_name=module_names.get(target_type_id, _UNKNOWN_MODULE_NAME),
                        )
                    else:
                        # If the interface addressed by the message is not configured to raise errors or process
                        # the data, ends processing. Otherwise, binds the reference to the targeted interface.
                        target_module = processing_map.get(target_type_id)
                        if target_module is None:
                            continue

                        # If the incoming message contains an event code matching one of the interface's error-codes,
                        # raises an error that carries the explanation the interface registered for that code.
                        explanation = target_module.error_codes.get(event)
                        if explanation is not None:
                            message = describe_custom_module_error(
                                message=incoming_data,
                                controller_id=controller_id,
                                controller_name=controller_name,
                                module_name=target_module.name,
                                description=explanation,
                            )
                            console.error(message=message, error=RuntimeError)

                        # Otherwise, if the incoming message is not an error and contains an event-code matching
                        # one of the interface's data-codes, calls the data processing method.
                        if event in target_module.data_codes:
                            target_module.process_received_data(message=incoming_data)

        # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
        # before re-raising the exception to terminate the process.
        except Exception as runtime_error:
            sys.stderr.write(str(runtime_error))
            sys.stderr.flush()
            raise

        # Ensures that local assets are always properly terminated
        finally:
            terminator_array.disconnect()

            # Terminates the custom assets of every interface that completed its own initialization. Each termination
            # is guarded, as an interface that raises here would otherwise strand the assets of every interface behind
            # it and replace the exception that started the shutdown.
            for module in initialized_modules:
                try:
                    module.terminate_remote_assets()
                except Exception as termination_error:
                    sys.stderr.write(
                        f"Unable to terminate the remote assets of the {module.name} module interface managed by the "
                        f"microcontroller {controller_id}. Encountered the following error: {termination_error}\n"
                    )
                    sys.stderr.flush()


def evaluate_port(port: str, baudrate: int = 115200) -> tuple[int, str | None]:
    """Determines whether the target serial port is connected to an Ataraxis MicroController.

    Args:
        port: The name of the port to evaluate.
        baudrate: The baudrate to use for communication if the microcontroller uses the UART serial interface.

    Returns:
        The microcontroller's unique identifier code when the port is connected to an Ataraxis MicroController, or -1
        otherwise, paired with an error message string when a connection error occurred or None when no error occurred.
    """
    try:
        # Initializes a fake multiprocessing queue to initialize communication.
        fake_queue: MPQueue = MPQueue()  # type: ignore[type-arg]  # MPQueue is not generic.

        # Initializes a timer to prevent stale identification attempts from running forever.
        timeout_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # Opens communication via the target port.
        communication = SerialCommunication(
            controller_id=np.uint8(123),
            microcontroller_serial_buffer_size=8192,
            port=port,
            baudrate=baudrate,
            logger_queue=fake_queue,
        )

        # Requests the microcontroller to identify itself. A valid microcontroller would respond with its ID. Any other
        # asset would either ignore the command or err.
        identify_controller_command = KernelCommand(
            command=np.uint8(KernelCommandCodes.IDENTIFY_CONTROLLER.value),
            return_code=np.uint8(_RuntimeParameters.DEFAULT_RETURN_CODE.value),
        )

        # Blocks until the microcontroller responds with its identification code.
        attempt = 0
        response = None
        while attempt < _RuntimeParameters.MAXIMUM_COMMUNICATION_ATTEMPTS.value and not isinstance(
            response, ControllerIdentification
        ):
            # Sends microcontroller identification command. This command requests the microcontroller to return its
            # id code.
            communication.send_message(message=identify_controller_command)
            attempt += 1

            # Waits for response with timeout
            timeout_timer.reset()
            while timeout_timer.elapsed < _RuntimeParameters.MICROCONTROLLER_ID_TIMEOUT.value:
                response = communication.receive_message()
                if isinstance(response, ControllerIdentification):
                    break

        # If the microcontroller did not respond to the identification request, returns -1 to indicate that the port is
        # likely not connected to a valid ataraxis microcontroller.
        if not isinstance(response, ControllerIdentification):
            return -1, None

        return int(response.controller_id), None

    except Exception as connection_error:
        # Catches any connection-related exceptions and returns an error message instead of propagating the exception.
        # This prevents individual port failures from aborting the entire evaluation process.
        error_type = type(connection_error).__name__
        error_message = str(connection_error) or error_type
        return -1, f"{error_type}: {error_message}"
