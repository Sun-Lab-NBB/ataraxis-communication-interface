"""Demonstrates the implementation of an AXCI-compatible hardware module interface class.

Showcases the process of writing interface classes for custom hardware modules managed by the
ataraxis-micro-controller (AXMC) library and interfaced through the AXCI (this) library. The implementation
demonstrates one of the many possible interface design patterns. The library is designed to work with any class design
and layout, as long as it subclasses the base ModuleInterface class and implements all abstract methods:
initialize_remote_assets, terminate_remote_assets, and process_received_data.

For the best learning experience, it is recommended to review this code side-by-side with the implementation of the
companion TestModule class defined in the ataraxis-micro-controller library:
https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart

See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface#quickstart for more details.
API documentation: https://ataraxis-communication-interface-api.netlify.app/
Authors: Ivan Kondratyev (Inkaros), Jacob Groner
"""

import numpy as np
from ataraxis_communication_interface import ModuleData, ModuleState, ModuleInterface
from ataraxis_data_structures import SharedMemoryArray


class TestModuleInterface(ModuleInterface):
    """Interfaces with the TestModule class from the companion ataraxis-micro-controller library.

    Subclasses the base ModuleInterface class to provide the PC-side interface for the TestModule hardware module
    running on the microcontroller. See the README file for more details about the module type and ID codes.

    Args:
        module_type: The type (family) code of the hardware module.
        module_id: The unique identifier code of the hardware module instance.

    Attributes:
        _shared_memory: Shared memory array used to transfer data between the communication and main processes.
        _previous_pin_state: Tracks the state of the digital output pin managed by the module.
    """

    def __init__(self, module_type: np.uint8, module_id: np.uint8) -> None:
        # Defines the set of event codes that require online processing. When the hardware module sends a message
        # containing one of these event codes to the PC, the interface calls the process_received_data() method to
        # process the received message. In this example, the online processing pipes the received messages to the main
        # process via the shared memory array.
        data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kHigh, kLow, and kEcho.

        # Initializes the superclass using the module-specific parameters.
        super().__init__(
            module_type=module_type,
            module_id=module_id,
            name="test_module",
            data_codes=data_codes,
            error_codes=None,  # The test module does not have any expected error states.
        )

        # Initializes the shared memory array used to transfer data from the remote communication process to the main
        # runtime control process. The shared memory array needs to be connected from both the main and the remote
        # communication processes.
        self._shared_memory: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self.type_id}_shm", prototype=np.zeros(shape=3, dtype=np.uint16), exists_ok=True
        )

        # Tracks the state of the digital output pin managed by the module.
        self._previous_pin_state: bool = False

    def initialize_remote_assets(self) -> None:
        """Initializes non-pickleable assets from the remote communication process before the communication cycle."""
        # Connects to the shared memory array from the remote process.
        self._shared_memory.connect()

    def terminate_remote_assets(self) -> None:
        """Terminates assets initialized at the beginning of the communication runtime from the remote process."""
        # Disconnects the shared memory array from the remote process to prevent runtime errors.
        self._shared_memory.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming module messages as they are received by the PC from the remote communication process."""
        # Event codes 52 and 53 are used to communicate the current state of the output pin managed by the example
        # module. State messages transmit these event-codes, so there is no additional data to parse other than
        # event codes.
        if message.event == 52 or message.event == 53:
            # Code 52 indicates that the pin outputs a HIGH signal, code 53 indicates the pin outputs a LOW signal.
            # If the pin state has changed from HIGH (52) to LOW (53), increments the pulse count stored in the shared
            # memory array.
            if message.event == 53 and self._previous_pin_state:
                self._shared_memory[0] += 1

            # Sets the previous pin state value to match the recorded pin state.
            self._previous_pin_state = True if message.event == 52 else False

        # The module uses code 54 messages to return its echo value to the PC.
        elif isinstance(message, ModuleData) and message.event == 54:
            # The echo value is transmitted by a Data message. In addition to the event code, Data messages include a
            # data_object. Upon reception, the data object is automatically deserialized into the appropriate
            # Python object, so it can be accessed directly.
            self._shared_memory[2] = message.data_object  # Records the received data value to the shared memory.
            self._shared_memory[1] += 1  # Increments the received echo value count.

    def start_shared_memory_array(self) -> None:
        """Connects to the shared memory array from the main process after the communication process starts."""
        self._shared_memory.connect()
        self._shared_memory.enable_buffer_destruction()

    def set_parameters(
        self,
        on_duration: np.uint32,
        off_duration: np.uint32,
        echo_value: np.uint16,
    ) -> None:
        """Packages and sends the input parameter values to the managed TestModule on the microcontroller.

        Args:
            on_duration: The time, in microseconds, to keep the pin HIGH when pulsing.
            off_duration: The time, in microseconds, to keep the pin LOW when pulsing.
            echo_value: The value sent to the PC as part of the echo() command's runtime.
        """
        # The order of parameter values in the tuple must match the order in the hardware module's parameter structure.
        self.send_parameters(parameter_data=(on_duration, off_duration, echo_value))

    def pulse(self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True) -> None:
        """Instructs the managed TestModule to emit a pulse via its output pin.

        Args:
            repetition_delay: The time, in microseconds, to wait before repeating the command. Set to 0 to execute once.
            noblock: Determines whether the microcontroller can execute other commands concurrently.
        """
        self.send_command(
            command=np.uint8(1),
            noblock=np.bool_(noblock),
            repetition_delay=repetition_delay,
        )

    def echo(self, repetition_delay: np.uint32 = np.uint32(0)) -> None:
        """Instructs the managed TestModule to respond with the current value of its echo_value parameter.

        Args:
            repetition_delay: The time, in microseconds, to wait before repeating the command. Set to 0 to execute once.
        """
        self.send_command(
            command=np.uint8(2),
            noblock=np.bool_(False),  # The echo command has no time-delays, so is always blocking.
            repetition_delay=repetition_delay,
        )

    @property
    def shared_memory(self) -> SharedMemoryArray:
        """Returns the shared memory array used to transfer data between the communication and main processes."""
        return self._shared_memory
