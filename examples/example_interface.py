# This example demonstrates the implementation of an AXCI-compatible hardware module interface class.
#
# This file demonstrates the process of writing interface classes for custom hardware modules managed by the
# ataraxis-micro-controller (AXMC) library and interfaced through the AXCI (this) library. This implementation showcases
# one of the many possible interface design patterns. The main advantage of this library, similar to the AXMC, is that
# it is designed to work with any class design and layout, as long as it subclasses the base ModuleInterface class and
# implements all abstract methods: initialize_remote_assets, terminate_remote_assets, and process_received_data.
#
# For the best learning experience, it is recommended to review this code side-by-side with the implementation
# of the companion TestModule class defined in the ataraxis-micro-controller library:
# https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart
#
# See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface#quickstart for more details.
# API documentation: https://ataraxis-communication-interface-api.netlify.app/.
# Authors: Ivan Kondratyev (Inkaros), Jacob Groner.

# Imports the required assets
import numpy as np
from ataraxis_communication_interface import ModuleData, ModuleState, ModuleInterface
from ataraxis_data_structures import SharedMemoryArray


# Defines the TestModuleInterface class by subclassing the base ModuleInterface class. This class interfaces with the
# TestModule class from the companion ataraxis-micro-controller library, running on the microcontroller.
class TestModuleInterface(ModuleInterface):
    # As a minimum, the initialization method accepts two arguments to pass them to the superclass (ModuleInterface)
    # initialization method. See the ReadMe file for more details about the module type and ID codes.
    def __init__(self, module_type: np.uint8, module_id: np.uint8) -> None:
        # Defines the set of event-codes that require online processing. When the hardware module sends a message
        # containing one of these event-codes to the PC, the interface calls the process_received_data() method to
        # process the received message. In this example, the 'online' processing is used to pipe the received messages
        # to the main process via the multiprocessing Queue.
        data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kHigh, kLow and kEcho.

        # Initializes the superclass using the module-specific parameters
        super().__init__(
            module_type=module_type,
            module_id=module_id,
            data_codes=data_codes,
            error_codes=None,  # The test module does not have any expected error states.
        )

        # Any pickleable interface asset can be initialized as part of the class instantiation. In this example, the
        # shared memory array is used to transfer the data from the remote communication process to the main runtime
        # control process. Note, the shared memory array needs to be connected from the main and the remote
        # communication processes.
        self._shared_memory: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self.type_id}_shm", prototype=np.zeros(shape=3, dtype=np.uint16), exists_ok=True
        )

        # Tracks the state of the digital output pin managed by the module.
        self._previous_pin_state = False

    # The MicroControllerInterface calls this method from the remote communication process before entering the
    # communication cycle. Use this method to initialize any non-pickleable asset and set up any assets that require
    # additional configuration before the communication cycle.
    def initialize_remote_assets(self) -> None:
        # Connects to the shared memory array from the remote process.
        self._shared_memory.connect()

    # The MicroControllerInterface calls this method from the remote communication process as part of its shutdown
    # sequence. Use this method to gracefully terminate any assets that require manual cleanup.
    def terminate_remote_assets(self) -> None:
        # The shared memory array must be manually disconnected from each process that uses it to prevent runtime
        # errors.
        self._shared_memory.disconnect()

    # The MicroControllerInterface calls this method from the remote communication process to process the input
    # message. This method is only called for messages whose event-codes match the data_codes set. Use this method to
    # implement the logic for processing the incoming hardware module data as it is received by the PC.
    def process_received_data(self, message: ModuleData | ModuleState) -> None:
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

    # This utility method should be called, after starting the remote communication process to connect to the
    # shared memory array from the main process. See the documentation for the ataraxis-data-structures library for
    # information about proper initialization and termination of the SharedMemoryArray instances.
    def start_shared_memory_array(self):
        self._shared_memory.connect()
        self._shared_memory.enable_buffer_destruction()

    # This method updates the runtime parameters of the managed TestModule instance by packaging and sending
    # the input parameter values to the microcontroller. Note; the input parameter values must use the same datatypes
    # as used by the hardware module's instance running on the microcontroller.
    def set_parameters(
        self,
        on_duration: np.uint32,  # The time, in microseconds, to keep the pin HIGH when pulsing.
        off_duration: np.uint32,  # The time, in microseconds, to keep the pin LOW when pulsing.
        echo_value: np.uint16,  # The value sent to the PC as part of the echo() command's runtime.
    ) -> None:
        # This inherited method packages and sends the data to the microcontroller. Note, the order in which
        # parameter values are stored in the tuple must match the order in which they are stored inside the hardware
        # module's parameter structure.
        self.send_parameters(parameter_data=(on_duration, off_duration, echo_value))

    # Instructs the managed TestModule to emit a pulse via its output pin. The pulse uses the on_duration
    # and off_duration TestModule parameters to determine the duration of High and Low phases.
    def pulse(self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True) -> None:
        self.send_command(
            command=np.uint8(1),
            noblock=np.bool(noblock),  # Determines whether the microcontroller can execute other commands concurrently.
            repetition_delay=repetition_delay,  # Determines whether to repeat the command at a certain interval.
        )

    # Instructs the managed TestModule to respond with the current value of its echo_value parameter.
    def echo(self, repetition_delay: np.uint32 = np.uint32(0)) -> None:
        self.send_command(
            command=np.uint8(2),
            noblock=np.bool(False),  # The echo command does not have any time-delays, so is always blocking.
            repetition_delay=repetition_delay,
        )

    # This helper property returns the shared memory array object of by the interface instance, so that the shared data
    # can be accessed from the main runtime process.
    @property
    def shared_memory(self) -> SharedMemoryArray:
        return self._shared_memory
