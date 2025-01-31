# This file showcases the implementation of a custom hardware module interface class.
#
# The interface is designed to be used together with the example TestModule class, available from
# the companion ataraxis-micro-controller library: https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart
#
# Each custom hardware module instance running on the microcontroller should be matched with a custom ModuleInterface
# class instance running on the PC. These classes can be viewed as two end-points of a communication chain,
# with ataraxis-communication-interface and ataraxis-micro-controller libraries jointly abstracting all intermediate
# steps connecting the module with its interface during runtime.
#
# See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface#quickstart for more details.
# API documentation: https://ataraxis-communication-interface-api.netlify.app/.
# Authors: Ivan Kondratyev (Inkaros), Jacob Groner.

# Imports the required assets
from multiprocessing import (
    Queue as MPQueue,
    Manager,
)
from multiprocessing.managers import SyncManager

import numpy as np
from ataraxis_time import PrecisionTimer

from ataraxis_communication_interface import (
    ModuleData,
    ModuleState,
    ModuleInterface,
    ModuleParameters,
    OneOffModuleCommand,
    RepeatedModuleCommand,
)


# Defines the TestModuleInterface class by subclassing the base ModuleInterface class. This class is designed to
# interface with the TestModule class from the companion ataraxis-micro-controller library, running on the
# microcontroller.
class TestModuleInterface(ModuleInterface):
    # As a minimum, the initialization method has to take in the module type and instance ID. Each user manually
    # assigns these values in microcontroller's main .cpp file and python script, the values are not inherently
    # meaningful. The values used on the PC and microcontroller have to match.
    def __init__(self, module_type: np.uint8, module_id: np.uint8) -> None:
        # Defines the set of event-codes that the interface will interpret as runtime error events. If the module sends
        # a message with one of the event-codes from this set to the PC, the interface will automatically raise a
        # RuntimeError.
        error_codes = {np.uint8(51)}  # kOutputLocked is the only error code used by TestModule.

        # Defines the set of event-codes that the interface will interpret as data events that require additional
        # processing. When the interface receives a message containing one of these event-codes, it will call the
        # process_received_data() method on that message. The method can then process the data as necessary and send it
        # to other destinations.
        data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kHigh, kLow and kEcho.

        # Messages with event-codes above 50 that are not in either of the sets above will be saved (logged) to disk,
        # but will not be processed further during runtime.

        # The base interface class also allows direct communication between the module and other clients over the MQTT
        # protocol. This example does not demonstrate this functionality, so sets to None to disable.
        mqtt_command_topics = None

        # Initializes the parent class, using the sets defined above
        super().__init__(
            module_type=module_type,
            module_id=module_id,
            mqtt_communication=False,  # Since this example does not work with other MQTT clients, sets to False.
            mqtt_command_topics=mqtt_command_topics,
            data_codes=data_codes,
            error_codes=error_codes,
        )

        # Initializes a multiprocessing Queue. In this example, we use the multiprocessing Queue to send the data
        # to the main process from the communication process. You can initialize any assets that can be pickled as part
        # of this method runtime.
        self._mp_manager: SyncManager = Manager()
        self._output_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        # Just for demonstration purposes, here is an example of an asset that CANNOT be pickled. Therefore, we have
        # to initialize the attribute to a placeholder and have the actual initialization as part of the
        # initialize_remote_assets() method.
        self._timer: PrecisionTimer | None = None

    # This abstract method acts as the gateway for interface developers to convert and direct the data received from
    # the hardware module for further real-time processing. For this example, we transfer all received
    # data into a multiprocessing queue, so that it can be accessed from the main process.
    def process_received_data(
        self,
        message: ModuleData | ModuleState,
    ) -> None:
        # This method will only receive messages with event-codes that match the content of the 'data_codes' set.

        # This case should not be possible, as we initialize the timer as part of the initialize_remote_assets() method.
        if self._timer is None:
            raise RuntimeError("PrecisionTimer not initialized.")

        timestamp = self._timer.elapsed  # Returns the number of milliseconds elapsed since timer initialization

        # Event codes 52 and 53 are used to communicate the current state of the output pin managed by the example
        # module.
        if message.event == 52 or message.event == 53:
            # These event-codes are transmitted by State messages, so there is no additional data to parse other than
            # event codes. The codes are transformed into boolean values and are exported via the multiprocessing queue.
            message_type = "pin state"
            state = True if message.event == 52 else False
            self._output_queue.put((self.module_id, message_type, state, timestamp))

        # Since there are only three possible data_codes and two are defined above, the only remaining data code is
        # 54: the echo value.
        elif isinstance(message, ModuleData) and message.event == 54:
            # The echo value is transmitted by a Data message. Data message also includes a data_object, in addition
            # to the event code. Upon reception, the data object is automatically deserialized into the appropriate
            # object, so it can be accessed directly.
            message_type = "echo value"
            value = message.data_object
            self._output_queue.put((self.module_id, message_type, value, timestamp))

    # Since this example does not receive commands from MQTT, this method is defined with a plain None return
    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    # Use this method to initialize or configure any assets that cannot be pickled and 'transferred' to the remote
    # Process. In a way, this is a secondary __init__ method called before the main runtime logic of the remote
    # communication process is executed.
    def initialize_remote_assets(self) -> None:
        # Initializes a milliseconds-precise timer. The timer cannot be passed to a remote process and has to be created
        # by the code running inside the process.
        self._timer = PrecisionTimer("ms")

    # The methods below function as a translation interface. Specifically, they take in the input arguments and package
    # them into the appropriate message structures that can be sent to the microcontroller. If you do not require a
    # dynamic interface, all messages can also be defined statically at initialization. Then, class methods can just
    # send the appropriate predefined structure to the communication process, the same way we do with the dequeue
    # command and the MicroControllerInterface commands.

    # This method takes in values for PC-addressable module runtime parameters, packages them into the ModuleParameters
    # message, and sends them to the microcontroller. Note, the arguments to this method match the parameter names used
    # in the microcontroller TestModule class implementation.
    def set_parameters(
        self,
        on_duration: np.uint32,  # The time the pin stays HIGH during pulses, in microseconds.
        off_duration: np.uint32,  # The time the pin stays LOW during pulses, in microseconds.
        echo_value: np.uint16,  # The value to be echoed back to the PC during echo() command runtimes.
    ) -> None:
        # The _input_queue is provided by the managing MicroControllerInterface during its initialization. This guard
        # prevents this command from running unless the MicroControllerInterface is initialized.
        if self._input_queue is None:
            raise RuntimeError("MicroControllerInterface that manages ModuleInterface is not initialized.")

        # Parameters have to be arranged in the exact order expected by the receiving structure. Additionally,
        # each parameter has to use the appropriate numpy type.
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
            parameter_data=(on_duration, off_duration, echo_value),
        )

        # Directly submits the message to the communication process. The process is initialized and managed by the
        # MicroControllerInterface class that also manages the runtime of this specific interface. Once both
        # TestModuleInterface AND MicroControllerInterface are initialized, TestModuleInterface will have access to some
        # MicroControllerInterface assets via private attributes inherited from the base ModuleInterface class.
        self._input_queue.put(message)

    # Instructs the managed TestModule to emit a pulse via the manged output pin. The pulse will use the on_duration
    # and off_duration TestModule parameters to determine the duration of High and Low phases. The arguments to this
    # method specify whether the pulse is executed once or is continuously repeated with a certain microsecond delay.
    # Additionally, they determine whether the microcontroller will block while executing the pulse or allow concurrent
    # execution of other commands.
    def pulse(self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True) -> None:
        # The _input_queue is provided by the managing MicroControllerInterface during its initialization. This guard
        # prevents this command from running unless the MicroControllerInterface is initialized.
        if self._input_queue is None:
            raise RuntimeError("MicroControllerInterface that manages ModuleInterface is not initialized.")

        # Repetition delay of 0 is interpreted as a one-time command (only runs once).
        command: RepeatedModuleCommand | OneOffModuleCommand
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )
        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
                command=np.uint8(1),
                noblock=np.bool(noblock),
                cycle_delay=repetition_delay,
            )

        # Directly submits the command to the communication process.
        self._input_queue.put(command)

    # This method returns a message that instructs the TestModule to respond with the current value of its echo_value
    # parameter. Unlike the pulse() command, echo() command does not require blocking, so the method does not have the
    # noblock argument. However, the command still supports recurrent execution.
    def echo(self, repetition_delay: np.uint32 = np.uint32(0)) -> None:
        # The _input_queue is provided by the managing MicroControllerInterface during its initialization. This guard
        # prevents this command from running unless the MicroControllerInterface is initialized.
        if self._input_queue is None:
            raise RuntimeError("MicroControllerInterface that manages ModuleInterface is not initialized.")

        command: RepeatedModuleCommand | OneOffModuleCommand
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
                command=np.uint8(2),
                noblock=np.bool(False),
            )

        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
                command=np.uint8(2),
                noblock=np.bool(False),
                cycle_delay=repetition_delay,
            )

        # Directly submits the command to the communication process.
        self._input_queue.put(command)

    @property
    def output_queue(self) -> MPQueue:  # type: ignore
        # A helper property that returns the output queue object used by the class to send data from the communication
        # process back to the central process.
        return self._output_queue
