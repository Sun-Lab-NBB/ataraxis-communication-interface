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
from multiprocessing import Queue as MPQueue

import numpy as np

from ataraxis_communication_interface import (
    ModuleData,
    ModuleState,
    ModuleInterface,
    ModuleParameters,
    MQTTCommunication,
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
        # to other processes via a local queue (as done here) or via the MQTT protocol. MQTT protocol interaction is
        # not demonstrated here, but is used by other Sun lab repositories.
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

    # This abstract method acts as the gateway for interface developers to convert and direct the data received from
    # the hardware module for further real-time processing. For this example, we transfer all received
    # data into a multiprocessing queue, so that it can be accessed from the main process.
    def process_received_data(
        self,
        message: ModuleData | ModuleState,
        mqtt_communication: MQTTCommunication,  # Not used by this example, but still has to be defined
        mp_queue: MPQueue,  # type: ignore
    ) -> None:
        # This method will only receive messages with event-codes that match the content of the 'data_codes' set.

        # Event codes 52 and 53 are used to communicate the current state of the output pin managed by the example
        # module.
        if message.event == 52 or message.event == 53:
            # These event-codes are transmitted by State messages, so there is no additional data to parse other than
            # event codes. The codes are transformed into boolean values and are exported via the multiprocessing queue.
            message_type = "pin state"
            state = True if message.event == 52 else False
            mp_queue.put((self.module_id, message_type, state))

        # Since there are only three possible data_codes and two are defined above, the only remaining data code is
        # 54: the echo value.
        elif isinstance(message, ModuleData) and message.event == 54:
            # The echo value is transmitted by a Data message. Data message also includes a data_object, in addition
            # to the event code. Upon reception, the data object is automatically deserialized into the appropriate
            # object, so it can be accessed directly.
            message_type = "echo value"
            value = message.data_object
            mp_queue.put((self.module_id, message_type, value))

    # Since this example does not receive commands from MQTT, this method is defined with a plain None return
    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    # The methods below function as a translation interface. Specifically, they take in the input arguments and package
    # them into the appropriate message structures that can be sent to the microcontroller. If you do not require a
    # dynamic interface, all messages can also be defined statically and exposed as class properties, as we do with
    # MicroControllerInterface class.

    # Note, the commands do not automatically end the message to the microcontroller. Instead, the message returned by
    # each of the methods below has to be passed to the send_data() method of the MicroControllerInterface class.

    # This method takes in values for PC-addressable module runtime parameters and returns the ModuleParameters message
    # that can be sent to the microcontroller to apply the input parameter values. Note, the arguments to this method
    # match the parameter names used in the microcontroller TestModule class implementation.
    def set_parameters(
        self,
        on_duration: np.uint32,  # The time the pin stays HIGH during pulses, in microseconds.
        off_duration: np.uint32,  # The time the pin stays LOW during pulses, in microseconds.
        echo_value: np.uint16,  # The value to be echoed back to the PC during echo() command runtimes.
    ) -> ModuleParameters:
        # Parameters have to be arranged in the exact order expected by the receiving structure. Additionally,
        # each parameter has to use the appropriate numpy type.
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
            parameter_data=(on_duration, off_duration, echo_value),
        )

    # This method returns a message that instructs the TestModule to emit a pulse via the manged output pin. The pulse
    # will use the on_duration and off_duration TestModule parameters to determine the duration of High and Low phases.
    # The arguments to this method specify whether the pulse is executed once or is continuously repeated with a
    # certain microsecond delay. Additionally, they determine whether the microcontroller will block while executing
    # the pulse or allow concurrent execution of other commands.
    def pulse(
        self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        # Repetition delay of 0 is interpreted as a one-time command (only runs once).
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
            command=np.uint8(1),
            noblock=np.bool(noblock),
            cycle_delay=repetition_delay,
        )

    # This method returns a message that instructs the TestModule to respond with the current value of its echo_value
    # parameter. Unlike the pulse() command, echo() command does not require blocking, so the method does not have the
    # noblock argument. However, the command still supports recurrent execution.
    def echo(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
                command=np.uint8(2),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Keep this set to 0, the functionality is only for debugging purposes.
            command=np.uint8(2),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )
