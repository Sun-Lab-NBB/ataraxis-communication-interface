# ataraxis-communication-interface

A Python library that enables interfacing with custom hardware modules running on Arduino or Teensy microcontrollers 
through Python interface clients.

![PyPI - Version](https://img.shields.io/pypi/v/ataraxis-communication-interface)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ataraxis-communication-interface)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/ataraxis-communication-interface)
![PyPI - Status](https://img.shields.io/pypi/status/ataraxis-communication-interface)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/ataraxis-communication-interface)
___

## Detailed Description

This library allows interfacing with custom hardware modules controlled by Arduino or Teensy microcontrollers via a 
local Python client or remote MQTT client. It is designed to work in tandem with the companion 
[microcontroller library](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) and allows hardware module 
developers to implement PC interfaces for their modules. To do so, the library exposes a shared API that can be 
integrated into custom interface classes by subclassing the ModuleInterface class. Additionally, the library offers 
the MicroControllerInterface class, which bridges microcontrollers managing custom hardware modules with local and 
remote clients, enabling efficient multi-directional communication and data logging.
___

## Features

- Supports Windows, Linux, and macOS.
- Provides an easy-to-implement API that integrates any user-defined hardware managed by the companion 
  [microcontroller library](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) with local and remote PC clients.
- Abstracts communication and microcontroller runtime management via the centralized microcontroller interface class.
- Contains many sanity checks performed at initialization time to minimize the potential for unexpected
  behavior and data corruption.
- Uses MQTT protocol to allow interfacing with microcontrollers over the internet or from non-Python processes.
- GPL 3 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
___

## Dependencies

- MQTT broker, if your interface needs to send or receive data over the MQTT protocol. This library was 
  tested and is intended to be used with a locally running [mosquitto MQTT broker](https://mosquitto.org/). If you have
  access to an external broker or want to use a different local broker implementation, this would also satisfy the 
  dependency.

For users, all other library dependencies are installed automatically by all supported installation methods 
(see [Installation](#installation) section).

For developers, see the [Developers](#developers) section for information on installing additional development 
dependencies.
___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.
Developers should see the [Developers](#Developers) section for more details on installing from source. The instructions
below assume you are ***not*** a developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/ataraxis-communication-interface/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file, to install the 
   wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install ataraxis-communication-interface```.
___

## Usage

### Quickstart
This section demonstrates how to use custom hardware module interfaces compatible with this library. See 
[this section](#implementing-custom-module-interfaces) for instructions on how to implement your own module interfaces. 
Note, the example below should be run together with the companion 
[microcontroller module](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart) example. 
See the [examples](./examples) folder for the .py files used in all sections of this ReadMe.
```
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
    def process_received_data(self, message: ModuleData | ModuleState) -> None:
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

    # This is the inverse of the initialize_remote_assets() that is used to clean up all custom assets initialized 
    # inside the communication process. It is called at the end of the communication runtime, before the process is 
    # terminated.
    def terminate_remote_assets(self) -> None:
        # The PrecisionTimer does not require any special cleanup. Other assets may need to have their stop() or 
        # disconnect() method called from within this method.
        pass

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
```

### User-Defined Variables
This library is designed to support many different use patterns. To do so, it intentionally avoids hardcoding
certain metadata variables that allow the PC interface to identify the managed microcontroller and specific hardware 
module instances running on that controller. As a user, you **have to** manually define these values **both** for the 
microcontroller and the PC. The PC and the Microcontroller have to have the **same** interpretation for these values 
to work as intended.

- `Controller ID`. This is a unique byte-code value between 1 and 255 that identifies the microcontroller during 
   communication. This ID code is used when logging the data received from the microcontroller, so it has to be unique 
   for all microcontrollers **and other** Ataraxis classes used at the same time that log data. For example, 
   [Video System](https://github.com/Sun-Lab-NBB/ataraxis-video-system) classes also use the byte-code ID system to 
   identify themselves during logging and **will clash** with microcontroller IDs if you are using 
   both at the same time. This code is provided as an argument when initializing the MicroControllerInterface instance.

- `Module Type` for each module. This is a byte-code between 1 and 255 that identifies the family of each module. For 
   example, all solenoid valves may use the type-code '1,' while all voltage sensors may use type-code '2.' The type 
   codes do not have an inherent meaning, they are assigned independently for each use case. Therefore, the
   same collection of custom module classes may have vastly different type-codes for two different projects. This 
   design pattern is intentional and allows developers to implement modules without worrying about clashing with 
   already existing modules. This code is provided as an argument when subclassing the ModuleInterface class.

- `Module ID` for each module. This byte-code between 1 and 255 has to be unique within the module type (family) and
   is used to identify specific module instances. For example, this code will be used to identify different voltage 
   sensors if more than one sensor is used by the same microcontroller at the same time. This code is provided as an 
   argument when subclassing the ModuleInterface class.

### Data Logging
Like some other Ataraxis libraries, this library relies on the 
[DataLogger](https://github.com/Sun-Lab-NBB/ataraxis-data-structures#datalogger) class to save all incoming and outgoing
messages in their byte-serialized forms to disk as `.npy` files. It is **highly** advised to study the documentation for
the class before using this library, especially if you want to parse the logged data manually instead of using the 
method exposed by each ModuleInterface class.

The DataLogger may be shared by multiple Ataraxis classes that generate log entries, such as 
[VideoSystem](https://github.com/Sun-Lab-NBB/ataraxis-video-system) classes. To support using 
the same logger class for multiple sources, each source (class) active at the same time has to use a unique byte-ID
(system id). These id-codes are used to identify the source class in log files and during further processing.

***Critically:*** Each MicroControllerInterface accepts a DataLogger instance at instantiation. Generally, it is advised
to use the same DataLogger instance for all MicroControllerInterface classes active at the same time, although this is
not required.

#### Log entries format
Each message is logged as a one-dimensional numpy uint8 array (.npy file). Inside the array, the data is organized in 
the following order:
1. The uint8 id of the data source. For this library, the source ID is the ID code of the microcontroller managed by the
   MicroControllerInterface that submits the data to be logged. The ID occupies the first byte of each logged array.
2. The uint64 timestamp that specifies the number of microseconds relative to the **onset** timestamp (see below). The 
   timestamp occupies **8** bytes following the ID byte.
3. The serialized message payload sent to the microcontroller or received from the microcontroller. The payload can 
   be deserialzied using the appropriate message structure. The payload occupies all remaining bytes, following the 
   source ID and the timestamp.

#### Onset timestamp:
Each MicroControllerInterface that logs its data generates an `onset` timestamp as part of its `start()` method runtime.
This log entry uses a modified data order and stores the current UTC time, accurate to microseconds. All further log 
entries for the same source use the timestamp section of their payloads to communicate the number of microseconds 
elapsed since the onset timestamp. The onset log entries follow the following order:
1. The uint8 id of the data source.
2. The uint64 value **0** that occupies 8 bytes following the source id. This is the only time when the timestamp value 
   of a log entry can be set to 0.
3. The uint64 value that stores the number of microseconds elapsed since the UTC epoch. This value specifies the 
   current time when the onset timestamp was generated.

#### Starting and stopping logging
Until the DataLogger is started through its `start()` method, the log entries will be buffered in the multiprocessing 
queue, which uses the host-computer’s RAM. To avoid running out of buffer space, **make sure** the DataLogger's 
`start()` method is called before calling the `start()` method of any MicroControllerInterface class. Once all sources
using the same DataLogger have finished their runtime, call the `stop()` method to end log saving and then call the
`compress_logs()` method to compress all individual `.npy` entries into an `.npz` archive. Compressing the logs is 
required to later parse logged module data for further analysis (see [quickstart](#quickstart)).

#### Reading custom module data from logs
The base ModuleInterface class exposes the `extract_logged_data()` method that allows parsing received ModuleState and 
ModuleData messages from compressed '.npz' archives. Currently, the method only works with messages that use 'event' 
byte-codes greater than 51 and only with messages sent by custom hardware module classes (children of base 
ModuleInterface class). The only exception to this rule is Command Completion events (event code 2), which are also
parsed for each hardware module.

***Note:*** to parse logged data, the ModuleInterface has to be used to initialize a MicroControllerInterface. The 
MicroControllerInterface overwrites certain attributes inside each managed ModuleInterface during its initialization, 
which is required for the log parser to find the target log file. Overall, it is advised to parse logged data 
immediately after finishing the communication runtime, as the class would be configured correctly for the parsing to 
work as intended.

### Custom Module Interfaces
For this library an interface is a class that contains the logic for sending the command and parameter data to the 
hardware module and receiving and processing the data sent by the module to the PC. The microcontroller and PC libraries
ensure that the data is efficiently moved between the module and the interface, but each custom hardware module 
developer is responsible for handling that data.

### Implementing Custom Module Interfaces
All module interfaces intended to be accessible through this library have to follow the implementation guidelines
described in the [example module interface implementation file](./examples/example_interface.py). Specifically, 
**all custom module interfaces have to subclass the ModuleInterface class from this library and implement all abstract
methods**. Additionally, all commands and parameter messages generated by the interface **have to use one of the valid
[message structures](#module-messages) exposed by this library**.

#### Abstract Methods
These methods act as a gateway that custom interface developers can use to execute custom logic to process incoming or
outgoing data. The MicroControllerInterface class that manages the communication will call these methods for incoming or
outgoing data according to the configuration of each managed ModuleInterface (see below for details). Currently, there 
are four abstract methods defined by the base ModuleInterface class: initialize_remote_assets(), 
terminate_remote_assets(), process_received_data() and parse_mqtt_command()

#### initialize_remote_assets
This method is called by the MicroControllerInterface once for each ModuleInterface at the beginning of the 
communication cycle. The method should be used to initialize or configure custom assets (queue, shared memory buffers, 
timer, etc.) that cannot be pickled and transferred to the communication Process. Any assets that can be pickled can be
initialized during the interface __init__ method runtime. All assets should be stored in class attributes, so that they
can be accessed from other abstract methods.
```
def initialize_remote_assets(self) -> None:
    # Initializes a milliseconds-precise timer. The timer cannot be passed to a remote process and has to be created
    # by the code running inside the process.
    self._timer = PrecisionTimer("ms")
```

#### terminate_remote_assets
This method is the inverse of the initialize_remote_assets() method. It is called by the MicroControllerInterface for
each ModuleInterface at the end of the communication cycle. This method should be used to clean up (terminate) any 
assets initialized at the beginning of the communication runtime to ensure all resources are released before the process
is terminated. The example below is not from the TestModuleInterface, but showcases the proper termination of the 
MQTTCommunication class used by some interfaces in the Sun lab.
```
def terminate_remote_assets(self) -> None:
    # This was called inside the initialize_remote_assets()
    # self._communication = MQTTCommunication()
    # self._communication.connect()
    
    # This is called in the terminate_remote_assets()
    self._communication.disconnect()
```

#### parse_mqtt_command
This method translates commands sent by other MQTT clients into ModuleCommand messages that are transmitted to the 
microcontroller for execution. MicroControllerInterface uses its MQTTCommunication class to monitor the topics listed
by each managed ModuleInterface. When one of the monitored topics receives a message, MicroControllerInterface calls 
this method for all ModuleInterfaces that listed that topic as their 'command topic.'

The purpose of the method is to parse the topic and/or payload of a received MQTT message and, based on this data, to
construct and return the command message to send to the Module. While the example TestModuleInterface does not 
demonstrate this functionality, consider this example implementation used to control water valves in the Sun Lab:
```
def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> OneOffModuleCommand | None:
    if topic == 'gimbl/reward':
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),  # Blocks to ensure reward delivery precision.
        )
```

Currently, the method is designed to only process commands and work with all valid module commands.

#### process_received_data
This method allows processing incoming ModuleState and ModuleData messages as they are received by the PC. 
MicroControllerInterface calls this method for any State or Data message received from the hardware module, if the 
event code from that messages matches one of the codes in the data_codes attribute of the ModuleInterface. Therefore, 
this method will only be called on the messages specified by teh ModuleInterface developer.

**Note:** The MicroControllerInterface class ***automatically*** saves (logs) each received and sent message to the PC
as a stream of bytes. Therefore, this method should ***not*** be used to save the data for post-runtime analysis. 
Instead, this method should be used to process the data in real time. For example, use this method to communicate the 
physical location of a real life object to the Unity game engine simulating the virtual reality (via MQTT). Or use this 
method to display a real-time graph for the microcontroller-recorded event, such as voltage detected by the voltage 
sensor.

Since all ModuleInterfaces used by the same MicroControllerInterface share the communication process, 
process_received_data should not use complex logic or processing. Treat this method as you would a hardware interrupt 
function: its main goal is to move the data to a different context, where it can be processed, as quickly as possible 
and allow the communication loop to run for other modules.

This example demonstrates the implementation of the processing method to send the data back to the main process. All 
assets other than the message are stored in class attributes. The timer is initialized via the 
initialize_remote_assets() method:
```
def process_received_data(
    self,
    message: ModuleData | ModuleState,
) -> None:
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
```

#### Module Messages
In addition to abstract methods, each interface may need to implement a number of messages that can be sent to the 
microcontroller. Unlike abstract methods, implementing custom command and parameter messages is optional: not all 
modules may need to receive data from the PC to function.

To communicate with the module, the interface has to define one of the valid Module-targeted messages:
OneOffModuleCommand, RepeatedModuleCommand, DequeueModuleCommand, or ModuleParameters. Each of these messages is a 
dataclass that as a minimum contains 3 fields: the type of the target module, the instance ID of the target module, and
a return_code. Since return_code is currently only used for debugging, **make sure the return_code is always set to 
0**. Check the [API documentation](https://ataraxis-communication-interface-api.netlify.app/) for details about
supported message structures.

It is not relevant how each interface defines its command and parameter messages. For example, in the 
TestModuleInterface, we define methods that translate user-input into command messages. This enables users to 
flexibly define commands to be sent to the module.
```
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
```

However, you can also statically hard-code a set of fixed commands and expose them as interface class properties or 
follow any other implementation that makes sense for your use case.

#### Submitting messages to the microcontroller

Since version 3.0.0, there are two ways for sending command or parameter messages to the microcontroller. The first way
is to submit the message instance to the `send_message()` method of the MicroControllerInterface instance managing the 
target microcontroller:
```
# This demonstrates creating and seinding a dequeue command to the hardware module with type 1 and id 3.
mc_interface.send_message(DequeueModuleCommand(np.uint8(1), np.uint8(3), np.uint8(0)))
```

The second way, introduced in version 3.0.0 is using the _input_queue attribute inherited from the base ModuleInterface
class. **Note**, this attribute is provided by the managing MicroControllerInterface class, so it is initially set to 
None. The ModuleInterface has to be submitted to the initialization method of the MicroControllerInterface class to be 
able to use this attribute for message submission:
```
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
```

___

## API Documentation

See the [API documentation](https://ataraxis-communication-interface-api.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library.

### Installing the library

The easiest way to ensure you have most recent development dependencies and library source files is to install the 
python environment for your OS (see below). All environments used during development are exported as .yml files and as 
spec.txt files to the [envs](envs) folder. The environment snapshots were taken on each of the three explicitly 
supported OS families: Windows 11, OSx Darwin, and GNU Linux.

**Note!** Since the OSx environment was built for the Darwin platform (Apple Silicon), it may not work on Intel-based 
Apple devices.

1. If you do not already have it installed, install [tox](https://tox.wiki/en/latest/user_guide.html) into the active
   python environment. The rest of this installation guide relies on the interaction of local tox installation with the
   configuration files included in with this library.
2. Download this repository to your local machine using your preferred method, such as git-cloning. If necessary, unpack
   and move the project directory to the appropriate location on your system.
3. ```cd``` to the root directory of the project using your command line interface of choice. Make sure it contains
   the `tox.ini` and `pyproject.toml` files.
4. Run ```tox -e import``` to automatically import the os-specific development environment included with the source 
   distribution. Alternatively, you can use ```tox -e create``` to create the environment from scratch and automatically
   install the necessary dependencies using pyproject.toml file. 
5. If either step 4 command fails, use ```tox -e provision``` to fix a partially installed environment.

**Hint:** while only the platforms mentioned above were explicitly evaluated, this project will likely work on any 
common OS, but may require additional configurations steps.

### Additional Dependencies

In addition to installing the development environment, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. These 
   versions will be installed in-addition to the main Python version installed in the development environment.
   The easiest way to get tox to work as intended is to have separate python distributions, but using 
   [pyenv](https://github.com/pyenv/pyenv) is a good alternative. This is needed for the 'test' task to work as 
   intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check [tox.ini file](tox.ini) for details about 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All commits to this project have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime duration for this task, use ```tox --parallel```.

For more information, check the 'Usage' section of the 
[ataraxis-automation project](https://github.com/Sun-Lab-NBB/ataraxis-automation#Usage) documentation.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself are prone to various failures. In 
most cases, this is related to their caching behavior. Despite a considerable effort to disable caching behavior known 
to be problematic, in some cases it cannot or should not be eliminated. If you run into an unintelligible error with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a cli command is very likely to fix the issue.
___

## Versioning

We use [semantic versioning](https://semver.org/) for this project. For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-communication-interface/tags).

---

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Jacob Groner ([Jgroner11](https://github.com/Jgroner11))

___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of all other projects used in our development automation pipelines and source code 
  [see pyproject.toml](pyproject.toml).

---
