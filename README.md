# ataraxis-communication-interface

A Python library that provides the centralized interface for exchanging commands and data between Arduino and Teensy
microcontrollers and host-computers.

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

This library allows interfacing with custom hardware modules controlled by Arduino or Teensy microcontrollers
running the companion [microcontroller library](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller). To do so, 
the library defines a shared API that can be integrated into user-defined interfaces by subclassing the (base) 
ModuleInterface class. It also provides the MicroControllerInterface class that manages the microcontroller-PC 
communication and the MQTTCommunication class that allows exchanging data between local and remote clients over the 
MQTT (TCP) protocol.

___

## Features

- Supports Windows, Linux, and macOS.
- Provides the framework for writing and deploying custom interfaces for the hardware module instances managed 
  by the companion [microcontroller library](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller).
- Abstracts communication and microcontroller runtime management via the centralized microcontroller interface class.
- Leverages MQTT protocol to support exchanging data between multiple local and remote clients.
- Uses JIT compilation and LRU caching to optimize the runtime efficiency of all library assets.
- Contains many sanity checks performed at initialization time to minimize the potential for unexpected
  behavior and data corruption.
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

- **MQTT broker**, if the library is intended to be used for sending and receiving data over the MQTT protocol. The 
  library was tested with a locally running [mosquitto MQTT broker](https://mosquitto.org/) version **2.0.22**.

For users, all other library dependencies are installed automatically by all supported installation methods 
(see [Installation](#installation) section).

***Note!*** Developers should see the [Developers](#developers) section for information on installing additional 
development dependencies.

___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the 
   [stable releases](https://github.com/Sun-Lab-NBB/ataraxis-communication-interface/releases).
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
   binaries, use ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file.

### pip
Use the following command to install the library using pip: ```pip install ataraxis-communication-interface```.

___

## Usage

### Quickstart
This section demonstrates how to use custom hardware module interfaces compatible with this library. See 
[this section](#implementing-custom-module-interfaces) for instructions on how to implement module interface classes. 
Note, the example below should be run together with the companion 
[microcontroller module](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart) example. 
See the [example_runtime.py](./examples/example_runtime.py) for the .py implementation of this example.
```
# Imports the necessary assets, including the TestModuleInterface class
from pathlib import Path

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from example_interface import TestModuleInterface
from ataraxis_data_structures import DataLogger, assemble_log_archives
from ataraxis_base_utilities import console, LogLevel
import tempfile

from ataraxis_communication_interface import MicroControllerInterface, extract_logged_hardware_module_data

# Since MicroControllerInterface uses multiple processes, it has to be called with the '__main__' guard
if __name__ == "__main__":
    # Enables the console module to communicate the example's runtime progress via the terminal.
    console.enable()

    # Specifies the directory where to save all incoming and outgoing messages processed by the MicroControllerInterface
    # instance for each hardware module.
    tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes
    output_directory = Path(tempdir.name)

    # Instantiates the DataLogger, which is used to save all incoming and outgoing MicroControllerInterface messages
    # to disk. See https://github.com/Sun-Lab-NBB/ataraxis-data-structures for more details on DataLogger class.
    data_logger = DataLogger(output_directory=output_directory, instance_name="AMC")
    data_logger.start()  # The DataLogger has to be started before it can save any log entries.

    # Defines two interface instances, one for each TestModule used at the same time. Note that each instance uses
    # different module_id codes, but the same type (family) id code.
    interface_1 = TestModuleInterface(module_type=np.uint8(1), module_id=np.uint8(1))
    interface_2 = TestModuleInterface(module_type=np.uint8(1), module_id=np.uint8(2))
    interfaces = (interface_1, interface_2)

    # Instantiates the MicroControllerInterface. This class functions similar to the Kernel class from the
    # ataraxis-micro-controller library and abstracts most inner-workings of the library. Note; example expects a
    # Teensy 4.1 microcontroller, and the parameters defined below may not be optimal for all supported
    # microcontrollers!
    mc_interface = MicroControllerInterface(
        controller_id=np.uint8(222),
        buffer_size=8192,
        port="/dev/ttyACM1",
        data_logger=data_logger,
        module_interfaces=interfaces,
        baudrate=115200,
        keepalive_interval=5000,
    )
    console.echo("Initializing the communication process...")

    # Starts the serial communication with the microcontroller by initializing a separate process that handles the
    # communication. This method may take up to 15 seconds to execute, as it verifies that the microcontroller is
    # configured correctly, given the MicroControllerInterface configuration.
    mc_interface.start()

    console.echo("Communication process: Initialized.", level=LogLevel.SUCCESS)
    console.echo("Updating hardware module runtime parameters...")

    # Due to the current SharedMemoryArray implementation, the SHM instances require additional setup after the
    # communication process is started.
    interface_1.start_shared_memory_array()
    interface_2.start_shared_memory_array()

    # Generates and sends new runtime parameters to both hardware module instances running on the microcontroller.
    # On and Off durations are in microseconds.
    interface_1.set_parameters(
        on_duration=np.uint32(1000000), off_duration=np.uint32(1000000), echo_value=np.uint16(121)
    )
    interface_2.set_parameters(
        on_duration=np.uint32(5000000), off_duration=np.uint32(5000000), echo_value=np.uint16(333)
    )

    console.echo("Hardware module runtime parameters: Updated.", level=LogLevel.SUCCESS)

    console.echo("Sending the 'echo' command to the TestModule 1...")

    # Requests instance 1 to return its echo value. By default, the echo command only runs once.
    interface_1.echo()

    # Waits until the microcontroller responds to the echo command. The interface is configured to update shared
    # memory array index 2 with the received echo value when it receives the response from the microcontroller.
    while interface_1.shared_memory[2] == 0:
        continue

    # Retrieves and prints the microcontroller's response. The returned value should match the parameter set above: 121.
    console.echo(message=f"TestModule 1 echo value: {interface_1.shared_memory[2]}.", level=LogLevel.SUCCESS)

    # Demonstrates the use of non-blocking recurrent commands.
    console.echo("Executing the example non-blocking runtime, standby for ~5 seconds...")

    # Instructs the first TestModule instance to start pulsing the managed pin (Pin 5 by default). With the parameters
    # sent earlier, it keeps the pin ON for 1 second and keeps it off for ~ 2 seconds (1 from off_duration,
    # 1 from waiting before repeating the command). The microcontroller repeats this command at regular intervals
    # until it is given a new command or receives a 'dequeue' command (see below).
    interface_1.pulse(repetition_delay=np.uint32(1000000), noblock=True)

    # Instructs the second TestModule instance to start sending its echo value to the PC every 500 milliseconds.
    interface_2.echo(repetition_delay=np.uint32(500000))

    # Delays for 5 seconds, accumulating echo values from TestModule 2 and pin On / Off notifications from TestModule
    # 1. Uses the PrecisionTimer instance to delay the main process for 5 seconds.
    delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    delay_timer.delay(delay=5, block=False)

    # Cancels both recurrent commands by issuing a dequeue command. Note, the dequeue command does not interrupt already
    # running commands, it only prevents further command repetitions.
    interface_1.reset_command_queue()
    interface_2.reset_command_queue()

    # The result seen here depends on the communication speed between the PC and the microcontroller and the precision
    # of the microcontroller's clock. For Teensy 4.1, which was used to write this example, the pin is expected to
    # pulse ~2 times and the echo value is expected to be transmitted ~10 times during the test period.
    console.echo(message="Non-blocking runtime: Complete.", level=LogLevel.SUCCESS)
    console.echo(f"TestModule 1 Pin pulses: {interface_1.shared_memory[0]}")
    console.echo(f"TestModule 2 Echo values: {interface_2.shared_memory[1]}")

    # Resets the pulse and echo counters before executing the demonstration below.
    interface_1.shared_memory[0] = 0
    interface_2.shared_memory[1] = 0

    # Repeats the example above, but now uses blocking commands instead of non-blocking.
    console.echo("Executing the example blocking runtime, standby for ~5 seconds...")
    interface_1.pulse(repetition_delay=np.uint32(1000000), noblock=False)
    interface_2.echo(repetition_delay=np.uint32(500000))
    delay_timer.delay(delay=5, block=False)  # Reuses the same delay timer
    interface_1.reset_command_queue()
    interface_2.reset_command_queue()

    # This time, since the pin pulsing performed by module 1 interferes with the echo command performed by module 2,
    # both pulse and echo counters are expected to be ~5.
    console.echo(message="Blocking runtime: Complete.", level=LogLevel.SUCCESS)
    console.echo(f"TestModule 1 Pin pulses: {interface_1.shared_memory[0]}")
    console.echo(f"TestModule 2 Echo values: {interface_2.shared_memory[1]}")

    # Stops the communication process and releases all resources used during runtime.
    mc_interface.stop()
    console.echo(f"Communication process: Stopped.", level=LogLevel.SUCCESS)

    # Stops the DataLogger and assembles all logged data into a single .npz archive file. This step is required to be
    # able to extract the logged message data for further analysis.
    data_logger.stop()
    console.echo(f"Assembling the message log archive...")
    assemble_log_archives(log_directory=data_logger.output_directory, remove_sources=True, verbose=True)

    # To process the data logged during runtime, it must be extracted from the archive created above. This can be
    # done with the help of the `extract_logged_hardware_module_data` function:
    console.echo(f"Extracting the logged message data...")
    log_data = extract_logged_hardware_module_data(
        log_path=data_logger.output_directory.joinpath(f"222_log.npz"),
        module_type_id=(
            (int(interface_1.module_type), int(interface_1.module_id)),
            (int(interface_2.module_type), int(interface_2.module_id)),
        ),
    )
    # Uses pulse off and echo event codes to determine the total number of TestModule 1 pulses and TestModule 2 echo
    # values encountered during runtime according to the processed log data.
    module_1_pulses = len(log_data[0].event_data[np.uint8(53)])
    module_2_echo_values = len(log_data[1].event_data[np.uint8(54)])
    console.echo(
        message=(
            f"According to the extracted data, during runtime the TestModule 1 emitted a total of {module_1_pulses} "
            f"pulses and the TestModule 2 sent {module_2_echo_values} echo values."
        ),
        level=LogLevel.SUCCESS,
    )
```

### User-Defined Variables
This library is designed to flexibly support many different use patterns. To do so, it intentionally avoids hardcoding
certain metadata variables that allow the PC interface to individuate and address the managed microcontroller and 
specific hardware module instances. **Each end user has to manually define these values both for the microcontroller 
and the PC.** 

Two of these variables, the `module_type` and the `module_id` are used by the (base) **ModuleInterface** class. The 
remaining `controlled_id` variable is used by the **MicroControllerInterface** class. See the 
[companion library’s](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#user-defined-variables) ReadMe for more 
details about each user-defined metadata variable. Typically, these variables are set in the microcontroller code and 
the PC code is adjusted to match the microcontroller code’s state.

### Keepalive
A major runtime safety feature of this library is the support for keepalive messaging. To work as intended, **both the 
PC (MicroControllerInterface instance) and the microcontroller (Kernel instance) must be configured to use the same 
keepalive interval.** 

When enabled, the MicroControllerInterface instance sends a 'keepalive' command at regular intervals, specified by the 
`keepalive_interval` initialization argument. If the microcontroller does not receive the command for 
**two consecutive interval windows**, it aborts the runtime by resetting the microcontroller’s hardware and software to 
the default state and sends an error message to the PC. If the PC does not receive the microcontroller’s acknowledgement
that it has received the keepalive command within **one interval windows from sending the previous command**, it aborts
the communication runtime with an error.

The keepalive functionality is **disabled** (set to 0) by default, but it is recommended to enable it for most use 
cases. See the [API documentation for the MicroControllerInterface class](#api-documentation) for more details on 
configuring the keepalive messaging.

***Note!*** The appropriate keepalive interval depends on the communication speed and the CPU frequency of the 
microcontroller. For a fast microcontroller (teensy4.1) that uses the USB communication interface, an appropriate 
keepalive interval is typically measured in milliseconds (100 to 500). For a slower microcontroller (arduino mega) with 
a UART communication interface using the baudrate of 115200, the appropriate keepalive interval is typically measured 
in seconds (2 to 5).

### Communication
During runtime, all communication with the microcontroller is routed via the MicroControllerInterface instance that 
implements the centralized communication and control interface for each microcontroller. To optimize runtime 
performance, the communication is managed by a daemonic process running in a separate CPU thread (core).

When the data is sent to the microcontroller, it is first transferred to the communication process, which then transmits
it to the microcontroller. When the data is received from the microcontroller, it is mostly handled by the communication
process, unless the end user implements the logic for routing it to other runtime processes.

### Data Logging
This library relies on the [DataLogger](https://github.com/Sun-Lab-NBB/ataraxis-data-structures#datalogger) class to 
save all incoming and outgoing messages to disk during PC-microcontroller communication. Each message sent or received 
by the PC is serialized and saved as an uncompressed **.npy** file.

The same DataLogger instance as used by the MicroControllerInterface instances may be shared by multiple other Ataraxis 
assets that generate log entries, such as [VideoSystem](https://github.com/Sun-Lab-NBB/ataraxis-video-system) classes. 
To support using the same logger instance for multiple concurrently active sources, 
**each source has to use a unique identifier value (controller id) when sending data to the logger instance**.

**Note!** Currently, only the MicroControllerInterface supports logging the data to disk.

#### Log Format
Each message is logged as a one-dimensional numpy uint8 array, saved as an .npy file. Inside the array, the data is 
organized in the following order:
1. The uint8 id of the data source (microcontroller). The ID occupies the first byte of each log entry.
2. The uint64 timestamp that specifies the number of microseconds elapsed since the acquisition of the **onset** 
   timestamp (see below). The timestamp occupies **8** bytes following the ID byte. This value communicates when each 
   message was sent or received by the PC.
3. The serialized message payload sent to the microcontroller or received from the microcontroller. The payload can 
   be deserialized using the appropriate message structure. The payload occupies all remaining bytes, following the 
   source ID and the timestamp.

#### Onset timestamp:
Each MicroControllerInterface generates an `onset` timestamp as part of its `start()` method runtime. This log entry 
uses a modified data order and stores the current UTC time, accurate to microseconds, as the total number of 
microseconds elapsed since the UTC epoch onset. All further log entries for the same source use the timestamp section 
of their payloads to communicate the number of microseconds elapsed since the onset timestamp acquisition. 

The onset log entry uses the following data organization order:
1. The uint8 id of the data source (video system instance).
2. The uint64 value **0** that occupies 8 bytes following the source id. A 'timestamp' value of 0 universally indicates 
   that the log entry stores the onset timestamp.
3. The uint64 value that stores the number of microseconds elapsed since the UTC epoch onset. This value specifies the 
   current time when the onset timestamp was generated.

#### Working with MicroControllerInterface Logs

See the [quickstart](#quickstart) example above for a demonstration on how to assemble and parse the frame acquisition
log archives generated by the VideoSystem instance at runtime. 

**Note!** Currently, the log parsing function only works with messages that use event-codes greater than 50 and only 
with messages sent by custom hardware module instances. The only exception to this rule is the **Command Completion** 
events (event code 2), which are also parsed for each hardware module.

The logged data is packaged into a hierarchical structure the segments messages by each custom hardware module instance,
packing each in the **ExtractedModuleData** dataclass instances. Each instance further segments the data into 'events'
by storing extracted data in a dictionary that uses event-codes as keys and tuples of **ExtractedMessageData** instances
as values. Each **ExtractedMessageData** stores the data of a single message received from the respective hardware 
module during runtime.

### Custom Module Interfaces
For this library, an interface is a class that contains the logic for sending the command and parameter data to the 
hardware module and receiving and processing the data sent by the module to the PC. The microcontroller and PC libraries
ensure that the data is efficiently moved between the module and the interface and saved (logged) to disk. The rest of 
the module-interface interaction is up to the end user (module / interface developer).

### Implementing Custom Module Interfaces
All module interfaces intended to be accessible through this library have to follow the implementation guidelines
described in the [example module interface implementation file](./examples/example_interface.py). Specifically, 
**all custom module interfaces have to subclass the ModuleInterface class from this library and implement all abstract
methods**.

#### Abstract Methods
These methods provide the inherited API used by the centralized microcontroller interface to connect hardware module 
interfaces to their hardware modules managed by the companion microcontroller. Specifically, the 
MicroControllerInterface calls these methods as part of the remote communication process’s runtime cycle to work with 
the data sent by the custom hardware module.

#### initialize_remote_assets
This method is called by the MicroControllerInterface once for each ModuleInterface at the beginning of the 
communication cycle. The method should be used to initialize or configure custom assets (queues, shared memory buffers, 
timers, etc.) that need to be processed from the (remote) communication process.
```
def initialize_remote_assets(self) -> None:
    # Connects to the shared memory array from the remote process.
    self._shared_memory.connect()
```

#### terminate_remote_assets
This method is the inverse of the initialize_remote_assets() method. It is called by the MicroControllerInterface for
each ModuleInterface at the end of the communication cycle. This method should be used to clean up (terminate) any 
assets initialized at the beginning of the communication runtime to ensure all resources are released before the process
is terminated.
```
def terminate_remote_assets(self) -> None:
    # The shared memory array must be manually disconnected from each process that uses it to prevent runtime
    # errors.
    self._shared_memory.disconnect()
```

#### process_received_data
This method allows processing incoming module messages as they are received by the PC. The MicroControllerInterface 
instance calls this method for any ModuleState or ModuleData message received from the hardware module, if the 
event code of the message matches one of the codes in the data_codes attribute of the module’s interface instance.

**Note!** The MicroControllerInterface class ***automatically*** saves (logs) each received and sent message to disk. 
Therefore, this method should ***not*** be used to save the data for post-runtime processing. Instead, this method 
should be used to process the data in real time or route it to other processes / machines for real time processing.

Since all ModuleInterfaces used by the same MicroControllerInterface share the communication process, 
**process_received_data() should not use complex logic or processing**. Treat this method as a hardware interrupt 
function: its main goal is to handle the incoming data as quickly as possible and allow the communication loop to run 
for other modules.

This example demonstrates the implementation of the processing method to send the data back to the main process:
```
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
```

#### Sending Data to the Microcontroller
In addition to abstract methods, each interface may need to send data to the microcontroller. Broadly, the outgoing 
messages are divided into two categories: **commands** and **parameter updates**. Command messages instruct the module 
to perform a specified action. Parameter updates are used to overwrite the module’s runtime parameters to broadly adjust
how the module behaves while executing commands.

Each interface should use the `send_parameters()` method inherited from the (base) ModuleInterface class to send 
parameter update messages to the managed module and the `send_command()` method to send command messages to the managed 
module. These utility method abstracts the necessary steps for packaging and transmitting the input data to the module.

**Note!** These methods use LRU caching and JIT compilation to optimize their runtime speed and minimize the delay 
between submitting the message for transmission and it being sent to the microcontroller. Therefore, most command and 
parameter update functions / methods should be simple wrappers around these inherited methods. See the API documentation
for the ModuleInterface class for the details about these methods inherited by each child interface class.

### Discovering Connected Microcontrollers
To help determining which microcontrollers are available for communication and at which ports, this library exposes the 
`axci-id` CLI command. This command is available from any environment that has the library installed and internally 
calls the `print_microcontroller_ids()` standalone function. This command evaluates each available serial port for 
whether it is connected to a valid Ataraxis microcontroller and, if so, queries the unique identifier of that 
microcontroller.

___

## API Documentation

See the [API documentation](https://ataraxis-communication-interface-api.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for project developers.

### Installing the Project

***Note!*** This installation method requires **mamba version 2.3.2 or above**. Currently, all Sun lab automation 
pipelines require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Install the core Sun lab development dependencies into the ***base*** mamba environment via the 
   ```mamba install tox uv tox-uv``` command.
5. Use the ```tox -e create``` command to create the project-specific development environment followed by 
   ```tox -e install``` command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project. 
   Currently, this library supports the three latest stable versions. It is recommended to use a tool like 
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check the [tox.ini file](tox.ini) for details about the 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All pull requests for this project have to successfully complete the ```tox``` task before being merged. 
To expedite the task’s runtime, use the ```tox --parallel``` command to run some tasks in-parallel.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself may experience runtime failures. In 
most cases, this is related to their caching behavior. If an unintelligible error is encountered with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a CLI command typically solves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the  
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-communication-interface/tags) for the available 
project releases.

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
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.

---
