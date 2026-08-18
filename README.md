# ataraxis-communication-interface

Provides the centralized interface for exchanging commands and data between Arduino and Teensy microcontrollers and
host-computers.

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

The library allows interfacing with custom hardware modules controlled by Arduino or Teensy microcontrollers running the
companion [microcontroller library](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller). To do so, the library
defines a shared API that can be integrated into user-defined interfaces by subclassing the (base) ModuleInterface
class. It also provides the MicroControllerInterface class that manages the microcontroller-PC communication and the
MQTTCommunication class that allows exchanging data between local and remote clients over the MQTT (TCP) protocol. This
library is part of the [Ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) framework for AI-assisted scientific hardware
control.

___

## Features

- Supports Windows, Linux, and macOS.
- Provides the framework for writing and deploying custom interfaces for the hardware module instances managed
  by the companion [microcontroller library](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller).
- Abstracts communication and microcontroller runtime management via the centralized microcontroller interface class.
- Leverages MQTT protocol to support exchanging data between multiple local and remote clients.
- Uses LRU caching to optimize the runtime efficiency of command and parameter message construction.
- Resolves every microcontroller status code into its firmware name and the condition it reports, covering the Kernel,
  base Module, Communication, and TransportLayer code families.
- Allows each custom module interface to register its own explanation for every error code it monitors.
- Contains many sanity checks performed at initialization time to minimize the potential for unexpected
  behavior and data corruption.
- Provides a log data processing pipeline for extracting hardware module and kernel event data from runtime log
  archives, with manifest-based discovery and parallel extraction to Feather (IPC) output files.
- Generates microcontroller manifest files that tag DataLogger output directories with source-to-name mappings,
  enabling downstream tools to identify which log archives were produced by ataraxis-communication-interface.
- Includes an MCP server for AI agent integration (compatible with Claude Desktop and other MCP clients).
- Apache 2.0 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Quickstart](#quickstart)
  - [User-Defined Variables](#user-defined-variables)
  - [Keepalive](#keepalive)
  - [Communication](#communication)
  - [Data Logging](#data-logging)
  - [Log Processing](#log-processing)
  - [Custom Module Interfaces](#custom-module-interfaces)
  - [Implementing Custom Module Interfaces](#implementing-custom-module-interfaces)
  - [Error Handling](#error-handling)
  - [CLI Commands](#cli-commands)
  - [MCP Server](#mcp-server)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

___

## Dependencies

- **MQTT broker**, if the library is intended to be used for sending and receiving data over the MQTT protocol. The
  library was tested with a locally running [mosquitto MQTT broker](https://mosquitto.org/) version **2.1.2**.

For users, all other library dependencies are installed automatically by all supported installation methods. For
developers, see the [Developers](#developers) section for information on installing additional development dependencies.

___

## Installation

### Source

***Note,*** installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the
   [stable releases](https://github.com/Sun-Lab-NBB/ataraxis-communication-interface/tags) that include precompiled
   binary and source code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Run `pip install .` to install the project and its dependencies.

### pip

Use the following command to install the library and all of its dependencies via [pip](https://pip.pypa.io/en/stable/):
`pip install ataraxis-communication-interface`

___

## Usage

### Quickstart

See the [Implementing Custom Module Interfaces](#implementing-custom-module-interfaces) section for instructions on how
to implement module interface classes. The example below should be run together with the companion
[microcontroller module](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart) example. See the
[example_runtime.py](./examples/example_runtime.py) for the .py implementation of this example.

```python
import tempfile
from pathlib import Path

import numpy as np
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, assemble_log_archives
from ataraxis_time import PrecisionTimer, TimerPrecisions

# Imports the TestModuleInterface class from the companion example file (examples/example_interface.py). Installing
# this library exposes that file as the 'ataraxis_communication_interface_examples' package. To run the example from a
# cloned repository instead, use 'from example_interface import TestModuleInterface' from inside the examples directory.
from ataraxis_communication_interface_examples.example_interface import TestModuleInterface

from ataraxis_communication_interface import (
    MICROCONTROLLER_MANIFEST_FILENAME,
    MicroControllerInterface,
    ModuleExtractionConfig,
    KernelExtractionConfig,
    ControllerExtractionConfig,
    create_extraction_config,
    run_log_processing_pipeline,
)

_ECHO_RESPONSE_TIMEOUT: int = 1000
"""The time, in milliseconds, to wait for the microcontroller to answer the one-off echo command before treating the
communication as broken. The firmware answers the echo inside the runtime cycle that dequeues the command, so the round
trip costs two short serial frames and stays two orders of magnitude below this bound."""


# Guards the runtime to support MicroControllerInterface's multiprocessing architecture.
if __name__ == "__main__":
    # Enables the console module to communicate the example's runtime progress via the terminal.
    console.enable()

    # Specifies the directory where to save all incoming and outgoing messages processed by the MicroControllerInterface
    # instance for each hardware module.
    tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes.
    output_directory = Path(tempdir.name)

    # Instantiates the DataLogger, which is used to save all incoming and outgoing MicroControllerInterface messages
    # to disk. See https://github.com/Sun-Lab-NBB/ataraxis-data-structures for more details on DataLogger class.
    data_logger = DataLogger(output_directory=output_directory, instance_name="AMC")
    data_logger.start()  # Starts the DataLogger before it can save any log entries.

    # Defines two interface instances, one for each TestModule used at the same time. Note that each instance uses
    # different module_id codes, but the same type (family) id code.
    interface_1 = TestModuleInterface(module_type=np.uint8(1), module_id=np.uint8(1))
    interface_2 = TestModuleInterface(module_type=np.uint8(1), module_id=np.uint8(2))
    interfaces = (interface_1, interface_2)

    # Instantiates the MicroControllerInterface. Functions similar to the Kernel class from the
    # ataraxis-micro-controller library and abstracts most inner-workings of the library. Expects a Teensy 4.1
    # microcontroller, and the parameters defined below may not be optimal for all supported microcontrollers.
    mc_interface = MicroControllerInterface(
        controller_id=np.uint8(222),
        buffer_size=8192,
        port="/dev/ttyACM0",
        data_logger=data_logger,
        module_interfaces=interfaces,
        name="test_controller",
        baudrate=115200,
        keepalive_interval=5000,
    )
    console.echo(message="Initializing the communication process...")

    # Guards the runtime below so that a failure still releases every asset reserved above. A microcontroller that
    # is absent or unplugged makes start() raise, and without this guard that exception would strand the DataLogger
    # process, its watchdog thread, and its shared memory buffer, and would discard every message already written
    # to disk.
    try:
        # Starts the serial communication with the microcontroller by initializing a separate process that handles the
        # communication. This method may take up to 30 seconds to execute, as it verifies that the microcontroller is
        # configured correctly, given the MicroControllerInterface configuration.
        mc_interface.start()

        console.echo(message="Communication process: Initialized.", level=LogLevel.SUCCESS)
        console.echo(message="Updating hardware module runtime parameters...")

        # The shared memory instances are connected in both processes already. Calling the setup here pins the
        # connection point to a moment the runtime chooses, after the communication process has started.
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

        console.echo(message="Hardware module runtime parameters: Updated.", level=LogLevel.SUCCESS)

        console.echo(message="Sending the 'echo' command to the TestModule 1...")

        # Requests instance 1 to return its echo value. By default, the echo command only runs once.
        interface_1.echo()

        # Waits until the microcontroller responds to the echo command. The interface is configured to update shared
        # memory array index 2 with the received echo value when it receives the response from the microcontroller.
        # The wait is bounded, since an echo the microcontroller never answers would otherwise spin this loop forever.
        echo_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
        while interface_1.shared_memory[2] == 0:
            if echo_timer.elapsed > _ECHO_RESPONSE_TIMEOUT:
                message = (
                    f"Unable to read the echo value from TestModule 1. The microcontroller did not answer the echo "
                    f"command within {_ECHO_RESPONSE_TIMEOUT} milliseconds, which indicates that the communication "
                    f"with the microcontroller is broken."
                )
                console.error(message=message, error=RuntimeError)

        # Retrieves and prints the microcontroller's response. The returned value should match the echo_value
        # parameter set above, which is 121.
        console.echo(message=f"TestModule 1 echo value: {interface_1.shared_memory[2]}.", level=LogLevel.SUCCESS)

        # Demonstrates the use of non-blocking recurrent commands.
        console.echo(message="Executing the example non-blocking runtime, standby for ~5 seconds...")

        # Instructs the first TestModule instance to start pulsing the managed pin (Pin 5 by default). With the
        # parameters sent earlier, it keeps the pin ON for 1 second and keeps it off for ~ 2 seconds (1 from
        # off_duration, 1 from waiting before repeating the command). The microcontroller repeats this command at
        # regular intervals until it is given a new command or receives a 'dequeue' command (see below).
        interface_1.pulse(repetition_delay=np.uint32(1000000), noblock=True)

        # Instructs the second TestModule instance to start sending its echo value to the PC every 500 milliseconds.
        interface_2.echo(repetition_delay=np.uint32(500000))

        # Delays for 5 seconds, accumulating echo values from TestModule 2 and pin On / Off notifications from
        # TestModule 1. Uses the PrecisionTimer instance to delay the main process for 5 seconds.
        delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
        delay_timer.delay(delay=5, block=False)

        # Cancels both recurrent commands by issuing a dequeue command. Note, the dequeue command does not interrupt
        # already running commands, it only prevents further command repetitions.
        interface_1.reset_command_queue()
        interface_2.reset_command_queue()

        # The result seen here depends on the communication speed between the PC and the microcontroller and the
        # precision of the microcontroller's clock. For Teensy 4.1, which was used to write this example, the pin is
        # expected to pulse ~2 times and the echo value is expected to be transmitted ~10 times during the test
        # period.
        console.echo(message="Non-blocking runtime: Complete.", level=LogLevel.SUCCESS)
        console.echo(message=f"TestModule 1 Pin pulses: {interface_1.shared_memory[0]}")
        console.echo(message=f"TestModule 2 Echo values: {interface_2.shared_memory[1]}")

        # Resets the pulse and echo counters before executing the demonstration below.
        interface_1.shared_memory[0] = 0
        interface_2.shared_memory[1] = 0

        # Repeats the example above, but now uses blocking commands instead of non-blocking.
        console.echo(message="Executing the example blocking runtime, standby for ~5 seconds...")
        interface_1.pulse(repetition_delay=np.uint32(1000000), noblock=False)
        interface_2.echo(repetition_delay=np.uint32(500000))
        delay_timer.delay(delay=5, block=False)  # Reuses the same delay timer
        interface_1.reset_command_queue()
        interface_2.reset_command_queue()

        # The pulse period is the same in both modes, so the pin is again expected to pulse ~2 times. This time the pin
        # pulsing performed by module 1 interferes with the echo command performed by module 2, so the echo counter is
        # expected to fall below the non-blocking figure above.
        console.echo(message="Blocking runtime: Complete.", level=LogLevel.SUCCESS)
        console.echo(message=f"TestModule 1 Pin pulses: {interface_1.shared_memory[0]}")
        console.echo(message=f"TestModule 2 Echo values: {interface_2.shared_memory[1]}")

    finally:
        # Retires the runtime assets in dependency order, which is what makes this teardown work on the failure path
        # as well as the success path. The communication process stops first, because it is the last writer feeding
        # log entries to the DataLogger, and stopping the logger underneath a live writer would drop the entries still
        # in flight. The DataLogger stops second, which drains its queue and flushes every remaining entry to disk as
        # a .npy file. The archive assembly runs last, since it consolidates those .npy files into .npz archives and
        # only sees a complete set once the logger process has exited. This example writes into a temporary directory
        # that goes away with the script, so the archive is discarded on the failure path along with its sources.
        mc_interface.stop()
        console.echo(message="Communication process: Stopped.", level=LogLevel.SUCCESS)

        # Stops the DataLogger and assembles all logged data into a single .npz archive file. This step is required to
        # be able to extract the logged message data for further analysis.
        data_logger.stop()
        console.echo(message="Assembling the message log archive...")
        assemble_log_archives(log_directory=data_logger.output_directory, remove_sources=True, verbose=True)

    # To process the data logged during runtime, first generate a precursor extraction configuration from the
    # microcontroller manifest. The manifest is automatically created by MicroControllerInterface during __init__.
    console.echo(message="Creating extraction configuration from manifest...")
    manifest_path = data_logger.output_directory / MICROCONTROLLER_MANIFEST_FILENAME
    config = create_extraction_config(manifest_path=manifest_path)

    # The generated config has empty event_codes for each module. Fill in the event codes that should be extracted.
    # Event codes 52 (kHigh), 53 (kLow), and 54 (kEcho) are the TestModule event codes demonstrated above.
    config.controllers[0] = ControllerExtractionConfig(
        controller_id=222,
        modules=(
            ModuleExtractionConfig(module_type=1, module_id=1, event_codes=(52, 53, 54)),
            ModuleExtractionConfig(module_type=1, module_id=2, event_codes=(52, 53, 54)),
        ),
        kernel=KernelExtractionConfig(event_codes=(1,)),  # Extracts kernel status code 1 (setup complete) events.
    )

    # Saves the filled-in config to disk. The pipeline reads it from disk to support both CLI and API usage.
    config_path = data_logger.output_directory / "extraction_config.yaml"
    config.to_yaml(file_path=config_path)
    console.echo(message=f"Extraction config written to: {config_path}", level=LogLevel.SUCCESS)

    # Runs the log processing pipeline. Extracts hardware module and kernel message data from the log archives and
    # writes the results to feather (IPC) files for downstream analysis.
    console.echo(message="Processing the logged message data...")
    output_path = Path(tempfile.mkdtemp())
    run_log_processing_pipeline(
        log_directory=data_logger.output_directory,
        output_directory=output_path,
        config=config_path,
    )
    console.echo(message=f"Processing complete. Feather output written to: {output_path}", level=LogLevel.SUCCESS)
```

### User-Defined Variables

The metadata variables that let the PC interface individuate and address the managed microcontroller and its hardware
module instances come from the end user. **Each end user has to manually define these values both for the
microcontroller and the PC.**

Two of these variables, the `module_type` and the `module_id` are used by the (base) **ModuleInterface** class. The
remaining `controller_id` variable is used by the **MicroControllerInterface** class. See the
[companion library's](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#user-defined-variables) README for more
details about each user-defined metadata variable. Typically, these variables are set in the microcontroller code and
the PC code is adjusted to match the microcontroller code's state.

### Keepalive

To work as intended, **both the PC (MicroControllerInterface instance) and the microcontroller (Kernel instance) must be
configured to use the same keepalive interval.**

When enabled, the MicroControllerInterface instance sends a 'keepalive' command at regular intervals, specified by the
`keepalive_interval` initialization argument. If the microcontroller does not receive the command for
**two consecutive interval windows**, it aborts the runtime by resetting the microcontroller's hardware and software to
the default state and sends an error message to the PC. If the PC does not receive the microcontroller's acknowledgement
that it has received the keepalive command within **one interval window from sending the previous command**, it aborts
the communication runtime with an error.

The keepalive functionality is **disabled** (set to 0) by default, but it is recommended to enable it for most use
cases. See the [API documentation for the MicroControllerInterface class](#api-documentation) for more details on
configuring the keepalive messaging.

***Note,*** the appropriate keepalive interval depends on the communication speed and the CPU frequency of the
microcontroller. For a fast microcontroller (Teensy 4.1) that uses the USB communication interface, an appropriate
keepalive interval is typically measured in milliseconds (100 to 500). For a slower microcontroller (Arduino Mega) with
a UART communication interface using the baudrate of 115200, the appropriate keepalive interval is typically measured
in seconds (2 to 5).

### Communication

During runtime, all communication with the microcontroller is routed via the MicroControllerInterface instance that
implements the centralized communication and control interface for each microcontroller. To optimize runtime
performance, the communication is managed by a daemonic process running on a separate CPU core.

When the data is sent to the microcontroller, it is first transferred to the communication process, which then transmits
it to the microcontroller. When the data is received from the microcontroller, it is mostly handled by the communication
process, unless the end user implements the logic for routing it to other runtime processes.

### Data Logging

This library relies on the [DataLogger](https://github.com/Sun-Lab-NBB/ataraxis-data-structures#datalogger) class to
save all incoming and outgoing messages to disk during PC-microcontroller communication. Each message sent or received
by the PC is serialized and saved as an uncompressed **.npy** file.

The same DataLogger instance as used by the MicroControllerInterface instances may be shared by multiple other Ataraxis
assets that generate log entries, such as [VideoSystem](https://github.com/Sun-Lab-NBB/ataraxis-video-system) classes.
To support using the same logger instance for multiple concurrently active sources, **each source has to use a unique
identifier value (controller id) when sending data to the logger instance**.

Each MicroControllerInterface instance automatically writes a `microcontroller_manifest.yaml` file into the DataLogger
output directory during initialization. The manifest associates the controller_id with the human-readable `name`
provided to the MicroControllerInterface constructor, along with the list of module sources. When multiple
MicroControllerInterface instances share the same DataLogger, each instance registers its entry in the same manifest
file, replacing the entry any earlier instance registered under the same controller_id. The manifest is required by the
[log processing](#log-processing) pipeline to validate which `.npz` archives were produced by
ataraxis-communication-interface and to resolve source IDs for processing.

***Note,*** currently, only the MicroControllerInterface supports logging data to disk.

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

#### Onset Timestamp

Each MicroControllerInterface generates an `onset` timestamp as part of its `start()` method runtime. This log entry
uses a modified data order and stores the current UTC time, accurate to microseconds, as the total number of
microseconds elapsed since the UTC epoch onset. All further log entries for the same source use the timestamp section
of their payloads to communicate the number of microseconds elapsed since the onset timestamp acquisition.

The onset log entry uses the following data organization order:
1. The uint8 id of the data source (microcontroller).
2. The uint64 value **0** that occupies 8 bytes following the source id. A 'timestamp' value of 0 universally indicates
   that the log entry stores the onset timestamp.
3. The uint64 value that stores the number of microseconds elapsed since the UTC epoch onset. This value specifies the
   current time when the onset timestamp was generated.

#### Working with MicroControllerInterface Logs

See the [quickstart](#quickstart) example above for a demonstration on how to assemble and process the message
log archives generated by the MicroControllerInterface instance at runtime.

### Log Processing

This library includes a log data processing pipeline for extracting hardware module and kernel event data from the
`.npz` log archives generated by MicroControllerInterface instances at runtime. The pipeline reads archives produced by
the [DataLogger](https://github.com/Sun-Lab-NBB/ataraxis-data-structures#datalogger), extracts messages matching the
event codes specified in an extraction configuration, and writes the results as [Polars](https://pola.rs/) DataFrames in
Apache Feather (IPC) format.

The pipeline uses the [microcontroller manifest](#data-logging) to validate that `.npz` archives were produced by
ataraxis-communication-interface. A `microcontroller_manifest.yaml` file must be present in the log directory for
processing to succeed. Controller IDs to process are resolved directly from the extraction configuration and validated
against the manifest. The `axci config create` command generates the precursor extraction configuration from that
manifest, populating every controller and module with empty event codes for the user to fill in.

One recording writes one MicroControllerInterface set to one DataLogger, so exactly one manifest is supported per
invocation. A log directory tree holding several manifests, or archives written by several DataLogger instances,
spans several recordings and is rejected with a diagnostic naming the topology it detected.

Processing is split across two entry points that share their job resolution but not their execution. The `axci process`
CLI command and the `run_log_processing_pipeline()` function target a single recording and run its archives one at a
time in the calling process, or run the single job a caller names by its canonical identifier. The
[MCP server](#mcp-server) log processing tools handle archive discovery, batch preparation, and status monitoring for
batches spanning many recordings, admitting jobs against a core budget and a memory budget and running them in one
shared process pool. Both write a YAML-based processing tracker that manages job lifecycle (scheduled, running,
succeeded, or failed), and both write every output file into a `microcontroller_data/` subdirectory under the specified
output directory.

Each job targets exactly one log archive. A caller that weighs jobs against a budget sizes each one from its own
archive. A batch mixing a long recording with a short one therefore gives each the width its own archive earns rather
than one width chosen for the whole run. The job resolution, the identifier generation, and the sizing model are
exported as callable functions, so an external scheduler derives the same values this library dispatches with instead
of re-deriving them.

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
MicroControllerInterface calls these methods as part of the remote communication process's runtime cycle to work with
the data sent by the custom hardware module.

#### initialize_remote_assets

This method is called by the MicroControllerInterface once for each ModuleInterface at the beginning of the
communication cycle. The method should be used to initialize or configure custom assets (queues, shared memory buffers,
timers, etc.) that need to be processed from the (remote) communication process.

```python
def initialize_remote_assets(self) -> None:
    # Connects to the shared memory array from the remote process.
    self._shared_memory.connect()
```

#### terminate_remote_assets

This method is the inverse of the initialize_remote_assets() method. It is called by the MicroControllerInterface for
each ModuleInterface at the end of the communication cycle. This method should be used to clean up (terminate) any
assets initialized at the beginning of the communication runtime to ensure all resources are released before the process
is terminated.

```python
def terminate_remote_assets(self) -> None:
    # The shared memory array must be manually disconnected from each process that uses it to prevent runtime
    # errors.
    self._shared_memory.disconnect()
```

#### process_received_data

This method allows processing incoming module messages as they are received by the PC. The MicroControllerInterface
instance calls this method for any ModuleState or ModuleData message received from the hardware module, if the
event code of the message matches one of the codes in the data_codes attribute of the module's interface instance.

***Note,*** the MicroControllerInterface class ***automatically*** saves (logs) each received and sent message to disk.
Therefore, this method should ***not*** be used to save the data for post-runtime processing. Instead, this method
should be used to process the data in real time or route it to other processes / machines for real time processing.

Since all ModuleInterfaces used by the same MicroControllerInterface share the communication process,
**process_received_data() should not use complex logic or processing**. Treat this method as a hardware interrupt
function: its main goal is to handle the incoming data as quickly as possible and allow the communication loop to run
for other modules.

This example demonstrates the implementation of the processing method to send the data back to the main process:

```python
from ataraxis_communication_interface import ModuleData, ModuleState


def process_received_data(self, message: ModuleData | ModuleState) -> None:
    # Event codes 52 and 53 are used to communicate the current state of the output pin managed by the example
    # module. State messages transmit these event-codes, so there is no additional data to parse other than
    # event codes.
    if message.event == 52 or message.event == 53:
        # Code 52 indicates that the pin outputs a HIGH signal, code 53 indicates the pin outputs a LOW signal.
        # If the pin state has changed from HIGH (52) to LOW (53), increments the pulse count stored in the shared
        # memory array.
        if message.event == 53 and self._previous_pin_state:
            # A compound update reads the counter and then writes the incremented value back. Subscript access
            # locks each of those halves on its own, so another process writing the same index between them
            # erases the increment. The array() context manager holds one lock across both halves instead.
            with self._shared_memory.array() as shared_data:
                shared_data[0] += 1

        # Sets the previous pin state value to match the recorded pin state.
        self._previous_pin_state = bool(message.event == 52)

    # The module uses code 54 messages to return its echo value to the PC.
    elif isinstance(message, ModuleData) and message.event == 54:
        # The echo value is transmitted by a Data message. In addition to the event code, Data messages include a
        # data_object. Upon reception, the data object is automatically deserialized into the appropriate
        # Python object, so it can be accessed directly.
        self._shared_memory[2] = message.data_object  # Records the received data value to the shared memory.

        # Increments the received echo value count. This is a compound update, so it takes one lock across the
        # read and the write, as the pulse counter above does.
        with self._shared_memory.array() as shared_data:
            shared_data[1] += 1
```

#### Sending Data to the Microcontroller

In addition to abstract methods, each interface may need to send data to the microcontroller. Broadly, the outgoing
messages are divided into two categories: **commands** and **parameter updates**. Command messages instruct the module
to perform a specified action. Parameter updates are used to overwrite the module's runtime parameters to broadly adjust
how the module behaves while executing commands.

Each interface should use the `send_parameters()` method inherited from the (base) ModuleInterface class to send
parameter update messages to the managed module and the `send_command()` method to send command messages to the managed
module. These utility methods abstract the necessary steps for packaging and transmitting the input data to the module.

***Note,*** these methods use LRU caching to optimize their runtime speed and minimize the delay between submitting
the message for transmission and it being sent to the microcontroller. Therefore, most command and parameter update
functions / methods should be simple wrappers around these inherited methods. See the API documentation for the
ModuleInterface class for the details about these methods inherited by each child interface class.

### Error Handling

The microcontroller reports every runtime fault as a byte-code, and the codes come from four families defined across
the companion microcontroller libraries. The Kernel and the base Module class each define their own status codes, and
a fault that originates in the serial link carries a second pair of codes from the Communication and TransportLayer
classes inside its message payload. This library resolves each received code into the firmware name of the code and
the condition that code reports, then raises the result as a RuntimeError from the communication process.

The messages state what the reported codes establish and stop there, because a status code records where a fault was
detected rather than what caused it. A packet corrupted between the microcontroller and the PC surfaces as the
following, rather than as the raw codes 3, 52, and 19:

```text
Microcontroller 3 ('actor_controller') Kernel status RECEPTION_ERROR (code 3) during Kernel command RECEIVE_DATA
(code 1). Reception failed. Communication RECEPTION_ERROR (code 52): could not read a complete message from the
serial stream. TransportLayer CRC_CHECK_FAILED (code 19): CRC mismatch, packet bytes likely corrupted in transit.
```

A code matching no member of its firmware enumeration reports that it falls outside the range this library resolves. The
KernelStatusCodes, ModuleStatusCodes, CommunicationStatusCodes, and TransportStatusCodes enumerations exported by this
library mirror the four firmware families, so post-runtime tooling reads a logged event code through the same
vocabulary.

#### Custom Module Error Codes

The status codes above are the ones the firmware defines for every module, and they occupy the reserved event code
range below MINIMUM_CUSTOM_STATUS_CODE. Each custom hardware module assigns its own event codes from the range that
MINIMUM_CUSTOM_STATUS_CODE and MAXIMUM_CUSTOM_STATUS_CODE bound, and only the module's author knows what those codes
mean.

To surface that knowledge to the operator, pass the `error_codes` argument of the ModuleInterface constructor a
dictionary that maps each monitored error code to its explanation. Receiving a message with one of those codes raises
a RuntimeError carrying the matching explanation alongside the reporting module, the command it was executing, and any
data object the message contained:

```python
import numpy as np

from ataraxis_communication_interface import ModuleInterface


class WaterValveInterface(ModuleInterface):
    def __init__(self) -> None:
        super().__init__(
            module_type=np.uint8(5),
            module_id=np.uint8(1),
            name="water_valve",
            data_codes={np.uint8(51), np.uint8(52)},  # kOpen and kClosed.
            error_codes={
                np.uint8(56): (  # kInvalidToneConfiguration.
                    "The valve was commanded to emit a tone while the instance is not configured for audible "
                    "tones. Set the tone duration parameter above 0 before issuing tone commands."
                ),
            },
        )

    # The three abstract methods covered above are omitted here for brevity.
```

***Note,*** the constructor rejects any error or data code outside the custom range. The runtime resolves every code
below that range through the service code handling described above, so a code declared from there never reaches the
interface that declared it.

### CLI Commands

This library provides the `axci` CLI that exposes the following commands:

| Command         | Description                                                              |
|-----------------|--------------------------------------------------------------------------|
| `id`            | Discovers all connected Ataraxis microcontrollers and returns their IDs  |
| `mqtt`          | Checks whether an MQTT broker is reachable at the specified host / port  |
| `config create` | Generates a precursor extraction config from a microcontroller manifest  |
| `config show`   | Displays the contents of an extraction configuration file                |
| `process`       | Processes log archives to extract hardware module and kernel event data  |
| `mcp`           | Starts the MCP server for AI agent integration                           |

Use `axci --help` or `axci COMMAND --help` for detailed usage information. A command whose execution fails reports the
reason through the console at the error level and exits zero, so a shell driving the CLI reads the reported message
rather than an interpreter traceback. Click rejects a missing or malformed option before the command runs and exits 2.

***Note,*** a script chaining commands with `set -e` or `&&` therefore continues past a failed command. Such a script
reads the reported output to decide whether the command succeeded.

A CLI-driven extraction generates the configuration from the manifest and then processes the archives that manifest
tags:

```bash
# Generates the precursor configuration. Fill in the event codes for each module entry before processing.
axci config create -m /path/to/logs/microcontroller_manifest.yaml -o /path/to/extraction_config.yaml

axci process -ld /path/to/logs -od /path/to/output -c /path/to/extraction_config.yaml
```

### MCP Server

This library provides an MCP server that exposes microcontroller discovery, MQTT connectivity checking, manifest
management, extraction configuration management, and log data processing functionality for AI agent integration.

#### Starting the Server

Start the MCP server using the CLI:

```bash
axci mcp
```

#### Available Tools

| Tool                                  | Description                                                                  |
|---------------------------------------|------------------------------------------------------------------------------|
| `list_microcontrollers_tool`          | Discovers serial ports connected to Ataraxis microcontrollers and returns IDs|
| `check_mqtt_broker_tool`              | Checks whether an MQTT broker is reachable at the specified host and port    |
| `assemble_log_archives_tool`          | Consolidates raw .npy log entries into .npz archives by source ID            |
| `read_microcontroller_manifest_tool`  | Reads a microcontroller manifest file and returns its contents               |
| `write_microcontroller_manifest_tool` | Writes or updates a microcontroller manifest file in a log directory         |
| `discover_microcontroller_data_tool`  | Discovers confirmed microcontroller recordings under a root directory        |
| `read_extraction_config_tool`         | Reads an extraction configuration from a YAML file and returns its contents  |
| `write_extraction_config_tool`        | Writes an extraction configuration to a YAML file from structured data       |
| `validate_extraction_config_tool`     | Validates an extraction config against a manifest for completeness           |
| `prepare_log_processing_batch_tool`   | Prepares a batch of log processing jobs across multiple directories          |
| `execute_log_processing_jobs_tool`    | Prepares and executes log processing jobs against a core and a memory budget |
| `get_log_processing_status_tool`      | Returns the current status of the active log processing session              |
| `get_log_processing_timing_tool`      | Returns timing information for all jobs in the active session                |
| `cancel_log_processing_tool`          | Cancels the active log processing execution session                          |
| `reset_log_processing_jobs_tool`      | Resets the named source IDs' jobs, or all jobs, in a tracker for re-execution|
| `get_batch_status_overview_tool`      | Summarizes processing status for all log directories under a root directory  |
| `verify_processing_output_tool`       | Verifies completeness and schema correctness of processed output             |
| `query_extracted_events_tool`         | Queries and samples extracted event data from feather output files           |
| `clean_log_processing_output_tool`    | Deletes processed output directories for clean re-processing                 |

#### Client Registration

MCP server registration and Claude Code skill assets for this library are distributed through the
[ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace as part of the **communication** plugin. The companion
**microcontroller** plugin provides firmware-side skills for implementing the C++ Module subclasses that pair with the
PC-side ModuleInterface subclasses defined by this library. Install both plugins from the marketplace to register the
MCP server with compatible clients and make all associated skills available.

___

## API Documentation

See the [API documentation](https://ataraxis-communication-interface-api.netlify.app/) for the detailed description of
the methods and classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to modify the
source code of this library.

### Installing the Project

***Note,*** this installation method requires **mamba version 2.3.2 or above**. Currently, all Ataraxis framework
automation pipelines require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge)
installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Install the core Ataraxis framework development dependencies into the ***base*** mamba environment via the
   `mamba install tox uv tox-uv` command.
5. Use the `tox -e create` command to create the project-specific development environment followed by `tox -e install`
   command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project.
   Currently, this library supports the three latest stable versions. It is recommended to use a tool like
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project uses `tox` for development automation. The following tox environments are available:

| Environment          | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `lint`               | Runs ruff formatting, ruff linting, and mypy type checking   |
| `stubs`              | Generates py.typed marker and .pyi stub files                |
| `{py312,...}-test`   | Runs the test suite via pytest for each supported Python     |
| `coverage`           | Aggregates test coverage and applies the 100% coverage gate  |
| `docs`               | Builds the API documentation via Sphinx                      |
| `build`              | Builds sdist and wheel distributions                         |
| `upload`             | Uploads distributions to PyPI via twine                      |
| `deploy`             | Uploads the built documentation to the Netlify site          |
| `install`            | Builds and installs the project into its mamba environment   |
| `uninstall`          | Uninstalls the project from its mamba environment            |
| `create`             | Creates the project's mamba development environment          |
| `remove`             | Removes the project's mamba development environment          |
| `provision`          | Recreates the mamba environment from scratch                 |
| `export`             | Exports the mamba environment as a .yml file                 |
| `import`             | Creates or updates the mamba environment from a .yml file    |

Run any environment using `tox -e ENVIRONMENT`. For example, `tox -e lint`.

***Note,*** all pull requests for this project have to successfully complete the `tox` task before being merged. To
expedite the task's runtime, use the `tox --parallel` command to run some tasks in parallel.

### AI-Assisted Development

Claude Code skills and other AI development assets for this project are distributed through the
[ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace across three plugins:

- **communication** plugin: Carries the `axci mcp` server registration and the communication workflow skills.
- **microcontroller** plugin: Carries the firmware-side skills for the companion
  [ataraxis-micro-controller](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) C++ library.
- **automation** plugin: Carries the shared development skills that enforce Ataraxis framework conventions.

Install all three plugins from the marketplace to make all associated skills and development tools available to
compatible AI coding agents.

### Automation Troubleshooting

Many packages used in `tox` automation pipelines (uv, mypy, ruff) and `tox` itself may experience runtime failures. In
most cases, this is related to their caching behavior. If an unintelligible error is encountered with any of the
automation components, deleting the corresponding cache directories (`.tox`, `.ruff_cache`, `.mypy_cache`, etc.)
manually or via a CLI command typically resolves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-communication-interface/tags) for the available
project releases.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Jacob Groner ([Jgroner11](https://github.com/Jgroner11))

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.

___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.
