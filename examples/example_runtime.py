# This file demonstrates the use of MicroControllerInterface with custom ModuleInterface classes.
#
# Note that this example is intentionally kept simple and does not cover all possible use cases. Overall, this example
# demonstrates how to use the PC client to control custom hardware modules running on the Arduino or Teensy
# microcontroller in real time. It also demonstrates how to access the data received from the microcontroller that is
# saved to disk via the DataLogger instance.
#
# This example is intended to be used together with a microcontroller running the module_integration.cpp from the
# companion ataraxis-micro-controller library: https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart
# See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface#quickstart for more details.
# API documentation: https://ataraxis-communication-interface-api.netlify.app/.
# Authors: Ivan Kondratyev (Inkaros), Jacob Groner.

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
