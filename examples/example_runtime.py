# This file demonstrates the usage of MicroControllerInterface with custom ModuleInterface classes.
#
# Note that this example is intentionally kept simple and does not cover all possible use cases. If you need a more
# complex example, check one of the Sun Lab libraries used for scientific data acquisition. Overall, this example
# demonstrates how to use the PC client to control custom hardware modules running on the microcontroller in real time.
# It also demonstrates how to access the data received from the microcontroller, that is saved to disk via the
# DataLogger class.
#
# This example is intended to be used together with a microcontroller running the module_integration.cpp from the
# companion ataraxis-micro-controller library: https://github.com/Sun-Lab-NBB/ataraxis-micro-controller#quickstart
# See https://github.com/Sun-Lab-NBB/ataraxis-communication-interface#quickstart for more details.
# API documentation: https://ataraxis-communication-interface-api.netlify.app/.
# Authors: Ivan Kondratyev (Inkaros), Jacob Groner.

# Imports the necessary assets, including the TestModuleInterface class
from pathlib import Path

import numpy as np
from ataraxis_time import PrecisionTimer
from example_interface import TestModuleInterface
from ataraxis_data_structures import DataLogger

from ataraxis_communication_interface import MicroControllerInterface

# Since MicroControllerInterface uses multiple processes, it has to be called with the '__main__' guard
if __name__ == "__main__":
    # Instantiates the DataLogger, which is used to save all incoming and outgoing MicroControllerInterface messages
    # to disk. See https://github.com/Sun-Lab-NBB/ataraxis-data-structures for more details on DataLogger class.
    output_directory = Path("/home/cyberaxolotl/Desktop/Demos/AXCI")  # Change this to your desired output directory
    data_logger = DataLogger(output_directory=output_directory, instance_name="AMC")

    # Defines two interface instances, one for each TestModule used at the same time. Note that each instance uses
    # different module_id codes, but the same type (family) id code. These codes match the values used on the
    # microcontroller.
    interface_1 = TestModuleInterface(module_type=np.uint8(1), module_id=np.uint8(1))
    interface_2 = TestModuleInterface(module_type=np.uint8(1), module_id=np.uint8(2))
    interfaces = (interface_1, interface_2)

    # Defines microcontroller parameters necessary to establish serial communication. Critically, this example uses a
    # Teensy 4.1 microcontroller, and the parameters defined below may not work for your microcontroller!
    # See MicroControllerInterface docstrings / API documentation for more details about each of these parameters.
    controller_id = np.uint8(222)  # Matches the microcontroller ID defined in the microcontroller's main.cpp file
    microcontroller_serial_buffer_size = 8192
    baudrate = 115200
    port = "/dev/ttyACM0"

    # Instantiates the MicroControllerInterface. This class functions similar to the Kernel class from the
    # ataraxis-micro-controller library and abstracts most inner-workings of the library. This interface also allows
    # issuing controller-wide commands and parameters.
    mc_interface = MicroControllerInterface(
        controller_id=controller_id,
        data_logger=data_logger,
        module_interfaces=interfaces,
        microcontroller_serial_buffer_size=microcontroller_serial_buffer_size,
        microcontroller_usb_port=port,
        baudrate=baudrate,
    )

    # Initialization can take some time. Notifies the user that the process is initializing.
    print("Initializing the communication process...")

    # Starts the logging process. By default, the process uses a separate core (process) and 5 concurrently active
    # threads to log all incoming data. The same data logger instance can be used by multiple MiroControllerInterface
    # instances and other Ataraxis classes that support logging data. Note, if this method is not called, no data
    # will be saved to disk.
    data_logger.start()

    # Starts the serial communication with the microcontroller. This method may take up to 15 seconds to execute, as
    # it verifies that the microcontroller is configured correctly, given the MicroControllerInterface configuration.
    # Also, this method JIT-compiles some assets as it runs, which speeds up all future communication.
    mc_interface.start()

    # As a safety feature, the microcontroller is locked when the communication starts. This prevents the
    # microcontroller from changing the states of output pins, but does not interfere with reading input pins or
    # setting runtime parameters. Since this demonstration manipulates an output pin, we need to unlock the
    # microcontroller before proceeding further.
    mc_interface.unlock_controller()

    # You have to manually generate and submit each module-addressed command (or parameter message) to the
    # microcontroller. This is in contrast to MicroControllerInterface commands, which are sent to the microcontroller
    # automatically (see unlock_controller above).

    # Generates and sends new runtime parameters to both hardware module instances running on the microcontroller.
    # On and Off durations are in microseconds. 1 second = 1_000_000 microseconds.
    interface_1.set_parameters(
        on_duration=np.uint32(1000000), off_duration=np.uint32(1000000), echo_value=np.uint16(121)
    )
    interface_2.set_parameters(
        on_duration=np.uint32(5000000), off_duration=np.uint32(5000000), echo_value=np.uint16(333)
    )

    # Requests instance 1 to return its echo value. By default, the echo command only runs once.
    interface_1.echo()

    # Since TestModuleInterface class used in this demonstration is configured to output all received data via
    # MicroControllerInterface's multiprocessing queue, we can access the queue to verify the returned echo value.

    # Waits until the microcontroller responds to the echo command.
    while interface_1.output_queue.empty():
        continue

    # Retrieves and prints the microcontroller's response. The returned value should match the parameter set above: 121.
    print(f"TestModule instance 1 returned {interface_1.output_queue.get()[2]}")

    # We can also set both instances to execute two different commands at the same time if both commands are noblock
    # compatible. The TestModules are written in a way that these commands are noblock compatible.

    # Instructs the first TestModule instance to start pulsing the managed pin (Pin 5 by default). With the parameters
    # we sent earlier, it will keep the pin ON for 1 second and then keep it off for ~ 2 seconds (1 from off_duration,
    # 1 from waiting before repeating the command). The microcontroller will repeat this command at regular intervals
    # until it is given a new command or receives a 'dequeue' command (see below).
    interface_1.pulse(repetition_delay=np.uint32(1000000), noblock=True)

    # Also instructs the second TestModule instance to start sending its echo value to the PC once every 500
    # milliseconds.
    interface_2.echo(repetition_delay=np.uint32(500000))

    # Delays for 10 seconds, accumulating echo values from TestModule 2 and pin On / Off notifications from TestModule
    # 1. Uses the PrecisionTimer class to delay the main process thread for 10 seconds, without blocking other
    # concurrent threads.
    delay_timer = PrecisionTimer("s")
    delay_timer.delay_noblock(10)

    # Cancels both recurrent commands by issuing a dequeue command. Note, the dequeue command does not interrupt already
    # running commands, it only prevents further command repetitions.
    interface_1.reset_command_queue()
    interface_2.reset_command_queue()

    # Counts the number of pin pulses and received echo values accumulated during the delay.
    pulse_count = 0
    echo_count = 0
    while not interface_1.output_queue.empty():
        message = interface_1.output_queue.get()
        # Pin pulses are counted when the microcontroller sends a notification that the pin was set to HIGH state.
        # The microcontroller also sends a notification when Pin state is LOW, but we do not consider it here.
        if message[0] == interface_1.module_id and message[1] == "pin state" and message[2]:
            pulse_count += 1

    while not interface_2.output_queue.empty():
        message = interface_2.output_queue.get()
        # Echo values are only counted if the echo value matches the value we set via the parameter message.
        if message[0] == interface_2.module_id and message[1] == "echo value" and message[2] == 333:
            echo_count += 1

    # The result seen here depends on the communication speed between the PC and the microcontroller and the precision
    # of microcontroller clocks. For Teensy 4.1, which was used to write this example, we expect the pin to pulse 4
    # times and the echo value to be transmitted 21 times during the test period. Note that these times are slightly
    # higher than the theoretically expected 3 and 20. This is because the modules are fast enough to start an extra
    # cycle for both pulse() and echo() commands in the time it takes the dequeue command to arrive to the
    # microcontroller.
    print("TestModule 1 Pin pulses:", pulse_count)
    print("TestModule 2 Echo values:", echo_count)

    # You can also try the same test as above, but this time with pulse noblock=False. In this case, pulsing the pin and
    # returning echo values will interfere with each other, which will drastically reduce the number of returned echo
    # values.

    # Stops the serial communication and the data logger processes.
    mc_interface.stop()
    data_logger.stop()

    # Compresses all logged data into a single .npz archive. This is a prerequisite for reading the logged data via the
    # ModuleInterface default methods!
    data_logger.compress_logs(remove_sources=True)  # Removes intermediate .npy log entries to save space.

    # If you want to process the data logged during runtime, you first need to extract it from the archive. To help
    # with this, the base ModuleInterface exposes a method that reads the data logged during runtime. The method
    # ONLY reads the data received from the module with the same type and ID as the ModuleInterface whose method is
    # called and only reads module messages with event-codes above 51. In other words, the method ignores
    # system-reserved messages that are also logged, but are likely not needed for further data analysis.

    # Log compression generates an '.npz' archive for each unique source. For MicroControllerInterface class, its
    # controlled_id is used as the source_id. In our case, the log is saved under '222_data_log.npz'.
    log_data = interface_1.extract_logged_data()
    print(f"Extracted event data: {log_data}")
