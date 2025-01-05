from abc import abstractmethod
from typing import Any
from pathlib import Path
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from ataraxis_data_structures import DataLogger, SharedMemoryArray

from .communication import (
    KernelData as KernelData,
    ModuleData as ModuleData,
    KernelState as KernelState,
    ModuleState as ModuleState,
    KernelCommand as KernelCommand,
    SerialProtocols as SerialProtocols,
    KernelParameters as KernelParameters,
    ModuleParameters as ModuleParameters,
    SerialPrototypes as SerialPrototypes,
    MQTTCommunication as MQTTCommunication,
    OneOffModuleCommand as OneOffModuleCommand,
    SerialCommunication as SerialCommunication,
    DequeueModuleCommand as DequeueModuleCommand,
    ModuleIdentification as ModuleIdentification,
    RepeatedModuleCommand as RepeatedModuleCommand,
    ControllerIdentification as ControllerIdentification,
)

class ModuleInterface:
    """The base class from which all custom ModuleInterface classes should inherit.

    Inheriting from this class grants all subclasses the static API that the MicroControllerInterface class uses to
    interface with specific module interfaces. It is essential that all abstract methods defined in this class are
    implemented for each custom module interface implementation that subclasses this class.

    Notes:
        Similar to the ataraxis-micro-controller (AXMC) library, the interface class has to be implemented separately
        for each custom module. The (base) class exposes the static API used by the MicroControllerInterface class to
        integrate each custom interface implementation with the general communication runtime cycle. To make this
        integration possible, this class defines some abstract (pure virtual) methods that developers have to implement
        for their interfaces. Follow the implementation guidelines in the docstrings of each abstract method and check
        the examples for further guidelines on how to implement each abstract method.

        When inheriting from this class, remember to call the parent's init method in the child class init method by
        using 'super().__init__()'! If this is not done, the MicroControllerInterface class will likely not be able to
        properly interact with your custom interface class!

        All data received from or sent to the microcontroller is automatically logged as byte-serialized numpy arrays.
        If you do not need any additional processing steps, such as sending or receiving data over MQTT, do not enable
        any custom processing flags when initializing this superclass!

        In addition to interfacing with the module, the class also contains methods used to parse logged module data.

    Args:
        module_type: The id-code that describes the broad type (family) of custom hardware modules managed by this
            interface class. This value has to match the code used by the custom module implementation on the
            microcontroller. Valid byte-codes range from 1 to 255.
        module_id: The code that identifies the specific custom hardware module instance managed by the interface class
            instance. This is used to identify unique modules in a broader module family, such as different rotary
            encoders if more than one is used at the same time. Valid byte-codes range from 1 to 255.
        mqtt_communication: Determines whether this interface needs to communicate with MQTT. If your implementation of
            the process_received_data() method requires sending data to MQTT via MQTTCommunication, set this flag to
            True when implementing the class. Similarly, if your interface is configured to receive commands from
            MQTT, set this flag to True.
        error_codes: A set that stores the numpy uint8 (byte) codes used by the interface module to communicate runtime
            errors. This set will be used during runtime to identify and raise error messages in response to
            managed module sending error State and Data messages to the PC. Note, status codes 0 through 50 are reserved
            for internal library use and should NOT be used as part of this set or custom hardware module class design.
            If the class does not produce runtime errors, set to None.
        data_codes: A set that stores the numpy uint8 (byte) codes used by the interface module to communicate states
            and data that needs additional processing. All incoming messages from the module are automatically logged to
            disk during communication runtime. Messages with event-codes from this set would also be passed to the
            process_received_data() method for additional processing. If the class does not require additional
            processing for any incoming data, set to None.
        mqtt_command_topics: A set of MQTT topics used by other MQTT clients to send commands to the module accessible
            through this interface instance. If the interface does not receive commands from mqtt, set this to None. The
            MicroControllerInterface set will use the set to initialize the MQTTCommunication class instance to
            monitor the requested topics and will use the use parse_mqtt_command() method to convert MQTT messages to
            module-addressed command structures.

    Attributes:
        _module_type: Stores the type (family) of the interfaced module.
        _module_id: Stores the specific module instance ID within the broader type (family).
        _type_id: Stores the type and id combined into a single uint16 value. This value should be unique for all
            possible type-id pairs and is used to ensure that each used module instance has a unique ID-type
            combination.
        _data_codes: Stores all event-codes that require additional processing.
        _mqtt_command_topics: Stores MQTT topics to monitor for incoming commands.
        _error_codes: Stores all expected error-codes as a set.
        _mqtt_communication: Determines whether this interface needs to communicate with MQTT.

    Raises:
        TypeError: If input arguments are not of the expected type.
    """

    _module_type: Incomplete
    _module_id: Incomplete
    _type_id: Incomplete
    _mqtt_command_topics: Incomplete
    _data_codes: Incomplete
    _error_codes: Incomplete
    _mqtt_communication: Incomplete
    def __init__(
        self,
        module_type: np.uint8,
        module_id: np.uint8,
        mqtt_communication: bool,
        error_codes: set[np.uint8] | None = None,
        data_codes: set[np.uint8] | None = None,
        mqtt_command_topics: set[str] | None = None,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns the string representation of the ModuleInterface instance."""
    @abstractmethod
    def parse_mqtt_command(
        self, topic: str, payload: bytes | bytearray
    ) -> OneOffModuleCommand | RepeatedModuleCommand | DequeueModuleCommand | None:
        """Packages and returns a ModuleCommand message to send to the microcontroller, based on the input MQTT
        command message topic and payload.

        This method is called by the MicroControllerInterface when other MQTT clients send command messages to one of
        the topics monitored by this ModuleInterface instance. This method resolves, packages, and returns the
        appropriate ModuleCommand message structure, based on the input message topic and payload.

        Notes:
            This method is called only if 'mqtt_command_topics' class argument was used to set the monitored topics
            during class initialization. This method will never receive a message with a topic that is not inside the
            'mqtt_command_topics' set.

            See the /examples folder included with the library for examples on how to implement this method.

        Args:
            topic: The MQTT topic to which the other MQTT client sent the module-addressed command.
            payload: The payload of the message.

        Returns:
            A OneOffModuleCommand or RepeatedModuleCommand instance that stores the message to be sent to the
            microcontroller. None, if the class instance is not configured to receive commands from MQTT.
        """
    @abstractmethod
    def process_received_data(
        self, message: ModuleData | ModuleState, mqtt_communication: MQTTCommunication, mp_queue: MPQueue
    ) -> None:
        """Processes the input message data and, if necessary, sends it to other MQTT clients and / or other Python
        processes.

        This method is called by the MicroControllerInterface when the ModuleInterface instance receives a message from
        the microcontroller that uses an event code provided at class initialization as 'data_codes' argument. This
        method processes the received message and uses the input MQTTCommunication instance or multiprocessing Queue
        instance to transmit the data to other Ataraxis systems or processes.

        Notes:
            To send the data to MQTT, call the send_data() method of the MQTTCommunication class. To send the data to
            other processes, call the put() method of the multiprocessing Queue object to pipe the data to other
            processes.

            This method is called only if 'data_codes' class argument was used to specify the event codes of messages
            that require further processing other than logging, which is done by default for all messages. This method
            will never receive a message with an event code that is not inside the 'data_codes' set.

            See the /examples folder included with the library for examples on how to implement this method.

        Args:
            message: The ModuleState or ModuleData object that stores the message received from the module instance
                running on the microcontroller.
            mqtt_communication: A fully configured instance of the MQTTCommunication class to use for sending the
                data to MQTT.
            mp_queue: An instance of the multiprocessing Queue class that allows piping data to parallel processes.
        """
    def extract_logged_data(self, log_path: Path) -> dict[Any, list[dict[str, np.uint64 | Any]]]:
        """Extracts the data received from the hardware module instance running on the microcontroller from the .npz
        log file generated during ModuleInterface runtime.

        This method reads the compressed '.npz' archives generated by the MicroControllerInterface class that works
        with this ModuleInterface during runtime and extracts all custom event-codes and data objects transmitted by
        the interfaced module instance from the microcontroller.

        Notes:
            The extracted data will NOT contain library-reserved events and messages. This includes all Kernel messages
            and module messages with event codes 0 through 50.

            This method should be used as a convenience abstraction for the inner workings of the DataLogger class.
            For each ModuleInterface, it will decode and return the logged runtime data sent to the PC by the specific
            hardware module instance controlled by the interface. You need to manually implement further data
            processing steps as necessary for your specific use case and module implementation.

        Args:
            log_path: The path to the compressed .npz file generated by the MicroControllerInterface that managed this
                ModuleInterface during runtime. Note, this has to be the compressed .npz archive, generated by
                DataLogger's compress_logs() method. The intermediate step of non-compressed '.npy 'files will not work.

        Returns:
            A dictionary that uses numpy uint8 event codes as keys and stores lists of dictionaries under each key.
            Each inner dictionary contains 3 elements. First, an uint64 timestamp, representing the number of
            microseconds since the UTC epoch onset. Second, the data object, transmitted with the message
            (or None, for state-only events). Third, the uint8 code of the command that the module was executing when
            it sent the message to the PC.

        Raises:
            ValueError: If the input path is not valid or does not point to an existing .npz archive.
        """
    @property
    def dequeue_command(self) -> DequeueModuleCommand:
        """Returns the command that instructs the microcontroller to clear all queued commands for the specific module
        instance managed by this ModuleInterface.
        """
    @property
    def module_type(self) -> np.uint8:
        """Returns the id-code that describes the broad type (family) of Modules managed by this interface class."""
    @property
    def module_id(self) -> np.uint8:
        """Returns the code that identifies the specific Module instance managed by the Interface class instance."""
    @property
    def data_codes(self) -> set[np.uint8]:
        """Returns the set of message event-codes that are processed during runtime, in addition to logging them to
        disk.
        """
    @property
    def mqtt_command_topics(self) -> set[str]:
        """Returns the set of MQTT topics this instance monitors for incoming MQTT commands."""
    @property
    def type_id(self) -> np.uint16:
        """Returns the unique 16-bit unsigned integer value that results from combining the type-code and the id-code
        of the instance.
        """
    @property
    def error_codes(self) -> set[np.uint8]:
        """Returns the set of error event-codes used by the module instance."""
    @property
    def mqtt_communication(self) -> bool:
        """Returns True if the class instance is configured to communicate with MQTT during runtime."""

class MicroControllerInterface:
    """Interfaces with an Arduino or Teensy microcontroller running ataraxis-micro-controller library.

    This class contains the logic that sets up a remote daemon process with SerialCommunication, MQTTCommunication,
    and DataLogger bindings to facilitate bidirectional communication and data logging between the microcontroller and
    concurrently active local (same PC) and remote (network) processes. Additionally, it exposes methods that send
    runtime parameters and commands to the Kernel and Module classes running on the connected microcontroller.

    Notes:
        An instance of this class has to be instantiated for each microcontroller active at the same time. The
        communication will not be started until the start() method of the class instance is called.

        This class uses SharedMemoryArray to control the runtime of the remote process, which makes it impossible to
        have more than one instance of this class with the same controller_id at a time. Make sure the class instance
        is stopped (to free SharedMemory buffer) before attempting to initialize a new class instance.

    Args:
        controller_id: The unique identifier code of the managed microcontroller. This information is hardcoded via the
            ataraxis-micro-controller (AXMC) library running on the microcontroller, and this class ensures that the
            code used by the connected microcontroller matches this argument when the connection is established.
            Critically, this code is also used as the source_id for the data sent from this class to the DataLogger.
            Therefore, it is important for this code to be unique across ALL concurrently active Ataraxis data
            producers, such as: microcontrollers and video systems. Valid codes are values between 1 and 255.
        microcontroller_serial_buffer_size: The size, in bytes, of the microcontroller's serial interface (UART or USB)
            buffer. This size is used to calculate the maximum size of transmitted and received message payloads. This
            information is usually available from the microcontroller's vendor.
        microcontroller_usb_port: The serial USB port to which the microcontroller is connected. This information is
            used to set up the bidirectional serial communication with the controller. You can use
            list_available_ports() function from ataraxis-transport-layer-pc library to discover addressable USB ports
            to pass to this argument. The function is also accessible through the CLI command: 'axtl-ports'.
        data_logger: An initialized DataLogger instance used to log the data produced by this Interface
            instance. The DataLogger itself is NOT managed by this instance and will need to be activated separately.
            This instance only extracts the necessary information to pipe the data to the logger.
        module_interfaces: A tuple of classes that inherit from the ModuleInterface class that interface with specific
            hardware module instances managed by the connected microcontroller.
        baudrate: The baudrate at which the serial communication should be established. This argument is ignored
            for microcontrollers that use the USB communication protocol, such as most Teensy boards. The correct
            baudrate for microcontrollers using the UART communication protocol depends on the clock speed of the
            microcontroller's CPU and the supported UART revision. Setting this to an unsupported value for
            microcontrollers that use UART will result in communication errors.
        mqtt_broker_ip: The ip address of the MQTT broker used for MQTT communication. Typically, this would be a
            'virtual' ip-address of the locally running MQTT broker, but the class can carry out cross-machine
            communication if necessary. MQTT communication will only be initialized if any of the input modules
            requires this functionality.
        mqtt_broker_port: The TCP port of the MQTT broker used for MQTT communication. This is used in conjunction
            with the mqtt_broker_ip argument to connect to the MQTT broker.

    Raises:
        TypeError: If any of the input arguments are not of the expected type.

    Attributes:
        _controller_id: Stores the id byte-code of the managed microcontroller.
        _usb_port: Stores the USB port to which the controller is connected.
        _baudrate: Stores the baudrate to use for serial communication with the controller.
        _microcontroller_serial_buffer_size: Stores the microcontroller's serial buffer size, in bytes.
        _mqtt_ip: Stores the IP address of the MQTT broker used for MQTT communication.
        _mqtt_port: Stores the port number of the MQTT broker used for MQTT communication.
        _mp_manager: Stores the multiprocessing Manager used to initialize and manage input and output Queue
            objects.
        _input_queue: Stores the multiprocessing Queue used to input the data to be sent to the microcontroller into
            the communication process.
        _output_queue: Stores the multiprocessing Queue used to output the data received from the microcontroller to
            other processes.
        _terminator_array: Stores the SharedMemoryArray instance used to control the runtime of the remote
            communication process.
        _communication_process: Stores the (remote) Process instance that runs the communication cycle.
        _watchdog_thread: A thread used to monitor the runtime status of the remote communication process.
        _reset_command: Stores the pre-packaged Kernel-addressed command that resets the microcontroller's hardware
            and software.
        _disable_locks: Stores the pre-packaged Kernel parameters configuration that disables all pin locks. This
            allows writing to all microcontroller pins.
        _enable_locks: Stores the pre-packaged Kernel parameters configuration that enables all pin locks. This
            prevents every Module managed by the Kernel from writing to any of the microcontroller pins.
        _started: Tracks whether the communication process has been started. This is used to prevent calling
            the start() and stop() methods multiple times.
        _start_mqtt_client: Determines whether to connect to MQTT broker during the main runtime cycle.
    """

    _reset_command: Incomplete
    _disable_locks: Incomplete
    _enable_locks: Incomplete
    _started: bool
    _controller_id: Incomplete
    _usb_port: Incomplete
    _baudrate: Incomplete
    _microcontroller_serial_buffer_size: Incomplete
    _mqtt_ip: Incomplete
    _mqtt_port: Incomplete
    _modules: Incomplete
    _logger_queue: Incomplete
    _mp_manager: Incomplete
    _input_queue: Incomplete
    _output_queue: Incomplete
    _terminator_array: Incomplete
    _communication_process: Incomplete
    _watchdog_thread: Incomplete
    _start_mqtt_client: bool
    def __init__(
        self,
        controller_id: np.uint8,
        microcontroller_serial_buffer_size: int,
        microcontroller_usb_port: str,
        data_logger: DataLogger,
        module_interfaces: tuple[ModuleInterface, ...],
        baudrate: int = 115200,
        mqtt_broker_ip: str = "127.0.0.1",
        mqtt_broker_port: int = 1883,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
    def __del__(self) -> None:
        """Ensures that all class resources are properly released when the class instance is garbage-collected."""
    def reset_controller(self) -> None:
        """Resets the connected MicroController to use default hardware and software parameters."""
    def lock_controller(self) -> None:
        """Configures connected MicroController parameters to prevent all modules from writing to any output pin."""
    def unlock_controller(self) -> None:
        """Configures connected MicroController parameters to allow all modules to write to any output pin."""
    def send_message(
        self,
        message: ModuleParameters
        | OneOffModuleCommand
        | RepeatedModuleCommand
        | DequeueModuleCommand
        | KernelParameters
        | KernelCommand,
    ) -> None:
        """Sends the input message to the microcontroller managed by the Interface instance.

        This is the primary interface for communicating with the Microcontroller. It allows sending all valid outgoing
        message structures to the Microcontroller for further processing.

        Raises:
            TypeError: If the input message is not a valid outgoing message structure.
        """
    @property
    def output_queue(self) -> MPQueue:
        """Returns the multiprocessing queue used by the communication process to output received data to all other
        processes that may need this data.
        """
    def _watchdog(self) -> None:
        """This function is used by the watchdog thread to ensure the communication process is alive during runtime.

        This function will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """
    def start(self) -> None:
        """Initializes the communication with the target microcontroller and the MQTT broker.

        The MicroControllerInterface class will not be able to carry out any communications until this method is called.
        After this method finishes its runtime, a watchdog thread is used to monitor the status of the process until
        stop() method is called, notifying the user if the process terminates prematurely.

        Notes:
            If send_message() was called before calling start(), all queued messages will be transmitted in one step.
            Multiple commands addressed to the same module sent in this fashion will likely interfere with each-other.

            As part of this method runtime, the interface will verify the target microcontroller's configuration to
            ensure compatibility.

        Raises:
            RuntimeError: If the instance fails to initialize the communication runtime.
        """
    def stop(self) -> None:
        """Shuts down the communication process, frees all reserved resources, and discards any unprocessed data stored
        inside input and output queues.
        """
    @staticmethod
    def _runtime_cycle(
        controller_id: np.uint8,
        module_interfaces: tuple[ModuleInterface, ...],
        input_queue: MPQueue,
        output_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        usb_port: str,
        baudrate: int,
        microcontroller_buffer_size: int,
        mqtt_ip: str,
        mqtt_port: int,
        start_mqtt_client: bool,
    ) -> None:
        """This method aggregates the communication runtime logic and is used as the target for the communication
        process.

        This method is designed to run in a remote Process. It encapsulates the steps for sending and receiving the
        data from the connected microcontroller. Primarily, the method routes the data between the microcontroller,
        the multiprocessing queues (inpout and output) managed by the Interface instance, and the MQTT
        broker. Additionally, it manages data logging by interfacing with the DataLogger class via the logger_queue.

        Args:
            controller_id: The byte-code identifier of the target microcontroller. This is used to ensure that the
                instance interfaces with the correct controller and to source-stamp logged data.
            module_interfaces: A tuple that stores ModuleInterface classes managed by this MicroControllerInterface
                instance.
            input_queue: The multiprocessing queue used to issue commands to the microcontroller.
            output_queue: The multiprocessing queue used to pipe received data to other processes.
            logger_queue: The queue exposed by the DataLogger class that is used to buffer and pipe received and
                outgoing messages to be logged (saved) to disk.
            terminator_array: The shared memory array used to control the communication process runtime.
            usb_port: The serial port to which the target microcontroller is connected.
            baudrate: The communication baudrate to use. This option is ignored for controllers that use USB interface,
                 but is essential for controllers that use the UART interface.
            microcontroller_buffer_size: The size of the microcontroller's serial buffer. This is used to determine
                the maximum size of the incoming and outgoing message payloads.
            mqtt_ip: The IP-address of the MQTT broker to use for communication with other MQTT processes.
            mqtt_port: The port number of the MQTT broker to use for communication with other MQTT processes.
            start_mqtt_client: Determines whether to start the MQTT client used by MQTTCommunication instance.
        """
    def vacate_shared_memory_buffer(self) -> None:
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed if the class is used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer. In turn, this would prevent the
        class remote Process from being started again. This method allows manually removing that buffer to reset the
        system. The method is designed to do nothing if the buffer with the same name as the microcontroller does not
        exist.
        """
