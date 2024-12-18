from abc import abstractmethod
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray
from ataraxis_data_structures import DataLogger, SharedMemoryArray

from .communication import (
    ModuleData as ModuleData,
    ModuleState as ModuleState,
    KernelCommand as KernelCommand,
    Identification as Identification,
    KernelParameters as KernelParameters,
    ModuleParameters as ModuleParameters,
    UnityCommunication as UnityCommunication,
    OneOffModuleCommand as OneOffModuleCommand,
    SerialCommunication as SerialCommunication,
    DequeueModuleCommand as DequeueModuleCommand,
    RepeatedModuleCommand as RepeatedModuleCommand,
)

class ModuleInterface:
    """The base class from which all custom ModuleInterface classes should inherit.

    Inheriting from this class grants all subclasses the static API that the MicroControllerInterface class uses to
    interface with specific modules. In addition to the inherited API, each ModuleInterface subclass should encapsulate
    module-specific parameters and data handling methods unique for each module family. See Notes for more details.

    Notes:
        Due to a high degree of custom module variability, it is currently not possible to provide a 'one-fits-all'
        module Interface that is also highly efficient for real time communication. Therefore, similar to
        ataraxis-micro-controller (AXMC) library, the interface for each custom module has to be implemented separately
        on a need-base method. The (base) class exposes the static API that MicroControllerInterface class can use to
        integrate each custom interface implementation with the general communication runtime cycle. To make this
        integration possible, this class defines some abstract (pure virtual) methods that developers have to implement
        for their interfaces. Follow the implementation guidelines in the docstrings of each abstract method and check
        the default modules included with the library distribution for guidance.

        When inheriting from this class, remember to call the parent's init method in the child class init method by
        using 'super().__init__()'! If this is not done, the MicroControllerInterface class will likely not be able to
        properly interact with your ModuleInterface!

        All data received from or sent to the microcontroller is automatically logged as byte-serialized numpy arrays.
        Therefore, if you do not need any additional processing steps, such as sending or receiving data from Unity,
        do not enable any custom processing flags. You will, however, have to implement all abstract methods, even if
        the class instance does not use them due to its flag-configuration.

    Args:
        module_type: The id-code that describes the broad type (family) of Modules managed by this interface class. This
            value has to match the code used by the module implementation on the microcontroller. Valid byte-codes range
            from 1 to 255.
        type_name: The human-readable name for the type (family) of Modules managed by this interface class, e.g.:
            'Rotary_Encoder'. This name is used in messages and some log files to help human operators in identifying
            the module family.
        module_id: The code that identifies the specific Module instance managed by the Interface class instance. This
            is used to identify unique instances of the same module family, such as different rotary encoders if more
            than one is used at the same time. Valid byte-codes range from 1 to 255.
        instance_name: The human-readable name for the specific Module instance managed by the Interface class instance,
            e.g.: 'Left_Corner_Touch_Sensor'. This name is used in messages and some log files to help human operators
            in identifying the module instance.
        unity_input_topics: A list of MQTT topics used by Unity to send commands to the module accessible through this
            Interface instance. If the module should not receive commands from Unity, set to None. This list will be
            used to initialize the UnityCommunication class instance to monitor the requested topics. If the module
            supports receiving commands from Unity, use get_from_unity() method to implement the logic for accessing
            and handling the incoming commands.
        output_data: Determines whether the module accessible through this Interface instance sends data to Unity or
            other processes. If the module is designed to output data in addition to logging it, use send_data()
            method to implement the logic for pre-processing and sending the data.

    Attributes:
        _module_type: Stores the type (family) of the interfaced module.
        _type_name: Stores the human-readable name of the module type (family).
        _module_id: Stores the specific module instance ID within the broader type (family).
        _instance_name: Stores the human-readable name of the specific module instance.
        _type_id: Stores the type and id combined into a single uint16 value. This value should be unique for all
            possible type-id pairs and is used to ensure that each used module instance has a unique ID-type
            combination.
        _output_data: Determines whether the instance outputs data to Unity or other processes.
        _unity_input: Determines whether to receive commands from Unity.
        _unity_input_topics: Stores the list of Unity topics to monitor for incoming commands.

    Raises:
        TypeError: If input arguments are not of the expected type.
    """

    _module_type: Incomplete
    _type_name: Incomplete
    _module_id: Incomplete
    _instance_name: Incomplete
    _type_id: Incomplete
    _unity_input_topics: Incomplete
    _unity_input: Incomplete
    _output_data: Incomplete
    def __init__(
        self,
        module_type: np.uint8,
        type_name: str,
        module_id: np.uint8,
        instance_name: str,
        unity_input_topics: tuple[str, ...] | None,
        *,
        output_data: bool = False,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns the string representation of the ModuleInterface instance."""
    @abstractmethod
    def get_from_unity(
        self, topic: str, payload: bytes | bytearray
    ) -> OneOffModuleCommand | RepeatedModuleCommand | None:
        """Packages and returns a ModuleCommand message to send to the microcontroller, based on the input Unity
        message topic and payload.

        This method is called by the MicroControllerInterface when Unity sends a message to one of the topics monitored
        by this ModuleInterface instance. This method then resolves, packages, and returns the appropriate ModuleCommand
        message structure, based on the input message topic and payload.

        Notes:
            This method is called only if 'unity_input_topics' argument was used to set the monitored topics during
            class initialization.

        Args:
            topic: The MQTT topic to which Unity sent the module-addressed message.
            payload: The payload of the message.

        Returns:
            A OneOffModuleCommand or RepeatedModuleCommand instance that stores the message to be sent to the
            microcontroller. None is a fallback return value used by ModuleInterface instances that are not
            configured to receive data from Unity.
        """
    @abstractmethod
    def send_data(
        self, message: ModuleData | ModuleState, unity_communication: UnityCommunication, mp_queue: MPQueue
    ) -> None:
        """Pre-processes the input message data and, if necessary, sends it to Unity and / or other processes.

        This method is called by the MicroControllerInterface when the ModuleInterface instance receives a message from
        the microcontroller and is configured to output data to Unity or other processes. This method pre-processes the
        received message and uses the input UnityCommunication instance or multiprocessing Queue instance to transmit
        the data.

        Notes:
            To send the data to Unity, call the send_data() method of the UnityCommunication class. Do not call any
            other methods as part of this method runtime, unless you know what you are doing. To send the data to other
            processes, call the put() method of the multiprocessing Queue object to pipe the data to other processes.

            This method is called only if 'output_data' flag was enabled during class initialization.

        Args:
            message: The ModuleState or ModuleData object that stores the message received from the microcontroller.
                This message always originates from the module with the same instance ID and type-code as used by the
                ModuleInterface instance.
            unity_communication: A fully configured instance of the UnityCommunication class to use for sending the
                data to Unity.
            mp_queue: An instance of the multiprocessing Queue class that allows piping data to parallel processes.
        """
    @abstractmethod
    def log_variables(self) -> NDArray[np.uint8] | None:
        """Serializes module-specific variable data into a byte numpy array.

        This method is called by the MicroControllerInterface during initialization for each ModuleInterface instance
        it manages. The array returned from this method is bundled with metadata that allows identifying the source of
        the data, and it is then sent to the DataLogger instance that saves this data to disk.

        Notes:
            This method is used to save instance-specific runtime data, such as conversion factors used to translate
            microcontroller-received data into a format expected by Unity. Use this method to save any information that
            may be helpful for post-processing or analyzing the rest of the data logged during runtime. The only
            requirement is that all data is serialized into a byte numpy array.

            If the instance does not need this functionality, implement this method with an empty return statement so
            that it returns Node.

        Returns:
            A shallow NumPy array that uses uint8 (byte) datatype and stores the serialized data to send to the logger.
            Note, the instance and type IDs of the module and the ID of the microcontroller will be combined and
            pre-pended to the data before it is sent to the logger. None, if the instance does not need to log any
            variable data.
        """
    def dequeue_command(self) -> DequeueModuleCommand:
        """Returns the command that instructs the microcontroller to clear all queued commands for the specific module
        instance managed by this ModuleInterface.
        """
    @property
    def module_type(self) -> np.uint8:
        """Returns the id-code that describes the broad type (family) of Modules managed by this interface class."""
    @property
    def type_name(self) -> str:
        """Returns the human-readable name for the type (family) of Modules managed by this interface class."""
    @property
    def module_id(self) -> np.uint8:
        """Returns the code that identifies the specific Module instance managed by the Interface class instance."""
    @property
    def instance_name(self) -> str:
        """Returns the human-readable name for the specific Module instance managed by the Interface class instance."""
    @property
    def output_data(self) -> bool:
        """Returns True if the class is configured to send the data received from the module instance to Unity or
        other processes.
        """
    @property
    def unity_input(self) -> bool:
        """Returns True if the class is configured to receive commands from Unity and send them to module instance."""
    @property
    def unity_input_topics(self) -> tuple[str, ...]:
        """Returns the tuple of MQTT topics this instance monitors for incoming Unity commands."""
    @property
    def type_id(self) -> np.uint16:
        """Returns the unique 16-bit unsigned integer value that results from combining the type-code and the id-code
        of the instance.
        """

class MicroControllerInterface:
    """Facilitates bidirectional communication between an Arduino or Teensy microcontroller, Python processes, and Unity
    game engine.

    This class contains the logic that sets up a remote daemon process with SerialCommunication, UnityCommunication,
    and DataLogger bindings to facilitate bidirectional communication and data logging between Unity, Python, and the
    microcontroller. Additionally, it exposes methods that send runtime parameters and commands to the Kernel and
    Module classes running on the connected microcontroller.

    Notes:
        An instance of this class has to be instantiated for each microcontroller active at the same time. The
        communication will not be started until the start() method of the class instance is called.

        This class uses SharedMemoryArray to control the runtime of the remote process, which makes it impossible to
        have more than one instance of this class with the same controller_name at a time. Make sure the class instance
        is stopped (to free SharedMemory buffer) before attempting to initialize a new class instance.

        During it's initialization, the class generates two unique log entry types. First, it builds and logs the map
        of all ID codes (controller, module-type, module-instance) to their human-readable names that follows the
        controller-module_type-module_instance hierarchy. Other Ataraxis libraries use this data to deserialize the
        logs into a human-readable dataset format. Second, for each managed ModuleInterface instance, the class calls
        its log_variables() method and logs the returned data if it is not None. These types of logs can be identified
        based on their unique timestamp values: 18446744073709551615 for variables and 18446744073709551614 for
        controller maps. These values are not meaningful as timestamps, instead they function as special identifiers.

    Args:
        controller_id: The unique identifier code of the managed microcontroller. This information is hardcoded via the
            AtaraxisMicroController (AXMC) firmware running on the microcontroller, and this class ensures that the code
            used by the connected microcontroller matches this argument when the connection is established. Critically,
            this code is also used as the source_id for the data sent from this class to the DataLogger. Therefore, it
            is important for this code to be unique across ALL concurrently active Ataraxis data producers, such as:
            microcontrollers, video systems, and Unity game engine instances. Valid codes are values between 1 and 255.
        controller_name: The human-readable name of the connected microcontroller. This information is used to better
            identify the microcontroller to human operators in error messages and log files.
        controller_usb_port: The serial USB port to which the microcontroller is connected. This information is used to
            set up the bidirectional serial communication with the controller. You can use list_available_ports()
            function from this library to discover addressable USB ports to pass to this argument.
        data_logger: An initialized DataLogger instance that will be used to log the data produced by this Interface
            instance. The DataLogger itself is NOT managed by this instance and will need to be activated separately.
            This instance only extracts the necessary information to buffer the data to the logger.
        modules: A tuple of classes that inherit from the (base) ModuleInterface class. These classes will be used by
            the main runtime cycle to handle the incoming data from the modules running on the microcontroller.
        baudrate: The baudrate at which the serial communication should be established. This argument is ignored
            for microcontrollers that use the USB communication protocol, such as most Teensy boards. The correct
            baudrate for microcontrollers using the UART communication protocol depends on the clock speed of the
            microcontroller's CPU and the supported UART revision. Setting this to an unsupported value for
            microcontrollers that use UART will result in communication errors.
        maximum_transmitted_payload_size: The maximum size of the message payload that can be sent to the
            microcontroller as one message. This should match the microcontroller's serial reception buffer size, even
            if the actual transmitted payloads do not reach that size. If the size is not set right, you may run into
            communication errors.
        unity_broker_ip: The ip address of the MQTT broker used for Unity communication. Typically, this would be a
            'virtual' ip-address of the locally running MQTT broker, but the class can carry out cross-machine
            communication if necessary. Unity communication will only be initialized if any of the input modules
            requires this functionality.
        unity_broker_port: The TCP port of the MQTT broker used for Unity communication. This is used in conjunction
            with the unity_broker_ip argument to connect to the MQTT broker.
        verbose: Determines whether the communication cycle reports runtime progress, including the contents of all
            incoming and outgoing messages, by printing messages to the console. This option is used during debugging
            and should be disabled during production runtimes.

    Raises:
        TypeError: If any of the input arguments are not of the expected type.

    Attributes:
            _controller_id: Stores the id byte-code of the managed microcontroller.
            _controller_name: Stores the human-readable name of the managed microcontroller.
            _usb_port: Stores the USB port to which the controller is connected.
            _baudrate: Stores the baudrate to use for serial communication with the controller.
            _max_tx_payload_size: Stores the maximum size the transmitted (outgoing) message payload can reach to fit
                inside the reception buffer of the microcontroller's Serial interface.
            _unity_ip: Stores the IP address of the MQTT broker used for Unity communication.
            _unity_port: Stores the port number of the MQTT broker used for Unity communication.
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
            _identify_command: Stores the pre-packaged Kernel-addressed command that requests the microcontroller to
                send back its id-code.
            _disable_locks: Stores the pre-packaged Kernel parameters configuration that disables all pin locks. This
                allows writing to all microcontroller pins.
            _enable_locks: Stores the pre-packaged Kernel parameters configuration that enables all pin locks. This
                prevents every Module managed by the Kernel from writing to any of the microcontroller pins.
            _started: Tracks whether the communication process has been started. This is used to prevent calling
                the start() and stop() methods multiple times.
    """

    _reset_command: Incomplete
    _identify_command: Incomplete
    _disable_locks: Incomplete
    _enable_locks: Incomplete
    _controller_id: Incomplete
    _controller_name: Incomplete
    _usb_port: Incomplete
    _baudrate: Incomplete
    _max_tx_payload_size: Incomplete
    _unity_ip: Incomplete
    _unity_port: Incomplete
    _verbose: Incomplete
    _started: bool
    _modules: Incomplete
    _logger_queue: Incomplete
    _mp_manager: Incomplete
    _input_queue: Incomplete
    _output_queue: Incomplete
    _terminator_array: Incomplete
    _communication_process: Incomplete
    _watchdog_thread: Incomplete
    def __init__(
        self,
        controller_id: np.uint8,
        controller_name: str,
        controller_usb_port: str,
        data_logger: DataLogger,
        modules: tuple[ModuleInterface, ...],
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
        unity_broker_ip: str = "127.0.0.1",
        unity_broker_port: int = 1883,
        verbose: bool = False,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
    def __del__(self) -> None:
        """Ensures that all class resources are properly released when the class instance is garbage-collected."""
    def _serialize_code_name_pair(self, code: np.uint8, name: str) -> NDArray[np.uint8]:
        """Serializes a single id_code-name pair into [code][name_length][name_bytes] byte block.

        The class uses this method to build the controller_layout log entry, which maps all ID codes (controller,
        module type, module instance ID) to human-readable data. This data is later used when reading the logs to
        form human-readable datasets.
        """
    def identify_controller(self) -> None:
        """Prompts the connected MicroController to identify itself by returning its id code."""
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
        """Initializes the communication with the target microcontroller, Unity game engine, and other processes.

        The MicroControllerInterface class will not be able to carry out any communications until this method is called.
        After this method finishes its runtime, a watchdog thread is used to monitor the status of the process until
        stop() method is called, notifying the user if the process terminates prematurely.

        Notes:
            If send_message() was called before calling start(), all queued messages will be transmitted in one step.
            Multiple commands addressed to the same module sent in this fashion will likely interfere with each-other.

            As part of this method runtime, the interface emits an identification request and ensures that the
            connected microcontroller responds with the id_code that exactly matches the id code used during class
            initialization.

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
        modules: tuple[ModuleInterface, ...],
        input_queue: MPQueue,
        output_queue: MPQueue,
        logger_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        usb_port: str,
        baudrate: int,
        payload_size: int,
        unity_ip: str,
        unity_port: int,
        verbose: bool = False,
    ) -> None:
        """This function aggregates the communication runtime logic and is used as the target for the communication
        process.

        This method is designed to run in a remote Process. It encapsulates the steps for sending and receiving the
        data from the connected microcontroller. Primarily, the method routes the data between the microcontroller and
        the multiprocessing queues (inpout and output) managed by the Interface instance and Unity game engine
        (via the binding of an MQTT client). Additionally, it manages data logging by interfacing with the DataLogger
        class via the logger_queue.

        Args:
            controller_id: The byte-code identifier of the target microcontroller. This is used to ensure that the
                instance interfaces with the correct controller.
            modules: A tuple that stores ModuleInterface classes managed by this MicroControllerInterface instance.
            input_queue: The multiprocessing queue used to issue commands to the microcontroller.
            output_queue: The multiprocessing queue used to pipe received data to other processes.
            logger_queue: The queue exposed by the DataLogger class that is used to buffer and pipe received and
                outgoing messages to be logged (saved) to disk.
            terminator_array: The shared memory array used to control the communication process runtime.
            usb_port: The serial port to which the target microcontroller is connected.
            baudrate: The communication baudrate to use. This option is ignored for controllers that use USB interface,
                 but is essential for controllers that use the UART interface.
            payload_size: The maximum size of the payload the managed microcontroller can receive. This is used to
                ensure all outgoing messages fit inside the Serial reception buffer of the microcontroller.
            unity_ip: The IP-address of the MQTT broker to use for communication with Unity game engine.
            unity_port: The port number of the MQTT broker to use for communication with Unity game engine.
            verbose: A flag that determines whether the contents of the incoming and outgoing messages should be
                printed to console. This is only used during debugging and should be disabled during most runtimes.
        """
    def vacate_shared_memory_buffer(self) -> None:
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed if the class is used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer. In turn, this would prevent the
        class remote Process from being started again. This method allows manually removing that buffer to reset the
        system. The method is designed to do nothing if the buffer with the same name as the microcontroller does not
        exist.
        """
