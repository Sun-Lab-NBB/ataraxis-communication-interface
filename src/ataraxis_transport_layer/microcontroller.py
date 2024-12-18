"""This module provides the ModuleInterface and MicroControllerInterface classes that aggregate the methods to
bidirectionally transmit data between PC, microcontrollers, and Unity game engine.

Each microcontroller module that manages physical hardware should be matched to a specialized interface derived from
the base ModuleInterface class. Similarly, for each concurrently active microcontroller, there has to be a specific
MicroControllerInterface instance that manages the ModuleInterface instances for the modules of that controller.
"""

from abc import abstractmethod
import sys
from threading import Thread
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, LogPackage, SharedMemoryArray

from .communication import (
    ModuleData,
    ModuleState,
    KernelCommand,
    Identification,
    KernelParameters,
    ModuleParameters,
    UnityCommunication,
    OneOffModuleCommand,
    SerialCommunication,
    DequeueModuleCommand,
    RepeatedModuleCommand,
)


class ModuleInterface:  # pragma: no cover
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

    def __init__(
        self,
        module_type: np.uint8,
        type_name: str,
        module_id: np.uint8,
        instance_name: str,
        unity_input_topics: tuple[str, ...] | None,
        *,
        output_data: bool = False,
    ) -> None:
        # Ensures that input byte-codes use valid value ranges
        if not isinstance(type_name, str):
            message = (
                f"Unable to initialize the ModuleInterface instance. Expected an string for 'type_name' argument, but "
                f"encountered {type_name} of type {type(type_name).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(instance_name, str):
            message = (
                f"Unable to initialize the ModuleInterface instance. Expected an string for 'instance_name' argument, "
                f"but encountered {instance_name} of type {type(instance_name).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(module_type, np.uint8) or not 1 <= module_type <= 255:
            message = (
                f"Unable to initialize the ModuleInterface instance for module {instance_name} of type {type_name}. "
                f"Expected an unsigned integer value between 1 and 255 for 'module_type' argument, but encountered "
                f"{module_type} of type {type(module_type).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(module_id, np.uint8) or not 1 <= module_id <= 255:
            message = (
                f"Unable to initialize the ModuleInterface instance for module {instance_name} of type {type_name}. "
                f"Expected an unsigned integer value between 1 and 255 for 'module_id' argument, but encountered "
                f"{module_id} of type {type(module_id).__name__}."
            )
            console.error(message=message, error=TypeError)
        if unity_input_topics is not None and not all(isinstance(topic, str) for topic in unity_input_topics):
            message = (
                f"Unable to initialize the ModuleInterface instance for module {instance_name} of type {type_name}. "
                f"Expected a tuple of strings or None for 'unity_input_topics' argument, but encountered "
                f"{unity_input_topics} of type {type(unity_input_topics).__name__} and / or at least one non-string "
                f"item."
            )
            console.error(message=message, error=TypeError)

        # Saves type and ID data into class attributes
        self._module_type: np.uint8 = module_type
        self._type_name: str = str(type_name)
        self._module_id: np.uint8 = module_id
        self._instance_name: str = str(instance_name)

        # Combines type and ID codes into a 16-bit value. This is used to ensure every module instance has a unique
        # ID + Type combination. This method is position-aware, so inverse type-id pairs will be coded as different
        # values e.g.: 4-5 != 5-4
        self._type_id: np.uint16 = np.uint16(
            (self._module_type.astype(np.uint16) << 8) | self._module_id.astype(np.uint16)
        )

        # Additional processing flags. Unity input is set based on whether there are input topics, other flags are
        # boolean and obtained from input arguments.
        self._unity_input_topics: tuple[str, ...] = unity_input_topics if unity_input_topics is not None else tuple()
        self._unity_input: bool = True if len(self._unity_input_topics) > 0 else False
        self._output_data: bool = output_data if isinstance(output_data, bool) else False

    def __repr__(self) -> str:
        """Returns the string representation of the ModuleInterface instance."""
        message = (
            f"ModuleInterface(type_code={self._module_type}, type_name={self._type_name}, "
            f"instance_code={self._module_id}, instance_name={self._instance_name}, output_data={self._output_data}, "
            f"unity_input_topics={self._unity_input_topics})"
        )
        return message

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
        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"get_from_unity() method for {self._type_name} module interface must be implemented when subclassing the "
            f"base ModuleInterface class."
        )

    @abstractmethod
    def send_data(
        self,
        message: ModuleData | ModuleState,
        unity_communication: UnityCommunication,
        mp_queue: MPQueue,  # type: ignore
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
        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"output_data() method for {self._type_name} module interface must be implemented when subclassing the "
            f"base ModuleInterface class."
        )

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
        raise NotImplementedError(
            f"log_variables() method for {self._type_name} module interface must be implemented when "
            f"subclassing the base ModuleInterface class."
        )

    def dequeue_command(self) -> DequeueModuleCommand:
        """Returns the command that instructs the microcontroller to clear all queued commands for the specific module
        instance managed by this ModuleInterface.
        """
        return DequeueModuleCommand(module_type=self._module_type, module_id=self._module_id, return_code=np.uint8(0))

    @property
    def module_type(self) -> np.uint8:
        """Returns the id-code that describes the broad type (family) of Modules managed by this interface class."""
        return self._module_type

    @property
    def type_name(self) -> str:
        """Returns the human-readable name for the type (family) of Modules managed by this interface class."""
        return self._type_name

    @property
    def module_id(self) -> np.uint8:
        """Returns the code that identifies the specific Module instance managed by the Interface class instance."""
        return self._module_id

    @property
    def instance_name(self) -> str:
        """Returns the human-readable name for the specific Module instance managed by the Interface class instance."""
        return self._instance_name

    @property
    def output_data(self) -> bool:
        """Returns True if the class is configured to send the data received from the module instance to Unity or
        other processes.
        """
        return self._output_data

    @property
    def unity_input(self) -> bool:
        """Returns True if the class is configured to receive commands from Unity and send them to module instance."""
        return self._unity_input

    @property
    def unity_input_topics(self) -> tuple[str, ...]:
        """Returns the tuple of MQTT topics this instance monitors for incoming Unity commands."""
        return self._unity_input_topics

    @property
    def type_id(self) -> np.uint16:
        """Returns the unique 16-bit unsigned integer value that results from combining the type-code and the id-code
        of the instance.
        """
        return self._type_id


class MicroControllerInterface:  # pragma: no cover
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

    # Pre-packages Kernel commands into attributes. Since Kernel commands are known and fixed at compilation,
    # they only need to be defined once.
    _reset_command = KernelCommand(
        command=np.uint8(2),
        return_code=np.uint8(0),
    )
    _identify_command = KernelCommand(
        command=np.uint8(3),
        return_code=np.uint8(0),
    )

    # Also pre-packages the two most used parameter configurations (all-locked and all-unlocked). The class can
    # also send messages with partial locks (e.g.: TTl ON, Action OFF), but those are usually not used outside
    # specific debugging and testing scenarios.
    _disable_locks = KernelParameters(
        action_lock=np.bool(False),
        ttl_lock=np.bool(False),
        return_code=np.uint8(0),
    )
    _enable_locks = KernelParameters(
        action_lock=np.bool(True),
        ttl_lock=np.bool(True),
        return_code=np.uint8(0),
    )

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
    ):
        # Ensures that input arguments have valid types. Only checks the arguments that are not passed to other classes,
        # such as TransportLayer, which has its own argument validation.
        if not isinstance(controller_id, np.uint8) or not 1 <= controller_id <= 255:
            message = (
                f"Unable to initialize the MicroControllerInterface instance. Expected an unsigned integer value "
                f"between 1 and 255 for 'controller_id' argument, but encountered {controller_id} of type "
                f"{type(controller_id).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(controller_name, str):
            message = (
                f"Unable to initialize the MicroControllerInterface instance. Expected a string for 'controller_name' "
                f"argument, but encountered {controller_name} of type {type(controller_name).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(modules, tuple) or not modules:
            message = (
                f"Unable to initialize the MicroControllerInterface instance for {controller_name} controller with id "
                f"{controller_id}. Expected a non-empty tuple of ModuleInterface instances for 'modules' argument, but "
                f"encountered {modules} of type {type(modules).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not all(isinstance(module, ModuleInterface) for module in modules):
            message = (
                f"Unable to initialize the MicroControllerInterface instance for {controller_name} controller with id "
                f"{controller_id}. All items in 'modules' tuple must be ModuleInterface instances."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(data_logger, DataLogger):
            message = (
                f"Unable to initialize the MicroControllerInterface instance for {controller_name} controller with id "
                f"{controller_id}. Expected an initialized DataLogger instance for 'data_logger' argument, but "
                f"encountered {data_logger} of type {type(data_logger).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Controller (kernel) ID information. Follows the same code-name-description format as module type and instance
        # values do.
        self._controller_id: np.uint8 = controller_id
        self._controller_name: str = controller_name

        # SerialCommunication parameters. This is used to initialize the communication in the remote process.
        self._usb_port: str = controller_usb_port
        self._baudrate: int = baudrate
        self._max_tx_payload_size: int = maximum_transmitted_payload_size

        # UnityCommunication parameters. This is used to initialize the unity communication from the remote process
        # if the managed modules need this functionality.
        self._unity_ip: str = unity_broker_ip
        self._unity_port: int = unity_broker_port

        # Verbose flag and started trackers
        self._verbose: bool = verbose if isinstance(verbose, bool) else False
        self._started: bool = False

        # Managed modules and data logger queue. Modules will be pre-processes as part of this initialization runtime.
        # Logger queue is fed directly into the SerialCommunication, which automatically logs all incoming and outgoing
        # data to disk.
        self._modules: tuple[ModuleInterface, ...] = modules

        # Extracts the queue from the logger instance. Other than for this step, this class does not use the instance
        # for anything else.
        self._logger_queue: MPQueue = DataLogger.input_queue  # type: ignore

        # Sets up the assets used to deploy the communication runtime on a separate core and bidirectionally transfer
        # data between the communication process and the main process managing the overall runtime.
        self._mp_manager: SyncManager = Manager()
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._output_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._terminator_array: None | SharedMemoryArray = None
        self._communication_process: None | Process = None
        self._watchdog_thread: None | Thread = None

        # Verifies that all input ModuleInterface instances have a unique type+id combination and pre-processes their
        # data. Specifically, logs their variable data (for module instances that support this process). Also, logs the
        # layout of the controller that maps controller, module type, module instance ID codes to meaningful names and
        # records their hierarchy. These log entries are then used when parsing the data logged by the controller
        # into a human-readable dataset.

        # Serializes the controller_id-name pair and uses it to pre-create a temporary list that stores serialized
        # code-name pairs. This is used to construct the controller layout log (see below).
        code_name_blocks = [self._serialize_code_name_pair(self._controller_id, self._controller_name)]

        # Stores the number of types as first byte after controller id-name data. This is essential to know how
        # many types to decode from the generated log entry.
        n_types = len(set(module.module_type for module in self._modules))
        code_name_blocks.append(np.array([n_types], dtype=np.uint8))

        processed_type_ids: set[np.uint16] = set()  # This is used to ensure each instance has a unique type+id pair.

        # This dictionary is used below to serialize instance_id-name pairs for each ModuleInterface instance and store
        # them under their type-codes. This preserves the hierarchy of which module instances belong to which
        # module families (types).
        processed_type_codes: dict[np.uint8, tuple[str, list[NDArray[np.uint8]]]] = {}

        # Loops over all module instances and processes their data
        for module in self._modules:
            # Extracts type and id codes of the module
            module_id = module.module_id
            module_type = module.module_type

            # If the module's combined type + id code is already inside the processed_types_id set, this means another
            # module with the same exact type and ID combination has already been processed.
            if module.type_id in processed_type_ids:
                message = (
                    f"Unable to initialize the MicroControllerInterface instance for {controller_name} controller with "
                    f"id {controller_id}. Encountered two ModuleInterface instances with the same type-code "
                    f"({module_type}) and id-code ({module_id}), which is not allowed. Make sure each type and id "
                    f"combination is only used by a single ModuleInterface class instance."
                )
                console.error(message=message, error=ValueError)

            # Adds each processed type+id code to the tracker set
            processed_type_ids.add(module.type_id)

            # For each unique module type, adds a new entry into the tracker dictionary. That entry contains the
            # human-readable name for the type and a list used to aggregate the code-name serialized data for all
            # instances of this module type.
            if module_type not in processed_type_codes:
                processed_type_codes[module_type] = (module.type_name, [])

            # Adds each module instance id-name serialization to its type's list inside the tracker dictionary
            processed_type_codes[module_type][1].append(self._serialize_code_name_pair(module_id, module.instance_name))

            # For each module instance, calls the method that returns the serialized variable data and logs it by
            # sending the data to the DataLogger
            variable_data = module.log_variables()

            # Appends type and id codes to the variable data package
            header = np.array([module_type, module_id], dtype=np.uint8)
            data_package = np.concatenate((header, variable_data), dtype=np.uint8)  # type: ignore

            # Uses controller_id as source ID, which is also done for all other log packages. The critical part here
            # is the timestamp set to the maximum possible value that still fits into numpy uint64 value.
            # Since during normal runtimes the timestamp is given as the time, in microseconds, relative to the
            # onset stamp log, this number would translate to ~584 years, which is not a reasonable number. Instead,
            # this timestamp value is co-opted to indicate that the log entry contains module instance variable
            # data.
            package = LogPackage(
                source_id=int(controller_id),
                time_stamp=np.iinfo(np.uint64).max,
                serialized_data=data_package,
            )

            # Logs variable data for each module instance by sending it to the logger queue.
            self._logger_queue.put(package)

        # After processing all modules, serializes type data in the order types were added to the storage dictionary
        # This forms type-blocks where the data for the type is followed by the data for all instances of that type.
        for type_code, (type_name, instance_blocks) in processed_type_codes.items():
            # Appends the serialized type_code-name pair to the beginning of each type block
            code_name_blocks.append(self._serialize_code_name_pair(type_code, type_name))
            # Then, appends the number of Interface instances that were found under that module_type code. This is
            # used in conjunction with the type-count recorded after adding controller information to decode the
            # serialized data.
            code_name_blocks.append(np.array([len(instance_blocks)], dtype=np.uint8))
            # Finally, appends all instance_id-name pairs after the instance count
            code_name_blocks.extend(instance_blocks)

        # Combines all blocks into the final bytes array
        serialized_code_names = np.concatenate(code_name_blocks)

        # Creates and send the resultant controller layout map (code-name map) to the logger queue. Similar to the
        # variable log package, uses a very large timestamp, which is exactly 1 value below the maximum number supported
        # by np.uint64
        code_name_package = LogPackage(
            source_id=int(self._controller_id),
            time_stamp=np.iinfo(np.uint64).max - 1,
            serialized_data=serialized_code_names,
        )
        self._logger_queue.put(code_name_package)

    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
        return (
            f"MicroControllerInterface(controller_id={self._controller_id}, controller_name={self._controller_name}, "
            f"usb_port={self._usb_port}, baudrate={self._baudrate}, unity_ip={self._unity_ip}, "
            f"unity_port={self._unity_port}, started={self._started})"
        )

    def __del__(self) -> None:
        """Ensures that all class resources are properly released when the class instance is garbage-collected."""
        self.stop()

    def _serialize_code_name_pair(self, code: np.uint8, name: str) -> NDArray[np.uint8]:
        """Serializes a single id_code-name pair into [code][name_length][name_bytes] byte block.

        The class uses this method to build the controller_layout log entry, which maps all ID codes (controller,
        module type, module instance ID) to human-readable data. This data is later used when reading the logs to
        form human-readable datasets.
        """
        # Converts string to bytes
        name_bytes = name.encode("utf-8")
        name_length = len(name_bytes)
        if name_length > 65535:  # uint16 max
            message = (
                f"Unable to serialize the input name {name} when constructing controller layout log entry for "
                f"{self._controller_name} MicroControllerInterface with id {self._controller_id}. Encoded name is "
                f"too long: {name_length} bytes. The maximum supported (encoded) name length is 65535 bytes."
            )
            console.error(message=message, error=ValueError)

        # Allocates the storage array: 1 byte code + 2 bytes length + n bytes name
        result = np.zeros(1 + 2 + name_length, dtype=np.uint8)

        # Stores byte-code
        result[0] = code

        # Stores name length as uint16
        result[1:3] = np.array([name_length & 0xFF, name_length >> 8], dtype=np.uint8)

        # Stores name bytes
        result[3:] = np.frombuffer(name_bytes, dtype=np.uint8)

        # Returns the serialized code-name pair
        return result

    def identify_controller(self) -> None:
        """Prompts the connected MicroController to identify itself by returning its id code."""
        self._input_queue.put(self._identify_command)

    def reset_controller(self) -> None:
        """Resets the connected MicroController to use default hardware and software parameters."""
        self._input_queue.put(self._reset_command)

    def lock_controller(self) -> None:
        """Configures connected MicroController parameters to prevent all modules from writing to any output pin."""
        self._input_queue.put(self._enable_locks)

    def unlock_controller(self) -> None:
        """Configures connected MicroController parameters to allow all modules to write to any output pin."""
        self._input_queue.put(self._disable_locks)

    def send_message(
        self,
        message: (
            ModuleParameters
            | OneOffModuleCommand
            | RepeatedModuleCommand
            | DequeueModuleCommand
            | KernelParameters
            | KernelCommand
        ),
    ) -> None:
        """Sends the input message to the microcontroller managed by the Interface instance.

        This is the primary interface for communicating with the Microcontroller. It allows sending all valid outgoing
        message structures to the Microcontroller for further processing.

        Raises:
            TypeError: If the input message is not a valid outgoing message structure.
        """
        # Verifies that the input message uses a valid type
        if not isinstance(
            message,
            (
                ModuleParameters,
                OneOffModuleCommand,
                RepeatedModuleCommand,
                DequeueModuleCommand,
                KernelParameters,
                KernelCommand,
            ),
        ):
            message = (
                f"Unable to send the message via the {self._controller_name} MicroControllerInterface with id "
                f"{self._controller_id}. Expected one of the valid outgoing message structures, but instead "
                f"encountered {message} of type {type(message).__name__}. Use one of the supported structures "
                f"available from the communication module."
            )
            console.error(message=message, error=TypeError)
        self._input_queue.put(message)

    @property
    def output_queue(self) -> MPQueue:  # type: ignore
        """Returns the multiprocessing queue used by the communication process to output received data to all other
        processes that may need this data.
        """
        return self._output_queue

    def _watchdog(self) -> None:
        """This function is used by the watchdog thread to ensure the communication process is alive during runtime.

        This function will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """
        timer = PrecisionTimer(precision="ms")

        # The watchdog function will run until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):  # type: ignore
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay_noblock(delay=20, allow_sleep=True)

            # Only monitors the Process state after the communication is initialized via the start() method.
            if not self._started:
                continue

            if self._communication_process is not None and not self._communication_process.is_alive():
                message = (
                    f"The communication process of the MicroControllerInterface {self._controller_name} with id "
                    f"{self._controller_id} has been prematurely shut down. This likely indicates that the process has "
                    f"encountered a runtime error that terminated the process."
                )
                console.error(message=message, error=RuntimeError)

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
        # If the process has already been started, returns without doing anything.
        if self._started:
            return

        # Instantiates the shared memory array used to control the runtime of the communication Process.
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._controller_name}_terminator_array",
            # Uses class name to ensure the array buffer name is unique
            prototype=np.zeros(shape=2, dtype=np.uint8),  # Index 0 = terminator, index 1 = initialization status
        )  # Instantiation automatically connects the main process to the array.

        # Sets up the communication process. This process continuously cycles through the communication loop until
        # terminated, enabling bidirectional communication with the controller.
        self._communication_process = Process(
            target=self._runtime_cycle,
            args=(
                self._controller_id,
                self._modules,
                self._input_queue,
                self._output_queue,
                self._logger_queue,
                self._terminator_array,
                self._usb_port,
                self._baudrate,
                self._max_tx_payload_size,
                self._unity_ip,
                self._unity_port,
                self._verbose,
            ),
            daemon=True,
        )

        # Creates the watchdog thread.
        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)

        # Initializes the communication process.
        self._communication_process.start()

        # Sends controller identification command and ensures the (connected) controller id matches expectation.
        self.identify_controller()

        start_timer = PrecisionTimer("s")
        start_timer.reset()
        # Blocks until the microcontroller has finished all initialization steps or encounters an initialization error.
        while self._terminator_array.read_data(1) != 1:
            # Generally, there are two ways initialization failure is detected. One is if the managed process
            # terminates, which would be the case if any subclass used in the communication process raises an exception.
            # Another way if the status tracker never reaches success code (1). This latter case would likely indicate
            # that there is a communication issue where the data does not reach the controller or the PC. The
            # initialization process should be VERY fast, likely on the order of hundreds of microseconds. Waiting for
            # 5 seconds is likely excessive.
            if not self._communication_process.is_alive() or start_timer.elapsed > 5:
                message = (
                    f"{self._controller_name} MicroControllerInterface with id {self._controller_id} has failed to "
                    f"initialize the communication with the microcontroller. If the class was initialized with "
                    f"the 'verbose' flag disabled, enable the flag and repeat the initialization to debug the "
                    f"initialization process error."
                )
                console.error(error=RuntimeError, message=message)

        # Starts the process watchdog thread once the initialization is complete
        self._watchdog_thread.start()

        # Sets the started flag
        self._started = True

    def stop(self) -> None:
        """Shuts down the communication process, frees all reserved resources, and discards any unprocessed data stored
        inside input and output queues.
        """
        # If the process has not been started, returns without doing anything.
        if not self._started:
            return

        # Resets the controller. This automatically prevents all modules from changing pin states (locks the controller)
        # and resets module and hardware states.
        self.reset_controller()

        # There is no need for additional delays as the communication loop will make sure the reset command is sent
        # to the controller before shutdown

        # Changes the started tracker value. Amongst other things this soft-inactivates the watchdog thread.
        self._started = False

        # Sets the terminator trigger to 1, which triggers communication process shutdown. This also shuts down the
        # watchdog thread.
        if self._terminator_array is not None:
            self._terminator_array.write_data(0, np.uint8(1))

        # Waits until the communication process terminates
        if self._communication_process is not None:
            self._communication_process.join()

        # Shuts down the multiprocessing manager. This collects all active queues and discards all unprocessed data.
        self._mp_manager.shutdown()

        # Waits for the watchdog thread to terminate.
        if self._watchdog_thread is not None:
            self._watchdog_thread.join()

        # Disconnects from the shared memory array and destroys its shared buffer.
        if self._terminator_array is not None:
            self._terminator_array.disconnect()
            self._terminator_array.destroy()

    @staticmethod
    def _runtime_cycle(
        controller_id: np.uint8,
        modules: tuple[ModuleInterface, ...],
        input_queue: MPQueue,  # type: ignore
        output_queue: MPQueue,  # type: ignore
        logger_queue: MPQueue,  # type: ignore
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
        # Connects to the terminator array. This is done early, as the terminator_array is used to track the
        # initialization and runtime status of the process.
        terminator_array.connect()

        # If the runtime is called in the verbose mode, ensures the console is enabled.
        was_enabled = console.enabled
        if verbose and not was_enabled:
            console.enable()

            # Also sends a message to notify that the initialization has started.
            console.echo(
                message=f"Starting MicroControllerInterface for controller with ID {controller_id}...",
                level=LogLevel.INFO,
            )

        # Precreates the assets used to optimize the communication runtime cycling. These assets are filled below to
        # support efficient interaction between the Communication classes and the ModuleInterface classes.
        unity_input_map: dict[str, list[int]] = {}
        output_map: dict[np.uint16, int] = {}

        for num, module in enumerate(modules):
            # If the module is configured to receive data from unity, configures the necessary data structures to enable
            # monitoring the necessary topics and allow passing the data received on that topic to the
            # appropriate module class for processing.
            if module.unity_input:
                for topic in module.unity_input_topics:
                    # Extends the list of module indices that listen for that particular topic. This allows addressing
                    # multiple modules at the same time, as long as they all listen to the same topic.
                    existing_modules = unity_input_map.get(topic, [])
                    unity_input_map[topic] = existing_modules + [num]

            # If the module is configured to output data to Unity or other processes, maps its type+id combined code
            # to its index number. This is used to quickly find the module interface instance addressed by incoming
            # data, so that they can then send the data to the appropriate output stream.
            if module.output_data:
                output_map[module.type_id] = num

        # Disables unused processing steps. For example, if none of the managed modules send data to Unity or other
        # processes, that processing step is disabled outright via a boolean 'if' check to speed up the runtime.
        unity_input = False
        data_output = False

        # Note, keys() essentially returns a set of keys, since the same hash-map optimizations are involved with
        # dictionary keys as with set values.
        if len(unity_input_map.keys()) != 0:
            unity_input = True
        if len(output_map.keys()) != 0:
            data_output = True

        # Initializes the serial communication class and connects to the target microcontroller.
        serial_communication = SerialCommunication(
            usb_port=usb_port,
            source_id=controller_id,
            logger_queue=logger_queue,
            baudrate=baudrate,
            maximum_transmitted_payload_size=payload_size,
            verbose=verbose,
        )

        # Initializes the unity_communication class and connects to the MQTT broker. If the interface does not
        # need Unity communication, this initialization will only statically reserve some RAM with no other
        # adverse effects. If the unity_input_map is empty, the class initialization method will correctly
        # interpret this as a case where no topics need to be monitored.
        unity_communication = UnityCommunication(
            ip=unity_ip, port=unity_port, monitored_topics=tuple(tuple(unity_input_map.keys()))
        )

        # Only connects to MQTT broker if managed modules need to send data to unity or receive data from Unity.
        if unity_input or data_output:
            unity_communication.connect()

        # This notifies the user that subclass initialization is complete. The interface still needs to verify that the
        # connected controller has the correct ID (see below).
        if verbose:
            console.echo(
                message=f"MicroControllerInterface {controller_id} initialization complete.", level=LogLevel.SUCCESS
            )

        try:
            # Initializes the main communication loop. This loop will run until the exit conditions are encountered.
            # The exit conditions for the loop require the first variable in the terminator_array to be set to True
            # and the main input queue of the interface to be empty. This ensures that all queued commands issued from
            # the central process are fully carried out before the communication is terminated.
            while not terminator_array.read_data(index=0, convert_output=True) or not input_queue.empty():
                # Main data sending loop. The method will sequentially retrieve the queued command and parameter data
                # and transmit them to the microcontroller.
                while not input_queue.empty():
                    out_data = input_queue.get()
                    serial_communication.send_message(out_data)  # Transmits the data to the microcontroller

                # Unity data sending loop. This loop will be O(1) if unity never has data. In turn, this will always be
                # the case if no module supports unity inputs. Therefore, there is no need to both have an 'if' and a
                # 'while' check here for optimal runtime speed.
                while unity_communication.has_data:
                    # If UnityCommunication has received data, loops over all interfaces that requested the data from
                    # this topic and calls their unity data processing method while passing it the topic and the
                    # received message payload.
                    topic, payload = unity_communication.get_data()  # type: ignore

                    # Each incoming message will be processed by each module subscribed to this topic. Since
                    # UnityCommunication is configured to only listen to topics submitted by the interface classes, the
                    # topic is guaranteed to be inside the unity_input_map dictionary and have at least one Module which
                    # can process its data.
                    for i in unity_input_map[topic]:
                        out_data = modules[i].get_from_unity(topic=topic, payload=payload)
                        if out_data is not None:
                            serial_communication.send_message(out_data)  # Transmits the data to the microcontroller

                # Attempts to receive the data from microcontroller
                in_data = serial_communication.receive_message()

                # If no data is available cycles the loop
                if in_data is None:
                    continue

                # Otherwise, resolves additional processing steps associated with incoming data. Currently, only Module
                # interfaces have additional data processing steps that are executed more than once during runtime. The
                # Kernel Identification message also has a unique processing step, but it should only be executed once,
                # during runtime initialization.
                if isinstance(in_data, (ModuleState, ModuleData)) and data_output:
                    # Computes the combined type and id code for the incoming data. This is used to find the specific
                    # ModuleInterface to which the message is addressed and, if necessary, invoke interface-specific
                    # additional processing method.
                    target_type_id: np.uint16 = np.uint16(
                        (in_data.module_type.astype(np.uint16) << 8) | in_data.module_id.astype(np.uint16)
                    )

                    # Depending on whether the combined code is inside the output_map, executes the target module's
                    # method to handle data output.
                    if target_type_id in output_map:
                        modules[output_map[target_type_id]].send_data(
                            message=in_data,
                            unity_communication=unity_communication,
                            mp_queue=output_queue,
                        )

                # Whenever the incoming message is the Identification message, ensures that the received controller_id
                # matches the ID expected by the class.
                elif isinstance(in_data, Identification):
                    if in_data.controller_id != controller_id:
                        # Raises the error.
                        message = (
                            f"Unexpected controller_id code received from the microcontroller managed by the "
                            f"MicroControllerInterface instance. Expected {controller_id}, but received "
                            f"{in_data.controller_id}."
                        )
                        console.error(message=message, error=ValueError)

                    else:
                        # Reports that communication class has been successfully initialized. Seeing this code means
                        # that the communication appears to be functioning correctly, at least in terms of the data
                        # reaching the Kernel and back. While this does not guarantee the runtime will continue running
                        # without errors, it is very likely to be so.
                        terminator_array.write_data(index=1, data=np.uint8(1))

        # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
        # before re-raising the exception to terminate the process.
        except Exception as e:
            sys.stderr.write(str(e))
            sys.stderr.flush()
            raise e

        # If this point is reached, the loop has received the shutdown command and successfully escaped the
        # communication cycle. Disconnects from the terminator array and shuts down Unity communication.
        terminator_array.disconnect()
        unity_communication.disconnect()

        # If this runtime had to enable the console to comply with 'verbose' flag, disables it before ending the
        # runtime.
        if verbose and not was_enabled:
            # Notifies the user that the runtime Process has been successfully terminated
            console.echo(
                message=f"MicroControllerInterface {controller_id} communication runtime terminated.",
                level=LogLevel.SUCCESS,
            )
            console.disable()

    def vacate_shared_memory_buffer(self) -> None:
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed if the class is used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer. In turn, this would prevent the
        class remote Process from being started again. This method allows manually removing that buffer to reset the
        system. The method is designed to do nothing if the buffer with the same name as the microcontroller does not
        exist.
        """
        try:
            buffer = SharedMemory(name=f"{self._controller_name}_terminator_array", create=False)
            buffer.close()
            buffer.unlink()
        except FileNotFoundError:
            pass
