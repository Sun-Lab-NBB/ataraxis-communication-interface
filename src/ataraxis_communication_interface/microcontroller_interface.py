"""This module provides the ModuleInterface and MicroControllerInterface classes. They aggregate the methods to
enable Python PC clients to bidirectionally interface with custom hardware modules managed by an Arduino or Teensy
microcontroller. Additionally, these classes contain the bindings to connect to the microcontrollers via the MQTT
protocol, facilitating indirect communication with local and remote processes over MQTT broker.

Each microcontroller hardware module that manages physical hardware should be matched to a specialized interface
derived from the base ModuleInterface class. Similarly, for each concurrently active microcontroller, there has to be a
specific MicroControllerInterface instance that manages the ModuleInterface instances for the modules of that
controller.
"""

from abc import abstractmethod
import sys
from typing import Any
from pathlib import Path
from threading import Thread
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from multiprocessing.managers import SyncManager

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from numpy.lib.npyio import NpzFile
from ataraxis_base_utilities import console
from ataraxis_data_structures import DataLogger, SharedMemoryArray

from .communication import (
    KernelData,
    ModuleData,
    KernelState,
    ModuleState,
    KernelCommand,
    SerialProtocols,
    KernelParameters,
    ModuleParameters,
    SerialPrototypes,
    MQTTCommunication,
    OneOffModuleCommand,
    SerialCommunication,
    DequeueModuleCommand,
    ModuleIdentification,
    RepeatedModuleCommand,
    ControllerIdentification,
)


class ModuleInterface:  # pragma: no cover
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

        In addition to interfacing with the module, the class also contains methods to parse logged module data. It is
        expected that these modules will be used immediately after runtime to parse raw logged data and transform it
        into the desired format for further processing and analysis.

        Some attributes of this class are assigned by the managing MicroControllerInterface class at its
        initialization. Therefore, to be fully functional, each ModuleInterface class has to be bound to an initialized
        MicroControllerInterface instance.

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
        _log_directory: Stores the path to the directory where the MicroControllerInterface that manages this class logs
            all received and transmitted messages related to this interface. The value for this attribute is assigned
            automatically by the managing MicroControllerInterface class during its initialization.
        _microcontroller_id: Stores the unique ID byte-code of the microcontroller that controls the hardware module
            interfaced by this class instance. The value for this attribute is assigned automatically by the managing
            MicroControllerInterface class during its initialization.
        _input_queue: Stores the multiprocessing queue that enables sending the data to the microcontroller
            via the managing MicroControllerInterface class. Putting messages into this queue is equivalent to
            submitting them to the send_data() method exposed by the managing MicroControllerInterface class. The value
            for this attribute is assigned automatically by the managing MicroControllerInterface class during its
            initialization.

    Raises:
        TypeError: If input arguments are not of the expected type.
    """

    def __init__(
        self,
        module_type: np.uint8,
        module_id: np.uint8,
        mqtt_communication: bool,
        error_codes: set[np.uint8] | None = None,
        data_codes: set[np.uint8] | None = None,
        mqtt_command_topics: set[str] | None = None,
    ) -> None:
        # Ensures that input byte-codes use valid value ranges
        if not isinstance(module_type, np.uint8) or not 1 <= module_type <= 255:
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected an unsigned integer value between 1 and 255 for 'module_type' argument, but encountered "
                f"{module_type} of type {type(module_type).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(module_id, np.uint8) or not 1 <= module_id <= 255:
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected an unsigned integer value between 1 and 255 for 'module_id' argument, but encountered "
                f"{module_id} of type {type(module_id).__name__}."
            )
            console.error(message=message, error=TypeError)
        if (mqtt_command_topics is not None and not isinstance(mqtt_command_topics, set)) or (
            isinstance(mqtt_command_topics, set) and not all(isinstance(topic, str) for topic in mqtt_command_topics)
        ):
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected a set of strings or None for 'mqtt_command_topics' argument, but encountered "
                f"{mqtt_command_topics} of type {type(mqtt_command_topics).__name__} and / or at least one "
                f"non-string item."
            )
            console.error(message=message, error=TypeError)
        if (error_codes is not None and not isinstance(error_codes, set)) or (
            isinstance(error_codes, set) and not all(isinstance(code, np.uint8) for code in error_codes)
        ):
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected a set of numpy uint8 values or None for 'error_codes' argument, but encountered "
                f"{error_codes} of type {type(error_codes).__name__} and / or at least one non-uint8 item."
            )
            console.error(message=message, error=TypeError)
        if (data_codes is not None and not isinstance(data_codes, set)) or (
            isinstance(data_codes, set) and not all(isinstance(code, np.uint8) for code in data_codes)
        ):
            message = (
                f"Unable to initialize the ModuleInterface instance for module {module_id} of type {module_type}. "
                f"Expected a set of numpy uint8 values or None for 'data_codes' argument, but encountered "
                f"{data_codes} of type {type(data_codes).__name__} and / or at least one non-uint8 item."
            )
            console.error(message=message, error=TypeError)

        # Saves type and ID data into class attributes
        self._module_type: np.uint8 = module_type
        self._module_id: np.uint8 = module_id

        # Combines type and ID codes into a 16-bit value. This is used to ensure every module instance has a unique
        # ID + Type combination. This method is position-aware, so inverse type-id pairs will be coded as different
        # values e.g.: 4-5 != 5-4
        self._type_id: np.uint16 = np.uint16(
            (self._module_type.astype(np.uint16) << 8) | self._module_id.astype(np.uint16)
        )

        # Resolves code and topic sets for additional data input and output processing
        self._mqtt_command_topics: set[str] = mqtt_command_topics if mqtt_command_topics is not None else set()
        self._data_codes: set[np.uint8] = data_codes if data_codes is not None else set()

        # Adds error-handling support. This allows raising errors when the module sends a message with an error code
        # from the microcontroller to the PC.
        self._error_codes: set[np.uint8] = error_codes if error_codes is not None else set()

        # If the class is configured to receive commands from MQTT, ensures that MQTT communication is enabled
        if len(self._mqtt_command_topics) > 0:
            mqtt_communication = True
        self._mqtt_communication: bool = mqtt_communication

        # These attributes are initialized to placeholder values. The actual values are assigned by the
        # MicroControllerInterface class that manages this ModuleInterface. During MicroControllerInterface
        # initialization, it updates these attributes for all managed interfaces via referencing. Therefore, after
        # initializing the main interface, all module interfaces will have these attributes configured properly.
        self._log_directory: Path | None = None
        self._microcontroller_id: np.uint8 | None = None
        self._input_queue: MPQueue | None = None  # type: ignore

    def __repr__(self) -> str:
        """Returns the string representation of the ModuleInterface instance."""
        message = (
            f"ModuleInterface(module_type={self._module_type}, module_id={self._module_id}, "
            f"combined_type_id={self._type_id}, mqtt_command_topics={self._mqtt_command_topics}, "
            f"data_codes={sorted(self._data_codes)}, error_codes={sorted(self._error_codes)})"  # type: ignore
        )
        return message

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

            Use this method to translate incoming MQTT messages into the appropriate command messages for the hardware
            module. While we currently do not explicitly support translating MQTT messages into parameter messages, this
            can be added in the future if enough interest is shown.

            See the /examples folder included with the library for examples on how to implement this method.

        Args:
            topic: The MQTT topic to which the other MQTT client sent the module-addressed command.
            payload: The payload of the message.

        Returns:
            A OneOffModuleCommand, RepeatedModuleCommand, or DequeueModuleCommand instance that stores the message to
            be sent to the microcontroller. None, if the class instance is not configured to receive commands from MQTT.
        """
        raise NotImplementedError(
            f"parse_mqtt_command() method must be implemented when subclassing the base ModuleInterface class."
        )

    @abstractmethod
    def initialize_remote_assets(self) -> None:
        """Initializes custom interface assets to be used in the remote process.

        This method is called at the beginning of the communication runtime by the managing MicroControllerInterface.
        Use this method to create and initialize any assets that cannot be pickled (to be transferred into the remote
        process).

        Notes:
            This method is called early in the preparation phase of the communication runtime, before any communication
            is actually carried out. Use this method to initialize unpickable assets, such as PrecisionTimer instances
            or connect to shared resources, such as SharedMemory buffers.

            All assets managed or created by this method should be stored in the ModuleInterface instance's attributes.
        """
        raise NotImplementedError(
            f"initialize_remote_assets() method must be implemented when subclassing the base ModuleInterface class."
        )

    @abstractmethod
    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes the incoming message and executes user-defined logic.

        This method is called by the MicroControllerInterface when the ModuleInterface instance receives a message from
        the microcontroller that uses an event code provided at class initialization as 'data_codes' argument. This
        method should be used to implement custom processing logic for the incoming data.

        Notes:
            Primarily, this method is intended to execute custom data transmission logic. For example, it can be used
            to send a message over MQTT (via a custom implementation or our MQTTCommunication class), put the data into
            a multithreading or multiprocessing queue, or use it to set a SharedMemory object. Use this method as a
            gateway to inject custom data handling into the communication runtime.

            Keep the logic inside this method as minimal as possible. All data from the microcontroller goes through the
            same communication process, so it helps to minimize real time processing of the data, as it allows for
            better communication throughput. Treat this method like you would treat a microcontroller hardware interrupt
            function.

            If your communication / processing assets cannot be pickled (to be transferred into the remote process
            used for communication), implement their initialization via the initialize_remote_assets() method.

            See the /examples folder included with the library for examples on how to implement this method.

        Args:
            message: The ModuleState or ModuleData object that stores the message received from the module instance
                running on the microcontroller.
        """
        raise NotImplementedError(
            f"process_received_data() method must be implemented when subclassing the base ModuleInterface class."
        )

    def extract_logged_data(self) -> dict[Any, list[dict[str, np.uint64 | Any]]]:
        """Extracts the data sent by the hardware module instance running on the microcontroller from the .npz
        log file generated during ModuleInterface runtime.

        This method reads the compressed '.npz' archives generated by the MicroControllerInterface class that works
        with this ModuleInterface during runtime and extracts all custom event-codes and data objects transmitted by
        the interfaced module instance from the microcontroller.

        Notes:
            The extracted data will NOT contain library-reserved events and messages. This includes all Kernel messages
            and module messages with event codes 0 through 50. The only exception to this rule is messages with event
            code 2, which report completion of commands. These messages are parsed in addition to custom messages
            sent by each hardware module.

            This method should be used as a convenience abstraction for the inner workings of the DataLogger class.
            For each ModuleInterface, it will decode and return the logged runtime data sent to the PC by the specific
            hardware module instance controlled by the interface. You need to manually implement further data
            processing steps as necessary for your specific use case and module implementation.

        Returns:
            A dictionary that uses numpy uint8 event codes as keys and stores lists of dictionaries under each key.
            Each inner dictionary contains three elements. First, an uint64 timestamp, representing the number of
            microseconds since the UTC epoch onset. Second, the data object, transmitted with the message
            (or None, for state-only events). Third, the uint8 code of the command that the module was executing when
            it sent the message to the PC.

        Raises:
            RuntimeError: If this method is called before the ModuleInterface is used to initialize a
                MicroControllerInterface class.
            ValueError: If the input path is not valid or does not point to an existing .npz archive.
        """
        if self._log_directory is None or self._microcontroller_id is None:
            error_message = (
                f"Unable to parse the log entries generated by the MicroControllerInterface for module "
                f"{self._module_id} of type {self._module_type}. The ModuleInterface has to be used to "
                f"initialize a MicroControllerInterface before calling this method."
            )
            console.error(message=error_message, error=RuntimeError)

        # Generates the log file path using the MicroControllerInterface id. Assumes that the log has been compressed
        # to the .npz format before calling this method.
        log_path = self._log_directory.joinpath(f"{self._microcontroller_id}_log.npz")  # type: ignore

        # If a compressed log archive does not exist, raises an error
        if not log_path.exists():
            error_message = (
                f"Unable to extract data for module {self._module_id} of type {self._module_type} from the log file "
                f"generated by the MicroControllerInterface with id {self._microcontroller_id}. This likely indicates "
                f"that the logs have not been compressed via DataLogger's compress_logs() method and are not available "
                f"for processing. Call log compression method before calling this method."
            )
            console.error(message=error_message, error=ValueError)

        # Loads the archive into RAM
        archive: NpzFile = np.load(file=log_path)

        # Precreates the dictionary to store the extracted data.
        event_data = {}

        # Locates the logging onset timestamp. The onset is used to convert the timestamps for logged module data into
        # absolute UTC timestamps. Originally, all timestamps other than onset are stored as elapsed time in
        # microseconds relative to onset timestamp.
        timestamp_offset = 0
        onset_us = np.uint64(0)
        timestamp: np.uint64
        for number, item in enumerate(archive.files):
            message: NDArray[np.uint8] = archive[item]  # Extracts message payload from the compressed .npy file

            # Recovers the uint64 timestamp value from each message. The timestamp occupies 8 bytes of each logged
            # message starting at index 1. If timestamp value is 0, the message contains the onset timestamp value
            # stored as 8-byte payload. Index 0 stores the source ID (uint8 value)
            if np.uint64(message[1:9].view(np.uint64)[0]) == 0:
                # Extracts the byte-serialized UTC timestamp stored as microseconds since epoch onset.
                onset_us = np.uint64(message[9:].view("<i8")[0].copy())

                # Breaks the loop onc the onset is found. Generally, the onset is expected to be found very early into
                # the loop
                timestamp_offset = number  # Records the item number at which the onset value was found.
                break

        # Once the onset has been discovered, processes the rest of the module instance data. Continues searching from
        # the position where the offset is found.
        for item in archive.files[timestamp_offset + 1 :]:
            message = archive[item]

            # Extracts the payload from each logged message.
            payload = message[9:]

            # Filters out the messages to exclusively process custom Data and State messages (event codes 51 and above).
            # The only exception to this rule is the CommandComplete state message, which uses the system-reserved code
            # '2'. In the future, if enough interest is shown, we may extend this list to also include outgoing
            # messages. For now, these messages need to be parsed manually by users that need this data.
            if (
                (payload[0] != SerialProtocols.MODULE_STATE and payload[0] != SerialProtocols.MODULE_DATA)
                or payload[1] != self._module_type
                or payload[2] != self._module_id
                or (payload[4] != 2 and payload[4] < 51)
            ):
                continue

            # Extracts the elapsed microseconds since timestamp and uses it to calculate the global timestamp for the
            # message, in microseconds since epoch onset.
            elapsed_microseconds = np.uint64(message[1:9].view(np.uint64)[0].copy())
            timestamp = onset_us + elapsed_microseconds

            # Extracts command, event, and, if supported, data object from the message payload.
            command_code = np.uint8(payload[3])
            event = np.uint8(payload[4])

            # This section is executed only if the parsed payload is MessageData. MessageState payloads are only 5 bytes
            # in size. Extracts and formats the data object, included with the logged payload.
            data: Any = None
            if len(payload) > 5:
                # noinspection PyTypeChecker
                prototype = SerialPrototypes.get_prototype_for_code(code=payload[5])

                # Depending on the prototype, reads the data object as an array or scalar
                if isinstance(prototype, np.ndarray):
                    data = payload[6:].view(prototype.dtype)[:]
                elif prototype is not None:
                    data = payload[6:].view(prototype.dtype)[0]
                else:
                    error_message = (
                        f"Unable to extract data for module {self._module_id} of type {self._module_type} from the log "
                        f"file. Failed to obtain the prototype to read the data object for message with "
                        f"event code {event} and command code {command_code}. No matching prototype was found for "
                        f"prototype code {payload[5]}."
                    )
                    console.error(message=error_message, error=ValueError)

            # Iteratively fills the dictionary with extracted data. Uses event byte-codes as keys. For each event code,
            # creates a list of tuples. Each tuple inside the list contains the timestamp, data object (or None) and
            # the active command code.
            if event not in event_data:
                event_data[event] = [{"timestamp": timestamp, "data": data, "command": command_code}]
            else:
                event_data[event].append({"timestamp": timestamp, "data": data, "command": command_code})

        return event_data

    def reset_command_queue(self) -> None:
        """Instructs the microcontroller to clear all queued commands for the specific module instance managed by this
        ModuleInterface.

        If the ModuleInterface has not been used to initialize the MicroControllerInterface, raises a RuntimeError.
        """
        if self._input_queue is not None:
            self._input_queue.put(
                DequeueModuleCommand(module_type=self._module_type, module_id=self._module_id, return_code=np.uint8(0))
            )

        message = (
            f"Unable to submit a deque command to module {self._module_id} of type {self._module_type}. The "
            f"ModuleInterface has to be used to initialize the MicroControllerInterface before calling this method."
        )
        console.error(message=message, error=RuntimeError)

    @property
    def module_type(self) -> np.uint8:
        """Returns the id-code that describes the broad type (family) of Modules managed by this interface class."""
        return self._module_type

    @property
    def module_id(self) -> np.uint8:
        """Returns the code that identifies the specific Module instance managed by the Interface class instance."""
        return self._module_id

    @property
    def data_codes(self) -> set[np.uint8]:
        """Returns the set of message event-codes that are processed during runtime, in addition to logging them to
        disk.
        """
        return self._data_codes

    @property
    def mqtt_command_topics(self) -> set[str]:
        """Returns the set of MQTT topics this instance monitors for incoming MQTT commands."""
        return self._mqtt_command_topics

    @property
    def type_id(self) -> np.uint16:
        """Returns the unique 16-bit unsigned integer value that results from combining the type-code and the id-code
        of the instance.
        """
        return self._type_id

    @property
    def error_codes(self) -> set[np.uint8]:
        """Returns the set of error event-codes used by the module instance."""
        return self._error_codes

    @property
    def mqtt_communication(self) -> bool:
        """Returns True if the class instance is configured to communicate with MQTT during runtime."""
        return self._mqtt_communication


class MicroControllerInterface:  # pragma: no cover
    """Interfaces with an Arduino or Teensy microcontroller running ataraxis-micro-controller library.

    This class contains the logic that sets up a remote daemon process with SerialCommunication, MQTTCommunication,
    and DataLogger bindings to facilitate bidirectional communication and data logging between the microcontroller and
    concurrently active local (same PC) and remote (network) processes. Additionally, it exposes methods that send
    runtime parameters and commands to the Kernel and Module classes running on the connected microcontroller.

    Notes:
        An instance of this class has to be instantiated for each microcontroller active at the same time. The
        communication will not be started until the start() method of the class instance is called.

        This class uses SharedMemoryArray to control the runtime of the remote process, which makes it impossible to
        have more than one instance of this class with the same controller_id at a time.

        Initializing MicroControllerInterface also completes the configuration of all ModuleInterface instances passed
        to the class constructor. It is essential to initialize both the interfaces and the MicroControllerInterface
        to have access to the full range of functionality provided by each ModuleInterface class.

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
        _input_queue: Stores the multiprocessing Queue used to pipe the data to be sent to the microcontroller to
            the remote communication process.
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

    # Pre-packages Kernel commands into attributes. Since Kernel commands are known and fixed at compilation,
    # they only need to be defined once.
    _reset_command = KernelCommand(
        command=np.uint8(2),
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
        microcontroller_serial_buffer_size: int,
        microcontroller_usb_port: str,
        data_logger: DataLogger,
        module_interfaces: tuple[ModuleInterface, ...],
        baudrate: int = 115200,
        mqtt_broker_ip: str = "127.0.0.1",
        mqtt_broker_port: int = 1883,
    ):
        # Initializes the started tracker. This is needed to avoid errors if initialization fails, and __del__ is called
        # for a partially initialized class.
        self._started: bool = False

        # Since the manager is now terminated via __del__ method, it makes sense to have it high in the initialization
        # order:
        self._mp_manager: SyncManager = Manager()

        # Ensures that input arguments have valid types. Only checks the arguments that are not passed to other classes,
        # such as TransportLayer, which has its own argument validation.
        if not isinstance(controller_id, np.uint8) or not 1 <= controller_id <= 255:
            message = (
                f"Unable to initialize the MicroControllerInterface instance. Expected an unsigned integer value "
                f"between 1 and 255 for 'controller_id' argument, but encountered {controller_id} of type "
                f"{type(controller_id).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(module_interfaces, tuple) or not module_interfaces:
            message = (
                f"Unable to initialize the MicroControllerInterface instance for microcontroller with id "
                f"{controller_id}. Expected a non-empty tuple of ModuleInterface instances for 'modules' argument, but "
                f"encountered {module_interfaces} of type {type(module_interfaces).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not all(isinstance(module, ModuleInterface) for module in module_interfaces):
            message = (
                f"Unable to initialize the MicroControllerInterface instance for microcontroller with id "
                f"{controller_id}. All items in 'modules' tuple must be ModuleInterface instances."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(data_logger, DataLogger):
            message = (
                f"Unable to initialize the MicroControllerInterface instance for microcontroller with id "
                f"{controller_id}. Expected an initialized DataLogger instance for 'data_logger' argument, but "
                f"encountered {data_logger} of type {type(data_logger).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Controller (kernel) ID information. Follows the same code-name-description format as module type and instance
        # values do.
        self._controller_id: np.uint8 = controller_id

        # SerialCommunication parameters. This is used to initialize the communication in the remote process.
        self._usb_port: str = microcontroller_usb_port
        self._baudrate: int = baudrate
        self._microcontroller_serial_buffer_size: int = microcontroller_serial_buffer_size

        # MQTTCommunication parameters. This is used to initialize the MQTT communication from the remote process
        # if the managed modules need this functionality.
        self._mqtt_ip: str = mqtt_broker_ip
        self._mqtt_port: int = mqtt_broker_port

        # Managed modules and data logger queue. Modules will be pre-processes as part of this initialization runtime.
        # Logger queue is fed directly into the SerialCommunication, which automatically logs all incoming and outgoing
        # data to disk.
        self._modules: tuple[ModuleInterface, ...] = module_interfaces

        # Extracts the queue from the logger instance. Other than for this step, this class does not use the instance
        # for anything else.
        self._logger_queue: MPQueue = data_logger.input_queue  # type: ignore

        # Sets up the assets used to deploy the communication runtime on a separate core and bidirectionally transfer
        # data between the communication process and the main process managing the overall runtime.
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._terminator_array: None | SharedMemoryArray = None
        self._communication_process: None | Process = None
        self._watchdog_thread: None | Thread = None

        # Verifies that all input ModuleInterface instances have a unique type+id combination and logs their runtime
        # constants (for module instances that support this process).
        processed_type_ids: set[np.uint16] = set()  # This is used to ensure each instance has a unique type+id pair.

        # Loops over all module instances and processes their data
        self._start_mqtt_client = False
        for module in self._modules:
            # If the module's combined type + id code is already inside the processed_types_id set, this means another
            # module with the same exact type and ID combination has already been processed.
            if module.type_id in processed_type_ids:
                message = (
                    f"Unable to initialize the MicroControllerInterface instance for microcontroller with "
                    f"id {controller_id}. Encountered two ModuleInterface instances with the same type-code "
                    f"({module.module_type}) and id-code ({module.module_id}), which is not allowed. Make sure each "
                    f"type and id combination is only used by a single ModuleInterface class instance."
                )
                console.error(message=message, error=ValueError)

            # Adds each processed type+id code to the tracker set
            processed_type_ids.add(module.type_id)
            if module.mqtt_communication:
                self._start_mqtt_client = True

            # Overwrites the attributes for each processed ModuleInterface with valid data. This effectively binds some
            # data and functionality realized through the main interface to each module interface. For example,
            # ModuleInterface classes can use their own _input_queue to
            module._microcontroller_id = self._controller_id
            module._log_directory = data_logger.output_directory
            module._input_queue = self._input_queue

    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
        return (
            f"MicroControllerInterface(controller_id={self._controller_id}, usb_port={self._usb_port}, "
            f"baudrate={self._baudrate}, mqtt_ip={self._mqtt_ip}, mqtt_port={self._mqtt_port}, "
            f"started={self._started})"
        )

    def __del__(self) -> None:
        """Ensures that all class resources are properly released when the class instance is garbage-collected."""
        self.stop()
        self._mp_manager.shutdown()

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
        """Sends the input message to the microcontroller managed by this interface instance.

        This is the primary interface for communicating with the Microcontroller. It allows sending all valid outgoing
        message structures to the Microcontroller for further processing. This is the only interface explicitly
        designed to communicate both with hardware modules and the Kernel class that manages the runtime of the
        microcontroller.

        Notes:
            During initialization, the MicroControllerInterface provides each managed ModuleInterface with the reference
            to the input_queue object. Each ModuleInterface can use its own _input_queue attribute to send the data
            to the communication process, eliminating the need for the data to go through this method. If you are
            developing a custom interface, you have the option for using either queue interface for submitting data to
            be sent to the microcontroller.

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
                f"Unable to send the message via the MicroControllerInterface with id {self._controller_id}. Expected "
                f"one of the valid outgoing message structures, but instead encountered {message} of type "
                f"{type(message).__name__}. Use one of the supported structures available from the communication "
                f"module."
            )
            console.error(message=message, error=TypeError)
        self._input_queue.put(message)

    def _watchdog(self) -> None:
        """This method is used by the watchdog thread to ensure the communication process is alive during runtime.

        This method will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.

        Notes:
            If the method detects that the communication process is not alive, it will carry out the necessary
            resource cleanup before raising the error and terminating the class runtime.
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
                # Prevents the __del__ method from running stop(), as the code below terminates all assets
                self._started = False

                # Activates the shutdown flag
                if self._terminator_array is not None:
                    self._terminator_array.write_data(0, np.uint8(1))

                # The process should already be terminated, but there are no downsides to making sure it is dead.
                self._communication_process.join()

                # Disconnects from the shared memory array and destroys its shared buffer.
                if self._terminator_array is not None:
                    self._terminator_array.disconnect()
                    self._terminator_array.destroy()

                # Raises the error
                message = (
                    f"The communication process of the MicroControllerInterface with id "
                    f"{self._controller_id} has been prematurely shut down. This likely indicates that the process has "
                    f"encountered a runtime error that terminated the process."
                )
                console.error(message=message, error=RuntimeError)

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
        # If the process has already been started, returns without doing anything.
        if self._started:
            return

        # Instantiates the shared memory array used to control the runtime of the communication Process.
        # Index 0 = terminator, index 1 = initialization status
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._controller_id}_terminator_array",
            prototype=np.zeros(shape=2, dtype=np.uint8),
            exist_ok=True,  # Automatically deals with already existing shared memory buffers
        )

        # Sets up the communication process. This process continuously cycles through the communication loop until
        # terminated, enabling bidirectional communication with the controller.
        self._communication_process = Process(
            target=self._runtime_cycle,
            args=(
                self._controller_id,
                self._modules,
                self._input_queue,
                self._logger_queue,
                self._terminator_array,
                self._usb_port,
                self._baudrate,
                self._microcontroller_serial_buffer_size,
                self._mqtt_ip,
                self._mqtt_port,
                self._start_mqtt_client,
            ),
            daemon=True,
        )

        # Creates the watchdog thread.
        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)

        # Initializes the communication process.
        self._communication_process.start()

        start_timer = PrecisionTimer("s")
        start_timer.reset()
        # Blocks until the microcontroller has finished all initialization steps or encounters an initialization error.
        while self._terminator_array.read_data(1) != 1:
            # Generally, there are two ways initialization failure is detected. One is if the managed process
            # terminates, which would be the case if any subclass used in the communication process raises an exception.
            # Another way if the status tracker never reaches success code (1). This latter case would likely indicate
            # that there is a communication issue where the data does not reach the controller or the PC. The
            # initialization process should be very fast, likely on the order of hundreds of microseconds. Waiting for
            # 15 seconds is likely excessive.
            if not self._communication_process.is_alive() or start_timer.elapsed > 15:
                # Ensures proper resource cleanup before terminating the process runtime, if this error is triggered:
                self._terminator_array.write_data(0, np.uint8(1))

                # Waits for at most 15 seconds before forcibly terminating the communication process to prevent
                # deadlocks
                self._communication_process.join(15)

                # Disconnects from the shared memory array and destroys its shared buffer.
                self._terminator_array.disconnect()
                self._terminator_array.destroy()

                message = (
                    f"MicroControllerInterface with id {self._controller_id} has failed to initialize the "
                    f"communication with the microcontroller. If the class did not display error messages in the "
                    f"terminal, activate the 'console' variable from ataraxis-base-utilities library to enable "
                    f"displaying error messages raised during daemon process runtimes."
                )
                console.error(error=RuntimeError, message=message)

        # Starts the process watchdog thread once the initialization is complete
        self._watchdog_thread.start()

        # Issues the global reset command. This ensures that the controller always starts with 'default' parameters.
        self.reset_controller()

        # Sets the started flag
        self._started = True

    def stop(self) -> None:
        """Shuts down the communication process and frees all reserved resources."""
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
        module_interfaces: tuple[ModuleInterface, ...],
        input_queue: MPQueue,  # type: ignore
        logger_queue: MPQueue,  # type: ignore
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

        Notes:
            Each managed ModuleInterface may contain custom logic for processing and routing the data. This method
            calls the custom logic bindings for each interface on a need-based method.

        Args:
            controller_id: The byte-code identifier of the target microcontroller. This is used to ensure that the
                instance interfaces with the correct controller and to source-stamp logged data.
            module_interfaces: A tuple that stores ModuleInterface classes managed by this MicroControllerInterface
                instance.
            input_queue: The multiprocessing queue used to issue commands to the microcontroller.
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

        # Constructs Kernel-addressed commands used to verify that the Interface and the microcontroller are
        # configured appropriately
        identify_controller_command = KernelCommand(
            command=np.uint8(3),
            return_code=np.uint8(0),
        )
        identify_modules_command = KernelCommand(
            command=np.uint8(4),
            return_code=np.uint8(0),
        )

        # Initializes the timer used during initialization to abort stale initialization attempts.
        timeout_timer = PrecisionTimer("ms")

        # Connects to the terminator array. This is done early, as the terminator_array is used to track the
        # initialization and runtime status of the process.
        terminator_array.connect()

        # Precreates the assets used to optimize the communication runtime cycling. These assets are filled below to
        # support efficient interaction between the Communication class and the ModuleInterface classes.
        mqtt_command_map: dict[str, tuple[ModuleInterface, ...] | list[ModuleInterface]] = {}
        processing_map: dict[np.uint16, ModuleInterface] = {}
        for module in module_interfaces:
            # For each module, initializes the assets that need to be configured / created inside the remote Process:
            module.initialize_remote_assets()

            # If the module is configured to receive commands from MQTT, sets up the necessary assets. For this,
            # extracts the monitored topics from each module
            for topic in module.mqtt_command_topics:
                # Extends the list of module interfaces that listen for that particular topic. This allows addressing
                # multiple modules at the same time, as long as they all listen to the same topic.
                existing_modules = mqtt_command_map.get(topic, [])
                mqtt_command_map[topic] = existing_modules + [module]  # type: ignore

            # If the module is configured to process incoming data or raise runtime errors, maps its type+id combined
            # code to the interface instance. This is used to quickly find the module interface instance addressed by
            # incoming data, so that it can handle teh data or error message.
            if len(module.data_codes) != 0 or len(module.error_codes) != 0:
                processing_map[module.type_id] = module

        # Converts the list of interface instance into a tuple for slightly higher runtime efficiency.
        for key in mqtt_command_map.keys():
            mqtt_command_map[key] = tuple(mqtt_command_map[key])

        # Initializes the serial communication class and connects to the target microcontroller.
        serial_communication = SerialCommunication(
            usb_port=usb_port,
            source_id=controller_id,
            logger_queue=logger_queue,
            baudrate=baudrate,
            microcontroller_serial_buffer_size=microcontroller_buffer_size,
        )

        # Sends microcontroller identification command. This command requests the microcontroller to return its
        # id code.
        # noinspection PyTypeChecker
        serial_communication.send_message(message=identify_controller_command)

        # Blocks until the microcontroller sends its identification code.
        timeout_timer.reset()
        response = None
        while not isinstance(response, ControllerIdentification):
            # If no response is received within 2 seconds, repeats the identification request. Older microcontrollers
            # that reset on serial connection may miss the first request if they were resetting their communication
            # hardware, but should receive the second request.
            if timeout_timer.elapsed > 2000:
                # noinspection PyTypeChecker
                serial_communication.send_message(message=identify_controller_command)

            # If there is no response after 4 seconds and 2 requests, aborts initialization with an error
            elif timeout_timer.elapsed > 4000:
                message = (
                    f"Unable to initialize the communication with the microcontroller {controller_id}. The "
                    f"microcontroller did not respond to the identification request in time (4 seconds) after two "
                    f"requests were sent."
                )
                console.error(message=message, error=RuntimeError)

            # The response will be None if there is no data to receive and a valid message otherwise.
            response = serial_communication.receive_message()

        # If response is received, but the ID contained in the received message does not match the expected ID,
        # raises an error
        if response.controller_id != controller_id:
            # Raises the error.
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. Expected "
                f"{controller_id} in response to the controller identification request, but "
                f"received a non-matching id {response.controller_id}."
            )
            console.error(message=message, error=ValueError)

        # Next, verifies that the microcontroller has a module instance expected by each managed interface class
        # noinspection PyTypeChecker
        serial_communication.send_message(message=identify_modules_command)

        # Sequentially receives the ID data for each module
        timeout_timer.reset()
        module_type_ids = []
        while timeout_timer.elapsed < 2000:
            # Receives the message. If the message is a module type+id code, adds it to the storage list
            response = serial_communication.receive_message()
            if isinstance(response, ModuleIdentification):
                module_type_ids.append(response.module_type_id)

                # Keeps the loop running as long as messages keep coming in within 2-second intervals.
                timeout_timer.reset()

        # If no message was received from the microcontroller, raises an error
        if len(module_type_ids) == 0:
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. The "
                f"microcontroller did not respond to module identification request in time (3 seconds)."
            )
            console.error(message=message, error=RuntimeError)

        # The microcontroller may have more modules than the number of managed interfaces, but it can never have fewer
        # modules than interfaces.
        if len(module_type_ids) < len(module_interfaces):
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. The number of "
                f"ModuleInterface instances ({len(module_interfaces)}) is greater than the number of physical hardware "
                f"module instances managed by the microcontroller ({len(module_type_ids)})."
            )
            console.error(message=message, error=ValueError)

        # Ensures that all type_id codes are unique on the microcontroller. This is done for ModuleInterfaces at class
        # instantiation. After this step, it is safe to assume that all module instances and interfaces are uniquely
        # identifiable.
        if len(module_type_ids) != len(set(module_type_ids)):
            message = (
                f"Unable to initialize the communication with the microcontroller {controller_id}. The microcontroller "
                f"contains multiple module instances with the same type + id combination. Make sure each module "
                f"instance has a unique type + id combination."
            )
            console.error(message=message, error=ValueError)

        # Ensures that each module interface has a matching hardware module on the microcontroller
        for module in module_interfaces:
            if module.type_id not in module_type_ids:
                message = (
                    f"Unable to initialize the communication with the microcontroller {controller_id}. "
                    f"The ModuleInterface class for module with type {module.module_type} and id {module.module_id}"
                    f"codes does not have a matching hardware module instance on the microcontroller."
                )
                console.error(message=message, error=ValueError)

        # Initializes the MQTTCommunication class. If the interface does not need MQTT communication, this
        # initialization will only statically reserve a minor portion of RAM with no other adverse effects. If the
        # mqtt_input_map is empty, the class initialization method will correctly interpret this as a case where no
        # topics need to be monitored.
        mqtt_communication = MQTTCommunication(
            ip=mqtt_ip, port=mqtt_port, monitored_topics=tuple(mqtt_command_map.keys())
        )

        # Connects to the MQTT broker if at least one interface requires this functionality
        if start_mqtt_client:
            mqtt_communication.connect()

        # Reports that communication class has been successfully initialized. Seeing this code means
        # that the communication appears to be functioning correctly and that the interface and the microcontroller
        # appear to be configured well. While this does not guarantee the runtime will continue running
        # without errors, it is very likely to be so.
        terminator_array.write_data(index=1, data=np.uint8(1))

        try:
            # Initializes the main communication loop. This loop will run until the exit conditions are encountered.
            # The exit conditions for the loop require the first variable in the terminator_array to be set to True
            # and the main input queue of the interface to be empty. This ensures that all queued commands issued from
            # the central process are fully carried out before the communication is terminated.
            while not terminator_array.read_data(index=0, convert_output=True) or not input_queue.empty():
                # Main data sending loop. The method will sequentially retrieve the queued messages and send them to
                # the microcontroller.
                while not input_queue.empty():
                    # Transmits the data to the microcontroller. Expects that the queue ONLY yields valid messages.
                    serial_communication.send_message(input_queue.get())

                # MQTT data sending loop
                while mqtt_communication.has_data:
                    # If MQTTCommunication has received data, loops over all interfaces that requested the data from
                    # this topic and calls their mqtt data processing method while passing it the topic and the
                    # received message payload.
                    topic, payload = mqtt_communication.get_data()

                    # Each incoming message will be processed by each module subscribed to this topic. Since
                    # MQTTCommunication is configured to only listen to topics submitted by the interface classes, the
                    # topic is guaranteed to be inside the mqtt_input_map dictionary and have at least one Module which
                    # can process its data.
                    for module in mqtt_command_map[topic]:
                        # Transmits the data to the microcontroller. Since parse_mqtt_command() is ONLY called for
                        # topics specified by the user, expects that the method ALWAYS returns a valid message.
                        serial_communication.send_message(
                            module.parse_mqtt_command(
                                topic=topic,
                                payload=payload,
                            )
                        )

                # Attempts to receive the data from microcontroller
                in_data = serial_communication.receive_message()

                # If no data is available cycles the loop
                if in_data is None:
                    continue

                # Converts valid KernelData and State messages into errors. This is used to raise runtime errors when
                # an appropriate error message is transmitted from the microcontroller. This clause does not evaluate
                # non-error codes.
                if isinstance(in_data, (KernelData, KernelState)):
                    # Note, event codes are taken directly from the microcontroller's Kernel class.

                    # kModuleSetupError
                    if in_data.event == 2:
                        message = (
                            f"The microcontroller {controller_id} encountered an error when executing command "
                            f"{in_data.command}. Error code: 2. The hardware module with type {in_data.data_object[0]} "
                            f"and id {in_data.data_object[1]} has failed its setup sequence. Firmware re-upload is "
                            f"required to restart the controller."
                        )
                        raise console.error(message=message, error=RuntimeError)

                    # kReceptionError
                    if in_data.event == 3:
                        message = (
                            f"The microcontroller {controller_id} encountered an error when executing command "
                            f"{in_data.command}. Error code: 3. "
                            f"The microcontroller was not able to receive (parse) the PC-sent data and had to "
                            f"abort the reception. Last Communication status code was {in_data.data_object[0]} and "
                            f"last TransportLayer status code was {in_data.data_object[1]}. Overall, this indicates "
                            f"broader issues with microcontroller-PC communication."
                        )
                        raise console.error(message=message, error=RuntimeError)

                    # kTransmissionError
                    if in_data.event == 4:
                        message = (
                            f"The microcontroller {controller_id} encountered an error when executing command "
                            f"{in_data.command}. Error code: 4. "
                            f"The microcontroller's Kernel class was not able to send data to the PC and had to abort "
                            f"the transmission. Last Communication status code was {in_data.data_object[0]} and "
                            f"last TransportLayer status code was {in_data.data_object[1]}. Overall, this indicates "
                            f"broader issues with microcontroller-PC communication."
                        )
                        raise console.error(message=message, error=RuntimeError)

                    # kInvalidMessageProtocol
                    if in_data.event == 5:
                        message = (
                            f"The microcontroller {controller_id} encountered an error when executing command "
                            f"{in_data.command}. Error code: 5. "
                            f"The microcontroller received a message with an invalid (unsupported) message protocol "
                            f"code {in_data.data_object[0]}."
                        )
                        raise console.error(message=message, error=RuntimeError)

                    # kModuleParametersError
                    if in_data.event == 8:
                        message = (
                            f"The microcontroller {controller_id} encountered an error when executing command "
                            f"{in_data.command}. Error code: 8. "
                            f"The microcontroller was not able to apply new runtime parameters received from the PC to "
                            f"the target hardware module with type {in_data.data_object[0]} and id "
                            f"{in_data.data_object[1]}."
                        )
                        raise console.error(message=message, error=RuntimeError)

                    # kCommandNotRecognized
                    if in_data.event == 9:
                        message = (
                            f"The microcontroller {controller_id} encountered an error when executing command "
                            f"{in_data.command}. Error code: 9. "
                            f"The microcontroller has received an invalid (unrecognized) command code "
                            f"{in_data.command}."
                        )
                        raise console.error(message=message, error=RuntimeError)

                    # kTargetModuleNotFound
                    if in_data.event == 10:
                        message = (
                            f"The microcontroller {controller_id} encountered an error when executing command "
                            f"{in_data.command}. Error code: 10. "
                            f"The microcontroller was not able to find the module addressed by the incoming command or "
                            f"parameters message. The target hardware module with type {in_data.data_object[0]} and id "
                            f"{in_data.data_object[1]} does not exist for that microcontroller."
                        )
                        raise console.error(message=message, error=RuntimeError)

                # Event codes from 0 through 50 are reserved for system use. These codes are handled by this clause,
                # which translates error messages sent by the base Module class into error codes, similar as to
                # how it is done by the Kernel.
                if isinstance(in_data, (ModuleState, ModuleData)) and in_data.event < 51:
                    # Note, event codes are taken directly from the microcontroller's (base) Module class.

                    # kTransmissionError
                    if in_data.event == 1:
                        message = (
                            f"The module with type {in_data.module_type} and id {in_data.module_id} managed by the "
                            f"{controller_id} encountered an error when executing command {in_data.command}. "
                            f"Error code: 1. The module was not able to send data to the PC and had to abort the "
                            f"transmission. Last Communication status code was {in_data.data_object[0]} and last "
                            f"TransportLayer status code was {in_data.data_object[1]}. Overall, this indicates broader "
                            f"issues with microcontroller-PC communication."
                        )
                        raise console.error(message=message, error=RuntimeError)

                    # kCommandNotRecognized
                    if in_data.event == 3:
                        message = (
                            f"The module with type {in_data.module_type} and id {in_data.module_id} managed by the "
                            f"{controller_id} encountered an error when executing command {in_data.command}. "
                            f"Error code: 3. The module has received an invalid (unrecognized) command code "
                            f"{in_data.command}."
                        )
                        raise console.error(message=message, error=RuntimeError)

                # Event codes 51 or above are used by the module developers to communicate states and errors.
                if isinstance(in_data, (ModuleState, ModuleData)) and in_data.event > 50:
                    # Computes the combined type and id code for the incoming data. This is used to find the specific
                    # ModuleInterface to which the message is addressed and, if necessary, invoke interface-specific
                    # additional processing methods.
                    target_type_id: np.uint16 = np.uint16(
                        (in_data.module_type.astype(np.uint16) << 8) | in_data.module_id.astype(np.uint16)
                    )

                    # If the interface addressed by the message is not configured to raise errors or process the data,
                    # ends processing
                    if target_type_id not in processing_map:
                        continue

                    # Otherwise, gets the reference to the targeted interface.
                    module = processing_map[target_type_id]

                    # If the incoming message contains an event code matching one of the interface's error-codes,
                    # raises an error message.
                    if in_data.event in module.error_codes:
                        if isinstance(in_data, ModuleData):
                            message = (
                                f"The module with type {in_data.module_type} and id {in_data.module_id} managed by the "
                                f"{controller_id} encountered an error when executing command {in_data.command}. "
                                f"Error code: {in_data.event}. The error message also contained the following data"
                                f"object: {in_data.data_object}."
                            )
                        else:
                            message = (
                                f"The module with type {in_data.module_type} and id {in_data.module_id} managed by the "
                                f"{controller_id} encountered an error when executing command {in_data.command}. "
                                f"Error code: {in_data.event}."
                            )
                        console.error(message=message, error=RuntimeError)

                    # Otherwise, if the incoming message is not an error and contains an event-code matching one of the
                    # interface's data-codes, calls the data processing method.
                    if in_data.event in module.data_codes:
                        module.process_received_data(message=in_data)

        # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
        # before re-raising the exception to terminate the process.
        except Exception as e:
            sys.stderr.write(str(e))
            sys.stderr.flush()
            raise e

        # Ensures that local assets are always properly terminated
        finally:
            terminator_array.disconnect()
            mqtt_communication.disconnect()

            if start_mqtt_client:
                mqtt_communication.disconnect()
