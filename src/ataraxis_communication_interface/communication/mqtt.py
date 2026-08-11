"""Provides the MQTTCommunication class for bidirectional MQTT communication over a shared MQTT broker."""

from __future__ import annotations

from queue import Queue
from typing import TYPE_CHECKING, Any

from paho.mqtt.enums import CallbackAPIVersion
import paho.mqtt.client as mqtt
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from paho.mqtt.properties import Properties
    from paho.mqtt.reasoncodes import ReasonCode


class MQTTCommunication:
    """Provides methods for bidirectionally communicating with other clients connected to the same MQTT broker using the
    MQTT protocol over the TCP interface.

    Notes:
        Primarily, the class is intended to be used alongside the SerialCommunication class to transfer the data between
        microcontrollers and the rest of the runtime infrastructure.

        The MQTT protocol requires a broker that facilitates the communication, which has to be available to this class
        at initialization. See https://mqtt.org/ for more details.

    Args:
        ip: The IP address of the MQTT broker. Defaults to "127.0.0.1" (localhost).
        port: The socket port used by the MQTT broker. Defaults to 1883, the standard MQTT port.
        monitored_topics: The tuple of MQTT topics to monitor for incoming messages. Defaults to None, which
            subscribes to no topics.

    Attributes:
        _ip: Stores the IP address of the MQTT broker.
        _port: Stores the port used by the broker's TCP socket.
        _connected: Tracks whether the class instance is currently connected to the MQTT broker.
        _monitored_topics: Stores the topics monitored by the instance for incoming messages.
        _output_queue: Buffers incoming messages received from other MQTT clients before their data is accessed via
            class methods.
        _client: The initialized MQTT client instance that carries out the communication.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 1883,
        monitored_topics: tuple[str, ...] | None = None,
    ) -> None:
        self._ip: str = ip
        self._port: int = port
        self._connected: bool = False
        self._monitored_topics: tuple[str, ...] = monitored_topics if monitored_topics is not None else ()

        # Initializes the queue to buffer incoming data. The queue may not be used if the class is not configured to
        # receive any data, but this is a fairly minor inefficiency.
        self._output_queue: Queue = Queue()  # type: ignore[type-arg]

        # Initializes the MQTT client. Note, it needs to be connected before it can send and receive messages!
        self._client: mqtt.Client = mqtt.Client(
            protocol=mqtt.MQTTv5,
            transport="tcp",
            callback_api_version=CallbackAPIVersion.VERSION2,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the MQTTCommunication instance."""
        return (
            f"MQTTCommunication(broker_ip={self._ip}, socket_port={self._port}, connected={self._connected}, "
            f"subscribed_topics={self._monitored_topics})"
        )

    def __del__(self) -> None:
        """Ensures that the instance disconnects from the broker before being garbage-collected."""
        self.disconnect()

    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:
        """Receives data from the MQTT broker and buffers it in the output queue.

        Args:
            _client: The MQTT client that received the message. Currently not used.
            _userdata: Custom user-defined data. Currently not used.
            message: The received MQTT message.
        """
        # Whenever a message is received, it is buffered via the local queue object.
        self._output_queue.put_nowait((message.topic, message.payload))

    def _on_disconnect(
        self,
        _client: mqtt.Client,
        _userdata: Any,
        _disconnect_flags: mqtt.DisconnectFlags,
        _reason_code: ReasonCode,
        _properties: Properties | None,
    ) -> None:
        """Clears the tracked connection state when the client loses its link to the MQTT broker.

        Args:
            _client: The MQTT client that lost the connection. Currently not used.
            _userdata: Custom user-defined data. Currently not used.
            _disconnect_flags: The flags that communicate whether the broker or the client initiated the
                disconnection. Currently not used.
            _reason_code: The code that communicates the reason for the disconnection. Currently not used.
            _properties: The MQTT v5 properties transmitted with the disconnection. Currently not used.
        """
        self._connected = False

    def connect(self) -> None:
        """Connects to the MQTT broker and subscribes to the requested list of monitored topics.

        Notes:
            This method has to be called after class initialization to start the communication process. Any message
            sent to the MQTT broker from other clients before this method is called may not reach this instance.

            If this instance is configured to subscribe (listen) to any topics, it starts a perpetually active thread
            with a listener callback to monitor the incoming traffic.

        Raises:
            ConnectionError: If the MQTT broker cannot be connected using the provided IP and Port.
        """
        # Guards against re-connecting an already connected client.
        if self._connected:
            return

        # Connects to the broker. Newer paho-mqtt versions surface socket-level failures as OSError subclasses instead
        # of returning an error code, which covers the timed out connection, the refused connection, and the
        # unresolvable host name.
        failure_reason: str | None = None
        try:
            result = self._client.connect(host=self._ip, port=self._port)
        except OSError as error:
            result = mqtt.MQTT_ERR_NO_CONN
            failure_reason = str(error)
        if result != mqtt.MQTT_ERR_SUCCESS:
            # Reports the socket-level text whenever the failure carried one, so that a failed name resolution stays
            # distinguishable from a refused connection.
            reason = failure_reason if failure_reason is not None else mqtt.error_string(result).removesuffix(".")
            message = (
                f"Unable to connect MQTTCommunication class instance to the MQTT broker. Failed to connect to MQTT "
                f"broker at {self._ip}:{self._port} with the error: {reason}. This likely indicates that the broker is "
                f"not running or that there is an issue with the provided IP and socket port."
            )
            console.error(message=message, error=ConnectionError)

        # Tracks broker-side and library-side link losses, which keeps the connection guards of the data exchange
        # methods accurate for publish-only instances as well as for subscribed ones.
        self._client.on_disconnect = self._on_disconnect

        # If the class is configured to connect to any topics, enables the message callback and starts the monitoring
        # thread.
        if self._monitored_topics:
            self._client.on_message = self._on_message
            self._client.loop_start()

        # Subscribes to necessary topics with qos of 0. Note, this assumes that the communication is happening over
        # a virtual TCP socket and, therefore, does not need qos.
        for topic in self._monitored_topics:
            self._client.subscribe(topic=topic, qos=0)

        self._connected = True

    def send_data(self, topic: str, payload: str | bytes | bytearray | float | None = None) -> None:
        """Publishes the input payload to the specified MQTT topic.

        Args:
            topic: The MQTT topic to publish the data to.
            payload: The data to be published. Setting this to None sends an empty message.

        Raises:
            ConnectionError: If the instance is not connected to the MQTT broker or if the client is unable to hand
                the payload to the broker.
        """
        if not self._connected:
            message = (
                f"Unable to send data to the MQTT broker at {self._ip}:{self._port} via the MQTTCommunication "
                f"instance. The MQTTCommunication instance is not connected to the MQTT broker, call connect() method "
                f"before sending data."
            )
            console.error(message=message, error=ConnectionError)
        # The client reports a dropped link through the returned status rather than by raising, so the status is the
        # only signal that the payload never reached the broker.
        information = self._client.publish(topic=topic, payload=payload, qos=0)
        if information.rc != mqtt.MQTT_ERR_SUCCESS:
            failure = mqtt.error_string(information.rc).removesuffix(".")
            message = (
                f"Unable to send data to the MQTT topic {topic} of the broker at {self._ip}:{self._port} via the "
                f"MQTTCommunication instance. The publication attempt failed with the error: {failure}."
            )
            console.error(message=message, error=ConnectionError)

    @property
    def has_data(self) -> bool:
        """Returns True if the instance's get_data() method can be used to retrieve a message received from another
        MQTT client.
        """
        return not self._output_queue.empty()

    def get_data(self) -> tuple[str, bytes] | None:
        """Extracts and returns the first available message stored inside the instance's buffer queue.

        Returns:
            A two-element tuple if there is data to retrieve. The first element is the MQTT topic of the received
            message. The second element is the payload of the message. If there is no data to retrieve, returns None.

        Raises:
            ConnectionError: If the instance is not connected to the MQTT broker.
        """
        if not self._connected:
            message = (
                f"Unable to get data from the MQTT broker at {self._ip}:{self._port} via the MQTTCommunication "
                f"instance. The MQTTCommunication instance is not connected to the MQTT broker, call connect() method "
                f"before retrieving data."
            )
            console.error(message=message, error=ConnectionError)

        if not self.has_data:
            return None

        data: tuple[str, bytes] = self._output_queue.get_nowait()
        return data

    def disconnect(self) -> None:
        """Disconnects the client from the MQTT broker."""
        # Prevents running the rest of the code if the client was not connected.
        if not self._connected:
            return

        # Stops the listener thread if the client was subscribed to receive topic data.
        if self._monitored_topics:
            self._client.loop_stop()

        # Disconnects from the client.
        self._client.disconnect()

        self._connected = False
