"""This module provides ModuleInterface implementations for the default hardware modules shipped with the
AtaraxisMicroController library. Primarily, these classes are intended to serve as examples for how to implement
custom module Interfaces.
"""

from json import dumps
import math
from multiprocessing import Queue as MPQueue

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console
from numpy.polynomial.polynomial import polyfit

from .communication import (
    ModuleData,
    ModuleState,
    ModuleParameters,
    UnityCommunication,
    OneOffModuleCommand,
    RepeatedModuleCommand,
)
from .microcontroller import ModuleInterface


class EncoderInterface(ModuleInterface):
    """Interfaces with EncoderModule instances running on Ataraxis MicroControllers.

    EncoderModule allows interfacing with quadrature encoders used to monitor the direction and magnitude of connected
    object's rotation. To achieve the highest resolution, the module relies on hardware interrupt pins to detect and
    handle the pulses sent by the two encoder channels.

    Notes:
        By default, EncoderModule instances are configured to interfere with any other library or class that uses
        AttachInterrupt() function. Usually it is advised to isolate EncoderModule onto a separate microcontroller
        board, if possible, to ensure maximum performance.

        This interface comes pre-configured to send incoming motion data to Unity. If you do not need this
        functionality, override the unity_output flag at class instantiation!

    Args:
        module_id: The identifier code of the interfaced EncoderModule instance.
        instance_name: The human-readable name of the interfaced EncoderModule instance.
        output_data: Determines whether the instance should send motion data to Unity.
        motion_topic: The MQTT topic to which the instance should send the motion data received from the
            microcontroller.
        encoder_ppr: The resolution of the managed quadrature encoder, in Pulses Per Revolution (PPR). This is the
            number of quadrature pulses the encoder emits per full 360-degree rotation. If this number is not known,
            provide a placeholder value and use get_ppr() command to estimate the PPR using the index channel of the
            encoder.
        object_diameter: The diameter of the rotating object connected to the encoder, in centimeters. This is used to
            convert encoder pulses into rotated distance in cm.
        cm_per_unity_unit: The conversion factor to translate the distance traveled by the edge of the circular object
             into Unity units. This value works together with object_diameter and encoder_ppr to translate raw
             encoder pulses received from the microcontroller into Unity-compatible units.

    Attributes:
        _motion_topic: Stores the MQTT motion topic.
        _ppr: Stores the resolution of the managed quadrature encoder.
        _object_diameter: Stores the diameter of the object connected to the encoder.
        _cm_per_unity_unit: Stores the conversion factor that translates centimeters into Unity units.
        _unity_unit_per_pulse: Stores the conversion factor to translate encoder pulses into Unity units.
    """

    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        output_data: bool = True,
        motion_topic: str = "LinearTreadmill/Data",
        encoder_ppr: int = 8192,
        object_diameter: float = 15.0333,  # 0333 is to account for the wheel wrap
        cm_per_unity_unit: float = 10.0,
    ) -> None:
        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="EncoderModule",
            module_type=np.uint8(2),
            module_id=module_id,
            instance_name=instance_name,
            unity_input_topics=None,
            output_data=output_data,
        )

        # Saves additional data to class attributes.
        self._motion_topic = motion_topic
        self._ppr = encoder_ppr
        self._object_diameter = object_diameter
        self._cm_per_unity_unit = cm_per_unity_unit

        # Computes the conversion factor to translate encoder pulses into unity units. Rounds to 12 decimal places for
        # consistency and to ensure repeatability.
        self._unity_unit_per_pulse = np.round(
            a=np.float64((math.pi * object_diameter) / (encoder_ppr * cm_per_unity_unit)),
            decimals=12,
        )

    def send_data(
        self,
        message: ModuleState | ModuleData,
        unity_communication: UnityCommunication,
        _mp_queue: MPQueue,  # type: ignore
    ) -> None:
        # ModuleState messages do not communicate any data that needs to be sent to Unity.
        if isinstance(message, ModuleState):
            return

        # If the incoming message is not a CCW or CW motion report, aborts processing:
        if message.event != np.uint8(51) and message.event != np.uint8(52):
            return

        # The rotation direction is encoded via the message event code. CW rotation (code 51) is interpreted as negative
        # and CCW as positive.
        sign = 1 if message.event == np.uint8(51) else -1

        # Translates the absolute motion into the CW / CCW vector and converts from raw pulse count to Unity units
        # using the precomputed conversion factor. Uses float64 and rounds to 12 decimal places for consistency and
        # precision
        signed_motion = np.round(
            a=np.float64(message.data_object) * self._unity_unit_per_pulse * sign,
            decimals=12,
        )

        # Encodes the motion data into the format expected by the GIMBL Unity module and serializes it into a
        # byte-string.
        json_string = dumps(obj={"movement": signed_motion})
        byte_array = json_string.encode("utf-8")

        # Publishes the motion to the appropriate MQTT topic.
        unity_communication.send_data(topic=self._motion_topic, payload=byte_array)

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def log_variables(self) -> NDArray[np.uint8] | None:
        # Serializes and returns the PPR, Object Diameter, and conversion factors used to translate pulses to cm and
        # cm to Unity units. Jointly, these values are enough to translate raw encoder pulses (which are logged to
        # disk) to other distance measurements.
        output_array: NDArray[np.uint8] = np.array(
            [
                np.uint32(self._ppr),
                np.float64(self._object_diameter),
                np.float64(self._cm_per_unity_unit),
                self._unity_unit_per_pulse,
            ],
            dtype=np.uint8,
        )
        return output_array

    def set_parameters(
        self,
        report_ccw: np.bool | bool = np.bool(True),
        report_cw: np.bool | bool = np.bool(True),
        delta_threshold: np.uint32 | int = np.uint32(10),
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the EncoderModule instance.

        Use this method to package and apply new PC-addressable parameters to the EncoderModule instance managed by
        this Interface class.

        Args:
            report_ccw: Determines whether to report rotation in the CCW (positive) direction.
            report_cw: Determines whether to report rotation in the CW (negative) direction.
            delta_threshold: The minimum number of pulses required for the motion to be reported. Depending on encoder
                resolution, this allows setting the 'minimum rotation distance' threshold for reporting. Note, if the
                change is 0 (the encoder readout did not change), it will not be reported, regardless of the
                value of this parameter. Sub-threshold motion will be aggregated (summed) across readouts until a
                significant overall change in position is reached to justify reporting it to the PC.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(np.bool(report_ccw), np.bool(report_cw), np.uint32(delta_threshold)),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Returns the number of pulses accumulated by the EncoderModule since the last check or reset.

        If there has been a significant change in the absolute count of pulses, reports the change and direction to the
        PC. It is highly advised to issue this command to repeat (recur) at a desired interval to continuously monitor
        the encoder state, rather than repeatedly calling it as a one-off command for best runtime efficiency.

        This command allows continuously monitoring the rotation of the object connected to the encoder. It is designed
        to return the absolute raw count of pulses emitted by the encoder in response to the object ration. This allows
        avoiding floating-point arithmetic on the microcontroller and relies on the PC to convert pulses to standard
        units,s uch as centimeters. The specific conversion algorithm depends on the encoder and motion diameter.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the
            command will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
            cycle_delay=np.uint32(repetition_delay),
        )

    def reset_pulse_count(self) -> OneOffModuleCommand:
        """Resets the EncoderModule pulse tracker to 0.

        This command allows resetting the encoder without evaluating its current pulse count. Currently, this command
        is designed ot only run once.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2),
            noblock=np.bool(False),
        )

    def get_ppr(self) -> OneOffModuleCommand:
        """Uses the index channel of the EncoderModule to estimate its Pulse-per-Revolution (PPR).

        The PPR allows converting raw pulse counts the EncoderModule sends to the PC to accurate displacement in
        standard distance units, such as centimeters. This is a service command not intended to be used during most
        runtimes if the PPR is already known. It relies on the object tracked by the encoder completing up to 11 full
        revolutions and uses the index channel of the encoder to measure the number of pulses per each revolution.

        Notes:
            Make sure the evaluated encoder rotates at a slow and stead speed until this command completes. Similar to
            other service commands, it is designed to deadlock the controller until the command completes. Note, the
            EncoderModule does not provide the rotation, this needs to be done manually.

            The direction of the rotation is not relevant for this command, as long as the object makes the full
            360-degree revolution.

            The command is optimized for the object to be rotated with a human hand at a steady rate, so it delays
            further index pin polling for 100 milliseconds each time the index pin is triggered. Therefore, if the
            object is moving too fast (or too slow), the command will not work as intended.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )


class TTLInterface(ModuleInterface):
    """Interfaces with TTLModule instances running on Ataraxis MicroControllers.

    TTLModule facilitates exchanging Transistor-to-Transistor Logic (TTL) signals between various hardware systems, such
    as microcontrollers, cameras and recording devices. The module contains methods for both sending and receiving TTL
    pulses, but each TTLModule instance can only perform one of these functions at a time.

    Args:
        module_id: The identifier code of the interfaced TTLModule instance.
        instance_name: The human-readable name of the interfaced TTLModule instance.
    """

    def __init__(self, module_id: np.uint8, instance_name: str) -> None:
        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="TTLModule",
            module_type=np.uint8(1),
            module_id=module_id,
            instance_name=instance_name,
            unity_input_topics=None,
            output_data=False,
        )

    def send_data(
        self,
        message: ModuleData | ModuleState,
        unity_communication: UnityCommunication,
        mp_queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def log_variables(self) -> NDArray[np.uint8] | None:
        """Not used."""
        return None

    def set_parameters(
        self, pulse_duration: np.uint32 = np.uint32(10000), averaging_pool_size: np.uint8 = np.uint8(0)
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the TTLModule instance.

        Use this method to package and apply new PC-addressable parameters to the TTLModule instance managed by
        this Interface class.

        Args:
            pulse_duration: The duration, in microseconds, of each emitted TTL pulse HIGH phase. This determines
                how long the TTL pin stays ON when emitting a pulse.
            averaging_pool_size: The number of digital pin readouts to average together when checking pin state. This
                is used during the execution of check_state() command to debounce the pin readout and acts in addition
                to any built-in debouncing.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(pulse_duration, averaging_pool_size),
        )

    def send_pulse(
        self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        """Triggers TTLModule to deliver a one-off or recurrent (repeating) digital TTL pulse.

        This command is well-suited to carry out most forms of TTL communication, but it is adapted for comparatively
        low-frequency communication at 10-200 Hz. This is in-contrast to PWM outputs capable of mHz or even Khz pulse
        oscillation frequencies.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
                will only run once. The exact repetition delay will be further affected by other modules managed by the
                same microcontroller and may not be perfectly accurate.
            noblock: Determines whether the command should block the microcontroller while emitting the high phase of
                the pulse or not. Blocking ensures precise pulse duration, non-blocking allows the microcontroller to
                perform other operations while waiting, increasing its throughput.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(noblock),
            cycle_delay=repetition_delay,
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Triggers the TTLModule to continuously deliver a digital HIGH or LOW signal.

        This command locks the TTLModule managed by this Interface into delivering the desired logical signal.

        Args:
            state: The signal to output. Set to True for HIGH and False for LOW.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=np.bool(False),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the state of the TTL signal received by the TTLModule.

        This command evaluates the state of the TTLModule's input pin and, if it is different from the previous state,
        reports it to the PC. This approach ensures that the module only reports signal level shifts (edges), preserving
        communication bandwidth.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
            will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(4),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )


class BreakInterface(ModuleInterface):
    """Interfaces with BreakModule instances running on Ataraxis MicroControllers.

    BreakModule allows interfacing with a break to dynamically control the motion of break-coupled objects. The module
    is designed to send PWM signals that trigger Field-Effect-Transistor (FET) gated relay hardware to deliver voltage
    that variably engages the break. The module can be used to either fully engage or disengage the breaks or to output
    a PWM signal to engage the break with the desired strength.

    Args:
        module_id: The identifier code of the interfaced BreakModule instance.
        instance_name: The human-readable name of the interfaced BreakModule instance.
        minimum_break_strength: The minimum torque applied by the break in grams / centimeter. This is the torque the
            break delivers at minimum voltage (break is disabled).
        maximum_break_strength: The maximum torque applied by the break in grams / centimeter. This is the torque the
            break delivers at maximum voltage (break is fully engaged).
        object_diameter: The diameter of the objected coupled to the break, in centimeters. This is used to
            estimate the torque applied by the break to the edge of the object and to estimate breaking force for each
            pwm level.

    Attributes:
        _minimum_break_strength: The minimum torque the break delivers at minimum voltage (break is disabled).
        _maximum_break_strength: The maximum torque the break delivers at maximum voltage (break is fully engaged).
        _object_diameter: The diameter of the objected coupled to the break, in centimeters.
        _newton_per_gram_centimeter: Conversion factor from torque force in g/cm to N/cm.
        _torque_per_pwm: Conversion factor from break pwm levels to breaking force in N/cm.
    """

    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        minimum_break_strength: float = 43.2,  # 0.6 in iz
        maximum_break_strength: float = 1152.13,  # 16 in oz
        object_diameter: float = 15.0333,  # 0333 is to account for the wheel wrap
    ) -> None:
        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="BreakModule",
            module_type=np.uint8(3),
            module_id=module_id,
            instance_name=instance_name,
            unity_input_topics=None,
            output_data=False,
        )

        # Hardcodes the conversion factor used to translate torque force in g/cm to N/cm
        self._newton_per_gram_centimeter: float = 0.0000981

        # Stores additional data into class attributes. Rounds to 12 decimal places for consistency and to ensure
        # repeatability.
        self._minimum_break_strength: np.float64 = np.round(
            a=minimum_break_strength / self._newton_per_gram_centimeter,
            decimals=12,
        )
        self._maximum_break_strength: np.float64 = np.round(
            a=maximum_break_strength / self._newton_per_gram_centimeter,
            decimals=12,
        )
        self._object_diameter: float = object_diameter

        # Computes the conversion factor to translate break pwm levels into breaking force in Newtons per cm. Rounds
        # to 12 decimal places for consistency and to ensure repeatability.
        self._torque_per_pwm = np.round(
            a=(self._maximum_break_strength - self._minimum_break_strength) / 255,
            decimals=12,
        )

    def send_data(
        self,
        message: ModuleData | ModuleState,
        unity_communication: UnityCommunication,
        queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def log_variables(self) -> NDArray[np.uint8] | None:
        # Saves break data and conversion factors used to translate pwm values into torque force.
        output_array: NDArray[np.uint8] = np.array(
            [
                self._minimum_break_strength,
                self._maximum_break_strength,
                np.float64(self._object_diameter),
                np.float64(self._newton_per_gram_centimeter),
                self._torque_per_pwm,
            ],
            dtype=np.uint8,
        )
        return output_array

    def get_pwm_from_force(self, target_force_n_cm: float) -> np.uint8:
        """Converts the desired breaking force in Newtons to the required PWM value (0-255) to be delivered to the break
        hardware by the BreakModule.

        Use this method to convert the desired breaking force into the PWM value that can be submitted to the
        BreakModule via the set_parameters() class method.

        Args:
            target_force_n_cm: Desired force in Newtons per centimeter at the edge of the object.

        Returns:
            The byte PWM value that would generate the desired amount of torque.

        Raises:
            ValueError: If the input force is not within the valid range for the BreakModule.
        """
        if self._maximum_break_strength > target_force_n_cm or self._minimum_break_strength < target_force_n_cm:
            message = (
                f"The requested force {target_force_n_cm} N/cm is outside the valid range for the BreakModule "
                f"{self._instance_name} with id {self._module_id}. Valid breaking force range is from "
                f"{self._minimum_break_strength} to {self._maximum_break_strength}."
            )
            console.error(message=message, error=ValueError)

        # Calculates PWM using the pre-computed torque_per_pwm conversion factor
        pwm_value = np.uint8(round((target_force_n_cm - self._minimum_break_strength) / self._torque_per_pwm))

        return pwm_value

    def set_parameters(self, breaking_strength: np.uint8 = np.uint8(255)) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the BreakModule instance.

        Use this method to package and apply new PC-addressable parameters to the BreakModule instance managed by this
        Interface class.

        Notes:
            Use set_breaking_power() command to apply the breaking-strength transmitted in this parameter message to the
            break. Until the command is called, the new breaking_strength will not be applied to the break hardware.

        Args:
            breaking_strength: The Pulse-Width-Modulation (PWM) value to use when the BreakModule delivers adjustable
                breaking power. Depending on this value, the breaking power can be adjusted from none (0) to maximum
                (255). Use get_pwm_from_force() to translate desired breaking force into the required PWM value.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(breaking_strength,),
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Triggers the BreakModule to be permanently engaged at maximum strength or permanently disengaged.

        This command locks the BreakModule managed by this Interface into the desired state.

        Notes:
            This command does NOT use the breaking_strength parameter and always uses either maximum or minimum breaking
            power. To set the break to a specific torque level, set the level via the set_parameters() method and then
            switch the break into the variable torque mode by using the set_breaking_power() method.

        Args:
            state: The desired state of the break. True means the break is engaged; False means the break is disengaged.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1 if state else 2),
            noblock=np.bool(False),
        )

    def set_breaking_power(self) -> OneOffModuleCommand:
        """Triggers the BreakModule to engage with the strength (torque) defined by the breaking_strength runtime
        parameter.

        Unlike the toggle() method, this method allows precisely controlling the torque applied by the break. This
        is achieved by pulsing the break control pin at the PWM level specified by breaking_strength runtime parameter
        stored in BreakModule's memory (on the microcontroller).

        Notes:
            This command switches the break to run in the variable strength mode and applies the current value of the
            breaking_strength parameter to the break, but it does not determine the breaking power. To adjust the power,
            use the set_parameters() class method to issue updated breaking_strength value. By default, the break power
            is set to 50% (PWM value 128).

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )


class ValveInterface(ModuleInterface):
    """Interfaces with ValveModule instances running on Ataraxis MicroControllers.

    ValveModule allows interfacing with a solenoid valve to controllably dispense precise amounts of fluid. The module
    is designed to send digital signals that trigger Field-Effect-Transistor (FET) gated relay hardware to deliver
    voltage that opens or closes the controlled valve. The module can be used to either permanently open or close the
    valve or to cycle opening and closing in a way that ensures a specific amount of fluid passes through the
    valve.

    Notes:
        This interface comes pre-configured to receive valve pulse triggers from Unity. If you do not need this
        functionality, set the input_unity_topics argument to None when initializing the class!

    Args:
        module_id: The unique identifier code of the managed ValveModule instance.
        instance_name: The human-readable name of the ValveModule instance.
        valve_calibration_data: A tuple of tuples that contains the data required to map pulse duration to delivered
            fluid volume. Each sub-tuple should contain the integer that specifies the pulse duration in microseconds
            and a float that specifies the delivered fluid volume in microliters. If you do not know this data,
            initialize the class using a placeholder calibration tuple and use calibration() class method to collect
            this data using the ValveModule.
        input_unity_topics: A tuple of Unity topics that the module should monitor to receive activation triggers.

    Attributes:
        _microliters_per_microsecond: The conversion factor from desired fluid volume in microliters to the pulse
            valve duration in microseconds.
    """

    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        valve_calibration_data: tuple[tuple[int, float], ...],
        input_unity_topics: tuple[str, ...] | None = ("Gimbl/Reward/",),
    ) -> None:
        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="ValveModule",
            module_type=np.uint8(5),
            module_id=module_id,
            instance_name=instance_name,
            unity_input_topics=input_unity_topics,
            output_data=False,
        )

        # Extracts pulse durations and fluid volumes into separate arrays
        pulse_durations: NDArray[np.float64] = np.array([x[0] for x in valve_calibration_data], dtype=np.float64)
        fluid_volumes: NDArray[np.float64] = np.array([x[1] for x in valve_calibration_data], dtype=np.float64)
        # noinspection PyTypeChecker

        # Computes the conversion factor by finding the slope of the calibration curve.
        # Rounds to 12 decimal places for consistency and to ensure repeatability
        slope: np.float64 = polyfit(x=pulse_durations, y=fluid_volumes, deg=1)[0]  # type: ignore
        self._microliters_per_microsecond: np.float64 = np.round(a=slope, decimals=12)

    def send_data(
        self,
        message: ModuleData | ModuleState,
        unity_communication: UnityCommunication,
        queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> OneOffModuleCommand:
        # Currently, the only processed topic is "Gimbl/Reward/". If more supported topics are needed, this needs to be
        # rewritten to use if-else conditions.

        # If the received message was sent to the reward topic, this is a binary (empty payload) trigger to
        # pulse the valve. It is expected that the valve parameters are configured so that this delivers the
        # desired amount of water reward.
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),  # Blocks to ensure reward delivery precision.
        )

    def log_variables(self) -> NDArray[np.uint8] | None:
        # Saves the valve calibration factor
        output_array: NDArray[np.uint8] = np.array(
            [self._microliters_per_microsecond],
            dtype=np.uint8,
        )
        return output_array

    def get_duration_from_volume(self, volume: float) -> np.uint32:
        """Converts the desired fluid volume in microliters to the valve pulse duration in microseconds that ValveModule
        will use to deliver that fluid volume.

        Use this method to convert the desired fluid volume into the pulse_duration value that can be submitted to the
        ValveModule via the set_parameters() class method.

        Args:
            volume: Desired fluid volume in microliters.

        Returns:
            The microsecond pulse duration that would be used to deliver the specified volume.
        """
        return np.uint32(np.round(volume / self._microliters_per_microsecond))

    def set_parameters(
        self,
        pulse_duration: np.uint32 = np.uint32(10000),
        calibration_delay: np.uint32 = np.uint32(10000),
        calibration_count: np.uint16 = np.uint16(100),
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the ValveModule instance.

        Use this method to package and apply new PC-addressable parameters to the ValveModule instance managed by this
        Interface class.

        Args:
            pulse_duration: The time, in microseconds, the valve stays open when it is pulsed (opened and closed). This
                is used during the execution of send_pulse() command to control the amount of dispensed fluid. Use
                get_duration_from_volume() method to convert the desired fluid volume into the pulse_duration value.
            calibration_delay: The time, in microseconds, to wait between consecutive pulses during calibration.
                Calibration works by repeatedly pulsing the valve the requested number of times. Delaying after closing
                the valve (ending the pulse) ensures the valve hardware has enough time to respond to the inactivation
                phase before starting the next calibration cycle.
            calibration_count: The number of times to pulse the valve during calibration. A number between 10 and 100 is
                enough for most use cases.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(pulse_duration, calibration_delay, calibration_count),
        )

    def send_pulse(
        self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = False
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        """Triggers ValveModule to deliver a precise amount of fluid by cycling opening and closing the valve once or
        repetitively (recurrently).

        After calibration, this command allows delivering precise amounts of fluid with, depending on the used valve and
        relay hardware microliter or nanoliter precision. This command is optimized to change valve states at a
        comparatively low frequency in the 10-200 Hz range.

        Notes:
            To ensure the accuracy of fluid delivery, it is recommended to run the valve in the blocking mode
            and, if possible, isolate it to a controller that is not busy with running other tasks.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
                will only run once. The exact repetition delay will be further affected by other modules managed by the
                same microcontroller and may not be perfectly accurate.
            noblock: Determines whether the command should block the microcontroller while the valve is kept open or
                not. Blocking ensures precise pulse duration and, by extension, delivered fluid volume. Non-blocking
                allows the microcontroller to perform other operations while waiting, increasing its throughput.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(noblock),
            cycle_delay=repetition_delay,
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Triggers the ValveModule to be permanently open or closed.

        This command locks the ValveModule managed by this Interface into the desired state.

        Args:
            state: The desired state of the valve. True means the valve is open; False means the valve is closed.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=np.bool(False),
        )

    def calibrate(self) -> OneOffModuleCommand:
        """Triggers ValveModule to repeatedly pulse the valve using the duration defined by the pulse_duration runtime
        parameter.

        This command is used to build the calibration map of the valve that matches pulse_duration to the volume of
        fluid dispensed during the time the valve is open. To do so, the command repeatedly pulses the valve to dispense
        a large volume of fluid which can be measured and averaged to get the volume of fluid delivered during each
        pulse. The number of pulses carried out during this command is specified by the calibration_count parameter, and
        the delay between pulses is specified by the calibration_delay parameter.

        Notes:
            When activated, this command will block in-place until the calibration cycle is completed. Currently, there
            is no way to interrupt the command, and it may take a prolonged period of time (minutes) to complete.

            This command does not set any of the parameters involved in the calibration process. Make sure the
            parameters are submitted to the ValveModule's hardware memory via the set_parameters() class method before
            running the calibration() command.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=np.bool(False),
        )


class LickInterface(ModuleInterface):
    """The class that exposes methods for interfacing with SensorModule instances running on Ataraxis MicroControllers.

    SensorModule facilitates receiving data recorded by any sensor that outputs analog unidirectional logic signals.
    This class functions similar to the TTLModule, but is specialized for working with analog signals. If you need to
    monitor a binary digital signal, use TTLModule instead for higher efficiency.

    Notes:
        This interface comes pre-configured to send triggers to Unity when it receives sensor data. If you do not need
        this functionality, override the unity_output flag at class instantiation!

    Args:
        module_id: The unique identifier code of the managed SensorModule instance.
        instance_name: The human-readable name of the SensorModule instance.
        output_data: Determines whether the SensorModule instance should send triggers to Unity when it receives sensor
            data.
        sensor_topic: The MQTT topic to which the instance should send the triggers based on the received sensor data.

    Attributes:
        _sensor_topic: Stores the output MQTT topic.
    """

    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        output_data: bool = True,
        sensor_topic: str = "LickPort/",
    ) -> None:
        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="SensorModule",
            module_type=np.uint8(4),
            module_id=module_id,
            instance_name=instance_name,
            unity_input_topics=None,
            output_data=output_data,
        )

        self._sensor_topic: str = sensor_topic

    def send_data(
        self,
        message: ModuleData | ModuleState,
        unity_communication: UnityCommunication,
        mp_queue: MPQueue,  # type: ignore
    ) -> None:
        # If the incoming message is not reporting a change in signal (code 51), aborts processing
        if message.event != np.uint8(51):
            return

        # Sends an empty message to the sensor MQTT topic, which acts as a binary trigger.
        unity_communication.send_data(topic=self._sensor_topic, payload=None)

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def log_variables(self) -> NDArray[np.uint8] | None:
        """Not used."""
        return None

    def set_parameters(
        self,
        lower_threshold: np.uint16 = np.uint16(0),
        upper_threshold: np.uint16 = np.uint16(65535),
        delta_threshold: np.uint16 = np.uint16(1),
        averaging_pool_size: np.uint8 = np.uint8(0),
    ) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Mostly, these parameters are used to filter incoming signals to minimize the number of messages sent to the PC.

        Notes:
            All threshold parameters are inclusive!

        Args:
            lower_threshold: The minimum signal level, in raw analog units, to be considered valid. Setting this
                threshold to a number above zero allows high-pass filtering the incoming signals.
            upper_threshold: The maximum signal level, in raw analog units, to be considered valid. Setting this
                threshold to a number below 65535 allows low-pass filtering the incoming signals. Note, due to the
                typically used analog readout resolution of 10-14 bytes, the practical ceiling of detected signals will
                likely not exceed 1000-5000 analog units.
            delta_threshold: The minimum value by which the signal has to change, relative to the previous check, for
                the change to be reported to the PC. Note, if the change is 0, the signal will not be reported to the
                PC, regardless of this parameter value.
            averaging_pool_size: The number of analog pin readouts to average together when checking pin state. This
                is used to smooth the recorded signal and eliminate analog communication noise. It is highly advised to
                have this enabled and set to at least 10 readouts.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(upper_threshold, lower_threshold, delta_threshold, averaging_pool_size),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the state of the analog pin managed by the module.

        This command will evaluate the signal received by the analog pin and, if it is significantly different from the
        signal recorded during a previous check, report it to the PC. This approach ensures that only significant
        changes are communicated to the PC, preserving communication bandwidth. It is highly advised to issue this
        command to repeat (recur) at a desired interval to continuously monitor the pin state, rather than repeatedly
        calling it as a one-off command for best runtime efficiency.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If this is set to 0,
                the command will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message to be sent to the microcontroller.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )


# class TorqueInterface(ModuleInterface):
#     pass
