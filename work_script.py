from pathlib import Path

import numpy as np
from ataraxis_data_structures import DataLogger
from ataraxis_time.precision_timer.timer_class import PrecisionTimer

from ataraxis_transport_layer import EncoderInterface, MicroControllerInterface, list_available_ports

print(list_available_ports())

# timeout = PrecisionTimer(precision="s")
# if __name__ == "__main__":
#     # Initializes and starts the logger
#     output_directory = Path("/home/cybermouse/Desktop/TestLog")
#
#     logger = DataLogger(output_directory)
#     logger._vacate_shared_memory_buffer()
#     logger.start()
#
#     # Initializes the tested module interface
#     module = EncoderInterface(module_id=np.uint8(1), instance_name="TestEncoder")
#
#     # Initializes and starts the microcontroller interface. Provides it with the tested module instance initialized
#     # above.
#     interface = MicroControllerInterface(
#         controller_id=np.uint8(123),
#         controller_name="TestController",
#         data_logger=logger,
#         modules=(module,),
#         controller_usb_port="/dev/ttyACM0",
#         baudrate=115200,
#         maximum_transmitted_payload_size=254,
#         unity_broker_ip="127.0.0.1",
#         unity_broker_port=1883,
#         verbose=True,
#     )
#     interface.vacate_shared_memory_buffer()
#     interface.start()
#     interface.unlock_controller()
#
#     # parameters
#     encoder_params = module.set_parameters(np.bool(True), np.bool(False), np.uint32(10))
#     interface.send_message(encoder_params)
#
#     # Sends out the tested command
#     check_command = module.check_state(repetition_delay=np.uint32(1000))
#     # noinspection PyTypeChecker
#     interface.send_message(check_command)
#
#     # calibrate = module.get_ppr()
#     # interface.send_message(calibrate)
#
#     # Statically blocks for 20 seconds while running recurrent commands.
#     timeout.delay_noblock(delay=60, allow_sleep=True)
#
#     # Shuts down the communication interface and the logger
#     interface.stop()
#     logger.stop()
#
#     # Compresses and lists the logged data. This is done mostly to verify the integrity of the logs by printing them to
#     # terminal.
#     logger.compress_logs(remove_sources=True, verbose=True)
