"""Provides the serial and MQTT communication classes alongside the message protocol and data prototype definitions used
to exchange commands and data between host-machines (PCs) and Arduino / Teensy microcontrollers.
"""

from .mqtt import MQTTCommunication
from .serial import SerialCommunication
from .messages import (
    KernelData,
    ModuleData,
    KernelState,
    ModuleState,
    KernelCommand,
    ReceptionCode,
    ModuleParameters,
    OneOffModuleCommand,
    DequeueModuleCommand,
    ModuleIdentification,
    RepeatedModuleCommand,
    ControllerIdentification,
)
from .protocols import PrototypeType, SerialProtocols, SerialPrototypes

__all__ = [
    "ControllerIdentification",
    "DequeueModuleCommand",
    "KernelCommand",
    "KernelData",
    "KernelState",
    "MQTTCommunication",
    "ModuleData",
    "ModuleIdentification",
    "ModuleParameters",
    "ModuleState",
    "OneOffModuleCommand",
    "PrototypeType",
    "ReceptionCode",
    "RepeatedModuleCommand",
    "SerialCommunication",
    "SerialProtocols",
    "SerialPrototypes",
]
