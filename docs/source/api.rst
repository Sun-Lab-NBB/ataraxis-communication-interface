.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Serial Protocols and Prototypes
===============================
.. automodule:: ataraxis_communication_interface.communication.protocols
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PrototypeType

Communication Messages
======================
.. automodule:: ataraxis_communication_interface.communication.messages
   :members:
   :undoc-members:
   :show-inheritance:

Serial Communication
====================
.. automodule:: ataraxis_communication_interface.communication.serial
   :members:
   :undoc-members:
   :show-inheritance:

MQTT Communication
==================
.. automodule:: ataraxis_communication_interface.communication.mqtt
   :members:
   :undoc-members:
   :show-inheritance:

MicroController Interface
=========================
.. automodule:: ataraxis_communication_interface.microcontroller.interface
   :members:
   :undoc-members:
   :show-inheritance:

Dataclasses
===========
.. automodule:: ataraxis_communication_interface.microcontroller.dataclasses
   :members:
   :undoc-members:
   :show-inheritance:

Log Processing
==============
.. automodule:: ataraxis_communication_interface.microcontroller.log_processing
   :members:
   :undoc-members:
   :show-inheritance:

CLI
===
.. click:: ataraxis_communication_interface.interfaces.cli:axci_cli
   :prog: axci
   :nested: full
