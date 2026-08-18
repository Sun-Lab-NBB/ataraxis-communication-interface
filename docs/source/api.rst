.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Communication
=============

.. automodule:: ataraxis_communication_interface.communication
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PrototypeType

.. Documents the message data type alias by hand, and excludes it from the automodule directive above, because autodoc
   renders a PEP 695 alias by evaluating its deferred value. That value names NDArray, which this library imports only
   under TYPE_CHECKING to keep it off the runtime import path, so letting autodoc reach it raises a NameError and
   fails the build. The canonical union and the description below therefore mirror the alias and its docstring in
   communication/protocols.py by hand, and both move whenever that alias does.
.. py:type:: PrototypeType
   :module: ataraxis_communication_interface
   :canonical: numpy.bool_ | numpy.uint8 | numpy.int8 | numpy.uint16 | numpy.int16 | numpy.uint32 | numpy.int32 | numpy.uint64 | numpy.int64 | numpy.float32 | numpy.float64 | NDArray[numpy.bool_] | NDArray[numpy.uint8] | NDArray[numpy.int8] | NDArray[numpy.uint16] | NDArray[numpy.int16] | NDArray[numpy.uint32] | NDArray[numpy.int32] | NDArray[numpy.uint64] | NDArray[numpy.int64] | NDArray[numpy.float32] | NDArray[numpy.float64]

   The union of every data object type this library can transmit to and receive from a microcontroller. Each
   serialized prototype code resolves to one member of this union, so the data object of any received ModuleData or
   KernelData message is an instance of one of these types.

MicroController
===============

.. automodule:: ataraxis_communication_interface.microcontroller
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, since the automodule directive above discovers module-level data through
   the source of the module it documents and therefore skips a constant this package re-exports. The directive names
   the defining module rather than the package, because autodoc reads the attribute docstring from that module's
   source and falls back to the docstring of the value's own type when it is pointed at the re-exporting package.
.. autodata:: ataraxis_communication_interface.microcontroller.dataclasses.MICROCONTROLLER_MANIFEST_FILENAME
.. autodata:: ataraxis_communication_interface.microcontroller.dataclasses.EXTRACTION_CONFIGURATION_FILENAME
.. autodata:: ataraxis_communication_interface.microcontroller.status_codes.MINIMUM_CUSTOM_STATUS_CODE
.. autodata:: ataraxis_communication_interface.microcontroller.status_codes.MAXIMUM_CUSTOM_STATUS_CODE

Orchestration
=============

.. automodule:: ataraxis_communication_interface.orchestration
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the MicroController autodata directives.
.. autodata:: ataraxis_communication_interface.orchestration.jobs.CONTROLLER_EXTRACTION_JOB_NAME
.. autodata:: ataraxis_communication_interface.orchestration.allocation.CONTROLLER_EXTRACTION_JOB_CORES

CLI
===

.. click:: ataraxis_communication_interface.interfaces.cli:axci_cli
   :prog: axci
   :nested: full
