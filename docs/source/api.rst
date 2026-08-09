.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Communication
=============

.. automodule:: ataraxis_communication_interface.communication
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PrototypeType

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

Orchestration
=============

.. automodule:: ataraxis_communication_interface.orchestration
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the MicroController autodata directives.
.. autodata:: ataraxis_communication_interface.orchestration.jobs.EXTRACTION_JOB_NAME
.. autodata:: ataraxis_communication_interface.orchestration.jobs.TRACKER_FILENAME
.. autodata:: ataraxis_communication_interface.orchestration.jobs.MICROCONTROLLER_DATA_DIRECTORY
.. autodata:: ataraxis_communication_interface.orchestration.jobs.CONTROLLER_FEATHER_PREFIX
.. autodata:: ataraxis_communication_interface.orchestration.jobs.MODULE_FEATHER_INFIX
.. autodata:: ataraxis_communication_interface.orchestration.jobs.KERNEL_FEATHER_INFIX
.. autodata:: ataraxis_communication_interface.orchestration.jobs.FEATHER_SUFFIX
.. autodata:: ataraxis_communication_interface.orchestration.allocation.RESERVED_CORES
.. autodata:: ataraxis_communication_interface.orchestration.allocation.EXTRACTION_JOB_CORES

CLI
===
.. click:: ataraxis_communication_interface.interfaces.cli:axci_cli
   :prog: axci
   :nested: full
