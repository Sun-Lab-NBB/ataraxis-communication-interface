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
.. autodata:: ataraxis_communication_interface.orchestration.allocation.PARALLEL_EXTRACTION_THRESHOLD
.. autodata:: ataraxis_communication_interface.orchestration.allocation.SPAWNED_CHILD_MEMORY_MB

CLI
===

.. click:: ataraxis_communication_interface.interfaces.cli:axci_cli
   :prog: axci
   :nested: full
