.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (autodoc, etc.).

.. Use the examples below to structure your API documentation. Section names are used as sub-directories in the
   index menu. Use appropriate module and automodule extensions as needed to document your code.

Python Module
=============

.. automodule:: YOUR_LIBRARY_NAME.module_name
   :members:
   :undoc-members:
   :show-inheritance:

CLI Interface
===============

.. automodule:: YOUR_LIBRARY_NAME.module_name
   :members:
   :undoc-members:
   :show-inheritance:

.. Nesting allows documenting click-options in-addition to the main function docstring. Interface_function is the name
   of the cli-wrapped function. Interface-name can be the same as function name or the shorthand name defined through
   pyproject.toml scripts section.
.. click:: YOUR_LIBRARY_NAME.module_name:interface_function
   :prog: interface-name
   :nested: full
