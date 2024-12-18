# Configuration file for the Sphinx documentation builder.
import importlib_metadata

# -- Project information -----------------------------------------------------
project = 'YOUR-PROJECT-NAME'
# noinspection PyShadowingBuiltins
copyright = '2024, YOUR_AUTHOR_NAME & Sun Lab'
author = 'YOUR_AUTHOR_NAME'
release = importlib_metadata.version("YOUR-PROJECT-NAME")  # Extracts project version from the metadata .toml file.

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # To build documentation from python source code docstrings.
    'sphinx.ext.napoleon',  # To read google-style docstrings (works with autodoc module).
    'sphinx_rtd_theme',  # To format the documentation html using ReadTheDocs format.
    'sphinx_click'  # To read docstrings and command-line arguments from click-wrapped python functions.
]

templates_path = ['_templates']
exclude_patterns = []

# Google-style docstring parsing configuration for napoleon extension
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Directs sphinx to use RTD theme
