# Sun Lab Python Style Guide

This guide defines the documentation and coding conventions used across Sun Lab Python projects.
Reference this during development to maintain consistency across all codebases.

---

## Docstrings

Use **Google-style docstrings** with the following sections (in order):

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """Brief one-line summary of what the function does.

    Extended description goes here if needed. This is optional and should only be used
    when the function's behavior is too complex or nuanced to be fully explained by
    the one-line summary. Most functions should not need this section.

    Notes:
        Additional context, background, or implementation details. Use this for
        explaining algorithms, referencing papers, or clarifying non-obvious behavior.
        Multi-sentence explanations go here.

    Args:
        param1: Description without repeating the type (types are in signature).
        param2: Description of parameter with default value behavior if relevant.

    Returns:
        Description of return value. For simple returns, one line is sufficient.
        For complex returns (tuples, dicts), describe each element.

    Raises:
        ValueError: When this error occurs and why.
        TypeError: When this error occurs and why.
    """
```

### General Rules

**Punctuation**: Always use proper punctuation in all documentation: docstrings, comments, and
argument descriptions.

### Section Guidelines

**Summary line**: Use imperative mood for ALL docstrings (functions, methods, classes, modules).
Use verbs like "Computes...", "Defines...", "Configures...", "Processes..." rather than noun
phrases like "A class that..." or "Configuration for...".

**Extended description**: Only include if the summary line is not sufficient to fully understand
what the function does. Use third person ("This method creates..."). Do not explain every step of
implementation; focus on the high-level purpose.

**Notes**: Use for usage guidance, non-obvious behavior, algorithms, references, and implementation
rationale. This section explains how the function is used or unique things to know about it, not
what it does.

**Args**: Don't repeat type info. Start with uppercase after the colon. Always use proper
punctuation. Descriptions must be clear enough to give the user a good understanding of what the
argument controls or does.

**Args (boolean)**: Use "Determines whether..." not "Whether..." for boolean parameters.

**Args (enum)**: Type hint should accept both the enum and its base type (e.g.,
`VisualizerMode | int` for IntEnum). Include "Must be a valid X enumeration member." at the end
of the description.

**Returns**: Describe what is returned, not the type. Start with uppercase if a sentence. For
complex returns (tuples, dicts), describe each element in prose form.

**Raises**: Only include if the function explicitly raises exceptions.

**Attributes**: Document all instance attributes, including private ones prefixed with `_`.

**Lists**: Do not use lists (numbered or bulleted) in docstrings. Write information in prose form
instead.

### Class Docstrings with Attributes

For classes, include an Attributes section listing all instance attributes:

```python
class DataProcessor:
    """Processes experimental data for analysis.

    Args:
        data_path: Path to the input data file.
        sampling_rate: The sampling rate in Hz.
        enable_filtering: Determines whether to apply bandpass filtering.

    Attributes:
        _data_path: Cached path to input data.
        _sampling_rate: Cached sampling rate parameter.
        _enable_filtering: Cached filtering flag.
        _processed_data: Dictionary storing processed results.
    """
```

### Enum and Dataclass Attributes

For enums and dataclasses, document each attribute inline using triple-quoted strings:

```python
class VisualizerMode(IntEnum):
    """Defines the display modes for the BehaviorVisualizer."""

    LICK_TRAINING = 0
    """Displays only lick sensor and valve plots."""
    RUN_TRAINING = 1
    """Displays lick, valve, and running speed plots."""
    EXPERIMENT = 2
    """Displays all plots including the trial performance panel."""


@dataclass
class SessionConfig:
    """Defines the configuration parameters for an experiment session."""

    animal_id: str
    """The unique identifier for the animal."""
    session_duration: float
    """The duration of the session in seconds."""
```

### Property Docstrings

```python
@property
def field_shape(self) -> tuple[int, int]:
    """Returns the shape of the data field as (height, width)."""
    return self._field_shape
```

### Module Docstrings

Follow the same imperative mood pattern as other docstrings:

```python
"""Provides assets for processing and analyzing neural imaging data."""
```

### CLI Command Docstrings

CLI commands and command groups use a specialized docstring format because Click parses these
docstrings into help messages displayed to users. Do not use standard docstring sections (Notes,
Args, Returns, Raises) as they will appear verbatim in the CLI help output.

```python
@click.command()
def process_data(input_path: Path, output_path: Path) -> None:
    """Processes raw experimental data and saves the results.

    This command reads data from the input path, applies standard preprocessing
    steps, and writes the processed output to the specified location. The input
    must be a valid .feather file containing session data.
    """
```

The first sentence serves as the short command description shown in command listings. The
remaining prose provides additional context shown in the detailed help for that specific command.

### MCP Server Tool Docstrings

MCP (Model Context Protocol) server tools expose library functionality to AI agents. Tool
docstrings serve dual purposes: they document the function for developers and provide instructions
to AI agents that call the tools. Use standard Google-style docstrings with additional sections
for agent guidance.

```python
@mcp.tool()
def start_video_session(
    output_directory: str,
    frame_rate: int = 30,
    display_frame_rate: int | None = 25,
) -> str:
    """Starts a video capture session with the specified parameters.

    Creates a VideoSystem instance and begins acquiring frames from the camera. Frames are not
    saved until start_frame_saving is called. Only one session can be active at a time.

    Important:
        The AI agent calling this tool MUST ask the user to provide the output_directory path
        before calling this tool. Do not assume or guess the output directory - always prompt
        the user for an explicit path.

    Args:
        output_directory: The path to the directory where video files will be saved. This must
            be provided by the user - the AI agent should always ask for this value explicitly.
        frame_rate: The target frame rate in frames per second. Defaults to 30.
        display_frame_rate: The rate at which to display acquired frames in a preview window.
            Defaults to 25 fps. Set to None to disable frame display.
    """
```

**Important section**: Use this section to provide explicit instructions to AI agents about how
to use the tool. Common guidance includes:

- Parameters that must be obtained from the user (not assumed or guessed)
- Prerequisites that must be checked before calling the tool
- Constraints on when or how the tool should be called

### MCP Server Response Formatting

MCP tool responses should be concise and information-dense. Avoid verbose multi-line formatting
with bullet points. Prefer single-line responses that pack relevant information efficiently.

```python
# Good - concise, information-dense
return f"Session started: {interface} #{camera_index} {width}x{height}@{frame_rate}fps -> {output_directory}"

# Avoid - verbose multi-line formatting
return (
    f"Video Session Started\n"
    f"• Interface: {interface}\n"
    f"• Camera: {camera_index}\n"
    f"• Resolution: {width}x{height} px\n"
    f"• Frame rate: {frame_rate} fps"
)
```

**Formatting conventions**:

- **Concise output**: Keep responses to a single line when possible
- **Key-value pairs**: Use `Key: value` format with `|` separators for multiple items
- **Errors**: Start with "Error:" followed by a brief description
- **Technical notation**: Use standard notation like `640x480@30fps` for resolution/framerate

```python
# Status with multiple properties
return f"FFMPEG: {ffmpeg_status} | GPU: {gpu_status} | CTI: {cti_status}"

# Success response
return "Recording started"

# Error response
return f"Error: Directory not found: {output_directory}"

# Camera listing (one per line for multiple items)
return f"OpenCV #{cam.camera_index}: {cam.frame_width}x{cam.frame_height}@{cam.acquisition_frame_rate}fps"
```

### Example Script Docstrings

Example scripts (files in the `examples/` directory) are an exception to the imperative mood rule.
These scripts use a descriptive format that starts with "This example script..." to clearly
identify their purpose to users browsing the examples.

```python
"""This example script demonstrates how to use the library to record and display frames from a
webcam.

See API documentation at https://example.com for additional configuration options.
Authors: Name1, Name2
"""
```

---

## Type Annotations

### General Rules

- All function parameters and return types must have annotations
- Use `-> None` for functions that don't return a value
- Use `| None` for optional types (not `Optional[T]`)
- Use lowercase `tuple`, `list`, `dict` (not `Tuple`, `List`, `Dict`)
- Avoid `any` type; use explicit union types instead

### NumPy Arrays

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

def process(data: NDArray[np.float32]) -> NDArray[np.float32]:
    ...
```

- Always specify dtype explicitly: `NDArray[np.float32]`, `NDArray[np.uint16]`,
  `NDArray[np.bool_]`, etc.
- Never use unparameterized `NDArray`
- Use `TYPE_CHECKING` block for `NDArray` to avoid runtime import overhead

### Class Attributes

```python
def __init__(self, height: int, width: int) -> None:
    self._field_shape: tuple[int, int] = (height, width)
    self._data: tuple[NDArray[np.float32], NDArray[np.float32]] = (
        np.zeros(self._shape, dtype=np.float32),
        np.zeros(self._shape, dtype=np.float32),
    )
```

---

## Naming Conventions

### Variables

Use **full words**, not abbreviations:

| Avoid             | Prefer                              |
|-------------------|-------------------------------------|
| `t`, `t_sq`       | `interpolation_factor`, `t_squared` |
| `coeff`, `coeffs` | `coefficient`, `coefficients`       |
| `pos`, `idx`      | `position`, `index`                 |
| `img`, `val`      | `image`, `value`                    |
| `num`, `dnum`     | `numerator`, `denominator`          |
| `gy`, `gx`        | `grid_index_y`, `grid_index_x`      |

### Functions

- Use descriptive verb phrases: `compute_coefficients`, `extract_features`
- Private functions start with underscore: `_process_batch`, `_validate_input`
- Avoid generic names like `process`, `handle`, `do_something`

### Constants

Module-level constants with type annotations and descriptive names:

```python
# Minimum number of samples required for statistical validity.
_MINIMUM_SAMPLE_COUNT: int = 100
```

---

## Function Calls

**Always use keyword arguments** for clarity:

```python
# Good
np.zeros((4,), dtype=np.float32)
np.empty((4,), dtype=np.float32)
compute_coefficients(interpolation_factor=t, output=result)
self._get_data(dimension=0)

# Avoid
np.zeros((4,), np.float32)
compute_coefficients(t, result)
self._get_data(0)
```

Exception: Single positional arguments for obvious cases like `range(4)`, `len(array)`.

---

## Error Handling

Use `console.error` from `ataraxis_base_utilities`:

```python
from ataraxis_base_utilities import console

def process_data(self, data: NDArray[np.float32], threshold: float) -> None:
    if not (0 < threshold <= 1):
        message = (
            f"Unable to process data with the given threshold. The threshold must be in range "
            f"(0, 1], but got {threshold}."
        )
        console.error(message=message, error=ValueError)
```

### Error Message Format

- Start with context: "Unable to [action] using [input]."
- Explain the constraint: "The [parameter] must be [constraint]"
- Show actual value: "but got {value}."
- Use f-strings for interpolation

---

## Numba Functions

### Decorator Patterns

```python
# Standard cached function
@numba.njit(cache=True)
def _compute_values(...) -> None:
    ...

# Parallelized function
@numba.njit(cache=True, parallel=True)
def _process_batch(...) -> None:
    for i in prange(data.shape[0]):  # Parallel outer loop
        for j in range(data.shape[1]):  # Sequential inner loop
            ...

# Inlined helper (for small, frequently-called functions)
@numba.njit(cache=True, inline="always")
def compute_coefficients(...) -> None:
    ...
```

### Guidelines

- Always use `cache=True` for disk caching (avoids recompilation)
- Use `parallel=True` with `prange` only when no race conditions exist
- Use `inline="always"` for small helper functions called in hot loops
- Don't use `nogil` unless explicitly using threading
- Use Python type hints (not Numba signature strings) for readability

### Variable Allocation in Parallel Loops

```python
for i in prange(data.shape[0]):
    # Allocate per-thread arrays INSIDE the parallel loop
    temp_y = np.empty((4,), dtype=np.float32)
    temp_x = np.empty((4,), dtype=np.float32)

    for j in range(data.shape[1]):
        ...
```

---

## Comments

### Inline Comments

- Use third person imperative ("Configures..." not "This section configures..." or "Configure...")
- Place above the code, not at end of line (unless very short)
- Use comments to explain non-obvious logic or provide context that aids understanding
- When explaining what code does, focus on the high-level purpose, not obvious implementation
  details

```python
# The constant 2.046392675 is the theoretical injectivity bound for 2D cubic B-splines.
# Values exceeding 1/K of the grid spacing can cause non-injective (folded) transformations.
limit = (1.0 / 2.046392675) * self._grid_sampling * factor

# Configures the speed axis, which only exists in RUN_TRAINING and experiment modes.
if self._speed_axis is not None:
    ...

# Creates the reinforcing trial rectangles in the bottom row.
for i in range(20):
    ...
```

### What to Avoid

- Don't reiterate the obvious (e.g., `# Set x to 5` before `x = 5`)
- Don't add docstrings/comments to code you didn't write or modify
- Don't add type annotations as comments (use actual type hints)

---

## Imports

### Organization

```python
"""Module docstring."""

from typing import TYPE_CHECKING

import numba
from numba import prange
import numpy as np
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from numpy.typing import NDArray
```

Order:
1. Future imports (if any)
2. Standard library
3. `TYPE_CHECKING` import from typing
4. Third-party imports (alphabetical)
5. Local imports
6. `if TYPE_CHECKING:` block for type-only imports

---

## Class Design

### Constructor Parameters

Use explicit parameters instead of tuples/dicts:

```python
# Good
def __init__(self, field_height: int, field_width: int, sampling: float) -> None:
    self._field_shape: tuple[int, int] = (field_height, field_width)

# Avoid
def __init__(self, field_shape: tuple[int, int], sampling: float) -> None:
    self._field_shape = field_shape
```

### Properties vs Methods

- Use `@property` for simple attribute access that may involve computation
- Use methods for operations that clearly "do something" or take parameters

```python
@property
def data_shape(self) -> tuple[int, int]:
    """Returns the shape of the data as (height, width)."""
    ...

def set_from_array(self, data: NDArray[np.float32], weights: NDArray[np.float32]) -> None:
    """Sets internal state from the provided arrays."""
    ...
```

### Method Types

Use the appropriate method decorator based on what the method accesses:

- **Instance methods** (no decorator): Use when the method accesses instance attributes (`self`).
- **`@staticmethod`**: Use when the method doesn't access instance or class attributes. Prefer
  this over instance methods when `self` is not needed.
- **`@classmethod`**: Use when the method needs access to class attributes but not instance
  attributes.

```python
class DataProcessor:
    _default_threshold: float = 0.5  # Class attribute.

    def process(self, data: NDArray[np.float32]) -> NDArray[np.float32]:
        """Processes data using instance configuration."""
        return data * self._scale_factor  # Uses self.

    @staticmethod
    def validate_input(data: NDArray[np.float32]) -> bool:
        """Validates input data format."""
        return data.ndim == 2  # No self or cls needed.

    @classmethod
    def from_config(cls, config: dict) -> "DataProcessor":
        """Creates an instance from a configuration dictionary."""
        return cls(threshold=config.get("threshold", cls._default_threshold))
```

### Visibility (Public vs Private)

- **Private** (`_` prefix): Use for anything internal to the class/module that should not be
  accessed externally. This includes helper methods, internal attributes, and implementation
  details.
- **Public** (no prefix): Use only for methods, functions, and classes that are intended to be
  used from other modules or by external code.

```python
class SessionManager:
    def __init__(self) -> None:
        self._session_id: str = ""  # Private attribute.
        self._is_active: bool = False  # Private attribute.

    def start_session(self) -> None:
        """Starts a new session."""  # Public - called externally.
        self._initialize_resources()
        self._is_active = True

    def _initialize_resources(self) -> None:
        """Initializes internal resources."""  # Private - internal helper.
        ...
```

---

## Line Length and Formatting

- Maximum line length: 120 characters
- Break long function calls across multiple lines:

```python
result = compute_transformation(
    input_data=self._data,
    parameters=self._get_parameters(dimension=dimension),
    weights=weights,
)
```

- Use parentheses for multi-line strings in error messages:

```python
message = (
    f"Unable to process the input data. The threshold must be in range "
    f"(0, 1], but got {threshold}."
)
```

---

## Linting and Code Quality

### Running the Linter

Run `tox -e lint` after making changes. All issues must either be resolved or marked with proper
`# noqa` ignore statements.

### Resolution Policy

Prefer resolving issues unless the resolution would:
- Make the code unnecessarily complex
- Hurt performance by adding redundant checks
- Harm codebase readability instead of helping it

### Magic Numbers (PLR2004)

For magic number warnings, prefer defining constants:

**Local constants**: Use when the value is specific to a single function or method.

```python
def calculate_threshold(self, value: float) -> float:
    """Calculates the adjusted threshold."""
    adjustment_factor = 1.5  # Empirically determined scaling factor.
    return value * adjustment_factor
```

**Module-level constants**: Use when the value is a configuration parameter that may need
adjustment later.

```python
# Maximum number of retry attempts for network operations.
_MAX_RETRY_ATTEMPTS: int = 3

# Default timeout in milliseconds for sensor polling.
_SENSOR_TIMEOUT_MS: int = 500
```

### Using noqa

When suppressing a warning, always include the specific error code:

```python
if mode == 3:  # noqa: PLR2004 - LICK_TRAINING mode value from VisualizerMode enum.
    ...
```

### Typos

- **Obvious typos**: Must be fixed immediately (e.g., "teh" -> "the", "fucntion" -> "function").
- **Ambiguous typos**: If a typo may be intentional (e.g., domain-specific terminology,
  abbreviations), flag it for user confirmation before changing.

---

## Test Files

Test files follow simplified documentation conventions compared to library code, as test functions
serve a different purpose and don't require the same level of API documentation.

### Module Docstrings

Test module docstrings use the "Contains tests for..." format rather than imperative mood:

```python
"""Contains tests for classes and methods provided by the saver.py module."""
```

### Test Function Docstrings

Test function docstrings use imperative mood with "Verifies..." as the preferred format:

```python
def test_video_saver_init_repr(tmp_path, has_ffmpeg):
    """Verifies the functioning of the VideoSaver __init__() and __repr__() methods."""
```

For complex tests that cover multiple scenarios, the docstring may include a description of
tested cases:

```python
def test_ensure_list(input_item: Any, expected: list) -> None:
    """Verifies the functioning of the ensure_list() method for all supported scenarios.

    Tests the following inputs:
        - Multi-item lists, tuples, and sets
        - One-item lists, tuples, and sets
        - One-dimensional and multidimensional numpy arrays
        - Scalar types (int, float, string, bool, None)
    """
```

**Important**: Test function docstrings do not include Args, Returns, or Raises sections. A
summary line (and optional extended description for complex tests) is sufficient.

### Fixture Docstrings

Pytest fixtures use imperative mood docstrings describing what the fixture provides:

```python
@pytest.fixture(scope="session")
def has_nvidia():
    """Checks for NVIDIA GPU availability in the test environment."""
    ...

@pytest.fixture
def data_logger(tmp_path) -> DataLogger:
    """Creates a DataLogger instance and returns it to the caller."""
    ...
```

### Comments in Tests

Comments within test functions follow the same conventions as library code (third person
imperative), but may include inline notes for non-obvious test logic:

```python
# Verifies that the saver was initialized properly.
assert saver._system_id == 1

# Note, this may change the frame_rate as the camera may not support the requested value.
camera.connect()
```

---

## README Files

README files follow a standardized structure and writing style to maintain consistency across all
Sun Lab projects.

### Structure

README files use the following section order. Sections marked as optional may be omitted based on
project type.

1. **Title**: Project name as H1 heading (`# project-name`)
2. **One-line description**: Brief summary immediately after the title
3. **Badges**: PyPI/language badges, tooling badges, license, status (no blank line before badges)
4. **Horizontal rule**: `___` to separate header from content
5. **Detailed Description**: Expanded explanation of the library's purpose
6. **Features** *(optional)*: Bulleted list of key features. Common in public `ataraxis-*`
   libraries, often omitted in internal `sl-*` libraries.
7. **Table of Contents**: Links to all major sections using Markdown anchors
8. **Dependencies**: External requirements and automatic installation notes
9. **Installation**: Source and pip installation instructions
10. **Usage**: Detailed usage instructions with subsections
    - **MCP Server** *(optional)*: Document MCP server functionality if the library provides one.
      See the MCP Server Documentation subsection below for details.
11. **API Documentation**: Link to hosted documentation
12. **Developers** *(optional)*: Development setup and automation. Include for public PyPI packages
    (`ataraxis-*`), omit for internal lab libraries (`sl-*`).
13. **Versioning**: Semantic versioning statement with link to repository tags
14. **Authors**: List of contributors with GitHub profile links
15. **License**: License type with link to LICENSE file
16. **Acknowledgments**: Credits to Sun lab members and dependency creators

Use horizontal rules (`___`) to separate major sections.

### Header Format

The header follows a specific format with no blank lines between title and badges:

```markdown
# project-name

One-line description of what the library does.

![PyPI - Version](https://img.shields.io/pypi/v/project-name)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/project-name)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/project-name)
![PyPI - Status](https://img.shields.io/pypi/status/project-name)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/project-name)

___
```

### Table of Contents

Include a Table of Contents section linking to all major sections:

```markdown
## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
```

### Writing Style

**Voice**: Use third person throughout. Refer to the project as "this library," "the library," or
by its name. Avoid first person ("I," "we") and second person ("you") where possible.

```markdown
<!-- Good -->
This library abstracts all necessary steps for acquiring and saving video data.
The library supports Windows, Linux, and macOS.

<!-- Avoid -->
We provide tools for acquiring and saving video data.
You can use this library on Windows, Linux, and macOS.
```

**Tense**: Use present tense as the default. Avoid "will" unless omitting it makes the sentence
awkward or unclear. Acceptable uses of "will" include:

- Describing automatic behavior: "Dependencies will be automatically resolved..."
- Conditional outcomes: "...will likely not work in other contexts"
- Future-facing statements where present tense sounds unnatural

```markdown
<!-- Good - present tense -->
The method returns a tuple of timestamps.
Calling start() arms the video system and begins frame acquisition.
This command generates a configuration file.

<!-- Good - "will" where natural -->
These dependencies will be automatically resolved when the library is installed.
The library will likely not work without extensive modification.

<!-- Avoid - unnecessary "will" -->
The method will return a tuple of timestamps.
Calling start() will arm the video system.
```

**Notes and warnings**: Use `**Note!**` or `***Note!***` for important information. Use
`**Warning!**`, `***Warning!***`, or `***Critical!***` for dangerous operations or essential
requirements.

```markdown
**Note!** The API documentation also includes details about the CLI interface.

***Note!*** Developers should see the Developers section for additional dependencies.

**Warning!** This command permanently deletes data and cannot be undone.

***Critical!*** Each data source must use a unique identifier value.
```

### Formatting Conventions

**Code and commands**: Use inline code for short references and code blocks for commands or
examples. Code blocks may omit the language identifier for simple shell commands.

```markdown
Use the `axvs --help` command to see available options.

Install via pip: ```pip install ataraxis-video-system```
```

For multi-line code examples, use fenced code blocks:

```markdown
    ```python
    from ataraxis_video_system import VideoSystem
    vs = VideoSystem(system_id=np.uint8(101), data_logger=logger)
    vs.start()
    ```
```

**Links**: Use descriptive link text rather than raw URLs. Link to specific sections using anchor
references.

```markdown
<!-- Good -->
See the [Installation](#installation) section for details.
Consult the [API documentation](https://example.com) for all parameters.

<!-- Avoid -->
See https://example.com for details.
```

**Lists**: Use bulleted lists for features and unordered items. Use numbered lists only for
sequential steps (like installation instructions).

**Subsections**: Use H3 (`###`) and H4 (`####`) headings to organize content within major sections.
Keep the hierarchy logical and consistent.

### Standard Sections

**Acknowledgments**: Use the standard format for crediting Sun lab members and dependencies:

```markdown
## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration
  and comments during the development of this library.
- The creators of all other dependencies and projects listed in the
  [pyproject.toml](pyproject.toml) file.
```

**Versioning**: Use the standard semantic versioning statement:

```markdown
## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/project-name/tags) for the available
project releases.
```

**License**: Use the standard license statement:

```markdown
## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
```

### MCP Server Documentation

Libraries that provide MCP (Model Context Protocol) servers for agentic interaction should document
this functionality in the README. Add an "MCP Server" subsection under the Usage section.

**Structure**: Include the following information:

1. Brief description of what the MCP server exposes
2. How to start the server (CLI command)
3. List of available tools with brief descriptions
4. Configuration instructions (e.g., Claude Desktop setup)

**Example**:

```markdown
### MCP Server

This library provides an MCP server that exposes camera discovery, video session management, and
runtime checks for AI agent integration.

#### Starting the Server

Start the MCP server using the CLI:

```
axvs mcp
```

#### Available Tools

| Tool                         | Description                                    |
|------------------------------|------------------------------------------------|
| `list_cameras`               | Discovers all compatible cameras on the system |
| `start_video_session`        | Starts a video capture session                 |
| `stop_video_session`         | Stops the active video session                 |
| `check_runtime_requirements` | Verifies FFMPEG and GPU availability           |

#### Claude Desktop Configuration

Add the following to the Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "ataraxis-video-system": {
      "command": "axvs",
      "args": ["mcp"]
    }
  }
}
```
```

**Table of Contents**: When an MCP Server subsection is included, add it to the table of contents
under the Usage section anchor or as a separate entry if it warrants its own top-level section.

---

## Commit Messages

Commit messages follow a consistent format across all Sun Lab repositories. Well-written commit messages make it
easier to understand project history and generate changelogs.

### Format

**Single-line commits**: Use for focused, single-purpose changes.

```
Added Python 3.14 support.
Fixed a bug that allowed valves to violate keepalive guard.
Optimized the behavior of camera ID discovery functionality.
```

**Multi-line commits**: Use for changes that bundle multiple related modifications. Separate items with `--`
prefixes.

```
Added MCP server module for agentic library interaction.

-- Added mcp_server.py exposing camera discovery and video session management.
-- Added 'axvs mcp' CLI command to start the MCP server.
-- Added frame display support to MCP video sessions.
-- Fixed various documentation and code style inconsistencies.
```

### Writing Style

**Verb tense**: Start with a past tense verb describing what the commit accomplishes:

| Verb       | Use Case                                    |
|------------|---------------------------------------------|
| Added      | New features, files, or functionality       |
| Fixed      | Bug fixes and error corrections             |
| Updated    | Modifications to existing functionality     |
| Refactored | Code restructuring without behavior changes |
| Optimized  | Performance improvements                    |
| Improved   | Enhancements to existing features           |
| Removed    | Deletions of code, files, or features       |
| Deprecated | Marking functionality for future removal    |
| Suppressed | Silencing warnings or output                |
| Prepared   | Release preparation tasks                   |
| Finalized  | Completing a feature or release             |

**Punctuation**: Always end commit messages with a period.

**Content**: Focus on *what* was changed and *why*, not *how*. The code diff shows the implementation details.

```
# Good - describes what and implies why
Fixed a bug that prevented trial decomposition from working.

# Avoid - describes how
Changed the if statement on line 45 to use >= instead of >.
```

**Scope**: Be specific about what was affected.

```
# Good - specific
Updated MCP response formatting with newlines and bullet points.

# Avoid - vague
Made some improvements.
```

### Common Patterns

**Bug fixes**: Describe the bug that was fixed, not just that something was fixed.

```
Fixed an issue that prevented camera discovery on systems without GPU.
Fixed a race condition in the frame saving loop.
```

**New features**: Describe what the feature does.

```
Added frame display support to MCP video sessions.
Added 'axvs cti-check' CLI command to verify CTI file configuration.
```

**Documentation**: Be specific about what documentation was updated.

```
Updated README installation instructions for Windows users.
Added MCP server conventions to Sun Lab Style Guide.
```

**Refactoring**: Describe what was refactored and optionally why.

```
Refactored the VideoSystem example script and README documentation section.
Streamlined library features and source code.
```

**Version bumps and releases**: Use standard phrasing.

```
Prepared all 2.2.0 release artifacts.
Bumped the version to 2.1.1.
Finalized the 4.0.0 release.
```
