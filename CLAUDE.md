# Claude Code Instructions

## Session Start Behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style Guide Compliance

Before writing, modifying, or reviewing any code or documentation, you MUST invoke the `/sun-lab-style` skill to load
the Sun Lab conventions. This applies to ALL file types including:
- Python source files (`.py`)
- Documentation files (`README.md`, docstrings)
- Configuration files when adding comments or descriptions

All contributions must strictly follow these conventions and all reviews must check for compliance. Key conventions
include:
- Google-style docstrings with proper sections
- Full type annotations with explicit array dtypes
- Keyword arguments for function calls
- Third person imperative mood for comments and documentation
- Proper error handling with `console.error()`
- README structure and formatting standards
- MCP server documentation format

## Cross-Referenced Library Verification

Sun Lab projects often depend on other `ataraxis-*` or `sl-*` libraries. These libraries may be stored
locally in the same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g.,
   `../ataraxis-time/`, `../ataraxis-base-utilities/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or
   main branch on GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch,
   notify the user with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative
   reference for API usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the
actual library state to prevent integration errors.

## MCP Server Integration

Sun Lab libraries may provide MCP (Model Context Protocol) servers that expose library functionality for agentic
interaction. When working with this project or its dependencies, prefer using available MCP tools over direct code
execution when appropriate.

**Guidelines for MCP usage:**

1. **Discover available tools**: At the start of a session, check which MCP servers are connected and what tools
   they provide. Use these tools when they offer functionality relevant to the current task.

2. **Prefer MCP for runtime operations**: For operations like microcontroller discovery, serial port management, and
   runtime checks, use MCP tools rather than writing and executing Python code directly. MCP tools provide:
   - Consistent, tested interfaces
   - Proper resource management and cleanup
   - Formatted output designed for user display

3. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

4. **Fall back to code when necessary**: Use direct code execution when:
   - No MCP tool exists for the required functionality
   - The task requires custom logic not covered by available tools
   - Writing or modifying library source code

**Example**: Instead of writing Python code to discover microcontrollers, use the `axci-id` CLI command or MCP tool
if available.

## Available Skills

- `/explore-codebase` - Perform in-depth codebase exploration
- `/sun-lab-style` - Apply Sun Lab coding and documentation conventions (REQUIRED for all code and documentation changes)

## Project Context

This is **ataraxis-communication-interface**, a Python library for managing bidirectional communication between
host computers and Arduino/Teensy microcontrollers. The library provides a robust abstraction layer over serial
communication with support for multiple hardware modules, automatic message logging, and MQTT networking.

### Key Areas

| Directory                                                           | Purpose                                      |
|---------------------------------------------------------------------|----------------------------------------------|
| `src/ataraxis_communication_interface/`                             | Main library source code                     |
| `src/ataraxis_communication_interface/communication.py`             | Message structures and serial/MQTT protocols |
| `src/ataraxis_communication_interface/microcontroller_interface.py` | Core controller interfaces                   |
| `examples/`                                                         | Usage examples and demonstrations            |
| `tests/`                                                            | Test suite                                   |

### Architecture

- **ModuleInterface**: Abstract base class for custom hardware module interfaces. Users subclass this to implement
  `initialize_remote_assets()`, `terminate_remote_assets()`, and `process_received_data()` methods.
- **MicroControllerInterface**: Manages communication with a microcontroller and all attached modules. Spawns a
  separate communication process for thread-safe serial I/O.
- **Message Protocol**: Binary-safe message structures using frozen dataclasses with 166 different data type
  prototypes (scalars and arrays of various numeric types).
- **MQTT Communication**: Network-based communication via MQTT broker for distributed systems.
- **MCP Server**: Model Context Protocol server for AI agent integration with microcontroller discovery and MQTT
  broker connectivity tools.
- **CLI Commands**: `axci-id` for microcontroller discovery, `axci-mqtt` for MQTT broker connectivity checking,
  `axci-mcp` for starting the MCP server.

### Key Patterns

- **Multiprocessing Architecture**: Communication runs in an isolated process to prevent GIL blocking on serial I/O.
  A watchdog thread monitors process health.
- **LRU Caching**: Commands and parameters are cached to avoid redundant serialization.
- **Message Logging**: All messages are automatically logged to disk as `.npy` files with timestamps.
- **Service vs Custom Messages**: Event codes <= 50 are handled by the framework; codes > 50 are routed to user code.
- **Keepalive Safety**: Optional bidirectional keepalive messaging at configurable intervals.

### Code Standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- See `/sun-lab-style` for complete conventions
