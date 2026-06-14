# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style guide compliance

Before writing, modifying, or reviewing any code or documentation, you MUST invoke the appropriate skill to load
Ataraxis framework conventions. This applies to ALL file types:

| Task                                | Skill to Invoke    |
|-------------------------------------|--------------------|
| Writing or modifying Python code    | `/python-style`    |
| Writing or modifying README files   | `/readme-style`    |
| Writing git commit messages         | `/commit`          |
| Writing or modifying pyproject.toml | `/pyproject-style` |
| Configuring tox.ini                 | `/tox-config`      |

All contributions must strictly follow these conventions. Key conventions include:
- Google-style docstrings with proper sections
- Full type annotations with explicit array dtypes
- Keyword arguments for function calls
- Third person imperative mood for comments and documentation
- Proper error handling with `console.error()`
- 120 character line limit

## Cross-referenced library verification

Ataraxis framework projects often depend on other `ataraxis-*` or `sl-*` libraries. These libraries may be stored
locally in the same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-time/`,
   `../ataraxis-base-utilities/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available skills

Skills are distributed through the ataraxis marketplace and are loaded into Claude Code via the plugin system. They are
**not** stored in this repository.

### Communication plugin skills (ataraxis/plugins/communication/)

| Skill                                  | Description                                                           |
|----------------------------------------|-----------------------------------------------------------------------|
| `/microcontroller-setup`               | MCP-based microcontroller discovery, MQTT verification, and manifests |
| `/microcontroller-interface`           | MicroControllerInterface and ModuleInterface API usage and lifecycle  |
| `/communication-mcp-environment-setup` | MCP server connectivity diagnostics and environment verification      |
| `/pipeline`                            | End-to-end pipeline orchestration and multi-controller planning       |
| `/extraction-configuration`            | ExtractionConfig parameters, generation, validation, and lifecycle    |
| `/log-input-format`                    | Reference for NPZ archive format, source IDs, and DataLogger output   |
| `/log-processing`                      | Orchestrate log archive processing workflow via MCP tools             |
| `/log-processing-results`              | Reference for output data formats and event distribution analysis     |

### Microcontroller plugin skills (ataraxis/plugins/microcontroller/)

| Skill              | Description                                                                   |
|--------------------|-------------------------------------------------------------------------------|
| `/firmware-module` | Firmware-side Module subclass implementation, command execution, and SendData |

### Automation plugin skills (ataraxis/plugins/automation/)

The automation plugin provides cross-cutting development skills; this table lists those relevant to this Python
library. Language-specific style skills (C++, C#) are omitted because they do not apply here.

| Skill                      | Description                                                                    |
|----------------------------|--------------------------------------------------------------------------------|
| `/explore-codebase`        | Perform in-depth codebase exploration at session start                         |
| `/explore-dependencies`    | Explore installed ataraxis dependency APIs for reuse opportunities             |
| `/python-style`            | Apply Ataraxis framework Python coding conventions (REQUIRED for code changes) |
| `/readme-style`            | Apply Ataraxis framework README conventions                                    |
| `/pyproject-style`         | Apply Ataraxis framework pyproject.toml conventions                            |
| `/tox-config`              | Apply Ataraxis framework tox.ini conventions                                   |
| `/api-docs`                | Apply Ataraxis framework Sphinx API documentation conventions                  |
| `/project-layout`          | Apply Ataraxis framework project directory structure conventions               |
| `/commit`                  | Draft Ataraxis framework style-compliant git commit messages                   |
| `/pr`                      | Draft a style-compliant pull request summary for the active branch             |
| `/release`                 | Draft style-compliant release notes from merged pull requests                  |
| `/skill-design`            | Generate and verify Claude Code skill files                                    |
| `/audit-facts`             | Audit documentation for factual accuracy against source code                   |
| `/audit-style`             | Audit files for style and convention compliance against framework checklists   |

## MCP server integration

This library provides an MCP server (`axci mcp`) that exposes microcontroller discovery, MQTT broker checking,
manifest management, extraction configuration management, and log data processing tools. When working with this project
or its dependencies, prefer using available MCP tools over direct code execution when appropriate.

**Guidelines for MCP usage:**

1. **Discover available tools**: At the start of a session, check which MCP servers are connected and what tools
   they provide. Use these tools when they offer functionality relevant to the current task.

2. **Prefer MCP for runtime operations**: For operations like microcontroller discovery, MQTT broker verification,
   log archive assembly, extraction configuration management, and batch log processing workflows, use MCP tools
   rather than writing and executing Python code directly. MCP tools provide:
   - Consistent, tested interfaces
   - Proper resource management and cleanup
   - Formatted output designed for user display

3. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

4. **Fall back to code when necessary**: Use direct code execution when:
   - No MCP tool exists for the required functionality
   - The task requires custom logic not covered by available tools
   - Writing or modifying library source code

## Distribution model

This project follows a dual distribution model. The library source code, tests, CLI, and MCP server implementation live
in this repository (`ataraxis-communication-interface`) and are distributed via PyPI. Claude Code skills and MCP server
registration are distributed separately through the [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace as
plugins:

- **communication** plugin (`ataraxis/plugins/communication/`): Registers the `axci mcp` server with compatible MCP
  clients and provides communication-specific skills for microcontroller setup, pipeline orchestration, log processing,
  extraction configuration, and post-processing verification.
- **microcontroller** plugin (`ataraxis/plugins/microcontroller/`): Provides firmware-side skills for implementing
  custom hardware Module subclasses in the companion
  [ataraxis-micro-controller](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) C++ library. The
  `/firmware-module` skill complements the `/microcontroller-interface` skill from the communication plugin, covering
  the firmware counterpart to the PC-side ModuleInterface.
- **automation** plugin (`ataraxis/plugins/automation/`): Provides shared development skills that enforce Ataraxis
  framework coding conventions (Python style, README style, commit messages, pyproject.toml, tox configuration) and
  general-purpose codebase exploration tools.

When modifying skills, edit the SKILL.md files in the ataraxis marketplace repository, not in this repository.
When modifying the MCP server implementation or library code, edit the source files in this repository.

## Project context

This is **ataraxis-communication-interface**, a Python library that provides the centralized interface for exchanging
commands and data between Arduino and Teensy microcontrollers and host computers. It abstracts hardware module
management, serial/USB communication, MQTT data exchange, and provides log processing for extracting hardware event
data from DataLogger archives.

### Key areas

| Directory                                   | Purpose                                                                       |
|---------------------------------------------|-------------------------------------------------------------------------------|
| `src/ataraxis_communication_interface/`     | Main library source code                                                      |
| `src/.../communication/`                    | Serial/MQTT communication package (`protocols`, `messages`, `serial`, `mqtt`) |
| `src/.../microcontroller/interface.py`      | Core MicroControllerInterface and ModuleInterface ABC                         |
| `src/.../microcontroller/dataclasses.py`    | Manifest and extraction configuration data structures                         |
| `src/.../microcontroller/log_processing.py` | Log data processing pipeline for extracting module/kernel events              |
| `src/.../interfaces/`                       | CLI (`axci`), MCP server entry point, shared instance, and MCP tool groups    |
| `tests/`                                    | Test suite (dataclasses, communication, log_processing)                       |
| `examples/`                                 | Example ModuleInterface subclass and runtime usage                            |
| `docs/`                                     | Sphinx API documentation source                                               |

### Architecture

- **MicroControllerInterface**: Multiprocessing architecture for bidirectional microcontroller communication.
  Constructor requires `controller_id`, `data_logger`, `module_interfaces`, `buffer_size`, `port`, and `name` as
  positional arguments; `__init__` writes a microcontroller manifest entry associating the controller_id with the
  human-readable name and its module list. A dedicated communication process handles serial I/O via
  `SerialCommunication` and dispatches received messages to the appropriate `ModuleInterface` based on
  `(module_type, module_id)` routing. A watchdog thread monitors process health. Commands and parameters flow from
  the main process through an `MPQueue` to the communication process.
- **ModuleInterface**: Abstract base class that users subclass to define hardware module behavior. Requires
  `module_type`, `module_id`, and `name` as positional arguments; optional `error_codes` and `data_codes` sets
  control message routing. Three abstract methods: `initialize_remote_assets()`, `terminate_remote_assets()`, and
  `process_received_data()`. Public methods `send_command()` and `send_parameters()` use LRU-cached message
  construction for performance. `reset_command_queue()` sends a dequeue command to the microcontroller. The
  `type_id` property combines `(type << 8) | id` for dispatch lookups.
- **Serial Communication**: `SerialCommunication` wraps `TransportLayer` from `ataraxis-transport-layer-pc` for
  CRC16-CCITT checksummed serial I/O. Supports 12 message protocols (`SerialProtocols` enum) and 252 numpy data
  prototypes (`SerialPrototypes` enum) covering all numpy scalar and array types. `send_message()` accepts
  command and parameter message objects; `receive_message()` returns typed message objects (`ModuleData`,
  `ModuleState`, `KernelData`, `KernelState`, `ReceptionCode`, `ControllerIdentification`,
  `ModuleIdentification`). All received data is timestamped via `PrecisionTimer` and logged to `DataLogger` through
  an `MPQueue`.
- **MQTT Communication**: `MQTTCommunication` provides publish/subscribe messaging over MQTT via `paho-mqtt`.
  Constructor takes `ip`, `port`, and optional `monitored_topics`. `get_data()` returns `(topic, message)`
  tuples from an internal `Queue` populated by the on_message callback.
- **Microcontroller Manifest**: `MicroControllerManifest` (`YamlConfig` subclass) associates controller IDs with
  human-readable names and their module lists in a `microcontroller_manifest.yaml` file alongside DataLogger
  archives. `MicroControllerSourceData` stores per-controller entries with a tuple of `ModuleSourceData` objects.
  `write_microcontroller_manifest()` creates or updates the manifest. Used by log processing discovery to identify
  which archives were produced by ataraxis-communication-interface and to route processing by source ID.
- **Extraction Configuration**: `ExtractionConfig` (`YamlConfig` subclass) specifies which controllers, modules,
  and event codes to extract from log archives. `ControllerExtractionConfig` contains per-controller settings with
  `ModuleExtractionConfig` (module_type, module_id, event_codes) and optional `KernelExtractionConfig`.
  `create_extraction_config()` generates a precursor config from a manifest with empty event codes.
- **Log Processing**: Pipeline for extracting hardware module and kernel event data from DataLogger `.npz` archives.
  Requires a `microcontroller_manifest.yaml` in the log directory for source ID resolution and validation. Supports
  sequential and parallel (`ProcessPoolExecutor`) processing with a 2000-message threshold for parallelization.
  Uses `LogArchiveReader` for archive access and `ProcessingTracker` for job lifecycle management.
  `run_log_processing_pipeline()` orchestrates local (all jobs) and remote (single job by ID) execution modes.
  Outputs Polars DataFrames as Feather (Arrow IPC) files in a `microcontroller_data/` subdirectory.
- **MCP Server**: `FastMCP` instance (`name="ataraxis-communication-interface"`, `json_response=True`) defined in
  `interfaces/mcp_instance.py`, with 19 tools split across `interfaces/discovery_tools.py`, `config_tools.py`,
  `processing_tools.py`, and `output_tools.py`. The tool modules register on the shared instance via `@mcp.tool()`
  decorators and are imported for their side effects by the thin `interfaces/mcp_server.py`, which also exposes
  `run_server()`. Tool categories: microcontroller discovery (2), log archive management (1), manifest management (2),
  recording discovery (1), extraction config management (3), batch processing execution (2), processing status and
  management (5), and output verification and cleanup (3). Batch log processing uses `JobExecutionState` (in
  `interfaces/mcp_execution.py`, accessed via `get_execution_state()` / `set_execution_state()`) with budget-based
  worker allocation: the execution manager divides the CPU budget evenly among concurrent parallel jobs (snapped to
  multiples of 5) with a sqrt-derived saturation floor. The MCP server is registered with MCP clients via the
  **communication** plugin in the ataraxis marketplace, not directly from this repository.
- **CLI**: Click command group (`axci`) with `id` for microcontroller discovery, `mqtt` for broker verification,
  `config` subgroup (`create`, `show`) for extraction configuration management, `process` for log data processing,
  and `mcp` for starting the MCP server.

### Key patterns

- **Daemon Communication Process**: The communication process is a daemon process requiring an explicit `stop()`
  call. Callers are responsible for setting an appropriate multiprocessing start method if needed.
- **Message Protocol Stack**: Four levels — `SerialCommunication` (USB/UART), `TransportLayer` (CRC checksums,
  frame encoding), message protocols (12 types via `SerialProtocols` enum), and data prototypes (252 numpy types
  via `SerialPrototypes` enum).
- **LRU Caching**: `ModuleInterface` caches command messages (`maxsize=32`) and parameter messages (`maxsize=16`)
  to avoid redundant serialization during repeated operations.
- **Type-ID Dispatch**: Received messages are routed to `ModuleInterface` instances via a `(module_type, module_id)`
  → `type_id` (`uint16`) lookup, where `type_id = (type << 8) | id`.
- **Manifest-Based Log Discovery**: `microcontroller_manifest.yaml` files tag DataLogger output directories with
  source-to-name mappings. Log processing discovery and batch preparation use manifests to identify which archives
  were produced by ataraxis-communication-interface and to route jobs by source ID.
- **Columnar Data Extraction**: Log processing accumulates data in parallel lists via `_ColumnAccumulator`, converts
  to numpy arrays, then builds Polars DataFrames for efficient Feather output.
- **Budget-Based Worker Allocation**: The MCP execution manager computes a worker budget as `available_cores - 2`,
  divides it among concurrent jobs snapped to multiples of 5, with a sqrt-derived saturation floor based on message
  count (`ceil(sqrt(messages / 1000))`).
- **Frozen Dataclasses**: Inner data classes (`ModuleSourceData`, `MicroControllerSourceData`,
  `ModuleExtractionConfig`, `KernelExtractionConfig`, `ControllerExtractionConfig`) use `frozen=True` for
  immutability and `slots=True` for performance. The top-level `MicroControllerManifest` and `ExtractionConfig`
  classes extend `YamlConfig` and are mutable (not frozen).

### Code standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.12, 3.13, 3.14 support
- See style skills for complete conventions

### Workflow guidance

Non-obvious facts for the most common modifications. Read the cited files for full context.

- **MicroControllerInterface** (`microcontroller/interface.py`): the communication loop runs in `_runtime_cycle()`,
  a static method executed in a spawned daemon process; a watchdog thread in the main process monitors liveness.
  Commands flow from the main process through an `MPQueue` to the communication process, which requires an explicit
  `stop()` call. Test against microcontroller hardware or in test mode.
- **ModuleInterface** (`microcontroller/interface.py`): subclasses must implement `initialize_remote_assets()`,
  `terminate_remote_assets()`, and `process_received_data()`. `send_command()` and `send_parameters()` use
  LRU-cached message construction; `reset_command_queue()` sends a dequeue command. See
  `examples/example_interface.py` for a reference subclass.
- **Serial communication** (`communication/protocols.py`, `messages.py`, `serial.py`): `SerialProtocols`
  (12 protocols) and `SerialPrototypes` (252 prototypes) define the protocol layer. Command classes pack bytes via
  the `packed_data` property; reception classes parse header bytes via properties.
- **MQTT communication** (`communication/mqtt.py`): `paho-mqtt` v2 client with callback reception into a `Queue`;
  `get_data()` returns `(topic, message)` tuples or `None`, and the `has_data` property checks queue state.
- **Data classes and manifests** (`microcontroller/dataclasses.py`): inner classes are frozen — create new instances
  rather than mutating; `MicroControllerManifest` and `ExtractionConfig` are mutable `YamlConfig` subclasses.
  `create_extraction_config()` builds a precursor config with empty event codes.
- **Log processing** (`microcontroller/log_processing.py`): `run_log_processing_pipeline()` supports local (all jobs)
  and remote (single job by ID) modes; the `config` parameter is a `Path` loaded internally; the parallelization
  threshold is 2000 messages; `ProcessingTracker` manages job lifecycle via YAML state files.
- **CLI** (`interfaces/cli.py`): use `console.echo()` for output and `console.error()` for errors; the `config`
  subgroup demonstrates nested Click command groups.
- **MCP tools** (`interfaces/*_tools.py`): register on the shared instance from `interfaces/mcp_instance.py` via
  `@mcp.tool()`, add new tool modules to the side-effect import list in `interfaces/mcp_server.py`, and return
  JSON-serializable `dict[str, Any]`. Execution uses `JobExecutionState` (`interfaces/mcp_execution.py`) with
  budget-based worker allocation.
