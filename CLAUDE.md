# Claude Code Instructions

## Session Start Behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style Guide Compliance

Before writing, modifying, or reviewing any code or documentation, you MUST invoke the appropriate skill to load Sun
Lab conventions. This applies to ALL file types:

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

## Cross-Referenced Library Verification

Sun Lab projects often depend on other `ataraxis-*` or `sl-*` libraries. These libraries may be stored locally in the
same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

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

## Distribution Model

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
- **automation** plugin (`ataraxis/plugins/automation/`): Provides shared development skills that enforce Sun Lab
  coding conventions (Python style, README style, commit messages, pyproject.toml, tox configuration) and
  general-purpose codebase exploration tools.

When modifying skills, edit the SKILL.md files in the ataraxis marketplace repository, not in this repository.
When modifying the MCP server implementation or library code, edit the source files in this repository.

## MCP Server Integration

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

## Available Skills

Skills are distributed through the ataraxis marketplace and are loaded into Claude Code via the plugin system. They are
**not** stored in this repository.

### Communication Plugin Skills (ataraxis/plugins/communication/)

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

### Microcontroller Plugin Skills (ataraxis/plugins/microcontroller/)

| Skill              | Description                                                                   |
|--------------------|-------------------------------------------------------------------------------|
| `/firmware-module` | Firmware-side Module subclass implementation, command execution, and SendData |

### Automation Plugin Skills (ataraxis/plugins/automation/)

| Skill                      | Description                                                              |
|----------------------------|--------------------------------------------------------------------------|
| `/explore-codebase`        | Perform in-depth codebase exploration at session start                   |
| `/explore-dependencies`    | Explore installed ataraxis dependency APIs for reuse opportunities       |
| `/python-style`            | Apply Sun Lab Python coding conventions (REQUIRED for code changes)      |
| `/readme-style`            | Apply Sun Lab README conventions                                         |
| `/commit`                  | Draft Sun Lab style-compliant git commit messages                        |
| `/pyproject-style`         | Apply Sun Lab pyproject.toml conventions                                 |
| `/tox-config`              | Apply Sun Lab tox.ini conventions                                        |
| `/skill-design`            | Generate and verify Claude Code skill files                              |
| `/project-layout`          | Apply Sun Lab project directory structure conventions                    |

## Project Context

This is **ataraxis-communication-interface**, a Python library that provides the centralized interface for exchanging
commands and data between Arduino and Teensy microcontrollers and host computers. It abstracts hardware module
management, serial/USB communication, MQTT data exchange, and provides log processing for extracting hardware event
data from DataLogger archives.

### Key Areas

| Directory                                | Purpose                                                            |
|------------------------------------------|--------------------------------------------------------------------|
| `src/ataraxis_communication_interface/`  | Main library source code                                           |
| `src/.../microcontroller_interface.py`   | Core MicroControllerInterface and ModuleInterface ABC              |
| `src/.../communication.py`               | Serial and MQTT communication, message protocol, data prototypes   |
| `src/.../dataclasses.py`                 | Manifest and extraction configuration data structures              |
| `src/.../log_processing.py`              | Log data processing pipeline for extracting module/kernel events   |
| `src/.../cli.py`                         | Click-based `axci` CLI with subcommand groups                      |
| `src/.../mcp_server.py`                  | FastMCP server with 19 tools for discovery, config, and processing |
| `tests/`                                 | Test suite (dataclasses, communication, log_processing)            |
| `examples/`                              | Example ModuleInterface subclass and runtime usage                 |
| `docs/`                                  | Sphinx API documentation source                                    |

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
  `type_id` property combines `(type << 8) | id` for
  dispatch lookups.
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
- **MCP Server**: `FastMCP` instance (`name="ataraxis-communication-interface"`, `json_response=True`) with 19
  tools and `_JobExecutionState` tracking for batch processing. Tool categories: microcontroller discovery (2),
  log archive management (1), manifest management (2), recording discovery (1), extraction config management (3),
  batch processing execution (2), processing status and management (5), and output verification and cleanup (3).
  Batch log processing uses `_JobExecutionState` with budget-based worker allocation: the execution manager divides
  the CPU budget evenly among concurrent parallel jobs (snapped to multiples of 5) with a sqrt-derived saturation
  floor. The MCP server is registered with MCP clients via the **communication** plugin in the ataraxis marketplace,
  not directly from this repository.
- **CLI**: Click command group (`axci`) with `id` for microcontroller discovery, `mqtt` for broker verification,
  `config` subgroup (`create`, `show`) for extraction configuration management, `process` for log data processing,
  and `mcp` for starting the MCP server.

### Key Patterns

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

### Code Standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- Ruff for formatting and linting
- Python 3.12, 3.13, 3.14 support
- See style skills for complete conventions

### Workflow Guidance

**Modifying MicroControllerInterface:**

1. Review `src/ataraxis_communication_interface/microcontroller_interface.py` for current implementation
2. Understand the multiprocessing architecture: main process sends commands via `MPQueue`, communication process
   handles serial I/O and dispatches received messages to `ModuleInterface` instances
3. The communication loop runs in `_runtime_cycle()` as a static method in a spawned process
4. The watchdog thread monitors process liveness from the main process
5. Test with actual microcontroller hardware or in test mode

**Modifying ModuleInterface:**

1. Review `src/ataraxis_communication_interface/microcontroller_interface.py` for the ABC definition
2. Subclasses must implement `initialize_remote_assets()`, `terminate_remote_assets()`, and
   `process_received_data()`
3. `send_command()` and `send_parameters()` use LRU-cached message construction; `reset_command_queue()` sends a 
   dequeue command
4. See `examples/example_interface.py` for a reference subclass implementation

**Modifying serial communication:**

1. Review `src/ataraxis_communication_interface/communication.py` for all message types and protocols
2. `SerialProtocols` (12 codes) and `SerialPrototypes` (252 codes) define the protocol layer
3. Command classes (`RepeatedModuleCommand`, `OneOffModuleCommand`, `DequeueModuleCommand`, `KernelCommand`,
   `ModuleParameters`) construct packed byte arrays via `packed_data` property
4. Reception classes (`ModuleData`, `ModuleState`, `KernelData`, `KernelState`, `ReceptionCode`,
   `ControllerIdentification`, `ModuleIdentification`) parse header bytes via properties

**Modifying MQTT communication:**

1. Review `src/ataraxis_communication_interface/communication.py` for `MQTTCommunication`
2. Uses `paho-mqtt` v2 client with callback-based message reception into a `Queue`
3. `connect()`/`disconnect()` manage the MQTT client lifecycle
4. `get_data()` returns `(topic, message)` tuples or `None`; `has_data` property checks queue state

**Modifying data classes and manifests:**

1. Review `src/ataraxis_communication_interface/dataclasses.py` for all data structures
2. `MicroControllerManifest` and `ExtractionConfig` extend `YamlConfig` from `ataraxis-data-structures`
3. Inner data classes are frozen — create new instances rather than mutating. Top-level `MicroControllerManifest`
   and `ExtractionConfig` are mutable `YamlConfig` subclasses
4. `create_extraction_config()` generates a precursor config from a manifest with empty event codes

**Modifying log processing:**

1. Review `src/ataraxis_communication_interface/log_processing.py` for the processing pipeline
2. `run_log_processing_pipeline()` supports local mode (all jobs) and remote mode (single job by ID)
3. `ProcessingTracker` manages job lifecycle (SCHEDULED → RUNNING → SUCCEEDED/FAILED) via YAML state files
4. `_process_message_batch()` runs in subprocess workers and is excluded from coverage (`# pragma: no cover`)
5. Parallelization threshold is 2000 messages; below that, processing runs sequentially
6. Log discovery uses manifest-based routing via `microcontroller_manifest.yaml` files
7. The `config` parameter is a `Path` — the pipeline loads the `ExtractionConfig` internally

**Adding or modifying CLI commands:**

1. Review `src/ataraxis_communication_interface/cli.py` for existing Click group structure
2. Follow existing patterns for option decorators and error handling
3. Use `console.echo()` for output and `console.error()` for error handling
4. The `config` subgroup demonstrates nested Click command groups

**Adding or modifying MCP tools:**

1. Review `src/ataraxis_communication_interface/mcp_server.py` for existing tool patterns
2. Log processing execution uses `_JobExecutionState` with budget-based worker allocation
3. The execution manager divides budget among parallel jobs via `_compute_sqrt_minimum()`
4. Return `dict[str, Any]` (JSON-serializable) for all tool responses
5. MCP server registration happens in the ataraxis marketplace communication plugin, not in this repository
