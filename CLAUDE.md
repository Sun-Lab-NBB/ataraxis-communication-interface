# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you MUST build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

## Style guide compliance

Before writing, modifying, or reviewing any code or documentation, you MUST invoke the appropriate skill to load
Ataraxis framework conventions. This applies to ALL file types:

| Task                                | Skill to invoke    |
|-------------------------------------|--------------------|
| Writing or modifying Python code    | `/python-style`    |
| Writing or modifying README files   | `/readme-style`    |
| Writing git commit messages         | `/commit`          |
| Writing or modifying pyproject.toml | `/pyproject-style` |
| Configuring tox.ini                 | `/tox-config`      |

## Cross-referenced library verification

Ataraxis framework projects often depend on other `ataraxis-*` or `sollertia-*` libraries. These libraries may be
stored locally in the same parent directory as this project, reachable as `../` from the repository root.

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

## Available skills

Skills live in the ataraxis marketplace repository and are loaded into Claude Code via the plugin system.

### Communication plugin skills (ataraxis/plugins/communication/)

| Skill                                  | Description                                                           |
|----------------------------------------|-----------------------------------------------------------------------|
| `/microcontroller-setup`               | MCP-based microcontroller discovery, MQTT verification, and manifests |
| `/microcontroller-interface`           | MicroControllerInterface and ModuleInterface API usage and lifecycle  |
| `/communication-mcp-environment-setup` | MCP server connectivity diagnostics and environment verification      |
| `/cli-reference`                       | Reference for every axci CLI command, option, and MCP tool mapping    |
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

| Skill                   | Description                                                                    |
|-------------------------|--------------------------------------------------------------------------------|
| `/explore-codebase`     | Perform in-depth codebase exploration at session start                         |
| `/explore-dependencies` | Explore installed ataraxis dependency APIs for reuse opportunities             |
| `/audit-correctness`    | Audit source code for bugs, edge cases, races, and leaks                       |
| `/audit-facts`          | Audit documentation for factual accuracy against source code                   |
| `/audit-performance`    | Audit source code for cost, speed, memory use, and dtype predictability        |
| `/audit-project`        | Orchestrate the four audits and merge their findings into one report           |
| `/audit-style`          | Audit files for style and convention compliance against framework checklists   |
| `/python-style`         | Apply Ataraxis framework Python coding conventions (REQUIRED for code changes) |
| `/readme-style`         | Apply Ataraxis framework README conventions                                    |
| `/pyproject-style`      | Apply Ataraxis framework pyproject.toml conventions                            |
| `/tox-config`           | Apply Ataraxis framework tox.ini conventions                                   |
| `/api-docs`             | Apply Ataraxis framework Sphinx API documentation conventions                  |
| `/skill-design`         | Generate and verify Claude Code skill files                                    |
| `/project-layout`       | Apply Ataraxis framework project directory structure conventions               |
| `/commit`               | Draft Ataraxis framework style-compliant git commit messages                   |
| `/pr`                   | Draft a style-compliant pull request summary for the active branch             |
| `/release`              | Draft style-compliant release notes from merged pull requests                  |

## MCP server

This library provides an MCP server (`axci mcp`) that exposes microcontroller discovery, MQTT broker checking, log
archive assembly, recording discovery, manifest management, extraction configuration management, log data processing,
output verification, output cleanup, and extracted event querying tools. When working with this project
or its dependencies, prefer using available MCP tools over direct code execution when appropriate.

1. **Discover available tools**: At the start of a session, check which MCP servers are connected and what tools
   they provide. Use these tools when they offer functionality relevant to the current task.

2. **Prefer MCP for runtime operations**: For operations like microcontroller discovery, MQTT broker verification,
   log archive assembly, extraction configuration management, and batch log processing workflows, use MCP tools
   rather than writing and executing Python code directly.

3. **Use MCP for cross-library operations**: When dependency libraries (e.g., `ataraxis-data-structures`,
   `ataraxis-time`) provide MCP servers, explore and use their tools for interacting with those libraries.

## Companion library synchronization

The companion [ataraxis-micro-controller](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) C++ library is the
firmware counterpart to this library, and parts of this codebase track it in lockstep. The enumerations in
`microcontroller/status_codes.py` mirror `kKernelStatusCodes`, `kCoreStatusCodes`, `kCommunicationStatusCodes`, the
microcontroller-side `kTransportStatusCodes`, and `kKernelCommands`. `communication/protocols.py` mirrors `kProtocols`
and `kPrototypes`, and the dataclasses in `communication/messages.py` mirror the packed message structs the firmware
declares in `axmc_shared_assets.h`. A firmware release that changes any of them requires the mirror to change with it.

## Distribution model

The library source code, tests, CLI, and MCP server implementation live in this repository
(`ataraxis-communication-interface`) and are distributed via PyPI. Claude Code skills and MCP server registration are
distributed separately through the [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace as plugins:

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

When modifying skills, edit the SKILL.md files in the ataraxis marketplace repository.
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
| `src/.../microcontroller/status_codes.py`   | Firmware status code mirrors and the message-to-explanation translators       |
| `src/.../microcontroller/dataclasses.py`    | Manifest and extraction configuration data structures                         |
| `src/.../microcontroller/log_processing.py` | Log data extraction algorithm and the columnar structures it returns          |
| `src/.../microcontroller/extracted_data.py` | Extracted message table schema, its writer, and the primitives that read it   |
| `src/.../orchestration/`                    | Job identity, sizing, discovery, the single-job runner, and execution paths   |
| `src/.../interfaces/`                       | CLI (`axci`), MCP entry point, shared instance, response machinery, tools     |
| `tests/`                                    | Test suite, grouped into per-package directories mirroring the source layout  |
| `examples/`                                 | Example ModuleInterface subclass and runtime usage                            |
| `docs/`                                     | Sphinx API documentation source                                               |

### Architecture

- **MicroControllerInterface**: Multiprocessing architecture for bidirectional microcontroller communication.
  `__init__` writes a microcontroller manifest entry associating the controller_id with the human-readable name and
  its module list. A dedicated communication process handles serial I/O via `SerialCommunication` and dispatches
  received messages to the appropriate `ModuleInterface` based on `(module_type, module_id)` routing. A watchdog thread
  monitors process health, and commands and parameters flow from the main process through an `MPQueue`.
- **ModuleInterface**: Abstract base class that users subclass to define hardware module behavior. Its three abstract
  methods, `initialize_remote_assets()`, `terminate_remote_assets()`, and `process_received_data()`, run inside the
  communication process, and the `type_id` property combines `(type << 8) | id` for dispatch lookups.
- **Status code resolution**: `microcontroller/status_codes.py` mirrors the four firmware status code families and
  resolves each received code into the description of a fault or an ordinary state. Membership in the private
  description tables defines which codes interrupt the runtime, and a code matching no enumeration member reports that
  it falls outside the resolved range rather than being ignored. The messages state what the reported codes establish
  and prescribe no response, because a status code records where a fault was detected rather than what caused it.
- **Serial communication**: `SerialCommunication` wraps `TransportLayer` from `ataraxis-transport-layer-pc` for
  CRC16-CCITT checksummed serial I/O. All received data is timestamped via `PrecisionTimer` and logged to `DataLogger`
  through an `MPQueue`.
- **MQTT communication**: `MQTTCommunication` provides publish/subscribe messaging over MQTT via `paho-mqtt`.
- **Microcontroller manifest**: `MicroControllerManifest` (`YamlConfig` subclass) associates controller IDs with
  human-readable names and their module lists in a `microcontroller_manifest.yaml` file alongside DataLogger archives.
- **Extraction configuration**: `ExtractionConfig` (`YamlConfig` subclass) specifies which controllers, modules, and
  event codes to extract from log archives.
- **Log data extraction**: `extract_logged_microcontroller_data()` reads a DataLogger `.npz` archive once and returns
  the matching module and kernel messages, choosing between the sequential and the `ProcessPoolExecutor` path through
  the `PARALLEL_PROCESSING_THRESHOLD` constant from ataraxis-data-structures.
- **Orchestration**: The `orchestration/` package owns everything above the extraction algorithm, resolving the job
  universe from the `microcontroller_manifest.yaml` and sizing each job from the archive it will read. It runs the
  shared-pool batch engine that admits jobs against a core and a memory budget, and runs one recording sequentially. It
  writes Feather (Arrow IPC) files through `atomic_write()` into a `microcontroller_data/` subdirectory.
- **MCP server**: The tool modules register on the shared `MCPServer` instance from `interfaces/mcp_instance.py` via
  `@mcp.tool()` decorators and are imported for their side effects by the thin `interfaces/mcp_server.py`, whose
  `run_server()` enables JSON responses when it starts the streamable-http transport.
- **CLI**: Click command group (`axci`) exposing microcontroller discovery, MQTT broker verification, extraction
  configuration management, log data processing, and MCP server startup.

### Key patterns

- **Daemon communication process**: `start()` spawns the daemon communication process, verifies the controller and
  module identity, and launches the watchdog thread, and the process requires an explicit `stop()` call. Callers are
  responsible for setting an appropriate multiprocessing start method if needed.
- **Message protocol stack**: Four levels: `SerialCommunication` (USB/UART), `TransportLayer` (CRC checksums,
  frame encoding), message protocols (13 types via `SerialProtocols` enum), and data prototypes (252 numpy types
  via `SerialPrototypes` enum).
- **LRU caching**: `ModuleInterface` caches command messages (`maxsize=32`) and parameter messages (`maxsize=16`)
  to avoid redundant serialization during repeated operations.
- **Type-ID dispatch**: Received messages are routed to `ModuleInterface` instances via a `(module_type, module_id)`
  → `type_id` (`uint16`) lookup, where `type_id = (type << 8) | id`.
- **Manifest-based log discovery**: `microcontroller_manifest.yaml` files tag DataLogger output directories with
  source-to-name mappings. Log processing discovery and batch preparation use manifests to identify which archives
  were produced by ataraxis-communication-interface and to route jobs by source ID.
- **Columnar data extraction**: Log processing accumulates data in parallel lists via `_ColumnAccumulator`, converts
  to numpy arrays, then builds Polars DataFrames for efficient Feather output.
- **Archive-derived job sizing**: Every job is sized before dispatch from the archive it will read.
  `resolve_archive_footprint()` reads the `.npz` zip directory and the file size, and `resolve_job_workers()` emits one
  of two shapes and nothing between them: a single core below `_PARALLEL_EXTRACTION_THRESHOLD` (15000 data messages) and
  the declared `CONTROLLER_EXTRACTION_JOB_CORES` width (4) at or above it. That threshold is distinct from the
  `PARALLEL_PROCESSING_THRESHOLD` governing message batching inside the archive reader. The width follows from the
  archive alone, so admission rather than the resolver holds a job to the cores its batch can spare.
  `estimate_job_memory_mb()` charges one spawned child baseline and one archive reader per core, plus the job body's
  own baseline and reader, and takes a sequential branch for a single-core job. The body and a pool child carry
  separate baselines, because a body assembles and writes the extracted output while a child returns what it decoded.
  The execution manager then admits jobs against both a core budget (`available_cores - 2`) and a memory budget (a share
  of the host's physical memory), admitting an oversized job alone rather than starving it. The declared width and every
  memory term carry the values the platform's own estimators were calibrated to against measured peaks, so this stage's
  jobs and the stages queued beside them are sized on one scale. Changing a constant here changes how this stage
  competes for admission against every other stage a scheduler plans with it.
- **Library-owned output contract**: This library owns both directions of the format it writes. `resolve_module_path()`
  and `resolve_kernel_path()` name the files, `find_module_paths()` and `find_kernel_paths()` discover them,
  `parse_module_path()` and `parse_kernel_path()` recover the identity each name encodes, all six in
  `orchestration/jobs.py`. `partition_events()`, `get_event_timestamps()`, and `get_event_data()` read the table through
  the `ExtractedDataColumns` enumeration rather than through string literals. A downstream consumer reads the extracted
  data through these rather than reimplementing the naming convention and the schema.
- **Frozen dataclasses**: Inner data classes (`ModuleSourceData`, `MicroControllerSourceData`, `ModuleExtractionConfig`,
  `KernelExtractionConfig`, `ControllerExtractionConfig`) use `frozen=True` for immutability and `slots=True` for
  performance. The top-level `MicroControllerManifest` and `ExtractionConfig` classes extend `YamlConfig` and are
  mutable.

### Code standards

- MyPy strict mode
- Ruff for formatting and linting
- Python 3.12, 3.13, 3.14 support
- See style skills for complete conventions

### Workflow guidance

- **MicroControllerInterface** (`microcontroller/interface.py`): the communication loop runs in `_runtime_cycle()`,
  a static method executed in a spawned daemon process, and a watchdog thread in the main process monitors liveness.
  Commands flow from the main process through an `MPQueue` to the communication process, which requires an explicit
  `stop()` call. Test against microcontroller hardware or in test mode.
- **Status codes** (`microcontroller/status_codes.py`): the Communication and TransportLayer codes never arrive as
  event codes and are read out of the two-byte payload every reception and transmission fault carries.
  `TransportStatusCodes` is distinct from `TransportLayerStatus` in ataraxis-transport-layer-pc, which covers the PC
  side and uses different values.
- **ModuleInterface** (`microcontroller/interface.py`): subclasses must implement `initialize_remote_assets()`,
  `terminate_remote_assets()`, and `process_received_data()`. `send_command()` and `send_parameters()` use
  LRU-cached message construction, and `reset_command_queue()` sends a dequeue command. See
  `examples/example_interface.py` for a reference subclass.
- **Serial communication** (`communication/protocols.py`, `messages.py`, `serial.py`): `SerialProtocols`
  (13 protocols) and `SerialPrototypes` (252 prototypes) define the protocol layer. Command classes pack bytes into
  the `packed_data` field their `__post_init__` populates, and reception classes parse header bytes via properties.
- **MQTT communication** (`communication/mqtt.py`): `paho-mqtt` v2 client with callback reception into a `Queue`.
  `get_data()` returns `(topic, message)` tuples or `None`, and the `has_data` property checks queue state.
- **Data classes and manifests** (`microcontroller/dataclasses.py`): inner classes are frozen, so create new instances
  rather than mutating. `MicroControllerManifest` and `ExtractionConfig` are mutable `YamlConfig` subclasses read and
  written through the inherited `from_yaml()` and `to_yaml()`, which creates the parent directory itself.
  `create_extraction_config()` builds a precursor config with empty event codes.
- **Log data extraction** (`microcontroller/log_processing.py`): `extract_logged_microcontroller_data()` returns
  columnar data and writes nothing, so the caller owns the output layout. Extraction rejects any data message whose
  payload size disagrees with the size its prototype code declares, because the extracted feather stores that
  prototype's dtype alongside the raw payload bytes and a mismatched pair cannot be decoded.
- **Extracted data access** (`microcontroller/extracted_data.py`): reading an extracted feather goes through
  `partition_events()` and then `get_event_timestamps()` or `get_event_data()`. `get_event_data()` decodes a whole
  event stream through one buffer read, which the firmware's one-data-type-per-event-code guarantee permits.
- **Orchestration** (`orchestration/`): `run_log_processing_pipeline()` runs one recording sequentially, or the single
  job a caller names by its canonical identifier. The `config` parameter is a `Path` loaded inside the job, and the
  tracker universe comes from the manifest rather than from the config, so a config requesting a subset never resets its
  sibling jobs. `execute_job()` wraps the work in `ProcessingTracker.run_job()`, which records the start, the
  completion, and the failure.
- **CLI** (`interfaces/cli.py`): use `console.echo()` for output, including for errors, which are reported at
  `LogLevel.ERROR` so a failed command exits zero rather than raising. The `config` subgroup demonstrates nested Click
  command groups.
- **MCP tools** (`interfaces/*_tools.py`): register on the shared instance from `interfaces/mcp_instance.py` via
  `@mcp.tool()`, add new tool modules to the side-effect import list in `interfaces/mcp_server.py`, and return
  JSON-serializable `dict[str, Any]`. The two discovery tools `list_microcontrollers_tool` and `check_mqtt_broker_tool`
  return a preformatted `str`. Execution uses `JobExecutionState` (`orchestration/execution.py`) with host-derived core
  and memory budgets, against which archive-derived per-job sizes are admitted. A read tool that lists items builds its
  response through `interfaces/responses.py`, which owns the bare, filtered, and detailed staging and the `rows`,
  `matched_rows`, `start_row`, and `next_start_row` paging fields.
