"""Provides MCP tools for reading, writing, and validating extraction configuration files."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from .mcp_instance import mcp
from ..microcontroller.dataclasses import (
    ExtractionConfig,
    KernelExtractionConfig,
    ModuleExtractionConfig,
    MicroControllerManifest,
    ControllerExtractionConfig,
)


@mcp.tool()
def read_extraction_config_tool(config_path: str) -> dict[str, Any]:
    """Reads an extraction configuration from a YAML file and returns its contents.

    Parses the ExtractionConfig file and returns a structured dictionary representation of all controller,
    module, and kernel extraction settings.

    Args:
        config_path: The absolute path to the extraction configuration YAML file.

    Returns:
        A dictionary containing the config path, a list of controller entries with their modules and
        kernel settings, and the total controller count. Returns an error dictionary if the file is
        missing or cannot be parsed.
    """
    path = Path(config_path)

    if not path.exists():
        return {"error": f"Config file not found: {config_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {config_path}"}

    try:
        config = ExtractionConfig.load(file_path=path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to read extraction config: {error}"}

    # Serializes each controller's modules, event codes, and optional kernel settings into dictionaries.
    controllers: list[dict[str, Any]] = []
    for controller in config.controllers:
        module_entries = [
            {
                "module_type": module.module_type,
                "module_id": module.module_id,
                "event_codes": list(module.event_codes),
            }
            for module in controller.modules
        ]

        controller_entry: dict[str, Any] = {
            "controller_id": controller.controller_id,
            "modules": module_entries,
        }

        # Includes kernel event codes when kernel extraction is configured for this controller.
        if controller.kernel is not None:
            controller_entry["kernel"] = {
                "event_codes": list(controller.kernel.event_codes),
            }
        else:
            controller_entry["kernel"] = None

        controllers.append(controller_entry)

    return {
        "config_path": config_path,
        "controllers": controllers,
        "total_controllers": len(controllers),
    }


@mcp.tool()
def write_extraction_config_tool(config_path: str, controllers: list[dict[str, Any]]) -> dict[str, Any]:
    """Writes an extraction configuration to a YAML file from structured controller data.

    Accepts a list of controller dictionaries, constructs an ExtractionConfig instance, and serializes it
    to the specified YAML file path. Each controller dictionary must contain 'controller_id' and 'modules'
    keys. Each module must have 'module_type' (type code), 'module_id' (ID code), and 'event_codes' keys.
    An optional 'kernel' key may contain a dictionary with 'event_codes'.

    Args:
        config_path: The absolute path where the extraction configuration YAML file will be written.
        controllers: A list of controller dictionaries, each with 'controller_id', 'modules' (list of
            dicts with 'module_type' (type code), 'module_id' (ID code), 'event_codes'), and optionally
            'kernel' (dict with 'event_codes').

    Returns:
        A dictionary containing a 'success' flag, the config file path, and the controller count.
        Returns an error dictionary if the input data is invalid or the file cannot be written.
    """
    # Converts raw controller dictionaries into typed ExtractionConfig dataclasses.
    try:
        controller_configs: list[ControllerExtractionConfig] = []
        for controller_dict in controllers:
            module_configs = tuple(
                ModuleExtractionConfig(
                    module_type=int(module["module_type"]),
                    module_id=int(module["module_id"]),
                    event_codes=tuple(int(code) for code in module["event_codes"]),
                )
                for module in controller_dict["modules"]
            )

            # Constructs the optional kernel extraction config when present.
            kernel_config = None
            kernel_data = controller_dict.get("kernel")
            if kernel_data is not None:
                kernel_config = KernelExtractionConfig(
                    event_codes=tuple(int(code) for code in kernel_data["event_codes"]),
                )

            controller_configs.append(
                ControllerExtractionConfig(
                    controller_id=int(controller_dict["controller_id"]),
                    modules=module_configs,
                    kernel=kernel_config,
                )
            )

        config = ExtractionConfig(controllers=controller_configs)
    except (KeyError, TypeError, ValueError) as error:
        return {"error": f"Invalid controller data: {error}"}

    output = Path(config_path)

    try:
        config.save(file_path=output)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to write extraction config: {error}"}

    return {
        "success": True,
        "config_path": config_path,
        "controller_count": len(controller_configs),
    }


@mcp.tool()
def validate_extraction_config_tool(
    config_path: str,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Validates an extraction configuration for structural correctness and optionally cross-references it against
    a microcontroller manifest.

    Checks that every controller has at least one extraction target (modules or kernel). Verifies that all module
    and kernel entries have non-empty event codes without duplicates. Confirms that module (type, id) pairs are
    unique within each controller. When a manifest path is provided, additionally verifies that every controller ID
    and module identifier in the config matches a registered entry in the manifest.

    Args:
        config_path: The absolute path to the extraction configuration YAML file to validate.
        manifest_path: An optional absolute path to the microcontroller_manifest.yaml file. When provided, enables
            cross-referencing controller IDs and module identifiers against the manifest.

    Returns:
        A dictionary containing a 'valid' flag, a 'config_path' key, a list of 'errors' (empty when valid), and
        a 'summary' with controller and module counts. Returns an error dictionary if the file cannot be read.
    """
    path = Path(config_path)

    if not path.exists():
        return {"error": f"Config file not found: {config_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {config_path}"}

    try:
        config = ExtractionConfig.load(file_path=path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"Unable to parse extraction config: {error}"}

    errors: list[str] = []
    total_modules = 0

    if not config.controllers:
        errors.append("Config contains no controller entries.")

    for controller in config.controllers:
        controller_label = f"Controller {controller.controller_id}"
        has_modules = bool(controller.modules)
        has_kernel = controller.kernel is not None

        if not has_modules and not has_kernel:
            errors.append(f"{controller_label}: No modules and no kernel configured. At least one is required.")

        seen_module_keys: set[tuple[int, int]] = set()
        for module in controller.modules:
            total_modules += 1
            module_label = f"{controller_label}, module ({module.module_type}, {module.module_id})"
            module_key = (module.module_type, module.module_id)

            if module_key in seen_module_keys:
                errors.append(f"{module_label}: Duplicate module (type, id) pair within this controller.")
            seen_module_keys.add(module_key)

            if not module.event_codes:
                errors.append(f"{module_label}: event_codes is empty.")
            elif len(module.event_codes) != len(set(module.event_codes)):
                errors.append(f"{module_label}: event_codes contains duplicates.")

        if controller.kernel is not None:
            kernel_label = f"{controller_label}, kernel"
            if not controller.kernel.event_codes:
                errors.append(f"{kernel_label}: event_codes is empty.")
            elif len(controller.kernel.event_codes) != len(set(controller.kernel.event_codes)):
                errors.append(f"{kernel_label}: event_codes contains duplicates.")

    # Cross-references against the manifest when provided.
    if manifest_path is not None:
        manifest_file = Path(manifest_path)

        if not manifest_file.exists():
            errors.append(f"Manifest file not found: {manifest_path}")
        elif not manifest_file.is_file():
            errors.append(f"Manifest path is not a file: {manifest_path}")
        else:
            try:
                manifest = MicroControllerManifest.load(file_path=manifest_file)
            except Exception as error:  # noqa: BLE001
                errors.append(f"Unable to read manifest for cross-referencing: {error}")
                manifest = None

            if manifest is not None:
                # Builds lookup structures from the manifest.
                manifest_controller_ids: set[int] = set()
                manifest_modules: dict[int, set[tuple[int, int]]] = {}

                for manifest_entry in manifest.controllers:
                    manifest_controller_ids.add(manifest_entry.id)
                    manifest_modules[manifest_entry.id] = {
                        (module.module_type, module.module_id) for module in manifest_entry.modules
                    }

                # Validates each config controller against the manifest.
                for config_controller in config.controllers:
                    controller_label = f"Controller {config_controller.controller_id}"

                    if config_controller.controller_id not in manifest_controller_ids:
                        errors.append(
                            f"{controller_label}: Not registered in manifest. "
                            f"Registered IDs: {sorted(manifest_controller_ids)}."
                        )
                        continue

                    registered_modules = manifest_modules.get(config_controller.controller_id, set())
                    for module in config_controller.modules:
                        module_key = (module.module_type, module.module_id)
                        if module_key not in registered_modules:
                            errors.append(
                                f"{controller_label}, module ({module.module_type}, {module.module_id}): "
                                f"Not registered in manifest for this controller."
                            )

    return {
        "valid": not errors,
        "config_path": config_path,
        "errors": errors,
        "summary": {
            "total_controllers": len(config.controllers),
            "total_modules": total_modules,
            "controllers_with_kernel": sum(1 for controller in config.controllers if controller.kernel is not None),
        },
    }
