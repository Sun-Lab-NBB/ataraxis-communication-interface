"""Provides MCP tools for verifying processing output, querying extracted events, and cleaning processing output."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, convert_time
from ataraxis_data_structures import delete_directory

from .mcp_instance import mcp, read_tracker_status
from ..orchestration import OutputLayout
from ..microcontroller import ExtractedDataColumns

_MINIMUM_ROWS_FOR_INTERVALS: int = 2
"""The minimum number of rows required in a feather file to compute inter-event timing intervals."""


@mcp.tool()
def verify_processing_output_tool(output_directory: str) -> dict[str, Any]:
    """Verifies the completeness and schema correctness of processed microcontroller data output.

    Scans the ``microcontroller_data/`` subdirectory under the specified output directory for feather files produced
    by the log processing pipeline. For each feather file, validates the expected 5-column schema (timestamp_us,
    command, event, dtype, data) and reports row counts. Also reads the processing tracker to report job statuses
    alongside the output file inventory.

    Args:
        output_directory: The absolute path to the output directory containing a ``microcontroller_data/``
            subdirectory with processed output.

    Returns:
        A dictionary containing a 'verified' flag, the 'output_directory' and 'data_path', per-file results in 'files'
        (each with path, filename, type, schema validity, row count, and column names, plus an 'error' for a file that
        cannot be read and 'missing_columns' and 'extra_columns' for a file whose schema mismatches), tracker status
        in 'tracker', and a 'total_files' count. Returns an error dictionary if the output directory is missing, is
        not a directory, or holds no ``microcontroller_data/`` subdirectory.
    """
    output_path = Path(output_directory)

    if not output_path.exists():
        return {"error": f"Directory does not exist: {output_directory}"}

    if not output_path.is_dir():
        return {"error": f"Path is not a directory: {output_directory}"}

    data_path = output_path / OutputLayout.DIRECTORY_NAME

    if not data_path.exists():
        return {
            "error": (
                f"No '{OutputLayout.DIRECTORY_NAME}' subdirectory found under '{output_directory}'. "
                f"Processing may not have been run yet."
            ),
        }

    expected_columns = set(ExtractedDataColumns)
    file_results: list[dict[str, Any]] = []
    all_valid = True

    feather_files = sorted(data_path.glob(f"*{OutputLayout.FILE_SUFFIX}"))

    for feather_file in feather_files:
        entry: dict[str, Any] = {"file": str(feather_file), "filename": feather_file.name}

        # Classifies each output file as module or kernel from its filename infix.
        name = feather_file.stem
        if OutputLayout.KERNEL_INFIX in name:
            entry["type"] = "kernel"
        elif OutputLayout.MODULE_INFIX in name:
            entry["type"] = "module"
        else:
            entry["type"] = "unknown"

        try:
            # Reads the schema block and a projected row count, so the payload column stays on disk. The verification
            # below reads the column names and the row count alone, which both of these supply.
            schema = pl.read_ipc_schema(source=feather_file)
            row_count = pl.scan_ipc(source=feather_file).select(pl.len()).collect().item()
        except Exception as error:
            entry["valid"] = False
            entry["error"] = f"Unable to read feather file: {error}"
            all_valid = False
            file_results.append(entry)
            continue

        columns = list(schema)
        actual_columns = set(columns)
        schema_valid = actual_columns == expected_columns

        entry["valid"] = schema_valid
        entry["columns"] = columns
        entry["row_count"] = row_count

        if not schema_valid:
            missing = expected_columns - actual_columns
            extra = actual_columns - expected_columns
            if missing:
                entry["missing_columns"] = sorted(missing)
            if extra:
                entry["extra_columns"] = sorted(extra)
            all_valid = False

        file_results.append(entry)

    # Reads tracker status if available.
    tracker_path = data_path / OutputLayout.TRACKER_FILENAME
    tracker_info: dict[str, Any] = {}
    if tracker_path.exists():
        try:
            tracker_info = read_tracker_status(tracker_path=tracker_path)
        except Exception:
            tracker_info = {"error": "Unable to read tracker file."}

    return {
        "verified": all_valid and bool(feather_files),
        "output_directory": output_directory,
        "data_path": str(data_path),
        "files": file_results,
        "total_files": len(file_results),
        "tracker": tracker_info,
    }


@mcp.tool()
def query_extracted_events_tool(
    feather_files: list[str],
    max_sample_rows: int = 10,
) -> dict[str, Any]:
    """Reads one or more processed microcontroller data feather files and returns event distribution, timing
    statistics, and sample rows.

    For each file, computes the total row count, time range, per-event-code frequency distribution, per-command-code
    frequency distribution, and inter-event timing statistics. Also returns a configurable number of sample rows
    (head of the file) with binary data payloads omitted for readability. Accepts feather file paths from the
    'files' list returned by verify_processing_output_tool.

    Args:
        feather_files: The list of absolute paths to feather files produced by the log processing pipeline.
            Expected filename pattern: ``controller_{source_id}_module_{type}_{id}.feather`` or
            ``controller_{source_id}_kernel.feather``.
        max_sample_rows: The maximum number of sample rows to include per file. Must not be negative. Defaults to 10.

    Returns:
        A dictionary containing a 'results' list with per-file statistics (each with 'file', 'summary',
        'event_distribution', 'command_distribution', 'inter_event_timing', and 'sample_rows' keys) and a
        'total_files' count. Files that cannot be read produce an entry with 'file' and 'error' keys. Returns an error
        dictionary if max_sample_rows is negative.
    """
    # Validates the sample bound before any file is read, so an out-of-range argument cannot discard the statistics
    # already computed for the preceding files.
    if max_sample_rows < 0:
        return {"error": f"Invalid max_sample_rows: {max_sample_rows}. The value must not be negative."}

    results = [
        _analyze_single_event_feather(feather_file=feather_file, max_sample_rows=max_sample_rows)
        for feather_file in feather_files
    ]

    return {"results": results, "total_files": len(results)}


@mcp.tool()
def clean_log_processing_output_tool(output_directories: list[str]) -> dict[str, Any]:
    """Deletes the microcontroller_data subdirectory under one or more output directories.

    Removes each ``microcontroller_data/`` subdirectory and all of its contents, including processed output files
    and the processing tracker. Uses ``delete_directory`` from ataraxis-data-structures for parallel file deletion
    with platform-safe retry logic. After cleanup, the output directories can be passed to
    prepare_log_processing_batch_tool to reinitialize from scratch. Accepts the same output directory paths that were
    supplied to prepare_log_processing_batch_tool.

    Args:
        output_directories: The list of absolute paths to output directories containing ``microcontroller_data/``
            subdirectories to delete.

    Returns:
        A dictionary containing a 'results' list with per-directory outcomes (each with 'output_directory' and a
        'cleaned' flag, carrying 'data_path' on a successful deletion, both 'data_path' and 'error' on a failed
        deletion, 'error' alone for a path that does not resolve to a directory, and 'message' when there is nothing
        to clean), a 'total_cleaned' count, and a 'total_directories' count.
    """
    results = [_clean_single_output(output_directory=directory) for directory in output_directories]
    total_cleaned = sum(1 for result in results if result.get("cleaned", False))

    return {"results": results, "total_cleaned": total_cleaned, "total_directories": len(results)}


def _clean_single_output(output_directory: str) -> dict[str, Any]:
    """Deletes the microcontroller_data subdirectory under a single output directory.

    Args:
        output_directory: The absolute path to the output directory.

    Returns:
        A dictionary containing 'output_directory' and a 'cleaned' flag, carrying 'data_path' on a successful
        deletion, both 'data_path' and 'error' on a failed deletion, 'error' alone for a path that does not resolve
        to a directory, and 'message' when there is nothing to clean.
    """
    output_path = Path(output_directory)

    if not output_path.exists():
        return {"output_directory": output_directory, "cleaned": False, "error": "Directory does not exist."}

    if not output_path.is_dir():
        return {"output_directory": output_directory, "cleaned": False, "error": "Path is not a directory."}

    data_path = output_path / OutputLayout.DIRECTORY_NAME

    if not data_path.exists():
        return {"output_directory": output_directory, "cleaned": True, "message": "Nothing to clean."}

    try:
        delete_directory(directory_path=data_path)
    except Exception as error:
        return {
            "output_directory": output_directory,
            "cleaned": False,
            "data_path": str(data_path),
            "error": f"Unable to delete: {error}",
        }

    return {"output_directory": output_directory, "cleaned": True, "data_path": str(data_path)}


def _analyze_single_event_feather(
    feather_file: str,
    max_sample_rows: int,
) -> dict[str, Any]:
    """Reads a single microcontroller data feather file and computes event statistics.

    Args:
        feather_file: The absolute path to the feather file.
        max_sample_rows: The maximum number of sample rows to include.

    Returns:
        A dictionary containing 'file', 'summary', 'event_distribution', 'command_distribution',
        'inter_event_timing', and 'sample_rows' keys, or 'file' and 'error' keys if the file cannot be read. For an
        empty file, 'summary' contains only 'total_rows' (0) and the distribution and timing collections are empty.
    """
    file_path = Path(feather_file)

    if not file_path.exists():
        return {"file": feather_file, "error": f"File does not exist: {feather_file}"}

    if not file_path.is_file():
        return {"file": feather_file, "error": f"Path is not a file: {feather_file}"}

    try:
        schema = pl.read_ipc_schema(source=file_path)
    except Exception as error:
        return {"file": feather_file, "error": f"Unable to read feather file: {error}"}

    columns = list(schema)
    if ExtractedDataColumns.TIMESTAMP not in columns:
        return {"file": feather_file, "error": f"Missing required 'timestamp_us' column. Found: {columns}"}

    # Projects the fixed-width columns every statistic below reads, so the dtype and payload columns are loaded only
    # for the handful of sample rows the response carries.
    statistic_columns = [
        column
        for column in (ExtractedDataColumns.TIMESTAMP, ExtractedDataColumns.COMMAND, ExtractedDataColumns.EVENT)
        if column in columns
    ]
    dataframe = pl.scan_ipc(source=file_path).select(statistic_columns).collect()

    total_rows = dataframe.height

    if total_rows == 0:
        return {
            "file": feather_file,
            "summary": {"total_rows": 0},
            "event_distribution": [],
            "command_distribution": [],
            "inter_event_timing": {},
            "sample_rows": [],
        }

    # Extracts the timestamp column and computes basic recording statistics.
    timestamps = dataframe[ExtractedDataColumns.TIMESTAMP].to_numpy()
    first_timestamp_us = int(timestamps[0])
    last_timestamp_us = int(timestamps[-1])
    duration_us = last_timestamp_us - first_timestamp_us
    duration_seconds = (
        round(
            convert_time(time=duration_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True),
            6,
        )
        if duration_us > 0
        else 0.0
    )

    summary: dict[str, Any] = {
        "total_rows": total_rows,
        "first_timestamp_us": first_timestamp_us,
        "last_timestamp_us": last_timestamp_us,
        "duration_us": duration_us,
        "duration_seconds": duration_seconds,
    }

    event_distribution: list[dict[str, Any]] = []
    if ExtractedDataColumns.EVENT in dataframe.columns:
        event_counts = dataframe.group_by(ExtractedDataColumns.EVENT).len().sort(ExtractedDataColumns.EVENT)
        event_distribution = [
            {"event_code": int(row[ExtractedDataColumns.EVENT]), "count": int(row["len"])}
            for row in event_counts.iter_rows(named=True)
        ]

    command_distribution: list[dict[str, Any]] = []
    if ExtractedDataColumns.COMMAND in dataframe.columns:
        command_counts = dataframe.group_by(ExtractedDataColumns.COMMAND).len().sort(ExtractedDataColumns.COMMAND)
        command_distribution = [
            {"command_code": int(row[ExtractedDataColumns.COMMAND]), "count": int(row["len"])}
            for row in command_counts.iter_rows(named=True)
        ]

    inter_event_timing: dict[str, Any] = {}
    if total_rows >= _MINIMUM_ROWS_FOR_INTERVALS:
        intervals_us = np.diff(timestamps).astype(np.int64)
        mean_us = round(float(np.mean(intervals_us)), 2)
        median_us = round(float(np.median(intervals_us)), 2)
        std_us = round(float(np.std(intervals_us)), 2)
        min_us = int(np.min(intervals_us))
        max_us = int(np.max(intervals_us))

        mean_ms = round(
            convert_time(time=mean_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True),
            4,
        )
        median_ms = round(
            convert_time(
                time=median_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.MILLISECOND, as_float=True
            ),
            4,
        )

        inter_event_timing = {
            "mean_us": mean_us,
            "median_us": median_us,
            "std_us": std_us,
            "min_us": min_us,
            "max_us": max_us,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
        }

    # Builds sample rows with binary data omitted for readability.
    sample_rows: list[dict[str, Any]] = []
    sample_count = min(max_sample_rows, total_rows)
    sample_df = pl.scan_ipc(source=file_path).head(sample_count).collect()

    for row in sample_df.iter_rows(named=True):
        sample_entry: dict[str, Any] = {ExtractedDataColumns.TIMESTAMP: int(row[ExtractedDataColumns.TIMESTAMP])}

        if ExtractedDataColumns.COMMAND in row:
            sample_entry[ExtractedDataColumns.COMMAND] = int(row[ExtractedDataColumns.COMMAND])
        if ExtractedDataColumns.EVENT in row:
            sample_entry[ExtractedDataColumns.EVENT] = int(row[ExtractedDataColumns.EVENT])
        if ExtractedDataColumns.DTYPE in row:
            sample_entry[ExtractedDataColumns.DTYPE] = row[ExtractedDataColumns.DTYPE]
        if ExtractedDataColumns.DATA in row:
            sample_entry["has_data"] = row[ExtractedDataColumns.DATA] is not None

        sample_rows.append(sample_entry)

    return {
        "file": feather_file,
        "summary": summary,
        "event_distribution": event_distribution,
        "command_distribution": command_distribution,
        "inter_event_timing": inter_event_timing,
        "sample_rows": sample_rows,
    }
