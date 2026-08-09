"""Provides the schema of the extracted message tables, the writer that builds them, and the primitives that read
them back.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .log_processing import ExtractedMessages


class ExtractedDataColumns(StrEnum):
    """Defines the columns every extracted message table carries, in the order the table stores them.

    Notes:
        Each member is a string, so a member indexes a table, names a series, and serializes into a report wherever
        the raw column name would. Iterating the enumeration yields the complete column set in storage order.
    """

    TIMESTAMP = "timestamp_us"
    """Holds the microseconds elapsed since the UTC epoch onset when each message arrived."""
    COMMAND = "command"
    """Holds the command code each message was sent under."""
    EVENT = "event"
    """Holds the event code of each message."""
    DTYPE = "dtype"
    """Holds the numpy dtype string of each message's data payload, or null for a state-only message."""
    DATA = "data"
    """Holds the raw payload bytes of each message, or null for a state-only message."""


def build_message_dataframe(messages: ExtractedMessages) -> pl.DataFrame:
    """Builds a polars DataFrame from an extracted columnar message block.

    Notes:
        The dtype column stores the numpy dtype string of each message's data payload, which allows a consumer to
        reconstruct the original array through ``np.frombuffer(payload, dtype=dtype_str)`` without depending on this
        library.

    Args:
        messages: The columnar message data to serialize.

    Returns:
        A polars DataFrame carrying the extracted message columns, with the timestamp stored as UInt64, the command
        and the event as UInt8, the dtype as String, and the data as Binary.
    """
    return pl.DataFrame(
        {
            ExtractedDataColumns.TIMESTAMP: pl.Series(values=messages.timestamps, dtype=pl.UInt64),
            ExtractedDataColumns.COMMAND: pl.Series(values=messages.commands, dtype=pl.UInt8),
            ExtractedDataColumns.EVENT: pl.Series(values=messages.events, dtype=pl.UInt8),
            ExtractedDataColumns.DTYPE: pl.Series(values=list(messages.dtypes), dtype=pl.String),
            ExtractedDataColumns.DATA: pl.Series(values=list(messages.data_payloads), dtype=pl.Binary),
        }
    )


def partition_events(module_dataframe: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """Partitions an extracted message table into one sub-table per event code, in a single pass.

    Args:
        module_dataframe: The message table read from a feather file this library wrote.

    Returns:
        The sub-table of each event code the table holds, keyed by that event code.
    """
    # Partitioning on a single column still keys the result by a one-element tuple, so the event code is always the
    # first element of the key.
    raw_partition = module_dataframe.partition_by(ExtractedDataColumns.EVENT, as_dict=True)
    return {int(key[0]): value for key, value in raw_partition.items()}


def get_event_timestamps(partition: dict[int, pl.DataFrame], event_code: int) -> NDArray[np.uint64]:
    """Reads the arrival timestamps of every message carrying the target event code.

    Notes:
        Serves a state-only event, whose messages carry a timestamp alone.

    Args:
        partition: The event-code-keyed partition produced by partition_events().
        event_code: The event code to look up.

    Returns:
        The timestamps of the requested event code, empty when the partition holds no such code.
    """
    event_dataframe = partition.get(event_code)
    if event_dataframe is None:
        return np.array([], dtype=np.uint64)
    return event_dataframe[ExtractedDataColumns.TIMESTAMP].to_numpy().astype(np.uint64)


def get_event_data[ScalarT: np.generic](
    partition: dict[int, pl.DataFrame],
    event_code: int,
    values_dtype: type[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]:
    """Reads the arrival timestamps and the decoded data values of every message carrying the target event code.

    Notes:
        The firmware assigns each event code a single data object type, so every message sharing an event code also
        shares a payload dtype. That lets the payloads of a whole event stream be concatenated and decoded through
        one buffer read rather than one read per message.

    Args:
        partition: The event-code-keyed partition produced by partition_events().
        event_code: The event code to look up.
        values_dtype: The numpy scalar type the decoded values are cast to.

    Returns:
        The timestamps of the requested event code and the values decoded from its payloads, both empty when the
        partition holds no such code.
    """
    event_dataframe = partition.get(event_code)
    if event_dataframe is None:
        return np.array([], dtype=np.uint64), np.array([], dtype=values_dtype)

    timestamps: NDArray[np.uint64] = event_dataframe[ExtractedDataColumns.TIMESTAMP].to_numpy().astype(np.uint64)

    payloads = event_dataframe[ExtractedDataColumns.DATA].to_list()
    payload_dtype = event_dataframe[ExtractedDataColumns.DTYPE].to_list()[0]
    values: NDArray[ScalarT] = np.frombuffer(b"".join(payloads), dtype=payload_dtype).astype(values_dtype)

    return timestamps, values
