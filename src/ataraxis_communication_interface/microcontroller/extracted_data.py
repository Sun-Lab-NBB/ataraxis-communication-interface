"""Provides the schema of the extracted message tables, the writer that builds them, and the primitives that read
them back.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from ataraxis_base_utilities import console

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
            ExtractedDataColumns.DTYPE: pl.Series(values=messages.dtypes, dtype=pl.String),
            ExtractedDataColumns.DATA: pl.Series(values=messages.data_payloads, dtype=pl.Binary),
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

        An event code declaring a scalar prototype has its trailing value axis squeezed, so it keeps returning a 1-D
        array holding one value per timestamp.

    Args:
        partition: The event-code-keyed partition produced by partition_events().
        event_code: The event code to look up.
        values_dtype: The numpy scalar type the decoded values are cast to.

    Returns:
        The timestamps of the requested event code and the values decoded from its payloads, both empty when the
        partition holds no such code. An event code declaring an array prototype yields one row of values per
        message, so the value array holds one row per timestamp.

    Raises:
        ValueError: If the requested event code is a state-only event, whose messages carry no data payload. If a
            message carrying the event code stores a null payload inside an otherwise decodable stream, which marks
            its prototype code as unrecognized. If the decoded value count is not a whole multiple of the message
            count.
    """
    event_dataframe = partition.get(event_code)
    if event_dataframe is None:
        return np.array([], dtype=np.uint64), np.array([], dtype=values_dtype)

    timestamps: NDArray[np.uint64] = event_dataframe[ExtractedDataColumns.TIMESTAMP].to_numpy().astype(np.uint64)

    payloads = event_dataframe[ExtractedDataColumns.DATA].to_list()
    payload_dtypes = event_dataframe[ExtractedDataColumns.DTYPE].to_list()
    payload_dtype = payload_dtypes[0]
    if payload_dtype is None:
        message = (
            f"Unable to read the data values of the messages carrying event code {event_code}. The messages of this "
            f"event code store no data payload, which marks the code as a state-only event. Read the arrival "
            f"timestamps of a state-only event through get_event_timestamps()."
        )
        console.error(message=message, error=ValueError)
    if any(payload is None for payload in payloads):
        message = (
            f"Unable to read the data values of the messages carrying event code {event_code}. At least one of these "
            f"messages stores a null payload while the rest store data under the {payload_dtype} dtype, which marks "
            f"the payload-free messages as carrying a prototype code this library does not recognize."
        )
        console.error(message=message, error=ValueError)
    if any(dtype != payload_dtype for dtype in payload_dtypes):
        message = (
            f"Unable to read the data values of the messages carrying event code {event_code}. The messages of this "
            f"event code store data under more than one dtype: {sorted(set(payload_dtypes))}. The firmware assigns "
            f"each event code a single data object type, so a code storing several dtypes marks a table this library "
            f"did not write or a firmware revision that reused the code."
        )
        console.error(message=message, error=ValueError)

    decoded_values: NDArray[ScalarT] = np.frombuffer(b"".join(payloads), dtype=payload_dtype).astype(values_dtype)
    if decoded_values.size % timestamps.size != 0:
        message = (
            f"Unable to pair the data values of the messages carrying event code {event_code} with their arrival "
            f"timestamps. The payloads of this event code decode into {decoded_values.size} values, which is not a "
            f"whole multiple of the {timestamps.size} messages the table stores for the code."
        )
        console.error(message=message, error=ValueError)

    # Reshapes the flat decode into one row per message, since an array prototype contributes multiple values per
    # message and the concatenated buffer preserves no boundary between them.
    values: NDArray[ScalarT] = decoded_values.reshape(timestamps.size, -1)
    if values.shape[1] == 1:
        values = values.reshape(-1)

    return timestamps, values
