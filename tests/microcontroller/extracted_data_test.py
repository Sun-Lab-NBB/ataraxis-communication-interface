"""Contains tests for the classes and functions provided by the extracted_data.py module."""

import json

import numpy as np
import polars as pl
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_communication_interface.microcontroller.extracted_data import (
    ExtractedDataColumns,
    get_event_data,
    partition_events,
    get_event_timestamps,
    build_message_dataframe,
)
from ataraxis_communication_interface.microcontroller.log_processing import (
    ExtractedMessages,
    _create_accumulator,
    _finalize_accumulator,
)


def _table(rows: list[tuple[int, int, int, str | None, bytes | None]]) -> pl.DataFrame:
    """Builds an extracted message table from the requested (timestamp, command, event, dtype, data) rows."""
    return pl.DataFrame(
        {
            ExtractedDataColumns.TIMESTAMP: pl.Series(
                name=ExtractedDataColumns.TIMESTAMP, values=[row[0] for row in rows], dtype=pl.UInt64
            ),
            ExtractedDataColumns.COMMAND: pl.Series(
                name=ExtractedDataColumns.COMMAND, values=[row[1] for row in rows], dtype=pl.UInt8
            ),
            ExtractedDataColumns.EVENT: pl.Series(
                name=ExtractedDataColumns.EVENT, values=[row[2] for row in rows], dtype=pl.UInt8
            ),
            ExtractedDataColumns.DTYPE: pl.Series(
                name=ExtractedDataColumns.DTYPE, values=[row[3] for row in rows], dtype=pl.String
            ),
            ExtractedDataColumns.DATA: pl.Series(
                name=ExtractedDataColumns.DATA, values=[row[4] for row in rows], dtype=pl.Binary
            ),
        }
    )


def test_extracted_data_columns() -> None:
    """Verifies that the column enumeration names the extracted message table columns in storage order."""
    assert tuple(ExtractedDataColumns) == ("timestamp_us", "command", "event", "dtype", "data")
    # Every member hashes as its own value, so a member and the raw column name are interchangeable as a set entry,
    # as a mapping key, and in a serialized report.
    assert set(ExtractedDataColumns) == {"timestamp_us", "command", "event", "dtype", "data"}
    assert json.dumps({ExtractedDataColumns.EVENT: 10}) == '{"event": 10}'


def test_partition_events() -> None:
    """Verifies that partition_events splits an extracted table into one sub-table per event code."""
    table = _table(
        [
            (100, 1, 10, None, None),
            (200, 1, 20, None, None),
            (300, 2, 10, None, None),
        ]
    )

    partition = partition_events(module_dataframe=table)

    assert set(partition) == {10, 20}
    assert partition[10].height == 2
    assert partition[20].height == 1
    assert list(partition[10][ExtractedDataColumns.TIMESTAMP]) == [100, 300]


def test_partition_events_empty_table() -> None:
    """Verifies that partition_events returns no partitions for a table holding no messages."""
    assert partition_events(module_dataframe=_table([])) == {}


def test_get_event_timestamps() -> None:
    """Verifies that get_event_timestamps returns the arrival timestamps of the requested event code."""
    partition = partition_events(module_dataframe=_table([(100, 1, 10, None, None), (300, 1, 10, None, None)]))

    timestamps = get_event_timestamps(partition=partition, event_code=10)

    assert timestamps.dtype == np.uint64
    assert list(timestamps) == [100, 300]


def test_get_event_timestamps_absent_code() -> None:
    """Verifies that get_event_timestamps returns an empty array for an event code the partition does not hold."""
    partition = partition_events(module_dataframe=_table([(100, 1, 10, None, None)]))

    timestamps = get_event_timestamps(partition=partition, event_code=99)

    assert timestamps.dtype == np.uint64
    assert timestamps.size == 0


def test_get_event_data() -> None:
    """Verifies that get_event_data decodes an event stream's payloads through the dtype the table records."""
    payloads = [np.uint16(value).tobytes() for value in (172, 5, 61_000)]
    table = _table([(100 * (index + 1), 1, 20, "uint16", payload) for index, payload in enumerate(payloads)])

    timestamps, values = get_event_data(
        partition=partition_events(module_dataframe=table), event_code=20, values_dtype=np.uint32
    )

    assert timestamps.dtype == np.uint64
    assert list(timestamps) == [100, 200, 300]
    assert values.dtype == np.uint32
    assert list(values) == [172, 5, 61_000]


def test_get_event_data_absent_code() -> None:
    """Verifies that get_event_data returns empty arrays for an event code the partition does not hold."""
    table = _table([(100, 1, 20, "uint16", np.uint16(7).tobytes())])

    timestamps, values = get_event_data(
        partition=partition_events(module_dataframe=table), event_code=99, values_dtype=np.float32
    )

    assert timestamps.dtype == np.uint64
    assert timestamps.size == 0
    assert values.dtype == np.float32
    assert values.size == 0


def test_get_event_data_array_prototype() -> None:
    """Verifies that get_event_data returns one row of values per message for an array prototype event code."""
    payloads = [np.array(values, dtype=np.uint16).tobytes() for values in ((1, 2), (3, 4), (5, 6))]
    table = _table([(100 * (index + 1), 1, 20, "uint16", payload) for index, payload in enumerate(payloads)])

    timestamps, values = get_event_data(
        partition=partition_events(module_dataframe=table), event_code=20, values_dtype=np.uint32
    )

    assert values.dtype == np.uint32
    assert values.shape == (3, 2)
    assert len(values) == len(timestamps)
    assert values.tolist() == [[1, 2], [3, 4], [5, 6]]

    # A scalar prototype has its trailing value axis squeezed, so the same three messages carrying one value each
    # still decode into a 1-D array holding one value per timestamp.
    scalar_payloads = [np.uint16(value).tobytes() for value in (1, 3, 5)]
    scalar_table = _table(
        [(100 * (index + 1), 1, 20, "uint16", payload) for index, payload in enumerate(scalar_payloads)]
    )

    scalar_timestamps, scalar_values = get_event_data(
        partition=partition_events(module_dataframe=scalar_table), event_code=20, values_dtype=np.uint32
    )

    assert scalar_values.shape == (3,)
    assert len(scalar_values) == len(scalar_timestamps)
    assert scalar_values.tolist() == [1, 3, 5]


def test_get_event_data_state_only_event() -> None:
    """Verifies that get_event_data refuses to read the data values of a state-only event code."""
    messages = ExtractedMessages(
        timestamps=np.array([100, 200], dtype=np.uint64),
        commands=np.array([1, 1], dtype=np.uint8),
        events=np.array([20, 20], dtype=np.uint8),
        dtypes=(None, None),
        data_payloads=(None, None),
    )
    partition = partition_events(module_dataframe=build_message_dataframe(messages=messages))

    message = (
        "Unable to read the data values of the messages carrying event code 20. The messages of this event code "
        "store no data payload, which marks the code as a state-only event. Read the arrival timestamps of a "
        "state-only event through get_event_timestamps()."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        get_event_data(partition=partition, event_code=20, values_dtype=np.uint32)


def test_get_event_data_null_payload_inside_decodable_stream() -> None:
    """Verifies that get_event_data refuses to decode a stream mixing a null payload into typed payloads."""
    messages = ExtractedMessages(
        timestamps=np.array([100, 200], dtype=np.uint64),
        commands=np.array([1, 1], dtype=np.uint8),
        events=np.array([20, 20], dtype=np.uint8),
        dtypes=("uint16", None),
        data_payloads=(np.uint16(172).tobytes(), None),
    )
    partition = partition_events(module_dataframe=build_message_dataframe(messages=messages))

    message = (
        "Unable to read the data values of the messages carrying event code 20. At least one of these messages "
        "stores a null payload while the rest store data under the uint16 dtype, which marks the payload-free "
        "messages as carrying a prototype code this library does not recognize."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        get_event_data(partition=partition, event_code=20, values_dtype=np.uint32)


def test_get_event_data_non_uniform_dtype() -> None:
    """Verifies that get_event_data refuses to decode an event stream storing data under more than one dtype."""
    table = _table(
        [
            (100, 1, 20, "uint16", np.uint16(172).tobytes()),
            (200, 1, 20, "uint32", np.uint32(5).tobytes()),
        ]
    )

    message = (
        "Unable to read the data values of the messages carrying event code 20. The messages of this event code "
        "store data under more than one dtype: ['uint16', 'uint32']. The firmware assigns each event code a single "
        "data object type, so a code storing several dtypes marks a table this library did not write or a firmware "
        "revision that reused the code."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        get_event_data(partition=partition_events(module_dataframe=table), event_code=20, values_dtype=np.uint32)


def test_get_event_data_indivisible_value_count() -> None:
    """Verifies that get_event_data refuses to pair a value count that is not a whole multiple of the message count."""
    # The extracted schema stores the dtype string but not the element count, so a stream whose messages share a dtype
    # while carrying different element counts is caught by the value count alone.
    table = _table(
        [
            (100, 1, 20, "uint16", np.uint16(172).tobytes()),
            (200, 1, 20, "uint16", np.array((5, 61_000), dtype=np.uint16).tobytes()),
        ]
    )

    message = (
        "Unable to pair the data values of the messages carrying event code 20 with their arrival timestamps. The "
        "payloads of this event code decode into 3 values, which is not a whole multiple of the 2 messages the table "
        "stores for the code."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        get_event_data(partition=partition_events(module_dataframe=table), event_code=20, values_dtype=np.uint32)


def test_build_message_dataframe() -> None:
    """Verifies that build_message_dataframe builds a correctly typed polars DataFrame."""
    messages = ExtractedMessages(
        timestamps=np.array([100, 200], dtype=np.uint64),
        commands=np.array([1, 2], dtype=np.uint8),
        events=np.array([10, 20], dtype=np.uint8),
        dtypes=("uint8", None),
        data_payloads=(b"\x2a", None),
    )

    dataframe = build_message_dataframe(messages=messages)

    assert isinstance(dataframe, pl.DataFrame)
    assert dataframe.shape == (2, 5)
    assert dataframe.columns == ["timestamp_us", "command", "event", "dtype", "data"]
    assert dataframe["timestamp_us"].dtype == pl.UInt64
    assert dataframe["command"].dtype == pl.UInt8
    assert dataframe["event"].dtype == pl.UInt8
    assert dataframe["dtype"].dtype == pl.String
    assert dataframe["data"].dtype == pl.Binary
    assert dataframe["timestamp_us"][0] == 100
    assert dataframe["command"][0] == 1
    assert dataframe["event"][0] == 10
    assert dataframe["dtype"][0] == "uint8"
    assert dataframe["data"][0] == b"\x2a"
    assert dataframe["dtype"][1] is None
    assert dataframe["data"][1] is None


def test_build_message_dataframe_empty() -> None:
    """Verifies that build_message_dataframe builds an empty DataFrame from an empty columnar block."""
    dataframe = build_message_dataframe(messages=_finalize_accumulator(accumulator=_create_accumulator()))

    assert dataframe.shape == (0, 5)
    assert dataframe.columns == ["timestamp_us", "command", "event", "dtype", "data"]
