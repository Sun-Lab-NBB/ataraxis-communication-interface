from enum import StrEnum

import numpy as np
import polars as pl
from numpy.typing import NDArray as NDArray

from .log_processing import ExtractedMessages as ExtractedMessages

class ExtractedDataColumns(StrEnum):
    TIMESTAMP = "timestamp_us"
    COMMAND = "command"
    EVENT = "event"
    DTYPE = "dtype"
    DATA = "data"

def build_message_dataframe(messages: ExtractedMessages) -> pl.DataFrame: ...
def partition_events(module_dataframe: pl.DataFrame) -> dict[int, pl.DataFrame]: ...
def get_event_timestamps(partition: dict[int, pl.DataFrame], event_code: int) -> NDArray[np.uint64]: ...
def get_event_data[ScalarT: np.generic](
    partition: dict[int, pl.DataFrame], event_code: int, values_dtype: type[ScalarT]
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]: ...
