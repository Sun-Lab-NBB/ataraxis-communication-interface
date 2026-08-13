"""Contains tests for the classes and functions provided by the log_processing.py module."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.log_archives import (
    DEFAULT_ONSET_US,
    create_test_archive,
    create_kernel_data_payload,
    create_module_data_payload,
    create_kernel_state_payload,
    create_module_state_payload,
)
from ataraxis_base_utilities import error_format
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX, PARALLEL_PROCESSING_THRESHOLD, LogArchiveReader

from ataraxis_communication_interface.communication import SerialPrototypes
from ataraxis_communication_interface.microcontroller import log_processing as log_processing_module
from ataraxis_communication_interface.microcontroller.log_processing import (
    ExtractedMessages,
    ExtractedModuleData,
    ExtractedControllerData,
    _finalize_batch,
    _ColumnAccumulator,
    _create_accumulator,
    _extend_accumulator,
    _create_accumulators,
    _finalize_accumulator,
    _process_message_batch,
    extract_logged_microcontroller_data,
)

_SOURCE_ID: int = 101
"""Stores the source ID used by every synthetic log archive built by this module."""

_MODULE_TYPE: int = 1
"""Stores the type code of the hardware module the module message builders of this file address."""

_MODULE_ID: int = 2
"""Stores the instance code of the hardware module the module message builders of this file address."""

_MODULE_KEY: tuple[int, int] = (_MODULE_TYPE, _MODULE_ID)
"""Stores the (type, id) pair the extraction filters and the returned module data are keyed by."""

_UINT16_PROTOTYPE_CODE: int = 7
"""Stores the prototype code that declares a single two-byte unsigned integer data object."""


def test_extracted_messages_count() -> None:
    """Verifies that the count property of ExtractedMessages reports the number of stored messages."""
    messages = ExtractedMessages(
        timestamps=np.array([100, 200, 300], dtype=np.uint64),
        commands=np.array([1, 2, 3], dtype=np.uint8),
        events=np.array([10, 20, 30], dtype=np.uint8),
        dtypes=(None, None, "uint8"),
        data_payloads=(None, None, b"\x01"),
    )

    assert messages.count == 3
    assert messages.dtypes[2] == "uint8"
    assert messages.data_payloads[2] == b"\x01"


def test_extracted_messages_count_empty() -> None:
    """Verifies that the count property of ExtractedMessages reports zero for an empty columnar block."""
    messages = ExtractedMessages(
        timestamps=np.array([], dtype=np.uint64),
        commands=np.array([], dtype=np.uint8),
        events=np.array([], dtype=np.uint8),
        dtypes=(),
        data_payloads=(),
    )

    assert messages.count == 0


def test_extracted_module_data() -> None:
    """Verifies that ExtractedModuleData stores the module identity alongside its columnar message block."""
    messages = ExtractedMessages(
        timestamps=np.array([100], dtype=np.uint64),
        commands=np.array([1], dtype=np.uint8),
        events=np.array([10], dtype=np.uint8),
        dtypes=(None,),
        data_payloads=(None,),
    )
    module_data = ExtractedModuleData(module_type=_MODULE_TYPE, module_id=_MODULE_ID, messages=messages)

    assert module_data.module_type == _MODULE_TYPE
    assert module_data.module_id == _MODULE_ID
    assert module_data.messages.count == 1


def test_extracted_controller_data() -> None:
    """Verifies that ExtractedControllerData stores the module blocks and the kernel block of one extraction pass."""
    kernel = _finalize_accumulator(accumulator=_build_accumulator(count=2))
    module_data = ExtractedModuleData(
        module_type=_MODULE_TYPE,
        module_id=_MODULE_ID,
        messages=_finalize_accumulator(accumulator=_build_accumulator(count=1)),
    )
    controller_data = ExtractedControllerData(modules=(module_data,), kernel=kernel)

    assert len(controller_data.modules) == 1
    assert controller_data.modules[0].module_id == _MODULE_ID
    assert controller_data.kernel.count == 2


def test_create_accumulator() -> None:
    """Verifies that _create_accumulator builds an accumulator holding no messages."""
    accumulator = _create_accumulator()

    assert accumulator.timestamps == []
    assert accumulator.commands == []
    assert accumulator.events == []
    assert accumulator.dtypes == []
    assert accumulator.data_payloads == []


def test_create_accumulators() -> None:
    """Verifies that _create_accumulators builds one empty accumulator per requested module."""
    accumulators = _create_accumulators(
        module_filters={_MODULE_KEY: frozenset({10}), (3, 4): frozenset({20})},
    )

    assert set(accumulators.keys()) == {_MODULE_KEY, (3, 4)}
    assert accumulators[_MODULE_KEY].timestamps == []
    assert accumulators[(3, 4)].data_payloads == []


def test_create_accumulators_without_filters() -> None:
    """Verifies that _create_accumulators builds no accumulators when module extraction is not requested."""
    assert _create_accumulators(module_filters=None) == {}


def test_extend_accumulator() -> None:
    """Verifies that _extend_accumulator appends every column of the source accumulator to the target."""
    target = _ColumnAccumulator(timestamps=[100], commands=[1], events=[10], dtypes=["uint8"], data_payloads=[b"\x01"])
    source = _ColumnAccumulator(timestamps=[200], commands=[2], events=[20], dtypes=[None], data_payloads=[None])

    _extend_accumulator(target=target, source=source)

    assert target.timestamps == [100, 200]
    assert target.commands == [1, 2]
    assert target.events == [10, 20]
    assert target.dtypes == ["uint8", None]
    assert target.data_payloads == [b"\x01", None]
    # The source accumulator is read-only, so extending the target must not alter it.
    assert source.timestamps == [200]


def test_finalize_accumulator() -> None:
    """Verifies that _finalize_accumulator converts the accumulator lists into typed numpy arrays and tuples."""
    accumulator = _ColumnAccumulator(
        timestamps=[100, 200, 300],
        commands=[1, 2, 3],
        events=[10, 20, 30],
        dtypes=[None, "uint8", None],
        data_payloads=[None, b"\x01", None],
    )

    result = _finalize_accumulator(accumulator=accumulator)

    assert isinstance(result, ExtractedMessages)
    assert result.count == 3
    assert result.timestamps.dtype == np.uint64
    assert result.commands.dtype == np.uint8
    assert result.events.dtype == np.uint8
    np.testing.assert_array_equal(result.timestamps, [100, 200, 300])
    np.testing.assert_array_equal(result.commands, [1, 2, 3])
    np.testing.assert_array_equal(result.events, [10, 20, 30])
    assert result.dtypes == (None, "uint8", None)
    assert result.data_payloads == (None, b"\x01", None)


def test_finalize_accumulator_empty() -> None:
    """Verifies that _finalize_accumulator converts an empty accumulator into empty typed arrays."""
    result = _finalize_accumulator(accumulator=_create_accumulator())

    assert result.count == 0
    assert result.timestamps.dtype == np.uint64
    assert result.dtypes == ()
    assert result.data_payloads == ()


def test_finalize_batch_skips_silent_modules() -> None:
    """Verifies that _finalize_batch reports only the modules that produced at least one matching message."""
    module_accumulators = {_MODULE_KEY: _build_accumulator(count=2), (3, 4): _create_accumulator()}

    result = _finalize_batch(
        module_accumulators=module_accumulators,
        kernel_accumulator=_build_accumulator(count=1),
        module_filters={_MODULE_KEY: frozenset({10}), (3, 4): frozenset({20})},
    )

    assert isinstance(result, ExtractedControllerData)
    assert len(result.modules) == 1
    assert result.modules[0].module_type == _MODULE_TYPE
    assert result.modules[0].module_id == _MODULE_ID
    assert result.modules[0].messages.count == 2
    assert result.kernel.count == 1


def test_finalize_batch_without_module_filters() -> None:
    """Verifies that _finalize_batch reports no modules when module extraction was not requested."""
    result = _finalize_batch(module_accumulators={}, kernel_accumulator=_create_accumulator(), module_filters=None)

    assert result.modules == ()
    assert result.kernel.count == 0


def test_process_message_batch_unknown_prototype_codes(tmp_path: Path) -> None:
    """Verifies that _process_message_batch stores no dtype or payload for an unrecognized prototype code."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_data(elapsed_us=1000, command=1, event=10, prototype_code=0, data_bytes=[1, 2]),
            (2000, create_kernel_data_payload(command=3, event=5, prototype_code=0, data_bytes=[7])),
        ],
    )
    reader = LogArchiveReader(archive_path=archive_path)

    module_accumulators, kernel_accumulator = _process_message_batch(
        log_path=archive_path,
        file_names=reader.get_batches(workers=1, batch_multiplier=1)[0],
        onset_us=reader.onset_timestamp_us,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=frozenset({5}),
    )

    assert module_accumulators[_MODULE_KEY].dtypes == [None]
    assert module_accumulators[_MODULE_KEY].data_payloads == [None]
    assert kernel_accumulator.dtypes == [None]
    assert kernel_accumulator.data_payloads == [None]


def test_extract_logged_microcontroller_data_invalid_path(tmp_path: Path) -> None:
    """Verifies that extraction rejects paths that do not point to an existing .npz file."""
    missing_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    message = (
        f"Unable to extract microcontroller message data from the log file {missing_path}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_microcontroller_data(
            log_path=missing_path, module_filters=None, kernel_event_codes=frozenset({5})
        )

    text_path = tmp_path / "controller_log.txt"
    text_path.touch()
    message = (
        f"Unable to extract microcontroller message data from the log file {text_path}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_microcontroller_data(log_path=text_path, module_filters=None, kernel_event_codes=frozenset({5}))

    directory_path = tmp_path / f"directory{LOG_ARCHIVE_SUFFIX}"
    directory_path.mkdir()
    message = (
        f"Unable to extract microcontroller message data from the log file {directory_path}, as it does not exist or "
        f"does not point to a valid .npz archive."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_microcontroller_data(
            log_path=directory_path, module_filters=None, kernel_event_codes=frozenset({5})
        )


def test_extract_logged_microcontroller_data_onset_only_archive(tmp_path: Path) -> None:
    """Verifies that an archive holding only the onset entry yields empty columns for every requested target."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(archive_path=archive_path, messages=[])

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=frozenset({5}),
        workers=1,
    )

    assert extracted.modules == ()
    assert extracted.kernel.count == 0
    assert extracted.kernel.timestamps.dtype == np.uint64


def test_extract_logged_microcontroller_data_module_only(tmp_path: Path) -> None:
    """Verifies that module-only extraction returns every matching module message and ignores all other traffic."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_state(elapsed_us=1000, command=1, event=10),
            _build_module_state(elapsed_us=2000, command=1, event=10),
            _build_module_data(
                elapsed_us=3000,
                command=2,
                event=20,
                prototype_code=int(SerialPrototypes.ONE_UINT8),
                data_bytes=[42],
            ),
            # An event code outside the module's filter, a message from an unrequested module, and a kernel message
            # that no filter admits. None of them are extracted.
            _build_module_state(elapsed_us=4000, command=1, event=99),
            (5000, create_module_state_payload(module_type=9, module_id=9, command=1, event=10)),
            (6000, create_kernel_state_payload(command=1, event=5)),
        ],
    )

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10, 20})},
        kernel_event_codes=None,
        workers=1,
    )

    assert len(extracted.modules) == 1
    module_data = extracted.modules[0]
    assert module_data.module_type == _MODULE_TYPE
    assert module_data.module_id == _MODULE_ID
    assert module_data.messages.count == 3
    np.testing.assert_array_equal(
        module_data.messages.timestamps,
        np.array([DEFAULT_ONSET_US + 1000, DEFAULT_ONSET_US + 2000, DEFAULT_ONSET_US + 3000], dtype=np.uint64),
    )
    np.testing.assert_array_equal(module_data.messages.commands, [1, 1, 2])
    np.testing.assert_array_equal(module_data.messages.events, [10, 10, 20])
    assert module_data.messages.dtypes == (None, None, "uint8")
    assert module_data.messages.data_payloads == (None, None, b"\x2a")
    assert extracted.kernel.count == 0


def test_extract_logged_microcontroller_data_kernel_only(tmp_path: Path) -> None:
    """Verifies that kernel-only extraction returns every matching kernel message and ignores all module traffic."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            (1000, create_kernel_state_payload(command=1, event=5)),
            (2000, create_kernel_state_payload(command=2, event=6)),
            (
                3000,
                create_kernel_data_payload(
                    command=3, event=5, prototype_code=_UINT16_PROTOTYPE_CODE, data_bytes=[172, 5]
                ),
            ),
            # An unmatched kernel event and a module message, neither of which is extracted.
            (4000, create_kernel_state_payload(command=1, event=99)),
            _build_module_state(elapsed_us=5000, command=1, event=10),
        ],
    )

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters=None,
        kernel_event_codes=frozenset({5, 6}),
        workers=1,
    )

    assert extracted.modules == ()
    assert extracted.kernel.count == 3
    np.testing.assert_array_equal(
        extracted.kernel.timestamps,
        np.array([DEFAULT_ONSET_US + 1000, DEFAULT_ONSET_US + 2000, DEFAULT_ONSET_US + 3000], dtype=np.uint64),
    )
    np.testing.assert_array_equal(extracted.kernel.commands, [1, 2, 3])
    np.testing.assert_array_equal(extracted.kernel.events, [5, 6, 5])
    assert extracted.kernel.dtypes == (None, None, "uint16")
    assert extracted.kernel.data_payloads == (None, None, b"\xac\x05")


def test_extract_logged_microcontroller_data_module_and_kernel(tmp_path: Path) -> None:
    """Verifies that a combined filter extracts the module and the kernel messages in a single pass."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_state(elapsed_us=1000, command=1, event=10),
            (2000, create_kernel_state_payload(command=1, event=5)),
            _build_module_state(elapsed_us=3000, command=2, event=10),
            (4000, create_kernel_state_payload(command=2, event=5)),
        ],
    )

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=frozenset({5}),
        workers=1,
    )

    assert len(extracted.modules) == 1
    assert extracted.modules[0].messages.count == 2
    assert extracted.kernel.count == 2
    np.testing.assert_array_equal(
        extracted.modules[0].messages.timestamps,
        np.array([DEFAULT_ONSET_US + 1000, DEFAULT_ONSET_US + 3000], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        extracted.kernel.timestamps,
        np.array([DEFAULT_ONSET_US + 2000, DEFAULT_ONSET_US + 4000], dtype=np.uint64),
    )


def test_extract_logged_microcontroller_data_isolates_the_per_module_event_codes(tmp_path: Path) -> None:
    """Verifies that each module is filtered against its own event codes rather than against the union of all sets."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    # Both modules emit both event codes, so a filter applied as a union would extract every one of these messages.
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_state(elapsed_us=1000, command=1, event=10),
            _build_module_state(elapsed_us=2000, command=1, event=20),
            (3000, create_module_state_payload(module_type=3, module_id=4, command=1, event=10)),
            (4000, create_module_state_payload(module_type=3, module_id=4, command=1, event=20)),
        ],
    )

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10}), (3, 4): frozenset({20})},
        kernel_event_codes=None,
        workers=1,
    )

    messages = {(module.module_type, module.module_id): module.messages for module in extracted.modules}

    assert set(messages.keys()) == {_MODULE_KEY, (3, 4)}
    np.testing.assert_array_equal(messages[_MODULE_KEY].events, [10])
    np.testing.assert_array_equal(messages[(3, 4)].events, [20])


def test_extract_logged_microcontroller_data_no_matching_messages(tmp_path: Path) -> None:
    """Verifies that a filter no message matches yields no module entries and an empty kernel block."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_state(elapsed_us=1000, command=1, event=99),
            (2000, create_kernel_state_payload(command=1, event=99)),
        ],
    )

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=frozenset({5}),
        workers=1,
    )

    assert extracted.modules == ()
    assert extracted.kernel.count == 0


def test_extract_logged_microcontroller_data_mismatched_data_payload(tmp_path: Path) -> None:
    """Verifies that extraction rejects a data payload of a different width than its prototype code declares."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    # The prototype code 7 declares a two-byte data object, so a four-byte payload cannot be decoded through it.
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_data(
                elapsed_us=1000,
                command=1,
                event=10,
                prototype_code=_UINT16_PROTOTYPE_CODE,
                data_bytes=[0, 0, 0, 0],
            )
        ],
    )

    message = (
        f"Unable to extract the data message logged by the module {_MODULE_TYPE} {_MODULE_ID} to '{archive_path}'. "
        f"The message declares the prototype code {_UINT16_PROTOTYPE_CODE}, whose data object occupies 2 bytes, but "
        f"it carries a 4-byte data payload."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_microcontroller_data(
            log_path=archive_path,
            module_filters={_MODULE_KEY: frozenset({10})},
            kernel_event_codes=None,
            workers=1,
        )


def test_extract_logged_microcontroller_data_mismatched_kernel_payload(tmp_path: Path) -> None:
    """Verifies that extraction rejects a kernel data payload of a different width than its prototype code declares."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            (
                1000,
                create_kernel_data_payload(
                    command=1, event=5, prototype_code=_UINT16_PROTOTYPE_CODE, data_bytes=[0, 0, 0, 0]
                ),
            )
        ],
    )

    message = (
        f"Unable to extract the data message logged by the kernel to '{archive_path}'. The message declares the "
        f"prototype code {_UINT16_PROTOTYPE_CODE}, whose data object occupies 2 bytes, but it carries a 4-byte data "
        f"payload."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        extract_logged_microcontroller_data(
            log_path=archive_path,
            module_filters=None,
            kernel_event_codes=frozenset({5}),
            workers=1,
        )


def test_extract_logged_microcontroller_data_matching_data_payload(tmp_path: Path) -> None:
    """Verifies that extraction accepts a data payload of the width its prototype code declares."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_data(
                elapsed_us=1000,
                command=1,
                event=10,
                prototype_code=_UINT16_PROTOTYPE_CODE,
                data_bytes=[172, 5],
            )
        ],
    )

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=None,
        workers=1,
    )

    assert len(extracted.modules) == 1
    messages = extracted.modules[0].messages
    assert messages.count == 1
    assert messages.dtypes == ("uint16",)
    assert messages.data_payloads == (b"\xac\x05",)
    assert np.frombuffer(messages.data_payloads[0], dtype=messages.dtypes[0])[0] == 1452


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_microcontroller_data_parallel_matches_sequential(tmp_path: Path) -> None:
    """Verifies that parallel extraction of an above-threshold archive reproduces the sequential result exactly."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    messages: list[tuple[int, NDArray[np.uint8]]] = [
        _build_module_state(elapsed_us=index * 10, command=1, event=10)
        for index in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
    ]
    messages.extend(
        (PARALLEL_PROCESSING_THRESHOLD * 10 + index, create_kernel_state_payload(command=2, event=5))
        for index in range(1, 5)
    )
    _build_archive(archive_path=archive_path, messages=messages)

    sequential = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=frozenset({5}),
        workers=1,
    )
    parallel = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=frozenset({5}),
        workers=2,
        display_progress=False,
    )

    assert sequential.modules[0].messages.count == PARALLEL_PROCESSING_THRESHOLD
    assert sequential.kernel.count == 4
    assert len(parallel.modules) == len(sequential.modules)
    np.testing.assert_array_equal(parallel.modules[0].messages.timestamps, sequential.modules[0].messages.timestamps)
    np.testing.assert_array_equal(parallel.modules[0].messages.commands, sequential.modules[0].messages.commands)
    np.testing.assert_array_equal(parallel.modules[0].messages.events, sequential.modules[0].messages.events)
    np.testing.assert_array_equal(parallel.kernel.timestamps, sequential.kernel.timestamps)


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_microcontroller_data_parallel_with_progress(tmp_path: Path) -> None:
    """Verifies that parallel extraction reports the same data when the progress bar is enabled."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_state(elapsed_us=index * 10, command=1, event=10)
            for index in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
        ],
    )

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=None,
        workers=2,
        display_progress=True,
    )

    assert extracted.modules[0].messages.count == PARALLEL_PROCESSING_THRESHOLD


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_microcontroller_data_resolves_the_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a worker count below one is auto-resolved before the message batches are generated."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_state(elapsed_us=index * 10, command=1, event=10)
            for index in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
        ],
    )

    resolution_requests: list[int] = []

    def _resolve_worker_count(requested_workers: int) -> int:
        """Records the requested worker count and resolves it to a pool size that keeps the test cheap."""
        resolution_requests.append(requested_workers)
        return 2

    monkeypatch.setattr(log_processing_module, "resolve_worker_count", _resolve_worker_count)

    extracted = extract_logged_microcontroller_data(
        log_path=archive_path,
        module_filters={_MODULE_KEY: frozenset({10})},
        kernel_event_codes=None,
        workers=-1,
        display_progress=False,
    )

    assert resolution_requests == [-1]
    assert extracted.modules[0].messages.count == PARALLEL_PROCESSING_THRESHOLD


@pytest.mark.xdist_group(name="orchestration")
def test_extract_logged_microcontroller_data_external_executor(tmp_path: Path) -> None:
    """Verifies that extraction submits batch work to a caller-owned executor and leaves that executor usable."""
    archive_path = tmp_path / f"{_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"
    _build_archive(
        archive_path=archive_path,
        messages=[
            _build_module_state(elapsed_us=index * 10, command=1, event=10)
            for index in range(1, PARALLEL_PROCESSING_THRESHOLD + 1)
        ],
    )

    executor = ProcessPoolExecutor(max_workers=2)
    try:
        extracted = extract_logged_microcontroller_data(
            log_path=archive_path,
            module_filters={_MODULE_KEY: frozenset({10})},
            kernel_event_codes=None,
            workers=2,
            display_progress=False,
            executor=executor,
        )

        assert extracted.modules[0].messages.count == PARALLEL_PROCESSING_THRESHOLD

        # The caller owns the pool, so extraction must not shut it down. A pool closed by extraction would instead
        # raise a RuntimeError when asked to accept more work.
        assert executor.submit(abs, -5).result() == 5
    finally:
        executor.shutdown(wait=True)


def _build_archive(archive_path: Path, messages: list[tuple[int, NDArray[np.uint8]]]) -> None:
    """Builds a synthetic log archive holding the requested messages."""
    create_test_archive(archive_path=archive_path, source_id=_SOURCE_ID, messages=messages)


def _build_module_state(elapsed_us: int, command: int, event: int) -> tuple[int, NDArray[np.uint8]]:
    """Builds a timestamped MODULE_STATE message for the shared test hardware module."""
    return (
        elapsed_us,
        create_module_state_payload(module_type=_MODULE_TYPE, module_id=_MODULE_ID, command=command, event=event),
    )


def _build_module_data(
    elapsed_us: int, command: int, event: int, prototype_code: int, data_bytes: list[int]
) -> tuple[int, NDArray[np.uint8]]:
    """Builds a timestamped MODULE_DATA message for the shared test hardware module."""
    return (
        elapsed_us,
        create_module_data_payload(
            module_type=_MODULE_TYPE,
            module_id=_MODULE_ID,
            command=command,
            event=event,
            prototype_code=prototype_code,
            data_bytes=data_bytes,
        ),
    )


def _build_accumulator(count: int) -> _ColumnAccumulator:
    """Builds a column accumulator holding the requested number of synthetic state-only messages."""
    return _ColumnAccumulator(
        timestamps=[100 * (index + 1) for index in range(count)],
        commands=[1] * count,
        events=[10] * count,
        dtypes=[None] * count,
        data_payloads=[None] * count,
    )
