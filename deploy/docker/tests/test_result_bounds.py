"""Byte budgets at the real result serialization and decompression boundary."""

import base64
import json
import random
import zlib

import pytest

from crawl_job_queue import CrawlJobQueueSettings
from utils import (
    MAX_SERIALIZED_RESULT_BYTES,
    MAX_STORED_RESULT_BYTES,
    ResultEncoding,
    ResultTooLarge,
    decode_task_result,
    encode_task_result,
)


@pytest.mark.parametrize("name", ["max_serialized_result_bytes", "max_stored_result_bytes"])
@pytest.mark.parametrize("value", [True, "1024", 1.5, None, 0, -1])
def test_result_settings_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=rf"crawl_jobs\.{name} must be a positive integer"):
        CrawlJobQueueSettings.from_config({"crawl_jobs": {name: value}})


@pytest.mark.parametrize("name,ceiling", [
    ("max_serialized_result_bytes", MAX_SERIALIZED_RESULT_BYTES),
    ("max_stored_result_bytes", MAX_STORED_RESULT_BYTES),
])
def test_result_settings_enforce_inclusive_reader_ceiling(name, ceiling):
    for value in (1, ceiling):
        settings = CrawlJobQueueSettings.from_config({"crawl_jobs": {name: value}})
        assert getattr(settings, name) == value
    with pytest.raises(ValueError, match=rf"crawl_jobs\.{name} must not exceed {ceiling}"):
        CrawlJobQueueSettings.from_config({"crawl_jobs": {name: ceiling + 1}})


def assert_limit(error, dimension, actual, limit):
    assert error.value.dimension == dimension
    assert error.value.actual == actual
    assert error.value.limit == limit


@pytest.mark.parametrize("encoding", list(ResultEncoding))
def test_custom_limits_include_exact_serialized_and_stored_sizes(encoding):
    result = {"html": "café 🌍"}
    serialized = json.dumps(result).encode("utf-8")
    stored = serialized if encoding == ResultEncoding.IDENTITY else zlib.compress(serialized, 6)
    encoded = encode_task_result(
        result, encoding, max_serialized_bytes=len(serialized), max_stored_bytes=len(stored)
    )
    assert (encoded.encode("utf-8") if isinstance(encoded, str) else encoded) == stored
    with pytest.raises(ResultTooLarge) as error:
        encode_task_result(
            result, encoding, max_serialized_bytes=len(serialized) - 1,
            max_stored_bytes=len(stored),
        )
    assert_limit(error, "serialized_bytes", len(serialized), len(serialized) - 1)
    with pytest.raises(ResultTooLarge) as error:
        encode_task_result(
            result, encoding, max_serialized_bytes=len(serialized),
            max_stored_bytes=len(stored) - 1,
        )
    assert_limit(error, "stored_bytes", len(stored), len(stored) - 1)


def test_default_identity_stored_limit_is_inclusive():
    result = "a" * (MAX_STORED_RESULT_BYTES - 2)  # JSON string quotes count too.
    encoded = encode_task_result(result, ResultEncoding.IDENTITY)
    assert len(encoded.encode("utf-8")) == MAX_STORED_RESULT_BYTES
    with pytest.raises(ResultTooLarge) as error:
        encode_task_result(result + "a", ResultEncoding.IDENTITY)
    assert_limit(error, "stored_bytes", MAX_STORED_RESULT_BYTES + 1, MAX_STORED_RESULT_BYTES)


def test_compressible_result_still_obeys_default_serialized_limit():
    result = "a" * (MAX_SERIALIZED_RESULT_BYTES - 2)
    encoded = encode_task_result(result, ResultEncoding.ZLIB)
    assert len(encoded) < MAX_STORED_RESULT_BYTES
    assert decode_task_result({"result": encoded, "result_encoding": "zlib"}, "task") == result
    with pytest.raises(ResultTooLarge) as error:
        encode_task_result(result + "a", ResultEncoding.ZLIB)
    assert_limit(
        error, "serialized_bytes", MAX_SERIALIZED_RESULT_BYTES + 1, MAX_SERIALIZED_RESULT_BYTES
    )


def test_incompressible_result_obeys_default_stored_limit():
    # Seeded random bytes keep this case reproducible while defeating zlib compression.
    result = base64.b64encode(random.Random(42).randbytes(MAX_STORED_RESULT_BYTES)).decode("ascii")
    serialized = json.dumps(result).encode("utf-8")
    stored_size = len(zlib.compress(serialized, 6))
    assert len(serialized) < MAX_SERIALIZED_RESULT_BYTES
    assert stored_size > MAX_STORED_RESULT_BYTES
    with pytest.raises(ResultTooLarge) as error:
        encode_task_result(result, ResultEncoding.ZLIB)
    assert_limit(error, "stored_bytes", stored_size, MAX_STORED_RESULT_BYTES)


@pytest.mark.parametrize("marker", [None, "identity", b"identity", "zlib", b"zlib"])
@pytest.mark.parametrize("as_bytes", [False, True])
def test_legacy_and_marked_results_count_utf8_bytes(marker, as_bytes):
    result = {"html": "café 🌍"}
    # Readers also accept legacy JSON emitted with unescaped Unicode.
    serialized = json.dumps(result, ensure_ascii=False)
    raw = serialized.encode("utf-8")
    task = {"result": raw if as_bytes else serialized}
    if marker is not None:
        task["result_encoding"] = marker
    if marker in ("zlib", b"zlib"):
        task["result"] = zlib.compress(raw, 6)
    assert len(raw) > len(serialized)
    assert decode_task_result(task, "task", max_serialized_bytes=len(raw)) == result
    with pytest.raises(ResultTooLarge) as error:
        decode_task_result(task, "task", max_serialized_bytes=len(raw) - 1)
    assert_limit(error, "serialized_bytes", len(raw), len(raw) - 1)


def test_default_decompression_limit_bounds_a_large_expansion():
    compressed = zlib.compress(b'"' + b"a" * MAX_SERIALIZED_RESULT_BYTES + b'"', 6)
    assert len(compressed) < MAX_STORED_RESULT_BYTES
    with pytest.raises(ResultTooLarge) as error:
        decode_task_result({"result": compressed, "result_encoding": "zlib"}, "task")
    # The observed extra byte is sufficient to reject, without allocating the full expansion.
    assert_limit(
        error, "serialized_bytes", MAX_SERIALIZED_RESULT_BYTES + 1, MAX_SERIALIZED_RESULT_BYTES
    )


def test_legacy_identity_result_obeys_default_decode_limit():
    serialized = '"' + "a" * (MAX_SERIALIZED_RESULT_BYTES - 2) + '"'
    assert decode_task_result({"result": serialized}, "task") == "a" * (MAX_SERIALIZED_RESULT_BYTES - 2)
    with pytest.raises(ResultTooLarge) as error:
        decode_task_result({"result": serialized + " "}, "task")
    assert_limit(
        error, "serialized_bytes", MAX_SERIALIZED_RESULT_BYTES + 1, MAX_SERIALIZED_RESULT_BYTES
    )


def test_truncated_zlib_retains_decode_failure():
    compressed = zlib.compress(b'{"success":true}', 6)
    with pytest.raises(zlib.error, match="incomplete or truncated"):
        decode_task_result({"result": compressed[:-1], "result_encoding": "zlib"}, "task")
