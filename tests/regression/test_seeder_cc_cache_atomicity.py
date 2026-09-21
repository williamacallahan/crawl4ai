"""Regression tests for the atomicity of the Common Crawl cache write.

``AsyncUrlSeeder._from_cc`` streams JSON-line URL records from
``index.commoncrawl.org`` and persists them to a per-domain cache file
``{index_id}_{safe}_{digest}.jsonl`` under ``seeder_cache/``. A subsequent
call with the default ``force=False`` trusts any existing cache file as
the *complete* URL set for the domain (no checksum, no completeness
marker, no per-file TTL).

Before the fix the cache file was opened in ``"w"`` mode *before* the
streaming download began, so a mid-stream failure (``httpx.ReadError``, or
caller cancellation via a ``max_urls``-triggered ``GeneratorExit``)
flushed and closed a partially-written file that the cache-read path then
served forever.

The fix streams into a sibling ``.tmp`` file and only ``replace``s the
live cache after the full response lands; on *any* exit (including
``GeneratorExit`` and ``asyncio.CancelledError``) the temp is unlinked,
so a previously-completed good cache (if any) is preserved and the next
call refetches instead of serving a truncated set.

These tests pin that contract by exercising ``_from_cc`` directly and
through the public ``urls(...)`` API with a fake ``client.stream`` whose
``aiter_lines`` deterministically raises transport errors / blocks /
returns canned JSON lines. No network is used and ``aiofiles`` is left
real so atomicity is proven on actual disk under ``tmp_path``.
"""

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from crawl4ai.async_url_seeder import AsyncUrlSeeder
from crawl4ai.async_configs import SeedingConfig


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

DOMAIN = "acme.io"
INDEX_ID = "CC-MAIN-test"
PATTERN = "*"
CC_URLS = [f"https://acme.io/blog/post-{i}" for i in range(5)]


def _cc_records(urls):
    """The JSON-lines payload the CC index API returns (one record per URL)."""
    return [
        json.dumps({"url": u, "timestamp": "20240101000000", "status": "200"})
        for u in urls
    ]


def _cc_cache_path(seeder, domain=DOMAIN, pattern=PATTERN):
    """Mirror of ``_from_cc``'s cache-path computation for assertions."""
    digest = hashlib.md5(pattern.encode()).hexdigest()[:8]
    raw = re.sub(r"^https?://", "", domain).split("#", 1)[0].split("?", 1)[0].lstrip(".")
    safe = re.sub("[/?#]+", "_", raw)
    return seeder.cache_dir / f"{seeder.index_id}_{safe}_{digest}.jsonl"


def _read_lines(p: Path):
    if not p.exists():
        return []
    return [l for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Fake httpx streaming client
# ---------------------------------------------------------------------------

class _FakeStreamResponse:
    """Stand-in for the httpx streaming response returned by ``client.stream``.

    Supports deterministic mid-stream raises and a pluggable async-line
    generator (for the cancellation test that needs to block forever).
    """

    def __init__(self, records=None, status_code=200, raise_after=None,
                 exc=None, lines_agen=None,
                 url="https://index.commoncrawl.org/CC-MAIN-test-index"):
        self.status_code = status_code
        self.url = url
        self._records = list(records or [])
        self.raise_after = raise_after
        self.exc = exc
        self._lines_agen = lines_agen

    def raise_for_status(self):
        if not 200 <= self.status_code < 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self,
            )

    async def aiter_lines(self):
        if self._lines_agen is not None:
            async for line in self._lines_agen():
                yield line
            return
        exc = self.exc if self.exc is not None else httpx.ReadError(
            "simulated mid-stream transport reset")
        for i, line in enumerate(self._records):
            if self.raise_after is not None and i >= self.raise_after:
                raise exc
            yield line


class _FakeStreamCM:
    """Async context manager matching ``async with client.stream(...) as r:``."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False  # never suppress the exception


class _FakeClient:
    """httpx.AsyncClient stand-in serving a queue of canned stream responses.

    ``stream(method, url)`` pops the next response from the queue. If the
    queue is empty it raises ``AssertionError`` — so a cache-hit test that
    must not touch the network fails loudly if it does.
    """

    def __init__(self, responses=None):
        self.stream_responses = list(responses or [])
        self.stream_calls = 0
        self.aclose = AsyncMock()

    def stream(self, method, url, **kw):
        self.stream_calls += 1
        if not self.stream_responses:
            raise AssertionError(
                f"client.stream must not be called (call #{self.stream_calls})")
        return _FakeStreamCM(self.stream_responses.pop(0))


def _make_seeder(tmp_path, responses=None):
    seeder = AsyncUrlSeeder(
        client=_FakeClient(responses),
        base_directory=tmp_path,
        cache_root=tmp_path / "cache",
    )
    # Bypass the network call to collinfo.json in ``urls()``.
    seeder.index_id = INDEX_ID
    return seeder


def _ok_response(records=None, **kw):
    return _FakeStreamResponse(records=records or _cc_records(CC_URLS), **kw)


def _status_response(status_code):
    return _FakeStreamResponse(status_code=status_code)


# ---------------------------------------------------------------------------
# Unit tests directly on _from_cc
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_from_cc_full_stream_writes_complete_cache_atomically(tmp_path):
    seeder = _make_seeder(tmp_path, [_ok_response()])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    yielded = []
    async for u in seeder._from_cc(DOMAIN, PATTERN, force=False):
        yielded.append(u)

    assert sorted(yielded) == sorted(CC_URLS)
    assert path.read_text().splitlines() == CC_URLS, "cache must preserve all URLs in order"
    assert not tmp.exists(), "temp file leaked after a successful fetch"
    assert seeder.client.stream_calls == 1


@pytest.mark.asyncio
async def test_from_cc_midstream_read_error_leaves_no_cache(tmp_path):
    seeder = _make_seeder(
        tmp_path, [_FakeStreamResponse(_cc_records(CC_URLS), raise_after=2)])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    yielded = []
    with pytest.raises(httpx.ReadError):
        async for u in seeder._from_cc(DOMAIN, PATTERN, force=False):
            yielded.append(u)

    assert sorted(yielded) == sorted(CC_URLS[:2])
    assert not path.exists(), "mid-stream error left a truncated cache file"
    assert not tmp.exists(), "mid-stream error left a temp file on disk"


@pytest.mark.asyncio
async def test_from_cc_midstream_error_preserves_existing_good_cache(tmp_path):
    seeder = _make_seeder(
        tmp_path, [_FakeStreamResponse(_cc_records(CC_URLS), raise_after=2)])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Pre-existing GOOD cache with the full URL set.
    path.write_text("".join(u + "\n" for u in CC_URLS))
    original = path.read_bytes()

    with pytest.raises(httpx.ReadError):
        async for _ in seeder._from_cc(DOMAIN, PATTERN, force=True):
            pass

    assert path.read_bytes() == original, "pre-existing good cache was truncated"
    assert _read_lines(path) == CC_URLS
    assert not tmp.exists(), "temp file leaked after mid-stream error"


@pytest.mark.asyncio
async def test_from_cc_early_aclose_leaves_no_cache(tmp_path):
    """``GeneratorExit`` via early ``aclose()`` (the ``max_urls`` path)."""
    seeder = _make_seeder(tmp_path, [_ok_response()])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    gen = seeder._from_cc(DOMAIN, PATTERN, force=False)
    yielded = []
    try:
        async for u in gen:
            yielded.append(u)
            if len(yielded) == 2:
                break
        await gen.aclose()
    finally:
        try:
            await gen.aclose()
        except Exception:
            pass

    assert sorted(yielded) == sorted(CC_URLS[:2])
    assert not path.exists(), "GeneratorExit left a partial cache file"
    assert not tmp.exists(), "GeneratorExit left a temp file on disk"


@pytest.mark.asyncio
async def test_from_cc_cancellation_leaves_no_cache(tmp_path):
    """``asyncio.CancelledError`` during the stream rolls back the temp file."""
    never = asyncio.Event()

    async def blocking_lines():
        for line in _cc_records(CC_URLS[:2]):
            yield line
        # Block forever on the 3rd line — simulates a slow upstream response.
        await never.wait()
        yield _cc_records(CC_URLS)[2]

    seeder = _make_seeder(
        tmp_path, [_FakeStreamResponse(lines_agen=blocking_lines)])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    collected = []

    async def driver():
        async for u in seeder._from_cc(DOMAIN, PATTERN, force=False):
            collected.append(u)

    task = asyncio.create_task(driver())
    # Let the driver consume the 2 available URLs and block on the 3rd.
    await asyncio.sleep(0.1)
    assert task.cancel(), "task was already finished before cancel()"
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sorted(collected) == sorted(CC_URLS[:2])
    assert not path.exists(), "CancelledError left a partial cache file"
    assert not tmp.exists(), "CancelledError left a temp file on disk"


@pytest.mark.asyncio
async def test_from_cc_cache_hit_serves_complete_set_no_network(tmp_path):
    seeder = _make_seeder(tmp_path)  # empty queue → stream must not be called
    path = _cc_cache_path(seeder)
    path.write_text("".join(u + "\n" for u in CC_URLS))

    yielded = []
    async for u in seeder._from_cc(DOMAIN, PATTERN, force=False):
        yielded.append(u)

    assert sorted(yielded) == sorted(CC_URLS)
    assert seeder.client.stream_calls == 0, "cache hit still hit the network"


@pytest.mark.asyncio
async def test_from_cc_force_refetches_over_complete_cache(tmp_path):
    seeder = _make_seeder(tmp_path, [_ok_response()])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    stale = [f"https://acme.io/old/{i}" for i in range(3)]
    path.write_text("".join(u + "\n" for u in stale))

    yielded = []
    async for u in seeder._from_cc(DOMAIN, PATTERN, force=True):
        yielded.append(u)

    assert sorted(yielded) == sorted(CC_URLS)
    assert path.read_text().splitlines() == CC_URLS, "force did not overwrite cache"
    assert not tmp.exists()
    assert seeder.client.stream_calls == 1


@pytest.mark.asyncio
async def test_from_cc_refetch_after_failure_with_default_force(tmp_path):
    failing = _FakeStreamResponse(_cc_records(CC_URLS), raise_after=2)
    seeder = _make_seeder(tmp_path, [failing, _ok_response()])
    path = _cc_cache_path(seeder)

    # First call: stream raises after 2 lines, no cache left.
    first = []
    with pytest.raises(httpx.ReadError):
        async for u in seeder._from_cc(DOMAIN, PATTERN, force=False):
            first.append(u)
    assert sorted(first) == sorted(CC_URLS[:2])
    assert not path.exists(), "failed first call left a truncated cache"

    # Second call with default force=False MUST refetch (path absent).
    second = []
    async for u in seeder._from_cc(DOMAIN, PATTERN, force=False):
        second.append(u)
    assert sorted(second) == sorted(CC_URLS)
    assert path.read_text().splitlines() == CC_URLS


@pytest.mark.asyncio
async def test_from_cc_503_then_success_writes_complete_cache(tmp_path, monkeypatch):
    sleeps = []
    orig_sleep = asyncio.sleep

    async def fake_sleep(d):
        sleeps.append(d)
        # yield control once so the retry loop can proceed deterministically
        await orig_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    seeder = _make_seeder(
        tmp_path,
        [_status_response(503), _ok_response()],
    )
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    yielded = []
    async for u in seeder._from_cc(DOMAIN, PATTERN, force=False):
        yielded.append(u)

    assert sorted(yielded) == sorted(CC_URLS)
    assert path.read_text().splitlines() == CC_URLS
    assert not tmp.exists()
    assert seeder.client.stream_calls == 2
    assert sleeps == [1], "first 503 must trigger exactly one back-off sleep"


@pytest.mark.asyncio
async def test_from_cc_persistent_503_raises_no_cache(tmp_path, monkeypatch):
    orig_sleep = asyncio.sleep

    async def fake_sleep(d):
        await orig_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # 4 attempts: three retries + the sentinel non-retry iteration.
    seeder = _make_seeder(tmp_path, [_status_response(503)] * 4)
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in seeder._from_cc(DOMAIN, PATTERN, force=False):
            pass

    assert seeder.client.stream_calls == 4
    assert not path.exists(), "persistent 503 left a cache file"
    assert not tmp.exists(), "persistent 503 left a temp file"


@pytest.mark.asyncio
async def test_from_cc_non_503_status_raises_no_cache(tmp_path):
    seeder = _make_seeder(tmp_path, [_status_response(404)])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in seeder._from_cc(DOMAIN, PATTERN, force=False):
            pass

    assert seeder.client.stream_calls == 1
    assert not path.exists()
    assert not tmp.exists()


# ---------------------------------------------------------------------------
# End-to-end through the public urls() API
# ---------------------------------------------------------------------------

def _cc_cfg(**overrides):
    base = dict(
        source="cc",
        pattern=PATTERN,
        live_check=False,
        extract_head=False,
        concurrency=1,
        hits_per_sec=None,
        force=False,
        filter_nonsense_urls=False,
    )
    base.update(overrides)
    return SeedingConfig(**base)


@pytest.mark.asyncio
async def test_urls_midstream_error_returns_partial_and_leaves_no_cache(tmp_path):
    failing = _FakeStreamResponse(_cc_records(CC_URLS), raise_after=2)
    seeder = _make_seeder(tmp_path, [failing, _ok_response()])
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    cfg = _cc_cfg()
    results = await seeder.urls(DOMAIN, cfg)
    returned = sorted(r["url"] for r in results)
    assert returned == sorted(CC_URLS[:2]), "user silently got a truncated set"
    assert len(returned) == 2 < len(CC_URLS)
    assert not path.exists(), "urls() left a truncated cache after mid-stream error"
    assert not tmp.exists()

    # Second call with the default force=False refetches (no cache present).
    cfg2 = _cc_cfg()
    results2 = await seeder.urls(DOMAIN, cfg2)
    assert sorted(r["url"] for r in results2) == sorted(CC_URLS), "second call did not refetch the full set"
    assert path.read_text().splitlines() == CC_URLS


@pytest.mark.asyncio
async def test_urls_max_urls_never_truncates_cache(tmp_path):
    # Slow per-line stream so the worker hits max_urls before the producer
    # finishes streaming — the exact race that triggered the truncated cache.
    async def slow_lines():
        for line in _cc_records(CC_URLS):
            await asyncio.sleep(0.02)
            yield line

    seeder = _make_seeder(
        tmp_path,
        [_FakeStreamResponse(lines_agen=slow_lines), _ok_response()],
    )
    path = _cc_cache_path(seeder)
    tmp = path.with_suffix(path.suffix + ".tmp")

    cfg = _cc_cfg(max_urls=1)
    results = await seeder.urls(DOMAIN, cfg)
    assert len(results) <= 1
    # Invariant: the cache is either COMPLETE or ABSENT — never truncated.
    if path.exists():
        assert path.read_text().splitlines() == CC_URLS, "cache was truncated"
    assert not tmp.exists(), "temp file leaked through urls()"

    # Second call asking for "everything" must return the full CC set.
    seeder.client.stream_responses = [_ok_response()]
    cfg2 = _cc_cfg(max_urls=-1)
    results2 = await seeder.urls(DOMAIN, cfg2)
    assert sorted(r["url"] for r in results2) == sorted(CC_URLS)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"]
    ))
