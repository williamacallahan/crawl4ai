import asyncio
import json
import time

import httpx
import pytest

import api
import brightdata_unlocker
from brightdata_unlocker import BrightDataUnlocker, UnlockerError, UnlockerSettings
from crawl4ai import CrawlerRunConfig
from utils import load_config


class BodyStream(httpx.AsyncByteStream):
    def __init__(self, *chunks, wait=None):
        self.chunks = chunks
        self.wait = wait
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.wait is not None:
            await self.wait.wait()

    async def aclose(self):
        self.closed = True


def make_unlocker(handler, **settings):
    unlocker = BrightDataUnlocker("test-key", UnlockerSettings(**settings))
    unlocker._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return unlocker


def test_success_streams_body_and_posts_provider_contract():
    async def run():
        requests = []
        stream = BodyStream(b"<html>", b"caf\xc3", b"\xa9</html>")

        def handler(request):
            requests.append(request)
            return httpx.Response(200, stream=stream)

        unlocker = make_unlocker(handler, zone="test_zone", country="gb")
        try:
            assert await unlocker.fetch("https://example.com/article") == "<html>café</html>"
            assert len(requests) == 1
            request = requests[0]
            assert request.method == "POST"
            assert str(request.url) == "https://api.brightdata.com/request"
            assert request.headers["authorization"] == "Bearer test-key"
            assert json.loads(request.content) == {
                "zone": "test_zone",
                "url": "https://example.com/article",
                "format": "raw",
                "country": "gb",
            }
            assert stream.closed
        finally:
            await unlocker.close()

    asyncio.run(run())


def test_credential_refusal_opens_breaker_after_configured_failures(caplog):
    async def run():
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(401, stream=BodyStream(b"Unauthorized"))

        unlocker = make_unlocker(handler, breaker_failures=2)
        try:
            for _ in range(2):
                with pytest.raises(UnlockerError, match="provider refused.*401"):
                    await unlocker.fetch("https://example.com")
            with pytest.raises(UnlockerError, match="circuit is open"):
                await unlocker.fetch("https://example.com")
            assert len(calls) == 2
            assert [r.message for r in caplog.records if r.name == "brightdata_unlocker"] == [
                "Web Unlocker credential refused; circuit opened"
            ]
        finally:
            await unlocker.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "headers", "opens_breaker"),
    [
        (502, {}, True),
        (403, {"x-brd-error": "upstream failure"}, True),
        (200, {"x-brd-error": ""}, True),
        (200, {"x-brd-err-code": "provider_failure"}, True),
        (404, {}, False),
    ],
)
def test_provider_failures_open_breaker_but_target_errors_do_not(
    status, headers, opens_breaker
):
    async def run():
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(status, headers=headers, stream=BodyStream(b"failure"))
            return httpx.Response(200, stream=BodyStream(b"recovered"))

        unlocker = make_unlocker(handler, breaker_failures=1)
        try:
            with pytest.raises(UnlockerError, match="provider refused|target returned"):
                await unlocker.fetch("https://example.com")
            if opens_breaker:
                with pytest.raises(UnlockerError, match="circuit is open"):
                    await unlocker.fetch("https://example.com")
                assert len(calls) == 1
            else:
                assert await unlocker.fetch("https://example.com") == "recovered"
                assert len(calls) == 2
        finally:
            await unlocker.close()

    asyncio.run(run())


def test_response_limit_applies_to_streamed_bytes_and_closes_response():
    async def run():
        stream = BodyStream(b"123", b"456")
        unlocker = make_unlocker(
            lambda request: httpx.Response(200, stream=stream),
            max_response_bytes=5,
            breaker_failures=1,
        )
        try:
            with pytest.raises(UnlockerError, match="exceeds max_response_bytes"):
                await unlocker.fetch("https://example.com")
            assert stream.closed
            assert unlocker._open_until == 0
        finally:
            await unlocker.close()

    asyncio.run(run())


def test_transport_timeout_does_not_replay_request():
    async def run():
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout("provider stalled", request=request)
            return httpx.Response(200, stream=BodyStream(b"would succeed on replay"))

        unlocker = make_unlocker(handler)
        try:
            with pytest.raises(UnlockerError, match="request timed out"):
                await unlocker.fetch("https://example.com")
        finally:
            await unlocker.close()

    asyncio.run(run())


@pytest.mark.parametrize("waiting_for", ["semaphore", "body"])
def test_total_timeout_includes_semaphore_wait_and_streaming_body(waiting_for):
    async def run():
        calls = []
        stream = BodyStream(b"partial", wait=asyncio.Event())

        def handler(request):
            calls.append(request)
            return httpx.Response(200, stream=stream)

        unlocker = make_unlocker(handler, max_concurrency=1, total_timeout_s=0.02)
        if waiting_for == "semaphore":
            await unlocker._semaphore.acquire()
        try:
            with pytest.raises(UnlockerError, match="request timed out"):
                await asyncio.wait_for(unlocker.fetch("https://example.com"), timeout=1)
            assert len(calls) == (0 if waiting_for == "semaphore" else 1)
            if waiting_for == "body":
                assert stream.closed
        finally:
            if waiting_for == "semaphore":
                unlocker._semaphore.release()
            await unlocker.close()

    asyncio.run(run())


def test_elapsed_breaker_admits_one_half_open_probe_then_recovers():
    async def run():
        calls = []
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        async def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, stream=BodyStream(b"provider unavailable"))
            if len(calls) == 2:
                probe_started.set()
                await release_probe.wait()
            return httpx.Response(200, stream=BodyStream(b"recovered"))

        unlocker = make_unlocker(handler, breaker_failures=1, max_concurrency=2)
        probe = None
        try:
            with pytest.raises(UnlockerError, match="provider refused"):
                await unlocker.fetch("https://example.com")
            unlocker._open_until = time.monotonic() - 1
            probe = asyncio.create_task(unlocker.fetch("https://example.com"))
            await asyncio.wait_for(probe_started.wait(), timeout=1)
            with pytest.raises(UnlockerError, match="circuit is open"):
                await unlocker.fetch("https://example.com")
            assert len(calls) == 2
            release_probe.set()
            assert await probe == "recovered"
            assert await unlocker.fetch("https://example.com") == "recovered"
            assert len(calls) == 3
        finally:
            release_probe.set()
            if probe is not None:
                await asyncio.gather(probe, return_exceptions=True)
            await unlocker.close()

    asyncio.run(run())


@pytest.mark.parametrize("enabled", [False, True])
def test_server_configs_install_only_server_unlocker(monkeypatch, enabled, caplog):
    monkeypatch.setattr(brightdata_unlocker, "_initialized", False)
    monkeypatch.setattr(brightdata_unlocker, "_unlocker", None)
    if enabled:
        monkeypatch.setenv("BRIGHTDATA_API_KEY", "test-key")
    else:
        monkeypatch.delenv("BRIGHTDATA_API_KEY", raising=False)
    config = load_config()
    with caplog.at_level("INFO", logger="brightdata_unlocker"):
        built = api.server_crawler_config(config)
        loaded = api.apply_server_crawler_defaults(CrawlerRunConfig(), {}, config)
    if enabled:
        assert built.fallback_fetch_function.__self__ is brightdata_unlocker._unlocker
        assert loaded.fallback_fetch_function == built.fallback_fetch_function
    else:
        assert built.fallback_fetch_function is None
        assert loaded.fallback_fetch_function is None
        assert sum("fallback disabled" in r.message for r in caplog.records) == 1
