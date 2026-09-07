"""Regression tests for streaming-crawl per-URL `crawler_configs` plumbing.

Bug: in `deploy/docker/server.py`, both streaming crawl entry points
(`/crawl` with `crawler_config.stream=true` and `/crawl/stream`) route
through `stream_process` -> `handle_stream_crawl_request`. On the multi-URL,
non-deep-crawl streaming branch (the only streaming path that reaches
`arun_many`), the request's inherited `crawler_configs` field (per-URL
`CrawlerRunConfig` list) was silently dropped: `stream_process` never
forwarded it and `handle_stream_crawl_request` did not accept the parameter.

A caller sending a multi-URL streaming request with `crawler_configs`
(per-URL extraction strategies / `url_matcher` patterns) got every URL
crawled with the single shared `crawler_config` -- the per-URL
customization was silently ignored, with no HTTP-level rejection and no
log entry recording that the supplied field was discarded.
"""

import asyncio
import inspect
from typing import List, Optional

import api
import crawler_pool
import egress_broker
import pytest
from fastapi import HTTPException

from crawl4ai import CrawlerRunConfig, MemoryAdaptiveDispatcher


def _stream_config() -> dict:
    """Minimal config for handle_stream_crawl_request (no rate limiter)."""
    return {
        "crawler": {
            "base_config": {"delay_before_return_html": 1.0},
            "pool": {"max_pages": 2},
            "memory_threshold_percent": 95,
            "recovery_threshold_percent": 80,
            "rate_limiter": {"enabled": False, "base_delay": [0, 0]},
            "browser": {"kwargs": {}},
        },
        "limits": {"wall_clock_s": 0},
    }


class _RecordingCrawler:
    """Minimal AsyncWebCrawler stand-in.

    Records the kwargs of every `arun` / `arun_many` call and returns an
    empty async generator (the streaming path's return shape), so
    `handle_stream_crawl_request` can complete setup without a real
    browser.
    """

    def __init__(self):
        self.arun_many_calls: list = []
        self.arun_calls: list = []

    async def arun_many(self, *, urls, config, dispatcher=None, **_kwargs):
        self.arun_many_calls.append(
            {"urls": list(urls), "config": config, "dispatcher": dispatcher}
        )

        async def _empty():
            if False:  # pragma: no cover - never iterated
                yield None

        return _empty()

    async def arun(self, url, *, config, **_kwargs):
        self.arun_calls.append({"url": url, "config": config})

        async def _empty():
            if False:  # pragma: no cover - never iterated
                yield None

        return _empty()


def _install_recording_crawler(monkeypatch, recording_crawler):
    """Patch the surface `handle_stream_crawl_request` touches, mirroring the
    pattern in `test_server_crawler_defaults.py::test_streaming_uses_server_render_readiness_default`.
    """

    async def get_crawler(_browser_config):
        return recording_crawler

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)
    # _track_request_start reaches out to the monitor; the real one swallows
    # exceptions but imports `monitor` lazily. Patching keeps the test
    # hermetic.
    monkeypatch.setattr(api, "_track_request_start", _no_op_track)


async def _no_op_track(_endpoint, _urls, _browser_config):
    return "req-test"


# ─── signature / acceptance ──────────────────────────────────────────────


def test_handle_stream_crawl_request_accepts_crawler_configs_parameter():
    """The handler must expose `crawler_configs: Optional[List[dict]] = None`."""
    sig = inspect.signature(api.handle_stream_crawl_request)
    assert "crawler_configs" in sig.parameters
    param = sig.parameters["crawler_configs"]
    assert param.default is None
    # Optional[List[dict]] (either typing form). Resolve aliases and compare
    # structurally so a typing-module prefix in repr doesn't make this brittle.
    assert param.annotation == Optional[List[dict]]


def test_stream_process_in_server_forwards_crawler_configs(server_module, monkeypatch):
    """`stream_process` in `server.py` must forward
    `crawl_request.crawler_configs` to `handle_stream_crawl_request`.

    This is the routing change that fixes the silent drop: previously the
    field was accepted by the inherited `CrawlRequestWithHooks` schema but
    never forwarded by `stream_process`, so it could not reach the library.
    """
    captured: dict = {}

    async def fake_handle_stream_crawl_request(**kwargs):
        captured.update(kwargs)

        async def _empty_gen():
            if False:  # pragma: no cover - never iterated
                yield None

        return object(), _empty_gen(), None, "req-test"

    monkeypatch.setattr(
        server_module,
        "handle_stream_crawl_request",
        fake_handle_stream_crawl_request,
    )

    crawler_configs = [
        {"url_matcher": "*site-a*", "cache_mode": "bypass"},
        {"url_matcher": "*site-b*", "cache_mode": "bypass"},
    ]
    crawl_request = server_module.CrawlRequestWithHooks(
        urls=["https://site-a.example/x", "https://site-b.example/x"],
        crawler_config={"stream": True},
        crawler_configs=crawler_configs,
    )

    response = asyncio.run(server_module.stream_process(crawl_request=crawl_request))

    # stream_process returns a StreamingResponse; we don't iterate the body
    # (the patched handler returns an empty generator). The contract under
    # test is that the kwarg made it across the server/router boundary.
    assert "crawler_configs" in captured
    assert captured["crawler_configs"] == crawler_configs
    # The other forwarding must remain intact (no regression).
    assert captured["urls"] == [
        "https://site-a.example/x",
        "https://site-b.example/x",
    ]
    assert captured["crawler_config"] == {"stream": True}
    assert captured["hooks_config"] is None
    assert response is not None


# ─── multi-URL streaming: crawler_configs honored ───────────────────────


def test_stream_multi_url_forwards_crawler_configs_as_list(monkeypatch):
    """Multi-URL streaming with `crawler_configs` must pass a *list* (not a
    single CrawlerRunConfig) to `arun_many`, so per-URL matching can apply."""
    crawler = _RecordingCrawler()
    _install_recording_crawler(monkeypatch, crawler)

    asyncio.run(
        api.handle_stream_crawl_request(
            urls=["https://site-a.example/x", "https://site-b.example/x"],
            browser_config={},
            crawler_config={"stream": True},
            config=_stream_config(),
            crawler_configs=[
                {"url_matcher": "*site-a*"},
                {"url_matcher": "*site-b*"},
            ],
        )
    )

    assert len(crawler.arun_many_calls) == 1
    config_arg = crawler.arun_many_calls[0]["config"]
    assert isinstance(config_arg, list)
    assert len(config_arg) == 2
    assert all(isinstance(c, CrawlerRunConfig) for c in config_arg)
    assert isinstance(crawler.arun_many_calls[0]["dispatcher"], MemoryAdaptiveDispatcher)


def test_stream_multi_url_select_config_routes_per_url(monkeypatch):
    """End-to-end: the dispatcher's `select_config` must route each URL to the
    config the caller supplied, proving the list actually reaches the
    library's per-URL selection logic. Unmatched URLs yield None, which
    `crawl_url` turns into a failed `CrawlResult` (parity with the non-stream
    `handle_crawl_request` path -- no silent wrong-config success)."""
    crawler = _RecordingCrawler()
    _install_recording_crawler(monkeypatch, crawler)

    asyncio.run(
        api.handle_stream_crawl_request(
            urls=["https://site-a.example/x", "https://site-b.example/x"],
            browser_config={},
            crawler_config={"stream": True},
            config=_stream_config(),
            crawler_configs=[
                {"url_matcher": "*site-a*", "css_selector": "article"},
                {"url_matcher": "*site-b*", "css_selector": "main"},
            ],
        )
    )

    config_list = crawler.arun_many_calls[0]["config"]
    dispatcher = crawler.arun_many_calls[0]["dispatcher"]
    a, b = config_list[0], config_list[1]
    # Per-URL routing: each URL lands on its caller-supplied config.
    assert dispatcher.select_config("https://site-a.example/x", config_list) is a
    assert dispatcher.select_config("https://site-b.example/x", config_list) is b
    # No-match -> None -> crawl_url synthesizes a failed CrawlResult with
    # error_message="No matching configuration found for URL: ..." (see
    # crawl4ai/async_dispatcher.py). This is the fail-loud behavior the
    # fix adopts for parity with the non-stream path.
    assert dispatcher.select_config("https://site-c.example/x", config_list) is None
    # Single shared config (the bug shape) would have returned itself for
    # every URL; with a list, the matched configs are distinct objects with
    # the caller's per-URL knobs intact (css_selector is not subject to the
    # untrusted clamps, so the values pass through verbatim).
    assert a.css_selector == "article"
    assert b.css_selector == "main"


def test_stream_multi_url_rejects_forbidden_untrusted_fields(monkeypatch):
    """Per-URL configs are loaded under `Provenance.UNTRUSTED` and must still
    reject forbidden power-fields (e.g. `proxy_config`,
    `deep_crawl_strategy`) with HTTP 400 rather than silently dropping them.
    Streaming does not weaken the untrusted gate."""
    crawler = _RecordingCrawler()
    _install_recording_crawler(monkeypatch, crawler)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            api.handle_stream_crawl_request(
                urls=["https://site-a.example/x", "https://site-b.example/x"],
                browser_config={},
                crawler_config={"stream": True},
                config=_stream_config(),
                crawler_configs=[
                    {
                        "url_matcher": "*site-a*",
                        "proxy_config": {"server": "http://evil:8080"},
                    },
                    {"url_matcher": "*site-b*"},
                ],
            )
        )
    assert exc.value.status_code == 400
    assert "Rejected request" in str(exc.value.detail)


# ─── backward compatibility ─────────────────────────────────────────────


def test_stream_multi_url_without_crawler_configs_uses_single_config(monkeypatch):
    """Multi-URL streaming without `crawler_configs` must use the single shared
    `CrawlerRunConfig` (the original behavior) -- backward compatible."""
    crawler = _RecordingCrawler()
    _install_recording_crawler(monkeypatch, crawler)

    asyncio.run(
        api.handle_stream_crawl_request(
            urls=["https://site-a.example/x", "https://site-b.example/x"],
            browser_config={},
            crawler_config={"stream": True, "cache_mode": "bypass"},
            config=_stream_config(),
        )
    )

    config_arg = crawler.arun_many_calls[0]["config"]
    assert isinstance(config_arg, CrawlerRunConfig)
    assert not isinstance(config_arg, list)
