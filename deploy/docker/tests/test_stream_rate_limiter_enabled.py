"""Regression coverage for the parity between ``handle_crawl_request`` and
``handle_stream_crawl_request`` around ``crawler.rate_limiter.enabled``.

Both handlers build a ``MemoryAdaptiveDispatcher`` from the same
``config["crawler"]["rate_limiter"]`` block to drive multi-URL crawls.  The
non-streaming handler gates the ``RateLimiter`` construction on the ``enabled``
flag (passing ``None`` when disabled); the streaming handler's multi-URL
non-deep-crawl branch must do the same, so that an operator who follows the
docs-recommended ``rate_limiter.enabled: false`` tuning gets consistent
behavior on ``/crawl`` and ``/crawl/stream``.

These tests capture the ``dispatcher`` argument the handlers actually build
and assert on ``dispatcher.rate_limiter`` for both endpoints under both
``enabled`` directions, guarding the parity invariant.
"""

import asyncio
import types

import api
import crawler_pool
import egress_broker

from crawl4ai import MemoryAdaptiveDispatcher, RateLimiter


# ---------------------------------------------------------------------------
# Fake stack: minimal, capturing the dispatcher each handler constructs.
# ---------------------------------------------------------------------------


class _StreamingCaptureCrawler:
    """Pooled crawler for the streaming handler: ``arun_many`` returns an
    async generator (the streaming contract) and records the dispatcher."""

    def __init__(self):
        self.captured = {}
        self.crawler_strategy = types.SimpleNamespace(hooks={})

    async def start(self):
        pass

    async def close(self):
        pass

    async def arun_many(self, *_args, dispatcher=None, **_kwargs):
        self.captured["dispatcher"] = dispatcher

        async def results():
            for url in ("https://example.com/a", "https://example.com/b"):
                yield types.SimpleNamespace(
                    url=url,
                    html="",
                    metadata={},
                    success=True,
                    status_code=200,
                    error_message=None,
                    model_dump=lambda u=url: {
                        "url": u,
                        "success": True,
                        "fit_html": None,
                        "pdf": None,
                    },
                )

        return results()


class _NonStreamingCaptureCrawler:
    """Pooled crawler for the non-streaming handler.  A single-URL request
    drives ``arun`` (the handler builds the dispatcher before choosing
    ``arun`` vs ``arun_many``), which records the dispatcher and returns one
    result."""

    def __init__(self):
        self.captured = {}
        self.crawler_strategy = types.SimpleNamespace(hooks={})

    async def start(self):
        pass

    async def close(self):
        pass

    async def arun(self, *_args, dispatcher=None, **_kwargs):
        self.captured["dispatcher"] = dispatcher
        return types.SimpleNamespace(
            url="https://example.com",
            html="",
            metadata={},
            success=True,
            status_code=200,
            error_message=None,
            model_dump=lambda: {
                "url": "https://example.com",
                "success": True,
                "fit_html": None,
                "pdf": None,
            },
        )


class _FakeBrowserConfig:
    verbose = False

    @classmethod
    def load(cls, _value, **_kwargs):
        return cls()


class _FakeCrawlerRunConfig:
    deep_crawl_strategy = None
    scraping_strategy = None
    stream = False

    @classmethod
    def load(cls, _value, **_kwargs):
        return cls()


def _config(*, enabled, base_delay=(1.0, 2.0)):
    return {
        "crawler": {
            "base_config": {},
            "pool": {"max_pages": 2},
            "memory_threshold_percent": 95,
            "recovery_threshold_percent": 80,
            "rate_limiter": {"enabled": enabled, "base_delay": list(base_delay)},
            "browser": {"kwargs": {}},
        },
        "limits": {"wall_clock_s": 0},
    }


def _install(monkeypatch, pooled):
    released = []

    async def get_crawler(_config):
        return pooled

    async def release_crawler(candidate):
        released.append(candidate)

    monkeypatch.setattr(api, "BrowserConfig", _FakeBrowserConfig)
    monkeypatch.setattr(api, "CrawlerRunConfig", _FakeCrawlerRunConfig)
    monkeypatch.setattr(api, "_apply_server_browser_policy", lambda value, _config: value)
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", release_crawler)
    return released


def run(coro):
    return asyncio.run(coro)


STREAM_URLS = ["https://example.com/a", "https://example.com/b"]


# ---------------------------------------------------------------------------
# Parity: /crawl/stream honors crawler.rate_limiter.enabled
# ---------------------------------------------------------------------------


def test_streaming_handler_passes_none_rate_limiter_when_disabled(monkeypatch):
    """The fix: the multi-URL non-deep-crawl streaming branch must pass
    ``rate_limiter=None`` when ``enabled`` is false, matching /crawl."""
    pooled = _StreamingCaptureCrawler()
    _install(monkeypatch, pooled)

    async def exercise():
        _, results_gen, _, _ = await api.handle_stream_crawl_request(
            STREAM_URLS, {}, {}, _config(enabled=False)
        )
        async for _ in results_gen:
            pass

    run(exercise())

    dispatcher = pooled.captured["dispatcher"]
    assert isinstance(dispatcher, MemoryAdaptiveDispatcher)
    assert dispatcher.rate_limiter is None, (
        "streaming handler must pass rate_limiter=None when enabled=false "
        "(parity with /crawl)"
    )


def test_streaming_handler_attaches_rate_limiter_when_enabled(monkeypatch):
    """Disabling is not a blanket kill-switch: with ``enabled: true`` the
    streaming branch still attaches a live ``RateLimiter``."""
    pooled = _StreamingCaptureCrawler()
    _install(monkeypatch, pooled)

    async def exercise():
        _, results_gen, _, _ = await api.handle_stream_crawl_request(
            STREAM_URLS, {}, {}, _config(enabled=True)
        )
        async for _ in results_gen:
            pass

    run(exercise())

    dispatcher = pooled.captured["dispatcher"]
    assert isinstance(dispatcher.rate_limiter, RateLimiter)
    assert tuple(dispatcher.rate_limiter.base_delay) == (1.0, 2.0)


# ---------------------------------------------------------------------------
# No regression: /crawl keeps honoring crawler.rate_limiter.enabled
# ---------------------------------------------------------------------------


def test_non_streaming_handler_passes_none_rate_limiter_when_disabled(monkeypatch):
    pooled = _NonStreamingCaptureCrawler()
    _install(monkeypatch, pooled)

    response = run(
        api.handle_crawl_request(
            ["https://example.com"], {}, {}, _config(enabled=False)
        )
    )

    assert response["success"] is True
    assert pooled.captured["dispatcher"].rate_limiter is None


def test_non_streaming_handler_attaches_rate_limiter_when_enabled(monkeypatch):
    pooled = _NonStreamingCaptureCrawler()
    _install(monkeypatch, pooled)

    response = run(
        api.handle_crawl_request(
            ["https://example.com"], {}, {}, _config(enabled=True)
        )
    )

    assert response["success"] is True
    assert isinstance(pooled.captured["dispatcher"].rate_limiter, RateLimiter)
    assert tuple(pooled.captured["dispatcher"].rate_limiter.base_delay) == (1.0, 2.0)
