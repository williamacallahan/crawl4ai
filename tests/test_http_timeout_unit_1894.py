"""Behavioral regression coverage for HTTP timeout unit conversion (#1894)."""

from contextlib import asynccontextmanager

import pytest

import crawl4ai.async_crawler_strategy as crawler_strategy
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("page_timeout", "expected_seconds"),
    [(60_000, 60.0), (5_000, 5.0), (500, 0.5), (0, 30)],
)
async def test_http_strategy_converts_page_timeout_to_aiohttp_seconds(
    monkeypatch,
    page_timeout,
    expected_seconds,
):
    """The real HTTP path must pass seconds, not Playwright milliseconds."""
    captured = {}

    @asynccontextmanager
    async def fake_session(_strategy):
        yield object()

    class TimeoutCaptured(Exception):
        pass

    def capture_timeout(**kwargs):
        captured.update(kwargs)
        raise TimeoutCaptured

    monkeypatch.setattr(AsyncHTTPCrawlerStrategy, "_session_context", fake_session)
    monkeypatch.setattr(crawler_strategy, "ClientTimeout", capture_timeout)

    strategy = AsyncHTTPCrawlerStrategy()
    with pytest.raises(TimeoutCaptured):
        await strategy._handle_http(
            "https://example.com",
            CrawlerRunConfig(page_timeout=page_timeout),
        )

    assert captured == {
        "total": expected_seconds,
        "connect": 10,
        "sock_read": 30,
    }
