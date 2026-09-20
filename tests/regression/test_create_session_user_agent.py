"""
Crawl4AI Regression Tests - create_session user_agent initialization

Regression coverage for a crash in `AsyncPlaywrightCrawlerStrategy.create_session`
introduced by commit 0982c63: the refactor that centralized the user agent into
``BrowserConfig`` dropped the ``self.user_agent`` initializer from ``__init__``
while leaving ``create_session`` reading ``kwargs.get("user_agent",
self.user_agent)``. Because Python evaluates the default-argument expression
eagerly, the missing attribute was accessed unconditionally on any strategy
instance that had not been warmed by a prior ``arun``/``crawl`` with a non-empty
``config.user_agent``.

These tests guard against the regression returning:
  * ``__init__`` seeds ``self.user_agent`` from ``BrowserConfig.user_agent``;
  * ``create_session`` no longer raises ``AttributeError`` on a cold strategy;
  * explicit ``user_agent``/``session_id`` kwargs are honored (a second, masked
    ``TypeError`` from duplicate-kwarg forwarding, exposed once the seed stopped
    the early ``AttributeError``);
  * the session produced by ``create_session`` is reusable by a subsequent crawl
    via ``CrawlerRunConfig(session_id=...)``.

Real-browser tests use the local HTTP server fixture and no behavior-replacing
mocks.
"""

import uuid

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


def test_user_agent_seeded_in_init():
    """A freshly constructed strategy must expose ``self.user_agent`` (regression: it was missing)."""
    strategy = AsyncPlaywrightCrawlerStrategy()
    try:
        assert hasattr(strategy, "user_agent"), "self.user_agent must be initialized in __init__"
        assert strategy.user_agent == strategy.browser_config.user_agent
    finally:
        # No browser was started, but keep the shape symmetric with other tests.
        del strategy


@pytest.mark.asyncio
async def test_create_session_on_cold_strategy_returns_uuid():
    """``create_session`` on a never-warmed strategy must return a valid UUID, not raise."""
    strategy = AsyncPlaywrightCrawlerStrategy(browser_config=BrowserConfig(headless=True, verbose=False))
    try:
        session_id = await strategy.create_session()
    finally:
        await strategy.close()
    assert isinstance(session_id, str)
    parsed = uuid.UUID(session_id)
    assert str(parsed) == session_id


@pytest.mark.asyncio
async def test_create_session_with_explicit_user_agent_kwarg_no_crash():
    """An explicit user_agent kwarg must not trigger AttributeError or duplicate-kwarg TypeError."""
    strategy = AsyncPlaywrightCrawlerStrategy(browser_config=BrowserConfig(headless=True, verbose=False))
    try:
        session_id = await strategy.create_session(user_agent="MyUA/1.0")
    finally:
        await strategy.close()
    assert isinstance(session_id, str)
    uuid.UUID(session_id)


@pytest.mark.asyncio
async def test_create_session_honors_explicit_session_id_kwarg():
    """An explicit session_id kwarg must be returned as-is so callers can pin a session."""
    strategy = AsyncPlaywrightCrawlerStrategy(browser_config=BrowserConfig(headless=True, verbose=False))
    try:
        session_id = await strategy.create_session(session_id="my-pinned-session")
    finally:
        await strategy.close()
    assert session_id == "my-pinned-session"


@pytest.mark.asyncio
async def test_create_session_reusable_by_subsequent_crawl(local_server):
    """A session_id from create_session must be reusable with CrawlerRunConfig(session_id=...)."""
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        session_id = await crawler.crawler_strategy.create_session()
        assert isinstance(session_id, str)
        uuid.UUID(session_id)

        config = CrawlerRunConfig(session_id=session_id)
        result = await crawler.arun(url=local_server + "/", config=config)
        assert result.success, f"Reuse crawl failed: {result.error_message}"
        assert "<h1>" in result.html

        result2 = await crawler.arun(url=local_server + "/", config=config)
        assert result2.success, f"Second reuse crawl failed: {result2.error_message}"
