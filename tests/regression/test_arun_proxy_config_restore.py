"""
Regression tests for AsyncWebCrawler.arun() proxy_config restoration.

arun() saves the caller's `config.proxy_config`, mutates it while rotating
proxies in the anti-bot retry loop, and must restore it before returning or
propagating — otherwise a failed crawl leaves the caller-owned
CrawlerRunConfig mutated, breaking reuse.

The restoration is performed in a `finally` block around the retry loop so it
runs on every exit path. These tests lock in that invariant for the two paths
where a naive alternative (e.g. restoring on the success path only, or just
before the inner bare `raise`) would silently leak the mutation:

  * the single-proxy / no-retry non-target exception that re-raises out of the
    loop (the reported bug), and
  * a BaseException such as asyncio.CancelledError that propagates straight
    through `except Exception` — only a `finally` restores here.

Both use a mocked crawler_strategy, so they need no browser, network, or
real proxy.
"""

import asyncio

import pytest

from crawl4ai.async_configs import CrawlerRunConfig, ProxyConfig
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.cache_context import CacheMode


class _RecordingLogger:
    def __init__(self):
        self.verbose = True
        self.events = []

    def info(self, message, tag="INFO", **kwargs):
        self.events.append(("info", tag, message))

    def warning(self, message, tag="WARNING", **kwargs):
        self.events.append(("warning", tag, message))

    def error_status(self, url, error, tag="ERROR", **kwargs):
        self.events.append(("error", tag, error))

    def url_status(self, url, success, timing, tag="FETCH", **kwargs):
        self.events.append(("success" if success else "error", tag, url))


class _ResponseStrategy:
    """Minimal AsyncCrawlerStrategy stub: raises a fixed exception."""

    def __init__(self, error):
        self.error = error

    async def crawl(self, url, config):
        raise self.error


def _make_crawler(tmp_path, error):
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(error=error),
        base_directory=str(tmp_path),
        logger=_RecordingLogger(),
    )
    crawler.ready = True
    return crawler


@pytest.mark.asyncio
async def test_proxy_config_restored_after_oneshot_raise(tmp_path):
    """Single-proxy list + no retries + non-target error triggers `raise`
    inside the retry loop. The outer handler returns a failure container, but
    the caller-owned config must be left untouched so it can be reused."""
    proxy_a = ProxyConfig(server="http://proxy-a.example:8080")
    config = CrawlerRunConfig(
        proxy_config=[proxy_a],
        cache_mode=CacheMode.BYPASS,
    )
    original = config.proxy_config
    assert isinstance(original, list) and len(original) == 1

    crawler = _make_crawler(tmp_path, error=RuntimeError("simulated crawl error"))
    result = await crawler.arun("https://example.com", config=config)

    assert not result.success
    # A list must stay a list — pre-fix the mutation silently flattened it
    # to a single ProxyConfig and never restored it.
    assert isinstance(config.proxy_config, list)
    assert config.proxy_config == original
    assert config._get_proxy_list() == [proxy_a]


@pytest.mark.asyncio
async def test_proxy_config_restored_on_cancelled_error(tmp_path):
    """asyncio.CancelledError is a BaseException: it is NOT caught by the
    loop's `except Exception`, so it propagates out of arun() unchanged —
    yet proxy_config must already be restored. This is the path that only a
    `finally` covers; a surgical "restore before the bare raise" would leave
    the config mutated here."""
    proxy_a = ProxyConfig(server="http://proxy-a.example:8080")
    config = CrawlerRunConfig(
        proxy_config=[proxy_a],
        cache_mode=CacheMode.BYPASS,
    )
    original = config.proxy_config

    crawler = _make_crawler(tmp_path, error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await crawler.arun("https://example.com", config=config)

    assert isinstance(config.proxy_config, list)
    assert config.proxy_config == original
