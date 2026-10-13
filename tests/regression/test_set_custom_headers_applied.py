"""Regression tests for ``AsyncPlaywrightCrawlerStrategy.set_custom_headers``.

Background (commit ``0982c639``): the refactor onto ``BrowserConfig`` /
``CrawlerRunConfig`` migrated every crawl-path read of ``self.headers`` over to
``self.browser_config.headers`` / ``self.config.headers`` but left
``set_custom_headers`` writing to the now-orphaned ``self.headers`` attribute.
``__init__`` no longer initialized ``self.headers`` and nothing in the crawl
path read it, so the documented public API ``crawler.crawler_strategy
.set_custom_headers({...})`` became a silent no-op: headers never reached the
Playwright context (the wire).

The fix routes ``set_custom_headers`` to the canonical store the crawl path
reads — ``self.browser_config.headers`` — and *merges* (``dict.update``) rather
than replaces so the ``sec-ch-ua`` default seeded in ``BrowserConfig.__init__``
(``self.headers.setdefault("sec-ch-ua", self.browser_hint)``) is preserved
unless explicitly overridden.

These tests pin the behavior at the wire-application point
(``BrowserManager.setup_context`` → ``context.set_extra_http_headers``) using a
mocked Playwright context, so they run in CI's Unit gate (``-m "not network and
not browser"``) with no browser and no external network — unlike
``tests/async/test_crawler_strategy.py::test_custom_headers``, a live-network
httpbin.org test that is not collected by any CI gate.
"""

import asyncio
from unittest.mock import AsyncMock

from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


def _applied_headers(mock_context: AsyncMock) -> dict:
    """Return the headers passed in the *last* ``set_extra_http_headers`` call.

    ``BrowserManager.setup_context`` calls ``set_extra_http_headers`` twice for
    a default ``BrowserConfig`` (once with ``self.config.headers`` at line 1276,
    once with the combined UA+hints+headers at line 1300). Playwright applies
    them in order, so the final call is the effective wire state.
    """
    assert mock_context.set_extra_http_headers.called, (
        "setup_context never pushed headers onto the context"
    )
    call_args = mock_context.set_extra_http_headers.call_args
    return call_args.args[0] if call_args.args else call_args.kwargs["headers"]


def test_set_custom_headers_reaches_the_wire():
    """Behavior-level proof: after ``set_custom_headers``, the header is pushed
    onto the Playwright context by ``BrowserManager.setup_context``. This is the
    exact regression — re-orphaning ``self.headers`` makes this assertion fail
    because nothing propagates the header to ``browser_config.headers``."""
    strategy = AsyncPlaywrightCrawlerStrategy()
    strategy.set_custom_headers({"X-Test-Header": "TestValue"})

    mock_context = AsyncMock()
    asyncio.run(strategy.browser_manager.setup_context(mock_context))

    assert _applied_headers(mock_context)["X-Test-Header"] == "TestValue"
    # The dead attribute must NOT be created.
    assert "headers" not in strategy.__dict__


def test_set_custom_headers_preserves_sec_ch_ua_default():
    """``update`` (merge) must not clobber the ``sec-ch-ua`` default that
    ``BrowserConfig.__init__`` seeds via ``setdefault``. Guards against a
    replace-instead-of-merge regression (``self.browser_config.headers = headers``
    would drop the seeded hint from the wire)."""
    strategy = AsyncPlaywrightCrawlerStrategy()
    expected_sec_ch_ua = strategy.browser_config.headers["sec-ch-ua"]
    assert expected_sec_ch_ua  # the default exists

    strategy.set_custom_headers({"X-Test-Header": "TestValue"})
    assert strategy.browser_config.headers["sec-ch-ua"] == expected_sec_ch_ua

    mock_context = AsyncMock()
    asyncio.run(strategy.browser_manager.setup_context(mock_context))
    assert _applied_headers(mock_context)["sec-ch-ua"] == expected_sec_ch_ua
