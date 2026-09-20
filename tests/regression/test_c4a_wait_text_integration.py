#!/usr/bin/env python3
"""
Integration tests for the C4A-Script WAIT text injection fix, exercised through
the full ``CrawlerRunConfig(c4a_script=...)`` → compile → ``js_code`` →
``robust_execute_user_script`` → browser pipeline.

Kept in a separate file from the sync-Playwright compiler tests because
``sync_playwright`` and ``pytest-asyncio`` run incompatible event loops in the
same module.

Preconditions: Chromium installed (playwright install chromium) and the
session-scoped ``local_server`` fixture from ``tests/regression/conftest.py``.
"""

import asyncio
import os
import sys

import pytest

pytestmark = pytest.mark.browser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


@pytest.mark.asyncio
async def test_c4a_script_wait_text_resolves_during_crawl(local_server):
    """A normal WAIT text resolves against the live page and the crawl completes."""
    from crawl4ai.async_configs import CrawlerRunConfig, BrowserConfig
    from crawl4ai import AsyncWebCrawler

    config = CrawlerRunConfig(c4a_script='WAIT "Welcome" 3')
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False, extra_args=["--no-sandbox"])) as crawler:
        result = await crawler.arun(local_server + "/", config=config)
        assert result.success, f"Crawl failed: {result.error_message}"
        assert "Welcome to the Crawl4AI Test Site" in result.markdown


@pytest.mark.asyncio
async def test_c4a_script_wait_text_with_dollar_curly_does_not_hang(local_server):
    """c4a_script with ${...} in WAIT text must not hang the crawl.

    Regression for the unbounded-hang defect: before the fix the emitted JS used
    a template literal so ``${never_defined_var}`` threw a ReferenceError on
    every setInterval tick, leaving the promise pending forever (no outer
    timeout wraps the call chain). After the fix the literal text is searched
    for, not found, and the promise rejects at ~1s.
    """
    from crawl4ai.async_configs import CrawlerRunConfig, BrowserConfig
    from crawl4ai import AsyncWebCrawler

    config = CrawlerRunConfig(c4a_script='WAIT "${never_defined_var}" 1')
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False, extra_args=["--no-sandbox"])) as crawler:
        try:
            result = await asyncio.wait_for(
                crawler.arun(local_server + "/", config=config),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            pytest.fail("c4a_script WAIT with ${...} hung the crawl — fix regressed")
        assert result.success, f"Crawl failed: {result.error_message}"
