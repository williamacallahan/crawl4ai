"""End-to-end verification: stale extracted_content on cache fallthrough.

Uses a REAL browser (Playwright/Chromium) + the REAL cache DB
(``async_db_manager`` singleton) + a counting extraction strategy, so the
fallthrough-and-re-extract path is exercised end-to-end (not mocked).

Crawl #1 seeds the cache (ENABLED) with extracted_content = {"extraction_call": 1}.
Crawl #2 re-crawls the same URL with pdf=True (pdf is never persisted, so the
cache entry is always missing pdf -> fallthrough).  With the fix, the strategy
re-runs (call #2) and returns fresh extracted_content; without the fix the
stale {"extraction_call": 1} is returned and the strategy is never called.
Crawl #3 is a pure cache hit (no artifact requested) confirming the fresh
value was re-persisted (durability, G3/T8).

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/regression/test_e2e_stale_extraction_fallthrough.py
"""
import uuid

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.cache_context import CacheMode
from crawl4ai.extraction_strategy import ExtractionStrategy


class CountingStrategy(ExtractionStrategy):
    """Returns a value tagged with its invocation count.

    A re-run on the fallthrough must increment the counter, so the returned
    marker distinguishes fresh (call N+1) from stale-carried (call N).
    """

    def __init__(self):
        super().__init__(input_format="markdown")
        self.calls = 0

    def extract(self, url, html, *q, **kwargs):
        self.calls += 1
        return [{"extraction_call": self.calls}]

    async def arun(self, url, sections, *q, **kwargs):
        self.calls += 1
        return [{"extraction_call": self.calls}]


@pytest.mark.browser
@pytest.mark.asyncio
async def test_e2e_pdf_fallthrough_reruns_extraction_and_persists_fresh(local_server):
    strategy = CountingStrategy()
    # Unique URL -> routes to the rich home page; unique query => unique cache
    # key so no pre-existing cache entry interferes.
    url = f"{local_server}/?e2e_stale={uuid.uuid4().hex}"

    async with AsyncWebCrawler(
        # The CI browser-gate container runs as appuser on a kernel without
        # unprivileged user namespaces, so the sandboxed Chromium zygote dies
        # ("No usable sandbox!"). extra_args mirrors the operator policy
        # test_real_browser_recycle_window.py applies for the same container.
        config=BrowserConfig(headless=True, verbose=False, extra_args=["--no-sandbox"])
    ) as crawler:
        # Crawl #1: seed cache with strategy output {call: 1}.
        r1 = await crawler.arun(
            url,
            config=CrawlerRunConfig(
                extraction_strategy=strategy, cache_mode=CacheMode.ENABLED
            ),
        )
        assert r1.success, f"crawl #1 failed: {r1.error_message}"
        assert strategy.calls == 1
        assert '"extraction_call": 1' in (r1.extracted_content or "")

        # Crawl #2: pdf=True forces the fallthrough (pdf never persisted to DB).
        r2 = await crawler.arun(
            url,
            config=CrawlerRunConfig(
                pdf=True,
                extraction_strategy=strategy,
                cache_mode=CacheMode.ENABLED,
            ),
        )
        assert r2.success, f"crawl #2 failed: {r2.error_message}"
        assert r2.pdf is not None and isinstance(r2.pdf, bytes)
        assert r2.pdf[:4] == b"%PDF", "pdf artifact must be produced on fallthrough"
        # The strategy must have re-run on the fresh fetch.
        assert strategy.calls == 2, (
            f"extraction strategy was not re-run on the pdf fallthrough "
            f"(calls={strategy.calls}); stale cached extracted_content was "
            f"carried into aprocess_html and short-circuited extraction"
        )
        # And the returned extracted_content must be the FRESH value (call 2),
        # not the stale cached value (call 1).
        assert '"extraction_call": 2' in (r2.extracted_content or ""), (
            "stale cached extracted_content was returned unchanged by the "
            "pdf fallthrough instead of the freshly re-extracted value"
        )
        assert '"extraction_call": 1' not in (r2.extracted_content or "")

        # Crawl #3: pure cache hit (no artifact requested) -> must return the
        # re-persisted fresh value (call 2), proving the stale value is no
        # longer durable in the cache (G3/T8).
        r3 = await crawler.arun(
            url,
            config=CrawlerRunConfig(
                extraction_strategy=strategy, cache_mode=CacheMode.ENABLED
            ),
        )
        assert r3.success, f"crawl #3 failed: {r3.error_message}"
        assert r3.cache_status == "hit", (
            f"crawl #3 should be a cache hit, got {r3.cache_status}"
        )
        assert strategy.calls == 2, (
            "crawl #3 must not re-run extraction on a pure cache hit"
        )
        assert '"extraction_call": 2' in (r3.extracted_content or ""), (
            "the freshly re-extracted value was not durably re-persisted to "
            "the cache; the stale value would otherwise compound across crawls"
        )
