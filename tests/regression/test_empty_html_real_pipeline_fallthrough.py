"""Real-pipeline regression: empty-HTML cache row from a binary download.

The trigger precondition — a cache row with ``html == ""`` AND a non-empty,
non-``"[]"`` ``extracted_content`` — is produced by a real empty-HTML producer
(binary download: ``html=""`` + ``downloaded_files=[...]``) combined with a
content-independent extraction strategy.  These tests exercise the REAL
``arun`` -> ``aprocess_html`` -> ``acache_url`` -> ``aget_cached_url`` pipeline
against a REAL, isolated SQLite cache DB; only the browser fetch
(``crawler_strategy.crawl``) is mocked (the same surface the existing
``f82b301`` regression tests mock) to deterministically return a download
response.

Crawl #1 (Producer A: binary download) writes the trigger row to the real DB:
``html==""``, ``extracted_content=='[{"extraction_call": 1}]'``.
Crawl #2 reads that row -> the ``not html`` fallthrough launches a fresh fetch
(now returning a real HTML page so the "fresh HTML alongside stale extraction"
symptom is visible).  With the fix, the stale ``extracted_content`` is cleared
and the strategy re-runs (call #2, fresh).  Without the fix the stale call #1
is returned with the fresh HTML and the strategy is never re-run.

The DB is isolated to a temp path (not the shared ``~/.crawl4ai`` DB) and fully
restored on teardown so other tests — including the browser-gate e2e that uses
the shared DB — are unaffected.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_empty_html_real_pipeline_fallthrough.py
"""
import os
import uuid

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
from crawl4ai.cache_context import CacheMode
from crawl4ai.extraction_strategy import ExtractionStrategy
from crawl4ai.models import AsyncCrawlResponse
from crawl4ai.utils import ensure_content_dirs

URL = "https://example.com"
# A non-empty binary downloaded "file" served by the mocked crawl strategy.
DOWNLOAD_BYTES = b"%PDF-1.4 BINARY_DOWNLOAD_BODY %%EOF"
# Real HTML returned by the SECOND crawl's fresh fetch, large enough to pass
# the anti-bot ``is_blocked`` check (>100 bytes, real visible text) so the
# "fresh HTML alongside stale extraction" symptom is observable.
LIVE_HTML = (
    "<html><head><title>Example Article</title></head><body><article>"
    "<h1>Live fetch completed successfully</h1>"
    "<p>This is the freshly fetched page content returned by the crawler "
    "strategy after the cached empty-HTML row was re-read and the not-html "
    "fallthrough launched a fresh fetch.</p></article></body></html>"
)


class CountingStrategy(ExtractionStrategy):
    """Content-independent strategy: tags output with its invocation count.

    A re-run on the fallthrough increments the counter, so the returned marker
    distinguishes fresh (call N+1) from stale-carried (call N).  Content-
    independent on purpose — only such a strategy can seed the trigger row
    (built-in content-dependent strategies return ``[]`` on empty HTML)."""

    def __init__(self):
        super().__init__(input_format="markdown")
        self.calls = 0

    def extract(self, url, html, *q, **kwargs):
        self.calls += 1
        return [{"extraction_call": self.calls}]

    async def arun(self, url, sections, *q, **kwargs):
        self.calls += 1
        return [{"extraction_call": self.calls}]


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

    def error(self, message, tag="ERROR", **kwargs):
        self.events.append(("error", tag, message))

    def success(self, message, tag="COMPLETE", **kwargs):
        self.events.append(("info", tag, message))


class _ResponseStrategy:
    """Minimal ``AsyncCrawlerStrategy`` stub returning scripted responses in order.

    Crawl #1 returns a binary-download response (empty HTML + a downloaded
    file); crawl #2 returns a real HTML page, mirroring the bug-report
    reproduction where the re-fetch returns real HTML to expose the symptom."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.crawl_calls = 0

    async def crawl(self, url, config):
        response = self.responses[min(self.crawl_calls, len(self.responses) - 1)]
        self.crawl_calls += 1
        return response

    def update_user_agent(self, *args, **kwargs):
        pass


async def _isolate_db(monkeypatch, tmp_path):
    """Redirect the ``async_db_manager`` singleton to an isolated temp SQLite DB
    with the full ``crawled_data`` schema, and skip the singleton's auto-init
    (which would run migrations).  Returns the temp download file path."""
    db_path = str(tmp_path / "test_crawl4ai.db")
    content_base = str(tmp_path / "content")
    downloads_dir = str(tmp_path / "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # Replace the singleton's DB path, content dirs, and init flag so that
    # the real ``aget_cached_url``/``acache_url`` hit the temp DB.  monkeypatch
    # restores every attribute at teardown, so the shared DB used by other
    # tests (including the browser-gate e2e) is untouched afterward.
    monkeypatch.setattr(async_db_manager, "db_path", db_path)
    monkeypatch.setattr(async_db_manager, "content_paths", ensure_content_dirs(content_base))
    monkeypatch.setattr(async_db_manager, "_initialized", True)
    monkeypatch.setattr(async_db_manager, "connection_pool", {})

    await async_db_manager.ainit_db()
    await async_db_manager.update_db_schema()

    return downloads_dir


def _make_crawler(tmp_path, responses):
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(responses),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    return crawler, logger


async def _read_db_row(url):
    """Re-read the cached ``CrawlResult`` for ``url`` via the REAL read path.

    ``acache_url`` stores content fields (html/extracted_content/...) on disk,
    keyed by content hash, NOT raw in the DB columns; ``aget_cached_url`` is the
    path that loads them back.  Reading through it (the same path ``arun`` uses)
    verifies the trigger row's *reconstructed* shape, which is what actually
    feeds the empty-HTML fallthrough on the next crawl.
    """
    return await async_db_manager.aget_cached_url(url)


@pytest.mark.asyncio
async def test_real_pipeline_download_producer_then_empty_html_fallthrough(
    tmp_path, monkeypatch
):
    """End-to-end through the real pipeline (real DB, real aprocess_html):

    1. Crawl #1 (binary download + content-independent strategy) persists the
       trigger row (html=="", extracted_content==call 1) to the REAL DB.
    2. Crawl #2 re-reads that row -> the not-html fallthrough launches a fresh
       fetch returning real HTML.  With the fix the stale extraction is cleared
       and the strategy re-runs (call 2) returning fresh extraction alongside
       the fresh HTML; without the fix the stale call 1 is returned and the
       strategy never re-runs.
    3. The fresh result is NOT re-persisted (cached_result stays truthy on the
       not-html path -> write guard stays False -> stickiness), so the DB row
       keeps the empty HTML / stale marker.
    """
    downloads_dir = await _isolate_db(monkeypatch, tmp_path)
    download_path = os.path.join(downloads_dir, "report.pdf")
    with open(download_path, "wb") as f:
        f.write(DOWNLOAD_BYTES)

    strategy = CountingStrategy()
    url = f"{URL}?empty_html_pipeline={uuid.uuid4().hex}"

    crawler, logger = _make_crawler(
        tmp_path,
        responses=[
            # Crawl #1: binary download (Producer A).  html="" by design;
            # downloaded_files populated; status_code 200.  success is
            # computed as bool(html) or bool(downloaded_files) -> True.
            AsyncCrawlResponse(
                html="",
                response_headers={
                    "Content-Type": "application/pdf",
                    "Content-Disposition": 'attachment; filename="report.pdf"',
                },
                status_code=200,
                downloaded_files=[download_path],
            ),
            # Crawl #2: real HTML page so the "fresh HTML alongside stale
            # extraction" symptom is observable on the fresh fetch.
            AsyncCrawlResponse(
                html=LIVE_HTML,
                response_headers={},
                status_code=200,
            ),
        ],
    )

    # --- Crawl #1: seed the cache with the trigger row (Producer A) ---
    r1 = await crawler.arun(
        url,
        CrawlerRunConfig(
            extraction_strategy=strategy,
            cache_mode=CacheMode.ENABLED,
        ),
    )
    assert r1.success, f"crawl #1 failed: {r1.error_message}"
    assert r1.html == "", "crawl #1 (download) should have empty HTML"
    assert r1.downloaded_files == [download_path], "download file must be captured"
    assert strategy.calls == 1, "extraction strategy must run once on the seed crawl"
    assert '"extraction_call": 1' in (r1.extracted_content or "")

    # The trigger row genuinely exists in the REAL isolated DB.  Read it back
    # via the same ``aget_cached_url`` path ``arun`` uses on the next crawl.
    cached1 = await _read_db_row(url)
    assert cached1 is not None, "crawl #1 must persist a cache row"
    assert cached1.html == "", "persisted row must have empty HTML (trigger)"
    assert (
        "extraction_call" in (cached1.extracted_content or "")
    ), "persisted row must carry the non-empty extracted_content (trigger)"

    # --- Crawl #2: re-read empty-HTML row -> not-html fallthrough -> fresh fetch ---
    r2 = await crawler.arun(
        url,
        CrawlerRunConfig(
            extraction_strategy=strategy,
            cache_mode=CacheMode.ENABLED,
        ),
    )
    assert r2.success, f"crawl #2 failed: {r2.error_message}"
    # Fresh HTML was fetched (the fallthrough re-ran the crawl strategy).
    assert r2.html == LIVE_HTML, "crawl #2 must return the freshly fetched HTML"
    assert crawler.crawler_strategy.crawl_calls == 2, "crawl #2 must perform a fresh fetch"
    # THE regression assertion: the strategy must have re-run on the fresh
    # fetch (call 2), returning fresh extraction — not the stale call 1.
    assert strategy.calls == 2, (
        f"extraction strategy was not re-run on the empty-HTML fallthrough "
        f"(calls={strategy.calls}); the stale cached extracted_content was "
        f"carried into aprocess_html and short-circuited extraction"
    )
    assert '"extraction_call": 2' in (r2.extracted_content or ""), (
        "stale cached extracted_content (call 1) was returned unchanged by the "
        "empty-HTML fallthrough instead of the freshly re-extracted value (call 2)"
    )
    assert '"extraction_call": 1' not in (r2.extracted_content or "")

    # Stickiness: the empty-HTML fallthrough keeps cached_result truthy, so the
    # cache-write guard stays False and the fresh result is NOT re-persisted.
    # The DB row still holds the empty HTML / stale marker from crawl #1.
    cached2 = await _read_db_row(url)
    assert cached2 is not None
    assert cached2.html == "", (
        "the empty-HTML cache row should not have been overwritten (stickiness: "
        "the not-html fallthrough keeps cached_result truthy -> write guard False)"
    )
    assert "extraction_call" in (cached2.extracted_content or "")
