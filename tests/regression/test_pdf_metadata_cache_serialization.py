"""Regression test for the PDF metadata cache-serialization bug.

Bug: ``PDFContentScrapingStrategy.scrap`` flattened ``PDFMetadata`` with a
bare ``dataclasses.asdict(result.metadata)``, preserving the native
``datetime.datetime`` objects that ``NaivePDFProcessorStrategy._parse_pdf_date``
places into ``created`` / ``modified`` when the PDF carries a ``/CreationDate``
or ``/ModDate`` matching the ``D:YYYYMMDDHHMMSS`` regex. The dict flowed
unchanged into ``CrawlResult.metadata``; the cache writer then called the
stdlib ``json.dumps(result.metadata or {})`` (no ``default=`` handler) in
``async_database.AsyncDatabaseManager.acache_url`` and raised
``TypeError: Object of type datetime is not JSON serializable``. The
exception was caught and logged ("Error caching URL") but swallowed, so the
crawl reported ``success=True`` while no cache row was written; every
subsequent crawl of the same URL re-downloaded and re-parsed the whole PDF.

Fix: ``scrap()`` now coerces the ``created`` / ``modified`` datetimes to
ISO-8601 strings at the strategy boundary (``_metadata_as_jsonable_dict``),
leaving the typed ``Optional[datetime]`` on ``PDFMetadata`` honest to its
declared type while guaranteeing the dict the cache writer receives is
JSON-native.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_pdf_metadata_cache_serialization.py
"""

import datetime
import json
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf", reason="requires the crawl4ai[pdf] extra")

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.async_database import AsyncDatabaseManager
from crawl4ai.processors.pdf import (
    PDFContentScrapingStrategy,
    PDFCrawlerStrategy,
)


def _pdf_date_str(dt: datetime.datetime) -> str:
    """The D:YYYYMMDDHHMMSS+00'00 form pypdf writes for a /CreationDate."""
    return dt.strftime("D:%Y%m%d%H%M%S+00'00")


@pytest.fixture
def dated_pdf(tmp_path):
    """A tiny, pypdf-valid single-page PDF carrying /CreationDate and /ModDate
    in the documented D:YYYYMMDDHHMMSS prefix form -- the exact trigger shape
    (verified against arXiv 2310.06825, which sets both fields to
    'D:20231011004817Z')."""
    from pypdf import PdfWriter

    path = tmp_path / "dated.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata(
        {
            "/Title": "Pdf Cache Test",
            "/Producer": "crawl4ai",
            "/CreationDate": _pdf_date_str(datetime.datetime(2024, 1, 1, 12, 0, 0)),
            "/ModDate": _pdf_date_str(datetime.datetime(2024, 1, 1, 12, 1, 0)),
        }
    )
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def dateless_pdf(tmp_path):
    """A PDF with no /CreationDate and no /ModDate -- the only fixture the
    pre-existing PDF crawl test (tests/test_placeholder_html_antibot.py) ever
    built. Its /Info has no dates, so _parse_pdf_date returns None for both
    fields; the fix must keep passing None through unchanged."""
    from pypdf import PdfWriter

    path = tmp_path / "dateless.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)
    return path


async def _setup_manager(tmp_path) -> AsyncDatabaseManager:
    """Build a fully-initialized, isolated AsyncDatabaseManager under tmp_path
    (mirrors tests/regression/test_async_database_markdown_roundtrip.py)."""
    from crawl4ai.utils import ensure_content_dirs

    mgr = AsyncDatabaseManager()
    mgr.db_path = str(tmp_path / "test.db")
    mgr.content_paths = ensure_content_dirs(str(tmp_path))
    await mgr.ainit_db()
    await mgr.update_db_schema()
    mgr._initialized = True
    return mgr


async def _crawl_file(pdf_file, manager, cache_mode=CacheMode.ENABLED):
    """Crawl a local ``file://`` PDF via the documented strategy pairing,
    routing the crawler's cache manager to the isolated ``manager``.

    ``file://`` URLs are *write-only* in the cache layer (``should_read`` is
    False for non-web URLs because local files have no ETag/Last-Modified to
    freshness-validate against), so this helper verifies the WRITE side of
    the bug: that the cache row is actually persisted.
    """
    config = CrawlerRunConfig(
        cache_mode=cache_mode,
        scraping_strategy=PDFContentScrapingStrategy(extract_images=False),
    )
    import crawl4ai.async_webcrawler as awc

    original = awc.async_db_manager
    awc.async_db_manager = manager
    try:
        async with AsyncWebCrawler(crawler_strategy=PDFCrawlerStrategy()) as crawler:
            return await crawler.arun(f"file://{pdf_file}", config=config)
    finally:
        awc.async_db_manager = original


def test_scrap_metadata_is_json_serializable_when_pdf_has_dates(dated_pdf):
    """The exact pre-fix crash: ``json.dumps(result.metadata)`` on a dated PDF.
    Pre-fix this raised ``TypeError: Object of type datetime is not JSON
    serializable`` because ``created``/``modified`` were raw datetimes. The fix
    coerces them to ISO-8601 strings at the strategy boundary; this assertion
    also catches any future non-JSON-native value sneaking into the dict."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    result = strategy.scrap(f"file://{dated_pdf}", html="")

    assert result.success
    assert isinstance(result.metadata, dict)
    assert isinstance(result.metadata["created"], str)
    assert isinstance(result.metadata["modified"], str)
    assert result.metadata["created"] == "2024-01-01T12:00:00"
    assert result.metadata["modified"] == "2024-01-01T12:01:00"
    # No datetime object survives anywhere in the metadata dict.
    assert not any(isinstance(v, datetime.datetime) for v in result.metadata.values())

    # The exact call acache_url makes at async_database.py:618 -- must not raise.
    json.dumps(result.metadata)


def test_scrap_metadata_for_dateless_pdf_keeps_none(dateless_pdf):
    """No-regression: a PDF with no /CreationDate / /ModDate must keep
    ``created`` / ``modified`` as ``None`` (the fix passes None through
    unchanged, not coerced to a string). Pre-existing dateless-PDF crawls
    already cached fine; the fix must not change that."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    result = strategy.scrap(f"file://{dateless_pdf}", html="")

    assert result.success
    assert result.metadata["created"] is None
    assert result.metadata["modified"] is None
    json.dumps(result.metadata)  # must not raise


@pytest.mark.asyncio
async def test_e2e_dated_pdf_writes_cache_row_on_default_path(tmp_path, dated_pdf):
    """The reported bug's core: a dated PDF crawled with the default
    CacheMode.ENABLED must WRITE the cache row. Pre-fix the row was silently
    dropped because ``acache_url``'s ``json.dumps(result.metadata)`` raised
    ``TypeError: Object of type datetime is not JSON serializable`` and the
    exception was swallowed by the ``execute_with_retry`` try/except.

    Wraps ``execute_with_retry`` to capture any exception ``acache_url``
    swallows (it catches -> logs -> returns), mirroring Evidence #1 in the
    bug report. Post-fix this list must stay empty AND the row must exist.
    """
    pdf_url = f"file://{dated_pdf}"
    mgr = await _setup_manager(tmp_path)

    captured = []
    orig_exec = mgr.execute_with_retry

    async def recording_exec(func):
        try:
            return await orig_exec(func)
        except Exception as e:
            captured.append((type(e).__name__, str(e)))
            raise

    mgr.execute_with_retry = recording_exec

    first = await _crawl_file(dated_pdf, mgr, cache_mode=CacheMode.ENABLED)
    assert first.success, f"first crawl failed: {first.error_message}"
    assert isinstance(first.metadata["created"], str)
    assert first.metadata["created"] == "2024-01-01T12:00:00"
    # The exact call acache_url makes at async_database.py:618 -- must not raise.
    json.dumps(first.metadata)

    cached = await mgr.aget_cached_url(pdf_url)
    assert cached is not None, (
        "cache row missing after a successful crawl -- the swallowed "
        "TypeError is back; created/modified were not coerced to str"
    )
    assert cached.metadata["created"] == "2024-01-01T12:00:00"
    assert cached.metadata["modified"] == "2024-01-01T12:01:00"

    assert captured == [], (
        f"acache_url swallowed an exception that should be gone: {captured}"
    )
