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

Extended regression coverage (residual hole left by the above fix):
``_metadata_as_jsonable_dict`` only coerced the two date fields, but
``NaivePDFProcessorStrategy._extract_metadata`` populated ``title`` /
``author`` / ``producer`` with the raw ``pypdf`` objects returned by
``reader.metadata.get('/...')`` -- a ``ByteStringObject`` (a ``bytes``
subclass pypdf returns when it cannot decode the /Info string's encoding)
or an ``IndirectObject`` (an unresolved indirect reference whose
``get_object()`` *seeks* the reader's stream). Those flowed through
``asdict`` untouched into the cache writer's ``json.dumps(result.metadata or {})``
(no ``default=`` handler), raised ``TypeError``, and were swallowed -- the
cache row was dropped with the same failure mode documented above. The
fix resolves the raw pypdf objects to ``Optional[str]`` inside
``_extract_metadata`` (where the reader stream is still open, so
``IndirectObject`` references can be dereferenced) and re-coerces them in
``_metadata_as_jsonable_dict`` as a backstop. The fixtures below exercise
both trigger shapes (``/Title`` & co. as ``ByteStringObject`` and as
indirect references) so the "JSON-native" guarantee the helper's docstring
makes actually holds.

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


def _write_info_object(writer, entries):
    """Inject raw ``pypdf`` objects into a ``PdfWriter``'s /Info dictionary.

    ``writer.add_metadata`` only accepts ``str -> str`` and always round-trips
    them through ``TextStringObject`` (a ``str`` subclass, JSON-safe). To force
    the two non-JSON-native return shapes the production read path can produce
    -- ``ByteStringObject`` (a ``bytes`` subclass) and ``IndirectObject`` (an
    unresolved reference) -- the /Info dict must be populated directly. This is
    the minimal way to reproduce the exact ``PdfObject`` subtypes the bug
    report's Evidence #5 describes ``reader.metadata.get('/...')`` returning.
    """
    from pypdf.generic import NameObject

    writer.add_metadata({})  # ensure the /Info dictionary exists
    info = writer._info.get_object()
    for name, value in entries.items():
        info[NameObject(name)] = value


@pytest.fixture
def bytestring_pdf(tmp_path):
    """A PDF whose /Title, /Author and /Producer are all ``ByteStringObject``
    -- the ``bytes`` subclass pypdf's ``create_string_object`` falls back to
    when the bytes have no UTF-16 BOM / NUL pattern and fail
    ``decode_pdfdocencoding``. ``0x7f`` is undefined in pypdf's pdfdoc
    table, so a leading ``\\x7f`` reliably triggers the fall-through on pypdf
    6.x. Pre-fix these reached ``json.dumps`` verbatim and raised
    ``TypeError: Object of type ByteStringObject is not JSON serializable``."""
    from pypdf import PdfWriter
    from pypdf.generic import ByteStringObject

    path = tmp_path / "bytestring.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    _write_info_object(writer, {
        "/Title": ByteStringObject(b"\x7fAB"),
        "/Author": ByteStringObject(b"\x7fEF"),
        "/Producer": ByteStringObject(b"\x7fGH"),
    })
    with open(path, "wb") as f:
        writer.write(f)

    # Sanity: the read-back path actually yields ByteStringObject for all three
    # fields. The regression is meaningless if the fixture silently decodes.
    from pypdf import PdfReader
    meta = PdfReader(path).metadata
    for key in ("/Title", "/Author", "/Producer"):
        assert type(meta.get(key)).__name__ == "ByteStringObject", (
            f"fixture failed to force ByteStringObject for {key} "
            f"(got {type(meta.get(key)).__name__})"
        )
    return path


@pytest.fixture
def indirect_pdf(tmp_path):
    """A PDF whose /Title, /Author and /Producer are all indirect references
    to ``TextStringObject`` values -- the second trigger shape. pypdf's
    ``DocumentInformation.get('/...')`` returns the raw ``IndirectObject``
    rather than resolving it; only the typed ``title`` property has a fallback
    that dereferences, so ``author`` / ``producer`` typed properties would
    *lose* the value (return None). Pre-fix the raw ``IndirectObject`` reached
    ``json.dumps`` and raised
    ``TypeError: Object of type IndirectObject is not JSON serializable``; the
    fix dereferences it inside ``_extract_metadata`` (while the reader stream
    is open) so the text is preserved for all three fields."""
    from pypdf import PdfWriter
    from pypdf.generic import TextStringObject

    path = tmp_path / "indirect.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    _write_info_object(writer, {
        "/Title": writer._add_object(TextStringObject("Indirect Title")),
        "/Author": writer._add_object(TextStringObject("Indirect Author")),
        "/Producer": writer._add_object(TextStringObject("Indirect Producer")),
    })
    with open(path, "wb") as f:
        writer.write(f)

    # Sanity: the read-back path actually yields IndirectObject for all three
    # fields.
    from pypdf import PdfReader
    meta = PdfReader(path).metadata
    for key in ("/Title", "/Author", "/Producer"):
        assert type(meta.get(key)).__name__ == "IndirectObject", (
            f"fixture failed to force IndirectObject for {key} "
            f"(got {type(meta.get(key)).__name__})"
        )
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


def _assert_metadata_is_json_native(metadata: dict) -> None:
    """Every value in the metadata dict the cache writer receives must be one
    of the JSON-native scalar/container types -- never a ``pypdf``
    ``ByteStringObject`` (``bytes`` subclass), ``IndirectObject``, or
    ``datetime``. This is the guarantee ``_metadata_as_jsonable_dict``'s
    docstring makes and what ``acache_url``'s bare ``json.dumps`` relies on."""
    json.dumps(metadata)  # the exact call acache_url makes -- must not raise
    for key, value in metadata.items():
        assert isinstance(value, (str, int, float, bool, type(None), list, dict)), (
            f"metadata[{key!r}] is not JSON-native: {type(value).__name__} = {value!r}"
        )


async def _assert_pdfcrawl_writes_cache_row(
    pdf_file, tmp_path, *, expected_fields=None
):
    """Crawl a local ``file://`` PDF through the real ``arun`` ->
    ``AsyncDatabaseManager.acache_url`` path against an isolated DB and assert
    the cache row is actually persisted.

    Wraps ``execute_with_retry`` to capture any exception ``acache_url``
    swallows (it catches -> logs -> returns without writing a row), mirroring
    Evidence #1 in the bug report. Post-fix the captured list must be empty AND
    the row must exist AND the persisted metadata must be JSON-native.

    ``expected_fields`` optionally asserts specific metadata values round-trip
    through the cache (used to prove indirect references are *resolved*, not
    degraded to None, for ``author`` / ``producer`` too).
    """
    pdf_url = f"file://{pdf_file}"
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

    result = await _crawl_file(pdf_file, mgr, cache_mode=CacheMode.ENABLED)
    assert result.success, f"crawl failed: {result.error_message}"
    _assert_metadata_is_json_native(result.metadata)

    cached = await mgr.aget_cached_url(pdf_url)
    assert cached is not None, (
        "cache row missing after a successful crawl -- a non-JSON-native "
        "value reached json.dumps and acache_url swallowed the TypeError/ValueError"
    )
    _assert_metadata_is_json_native(cached.metadata)
    if expected_fields:
        for key, expected in expected_fields.items():
            assert cached.metadata[key] == expected, (
                f"cached metadata[{key!r}] = {cached.metadata[key]!r}, "
                f"expected {expected!r}"
            )

    assert captured == [], (
        f"acache_url swallowed an exception that should be gone: {captured}"
    )
    return result


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


def test_info_value_as_text_resolves_each_shape_without_a_live_reader():
    """Direct unit coverage for ``NaivePDFProcessorStrategy._info_value_as_text``
    for the shapes that do NOT need an open reader stream: ``None`` (absent
    key), ``TextStringObject`` (``str`` subclass), and ``ByteStringObject``
    (``bytes`` subclass that ``str()`` decodes via a charset fallback). The
    ``IndirectObject`` shape needs a live reader stream and is covered by the
    ``indirect_pdf`` e2e test below instead."""
    from crawl4ai.processors.pdf.processor import NaivePDFProcessorStrategy
    from pypdf.generic import ByteStringObject, TextStringObject

    resolve = NaivePDFProcessorStrategy._info_value_as_text

    assert resolve(None) is None
    assert resolve(TextStringObject("plain title")) == "plain title"
    decoded = resolve(ByteStringObject(b"\x7fAB"))
    assert isinstance(decoded, str)
    assert decoded == str(b"\x7fAB".decode("latin-1"))
    # A plain Python str (defensive: helpers must not assume a pypdf object).
    assert resolve("just a string") == "just a string"


def test_metadata_as_jsonable_dict_backstops_non_native_text_fields():
    """Defense-in-depth for ``_metadata_as_jsonable_dict``: even if a raw
    ``ByteStringObject`` ever reaches ``PDFMetadata`` (e.g. via a code path
    that bypasses ``_extract_metadata``), the helper must still hand the cache
    writer a JSON-native dict. (An ``IndirectObject`` cannot be exercised here
    because dereferencing it requires an open reader stream, which the helper
    -- called after ``scrap()``'s ``with open(...)`` block has closed -- does
    not have; that shape is covered end-to-end by the ``indirect_pdf`` tests.)"""
    from crawl4ai.processors.pdf import _metadata_as_jsonable_dict
    from crawl4ai.processors.pdf.processor import PDFMetadata
    from pypdf.generic import ByteStringObject, TextStringObject

    meta = PDFMetadata(
        title=ByteStringObject(b"\x7fAB"),
        author=TextStringObject("an author"),
        producer=ByteStringObject(b"\x7fGH"),
        created=None,
        modified=None,
        pages=1,
        encrypted=False,
        file_size=42,
    )
    d = _metadata_as_jsonable_dict(meta)
    _assert_metadata_is_json_native(d)
    assert isinstance(d["title"], str)
    assert isinstance(d["author"], str)
    assert isinstance(d["producer"], str)
    assert d["author"] == "an author"


@pytest.mark.asyncio
async def test_e2e_bytestring_pdf_writes_cache_row(tmp_path, bytestring_pdf):
    """End-to-end (shape A): a PDF with ``ByteStringObject`` /Info values,
    crawled on the default ``CacheMode.ENABLED`` path, must WRITE the cache
    row. Pre-fix ``acache_url``'s ``json.dumps(result.metadata)`` raised
    ``TypeError: Object of type ByteStringObject is not JSON serializable``
    and the crawl reported ``success=True`` with no row written."""
    await _assert_pdfcrawl_writes_cache_row(
        bytestring_pdf,
        tmp_path,
        expected_fields={"title": str(b"\x7fAB".decode("latin-1"))},
    )


@pytest.mark.asyncio
async def test_e2e_indirect_pdf_writes_cache_row(tmp_path, indirect_pdf):
    """End-to-end (shape B): a PDF with indirect-reference /Info values,
    crawled on the default ``CacheMode.ENABLED`` path, must WRITE the cache
    row. Pre-fix ``acache_url``'s ``json.dumps(result.metadata)`` raised
    ``TypeError: Object of type IndirectObject is not JSON serializable`` (or,
    with a naive ``str()``-only fix applied after the reader closed,
    ``ValueError: seek of closed file``) and the crawl reported
    ``success=True`` with no row written. Asserting all three text fields
    round-trip proves the indirect references were dereferenced *while the
    reader stream was open* (inside ``_extract_metadata``), not later."""
    await _assert_pdfcrawl_writes_cache_row(
        indirect_pdf,
        tmp_path,
        expected_fields={
            "title": "Indirect Title",
            "author": "Indirect Author",
            "producer": "Indirect Producer",
        },
    )

