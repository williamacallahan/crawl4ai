"""Cache must not silently return empty content when a content file is missing.

When the on-disk content files referenced by a cache row's stored hash cannot
be read (the file was deleted, moved, made unreadable, or the read otherwise
fails) but the SQLite ``crawled_data`` row survives, ``aget_cached_url`` used
to substitute ``""`` for the missing field via ``row_dict[field] = content or
""`` and still return a populated ``CrawlResult``.  Because ``arun`` only
falls through to a fresh crawl when ``html`` is empty, a row whose ``html``
loaded fine but whose ``markdown`` / ``extracted_content`` / ``screenshot`` /
``cleaned_html`` file did not was served as a cache ``hit`` with silently empty
content fields, no fresh fetch, no exception, and only a verbose-only log
line.

The fix (see ``crawl4ai/async_database.py``):

1. ``_load_content`` narrows its ``except:`` to ``(OSError, UnicodeDecodeError)``
   so cancellation (``asyncio.CancelledError`` / ``KeyboardInterrupt`` /
   ``SystemExit``) and other unexpected exceptions are no longer swallowed.
2. ``aget_cached_url`` distinguishes ''no stored hash'' (a legitimately-empty
   field) from ''hash present but the file is unreadable'' (a corrupted cache
   row).  The latter now returns ``None`` so ``arun`` falls through to a fresh
   crawl and re-persists a healthy row, rather than returning a silent-empty
   cache hit.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_silent_content_loss_cache_hit.py
"""
import asyncio
import os
import uuid

import aiofiles
import pytest

from crawl4ai.async_database import AsyncDatabaseManager, generate_content_hash
from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import (
    AsyncCrawlResponse,
    CrawlResult,
    MarkdownGenerationResult,
    StringCompatibleMarkdown,
)
from crawl4ai.utils import ensure_content_dirs


URL = "https://example.com"
# Cached HTML / markdown / extraction / screenshot / cleaned_html as seeded
# by the real ``acache_url`` path.  The cached HTML is never run through
# ``is_blocked`` (the cache-hit branch returns it directly), so it can be
# small.  LIVE HTML is large enough (>100 bytes, real visible text) to pass
# the anti-bot ``is_blocked`` check on the fresh-fetch path so the regression
# under test is not obscured by an anti-bot failure.
CACHED_HTML = (
    "<html><body><h1>Cached page</h1>"
    "<p>Cached content here for the regression test.</p></body></html>"
)
CACHED_CLEANED_HTML = (
    "<h1>Cached page</h1><p>Cached content here for the regression test.</p>"
)
CACHED_MD = MarkdownGenerationResult(
    raw_markdown="# Cached page\n\nCached content here for the regression test.",
    markdown_with_citations="",
    references_markdown="",
)
CACHED_EXTRACTION = '[{"cached_extraction": true}]'
CACHED_SCREENSHOT_B64 = "CACHED_SCREENSHOT_BASE64_PNG"

LIVE_HTML = (
    "<html><head><title>Example Article</title></head><body><article>"
    "<h1>Live fetch completed successfully</h1>"
    "<p>This is the freshly fetched page content returned by the crawler "
    "strategy after the cached entry was found to be missing its on-disk "
    "content file and the cache hit was invalidated so a fresh fetch could "
    "self-heal the row.</p></article></body></html>"
)


class _RecordingLogger:
    """Stub logger that records events for assertion, mirroring sibling tests."""

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
    """Minimal ``AsyncCrawlerStrategy`` stub that records ``crawl`` calls.

    ``crawl_calls`` is the canonical "did a fresh fetch happen?" assertion
    target across the regression suite.
    """

    def __init__(self, response=None):
        self.response = response
        self.crawl_calls = 0

    async def crawl(self, url, config):
        self.crawl_calls += 1
        return self.response

    def update_user_agent(self, *args, **kwargs):
        pass


def _make_crawler(tmp_path, response):
    """Build an ``AsyncWebCrawler`` wired for a real ``aprocess_html`` run.

    ``aprocess_html`` is intentionally left un-mocked so the real markdown
    generator runs on the freshly fetched HTML and the test can assert that
    the returned markdown is the *fresh* content (not silently empty).
    """
    logger = _RecordingLogger()
    crawler = AsyncWebCrawler(
        crawler_strategy=_ResponseStrategy(response=response),
        base_directory=str(tmp_path),
        logger=logger,
    )
    crawler.ready = True
    return crawler, logger


async def _isolate_db(monkeypatch, tmp_path):
    """Redirect the ``async_db_manager`` singleton to an isolated temp DB.

    Mirrors the helper in ``test_empty_html_real_pipeline_fallthrough.py`` so
    the E2E ``arun`` tests exercise the real ``aget_cached_url`` /
    ``acache_url`` round-trip against a throwaway SQLite DB + content store,
    leaving the shared ``~/.crawl4ai`` DB untouched.
    """
    db_path = str(tmp_path / "test_crawl4ai.db")
    content_base = str(tmp_path / "content")
    monkeypatch.setattr(async_db_manager, "db_path", db_path)
    monkeypatch.setattr(async_db_manager, "content_paths", ensure_content_dirs(content_base))
    monkeypatch.setattr(async_db_manager, "_initialized", True)
    monkeypatch.setattr(async_db_manager, "connection_pool", {})
    await async_db_manager.ainit_db()
    await async_db_manager.update_db_schema()


async def _setup_manager(tmp_path):
    """Build a fully-initialized, isolated ``AsyncDatabaseManager`` for unit
    tests that don't go through ``arun``.

    Mirrors ``_setup_manager`` in ``test_async_database_markdown_roundtrip.py``.
    """
    mgr = AsyncDatabaseManager()
    mgr.db_path = str(tmp_path / "test.db")
    mgr.content_paths = ensure_content_dirs(str(tmp_path))
    await mgr.ainit_db()
    await mgr.update_db_schema()
    mgr._initialized = True
    return mgr


def _seed_result(
    url,
    *,
    html=CACHED_HTML,
    cleaned_html=CACHED_CLEANED_HTML,
    markdown=CACHED_MD,
    extracted_content=CACHED_EXTRACTION,
    screenshot=CACHED_SCREENSHOT_B64,
):
    """Build a fully-populated ``CrawlResult`` for seeding the cache via
    ``acache_url`` (every content field non-empty so every content hash is
    stored)."""
    return CrawlResult(
        url=url,
        html=html,
        cleaned_html=cleaned_html,
        success=True,
        markdown=markdown,
        extracted_content=extracted_content,
        screenshot=screenshot,
    )


def _content_file_path(mgr, content_type, stored_content):
    """Path of the on-disk content file ``acache_url`` writes for
    ``stored_content`` keyed by ``generate_content_hash(content)``."""
    return os.path.join(mgr.content_paths[content_type], generate_content_hash(stored_content))


async def _seed_cache_via_acache_url(url):
    """Seed a cache row via the real ``acache_url`` path (the singleton)."""
    await async_db_manager.acache_url(_seed_result(url))
    cached = await async_db_manager.aget_cached_url(url)
    assert cached is not None, "seed must produce a cache hit"
    assert cached.html == CACHED_HTML
    assert cached.markdown.raw_markdown == CACHED_MD.raw_markdown
    return cached


# ---------------------------------------------------------------------------
# Layer 1: direct aget_cached_url against an isolated AsyncDatabaseManager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_cached_url_returns_full_result_when_all_files_present(tmp_path):
    """No regression: when every content file is readable, ``aget_cached_url``
    returns the full ``CrawlResult`` reconstructed from disk."""
    mgr = await _setup_manager(tmp_path)
    url = f"{URL}?unit_all_present={uuid.uuid4().hex}"
    await mgr.acache_url(_seed_result(url))

    cached = await mgr.aget_cached_url(url)
    assert cached is not None
    assert cached.html == CACHED_HTML
    assert cached.cleaned_html == CACHED_CLEANED_HTML
    assert isinstance(cached.markdown, StringCompatibleMarkdown)
    assert cached.markdown.raw_markdown == CACHED_MD.raw_markdown
    assert cached.extracted_content == CACHED_EXTRACTION
    assert cached.screenshot == CACHED_SCREENSHOT_B64


@pytest.mark.asyncio
async def test_aget_cached_url_no_false_miss_when_optional_fields_empty(tmp_path):
    """No false cache miss: a row whose optional content fields were stored
    with empty hashes (e.g. ``cleaned_html=""``, ``extracted_content=""``,
    ``screenshot=""``) must round-trip as a hit, NOT be mistaken for a
    corrupted row.  Only a *non-empty* hash whose file is unreadable is a
    corrupted row."""
    mgr = await _setup_manager(tmp_path)
    url = f"{URL}?unit_empty_optional={uuid.uuid4().hex}"
    await mgr.acache_url(
        CrawlResult(
            url=url,
            html=CACHED_HTML,
            success=True,
            markdown=CACHED_MD,
            cleaned_html=None,
            extracted_content=None,
            screenshot=None,
        )
    )

    cached = await mgr.aget_cached_url(url)
    assert cached is not None, "empty optional hashes are NOT corruption"
    assert cached.html == CACHED_HTML
    assert cached.cleaned_html == ""
    assert cached.markdown.raw_markdown == CACHED_MD.raw_markdown
    assert cached.extracted_content == ""
    assert cached.screenshot == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type,stored_content,field",
    [
        # Markdown covers the canonical repro from the bug report (html file
        # intact, markdown file missing).
        ("markdown", CACHED_MD.model_dump_json(), "markdown"),
        # Screenshot covers a distinct content_type key (``screenshot`` /
        # ``screenshots`` map to ``screenshots/`` while the field name in the
        # load loop is ``screenshot``) — guards against a regression that
        # only fixes the markdown branch.
        ("screenshot", CACHED_SCREENSHOT_B64, "screenshot"),
    ],
    ids=["markdown", "screenshot"],
)
async def test_aget_cached_url_returns_none_when_a_content_file_is_missing(
    tmp_path, content_type, stored_content, field
):
    """THE regression: when an on-disk content file is missing while the DB
    row still references its hash, ``aget_cached_url`` must return ``None``
    (cache miss), NOT a silently-empty ``CrawlResult``.

    Before the fix, ``_load_content`` returned ``None`` (indistinguishable
    from ''no hash stored''), the load loop normalized it to ``""`` via
    ``content or ""``, and a populated ``CrawlResult`` was returned whose
    other fields (e.g. ``html``) were intact — letting ``arun`` treat the row
    as a cache hit with the missing field silently empty."""
    mgr = await _setup_manager(tmp_path)
    url = f"{URL}?unit_missing_{field}={uuid.uuid4().hex}"
    await mgr.acache_url(_seed_result(url))

    cached_before = await mgr.aget_cached_url(url)
    assert cached_before is not None
    file_path = _content_file_path(mgr, content_type, stored_content)
    assert os.path.exists(file_path), "seed must have written the content file"

    # Delete the file (simulating manual cache-dir cleanup / partial restore).
    os.remove(file_path)
    assert not os.path.exists(file_path)

    # THE regression assertion: a corrupted cache row is a cache miss.
    cached_after = await mgr.aget_cached_url(url)
    assert cached_after is None, (
        f"aget_cached_url must return None when the {field} content file is "
        f"missing but its hash is still stored in the DB — silently returning "
        f"a partial CrawlResult (with {field} emptied) is the bug"
    )


# ---------------------------------------------------------------------------
# Layer 2: _load_content narrow catch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_content_returns_none_for_empty_hash(tmp_path):
    """An empty content hash means ''no content was stored''; ``_load_content``
    short-circuits and returns ``None`` on the early guard."""
    mgr = await _setup_manager(tmp_path)
    assert await mgr._load_content("", "markdown") is None


@pytest.mark.asyncio
async def test_load_content_returns_none_for_missing_file(tmp_path):
    """A missing file (``FileNotFoundError``, an ``OSError`` subclass) is one
    of the two expected failure modes — must return ``None``, not raise."""
    mgr = await _setup_manager(tmp_path)
    assert await mgr._load_content("nonexistent_hash", "markdown") is None


@pytest.mark.asyncio
async def test_load_content_returns_none_for_undecodable_file(tmp_path):
    """A corrupted/truncated UTF-8 file (``UnicodeDecodeError``, the second
    expected failure mode) must return ``None``, not raise."""
    mgr = await _setup_manager(tmp_path)
    hash_value = "deadbeef_undecodable"
    file_path = os.path.join(mgr.content_paths["markdown"], hash_value)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(b"\xff\xfe\xfd\xfb")  # invalid UTF-8 sequences

    assert await mgr._load_content(hash_value, "markdown") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [asyncio.CancelledError(), ValueError("totally unrelated error")],
    ids=["CancelledError", "ValueError"],
)
async def test_load_content_does_not_swallow_unexpected_exceptions(
    tmp_path, monkeypatch, exc
):
    """The narrowed ``except (OSError, UnicodeDecodeError)`` must NOT swallow
    ``BaseException`` subclasses (``CancelledError`` masks cancellation — the
    documented code-quality concern) or any other unexpected exception.  The
    bare ``except:`` that existed before the fix caught all of these, hiding
    programming errors as silent cache misses."""

    def _raise(*args, **kwargs):
        raise exc

    # ``_load_content`` resolves ``aiofiles.open`` at call time, so
    # monkeypatching the module attribute is sufficient.  Set on the module
    # imported by ``async_database`` directly (no shadowing local import).
    import crawl4ai.async_database as db_module

    monkeypatch.setattr(db_module.aiofiles, "open", _raise)

    mgr = await _setup_manager(tmp_path)
    with pytest.raises(type(exc)):
        await mgr._load_content("any_hash", "markdown")


# ---------------------------------------------------------------------------
# Layer 3: end-to-end via arun (real aget_cached_url / acache_url)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_markdown_file_missing_triggers_recrawl_not_silent_hit(tmp_path, monkeypatch):
    """THE end-to-end repro: deleting the markdown content file then calling
    ``arun`` must trigger a fresh fetch (``cache_status == "miss"``,
    ``crawl_calls == 1``) and return the freshly generated markdown — not a
    cache ``hit`` with ``markdown.raw_markdown == ""`` and no fresh fetch.

    On the unpatched code this observes ``cache_status == "hit"``,
    ``crawl_calls == 0``, and ``markdown.raw_markdown == ""``.
    """
    await _isolate_db(monkeypatch, tmp_path)
    url = f"{URL}?e2e_md_missing={uuid.uuid4().hex}"
    await _seed_cache_via_acache_url(url)

    md_file = _content_file_path(async_db_manager, "markdown", CACHED_MD.model_dump_json())
    assert os.path.exists(md_file)
    os.remove(md_file)

    crawler, _ = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML, response_headers={}, status_code=200
        ),
    )

    result = await crawler.arun(url, CrawlerRunConfig(cache_mode=CacheMode.ENABLED))

    assert result.cache_status == "miss", (
        "a cache row whose markdown content file is missing must NOT be "
        "served as a cache hit"
    )
    assert crawler.crawler_strategy.crawl_calls == 1, (
        "a cache row whose markdown content file is missing must trigger a "
        "fresh fetch"
    )
    assert result.markdown is not None
    assert result.markdown.raw_markdown != "", (
        "the returned markdown must be the freshly generated content, not "
        "silently empty"
    )
    assert "Live fetch completed successfully" in result.markdown.raw_markdown
    assert result.html == LIVE_HTML


@pytest.mark.asyncio
async def test_corrupted_row_self_heals_after_recrawl(tmp_path, monkeypatch):
    """After a fresh fetch re-persists the row, the corrupted content file is
    recreated from the fresh result and a subsequent ``arun`` returns a cache
    hit with the fresh content — full self-heal, no manual cleanup required.

    On the unpatched code the first ``arun`` returns a silent cache hit and
    never re-fetches, so the DB row is never overwritten and the second
    ``arun`` keeps returning the silently-empty markdown.
    """
    await _isolate_db(monkeypatch, tmp_path)
    url = f"{URL}?e2e_self_heal={uuid.uuid4().hex}"
    await _seed_cache_via_acache_url(url)

    md_file = _content_file_path(async_db_manager, "markdown", CACHED_MD.model_dump_json())
    os.remove(md_file)

    crawler, _ = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML, response_headers={}, status_code=200
        ),
    )

    # First arun: cache miss → fresh fetch → re-persist (write guard fires
    # because cached_result is None under the fix).
    r1 = await crawler.arun(url, CrawlerRunConfig(cache_mode=CacheMode.ENABLED))
    assert r1.cache_status == "miss"
    assert crawler.crawler_strategy.crawl_calls == 1
    assert r1.html == LIVE_HTML
    assert "Live fetch completed successfully" in r1.markdown.raw_markdown

    fresh_md_payload = (
        r1.get_markdown_generation_result().model_dump_json()
        if r1.get_markdown_generation_result() is not None
        else ""
    )
    fresh_md_file = _content_file_path(async_db_manager, "markdown", fresh_md_payload)
    assert os.path.exists(fresh_md_file), (
        "the fresh arun must re-persist the markdown content file"
    )

    # Second arun: cache HIT — no fresh fetch, fresh content served from the
    # self-healed row.
    r2 = await crawler.arun(url, CrawlerRunConfig(cache_mode=CacheMode.ENABLED))
    assert r2.cache_status == "hit"
    assert crawler.crawler_strategy.crawl_calls == 1, "second arun is a cache hit"
    assert r2.html == LIVE_HTML
    assert "Live fetch completed successfully" in r2.markdown.raw_markdown
    assert r2.markdown.raw_markdown != ""


@pytest.mark.asyncio
async def test_all_files_present_returns_cache_hit_without_refetch(tmp_path, monkeypatch):
    """No regression: on the happy path (every content file present),
    ``arun`` returns ``cache_status == "hit"`` with ``crawl_calls == 0`` and
    the cached content verbatim.  Confirms the fix does not accidentally
    invalidate healthy cache rows."""
    await _isolate_db(monkeypatch, tmp_path)
    url = f"{URL}?e2e_happy={uuid.uuid4().hex}"
    await _seed_cache_via_acache_url(url)

    crawler, _ = _make_crawler(
        tmp_path,
        response=AsyncCrawlResponse(
            html=LIVE_HTML, response_headers={}, status_code=200
        ),
    )

    result = await crawler.arun(url, CrawlerRunConfig(cache_mode=CacheMode.ENABLED))

    assert result.cache_status == "hit"
    assert crawler.crawler_strategy.crawl_calls == 0
    assert result.html == CACHED_HTML
    assert result.markdown.raw_markdown == CACHED_MD.raw_markdown
    assert result.extracted_content == CACHED_EXTRACTION
    assert result.screenshot == CACHED_SCREENSHOT_B64
