"""AsyncDatabaseManager must round-trip the full MarkdownGenerationResult.

Regression coverage for the cache markdown round-trip bug introduced in
commit a9e2430 ("Release prep (#749)"). The `markdown` property always returns a
`StringCompatibleMarkdown` wrapper (a `str` subclass) whose string value is only
`raw_markdown`, so the old `acache_url` write branch that tested
`isinstance(result.markdown, StringCompatibleMarkdown)` first always won and
persisted only `raw_markdown`, silently dropping `markdown_with_citations`,
`references_markdown`, `fit_markdown` and `fit_html`. On read,
`aget_cached_url` ran `json.loads` on the stored value; when the cached
`raw_markdown` happened to be JSON-parseable, the dict-downgrade either crashed
the `markdown` property (`AttributeError`) or raised a `ValidationError` that
the outer try/except swallowed into a silent cache miss.

These tests exercise the real `AsyncDatabaseManager` against an isolated
SQLite DB + filesystem content store under `tmp_path` (no mocks of the DB
layer).

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_async_database_markdown_roundtrip.py
"""

import json
import os

import aiofiles
import pytest

from crawl4ai.async_database import AsyncDatabaseManager, generate_content_hash
from crawl4ai.models import (
    CrawlResult,
    MarkdownGenerationResult,
    StringCompatibleMarkdown,
)
from crawl4ai.utils import ensure_content_dirs


async def _setup_manager(tmp_path) -> AsyncDatabaseManager:
    """Build a fully-initialized, isolated AsyncDatabaseManager under tmp_path."""
    mgr = AsyncDatabaseManager()
    mgr.db_path = str(tmp_path / "test.db")
    mgr.content_paths = ensure_content_dirs(str(tmp_path))
    await mgr.ainit_db()
    await mgr.update_db_schema()
    mgr._initialized = True
    return mgr


def _md(raw, citations="", refs="", fit=None, fithtml=None) -> MarkdownGenerationResult:
    return MarkdownGenerationResult(
        raw_markdown=raw,
        markdown_with_citations=citations,
        references_markdown=refs,
        fit_markdown=fit,
        fit_html=fithtml,
    )


def _result(url, md) -> CrawlResult:
    return CrawlResult(url=url, html="<h1>x</h1>", success=True, markdown=md)


# ---------------------------------------------------------------------------
# Core round-trip: enriched sub-fields survive a cache write + read.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_markdown_roundtrips_through_cache(tmp_path):
    """All enriched MarkdownGenerationResult sub-fields survive the round-trip."""
    mgr = await _setup_manager(tmp_path)
    md = _md(
        raw="# Heading\n\nBody text.",
        citations="# Heading\n\nBody text. \u20064\u2007",
        refs="## References\n[1] https://example.com",
        fit="# Fit\nbody.",
        fithtml="<article>fit</article>",
    )
    await mgr.acache_url(_result("https://e.com/full", md))

    cached = await mgr.aget_cached_url("https://e.com/full")
    assert cached is not None
    assert isinstance(cached.markdown, StringCompatibleMarkdown)
    # String-compatible behaviour preserved.
    assert str(cached.markdown) == md.raw_markdown
    assert isinstance(cached.markdown, str)
    # Full-fidelity sub-fields preserved (all five).
    assert cached.markdown.raw_markdown == md.raw_markdown
    assert cached.markdown.markdown_with_citations == md.markdown_with_citations
    assert cached.markdown.references_markdown == md.references_markdown
    assert cached.markdown.fit_markdown == md.fit_markdown
    assert cached.markdown.fit_html == md.fit_html


@pytest.mark.asyncio
async def test_model_dump_of_cached_result_contains_full_markdown(tmp_path):
    """Downstream `model_dump()` serialises the full markdown dict from cache."""
    mgr = await _setup_manager(tmp_path)
    md = _md(raw="raw", citations="cit", refs="ref", fit="fit", fithtml="<a>")
    await mgr.acache_url(_result("https://e.com/dump", md))

    cached = await mgr.aget_cached_url("https://e.com/dump")
    dumped = cached.model_dump()["markdown"]
    assert isinstance(dumped, dict)
    assert dumped["raw_markdown"] == "raw"
    assert dumped["markdown_with_citations"] == "cit"
    assert dumped["references_markdown"] == "ref"
    assert dumped["fit_markdown"] == "fit"
    assert dumped["fit_html"] == "<a>"


# ---------------------------------------------------------------------------
# JSON-parseable raw_markdown: previously crashed or silently missed.
# After the fix, the full JSON wrapper is stored, so the page's raw_markdown is
# preserved verbatim and `.markdown` access never raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_markdown",
    [
        '"quoted JSON string"',
        "42",
        "3.14",
        "true",
        "null",
        '["a","list"]',
        json.dumps({"raw_markdown": "hello", "fit_markdown": "y"}),
        json.dumps({"fit_markdown": "x"}),
        json.dumps({"foo": "bar"}),
    ],
)
async def test_json_parseable_raw_markdown_roundtrips_without_crash(tmp_path, raw_markdown):
    """A new entry whose raw_markdown is JSON-parseable round-trips cleanly.

    Before the fix these either crashed the `markdown` property (parsed scalar
    / array / dict-with-raw_markdown became a bare str/int/list in _markdown)
    or silently returned None (dict without raw_markdown raised ValidationError,
    swallowed by the outer try/except).
    """
    mgr = await _setup_manager(tmp_path)
    url = f"https://e.com/json/{abs(hash(raw_markdown))}"
    await mgr.acache_url(_result(url, _md(raw=raw_markdown)))

    cached = await mgr.aget_cached_url(url)
    assert cached is not None, "cached entry must not silently become a miss"
    # `.markdown` access must not raise for any JSON-parseable raw_markdown.
    assert cached.markdown is not None
    assert cached.markdown.raw_markdown == raw_markdown
    # Enriched sub-fields were empty at write time, stay empty at read time.
    assert cached.markdown.markdown_with_citations == ""
    assert cached.markdown.references_markdown == ""
    # Backward-compatible string form preserved.
    assert str(cached.markdown) == raw_markdown


# ---------------------------------------------------------------------------
# None / bare-string markdown: previously caused silent total cache-write
# failure (MarkdownGenerationResult() raised, the whole write was dropped).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_markdown_caches_without_error(tmp_path):
    """Caching a CrawlResult with no markdown no longer silently drops the row."""
    mgr = await _setup_manager(tmp_path)
    result = CrawlResult(url="https://e.com/none", html="<h1>x</h1>", success=True)
    assert result.markdown is None

    await mgr.acache_url(result)

    cached = await mgr.aget_cached_url("https://e.com/none")
    assert cached is not None, "row must be persisted even when markdown is None"
    # The round-trip produces an empty (non-None) MarkdownGenerationResult, which
    # is safe to access and never crashes the property.
    assert cached.markdown is not None
    assert cached.markdown.raw_markdown == ""


@pytest.mark.asyncio
async def test_bare_string_markdown_roundtrips(tmp_path):
    """A bare-string `_markdown` (defensive branch) is wrapped and round-trips."""
    mgr = await _setup_manager(tmp_path)
    # Constructing with markdown="<str>" leaves _markdown as a bare str, which
    # the property cannot wrap; acache_url must handle it without accessing the
    # property.
    result = CrawlResult(
        url="https://e.com/bare", html="<h1>x</h1>", success=True, markdown="# bare"
    )
    await mgr.acache_url(result)

    cached = await mgr.aget_cached_url("https://e.com/bare")
    assert cached is not None
    assert cached.markdown.raw_markdown == "# bare"
    assert cached.markdown.markdown_with_citations == ""
    assert cached.markdown.references_markdown == ""


# ---------------------------------------------------------------------------
# Overwrite / UPSERT path: the second write replaces the first entirely.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overwrite_replaces_markdown(tmp_path):
    mgr = await _setup_manager(tmp_path)
    url = "https://e.com/overwrite"
    await mgr.acache_url(_result(url, _md(raw="first", citations="c1")))
    await mgr.acache_url(_result(url, _md(raw="second", citations="c2", fit="f2")))

    cached = await mgr.aget_cached_url(url)
    assert cached is not None
    assert cached.markdown.raw_markdown == "second"
    assert cached.markdown.markdown_with_citations == "c2"
    assert cached.markdown.fit_markdown == "f2"
    # First-write citation must not leak through.
    assert cached.markdown.raw_markdown != "first"


# ---------------------------------------------------------------------------
# Legacy entries (created before the fix). These are written by hand directly
# to the content file + DB to simulate pre-existing caches.
# ---------------------------------------------------------------------------


async def _insert_legacy_entry(mgr, url, stored_content):
    """Write a raw markdown content blob (as if from an older code path)."""
    content_hash = generate_content_hash(stored_content)
    md_path = os.path.join(mgr.content_paths["markdown"], content_hash)
    async with aiofiles.open(md_path, "w", encoding="utf-8") as f:
        await f.write(stored_content)
    async with mgr.get_connection() as db:
        await db.execute(
            "INSERT INTO crawled_data (url, html, cleaned_html, markdown, "
            "extracted_content, screenshot, success, downloaded_files) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (url, "", "", content_hash, "", "", 1, "[]"),
        )
        await db.commit()


@pytest.mark.asyncio
async def test_legacy_full_json_entry_preserved(tmp_path):
    """A pre-fix full MarkdownGenerationResult JSON blob round-trips intact.

    Before the fix, the dict-with-raw_markdown was downgraded to a bare string
    and the `markdown` property raised AttributeError.
    """
    mgr = await _setup_manager(tmp_path)
    legacy = _md(
        raw="# Legacy",
        citations="# Legacy \u20064\u2007",
        refs="[1] https://legacy.example.com",
        fit="# Fit",
        fithtml="<a>fit</a>",
    )
    await _insert_legacy_entry(mgr, "https://e.com/legacy", legacy.model_dump_json())

    cached = await mgr.aget_cached_url("https://e.com/legacy")
    assert cached is not None
    assert cached.markdown.raw_markdown == "# Legacy"
    assert cached.markdown.markdown_with_citations == "# Legacy \u20064\u2007"
    assert cached.markdown.references_markdown == "[1] https://legacy.example.com"
    assert cached.markdown.fit_markdown == "# Fit"
    assert cached.markdown.fit_html == "<a>fit</a>"


@pytest.mark.asyncio
async def test_legacy_plain_text_raw_markdown_preserved(tmp_path):
    """A legacy lossy entry storing plain-text raw_markdown is recovered as-is."""
    mgr = await _setup_manager(tmp_path)
    stored = "# Just plain text markdown\n\nNo JSON here."
    await _insert_legacy_entry(mgr, "https://e.com/plain", stored)

    cached = await mgr.aget_cached_url("https://e.com/plain")
    assert cached is not None
    assert cached.markdown.raw_markdown == stored
    assert cached.markdown.markdown_with_citations == ""
    assert cached.markdown.references_markdown == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [
        '"quoted JSON string"',
        "42",
        "true",
        "null",
        '["a","list"]',
        json.dumps({"raw_markdown": "hello", "fit_markdown": "y"}),
        json.dumps({"fit_markdown": "x"}),
        json.dumps({"only": "fields"}),
    ],
)
async def test_legacy_json_parseable_raw_markdown_preserved(tmp_path, stored):
    """A legacy lossy entry whose raw_markdown happens to be JSON-parseable.

    Before the fix these crashed (parsed scalar/array/dict-with-raw_markdown
    became a bare value in _markdown) or silently returned None (dict without
    raw_markdown raised ValidationError swallowed by the outer try/except).
    They must instead preserve the original string as raw_markdown.
    """
    mgr = await _setup_manager(tmp_path)
    await _insert_legacy_entry(mgr, f"https://e.com/legacy/{abs(hash(stored))}", stored)

    cached = await mgr.aget_cached_url(f"https://e.com/legacy/{abs(hash(stored))}")
    assert cached is not None, "valid cached entry must not become a silent miss"
    assert cached.markdown is not None
    assert cached.markdown.raw_markdown == stored
    assert str(cached.markdown) == stored


@pytest.mark.asyncio
async def test_legacy_empty_markdown_preserved(tmp_path):
    """An empty stored markdown round-trips to an empty MarkdownGenerationResult."""
    mgr = await _setup_manager(tmp_path)
    await _insert_legacy_entry(mgr, "https://e.com/empty", "")

    cached = await mgr.aget_cached_url("https://e.com/empty")
    assert cached is not None
    assert cached.markdown.raw_markdown == ""


# ---------------------------------------------------------------------------
# Backward-compatibility: `str(result.markdown)` (= raw_markdown) is unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_string_form_backward_compat(tmp_path):
    """The common `str(result.markdown)` call still returns raw_markdown."""
    mgr = await _setup_manager(tmp_path)
    md = _md(raw="raw text", citations="cited", refs="refs", fit="fit", fithtml="<x>")
    await mgr.acache_url(_result("https://e.com/str", md))

    cached = await mgr.aget_cached_url("https://e.com/str")
    assert str(cached.markdown) == "raw text"
    assert cached.markdown == "raw text"  # str equality via StringCompatibleMarkdown


@pytest.mark.asyncio
async def test_cached_result_other_fields_preserved(tmp_path):
    """Non-markdown fields are unaffected by the markdown round-trip fix."""
    mgr = await _setup_manager(tmp_path)
    result = CrawlResult(
        url="https://e.com/other",
        html="<h1>title</h1>",
        success=True,
        markdown=_md(raw="md"),
        media={"images": [{"src": "http://x/y.png"}]},
        links={"internal": [{"href": "/a", "text": "a"}]},
        metadata={"title": "T"},
        response_headers={"etag": "abc"},
    )
    await mgr.acache_url(result)

    cached = await mgr.aget_cached_url("https://e.com/other")
    assert cached is not None
    assert cached.html == "<h1>title</h1>"
    assert cached.url == "https://e.com/other"
    assert cached.success is True
    assert cached.media["images"][0]["src"] == "http://x/y.png"
    assert cached.links["internal"][0]["href"] == "/a"
    assert cached.metadata["title"] == "T"
    assert cached.response_headers["etag"] == "abc"
    assert cached.markdown.raw_markdown == "md"
