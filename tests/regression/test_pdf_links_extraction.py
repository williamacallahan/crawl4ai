"""Regression test for the silent PDF-link-drop bug.

Bug: ``PDFContentScrapingStrategy.scrap`` (commit f8fd9d9e) accumulated the
URIs ``NaivePDFProcessorStrategy._extract_links`` returns under an
undocumented ``links["urls"]`` key in a plain dict, but
``ScrapingResult.links`` is the pydantic ``Links`` model whose only fields
are ``internal`` and ``external``. Under pydantic v2's default
``extra='ignore'`` the entire ``"urls"`` payload is silently discarded at
construction, so every PDF crawl reports
``links == {"internal": [], "external": []}`` and deep crawling from PDF
seeds discovers nothing. The docs (``docs/md_v2/advanced/pdf-parsing.md``)
meanwhile told users to read ``links["urls"]``, which raised ``KeyError``
on a ``CrawlResult`` (whose ``links`` is ``Dict[str, List]``) and was not
even subscriptable on a ``ScrapingResult`` (whose ``links`` is a ``Links``
model).

Fix:
1. ``crawl4ai/processors/pdf/__init__.py`` now builds the standard
   ``{"internal": [...], "external": [...]}`` shape the rest of the
   codebase consumes (``LXMLWebScrapingStrategy``, the deep crawlers, the
   adaptive crawler) using ``is_external_url`` against the PDF source
   URL's base domain. Each entry uses ``href`` (the field ``Link``
   declares -- ``url`` would also be dropped) and ``page`` (the 1-based
   PDF page number it was extracted from).
2. ``crawl4ai/models.py`` gains ``page: Optional[int] = None`` on ``Link``
   so per-page provenance survives ``Links`` validation instead of being
   dropped by ``extra='ignore'``. The field is optional and defaults to
   ``None``, so existing HTML link producers are unaffected.
3. ``docs/md_v2/advanced/pdf-parsing.md`` now documents the actual
   ``internal``/``external``/``href``/``page`` shape.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_pdf_links_extraction.py
"""

import json
from pathlib import Path
from typing import List

import pytest

pypdf = pytest.importorskip("pypdf", reason="requires the crawl4ai[pdf] extra")

from pypdf import PdfWriter
from pypdf.annotations import Link as PypdfLink
from pypdf.generic import RectangleObject

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.models import Link, Links, ScrapingResult
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _build_pdf(path: Path, per_page_links: List[List[str]]) -> Path:
    """Write a PDF with the given URI link annotations per page.

    ``per_page_links[i]`` is a list of absolute URI strings to attach as
    ``/Annots`` link annotations on page ``i+1``. Pages beyond the supplied
    list are skipped, so passing ``[["https://a"], ["https://b"]]`` produces
    a 2-page PDF with one link on each page.
    """
    writer = PdfWriter()
    for page_links in per_page_links:
        writer.add_blank_page(width=612, height=792)
    for page_idx, page_links in enumerate(per_page_links):
        for uri in page_links:
            annot = PypdfLink(rect=RectangleObject((10, 10, 100, 100)), url=uri)
            writer.add_annotation(page_idx, annot)
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def single_link_pdf(tmp_path):
    return _build_pdf(tmp_path / "single.pdf", [["https://example.com/foo"]])


@pytest.fixture
def no_link_pdf(tmp_path):
    return _build_pdf(tmp_path / "no_link.pdf", [[]])


@pytest.fixture
def multipage_pdf(tmp_path):
    return _build_pdf(
        tmp_path / "multipage.pdf",
        [["https://example.com/page1"], ["https://example.com/page2"]],
    )


@pytest.fixture
def mixed_host_pdf(tmp_path):
    """A single page with three URI links: same-host, subdomain, other-host."""
    return _build_pdf(
        tmp_path / "mixed.pdf",
        [[
            "https://example.com/a",
            "https://sub.example.com/b",
            "https://other.com/c",
        ]],
    )


# ---------------------------------------------------------------------------
# Unit tests for PDFContentScrapingStrategy.scrap — pins G1-G5, G13, G14
# ---------------------------------------------------------------------------

def test_scrap_returns_links_in_internal_external_shape(single_link_pdf):
    """The link survives ``ScrapingResult`` construction and appears under
    ``external`` (file:// source has no host, so absolute http links are
    external). The entry uses ``href``/``page``, not ``url``/``urls``."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    result = strategy.scrap(f"file://{single_link_pdf}", html="")

    assert isinstance(result, ScrapingResult)
    assert result.success
    assert isinstance(result.links, Links)

    dumped = result.links.model_dump()
    assert set(dumped.keys()) == {"internal", "external"}, (
        f"links keys must be exactly internal/external (no 'urls'): {dumped.keys()}"
    )
    assert dumped["internal"] == []
    assert len(dumped["external"]) == 1
    entry = dumped["external"][0]
    assert entry["href"] == "https://example.com/foo"
    assert entry["text"] == ""
    assert entry["page"] == 1
    # The producer must NOT emit a "url" key (Link has no "url" field, so
    # the field name itself would survive only as silently dropped).
    assert "url" not in entry


def test_scrap_classifies_internal_vs_external_by_host(mixed_host_pdf, monkeypatch):
    """Same-host and subdomain links go to internal; different-host links
    go to external, using is_external_url's same convention as the HTML
    scraper. The source URL is ``https://example.com/doc.pdf`` without
    actually downloading it (we point _get_pdf_path at the local fixture
    and neuter _discard_temp_file so the finally clause keeps the file)."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    monkeypatch.setattr(strategy, "_get_pdf_path", lambda url: str(mixed_host_pdf))
    monkeypatch.setattr(strategy, "_discard_temp_file", lambda path: None)

    result = strategy.scrap("https://example.com/doc.pdf", html="")

    dumped = result.links.model_dump()
    internal_hrefs = sorted(e["href"] for e in dumped["internal"])
    external_hrefs = sorted(e["href"] for e in dumped["external"])
    assert internal_hrefs == [
        "https://example.com/a",
        "https://sub.example.com/b",
    ]
    assert external_hrefs == ["https://other.com/c"]
    # Every entry on page 1.
    for entry in dumped["internal"] + dumped["external"]:
        assert entry["page"] == 1


def test_scrap_no_links_yields_empty_internal_external(no_link_pdf):
    """A PDF without /Annots must produce the standard empty shape, never
    None and never a leftover 'urls' key."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    result = strategy.scrap(f"file://{no_link_pdf}", html="")

    dumped = result.links.model_dump()
    assert dumped == {"internal": [], "external": []}
    assert "urls" not in dumped


def test_scrap_preserves_per_page_provenance_for_multipage_pdf(multipage_pdf):
    """A two-page PDF with a link on page 1 and a link on page 2 must
    carry page=1 and page=2 respectively -- the original INTENT of the
    buggy ``{"page": page.page_number}`` entry, now preserved instead of
    being dropped by extra='ignore'."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    result = strategy.scrap(f"file://{multipage_pdf}", html="")

    external = result.links.model_dump()["external"]
    by_page = {e["page"]: e["href"] for e in external}
    assert by_page == {
        1: "https://example.com/page1",
        2: "https://example.com/page2",
    }, f"per-page provenance lost: {by_page}"


def test_scrap_cleaned_html_media_metadata_unchanged(single_link_pdf):
    """Pinning G8: non-link ScrapingResult fields stay correct -- the
    fix touches only the links path. ``cleaned_html`` carries the
    pdf-pages meta, ``media.images`` is empty (extract_images=False),
    and metadata has the [pages] field with value 1."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    result = strategy.scrap(f"file://{single_link_pdf}", html="")

    assert 'pdf-pages' in result.cleaned_html
    assert 'content="1"' in result.cleaned_html
    media = result.media.model_dump() if hasattr(result.media, "model_dump") else result.media
    assert media["images"] == []
    assert result.metadata["pages"] == 1


def test_model_dumped_link_entries_subscriptable_with_get(single_link_pdf):
    """Pinning G7: every link dict that deep crawlers / the adaptive
    crawler will iterate supports ``link.get("href")`` and
    ``link.get("page")`` exactly as ``bfs_strategy`` / ``adaptive_crawler``
    read them."""
    strategy = PDFContentScrapingStrategy(extract_images=False)
    result = strategy.scrap(f"file://{single_link_pdf}", html="")

    all_links = (
        result.links.model_dump()["internal"]
        + result.links.model_dump()["external"]
    )
    assert all_links, "no links to test consumer pattern against"
    for link in all_links:
        assert isinstance(link, dict)
        href = link.get("href")
        page = link.get("page")
        assert isinstance(href, str) and href.startswith(("http://", "https://"))
        assert isinstance(page, int) and page >= 1


# ---------------------------------------------------------------------------
# Link model regression (G9): HTML-path producers must not regress.
# ---------------------------------------------------------------------------

def test_link_model_html_producer_yields_page_none():
    """A web-scraping producer that builds Link from {"href": ..., "text":
    ...} (no "page") must get page=None -- the new field is benign."""
    link = Link(**{"href": "/internal", "text": "go"})
    assert link.href == "/internal"
    assert link.text == "go"
    assert link.page is None

    links = Links(**{"internal": [{"href": "/x"}, {"href": "/y", "text": "y"}],
                    "external": [{"href": "https://ext.com/z"}]})
    dumped = links.model_dump()
    assert all(e["page"] is None for e in dumped["internal"])
    assert all(e["page"] is None for e in dumped["external"])


def test_link_model_pdf_producer_carries_page_int():
    """A producer that DOES supply "page" (the PDF path) preserves it,
    not None -- proves the field addition is what makes provenance
    survive Links validation."""
    links = Links(**{
        "internal": [{"href": "https://example.com/a", "text": "", "page": 5}],
        "external": [{"href": "https://other.com/b", "text": "", "page": 1}],
    })
    dumped = links.model_dump()
    assert dumped["internal"][0]["page"] == 5
    assert dumped["external"][0]["page"] == 1


# ---------------------------------------------------------------------------
# End-to-end via AsyncWebCrawler.arun — pins G6, G7, G12
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_arun_carries_pdf_links_into_crawl_result(single_link_pdf):
    """The real AsyncWebCrawler.arun path must surface PDF links in
    CrawlResult.links (Dict[str, List[Dict]]) under the same shape -- the
    silent drop at ScrapingResult was the bug, but the conversion
    ``result.links.model_dump()`` in async_webcrawler.py also must not
    re-drop them. Cache is bypassed to keep this a pure in-process test."""
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        scraping_strategy=PDFContentScrapingStrategy(extract_images=False),
    )
    async with AsyncWebCrawler(crawler_strategy=PDFCrawlerStrategy()) as crawler:
        result = await crawler.arun(f"file://{single_link_pdf}", config=config)

    assert result.success, f"crawl failed: {result.error_message}"
    assert isinstance(result.links, dict), (
        f"CrawlResult.links must be a dict (Dict[str, List[Dict]]); "
        f"got {type(result.links).__name__}"
    )
    assert set(result.links.keys()) == {"internal", "external"}, (
        f"CrawlResult.links keys must be exactly internal/external (no 'urls'): "
        f"{result.links.keys()}"
    )
    # The link URI from /Annots survived all the way through.
    all_hrefs = [e.get("href") for e in result.links["internal"] + result.links["external"]]
    assert "https://example.com/foo" in all_hrefs
    # Per-page provenance is preserved through the conversion.
    matching = [e for e in result.links["internal"] + result.links["external"]
               if e.get("href") == "https://example.com/foo"]
    assert matching, "link dropped between ScrapingResult and CrawlResult"
    assert matching[0].get("page") == 1
    # The doc'd consumer pattern works.
    assert result.links.get("internal", []) is not None
    assert result.links.get("external", []) is not None


@pytest.mark.asyncio
async def test_arun_pdf_links_are_json_serializable(single_link_pdf):
    """The cache writer calls ``json.dumps(result.links)`` in
    ``AsyncDatabaseManager.acache_url`` -- the new links shape must not
    break it (the previous ``{"urls": [...]}`` shape, had it survived,
    would have been fine too, but the page ints must also be
    JSON-serializable)."""
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        scraping_strategy=PDFContentScrapingStrategy(extract_images=False),
    )
    async with AsyncWebCrawler(crawler_strategy=PDFCrawlerStrategy()) as crawler:
        result = await crawler.arun(f"file://{single_link_pdf}", config=config)

    serialized = json.dumps(result.links)  # the exact call acache_url makes
    assert "https://example.com/foo" in serialized
    assert '"page": 1' in serialized or '"page":1' in serialized
