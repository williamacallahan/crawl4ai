"""
Regression tests for the BM25ContentFilter excluded-tag bug.

Bug: ``BM25ContentFilter.filter_content`` never removed the
``RelevantContentFilter.excluded_tags`` boilerplate
(``nav`` / ``footer`` / ``header`` / ``aside`` / ``script`` / ``style`` /
``form`` / ``iframe`` / ``noscript``) before extracting and scoring text
chunks, so on real pages where the boilerplate text overlaps the page
query (very common with navigation that repeats the product / topic
name) the boilerplate was returned as a "content" block and polluted
``MarkdownGenerationResult.fit_html`` / ``fit_markdown``.

Fix: mirror ``PruningContentFilter.filter_content`` by calling
``_remove_comments`` + ``_remove_unwanted_tags`` on the soup before
chunk extraction (the helpers were hoisted into the shared
``RelevantContentFilter`` base class so both strategies share them).

These tests guard against the specific failure modes that the original
``tests/async/test_content_filter_bm25.py::test_excluded_tags`` could
not catch (it passed vacuously because its HTML produced an empty
query, so ``filter_content`` returned ``[]`` before any scoring). The
parametrized tag walk in that strengthened test covers the lenient
-threshold verbatim-leak case for ``script``/``style``/``nav``/
``footer``/``header``; the tests here cover the production-default
-threshold failure mode, the realistic docs-page shape, the
end-to-end ``fit_markdown`` pipeline, and the helper-hoist refactor.
"""

from bs4 import BeautifulSoup

from crawl4ai.content_filter_strategy import (
    BM25ContentFilter,
    PruningContentFilter,
    RelevantContentFilter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FILLER_PARAGRAPHS = [
    "<p>JavaScript powers interactive websites and runs in every modern browser environment.</p>",
    "<p>Rust provides memory safety without garbage collection through its ownership model.</p>",
    "<p>Go was designed at Google for building scalable network services and cloud infrastructure.</p>",
    "<p>Ruby on Rails popularized convention over configuration in web application frameworks.</p>",
    "<p>TypeScript adds static type checking to JavaScript for large-scale application development.</p>",
    "<p>Java remains dominant in enterprise software and Android mobile application development.</p>",
    "<p>C++ is essential for game engines, operating systems, and high-performance computing.</p>",
    "<p>Python is widely adopted for data science, automation, scripting, and machine learning workflows.</p>",
]
FILLER_BLOCK = "\n".join(FILLER_PARAGRAPHS)


def _wrap_html(body_inner: str, title: str = "Python Tutorial") -> str:
    """Wrap body HTML with head metadata so ``extract_page_query`` is non-empty."""
    return (
        f"<html><head><title>{title}</title>"
        f'<meta name="description" content="{title} for beginners"></head>'
        f"<body>{body_inner}</body></html>"
    )


# ---------------------------------------------------------------------------
# Production-default threshold (the user-visible failure mode)
# ---------------------------------------------------------------------------


def test_nav_does_not_leak_at_default_threshold_realworld_corpus():
    """Reproduces the production failure mode from the bug report: with a
    realistic docs-site corpus the ``<nav>`` chunk scored above the default
    ``bm25_threshold=1.0`` and was returned as the first content block.

    After the fix the nav (and footer) must be absent at the default
    threshold while the real article body survives. The strengthened
    ``test_excluded_tags`` uses a lenient threshold; this test is the
    only one that catches a regression where nav leaks only at the
    default threshold.
    """
    nav_links = (
        "<nav>"
        '<a href="/docs/">Acme Rate Limiter</a>'
        '<a href="/docs/sdk">Acme SDK</a>'
        '<a href="/docs/headers">Rate Limiter Headers</a>'
        '<a href="/docs/quota">Quota & Usage</a>'
        '<a href="/docs/errors">Error Codes</a>'
        "</nav>"
    )
    article_paragraphs = [
        "Acme SDK installation steps for getting started with the platform and API usage basics.",
        "Acme API authentication workflow describes how to obtain tokens and refresh credentials programmatically.",
        "Acme SDK response codes for common rate limit conditions explain how the limiter behaves client-side.",
        "Acme dashboard usage analytics give insight into quota consumption across all your services and projects.",
        "Acme API error codes reference lists every status code returned by the rate limiter and the gateway service.",
        "Acme rate limiter headers tell clients when their quota resets and how many calls remain in the window.",
        "Acme SDK language bindings include Python, JavaScript, Go, and Java examples for the rate limiting methods.",
        "Acme developer portal organizes API documentation, SDK downloads, quota tools, and rate limit guidance.",
        "Acme support channel answers API rate limit questions, SDK installation problems, and quota troubleshooting.",
        "Acme release notes summarize API rate limiter improvements, SDK bug fixes, and quota dashboard enhancements.",
        "Acme API endpoints accept JSON payloads with rate limit headers and return structured quota usage reports.",
        "Acme webhook integration uses the SDK to forward quota events and rate limiter status to your backend.",
        "Acme CLI tool lets you inspect API quota, adjust rate limiter thresholds, and test SDK calls from terminal.",
        "Acme status page reports API uptime, rate limiter health, SDK compatibility, and quota dashboard live.",
        "Acme migration guide helps SDK users move between API versions while preserving quota settings and limits.",
        "Acme best practices recommend SDK retries on rate limiter responses and cache patterns that respect quota rules.",
    ]
    article = "<article><h1>Acme API Rate Limiter</h1>"
    for p in article_paragraphs:
        article += f"<p>{p}</p>"
    article += "</article>"
    footer = "<footer>Copyright Acme Corporation. All rights reserved.</footer>"
    html = (
        f"<html><head><title>Acme API Rate Limiter</title></head>"
        f"<body>{nav_links}{article}{footer}</body></html>"
    )

    results = BM25ContentFilter(bm25_threshold=1.0).filter_content(html)
    combined = " ".join(results)

    assert "<nav" not in combined, "Nav wrapper leaked at default bm25_threshold=1.0"
    assert "Acme Rate Limiter</a>" not in combined, (
        "Nav anchor text leaked at default bm25_threshold=1.0"
    )
    assert "Rate Limiter Headers</a>" not in combined, (
        "Nav anchor text leaked at default bm25_threshold=1.0"
    )
    assert "Copyright Acme Corporation" not in combined, (
        "Footer text leaked at default bm25_threshold=1.0"
    )
    assert "Acme SDK installation steps" in combined, (
        "Article body should survive at the default threshold once boilerplate is removed"
    )


def test_realistic_docs_page_excludes_boilerplate():
    """Realistic docs-page shape: a breadcrumb ``<nav>`` whose link repeats
    the page topic, a sidebar ``<aside>`` with TOC entries, a copyright
    ``<footer>``, plus the article body. At a lenient threshold none of
    the boilerplate should appear in the output while the article body
    remains. Guards against a regression in this canonical real-world
    page shape.
    """
    breadcrumb = (
        '<nav class="breadcrumb"><ul><li>'
        '<a href="/tutorial/index.html">The Python Tutorial</a></li></ul></nav>'
    )
    sidebar_toc = (
        '<aside class="sidebar"><ul>'
        '<li><a href="interpreter.html#argument-passing">'
        "2.1.1. Argument Passing - Python Tutorial</a></li>"
        '<li><a href="introduction.html#numbers">'
        "3.1.1. Numbers - Python Tutorial</a></li>"
        "</ul></aside>"
    )
    footer = (
        "<footer>See the Python Tutorial History and License. "
        "Copyright Python Software Foundation.</footer>"
    )
    article = (
        "<article><h1>The Python Tutorial</h1>"
        "<p>Python is an easy to learn, powerful programming language. "
        "It has efficient high-level data structures and a simple but "
        "effective approach to object-oriented programming. This tutorial "
        "introduces the reader informally to the basic concepts and "
        "features of the Python language and system.</p>"
        "</article>"
    )
    html = _wrap_html(
        breadcrumb + sidebar_toc + article + footer + FILLER_BLOCK,
        title="The Python Tutorial",
    )

    results = BM25ContentFilter(bm25_threshold=0.01).filter_content(html)
    combined = " ".join(results).lower()

    assert "the python tutorial</a>" not in combined, "Breadcrumb leaked"
    assert "argument passing" not in combined, "Sidebar TOC entry leaked"
    assert "2.1.1." not in combined, "Sidebar TOC numbering leaked"
    assert "3.1.1." not in combined, "Sidebar TOC numbering leaked"
    assert "copyright python software foundation" not in combined, (
        "Footer copyright leaked"
    )
    assert "python is an easy to learn" in combined, (
        "Real article body must survive once boilerplate is stripped"
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline integration
# ---------------------------------------------------------------------------


def test_fit_markdown_pipeline_excludes_nav():
    """End-to-end via ``DefaultMarkdownGenerator``: the production
    ``fit_html`` / ``fit_markdown`` path (the symptom the bug report
    describes) must not contain nav link labels, and must contain the
    article body. ``raw_markdown`` / ``markdown_with_citations`` are
    produced by running ``CustomHTML2Text`` directly on the input HTML
    and are unaffected by the content filter; this test only asserts on
    the fit path.
    """
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    nav = (
        "<nav>"
        '<a href="/docs/">Python Tutorial</a>'
        '<a href="/docs/sdk">Python SDK</a>'
        '<a href="/docs/errors">Error Codes</a>'
        "</nav>"
    )
    article = (
        "<article><h1>Python Tutorial</h1>"
        "<p>The Python tutorial walks beginners through syntax, data "
        "structures, functions, modules, classes, and the standard library "
        "with worked examples that build on one another across chapters.</p>"
        "</article>"
    )
    cleaned_html = _wrap_html(nav + article + FILLER_BLOCK)

    generator = DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(
            user_query="Python Tutorial", bm25_threshold=0.01
        )
    )
    result = generator.generate_markdown(
        cleaned_html, base_url="https://docs.python.org", citations=False
    )

    assert result.fit_html is not None
    assert result.fit_markdown is not None
    assert "Error Codes" not in result.fit_markdown, "Nav link label leaked into fit_markdown"
    assert "Python SDK" not in result.fit_html, "Nav link label leaked into fit_html"
    assert "Python SDK" not in result.fit_markdown, "Nav link label leaked into fit_markdown"
    assert "walks beginners through syntax" in result.fit_markdown, (
        "Article body should appear in fit_markdown"
    )


# ---------------------------------------------------------------------------
# Comment removal (defensive against inflation of scored chunk text)
# ---------------------------------------------------------------------------


def test_comments_removed_before_bm25_chunk_extraction():
    """HTML comments are ``NavigableString`` subclasses that
    ``extract_text_chunks`` would otherwise fold into a chunk's text and
    inflate scoring. The fix mirrors ``PruningContentFilter`` by calling
    ``_remove_comments`` before chunk extraction; this test verifies the
    chunk corpus does not contain the comment text and that the article
    chunk still extracts as expected.
    """
    commented_html = _wrap_html(
        "<!-- Python Tutorial secret comment payload -->"
        "<article><h1>Python Tutorial</h1>"
        "<p>The Python tutorial paragraph with enough words for BM25 to score.</p>"
        "</article>" + FILLER_BLOCK
    )
    filt = BM25ContentFilter(user_query="Python Tutorial", bm25_threshold=0.01)
    soup = BeautifulSoup(commented_html, "lxml")
    filt._remove_comments(soup)
    filt._remove_unwanted_tags(soup)
    body = soup.find("body")
    chunks = filt.extract_text_chunks(body, None)

    chunk_texts = [chunk[1].lower() for chunk in chunks]
    assert not any("secret comment payload" in t for t in chunk_texts), (
        "HTML comment text survived into chunk corpus"
    )
    assert any(
        "python tutorial paragraph with enough words" in t for t in chunk_texts
    ), "Article paragraph chunk should still be extracted"


# ---------------------------------------------------------------------------
# Helper-hoist refactor regression guards
# ---------------------------------------------------------------------------


def test_exclusion_helpers_live_on_base_class():
    """``_remove_comments`` and ``_remove_unwanted_tags`` must be defined
    once on the shared ``RelevantContentFilter`` base class and inherited
    (not duplicated) by both concrete strategies. Guards against a future
    refactor that re-duplicates the helpers (which would silently let the
    BM25 and Pruning implementations drift apart again — the original bug).
    """
    assert callable(RelevantContentFilter._remove_comments)
    assert callable(RelevantContentFilter._remove_unwanted_tags)
    assert PruningContentFilter._remove_comments is RelevantContentFilter._remove_comments
    assert PruningContentFilter._remove_unwanted_tags is RelevantContentFilter._remove_unwanted_tags


def test_pruning_still_removes_excluded_tags_after_helper_hoist():
    """``PruningContentFilter`` previously owned the helper definitions;
    after they were hoisted into the base class it must continue to drop
    excluded-tag text and keep the article body, exactly as before. Guards
    against a regression in Pruning introduced by the helper move.
    """
    html = (
        "<html><body>"
        "<nav>Python Tutorial nav boilerplate text here.</nav>"
        "<article><h1>Python Tutorial</h1>"
        "<p>This is substantive article prose with enough words to clear "
        "any reasonable minimum threshold for retention of meaningful "
        "content for the pruning process to keep it.</p>"
        "</article>"
        "<footer>Copyright Python Software Foundation</footer>"
        "</body></html>"
    )
    results = PruningContentFilter(min_word_threshold=5).filter_content(html)
    combined = " ".join(results).lower()
    assert "boilerplate text here" not in combined, "Nav leaked via Pruning"
    assert "copyright python software foundation" not in combined, (
        "Footer leaked via Pruning"
    )
    assert "substantive article prose" in combined, "Pruning must keep the article body"
