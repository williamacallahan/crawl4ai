"""Regression tests for ``LXMLWebScrapingStrategy._scrap`` honoring the
``word_count_threshold`` parameter.

The bug (introduced in f3ae5a65, Jan 2025): ``_scrap`` accepts a
``word_count_threshold`` parameter (forwarded from
``CrawlerRunConfig.word_count_threshold``) but its only word-count pruning
call hardcoded the threshold to ``1``::

    self.remove_empty_elements_fast(content_element, 1)  # ignored threshold

Setting ``word_count_threshold=N`` (>1) therefore had no effect on the
default LXML scraping path; the output was byte-identical to ``N=1``. The
legacy BeautifulSoup path (``crawl4ai/utils.py::get_content_of_website``)
forwards the parameter into ``remove_empty_and_low_word_count_elements``
and ``remove_small_text_tags`` and removes short blocks as documented.

The fix forwards the parameter to the pruning helper::

    self.remove_empty_elements_fast(content_element, word_count_threshold)

These tests guard the parameter is honored (T1), the helper's ``<`` (not
``<=``) boundary semantics (T2), the bypass-tag and ``<pre>``/``<code>``
non-pruning contracts at a high threshold (T3, T4), and the full
``CrawlerRunConfig`` -> ``arun`` -> ``ascrap`` -> ``scrap`` -> ``_scrap``
plumbing chain (T5). The unit-level tests use synthetic HTML; the
end-to-end test mocks the browser fetch (no Playwright required) and
isolates the cache DB.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_reg_word_count_threshold.py
"""

import pytest

from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


@pytest.fixture
def scraper():
    return LXMLWebScrapingStrategy()


def _paragraph(n_words, prefix="w"):
    """Exactly ``n_words`` whitespace-separated tokens."""
    return " ".join(f"{prefix}{i}" for i in range(n_words))


# Synthetic HTML fixtures (paragraphs of exact word counts).

# Bug-report reproduction page: three paragraphs of 4, 8, and 60 words.
SHORT_A = "alpha bravo charlie delta"  # 4 words
SHORT_B = "echo foxtrot golf hotel india juliet kilo"  # 8 words
LONG_60 = _paragraph(60)  # 60 words exactly

BUG_REPORT_HTML = (
    "<html><head><title>T</title></head><body>"
    f'<p id="short4">{SHORT_A}</p>'
    f'<p id="short8">{SHORT_B}</p>'
    f'<p id="long60">{LONG_60}</p>'
    "</body></html>"
)

# Two paragraphs of 30 and 31 words. The boundary between `<` (kept) and
# `<=` (would-be-removed) is asserted: a 30-word block survives at threshold
# 30 (30 < 30 is False) and is pruned at 31. Distinct prefixes prevent one
# block's text from being a substring of the other's.
BOUNDARY_HTML = (
    "<html><body>"
    f'<p id="p30">{_paragraph(30, prefix="thirty_")}</p>'
    f'<p id="p31">{_paragraph(31, prefix="thirtyone_")}</p>'
    "</body></html>"
)


def _cleaned(result):
    return result.get("cleaned_html", "")


# T1: the parameter is honored (core bug fix + footgun guard against
# re-introducing the hardcoded literal ``1``).

class TestWordCountThresholdHonored:
    """``word_count_threshold=N`` must prune leaf blocks below N words."""

    def test_threshold_prunes_short_leaf_blocks(self, scraper):
        ch = _cleaned(
            scraper._scrap("raw://test", BUG_REPORT_HTML, word_count_threshold=50)
        )
        assert SHORT_A not in ch, "4-word block must be pruned at threshold=50"
        assert SHORT_B not in ch, "8-word block must be pruned at threshold=50"
        assert LONG_60 in ch, "60-word block must survive at threshold=50"


# T2: helper uses ``<`` (not ``<=``). Pins the off-by-one semantics so a
# refactor that flips the comparison is caught.

class TestThresholdBoundary:
    """``remove_empty_elements_fast`` uses ``len(words) < threshold``.
    A 30-word block survives at threshold=30 and is pruned at threshold=31."""

    def test_threshold_at_boundary_uses_strict_less_than(self, scraper):
        at_30 = _cleaned(
            scraper._scrap("raw://test", BOUNDARY_HTML, word_count_threshold=30)
        )
        at_31 = _cleaned(
            scraper._scrap("raw://test", BOUNDARY_HTML, word_count_threshold=31)
        )
        assert "thirty_0" in at_30, (
            "30-word block must survive at threshold=30 (helper uses `<`)"
        )
        assert "thirtyone_0" in at_30
        assert "thirty_0" not in at_31, "30-word block pruned at threshold=31"
        assert "thirtyone_0" in at_31, "31-word block survives at threshold=31"


# T3 + T4: the pruning helper's bypass contracts — bypass-tag set and
# ``<pre>``/``<code>`` descendant protection — must not regress at a high
# threshold. These contracts are inside ``remove_empty_elements_fast``
# (the function the fix now calls with the user's threshold value), so a
# future refactor that drops a bypass tag or removes the ancestor check
# would silently strip link/image/table-cell scaffolding or code blocks.

class TestBypassTagsAndCodeBlocks:
    """The pruning helper skips a bypass-tag set (a, img, br, hr, input,
    meta, link, source, track, wbr, tr, td, th) and any descendant of a
    ``<pre>``/``<code>`` block. The fix must not regress those rules."""

    def test_bypass_tags_survive_high_threshold(self, scraper):
        # Padding paragraph keeps the body non-empty after the short
        # bypass-tag leaves are processed. The bypass tags themselves are
        # 1 word or less so they would be pruned if the bypass rule broke.
        html = (
            '<html><body>'
            '<p>' + _paragraph(60, prefix="pad") + '</p>'
            '<a href="https://example.com/x">x</a>'
            '<img src="/i.png" alt="alt"><br><hr>'
            '<table><tr><td>cell</td><th>h</th></tr></table>'
            '</body></html>'
        )
        ch = _cleaned(scraper._scrap("raw://test", html, word_count_threshold=50))
        assert "<a" in ch, "<a> (bypass tag) must survive a high threshold"
        assert "<img" in ch, "<img> (bypass tag) must survive a high threshold"
        assert "<br" in ch, "<br> (bypass tag) must survive a high threshold"
        assert "<hr" in ch, "<hr> (bypass tag) must survive a high threshold"
        assert "<tr" in ch, "<tr> (bypass tag) must survive a high threshold"
        assert "<td" in ch, "<td> (bypass tag) must survive a high threshold"
        assert "<th" in ch, "<th> (bypass tag) must survive a high threshold"

    def test_pre_code_block_descendants_survive_high_threshold(self, scraper):
        """The bypass rule protects *descendants* of ``<pre>`` / ``<code>``
        only. A ``<pre>`` containing short ``<span>`` leaves (1 word each)
        must keep those spans even at a high threshold."""
        html = (
            '<html><body>'
            '<p>' + _paragraph(60, prefix="pad") + '</p>'
            '<pre><span class="w">short</span> <span class="w">code</span></pre>'
            '<code><span class="w">a</span> <span class="w">b</span></code>'
            '</body></html>'
        )
        ch = _cleaned(scraper._scrap("raw://test", html, word_count_threshold=50))
        assert "short" in ch, (
            "short span inside <pre> must survive (descendants are bypassed)"
        )
        assert "code" in ch
        assert '<pre' in ch, "<pre> container survives (only its descendants bypass)"
        assert "<span" in ch, (
            "code-block spans must survive (descendants of <pre>/<code> are bypassed)"
        )


# T4b: inline elements inside a long paragraph and childless ``<pre>``/``<code>``
# must survive a high threshold. The threshold applies to block-level elements
# only; short inline leaves (``<strong>``, ``<code>``, ``<span>``, ...) are kept
# with a threshold of 1, and childless ``<pre>``/``<code>`` are protected
# directly (the ancestor-based code-block check only covers descendants).

INLINE_IN_LONG_PARA_HTML = (
    "<html><body>"
    '<p id="long">' + _paragraph(60, prefix="pad") + "</p>"
    "<p>This sentence has a <strong>bold</strong> word and a "
    "<code>code</code> word and an <em>em</em> word.</p>"
    "</body></html>"
)

CHILDLESS_PRE_HTML = (
    "<html><body>"
    '<p>' + _paragraph(60, prefix="pad") + '</p>'
    "<pre>short</pre>"
    "<code>x</code>"
    "</body></html>"
)


class TestInlineAndChildlessCodeBlocks:
    """``word_count_threshold`` must not strip short inline text inside a
    longer paragraph nor childless ``<pre>``/``<code>`` blocks."""

    def test_inline_elements_survive_high_threshold(self, scraper):
        ch = _cleaned(
            scraper._scrap(
                "raw://test", INLINE_IN_LONG_PARA_HTML, word_count_threshold=50
            )
        )
        assert "<strong" in ch, "<strong> inline text must survive"
        assert "<code" in ch, "<code> inline text must survive"
        assert "<em" in ch, "<em> inline text must survive"
        assert "bold" in ch, "<strong> word text must be preserved"
        assert "code" in ch, "<code> word text must be preserved"

    def test_childless_pre_code_survive_high_threshold(self, scraper):
        ch = _cleaned(
            scraper._scrap(
                "raw://test", CHILDLESS_PRE_HTML, word_count_threshold=50
            )
        )
        assert "<pre" in ch, "childless <pre> must survive a high threshold"
        assert "short" in ch, "<pre> text must be preserved"
        assert "<code" in ch, "childless <code> must survive a high threshold"
        assert ">x<" in ch, "<code> text must be preserved"


# T5: end-to-end through ``AsyncWebCrawler.arun``. Pins the full plumbing
# chain ``CrawlerRunConfig.word_count_threshold`` -> ``arun`` -> ``ascrap``
# -> ``scrap`` -> ``_scrap(word_count_threshold=...)``. The browser fetch
# is mocked so the test runs headlessly; the real ``LXMLWebScrapingStrategy``,
# real ``aprocess_html`` and real ``DefaultMarkdownGenerator`` are NOT
# mocked. This is the offline deterministic mirror of the live-network
# ``tests/async/test_parameters_and_options.py::test_word_count_threshold``
# (which is not collected by CI and passes the deprecated ``bypass_cache``
# kwarg that ``CrawlerRunConfig._UNWANTED_PROPS`` rejects).

class TestEndToEndThroughArun:
    """Full plumbing chain from ``CrawlerRunConfig`` to ``_scrap``."""

    @pytest.mark.asyncio
    async def test_word_count_threshold_via_CrawlerRunConfig_arun(
        self, tmp_path, monkeypatch
    ):
        from crawl4ai.async_configs import CrawlerRunConfig
        from crawl4ai.async_webcrawler import AsyncWebCrawler, async_db_manager
        from crawl4ai.cache_context import CacheMode
        from crawl4ai.models import AsyncCrawlResponse
        from crawl4ai.utils import ensure_content_dirs

        # Isolate the cache DB so the test does not touch ~/.crawl4ai.
        db_path = str(tmp_path / "test_crawl4ai.db")
        content_base = str(tmp_path / "content")
        monkeypatch.setattr(async_db_manager, "db_path", db_path)
        monkeypatch.setattr(
            async_db_manager, "content_paths", ensure_content_dirs(content_base)
        )
        monkeypatch.setattr(async_db_manager, "_initialized", True)
        monkeypatch.setattr(async_db_manager, "connection_pool", {})
        await async_db_manager.ainit_db()
        await async_db_manager.update_db_schema()

        # LIVE_HTML mirrors the bug-report page (short + long paragraphs),
        # large enough (>100 bytes, real visible text) to pass the anti-bot
        # ``is_blocked`` check the fresh-fetch path runs.
        short_a = "alpha bravo"  # 2 words
        short_b = "charlie delta echo foxtrot golf"  # 5 words
        long_p = _paragraph(60)  # 60 words
        url = "https://example.com/article"
        live_html = (
            "<html><head><title>T</title></head><body>"
            f"<p>{short_a}</p><p>{short_b}</p><p>{long_p}</p></body></html>"
        )

        class _ResponseStrategy:
            async def crawl(self, url, config):
                return AsyncCrawlResponse(
                    html=live_html, response_headers={}, status_code=200
                )

            def update_user_agent(self, *args, **kwargs):
                pass

        crawler = AsyncWebCrawler(
            crawler_strategy=_ResponseStrategy(), base_directory=str(tmp_path)
        )
        crawler.ready = True

        cfg_low = CrawlerRunConfig(
            word_count_threshold=1, cache_mode=CacheMode.BYPASS, verbose=False
        )
        cfg_high = CrawlerRunConfig(
            word_count_threshold=50, cache_mode=CacheMode.BYPASS, verbose=False
        )
        r_low = await crawler.arun(url=url, config=cfg_low)
        r_high = await crawler.arun(url=url, config=cfg_high)

        assert r_low.success, f"low-threshold crawl failed: {r_low.error_message}"
        assert r_high.success, f"high-threshold crawl failed: {r_high.error_message}"
        # Headline regression (mirrors
        # tests/async/test_parameters_and_options.py::test_word_count_threshold
        # against the live NBC URL, deterministically offline):
        assert len(r_high.markdown.raw_markdown) < len(r_low.markdown.raw_markdown), (
            "word_count_threshold=50 via CrawlerRunConfig must shrink the "
            "generated markdown vs threshold=1."
        )
        # Pin the pruning semantics directly on cleaned_html.
        assert short_a in r_low.cleaned_html
        assert short_b in r_low.cleaned_html
        assert long_p in r_low.cleaned_html
        assert short_a not in r_high.cleaned_html, "2-word block pruned at 50"
        assert short_b not in r_high.cleaned_html, "5-word block pruned at 50"
        assert long_p in r_high.cleaned_html, "60-word block must survive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
