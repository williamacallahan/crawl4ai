r"""Regression tests for ``CustomHTML2Text`` fence-language labeling.

Background (issue: mermaid fence label lost on default crawl path)
-------------------------------------------------------------------
``crawl4ai/content_scraping_strategy.py`` writes, for each labeled unwrapped
mermaid SVG, a placeholder::

    <pre data-language="mermaid"><code class="language-mermaid">…</code></pre>

The default crawl path runs ``remove_unwanted_attributes_fast`` with
``keep_data_attributes=False`` (the ``CrawlerRunConfig`` default), which strips
``data-language`` (not in ``IMPORTANT_ATTRS``) but keeps ``class`` (in
``IMPORTANT_ATTRS``). ``CustomHTML2Text.handle_tag`` previously read the fence
language *only* from ``<pre data-language="…">``, so on the default path the
fence came out as a bare ``\`\`\`` instead of ``\`\`\`mermaid``.

The fix defers emitting the opening fence until the first child of ``<pre>`` is
seen, and falls back to the inner ``<code class="language-…">`` (the GitHub
convention, which the scraper already writes and which survives stripping) when
``data-language`` is absent. These tests pin both the fix and the
no-regression behaviors around ``<pre>``/``<code>`` fence handling.
"""

import pytest

from crawl4ai.html2text import CustomHTML2Text

_MER = "```"


def _convert(html: str) -> str:
    h = CustomHTML2Text()
    h.update_params(
        body_width=0,
        ignore_links=False,
        single_line_break=True,
        mark_code=True,
    )
    return h.handle(html)


# ---------------------------------------------------------------------------
# The bug: <pre> whose data-language has been stripped must still be labeled
# from the inner <code class="language-…"> (GitHub convention).
# ---------------------------------------------------------------------------


def test_pre_with_code_class_language_fallback_when_data_language_stripped():
    md = _convert('<pre><code class="language-mermaid">%% flowchart diagram\nA\nB</code></pre>')
    assert _MER + "mermaid" in md, repr(md)
    assert "%% flowchart diagram" in md
    assert "A" in md and "B" in md


def test_pre_with_code_class_language_python_fallback():
    md = _convert('<pre><code class="language-python">print(1)</code></pre>')
    assert _MER + "python" in md, repr(md)


def test_pre_data_language_takes_precedence_over_code_class():
    """An explicit data-language on <pre> wins over any inner <code> class."""
    md = _convert(
        '<pre data-language="rust"><code class="language-python">x</code></pre>'
    )
    assert _MER + "rust" in md, repr(md)
    assert _MER + "python" not in md, repr(md)


def test_pre_data_language_present_no_code_still_labeled():
    md = _convert('<pre data-language="bash">echo hi</pre>')
    assert _MER + "bash" in md, repr(md)


def test_pre_no_language_signal_remains_unlabeled():
    """No false labels: a <pre> with neither data-language nor a language-*
    class on its <code> child stays a bare ``` fence."""
    md = _convert("<pre>plain\ncode</pre>")
    assert _MER + "\nplain\ncode\n" + _MER in md, repr(md)
    # And a <code> child without a language-* class does not introduce a label.
    md2 = _convert("<pre><code>def foo():\n    return 1</code></pre>")
    assert _MER + "\ndef foo():\n    return 1\n" + _MER in md2, repr(md2)


def test_empty_pre_emits_paired_fences():
    md = _convert("<pre></pre>")
    assert md.count(_MER) == 2, repr(md)


# ---------------------------------------------------------------------------
# Indentation / container regressions: language recovery must not disturb the
# in-list / in-blockquote fence-indenting behavior.
# ---------------------------------------------------------------------------


def test_pre_with_code_class_in_list_indented_and_labeled():
    md = _convert(
        "<ul><li>x<pre><code class=\"language-python\">def foo():\n    return 1"
        "</code></pre></li></ul>"
    )
    assert "    " + _MER + "python" in md, repr(md)
    assert "    def foo():" in md, repr(md)


def test_pre_with_code_class_in_blockquote_indented_and_labeled():
    md = _convert(
        "<blockquote><pre><code class=\"language-python\">x=1</code></pre></blockquote>"
    )
    assert "> " + _MER + "python" in md, repr(md)
    assert "> x=1" in md, repr(md)


# ---------------------------------------------------------------------------
# State invariants: the deferred-fence machinery resets after </pre>.
# ---------------------------------------------------------------------------


def test_pre_state_resets_after_close():
    h = CustomHTML2Text()
    h.update_params(body_width=0, single_line_break=True)
    h.handle("<pre><code class=\"language-python\">x</code></pre><p>after</p>")
    assert h.inside_pre is False
    assert h._pre_prefix is None
    assert h._pre_fence_pending is False
    assert h._pre_lang == ""


def test_consecutive_pre_blocks_each_independent():
    md = _convert(
        '<pre><code class="language-python">a</code></pre>'
        '<pre><code class="language-bash">b</code></pre>'
    )
    assert _MER + "python" in md, repr(md)
    assert _MER + "bash" in md, repr(md)


def test_pre_then_plain_pre_does_not_inherit_lang():
    """A labeled <pre> must not bleed its label into a later unlabeled <pre>."""
    md = _convert(
        '<pre><code class="language-python">a</code></pre>'
        '<pre>plain</pre>'
    )
    # The second fence must be bare (no python label leaked into it).
    assert _MER + "\nplain\n" + _MER in md, repr(md)
