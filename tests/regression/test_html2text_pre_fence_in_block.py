"""Regression tests for fenced code blocks (````` ```` `````) emitted by
``CustomHTML2Text`` when a ``<pre>`` appears inside an enclosing ``<li>`` or
``<blockquote>``.

Background
----------
``crawl4ai/html2text/__init__.py`` overrides ``<pre>`` handling in
``CustomHTML2Text.handle_tag`` to emit CommonMark fenced (`` ``` ``) code
blocks instead of the base ``HTML2Text``'s 4-space indented code blocks. The
override short-circuits with ``self.o(f"\\n```{lang}\\n")`` and
``self.inside_pre = True`` and never calls ``super().handle_tag("pre", ...)``,
so the base class's ``self.pre = True`` flag (and thus the per-list/blockquote
indent branch in ``HTML2Text.o``) never engages. Net result: the opening
fence, every code line, and the closing fence were all written starting at
column 0, so a strict CommonMark renderer (e.g. ``markdown_it``) treats the
opening fence as terminating the enclosing list/blockquote, and the code
block (plus its closing fence) is hoisted out to the top level - list items
render empty and the blockquotes collapse.

The three production paths (``DefaultMarkdownGenerator.generate_markdown``,
``crawl4ai/utils.py::get_content_of_website_optimized`` and the legacy
``get_content_of_website``) also applied a string-level strip
``markdown.replace("    ```", "```")`` after conversion. This strip is a
no-op on the broken output, but it would *undo* the indenting fix once the
override starts emitting 4-space-indented fences for list-nested ``<pre>``.
The strip is therefore removed at all three call sites; one of the tests
here guards against it being re-introduced.
"""

import os
import subprocess
import sys

import pytest

markdown_it = pytest.importorskip("markdown_it")

from crawl4ai.html2text import HTML2Text, CustomHTML2Text  # noqa: E402
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator  # noqa: E402
from crawl4ai.utils import get_content_of_website_optimized  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CLI_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH", "")])
    ),
}


def _convert(html: str) -> str:
    """Direct ``CustomHTML2Text`` conversion with production-equivalent options."""
    h = CustomHTML2Text()
    h.update_params(
        body_width=0,
        ignore_links=False,
        single_line_break=True,
        mark_code=True,
    )
    return h.handle(html)


def _convert_base(html: str) -> str:
    """Base ``HTML2Text`` conversion (used by the CLI)."""
    h = HTML2Text()
    h.body_width = 0
    return h.handle(html)


def _render(md: str) -> str:
    return markdown_it.MarkdownIt().render(md)


def _gen_markdown(html: str, base_url: str = "") -> str:
    gen = DefaultMarkdownGenerator()
    return gen.generate_markdown(html, base_url=base_url, citations=False).raw_markdown


def _wrap_html(fragment: str) -> str:
    return f"<html><head><title>tmp</title></head><body>{fragment}</body></html>"


# ---------------------------------------------------------------------------
# Core bug: <pre> inside <li> must keep its fences inside the list item
# ---------------------------------------------------------------------------


def test_pre_in_list_item_renders_inside_li():
    md = _convert("<ul><li>Step1 <pre>npm install</pre></li></ul>")
    assert "    ```\n    npm install\n    ```" in md, repr(md)
    assert "Step1 \n```\n" not in md, repr(md)
    rendered = _render(md)
    assert "<pre><code>npm install\n</code></pre>" in rendered, repr(rendered)
    assert "<ul>\n<li></li>\n</ul>" not in rendered
    assert rendered.rstrip().endswith("</ul>"), repr(rendered)


def test_pre_in_two_item_list_both_indented():
    html = (
        "<ul><li>Step1 <pre>npm install</pre></li>"
        "<li>Step2 <pre>npm run dev</pre></li></ul>"
    )
    md = _convert(html)
    assert "    ```\n    npm install\n    ```" in md, repr(md)
    assert "    ```\n    npm run dev\n    ```" in md, repr(md)
    rendered = _render(md)
    assert rendered.count("<pre><code>npm install") == 1, repr(rendered)
    assert rendered.count("<pre><code>npm run dev") == 1, repr(rendered)
    assert rendered.count("<li>") == 2, repr(rendered)
    assert "<ul>\n<li></li>\n</ul>\n<pre>" not in rendered, repr(rendered)


def test_pre_in_ordered_list_item_stays_inside_li():
    html = "<ol><li>Step1 <pre>npm install</pre></li></ol>"
    md = _convert(html)
    rendered = _render(md)
    assert "<pre><code>npm install\n</code></pre>" in rendered, repr(rendered)
    li_pos = rendered.find("<li>")
    pre_pos = rendered.find("<pre><code>")
    ol_end = rendered.find("</ol>")
    assert li_pos < pre_pos < ol_end, repr(rendered)


def test_pre_in_ordered_list_two_digit_number_stays_inside_li():
    html = '<ol start="10"><li>v10 <pre>build v10</pre></li></ol>'
    md = _convert(html)
    # Two-digit ol markers take 4 chars ("10. ") + 2-space indent = 6 chars,
    # so the content column (and the fence indent) is 6.
    assert "      ```\n      build v10\n      ```" in md, repr(md)
    rendered = _render(md)
    assert "<pre><code>build v10\n</code></pre>" in rendered, repr(rendered)
    assert "</ol>\n<pre>" not in rendered, repr(rendered)


# ---------------------------------------------------------------------------
# Core bug: <pre> inside <blockquote> must keep its fences inside the
# blockquote
# ---------------------------------------------------------------------------


def test_pre_in_blockquote_renders_inside_blockquote():
    md = _convert("<blockquote><pre>quoted\ncode</pre></blockquote>")
    assert "> ```\n> quoted\n> code\n> ```" in md, repr(md)
    assert "\n```\nquoted" not in md, repr(md)
    rendered = _render(md)
    assert "<pre><code>quoted\ncode\n</code></pre>" in rendered, repr(rendered)
    assert "<blockquote>\n</blockquote>" not in rendered
    assert "<blockquote></blockquote>\n<pre>" not in rendered, repr(rendered)


def test_pre_in_nested_blockquote_renders_inside_outer_blockquote():
    html = "<blockquote><blockquote><pre>quoted\ncode</pre></blockquote></blockquote>"
    md = _convert(html)
    rendered = _render(md)
    assert "<pre><code>quoted\ncode\n</code></pre>" in rendered, repr(rendered)
    assert "</blockquote>\n<pre>" not in rendered, repr(rendered)


# ---------------------------------------------------------------------------
# No regression: top-level <pre> continues to produce a single top-level
# fenced code block; data-language attribute is preserved
# ---------------------------------------------------------------------------


def test_pre_at_top_level_not_regressed():
    md = _convert("<pre>line1\nline2</pre>")
    assert "\n```\nline1\nline2\n```\n" in md, repr(md)
    rendered = _render(md)
    assert rendered == "<pre><code>line1\nline2\n</code></pre>\n", repr(rendered)


def test_pre_at_top_level_with_language_attribute():
    md = _convert('<pre data-language="python">print("hi")</pre>')
    assert "\n```python\nprint(\"hi\")\n```\n" in md, repr(md)
    rendered = _render(md)
    assert 'class="language-python"' in rendered, repr(rendered)


def test_language_attribute_preserved_for_in_list_pre():
    md = _convert('<ul><li>x<pre data-language="bash">echo hi</pre></li></ul>')
    assert "```bash" in md, repr(md)
    rendered = _render(md)
    assert 'class="language-bash"' in rendered, repr(rendered)


# ---------------------------------------------------------------------------
# Edge cases that guard against accidental regressions in the prefix logic
# ---------------------------------------------------------------------------


def test_pre_with_empty_line_in_blockquote_keeps_blockquote_open():
    html = "<blockquote><pre>import os\n\nimport sys</pre></blockquote>"
    md = _gen_markdown(html)
    assert md == "> \n> ```\n> import os\n>\n> import sys\n> ```\n", repr(md)
    rendered = _render(md)
    assert rendered.count("<blockquote>") == 1, repr(rendered)
    assert "<code>import os\n\nimport sys" in rendered, repr(rendered)


def test_pre_with_leading_and_trailing_empty_lines_in_blockquote_keeps_open():
    html = "<blockquote><pre>\n\ncode\n\n</pre></blockquote>"
    md = _gen_markdown(html)
    assert "> ```\n>\n>\n> code\n>\n>\n> ```" in md, repr(md)
    assert "> ```\n>\n>\n> code\n\n\n> ```" not in md, repr(md)
    rendered = _render(md)
    assert rendered.count("<blockquote>") == 1, repr(rendered)
    assert "<pre><code>" in rendered and "code" in rendered, repr(rendered)


def test_pre_with_empty_line_in_nested_blockquote_keeps_both_open():
    html = "<blockquote><blockquote><pre>a\n\nb</pre></blockquote></blockquote>"
    md = _gen_markdown(html)
    assert ">> ```\n>> a\n>>\n>> b\n>> ```" in md, repr(md)
    assert ">> a\n\n>> b" not in md, repr(md)
    rendered = _render(md)
    assert rendered.count("<blockquote>") == 2, repr(rendered)
    assert "<code>a\n\nb" in rendered, repr(rendered)


def test_pre_with_empty_line_in_blockquote_plus_list_drops_list_indent_only():
    html = "<blockquote><ul><li><pre>a\n\nb</pre></li></ul></blockquote>"
    md = _gen_markdown(html)
    assert ">     ```\n>     a\n>\n>     b\n>     ```" in md, repr(md)
    assert ">     a\n\n>     b" not in md, repr(md)
    assert ">     \n" not in md, repr(md)
    rendered = _render(md)
    assert rendered.count("<blockquote>") == 1, repr(rendered)
    assert "<code>a\n\nb" in rendered, repr(rendered)


def test_pre_with_empty_line_in_list_indents_real_lines_only():
    md = _convert("<ul><li><pre>a\n\nb</pre></li></ul>")
    # Non-empty lines must be indented; the empty line between them must NOT
    # be prefixed (an empty "> " line would close the surrounding blockquote,
    # an empty "    " line would add trailing whitespace inside the code).
    assert "    a\n\n    b" in md, repr(md)
    assert "    a\n    \n    b" not in md, repr(md)
    rendered = _render(md)
    assert "a" in rendered and "b" in rendered
    assert "</li>\n<pre>" not in rendered and "</ul>\n<pre>" not in rendered, repr(rendered)


def test_code_inside_pre_in_list_indented_correctly():
    html = "<ul><li>x<pre><code>def foo():\n    return 1</code></pre></li></ul>"
    md = _convert(html)
    # The inner <code> tags are ignored (handle_code_in_pre=False default),
    # and the first non-blank code line must keep its 4-space indent.
    assert "    def foo():" in md, repr(md)
    rendered = _render(md)
    assert "<pre><code>" in rendered
    assert "</li>\n<pre>" not in rendered and "</ul>\n<pre>" not in rendered, repr(rendered)


# ---------------------------------------------------------------------------
# Production path: DefaultMarkdownGenerator.generate_markdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "html,needle_in_md,needle_in_html",
    [
        (
            "<ul><li>Step1 <pre>npm install</pre></li>"
            "<li>Step2 <pre>npm run dev</pre></li></ul>",
            "    ```\n    npm install\n    ```",
            "<pre><code>npm install\n</code></pre>",
        ),
        (
            "<blockquote><pre>quoted\ncode</pre></blockquote>",
            "> ```\n> quoted\n> code\n> ```",
            "<pre><code>quoted\ncode\n</code></pre>",
        ),
        (
            "<pre>line1\nline2</pre>",
            "\n```\nline1\nline2\n```\n",
            "<pre><code>line1\nline2\n</code></pre>",
        ),
        (
            '<pre data-language="python">print(1)</pre>',
            "\n```python\nprint(1)\n```\n",
            'class="language-python"',
        ),
    ],
    ids=["list-2item", "blockquote", "top-level", "data-language"],
)
def test_default_markdown_generator_keeps_pre_nested(
    html, needle_in_md, needle_in_html
):
    md = _gen_markdown(html)
    assert needle_in_md in md, repr(md)
    rendered = _render(md)
    assert needle_in_html in rendered, repr(rendered)


def test_default_markdown_generator_no_extra_strip_applied_to_indented_fences():
    """The post-conversion strip `md.replace("    ```", "```")` must NOT be
    applied at the DefaultMarkdownGenerator site, otherwise the in-list
    fences we emit have their 4-space indent stripped and the bug regresses."""
    md = _gen_markdown("<ul><li>x<pre>code</pre></li></ul>")
    assert "    ```\n    code\n    ```" in md, repr(md)


# ---------------------------------------------------------------------------
# Production path: crawl4ai.utils.get_content_of_website_optimized
# ---------------------------------------------------------------------------


def test_get_content_of_website_optimized_pre_in_list_stays_nested():
    html = _wrap_html(
        "<ul><li>Step1 <pre>npm install</pre></li>"
        "<li>Step2 <pre>npm run dev</pre></li></ul>"
    )
    res = get_content_of_website_optimized(url="http://test", html=html)
    md = res["markdown"]
    rendered = _render(md)
    assert "<li>" in rendered and "<pre><code>npm install" in rendered, repr(rendered)
    assert "<ul>\n<li>Step1</li>\n</ul>\n<pre>" not in rendered, repr(rendered)


# ---------------------------------------------------------------------------
# Internal contract: the subclass deliberately bypasses the base class's
# `self.pre` machinery; verify state invariants so a future maintainer does
# not accidentally re-engage it (which would re-introduce the bug).
# ---------------------------------------------------------------------------


def test_pre_handling_does_not_set_base_pre_flag_on_subclass():
    h = CustomHTML2Text()
    h.update_params(body_width=0, single_line_break=True)
    h.handle("<ul><li>x<pre>code</pre></li></ul>")
    assert h.pre is False
    assert h.inside_pre is False
    # The per-pre prefix must reset to None after </pre>.
    assert h._pre_prefix is None
    # Counters must return to baseline after closing tags.
    assert h.blockquote == 0
    assert h.list == []


# ---------------------------------------------------------------------------
# No regression: the base HTML2Text class (used by `python -m crawl4ai.html2text`
# CLI) is unchanged by this fix.
# ---------------------------------------------------------------------------


def test_base_html2text_pre_in_list_unchanged_shape():
    """The base class emits 4-space indented code (NOT fences) for <pre>;
    its output shape must not be changed by this fix (which only touches
    ``CustomHTML2Text``)."""
    md = _convert_base("<ul><li>x<pre>code</pre></li></ul>")
    rendered = _render(md)
    assert "x" in rendered
    assert "<li>" in rendered or "<pre>" in rendered


def test_cli_base_html2text_smoke_test():
    proc = subprocess.run(
        [sys.executable, "-m", "crawl4ai.html2text", "-b", "0"],
        input="<p>hello</p>",
        capture_output=True,
        text=True,
        check=True,
        env=_CLI_ENV,
    )
    assert "hello" in proc.stdout
