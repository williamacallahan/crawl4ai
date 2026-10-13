"""Regression tests for ``<br>`` inside ``<td>``/``<th>`` table cells.

Background
----------
``crawl4ai/html2text/__init__.py`` powers Markdown generation for the whole
project (via ``DefaultMarkdownGenerator`` → ``CustomHTML2Text``). The inherited
``<br>`` handler unconditionally emitted a Markdown hard line break (``"  \\n"``)
regardless of context. GFM tables require each ``<tr>`` to occupy a single
physical line, so a ``<br>`` inside a ``<td>``/``<th>`` split the cell across two
physical lines and produced structurally invalid GFM: a downstream
GFM-compliant parser (``markdown-it-py`` in commonmark+``table`` mode) re-parsed
a 3-row source table as a 4-row table with a phantom extra row and shifted
cells (the value after ``<br>`` became column 1 of the phantom row).

The fix tracks an ``in_table_cell`` depth counter (incremented on ``<td>``/``<th>``
start, decremented on their end, on the GFM/``pad_tables`` path — the
``ignore_tables``/``bypass_tables`` branches are untouched) and emits a
cell-safe raw inline ``<br>`` (GFM-permitted raw inline HTML) when inside a cell,
keeping each row on one physical line.

These tests confirm the fix and guard against regressions for ``<br>`` outside
cells (paragraphs and blockquotes) and for the other table modes.
"""

import os
import subprocess
import sys

import pytest

markdown_it = pytest.importorskip("markdown_it")

from crawl4ai.html2text import HTML2Text, CustomHTML2Text  # noqa: E402
from crawl4ai.markdown_generation_strategy import (  # noqa: E402
    DefaultMarkdownGenerator,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CLI_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH", "")])
    ),
}

# Realistic e-commerce spec table whose multi-value cell uses <br> to separate
# metric/imperial values (the case from the bug report).
_ECOMMERCE_HTML = (
    "<table>\n"
    "<tr><th>Attribute</th><th>Value</th></tr>\n"
    "<tr><td>Dimensions</td><td>10 x 5 x 2 cm<br>4 x 2 x 0.8 in</td></tr>\n"
    "<tr><td>Weight</td><td>120 g</td></tr>\n"
    "</table>"
)


def _convert_base(html: str) -> str:
    """Base ``HTML2Text`` conversion (GFM default path)."""
    h = HTML2Text()
    h.body_width = 0
    return h.handle(html)


def _convert_custom(html: str) -> str:
    """Direct ``CustomHTML2Text`` conversion with production-equivalent options."""
    h = CustomHTML2Text()
    h.update_params(
        body_width=0,
        ignore_links=False,
        single_line_break=True,
        mark_code=True,
    )
    return h.handle(html)


def _gen_markdown(html: str, base_url: str = "") -> str:
    return DefaultMarkdownGenerator().generate_markdown(
        html, base_url=base_url, citations=False
    ).raw_markdown


def _row_count(md: str) -> int:
    """Number of ``tr_open`` tokens when a GFM-compliant parser parses ``md``."""
    tokens = (
        markdown_it.MarkdownIt("commonmark").enable("table").parse(md)
    )
    return sum(1 for t in tokens if t.type == "tr_open")


# ---------------------------------------------------------------------------
# Core bug: <br> inside a table cell must not split the GFM row across lines
# ---------------------------------------------------------------------------


def test_br_in_td_emits_inline_br_not_hard_break():
    md = _convert_base(_ECOMMERCE_HTML)
    # The multi-value cell stays on one physical line as raw inline <br>...
    assert "10 x 5 x 2 cm<br>4 x 2 x 0.8 in" in md, repr(md)
    # ...and is NOT split by a hard line break (which would corrupt the row).
    assert "10 x 5 x 2 cm  \n4 x 2 x 0.8 in" not in md, repr(md)


def test_br_in_td_preserves_gfm_row_structure():
    md = _convert_base(_ECOMMERCE_HTML)
    # The 3-row source table must re-parse as exactly 3 rows (header + 2 data),
    # not 4 (the phantom-extra-row bug).
    assert _row_count(md) == 3, repr(md)


def test_br_in_th_keeps_header_on_single_line():
    html = (
        "<table><tr><th>Apple production<br>2022, millions of tonnes</th>"
        "<th>Share</th></tr>"
        "<tr><td>China</td><td>50%</td></tr></table>"
    )
    md = _convert_base(html)
    header_lines = [
        ln for ln in md.split("\n") if "Apple production" in ln and "Share" in ln
    ]
    assert len(header_lines) == 1, repr(md)
    assert "Apple production<br>2022, millions of tonnes" in header_lines[0], repr(md)
    assert _row_count(md) == 2, repr(md)


def test_br_in_cell_via_default_markdown_generator():
    # The production path used by AsyncWebCrawler.
    md = _gen_markdown(_ECOMMERCE_HTML, base_url="https://example.com/")
    assert "10 x 5 x 2 cm<br>4 x 2 x 0.8 in" in md, repr(md)
    assert _row_count(md) == 3, repr(md)


def test_custom_html2text_br_in_cell_inline():
    md = _convert_custom(_ECOMMERCE_HTML)
    assert "10 x 5 x 2 cm<br>4 x 2 x 0.8 in" in md, repr(md)
    assert "10 x 5 x 2 cm  \n4 x 2 x 0.8 in" not in md, repr(md)


# ---------------------------------------------------------------------------
# Regression guards: <br> outside a cell must keep the old hard-line behavior
# ---------------------------------------------------------------------------


def test_br_in_paragraph_still_hard_break():
    md = _convert_base("<p>line one<br>line two</p>")
    assert "line one  \nline two" in md, repr(md)
    assert "line one<br>line two" not in md, repr(md)


def test_br_in_blockquote_still_uses_quote_prefix():
    md = _convert_base("<blockquote>quoted one<br>quoted two</blockquote>")
    assert "  \n> " in md, repr(md)
    assert "quoted one<br>quoted two" not in md, repr(md)


def test_in_table_cell_counter_returns_to_zero_after_table():
    """After a table with <br> in a cell closes, a later <br> in a paragraph
    reverts to the hard-line-break behavior (no state leak across tables)."""
    html = (
        "<table><tr><th>A</th></tr><tr><td>x<br>y</td></tr></table>"
        "<p>after1<br>after2</p>"
    )
    md = _convert_base(html)
    assert "x<br>y" in md, repr(md)
    assert "after1  \nafter2" in md, repr(md)
    assert "after1<br>after2" not in md, repr(md)


# ---------------------------------------------------------------------------
# Table-mode guards: the fix is scoped to the GFM/`pad_tables` path
# ---------------------------------------------------------------------------


def test_br_in_cell_pad_tables_mode_single_line():
    html = (
        "<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>Dimensions</td><td>10 x 5 x 2 cm<br>4 x 2 x 0.8 in</td></tr>"
        "<tr><td>Weight</td><td>120 g</td></tr></table>"
    )
    h = HTML2Text()
    h.body_width = 0
    h.pad_tables = True
    md = h.handle(html)
    rows = [ln for ln in md.split("\n") if ln.strip().startswith("|")]
    assert any("10 x 5 x 2 cm<br>4 x 2 x 0.8 in" in ln for ln in rows), repr(md)
    # The <br> remnant must not create a phantom extra row.
    assert md.count("4 x 2 x 0.8 in") == 1, repr(md)
    # Padded (non-separator) rows must remain width-aligned.
    widths = {len(ln) for ln in rows if "---" not in ln}
    assert len(widths) == 1, f"padded rows not aligned: {widths}"


def test_br_in_cell_bypass_tables_unchanged():
    """bypass_tables emits raw HTML; the fix must not inject GFM pipes there."""
    html = (
        "<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>x<br>y</td><td>z</td></tr></table>"
    )
    h = HTML2Text()
    h.body_width = 0
    h.bypass_tables = True
    md = h.handle(html)
    assert "<table" in md and "<td" in md, repr(md)
    assert "| ---" not in md, repr(md)


def test_nested_table_inner_br_is_inline():
    """The in_table_cell depth counter handles nesting: a <br> in an inner
    cell is still emitted inline even while an outer cell is open."""
    html = (
        "<table><tr><th>Outer</th><th>Inner</th></tr>"
        "<tr><td>a</td><td>"
        "<table><tr><td>i1<br>i2</td><td>j</td></tr></table>"
        "</td></tr></table>"
    )
    md = _convert_base(html)
    assert "i1<br>i2" in md, repr(md)
    assert "i1  \ni2" not in md, repr(md)
