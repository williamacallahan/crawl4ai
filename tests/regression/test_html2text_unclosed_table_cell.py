"""Regression tests for the ``in_table_cell`` counter after a table with
omitted (legally HTML-spec) ``</td>``/``</th>`` end tags.

``crawl4ai/html2text/__init__.py`` keeps an ``in_table_cell`` depth counter so
``<br>`` inside a GFM cell is emitted as a raw inline ``<br>`` (keeping each
``<tr>`` on one physical line). The counter was decremented only on explicit
``</td>``/``</th>`` end tags; since ``html.parser`` never synthesizes implied
end tags, any legally-omitted ``</td>`` left the counter elevated, leaking
literal ``<br>`` into paragraphs and blockquotes (and shadowing the blockquote
branch). Via ``DefaultMarkdownGenerator(content_source="raw_html")`` this
reached end-to-end conversion. The fix snapshots ``in_table_cell`` per
``<table>`` on a stack and restores it on ``</table>`` — restoring (not
zeroing) so the nested-table case stays correct: after an inner ``</table>``
the counter returns to the enclosing cell's depth. The well-formed (balanced
end-tag) case is covered by ``test_html2text_br_in_table_cell.py``.
"""

import pytest

from crawl4ai.html2text import HTML2Text  # noqa: E402
from crawl4ai.markdown_generation_strategy import (  # noqa: E402
    DefaultMarkdownGenerator,
)


# --- Conversion entry points ------------------------------------------------


def _convert_base(html):
    h = HTML2Text()
    h.body_width = 0
    return h.handle(html)


def _gen_raw_markdown(html, base_url=""):
    # ``content_source="raw_html"`` is the documented option that bypasses
    # lxml normalization (the path the bug report identifies for end-to-end
    # reach).
    return DefaultMarkdownGenerator(
        content_source="raw_html"
    ).generate_markdown(
        html, base_url=base_url, citations=False
    ).raw_markdown


class _Tracing(HTML2Text):
    """Snapshots ``in_table_cell`` after each ``</table>`` end tag."""

    def __init__(self):
        super().__init__()
        self.in_cell_after_table_close: list[int] = []

    def handle_tag(self, tag, attrs, start):
        super().handle_tag(tag, attrs, start)
        if tag == "table" and not start:
            self.in_cell_after_table_close.append(self.in_table_cell)


# --- Core bug: omitted end tag must not leak <br> past the table ------------


def test_unclosed_td_then_br_in_paragraph_uses_hard_break():
    md = _convert_base(
        "<table><tr><td>a<tr><td>b</table><p>line1<br>line2</p>"
    )
    assert "line1  \nline2" in md, repr(md)
    assert "line1<br>line2" not in md, repr(md)


def test_unclosed_td_then_br_in_blockquote_uses_quote_prefix():
    md = _convert_base(
        "<table><tr><td>a<tr><td>b</table>"
        "<blockquote>q1<br>q2</blockquote>"
    )
    # The blockquote branch ("  \n> ") must not be shadowed by the leaked
    # raw-<br> branch.
    assert "q1  \n> q2" in md, repr(md)
    assert "q1<br>q2" not in md, repr(md)


@pytest.mark.parametrize("cell_tag", ["td", "th"])
def test_unclosed_cell_tag_then_br_in_prose_uses_hard_break(cell_tag):
    md = _convert_base(
        "<table><tr><%s>h<tr><%s>i</table><p>L1<br>L2</p>"
        % (cell_tag, cell_tag)
    )
    assert "L1  \nL2" in md, repr(md)
    assert "L1<br>L2" not in md, repr(md)


def test_unclosed_cells_in_multiple_rows_then_br_in_prose():
    # Guards against the counter accumulating across rows of an unclosed
    # table and staying elevated past the table boundary.
    md = _convert_base(
        "<table><tr><td>a<tr><td>b<tr><td>c</table><p>p1<br>p2</p>"
    )
    assert "p1  \np2" in md, repr(md)
    assert "p1<br>p2" not in md, repr(md)


def test_unclosed_cell_with_br_inside_cell_still_inline():
    # Over-correction guard: a <br> inside an (still-open) cell stays inline;
    # only the post-table <br> reverts to a hard break. A future "always
    # reset in_table_cell on the next <tr>" change would break this.
    md = _convert_base(
        "<table><tr><td>a<br>x<tr><td>b</table><p>p1<br>p2</p>"
    )
    assert "a<br>x" in md, repr(md)
    assert "p1  \np2" in md, repr(md)
    assert "p1<br>p2" not in md, repr(md)


# --- in_table_cell counter invariants ---------------------------------------


def test_in_table_cell_zero_after_table_with_unclosed_td():
    h = HTML2Text()
    h.body_width = 0
    h.handle("<table><tr><td>a<tr><td>b</table><p>x<br>y</p>")
    assert h.in_table_cell == 0, h.in_table_cell
    assert h._in_table_cell_stack == [], h._in_table_cell_stack


def test_tracing_counter_restored_to_outer_depth_after_inner_table():
    # The snapshot-stack variant's invariant: after the inner </table> the
    # counter must be 1 (the outer cell is still open); after the outer
    # </table> it must be 0. A naive ``self.in_table_cell = 0`` reset on
    # </table> would leave 0 after the inner close and corrupt the outer
    # cell's <br> handling.
    h = _Tracing()
    h.body_width = 0
    h.handle(
        "<table><tr><td>a"
        "<table><tr><td>i1<br>i2</td><td>j</td></tr></table>"
        "after<br>cell</td></tr></table>"
    )
    assert h.in_cell_after_table_close == [1, 0], (
        h.in_cell_after_table_close
    )
    assert h.in_table_cell == 0
    assert h._in_table_cell_stack == []


def test_stray_table_end_tag_does_not_hijack_later_br():
    # A </table> with no matching <table> must not leave the counter
    # negative or elevated; later <br>s in prose must be hard breaks.
    md = _convert_base("<table><tr><td>a<td>cell</table><p>x<br>y</p>")
    assert "x  \ny" in md, repr(md)
    assert "x<br>y" not in md, repr(md)


# --- Nested-table robustness (requires the snapshot stack, not naive reset)


def test_br_in_outer_cell_after_inner_table_still_inline():
    # A <br> between the inner </table> and the outer </td> is still inside
    # the outer cell and must emit an inline <br>, not a hard break. A naive
    # reset of in_table_cell to 0 on </table> would corrupt this case.
    md = _convert_base(
        "<table><tr><th>Outer</th><th>Inner</th></tr>"
        "<tr><td>a</td><td>"
        "<table><tr><td>i1<br>i2</td><td>j</td></tr></table>"
        "after<br>cell"
        "</td></tr></table>"
    )
    assert "after<br>cell" in md, repr(md)
    assert "after  \ncell" not in md, repr(md)


def test_nested_table_with_unclosed_inner_cell_does_not_leak_outer():
    # The inner </table> must restore the counter to the *outer* cell's
    # depth — neither leak past the outer cell nor zero the counter (which
    # would corrupt the outer cell), and the post-outer-</table> <br> must
    # be a hard break.
    md = _convert_base(
        "<table><tr><td>outer-before"
        "<table><tr><td>inner1<tr><td>inner2</table>"
        "outer-after<br>cell"
        "</td></tr></table>"
        "<p>after-table1<br>after-table2</p>"
    )
    assert "outer-after<br>cell" in md, repr(md)
    assert "after-table1  \nafter-table2" in md, repr(md)
    assert "after-table1<br>after-table2" not in md, repr(md)


# --- Production path (the bug report's end-to-end reach) -------------------


def test_default_markdown_generator_raw_html_unclosed_cell_no_leak():
    # ``content_source="raw_html"`` is the documented option that bypasses
    # lxml normalization and feeds un-normalized HTML straight to the
    # converter — the path the bug report identifies for end-to-end reach.
    md = _gen_raw_markdown(
        "<table><tr><th>Attribute</th><th>Value</th></tr>"
        "<tr><td>Weight</td><td>120 g</td></tr>"
        "<tr><td>Dimensions</td><td>10 x 5 x 2 cm<br>4 x 2 x 0.8 in"
        "</table>"
        "<p>line one<br>line two</p>",
        base_url="https://example.com/",
    )
    assert "line one  \nline two" in md, repr(md)
    assert "line one<br>line two" not in md, repr(md)
    assert "10 x 5 x 2 cm<br>4 x 2 x 0.8 in" in md, repr(md)


# --- Table-mode scoping guards (the fix touches the GFM / pad_tables path) ---


def test_bypass_tables_mode_unclosed_table_does_not_break_output():
    # ``bypass_tables`` emits raw HTML and never touches in_table_cell; the
    # fix's snapshot stack never enters this code path, so an unclosed cell
    # must not crash and a later <br> in a paragraph still emits a hard
    # break. Guards a future "let me also push to the snapshot stack here"
    # mistake.
    h = HTML2Text()
    h.body_width = 0
    h.bypass_tables = True
    md = h.handle(
        "<table><tr><td>a<tr><td>b</table><p>line1<br>line2</p>"
    )
    assert "<table" in md, repr(md)
    assert "line1  \nline2" in md, repr(md)
    assert "line1<br>line2" not in md, repr(md)
    assert h.in_table_cell == 0, h.in_table_cell


def test_ignore_tables_mode_unclosed_table_does_not_break_output():
    h = HTML2Text()
    h.body_width = 0
    h.ignore_tables = True
    md = h.handle(
        "<table><tr><td>a<tr><td>b</table><p>line1<br>line2</p>"
    )
    assert "line1  \nline2" in md, repr(md)
    assert "line1<br>line2" not in md, repr(md)
    assert h.in_table_cell == 0, h.in_table_cell


# --- No-regression on the previously-fixed well-formed nested case ----------


def test_well_formed_nested_table_inner_br_inline():
    md = _convert_base(
        "<table><tr><th>Outer</th><th>Inner</th></tr>"
        "<tr><td>a</td><td>"
        "<table><tr><td>i1<br>i2</td><td>j</td></tr></table>"
        "</td></tr></table>"
    )
    assert "i1<br>i2" in md, repr(md)
    assert "i1  \ni2" not in md, repr(md)
