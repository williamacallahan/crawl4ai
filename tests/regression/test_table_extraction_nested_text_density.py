"""Regression tests for DefaultTableExtraction text-density scoring under
nested-subtable load.

Commit 3b2c79b rescoped the text-density numerator (``total_text``) via the
depth-scoped ``_cell_text`` helper so nested-subtable text no longer bleeds
into the outer table's score, but left the denominator as
``total_tags = sum(1 for _ in table.iterdescendants())``, which still counts
every descendant including tags inside nested subtables. The resulting
``text_ratio`` became a ratio of two differently-scoped quantities: a tag-heavy
nested subtable inflated the denominator while the numerator stayed scoped,
collapsing the ratio below the ``> 10`` / ``> 20`` text-bonus thresholds and
silently dropping otherwise-admissible data tables from ``extract_tables()``
(the only site populating ``media["tables"]``).

These tests pin the symmetric scoping of numerator and denominator (both count
only content owned by the table currently being extracted, via
``count(ancestor::table) = table_depth``) and guard against the asymmetry
regressing. They run on real lxml-parsed HTML and need no browser or network.
"""

from lxml import html

from crawl4ai.table_extraction import DefaultTableExtraction


def _extract(table_html, table_score_threshold=7):
    """Parse a <table> fragment and run the default extraction strategy."""
    root = html.fromstring("<html><body>" + table_html + "</body></html>")
    table = root.xpath(".//table")[0]
    strategy = DefaultTableExtraction(table_score_threshold=table_score_threshold)
    is_data_table = strategy.is_data_table(table, table_score_threshold=table_score_threshold)
    data = strategy.extract_table_data(table)
    return is_data_table, data


def _extract_all(table_html, table_score_threshold=7):
    """End-to-end via the public extract_tables() entry point."""
    root = html.fromstring("<html><body>" + table_html + "</body></html>")
    strategy = DefaultTableExtraction(table_score_threshold=table_score_threshold)
    return strategy.extract_tables(root)


def _denominator_counts(table_html):
    """(scoped, unscoped) tag counts for the first table in a fragment.

    The scoped count mirrors the production XPath
    ``.//*[count(ancestor::table) = table_depth]``; the unscoped count mirrors
    the pre-fix ``iterdescendants()`` total (which also yields comment nodes).
    Both are derived here independently so the tests cross-check the two
    definitions against each other.
    """
    root = html.fromstring("<html><body>" + table_html + "</body></html>")
    table = root.xpath(".//table")[0]
    table_depth = len(table.xpath("ancestor::table")) + 1
    scoped = len(table.xpath(f".//*[count(ancestor::table) = {table_depth}]"))
    return scoped, sum(1 for _ in table.iterdescendants())


def _browser_base5_table(own_len):
    """A browser-serialized shape: <tbody> present, <thead>/<caption> absent,
    first <tbody> row uses <th>, one cell embeds a 2x5 (10-cell) nested
    subtable. Structural base score is 5 (below the default threshold 7);
    admission depends on the text-bonus awarded by the scoped text_ratio."""
    t = "t" * own_len
    return (
        "<table><tbody>"
        '<tr><th>H0 hhhhhhhh</th><th>H1 hhhhhhhh</th><th>H2 hhhhhhhh</th>'
        '<th>H3 hhhhhhhh</th><th>H4 hhhhhhhh</th></tr>'
        f'<tr><td>D0 {t}</td><td>D1 {t}</td><td>D2 {t}</td>'
        f'<td>D3 {t}</td><td>D4 {t}</td></tr>'
        "<tr><td><table>"
        '<tr><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr>'
        '<tr><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr>'
        "</table> tail</td><td>d0</td><td>d1</td><td>d2</td><td>d3</td></tr>"
        "</tbody></table>"
    )


def test_nested_subtable_in_base5_band_passes_threshold():
    # The primary reproduction: under the buggy (asymmetrically-scoped)
    # denominator the 10 nested-subtable tags inflated the denominator
    # (text_ratio 282/32 = 8.81 <= 10, +0 bonus -> score 5 < 7 -> DROPPED).
    # With the symmetric scoping the nested tags are excluded
    # (text_ratio 282/20 = 14.10 > 10, +2 bonus -> score 7 >= 7 -> admitted).
    tables = _extract_all(_browser_base5_table(40))

    assert len(tables) == 1
    is_data_table, data = _extract(_browser_base5_table(40))
    assert is_data_table is True
    assert data["headers"] == [
        "H0 hhhhhhhh", "H1 hhhhhhhh", "H2 hhhhhhhh",
        "H3 hhhhhhhh", "H4 hhhhhhhh",
    ]
    # Nested-subtable text stays decontaminated: only "tail" survives from
    # the cell that hosted the nested table.
    assert data["rows"][1] == ["tail", "d0", "d1", "d2", "d3"]


def test_scoped_denominator_excludes_nested_subtable_descendants():
    # The denominator must be scoped symmetrically with the _cell_text
    # numerator: tags *inside* a nested subtable belong to that subtable's
    # own extraction and must NOT inflate the outer table's total_tags. The
    # nested <table> element itself sits at the outer table's depth (its
    # only table ancestor is the outer table), so it is correctly counted
    # once; its descendants are excluded.
    nested = (
        "<table><tbody>"
        "<tr><th>H0</th><th>H1</th></tr>"
        "<tr><td><table><tr><td>x</td><td>y</td></tr>"
        "<tr><td>z</td><td>w</td></tr></table> tail</td><td>d0</td></tr>"
        "</tbody></table>"
    )
    scoped, everything = _denominator_counts(nested)

    assert scoped < everything
    # excluded = the nested subtable's 2 <tr> + 4 <td> descendants (6 tags).
    assert scoped == everything - 6


def test_scoped_denominator_matches_unscoped_without_nested_subtable():
    # When no nested <table> exists, the depth-scoped XPath
    # `.//*[count(ancestor::table) = table_depth]` matches every
    # element descendant. iterdescendants() additionally yields comment
    # nodes, so on comment-bearing non-nested tables the scoped
    # denominator is smaller by the comment count - a slight loosening
    # toward admission, never a rejection. On comment-free non-nested
    # tables the fix is a no-op: text_ratio and score are unchanged.
    non_nested = (
        "<table><tbody>"
        "<tr><th>H0</th><th>H1</th></tr>"
        "<tr><td>a</td><td>b</td></tr>"
        "</tbody></table>"
    )
    scoped, everything = _denominator_counts(non_nested)

    assert scoped == everything

    # One in-table comment node: iterdescendants() counts it, the XPath
    # does not, so the scoped denominator is exactly one smaller. Pinned
    # so the divergence is known behavior, not an accident.
    commented = (
        "<table><tbody>"
        "<tr><th>H0</th><!-- c --><th>H1</th></tr>"
        "<tr><td>a</td><td>b</td></tr>"
        "</tbody></table>"
    )
    scoped, everything = _denominator_counts(commented)

    assert scoped == everything - 1


def test_low_text_density_nested_table_still_rejected():
    # No false positives: a sparse data table with a tag-heavy nested
    # subtable whose scoped text_ratio stays <= 10 must STILL be rejected
    # at the default threshold. The fix narrows the denominator but does
    # not inflate the numerator, so a genuinely low-density table remains
    # below the bonus cutoff.
    table = (
        "<table><tbody>"
        '<tr><th>H0 hhhhhhhh</th><th>H1 hhhhhhhh</th><th>H2 hhhhhhhh</th>'
        '<th>H3 hhhhhhhh</th><th>H4 hhhhhhhh</th></tr>'
        '<tr><td>D0 tttttttttt</td><td>D1 tttttttttt</td><td>D2 tttttttttt</td>'
        '<td>D3 tttttttttt</td><td>D4 tttttttttt</td></tr>'
        "<tr><td><table>"
        '<tr><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr>'
        '<tr><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr>'
        "</table></td><td>d0</td><td>d1</td><td>d2</td><td>d3</td></tr>"
        "</tbody></table>"
    )
    is_data_table, _ = _extract(table)

    assert is_data_table is False
    assert _extract_all(table) == []
