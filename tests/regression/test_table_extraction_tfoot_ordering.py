"""Direct regression tests for DefaultTableExtraction ``<tfoot>`` ordering.

These guard against ``extract_table_data`` emitting ``<tfoot>`` rows in
source (document) order rather than in the HTML table-model order
(``<thead>`` -> body rows -> ``<tfoot>``). The XPath union
``./tr | ./thead/tr | ./tbody/tr | ./tfoot/tr`` returns matches in document
order, and lxml preserves the source position of ``<tfoot>``, so a
``<tfoot>`` placed before ``<tbody>`` (valid HTML4/HTML5; HTML4.01 Sec11.2.1
specified the table content model as ``CAPTION?, COLGROUP*, THEAD?, TFOOT?,
TBODY+``) was emitted at the TOP of ``rows`` instead of at the end.

The same document-order bug let a leading ``<tfoot>`` row be adopted as
``headers`` in the ``<thead>``-less fallback path
(``first_row = table.xpath("./tr | ./tbody/tr | ./tfoot/tr")``), silently
replacing the column headers with the footer/totals row.

The WHATWG HTML Living Standard ``table.rows`` getter defines the canonical
ordering: ``<thead>`` rows first, then body rows (direct ``<tr>`` and
``<tbody>``) in tree order, then ``<tfoot>`` rows in tree order -- regardless
of source position. The fix iterates the non-footer row groups in their
existing document order -- preserving the ``current_group`` /
``pending.clear()`` group-transition semantics the rowspan carry-down
regression suite depends on -- then appends ``<tfoot>`` rows, and excludes
``<tfoot>`` from the ``<thead>``-less header fallback. These tests run on
real lxml-parsed HTML and need no browser or network.
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


def test_tfoot_before_tbody_emitted_after_body_rows():
    # The exact reproduction from the bug report: <thead> + <tfoot> placed
    # BEFORE <tbody> (valid HTML4/HTML5). Under the bug, the footer row was
    # emitted at the TOP of rows; the HTML table model requires it last.
    table = (
        "<table><caption>Quarterly Results</caption>"
        "<thead><tr><th>Region</th><th>Sales</th></tr></thead>"
        "<tfoot><tr><td>Total</td><td>1,000</td></tr></tfoot>"
        "<tbody><tr><td>North</td><td>400</td></tr>"
        "<tr><td>South</td><td>600</td></tr></tbody></table>"
    )
    is_data_table, data = _extract(table)

    assert is_data_table is True
    assert data["headers"] == ["Region", "Sales"]
    assert data["rows"] == [
        ["North", "400"],
        ["South", "600"],
        ["Total", "1,000"],
    ]
    # The footer must NOT be the first data row (the bug put it at index 0).
    assert data["rows"][0] != ["Total", "1,000"]
    assert data["rows"][-1] == ["Total", "1,000"]


def test_tfoot_before_tbody_end_to_end_via_extract_tables():
    # The production path: content_scraping_strategy.py populates
    # media["tables"] via the public extract_tables() entry point. The
    # tfoot-before-tbody table must pass is_data_table and be returned
    # exactly once with the footer emitted after the body rows.
    table_html = (
        "<table><caption>Quarterly Results</caption>"
        "<thead><tr><th>Region</th><th>Sales</th></tr></thead>"
        "<tfoot><tr><td>Total</td><td>1,000</td></tr></tfoot>"
        "<tbody><tr><td>North</td><td>400</td></tr>"
        "<tr><td>South</td><td>600</td></tr></tbody></table>"
    )
    tables = _extract_all(table_html)

    assert len(tables) == 1
    data = tables[0]
    assert data["headers"] == ["Region", "Sales"]
    assert data["rows"] == [
        ["North", "400"],
        ["South", "600"],
        ["Total", "1,000"],
    ]
    assert data["caption"] == "Quarterly Results"
    assert data["metadata"]["has_headers"] is True


def test_tfoot_with_th_before_tbody_not_adopted_as_headers():
    # The <thead>-less header fallback: when no <thead> is present, the first
    # body row may be adopted as headers ONLY if it is all <th>. A <tfoot>
    # row (even an all-<th> one) placed before <tbody> used to become
    # first_row[0] via the "./tr | ./tbody/tr | ./tfoot/tr" document-order
    # union, promoting the footer to column headers. <tfoot> must be
    # excluded: a footer is never a column header.
    table = (
        "<table><caption>c</caption>"
        "<tfoot><tr><th>Total</th><th>1,000</th></tr></tfoot>"
        "<tbody><tr><td>North</td><td>400</td></tr>"
        "<tr><td>South</td><td>600</td></tr></tbody></table>"
    )
    _, data = _extract(table)

    # The footer (all-<th>) must NOT have been promoted to headers.
    assert data["headers"] != ["Total", "1,000"]
    assert data["headers"] == ["Column 1", "Column 2"]
    assert data["metadata"]["has_headers"] is False


def test_body_rowspan_does_not_carry_into_tfoot():
    # A <tfoot> placed before a <tbody> whose first cell carries rowspan.
    # The transition into <tfoot> must still clear the pending rowspan
    # grid (existing group-boundary semantics, preserved by the fix), so a
    # body rowspan cannot bleed up into the reordered footer. The footer's
    # own <td> values land in their own columns, not under a carried-down
    # body span. This is the interaction the bug report warned a naive
    # ``./tr + ./thead/tr + ./tbody/tr + ./tfoot/tr`` concat would disturb.
    table = (
        "<table><caption>c</caption>"
        "<thead><tr><th>A</th><th>B</th></tr></thead>"
        "<tfoot><tr><td>FT</td><td>n1</td></tr></tfoot>"
        '<tbody><tr><td rowspan="5">BIG</td><td>1</td></tr>'
        "<tr><td>2</td></tr><tr><td>3</td></tr></tbody></table>"
    )
    _, data = _extract(table)

    # The body rowspan covers only the 3 continuation rows in the same
    # <tbody>; the remaining 2 slots of rowspan="5" are dropped by the group
    # boundary, and the footer row is NOT prefixed with "BIG".
    assert data["rows"] == [
        ["BIG", "1"],
        ["BIG", "2"],
        ["BIG", "3"],
        ["FT", "n1"],
    ]
    assert data["rows"][-1][0] == "FT"


def test_tfoot_before_bare_direct_rows_emitted_last():
    # A <thead>-less table with bare <tr> body children and a <tfoot>
    # placed before them. Bare <tr> children form an implicit <tbody> that
    # the table model places between <thead> and <tfoot>; the footer must
    # follow them even when it appears first in source.
    table = (
        "<table><caption>c</caption>"
        "<tfoot><tr><td>Footer</td><td>row</td></tr></tfoot>"
        "<tr><td>r1a</td><td>r1b</td></tr>"
        "<tr><td>r2a</td><td>r2b</td></tr>"
        "</table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"] == [
        ["r1a", "r1b"],
        ["r2a", "r2b"],
        ["Footer", "row"],
    ]
    assert data["rows"][-1] == ["Footer", "row"]
