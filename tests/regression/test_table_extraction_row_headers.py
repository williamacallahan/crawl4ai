"""Direct regression tests for DefaultTableExtraction row-header handling.

These guard against ``<th scope="row">`` row-header cells in ``<tbody>`` being
dropped when a ``<thead>`` is present. The body-row grid builder iterated only
``./td`` cells, so for a table using the HTML Living Standard §4.9.1.1
"Techniques for describing tables" row-header pattern, every row's leading
``<th>`` label vanished and the surviving ``<td>`` values left-shifted under
the wrong headers, padded with a fabricated trailing ``""``. The table's own
``is_data_table`` scorer counted ``<th>`` as a cell (``./td|./th``), so the
asymmetry silently reached ``media["tables"]`` verbatim via
``content_scraping_strategy.py``.

The fix selects ``./td|./th`` when a ``<thead>`` is present and keeps ``./td``
on the deliberately-settled headerless path (pinned by
``test_no_thead_mixed_th_td_first_row_keeps_all_columns``). These tests run on
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


def test_thead_with_th_row_headers_preserves_labels():
    # The spec §4.9.1.1 "Techniques for describing tables" canonical layout:
    # <thead> column headers + <th scope="row"> row labels in <tbody>. Each row
    # label must occupy the row's first grid column and each <td> value must
    # land under its correct header (previously the labels were dropped and
    # every value left-shifted under the wrong header with a trailing "").
    table = (
        "<table><caption>Characteristics with positive and negative sides</caption>"
        "<thead><tr><th>Characteristic</th><th>Negative</th><th>Positive</th></tr></thead>"
        "<tbody>"
        "<tr><th>Mood</th><td>Sad</td><td>Happy</td></tr>"
        "<tr><th>Grade</th><td>Failing</td><td>Passing</td></tr>"
        "</tbody></table>"
    )
    is_data_table, data = _extract(table)

    assert is_data_table is True
    assert data["headers"] == ["Characteristic", "Negative", "Positive"]
    assert data["rows"] == [
        ["Mood", "Sad", "Happy"],
        ["Grade", "Failing", "Passing"],
    ]
    # Row labels must not vanish and rows must not be left-shifted or padded.
    assert all(row[0] != "" for row in data["rows"])
    assert all(row[-1] != "" for row in data["rows"])


def test_th_row_headers_emitted_end_to_end_via_extract_tables():
    # The production path: content_scraping_strategy.py populates
    # media["tables"] via the public extract_tables() entry point. The
    # row-header table must pass is_data_table and be returned exactly once
    # with the labels intact.
    table_html = (
        "<table><caption>Population</caption>"
        "<thead><tr><th>City</th><th>Population</th></tr></thead><tbody>"
        '<tr><th scope="row">Metropolis</th><td>1,000,000</td></tr>'
        '<tr><th scope="row">Gotham</th><td>500,000</td></tr>'
        "</tbody></table>"
    )
    tables = _extract_all(table_html)

    assert len(tables) == 1
    data = tables[0]
    assert data["headers"] == ["City", "Population"]
    assert data["rows"] == [
        ["Metropolis", "1,000,000"],
        ["Gotham", "500,000"],
    ]
    assert data["metadata"]["has_headers"] is True
    assert all(row[0] != "" for row in data["rows"])


def test_th_rowheader_with_rowspan_carries_down_correctly():
    # A row-header <th> carrying rowspan must carry its label down into the
    # continuation rows it occupies (HTML table model), exactly like a <td>
    # rowspan. Only the label carry-down this fix owns is asserted; the body
    # grid here is wider than <thead>, so trailing-column width is governed by
    # the separate max_columns=len(headers) behavior and is intentionally not
    # pinned.
    table = (
        "<table><caption>c</caption>"
        "<thead><tr><th>Group</th><th>Metric</th><th>Value</th></tr></thead><tbody>"
        '<tr><th rowspan="2">MegaCorp</th>'
        '<th scope="row">Bought</th><td>1</td><td>0</td></tr>'
        '<tr><th scope="row">Sold</th><td>3</td><td>2</td></tr>'
        "</tbody></table>"
    )
    _, data = _extract(table)

    rows = data["rows"]
    # The <th rowspan="2"> group label carries down to the continuation row.
    assert [r[0] for r in rows] == ["MegaCorp", "MegaCorp"]
    # The <th scope="row"> sub-labels are preserved (previously dropped +
    # left-shifted into the group label's column).
    assert [r[1] for r in rows] == ["Bought", "Sold"]


def test_scope_rowgroup_multi_tbody_recovers_all_labels():
    # The spec's §4.9.10 scope=rowgroup example: <th scope="rowgroup"
    # rowspan="2"> groups across two <tbody> row groups. The body-loop fix
    # recovers MegaCorp/MiniCorp (rowgroup) and Bought/Sold (row) labels with
    # correct rowspan carry-down across each row group. Only the label
    # recovery this fix owns is asserted; the trailing-column width cap set by
    # max_columns=len(headers) is a separate pre-existing issue and is
    # intentionally not pinned.
    table = (
        "<table>"
        "<thead><tr><th>&nbsp;</th>"
        '<th scope="col">Home starter</th>'
        '<th scope="col">Home brew</th>'
        '<th scope="col">Home builder</th></tr></thead>'
        "<tbody>"
        '<tr><th scope="rowgroup" rowspan="2">MegaCorp</th>'
        '<th scope="row">Bought</th><td>1</td><td>0</td><td>0</td></tr>'
        '<tr><th scope="row">Sold</th><td>3</td><td>2</td><td>5</td></tr>'
        "</tbody>"
        "<tbody>"
        '<tr><th scope="rowgroup" rowspan="2">MiniCorp</th>'
        '<th scope="row">Bought</th><td>4</td><td>12</td><td>0</td></tr>'
        '<tr><th scope="row">Sold</th><td>4</td><td>2</td><td>0</td></tr>'
        "</tbody></table>"
    )
    _, data = _extract(table)

    rows = data["rows"]
    # scope=rowgroup labels carry down within each <tbody> row group.
    assert [r[0] for r in rows] == ["MegaCorp", "MegaCorp", "MiniCorp", "MiniCorp"]
    # scope=row sub-labels are preserved (previously dropped + left-shifted).
    assert [r[1] for r in rows] == ["Bought", "Sold", "Bought", "Sold"]
