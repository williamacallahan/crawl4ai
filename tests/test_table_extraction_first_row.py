"""Direct regression tests for DefaultTableExtraction header handling.

These guard against the first row of a headerless (<thead>-less) table being
duplicated in both ``headers`` and ``rows`` — a regression introduced in 9d69fce.
They run on real lxml-parsed HTML and need no browser or network.
"""

import pytest
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


def test_no_thead_first_row_td_not_duplicated_in_headers_and_rows():
    # The reported bug: no <thead>, first row uses <td> -> must appear once,
    # only as data, never as headers.
    table = (
        "<table><caption>Sales Data</caption><tbody>"
        "<tr><td>Product</td><td>Q1</td><td>Q2</td></tr>"
        "<tr><td>Widget A</td><td>100</td><td>150</td></tr>"
        "</tbody></table>"
    )
    is_data_table, data = _extract(table)

    assert is_data_table is True
    assert data["rows"] == [
        ["Product", "Q1", "Q2"],
        ["Widget A", "100", "150"],
    ]
    assert data["headers"] != data["rows"][0]
    assert data["headers"] == ["Column 1", "Column 2", "Column 3"]
    assert data["metadata"]["has_headers"] is False


def test_no_thead_first_row_th_acts_as_headers():
    # A first row using semantic <th> (but no <thead>) must still be treated as
    # headers and must not reappear as a data row.
    table = (
        "<table><caption>X</caption><tbody>"
        "<tr><th>Name</th><th>Age</th></tr>"
        "<tr><td>Alice</td><td>30</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Name", "Age"]
    assert data["rows"] == [["Alice", "30"]]
    assert ["Name", "Age"] not in data["rows"]
    assert data["metadata"]["has_headers"] is True


def test_thead_headers_still_extracted_and_not_in_rows():
    # The <thead> path is unchanged: header cells come from <thead>, the header
    # row is not emitted as data.
    table = (
        "<table><thead>"
        "<tr><th>Quarter</th><th>Revenue</th></tr>"
        "</thead><tbody>"
        "<tr><td>Q1</td><td>1234</td></tr>"
        "<tr><td>Q2</td><td>5678</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Quarter", "Revenue"]
    assert data["rows"] == [["Q1", "1234"], ["Q2", "5678"]]
    assert data["metadata"]["has_headers"] is True


def test_no_thead_first_row_td_with_colspan_keeps_all_columns():
    # Guards against truncating columns when a <td>-only first row uses colspan:
    # default headers must be derived from the widest (colspan-expanded) row,
    # not the raw <td> count of the first row.
    table = (
        "<table><caption>c</caption><tbody>"
        "<tr><td colspan=\"2\">H</td><td>X</td></tr>"
        "<tr><td>A</td><td>B</td><td>C</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Column 1", "Column 2", "Column 3"]
    assert data["rows"] == [["H", "H", "X"], ["A", "B", "C"]]
    assert data["metadata"]["column_count"] == 3


def test_extract_tables_no_thead_no_duplication():
    # End-to-end via the public extract_tables() entry point: the table passes
    # the data-table threshold and is returned exactly once, unduplicated.
    table_html = (
        "<table><caption>Sales Data</caption><tbody>"
        "<tr><td>Product</td><td>Q1</td><td>Q2</td></tr>"
        "<tr><td>Widget A</td><td>100</td><td>150</td></tr>"
        "</tbody></table>"
    )
    root = html.fromstring("<html><body>" + table_html + "</body></html>")
    strategy = DefaultTableExtraction(table_score_threshold=7)
    tables = strategy.extract_tables(root)

    assert len(tables) == 1
    data = tables[0]
    assert data["rows"] == [
        ["Product", "Q1", "Q2"],
        ["Widget A", "100", "150"],
    ]
    assert data["headers"] == ["Column 1", "Column 2", "Column 3"]
    assert data["rows"][0] != data["headers"]
