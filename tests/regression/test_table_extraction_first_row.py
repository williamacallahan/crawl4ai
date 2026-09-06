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


def test_no_thead_mixed_th_td_first_row_keeps_all_columns():
    # Row-header pattern: <th scope="row"> label + <td> data cells, no <thead>.
    # The lone <th> must not become the only header — that would cap
    # max_columns at 1 and truncate every data row.
    table = (
        "<table><caption>c</caption><tbody>"
        "<tr><th>Region</th><td>North</td><td>South</td></tr>"
        "<tr><th>Sales</th><td>100</td><td>50</td></tr>"
        "<tr><th>Costs</th><td>80</td><td>30</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Column 1", "Column 2"]
    assert data["rows"] == [["North", "South"], ["100", "50"], ["80", "30"]]
    assert data["metadata"]["column_count"] == 2
    assert data["metadata"]["has_headers"] is False


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


def _nested_in_th_table_html():
    """A headerless table whose first row is all <th>, with a nested subtable
    inside one <th>. cell text is intentionally verbose so the outer table still
    clears the is_data_table text-density bonus despite the nested-table -3
    penalty (matches the report's conditional-reachability trigger).
    """
    return (
        '<table border="1" summary="Quarterly sales summary across all regions" '
        'data-report="sales" data-fiscal="2024">'
        "<caption>Quarterly sales summary across all regions including revenue "
        "breakdown and growth indicators</caption>"
        "<tr>"
        "<th>Region Name Descriptor</th>"
        "<th>Performance Breakdown Measurement"
        "<table><tr><td>sparkline one</td><td>sparkline two</td>"
        "<td>sparkline three</td><td>sparkline four</td></tr></table>"
        "</th>"
        "<th>Total Revenue in dollars</th>"
        "<th>Growth Percentage year over year</th>"
        "</tr>"
        "<tr><td>North America Region eastern territory</td>"
        "<td>one hundred to two hundred dollars range</td>"
        "<td>three hundred forty five dollars total</td>"
        "<td>twenty five percent growth measured</td></tr>"
        "<tr><td>South America Region western territory</td>"
        "<td>fifty to one hundred fifty dollars range</td>"
        "<td>two hundred twelve dollars total</td>"
        "<td>fifteen percent growth measured</td></tr>"
        "<tr><td>Europe Region central territory</td>"
        "<td>two hundred to four hundred dollars range</td>"
        "<td>five hundred ten dollars total</td>"
        "<td>thirty two percent growth measured</td></tr>"
        "</table>"
    )


def test_no_thead_first_row_all_th_with_nested_table_in_th_adopted_as_headers():
    # Regression for the descendant-axis guard bug: a headerless (<thead>-less)
    # table whose first row is all <th> but one <th> nests a <table> must still
    # have its <th> cells adopted as headers. The guard must use the child axis
    # (./td) so that <td> belonging to a nested subtable inside a <th> do NOT
    # defeat header adoption. Before the fix, .//td descended into the nested
    # subtable's <td> and rejected a genuine all-<th> header row, emitting
    # generic "Column N" placeholders and has_headers=False.
    _, data = _extract(_nested_in_th_table_html())

    assert data["metadata"]["has_headers"] is True
    # The three <th> cells without a nested subtable must be adopted verbatim.
    assert data["headers"][0] == "Region Name Descriptor"
    assert data["headers"][2] == "Total Revenue in dollars"
    assert data["headers"][3] == "Growth Percentage year over year"
    # The <th> that nests the subtable is corrupted by text_content() descendant
    # bleed (a separate, pre-existing defect), but it must still carry its own
    # <th> text rather than a generic "Column N" placeholder.
    assert data["headers"][1].startswith("Performance Breakdown Measurement")
    assert "Column 1" not in data["headers"]
    # A real data row must be present (the pre-existing descendant-axis row loop
    # also emits bogus sparkline rows; only assert membership of genuine rows).
    north = [
        "North America Region eastern territory",
        "one hundred to two hundred dollars range",
        "three hundred forty five dollars total",
        "twenty five percent growth measured",
    ]
    assert north in data["rows"]


def test_no_thead_all_td_first_row_with_nested_table_still_rejected():
    # Control for the fix: a first row whose direct children are <td> (data
    # row) must STILL be rejected even when a nested <table> lives inside one
    # of those <td>. The child-axis (./td) guard must find the direct <td>
    # children and fall back to generic headers + has_headers=False.
    table = (
        "<table><caption>c</caption>"
        "<tr><td>Outer A"
        "<table><tr><td>inner one</td><td>inner two</td></tr></table>"
        "</td><td>Outer B</td></tr>"
        "<tr><td>r2a</td><td>r2b</td></tr>"
        "</table>"
    )
    _, data = _extract(table)

    # The guard correctly rejects the <td>-first row: no real headers are
    # adopted. (The exact column count is inflated by the pre-existing
    # descendant-axis row loop, which is out of scope for this fix; only the
    # guard's adoption decision is asserted here.)
    assert data["metadata"]["has_headers"] is False
    assert all(h.startswith("Column ") for h in data["headers"])
    assert not any("Outer" in h for h in data["headers"])
