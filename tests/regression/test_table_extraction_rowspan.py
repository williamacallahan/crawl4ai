"""Direct regression tests for DefaultTableExtraction rowspan handling.

These guard against a body cell carrying ``rowspan="N"`` failing to be
duplicated into the N-1 continuation rows it occupies. Without carry-down,
each continuation row was left-shifted by one column and padded with a
fabricated trailing ``""`` so values landed under the wrong headers with no
signal. The regression was introduced in a51545c, which created
``crawl4ai/table_extraction.py`` with a body-row loop that read only ``colspan``
and never ``rowspan`` despite the class docstring claiming rowspan support.

The HTML Living Standard table model ("Forming a table") and the project's
own ``LLMTableExtraction`` prompt both require that a cell with rowspan be
duplicated down into all rows it spans. These tests pin that contract for the
default strategy via real lxml-parsed HTML; no browser or network is needed.
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


def test_rowspan_basic_carry_down():
    # Minimal repro: a single rowspan="2" cell must be duplicated into the
    # continuation row; the continuation row's own <td> must land in the
    # column AFTER the span, not in the span's column.
    table = (
        "<table><caption>c</caption><thead>"
        "<tr><th>Cat</th><th>Item</th><th>Val</th></tr>"
        "</thead><tbody>"
        '<tr><td rowspan="2">A</td><td>Item1</td><td>$100</td></tr>'
        "<tr><td>Item2</td><td>$200</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Cat", "Item", "Val"]
    assert data["rows"] == [
        ["A", "Item1", "$100"],
        ["A", "Item2", "$200"],
    ]
    # The continuation row must NOT be left-shifted or padded with "".
    assert data["rows"][1][0] == "A"
    assert data["rows"][1][-1] != ""


def test_rowspan_grouped_department_table_end_to_end():
    # The exact end-to-end reproduction from the bug report: a realistic
    # department-grouped data table passing the default data-table threshold,
    # extracted through the public extract_tables() entry point (the path
    # content_scraping_strategy.py uses to populate media["tables"]).
    table_html = (
        "<table>"
        "<caption>2024 Department Performance</caption>"
        "<thead><tr>"
        "<th>Department</th><th>Team/Region</th>"
        "<th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th>"
        "</tr></thead><tbody>"
        '<tr><td rowspan="2">Sales</td><td>North</td>'
        "<td>$1.2M</td><td>$1.5M</td><td>$1.8M</td><td>$2.0M</td></tr>"
        "<tr><td>South</td>"
        "<td>$0.9M</td><td>$1.1M</td><td>$1.3M</td><td>$1.4M</td></tr>"
        '<tr><td rowspan="2">Engineering</td><td>Alpha</td>'
        "<td>85%</td><td>88%</td><td>92%</td><td>95%</td></tr>"
        "<tr><td>Beta</td>"
        "<td>78%</td><td>82%</td><td>86%</td><td>90%</td></tr>"
        "</tbody></table>"
    )
    tables = _extract_all(table_html)

    assert len(tables) == 1
    data = tables[0]
    assert data["headers"] == [
        "Department", "Team/Region", "Q1", "Q2", "Q3", "Q4",
    ]
    assert data["rows"] == [
        ["Sales", "North", "$1.2M", "$1.5M", "$1.8M", "$2.0M"],
        ["Sales", "South", "$0.9M", "$1.1M", "$1.3M", "$1.4M"],
        ["Engineering", "Alpha", "85%", "88%", "92%", "95%"],
        ["Engineering", "Beta", "78%", "82%", "86%", "90%"],
    ]
    assert all("" not in row for row in data["rows"])
    assert data["rows"][0][0] == data["rows"][1][0] == "Sales"
    assert data["rows"][2][0] == data["rows"][3][0] == "Engineering"


def test_rowspan_and_colspan_combined_same_cell():
    # A single cell carrying BOTH rowspan and colspan must fill `colspan`
    # columns in this row and carry all of them down into the continuation row.
    table = (
        "<table><caption>c</caption><thead>"
        "<tr><th>A</th><th>B</th><th>C</th></tr>"
        "</thead><tbody>"
        '<tr><td rowspan="2" colspan="2">Merged</td><td>X</td></tr>'
        "<tr><td>Y</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["rows"] == [
        ["Merged", "Merged", "X"],
        ["Merged", "Merged", "Y"],
    ]


def test_rowspan_at_end_of_row_uses_flush_loop():
    # The rowspan cell is the LAST cell in the row. The continuation row has
    # one <td> that must occupy the FIRST column, and the pending rowspan
    # value must be flushed AFTER it into the second column. This exercises
    # the trailing-flush branch of the grid builder.
    table = (
        "<table><caption>c</caption><thead>"
        "<tr><th>Item</th><th>Cat</th></tr>"
        "</thead><tbody>"
        '<tr><td>I1</td><td rowspan="2">A</td></tr>'
        "<tr><td>I2</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Item", "Cat"]
    assert data["rows"] == [
        ["I1", "A"],
        ["I2", "A"],
    ]


def test_rowspan_does_not_overconsume_following_full_row():
    # A row whose cells are ALL full (after a span window has ended) must not
    # be shifted or padded. Guards against the fix over-applying pending
    # spans past their declared row count.
    table = (
        "<table><caption>c</caption><thead>"
        "<tr><th>Cat</th><th>Item</th><th>Val</th></tr>"
        "</thead><tbody>"
        '<tr><td rowspan="2">A</td><td>I1</td><td>$1</td></tr>'
        "<tr><td>I2</td><td>$2</td></tr>"
        # span window ends here. Following row has full <td> count and must NOT be shifted.
        "<tr><td>B</td><td>I3</td><td>$3</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["rows"] == [
        ["A", "I1", "$1"],
        ["A", "I2", "$2"],
        ["B", "I3", "$3"],
    ]
    assert data["rows"][2][0] == "B"
    assert data["rows"][2][-1] == "$3"


def test_no_rowspan_table_is_unaffected_by_fix():
    # Regression guard: a plain table with no rowspan/colspan must produce
    # identical output to the pre-fix behavior (each row [<td>...] in order).
    table = (
        "<table><caption>c</caption><thead>"
        "<tr><th>A</th><th>B</th><th>C</th></tr>"
        "</thead><tbody>"
        "<tr><td>1</td><td>2</td><td>3</td></tr>"
        "<tr><td>4</td><td>5</td><td>6</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["A", "B", "C"]
    assert data["rows"] == [
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]


def test_colspan_only_table_still_works():
    # Regression guard: colspan expansion must continue to work after the
    # rowspan fix (the grid builder must read colspan as well).
    table = (
        "<table><caption>c</caption><thead>"
        "<tr><th>A</th><th>B</th><th>C</th></tr>"
        "</thead><tbody>"
        '<tr><td colspan="2">merged</td><td>6</td></tr>'
        "<tr><td>1</td><td>2</td><td>3</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["rows"] == [
        ["merged", "merged", "6"],
        ["1", "2", "3"],
    ]
