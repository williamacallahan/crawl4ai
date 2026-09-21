"""Direct regression tests for DefaultTableExtraction multi-row ``<thead>`` handling.

These guard against ``extract_table_data`` reading only the FIRST ``<tr>`` of
``<thead>`` when building ``headers``. Rows 2..N of a multi-row ``<thead>``
were dropped from the output entirely -- they contributed neither to
``headers`` nor to ``rows`` -- so hierarchical column headers (the ``Q1`` /
``Q2`` leaf-row labels that body cells semantically align to) were silently
lost, and the surface-level ``headers`` list kept duplicate/degenerate
top-row labels (e.g. ``['Dept', '2024', '2024']``) with no warning and no
metadata flag (``has_headers: True``, but no ``header_row_count``).

The fix resolves ALL ``<thead>`` rows through the same colspan/rowspan grid
expansion used for the body (the WHATWG "forming a table" algorithm) and
takes the leaf (deepest resolved) row as ``headers`` -- policy (a) in the
report, the convention used by ``pandas.read_html``'s default single-level
output. The data loop's ``if row_group.tag == "thead": continue`` is
unchanged: ``<thead>`` rows still never appear in ``rows``.

The bug was introduced in a51545c (which created ``table_extraction.py``
with the ``thead_rows[0].xpath("./th")`` header line) and preserved by
3b2c79b (which restructured the row loop but kept the multi-row-``<thead>``
drop). These tests run on real lxml-parsed HTML and need no browser or
network.
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


# ---------------------------------------------------------------------------
# The exact reproduction from the bug report.
# ---------------------------------------------------------------------------

def test_multirow_thead_keeps_leaf_header_row():
    # The reported bug: a two-row <thead> with a rowspan="2" group header and
    # a colspan="2" top group. Under the bug, headers were the FIRST resolved
    # grid row (['Dept', '2024', '2024']); the leaf row ['Dept', 'Q1', 'Q2']
    # was dropped.
    table = (
        "<table><thead>"
        "<tr><th rowspan='2'>Dept</th><th colspan='2'>2024</th></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead><tbody>"
        "<tr><td>Sales</td><td>100</td><td>200</td></tr>"
        "<tr><td>Eng</td><td>300</td><td>400</td></tr>"
        "</tbody></table>"
    )
    is_data_table, data = _extract(table)

    assert is_data_table is True
    # Leaf-row flattening per the report's policy (a).
    assert data["headers"] == ["Dept", "Q1", "Q2"]
    assert data["rows"] == [["Sales", "100", "200"], ["Eng", "300", "400"]]
    assert data["metadata"]["has_headers"] is True
    # The duplicate/degenerate top-row labels must NOT survive into headers.
    assert data["headers"] != ["Dept", "2024", "2024"]
    assert "2024" not in data["headers"]


def test_three_row_thead_keeps_leaf_header_row():
    # A three-row <thead> (Dept/2024/2025 -> H1/H2 -> Q1..Q4). Under the bug,
    # BOTH the 2nd and 3rd header rows were dropped; only the top row
    # ['Dept','2024','2024','2025','2025'] survived. The intermediate
    # (H1,H2) and leaf (Q1..Q4) header cells were absent from the output.
    table = (
        "<table><caption>Financials</caption><thead>"
        "<tr><th rowspan='3'>Dept</th><th colspan='2'>2024</th>"
        "<th colspan='2'>2025</th></tr>"
        "<tr><th colspan='2'>H1</th><th colspan='2'>H2</th></tr>"
        "<tr><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th></tr>"
        "</thead><tbody>"
        "<tr><td>Sales</td><td>10</td><td>20</td><td>30</td><td>40</td></tr>"
        "</tbody></table>"
    )
    is_data_table, data = _extract(table)

    assert is_data_table is True
    # The leaf row carries 'Dept' down (rowspan=3) and fills cols 1..4 with
    # the quarterly cells actually aligned to body data.
    assert data["headers"] == ["Dept", "Q1", "Q2", "Q3", "Q4"]
    assert data["rows"] == [["Sales", "10", "20", "30", "40"]]
    assert data["metadata"]["has_headers"] is True
    # Intermediate row values must not leak into the (flat) header contract.
    assert "H1" not in data["headers"]
    assert "H2" not in data["headers"]
    assert "2024" not in data["headers"]
    assert "2025" not in data["headers"]


# ---------------------------------------------------------------------------
# End-to-end via the production entry point (content_scraping_strategy.py).
# ---------------------------------------------------------------------------

def test_multirow_thead_extracted_end_to_end_via_extract_tables():
    # The production path: content_scraping_strategy.py populates
    # media["tables"] via the public extract_tables() entry point. A
    # multi-row-<thead> table passing the default data-table threshold must
    # be returned exactly once with leaf-row headers intact.
    table_html = (
        "<table><caption>2024 Department Performance</caption><thead>"
        "<tr><th rowspan='2'>Department</th><th colspan='4'>2024 Performance</th></tr>"
        "<tr><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th></tr>"
        "</thead><tbody>"
        "<tr><td>Sales</td><td>$1.2M</td><td>$1.5M</td><td>$1.8M</td><td>$2.0M</td></tr>"
        "<tr><td>Engineering</td><td>85%</td><td>88%</td><td>92%</td><td>95%</td></tr>"
        "</tbody></table>"
    )
    tables = _extract_all(table_html)

    assert len(tables) == 1
    data = tables[0]
    assert data["headers"] == ["Department", "Q1", "Q2", "Q3", "Q4"]
    assert data["rows"] == [
        ["Sales", "$1.2M", "$1.5M", "$1.8M", "$2.0M"],
        ["Engineering", "85%", "88%", "92%", "95%"],
    ]
    assert data["metadata"]["has_headers"] is True
    assert data["metadata"]["has_caption"] is True
    assert "2024 Performance" not in data["headers"]


# ---------------------------------------------------------------------------
# Leaf-row flattening mechanics: rowspan carry-down into the leaf row.
# ---------------------------------------------------------------------------

def test_rowspan_in_thead_carries_group_label_down_to_leaf_row():
    # The group label 'A' carries down (rowspan=2) so the leaf row resolves
    # to ['A', 'C'] even though the leaf row's own <th> is only 'C'.
    table = (
        "<table><thead>"
        "<tr><th rowspan='2'>A</th><th>B</th></tr>"
        "<tr><th>C</th></tr>"
        "</thead><tbody>"
        "<tr><td>x</td><td>y</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["A", "C"]
    assert data["rows"] == [["x", "y"]]


# ---------------------------------------------------------------------------
# Interaction with body-side features (rowspan carry-down, row headers).
# ---------------------------------------------------------------------------

def test_multirow_thead_with_body_rowspan_is_independent():
    # Header-side rowspan (Dept carries down through the two header rows)
    # and body-side rowspan (Sales carries down through two body rows) must
    # be resolved independently by their own grid expansions.
    table = (
        "<table><thead>"
        "<tr><th rowspan='2'>Dept</th><th colspan='2'>2024</th></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead><tbody>"
        "<tr><td rowspan='2'>Sales</td><td>100</td><td>200</td></tr>"
        "<tr><td>300</td><td>400</td></tr>"
        "<tr><td>Eng</td><td>500</td><td>600</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Dept", "Q1", "Q2"]
    assert data["rows"] == [
        ["Sales", "100", "200"],
        ["Sales", "300", "400"],
        ["Eng", "500", "600"],
    ]


def test_multirow_thead_with_th_row_headers_in_tbody_preserves_both():
    # Multi-row <thead> column headers + <th scope="row"> row labels in
    # <tbody>. The column headers must resolve to the leaf row while the
    # row labels must each occupy the first grid column of their body row.
    table = (
        "<table><caption>Characteristics</caption><thead>"
        "<tr><th rowspan='2'>Group</th><th colspan='2'>Reading</th></tr>"
        "<tr><th>Negative</th><th>Positive</th></tr>"
        "</thead><tbody>"
        "<tr><th scope='row'>Mood</th><td>Sad</td><td>Happy</td></tr>"
        "<tr><th scope='row'>Grade</th><td>Failing</td><td>Passing</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Group", "Negative", "Positive"]
    assert data["rows"] == [
        ["Mood", "Sad", "Happy"],
        ["Grade", "Failing", "Passing"],
    ]
    assert all(row[0] != "" for row in data["rows"])


# ---------------------------------------------------------------------------
# rowspan="0" in <thead> (HTML "span to end of row group" semantics).
# ---------------------------------------------------------------------------

def test_rowspan_zero_in_thead_extends_through_all_thead_rows():
    # rowspan="0" means "span to the end of the row group" (the <thead>).
    # The 'Dept' label must carry down through both following <thead> rows
    # so the leaf row resolves to ['Dept', 'Q1', 'Q2'].
    table = (
        "<table><thead>"
        "<tr><th rowspan='0'>Dept</th><th colspan='2'>2024</th></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead><tbody>"
        "<tr><td>Sales</td><td>100</td><td>200</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Dept", "Q1", "Q2"]
    assert data["rows"] == [["Sales", "100", "200"]]


# ---------------------------------------------------------------------------
# Single-row <thead> regression guards (behavior unchanged by the fix).
# ---------------------------------------------------------------------------

def test_single_row_thead_headers_unchanged():
    # The most common case: a single <thead> row must produce the same
    # headers as before the fix (no multi-row machinery should perturb it).
    table = (
        "<table><thead>"
        "<tr><th>Quarter</th><th>Revenue</th><th>Growth</th></tr>"
        "</thead><tbody>"
        "<tr><td>Q1</td><td>1234</td><td>12.5%</td></tr>"
        "<tr><td>Q2</td><td>5678</td><td>18.0%</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Quarter", "Revenue", "Growth"]
    assert data["rows"] == [["Q1", "1234", "12.5%"], ["Q2", "5678", "18.0%"]]
    assert data["metadata"]["has_headers"] is True


def test_empty_thead_with_no_th_still_has_headers_true():
    # An empty <thead> (or <thead> with rows but no <th>) must still report
    # has_headers=True via the thead_rows presence guard, and headers must
    # fall back to the generic 'Column N' default. Regression guard for the
    # has_headers = bool(thead_rows) or bool(headers) computation.
    table = (
        "<table><thead><tr></tr></thead><tbody>"
        "<tr><td>1</td><td>2</td></tr>"
        "<tr><td>3</td><td>4</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["metadata"]["has_headers"] is True
    assert data["headers"] == ["Column 1", "Column 2"]
    assert data["rows"] == [["1", "2"], ["3", "4"]]
    assert data["metadata"]["column_count"] == 2


# ---------------------------------------------------------------------------
# Nested-subtable depth scoping under multi-row <thead>.
# ---------------------------------------------------------------------------

def test_multirow_thead_nested_subtable_does_not_contaminate_headers():
    # A nested <table> embedded inside a <th> of a multi-row <thead> must
    # stay excluded from the outer table's header text (depth-scoped
    # _cell_text). Only the outer th's own text contributes to headers.
    table = (
        "<table><caption>Outer report</caption><thead>"
        "<tr><th rowspan='2'>Dept</th>"
        "<th colspan='2'>2024"
        "<table><tr><td>nested-one</td><td>nested-two</td></tr></table>"
        "</th></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "</thead><tbody>"
        "<tr><td>Sales</td><td>100</td><td>200</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table)

    assert data["headers"] == ["Dept", "Q1", "Q2"]
    assert data["rows"] == [["Sales", "100", "200"]]
    assert "nested-one" not in data["headers"]
    assert "nested-two" not in data["headers"]
    assert "2024" not in data["headers"]
