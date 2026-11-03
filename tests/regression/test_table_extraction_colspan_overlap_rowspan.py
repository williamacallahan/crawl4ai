"""Direct regression tests for ``DefaultTableExtraction`` colspan-over-rowspan
overlap handling.

The grid-expansion algorithm in ``crawl4ai/table_extraction.py`` carries an
earlier ``rowspan="N"`` cell's value down through ``N-1`` continuation rows via
a ``pending`` dict keyed by column index. Before this fix, pending entries
were consumed ONLY at each cell's *start column* via
``while col in pending: append_pending()``. A ``colspan="K"`` cell that starts
BEFORE a pending column and extends past it incremented ``col`` across the
spanned columns without inspecting ``pending`` for the intermediate indices,
leaving the entry stale. The stale entry then leaked into a later row
(shifting that row's cells one column to the right) or -- on the last row of a
group -- was silently dropped by the trailing-fill loop (which considers only
pending indices ``>= col``), losing the continuation value entirely. The same
buggy pattern in the multi-row ``<thead>`` header loop made a column header
vanish from ``headers`` completely.

Per the module's documented contract ("A cell with rowspan=N is duplicated down
into the N-1 continuation rows it occupies within its row group"), the
continuation value belongs in the row it spans into -- never below it, never
absent. WHATWG §4.9.12.1 treats a ``colspan`` over an occupied slot as a
table-model error; this fix lets the rowspan continuation keep the slot and
the ``colspan`` text fill only the free columns of its span. Because
``append_pending()`` already advances ``col``, the colspan-loop's ``col += 1``
runs only in the else branch to avoid a double-increment that would otherwise
skip a spanned pending column.

Introduced for the body loop by commit 3b2c79bc and duplicated into the
``<thead>`` header loop by 3f478439. These tests run on real lxml-parsed HTML;
no browser or network is needed.
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


def test_body_colspan_starts_before_pending_rowspan_column():
    # The exact repro: row 1 holds B with rowspan="2" at col 1; row 2's D
    # carries colspan="2" starting at col 0, spanning across col 1 where B's
    # continuation belongs. Under the bug, B leaked into row 3, shifting F.
    # After the fix, B occupies col 1 of row 2, D fills only free col 0, and
    # nothing leaks into row 3.
    table = (
        "<table><tbody>"
        '<tr><td>A</td><td rowspan="2">B</td><td>C</td></tr>'
        '<tr><td colspan="2">D</td></tr>'
        "<tr><td>E</td><td>F</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"] == [
        ["A", "B", "C"],
        ["D", "B", ""],
        ["E", "F", ""],
    ]
    assert data["rows"][1][1] == "B"
    assert data["rows"][2][0] == "E"
    assert data["rows"][2][1] == "F"
    assert "B" not in data["rows"][2]


def test_body_colspan_overlap_does_not_double_append():
    # Guard against the wrong alternative fix ("consume then append" inside
    # the colspan loop): that variant would emit D a second time after
    # consuming B, producing ['D','B','D'] -- placing D under a column its
    # span already ceded to the rowspan continuation. The correct behavior
    # fills ONLY the free columns of the colspan span.
    table = (
        "<table><tbody>"
        '<tr><td>A</td><td rowspan="2">B</td><td>C</td></tr>'
        '<tr><td colspan="2">D</td></tr>'
        "<tr><td>E</td><td>F</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"][1] == ["D", "B", ""]
    assert data["rows"][1].count("D") == 1


def test_body_colspan_overlap_drops_pending_on_last_row_under_bug():
    # colspan="3" spans across pending col 1 AND reaches the table's last
    # row. Under the bug, the stale pending B survived to the end and was
    # silently dropped by the trailing-fill loop (which considers only
    # pending indices >= col), so row 2 became ['D','D','D'] with B lost
    # entirely. After the fix, B is preserved at col 1 of row 2 and D fills
    # the two free columns of its span.
    table = (
        "<table><tbody>"
        '<tr><td>A</td><td rowspan="2">B</td><td>C</td></tr>'
        '<tr><td colspan="3">D</td></tr>'
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"] == [["A", "B", "C"], ["D", "B", "D"]]
    assert data["rows"][1][1] == "B"


def test_body_colspan_spans_across_multiple_pending_rowspan_columns():
    # Row 1 establishes two pending continuations (B at col 1, C at col 2).
    # Row 2's colspan="3" starting at col 0 spans across both pending
    # columns. After the fix, the colspan yields to BOTH pending
    # continuations and D fills only col 0. Guards against the
    # double-increment variant (which would skip col 2's pending C, leaving
    # C to leak into row 3).
    table = (
        "<table><tbody>"
        '<tr><td>A</td><td rowspan="2">B</td><td rowspan="2">C</td></tr>'
        '<tr><td colspan="3">D</td></tr>'
        "<tr><td>E</td><td>F</td><td>G</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"] == [
        ["A", "B", "C"],
        ["D", "B", "C"],
        ["E", "F", "G"],
    ]
    assert data["rows"][2] == ["E", "F", "G"]


def test_body_colspan_overlap_cell_itself_carries_rowspan():
    # Row 1: B has rowspan="2" at col 1 (1 continuation pending).
    # Row 2: D has colspan="3" AND rowspan="2". Col 1 is pending (B), so D
    # fills cols 0 and 2 (free) while B keeps col 1. D's own rowspan="2"
    # carries D down in cols 0 and 2 to row 3.
    table = (
        "<table><thead><tr><th>0</th><th>1</th><th>2</th></tr></thead><tbody>"
        '<tr><td>A</td><td rowspan="2">B</td><td>C</td></tr>'
        '<tr><td colspan="3" rowspan="2">D</td></tr>'
        "<tr><td>F</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"] == [
        ["A", "B", "C"],
        ["D", "B", "D"],
        ["D", "F", "D"],
    ]
    assert "B" not in data["rows"][2]


def test_body_colspan_starting_at_pending_column_unaffected():
    # The ``while col in pending`` check at the cell start consumes the
    # pending slot BEFORE the colspan loop runs, so the overlap fix must not
    # perturb this non-overlap path. B's continuation occupies col 0 of row
    # 2, then D's colspan=2 spans cols 1 and 2 (expanding the grid to 3
    # wide), so row 1 is padded.
    table = (
        "<table><tbody>"
        '<tr><td rowspan="2">B</td><td>C</td></tr>'
        '<tr><td colspan="2">D</td></tr>'
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"] == [["B", "C", ""], ["B", "D", "D"]]


def test_thead_colspan_starts_before_pending_rowspan_header_preserved():
    # Row 1 of thead: X at col 0, H2 with rowspan="2" at col 1 (pending).
    # Row 2 of thead: HX with colspan="2" starting at col 0 spans across
    # col 1 where H2's continuation belongs. Under the bug, H2 VANISHED
    # from headers; after the fix, H2 is preserved at col 1 of the leaf
    # resolved header row. The <thead> header loop duplicates the same
    # colspan/rowspan grid logic as the body loop and had the same bug.
    table = (
        "<table><thead>"
        '<tr><th>X</th><th rowspan="2">H2</th></tr>'
        '<tr><th colspan="2">HX</th></tr>'
        "</thead><tbody>"
        "<tr><td>a</td><td>b</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["headers"] == ["HX", "H2"]
    assert data["headers"].count("H2") == 1


def test_real_world_summary_row_overlaps_pending_continuation():
    # The "Impact" scenario from the bug report: a Category rowspan sits in
    # the MIDDLE column (col 1) and is still pending (its continuation belongs
    # in row 3), while row 3 is a full-width "<td colspan=3>" that starts at
    # col 0 and spans across col 1 where the continuation belongs. The leading
    # ``while col in pending`` consumes col 0 but leaves col 1 pending, so the
    # overlap is exercised inside the colspan loop itself. Under the bug,
    # Category A leaked past the summary row into row 4. After the fix,
    # Category A's continuation occupies col 1 of the summary row, Total A
    # fills cols 0 and 2 only, and row 4 is unshifted.
    table = (
        "<table><thead>"
        "<tr><th>Cat</th><th>Metric</th><th>Value</th></tr>"
        "</thead><tbody>"
        '<tr><td>Count</td><td rowspan="3">Category A</td><td>10</td></tr>'
        "<tr><td>Volume</td><td>20</td></tr>"
        '<tr><td colspan="3">Total A</td></tr>'
        "<tr><td>Count</td><td>Category B</td><td>5</td></tr>"
        "</tbody></table>"
    )
    _, data = _extract(table, table_score_threshold=0)

    assert data["rows"] == [
        ["Count", "Category A", "10"],
        ["Volume", "Category A", "20"],
        ["Total A", "Category A", "Total A"],
        ["Count", "Category B", "5"],
    ]
    assert data["rows"][2][1] == "Category A"
    assert data["rows"][3] == ["Count", "Category B", "5"]
