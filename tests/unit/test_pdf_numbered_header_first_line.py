"""Regression test for section-number loss in first-line numbered PDF headings.

``clean_pdf_text`` (crawl4ai/processors/pdf/utils.py) converts a PDF page's
extracted text to Markdown. Its numbered-header branch read ``lines[i-1]`` to
decide whether a line was a standalone numbered heading, but it lacked the
``i > 0`` guard that the sibling ``clean_pdf_text_to_html`` already had. At
``i == 0`` Python's negative indexing resolves ``lines[i-1]`` to ``lines[-1]``
-- the LAST line of the (preprocessed) page text -- so whether the first line
was classified as a numbered header (and had its leading section number
dropped) depended on the content of the *last* line of the document. Because
the branch sits before the section-header branch, when it fired on line 0 it
also pre-empted the section-header branch that would otherwise have kept the
number. It only triggers for 2-word numbered first-line headings
(``1 Introduction``, ``1 methods``, ``2.1 Methods``): 3+ word numbered headings
are caught first by the title heuristic (3-8 words).

The preprocessing ``re.sub(r'\\.\\n', '.\\n\\n', decoded)`` widens a final
``.\\n`` into ``.\\n\\n``, which makes ``lines[-1]`` empty -- the exact shape
that trips the branch on line 0. The fix adds the ``i > 0`` guard so first-
line classification is independent of the last line's content, consistent with
``clean_pdf_text_to_html``.

Introduced in commit f8fd9d9e (PR #657); the guard was present in the HTML twin
from day one and was omitted from the Markdown twin.
"""

import pytest

from crawl4ai.processors.pdf.utils import clean_pdf_text


# Body ending with "two.\n" -- after the ``re.sub(r'\.\n', '.\n\n', ...)``
# preprocessing step, ``lines[-1]`` becomes '' (the empty trigger line). This
# is the shape that exposed the negative-index read on line 0.
BODY_TAIL_EMPTY_LAST = "This is the body text line one.\nThis is body text line two.\n"

# Same body, but the last line is non-empty -- the pre-bug classification.
BODY_TAIL_NONEMPTY_LAST = "This is the body text line one.\nThis is body text line two.\nEnd"


@pytest.mark.parametrize(
    "heading, must_absent",
    [
        ("1 Introduction", "## Introduction"),
        ("1 methods", "## methods"),
        ("2.1 Methods", "### Methods"),
    ],
)
def test_clean_pdf_text_first_line_numbered_header_keeps_number(heading, must_absent):
    """The leading section number must survive; the number-dropping header
    form produced by the buggy branch must be absent. Pins the three triggering
    2-word numbered first-line shapes (upper-1word-number, lowercase title,
    multi-component number)."""
    out = clean_pdf_text(1, heading + "\n" + BODY_TAIL_EMPTY_LAST)
    assert heading in out, f"section number dropped from output: {out!r}"
    assert must_absent not in out, f"buggy promotion present: {out!r}"


@pytest.mark.parametrize(
    "heading",
    ["1 Introduction", "1 methods", "2.1 Methods"],
)
def test_first_line_classification_independent_of_last_line(heading):
    """The invariant of the fix: first-line classification must not depend on
    the content of the last line of the document. Both variants must produce
    the same first output line (which must keep the number). Guards against
    reordering the branches or changing the preprocessing in a way that
    reintroduces the last-line dependence."""
    out_empty = clean_pdf_text(1, heading + "\n" + BODY_TAIL_EMPTY_LAST)
    out_nonempty = clean_pdf_text(1, heading + "\n" + BODY_TAIL_NONEMPTY_LAST)
    assert heading in out_empty and heading in out_nonempty
    first_empty = out_empty.split("\n", 1)[0]
    first_nonempty = out_nonempty.split("\n", 1)[0]
    assert first_empty == first_nonempty, (
        f"first-line classification depends on last line: "
        f"empty_last={first_empty!r} vs nonempty_last={first_nonempty!r}"
    )


def test_non_first_line_numbered_header_still_promoted():
    """The ``i > 0`` guard must NOT disable the numbered-header branch for
    legitimate mid-page numbered headings (a numbered heading preceded by a
    blank line, not on line 0). These keep getting promoted to ``#``-headers
    with the number dropped -- the intended behavior of that branch. Guards
    against over-correcting the fix by removing the branch entirely."""
    text = "Some title line one.\n\n2.1 Background\nBody text here.\n"
    out = clean_pdf_text(1, text)
    assert "### Background" in out, f"numbered-header branch broken: {out!r}"
