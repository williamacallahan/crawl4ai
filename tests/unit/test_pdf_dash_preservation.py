"""Regression test for ASCII dash (" - ") destruction in PDF post-processing.

clean_pdf_text_to_html and clean_pdf_text each used to run
``re.sub(r'\\s+-\\s+', '', text)`` (commented "Join hyphenated words"). That pattern matches a
space-delimited ASCII dash -- the shape of arithmetic, ranges, and header separators -- rather
than a PDF line-wrap hyphenation (already handled earlier by the ``line.endswith('-')`` branch).
The substitution removed the hyphen AND both surrounding spaces, fusing adjacent tokens with no
separator: ``"2020 - 2024" -> "20202024"``, ``"5 - 3 = 2" -> "53 = 2"``,
``"Smith - Jones" -> "SmithJones"``.

clean_pdf_text_to_html is user-facing (pdf_page.html -> ScrapingResult.cleaned_html -> crawl JSON);
clean_pdf_text's output (pdf_page.markdown) is currently unused by the production crawl path but
is fixed in lockstep to prevent a future regression. These tests pin dash preservation in both
sinks and guard the line-wrapped hyphenation heuristic against overcorrection.
"""

from crawl4ai.processors.pdf.utils import clean_pdf_text, clean_pdf_text_to_html

# A leading short line becomes an <h2> title (its own code path); the blank line
# then starts a fresh paragraph, which is the user-facing sink under test.
TITLE = "My Paper Title\n\n"


def _html(body):
    return clean_pdf_text_to_html(1, TITLE + body)


def _markdown(body):
    return clean_pdf_text(1, TITLE + body)


def test_ascii_dash_preserved_in_cleaned_html():
    body = (
        "The experiment ran from 2020 - 2024.\n"
        "We computed 5 - 3 = 2 as a check.\n"
        "The author list is Smith - Jones et al.\n"
        "Section 1 - Introduction\n"
    )

    html = _html(body)

    assert "2020 - 2024" in html
    assert "5 - 3 = 2" in html
    assert "Smith - Jones" in html
    assert "Section 1 - Introduction" in html
    # The corruption signature: surrounding tokens fused with no separator.
    assert "20202024" not in html
    assert "53 = 2" not in html
    assert "SmithJones" not in html
    assert "Section 1Introduction" not in html


def test_ascii_dash_preserved_in_markdown():
    body = (
        "The experiment ran from 2020 - 2024.\n"
        "We computed 5 - 3 = 2 as a check.\n"
        "The author list is Smith - Jones et al.\n"
        "Section 1 - Introduction\n"
    )

    markdown = _markdown(body)

    assert "2020 - 2024" in markdown
    assert "5 - 3 = 2" in markdown
    assert "Smith - Jones" in markdown
    assert "Section 1 - Introduction" in markdown
    assert "20202024" not in markdown
    assert "53 = 2" not in markdown
    assert "SmithJones" not in markdown
    assert "Section 1Introduction" not in markdown


def test_line_wrapped_hyphenation_still_rejoined():
    # The intended dehyphenation heuristic (``line.endswith('-')``) runs before
    # post-processing and must keep working after the dash-stripping regex is
    # removed. A line ending in ``-`` is rejoined into a single paragraph; the
    # literal hyphen-at-line-break must not survive into cleaned_html.
    body = "The measure-\nments were taken over several days."

    html = _html(body)

    assert "measure" in html
    assert "ments" in html
    assert html.count("<p>") == 1
    assert "measure-\nments" not in html
