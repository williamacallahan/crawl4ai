"""Preserve word separators when removing whitespace-only inline elements."""

import pytest

from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


@pytest.fixture
def strategy():
    return LXMLWebScrapingStrategy()


SPLIT_HTML = (
    "<html><body><p><span>H</span><span>i</span><span> </span>"
    "<span>t</span><span>h</span><span>e</span><span>r</span><span>e</span></p>"
    "<p>plain text stays fine</p></body></html>"
)


def test_whitespace_span_between_words_is_kept_in_markdown(strategy):
    result = strategy._scrap("http://test.com", SPLIT_HTML)
    md = DefaultMarkdownGenerator().generate_markdown(
        input_html=result["cleaned_html"], base_url="http://test.com"
    )

    assert "Hi there" in md.raw_markdown
    assert "plain text stays fine" in md.raw_markdown


def test_whitespace_spans_between_several_words(strategy):
    html = (
        "<html><body><p><span>one</span><span> </span><span>two</span>"
        "<span> </span><span>three</span></p></body></html>"
    )
    result = strategy._scrap("http://test.com", html)
    md = DefaultMarkdownGenerator().generate_markdown(
        input_html=result["cleaned_html"], base_url="http://test.com"
    )

    assert "one two three" in md.raw_markdown


def test_whitespace_first_child_separates_parent_text_from_tail(strategy):
    html = "<html><body><p>Hello<span> \n\t </span>world</p></body></html>"
    result = strategy._scrap("http://test.com", html)
    md = DefaultMarkdownGenerator().generate_markdown(
        input_html=result["cleaned_html"], base_url="http://test.com"
    )

    assert "Hello world" in md.raw_markdown


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('<p>See <a href="https://example.com/blog"><span> </span>Blog</a> now</p>', "[Blog](https://example.com/blog)"),
        ('<p>See <a href="https://example.com/blog">Blog<span> </span></a> now</p>', "[Blog](https://example.com/blog)"),
        ("<h2><span> </span>Title</h2>", "## Title"),
    ],
    ids=["link-leading", "link-trailing", "heading-leading"],
)
def test_edge_whitespace_element_adds_no_space_inside_markdown(body, expected):
    html = f"<html><body><article>{body}<p>Padding paragraph with enough words.</p></article></body></html>"
    cleaned = LXMLWebScrapingStrategy().scrap("https://example.com/", html).cleaned_html
    markdown = DefaultMarkdownGenerator().generate_markdown(
        input_html=cleaned, base_url="https://example.com/"
    ).raw_markdown
    assert expected in markdown
