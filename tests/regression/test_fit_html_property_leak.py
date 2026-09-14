"""``CrawlResult.fit_html`` must never leak a ``property`` descriptor into
serialized output.

Regression for a field/property name collision: ``CrawlResult`` declared
``fit_html`` both as a pydantic field *and* as a deprecation ``@property``.
Pydantic v2 captured the live ``property`` descriptor as the field's default
value, so any ``CrawlResult(...)`` constructed without an explicit
``fit_html=`` argument received the ``property`` object itself instead of
``None``.  ``model_dump()`` then stringified the descriptor via ``str()``,
producing the garbage string ``"<property object at 0x...>"`` in serialized
JSON, and ``model_dump_json()`` raised ``PydanticSerializationError``.

The fix removed the deprecation ``@property fit_html`` getter (the field was
re-intentionally re-added for ``RegexExtractionStrategy``) and removed the
``str(result[key])`` stringify loop in ``model_dump()``.

Run with:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/regression/test_fit_html_property_leak.py
"""
import json

import pytest

from crawl4ai.models import CrawlResult, MarkdownGenerationResult


URL = "https://example.com"
HTML = "<html><head><title>x</title></head><body><p>hello</p></body></html>"


def _has_property_object(value):
    """Recursively check whether ``value`` (or any nested dict/list value)
    contains a ``property`` descriptor — the implementation-artifact that must
    never reach serialized output."""
    if isinstance(value, property):
        return True
    if isinstance(value, dict):
        return any(_has_property_object(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_property_object(v) for v in value)
    return False


class TestFitHtmlFieldDefault:
    """The pydantic field default must be ``None``, not a ``property``.

    Guards against re-introducing a same-named ``@property`` on the class,
    which pydantic v2 silently captures as the field's default value.
    """

    def test_field_default_is_none(self):
        assert CrawlResult.model_fields["fit_html"].default is None

    def test_field_default_is_not_property_descriptor(self):
        assert not isinstance(CrawlResult.model_fields["fit_html"].default, property)


class TestBareConstruction:
    """Constructing ``CrawlResult`` without ``fit_html=`` was the primary
    trigger for the property-descriptor leak (multiple paths in
    ``async_webcrawler.py`` construct bare ``CrawlResult(...)``)."""

    def test_bare_construction_fit_html_is_none(self):
        cr = CrawlResult(url=URL, html=HTML, success=True)
        assert cr.fit_html is None
        assert not isinstance(cr.fit_html, property)

    def test_model_dump_fit_html_is_none_and_garbage_free(self):
        cr = CrawlResult(url=URL, html=HTML, success=True)
        assert cr.model_dump()["fit_html"] is None
        assert _has_property_object(cr.model_dump()) is False

    def test_model_dump_json_round_trips_and_emits_null(self):
        """Latent failure mode: ``model_dump_json()`` previously raised
        ``PydanticSerializationError`` because a ``property`` object is not
        serializable by pydantic's recommended API.  It must now succeed."""
        from pydantic_core import PydanticSerializationError

        cr = CrawlResult(url=URL, html=HTML, success=True)
        try:
            parsed = json.loads(cr.model_dump_json())
        except PydanticSerializationError as e:
            pytest.fail(f"model_dump_json() raised PydanticSerializationError: {e}")
        assert parsed["fit_html"] is None

    def test_explicit_fit_html_round_trips_through_both_serializers(self):
        cr = CrawlResult(url=URL, html=HTML, success=True, fit_html="<p>fit</p>")
        assert cr.fit_html == "<p>fit</p>"
        assert cr.model_dump()["fit_html"] == "<p>fit</p>"
        assert json.loads(cr.model_dump_json())["fit_html"] == "<p>fit</p>"


class TestDeprecatedPropertiesPreserved:
    """The sibling deprecation ``@property`` accessors (``fit_markdown``,
    ``markdown_v2``) remain defined and must still raise ``AttributeError``
    to direct users to ``markdown.*``.  They must NOT leak into
    ``model_dump()`` output because they are properties, not fields."""

    def test_fit_markdown_access_raises_attribute_error(self):
        cr = CrawlResult(url=URL, html=HTML, success=True)
        with pytest.raises(AttributeError):
            cr.fit_markdown

    def test_markdown_v2_access_raises_attribute_error(self):
        cr = CrawlResult(url=URL, html=HTML, success=True)
        with pytest.raises(AttributeError):
            cr.markdown_v2

    def test_deprecated_property_names_absent_from_model_dump(self):
        cr = CrawlResult(url=URL, html=HTML, success=True)
        assert "fit_markdown" not in cr.model_dump()
        assert "markdown_v2" not in cr.model_dump()


class TestMarkdownPrivateAttrSerialization:
    """The ``_markdown`` ``PrivateAttr`` is still serialized into the
    ``markdown`` key by the ``model_dump()`` override — confirmed unaffected
    by removing the descriptor-stringify loop."""

    def test_markdown_serialized_when_set(self):
        mr = MarkdownGenerationResult(
            raw_markdown="# hi",
            markdown_with_citations="",
            references_markdown="",
            fit_markdown="hi",
            fit_html="<p>hi</p>",
        )
        cr = CrawlResult(url=URL, html=HTML, success=True, markdown=mr)
        d = cr.model_dump()
        assert d["markdown"]["fit_html"] == "<p>hi</p>"
        assert d["markdown"]["raw_markdown"] == "# hi"
        assert _has_property_object(d) is False

    def test_markdown_absent_from_dump_when_not_set(self):
        cr = CrawlResult(url=URL, html=HTML, success=True)
        assert "markdown" not in cr.model_dump()


@pytest.mark.asyncio
async def test_crawl_web_exception_path_returns_none_fit_html():
    """End-to-end regression: the ``_crawl_web`` exception handler constructs a
    bare ``CrawlResult`` (no ``fit_html=``) on any unhandled exception during
    crawling.  The serialized ``fit_html`` must be ``None``, not the garbage
    ``"<property object at 0x...>"`` string."""
    from unittest.mock import AsyncMock, MagicMock

    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    crawler = AsyncWebCrawler(verbose=False)
    crawler.crawler_strategy = MagicMock()
    crawler.crawler_strategy.crawl = AsyncMock(side_effect=RuntimeError("simulated"))
    result = await crawler.arun(url="https://example.com", config=CrawlerRunConfig())

    assert result.success is False
    d = result.model_dump()
    assert d["success"] is False
    assert d["fit_html"] is None
    payload = json.dumps(d)
    assert "<property object" not in payload
    assert json.loads(payload)["fit_html"] is None
