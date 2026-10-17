"""``CrawlResult.model_dump_json()`` must include the ``markdown`` field.

Regression for the silent omission of ``markdown`` from JSON
serialization introduced in commit a9e24307 ("Release prep (#749)").
The PR refactored ``CrawlResult.markdown`` from a regular pydantic
field into a private attribute (``_markdown``) and compensated with a
Python-level ``model_dump()`` override. Because Pydantic v2's
``model_dump_json()`` dispatches via the core-schema serializer and does
NOT call ``model_dump()``, the override was invisible to JSON
serialization — ``model_dump_json()`` silently dropped ``markdown``. The
same root cause also made the override ignore ``include``/``exclude``
for the injected ``markdown`` key (it injected unconditionally whenever
``_markdown`` was not None).

The fix replaces the ``model_dump`` override with a
``@model_serializer(mode="wrap")`` that injects ``markdown`` inside the
core schema with access to ``SerializationInfo`` for
``include`` / ``exclude`` / ``mode``.

Run with:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/regression/test_crawl_result_model_dump_json_markdown.py
"""
import json

import pytest
from pydantic import BaseModel

from crawl4ai.models import (
    CrawlResult,
    MarkdownGenerationResult,
)


URL = "https://example.com"
HTML = "<html><head><title>x</title></head><body><p>hello</p></body></html>"


class _Outer(BaseModel):
    """Reference model that carries a ``MarkdownGenerationResult`` as a
    *standard* nested field. Used to assert that ``CrawlResult``'s injected
    ``markdown`` sub-object matches pydantic's own field-serialization path
    under each forwarded serialization option."""

    inner: MarkdownGenerationResult


def _md(raw="hello", citations="c", refs="r", fit="fm", fithtml="<p>fh</p>"):
    return MarkdownGenerationResult(
        raw_markdown=raw,
        markdown_with_citations=citations,
        references_markdown=refs,
        fit_markdown=fit,
        fit_html=fithtml,
    )


def _result(md=None, **extra) -> CrawlResult:
    kwargs = {"url": URL, "html": HTML, "success": True}
    if md is not None:
        kwargs["markdown"] = md
    kwargs.update(extra)
    return CrawlResult(**kwargs)


class TestModelDumpJsonIncludesMarkdown:
    """``model_dump_json()`` must emit ``markdown`` whenever ``_markdown``
    is set, with the full ``MarkdownGenerationResult`` sub-fields."""

    def test_model_dump_json_includes_markdown_when_set(self):
        """Headline regression: ``"markdown" in json.loads(model_dump_json())``
        was False before the fix; it must now be True, and all five
        ``MarkdownGenerationResult`` sub-fields must round-trip through JSON."""
        cr = _result(md=_md(
            raw="# Heading",
            citations="# Heading \u20064\u2007",
            refs="[1] https://example.com",
            fit="# Fit",
            fithtml="<article>fit</article>",
        ))
        md = json.loads(cr.model_dump_json())["markdown"]
        assert md["raw_markdown"] == "# Heading"
        assert md["markdown_with_citations"] == "# Heading \u20064\u2007"
        assert md["references_markdown"] == "[1] https://example.com"
        assert md["fit_markdown"] == "# Fit"
        assert md["fit_html"] == "<article>fit</article>"

    def test_model_dump_json_omits_markdown_when_not_set(self):
        """When ``_markdown`` is None, ``model_dump_json()`` must NOT
        inject a spurious ``"markdown": null`` entry."""
        cr = _result()
        assert "markdown" not in json.loads(cr.model_dump_json())


class TestSerializationParity:
    """Both standard pydantic serialization methods must agree on
    ``markdown`` and honor ``include``/``exclude`` on both surfaces —
    the kwargs-dropping corollary was the secondary defect from the same
    root cause."""

    def test_model_dump_mode_json_matches_model_dump_json(self):
        """The wrap serializer must forward ``mode`` to the nested
        ``MarkdownGenerationResult.model_dump(...)`` so the dict surface
        with ``mode='json'`` matches the JSON surface exactly."""
        cr = _result(md=_md())
        assert cr.model_dump(mode="json") == json.loads(cr.model_dump_json())

    def test_model_dump_json_exclude_markdown_drops_key(self):
        cr = _result(md=_md())
        assert "markdown" not in json.loads(cr.model_dump_json(exclude={"markdown"}))

    def test_model_dump_json_exclude_other_field_keeps_markdown(self):
        """Excluding an unrelated top-level field must not silently drop
        ``markdown`` — the headline bug."""
        cr = _result(md=_md())
        out = json.loads(cr.model_dump_json(exclude={"url"}))
        assert "url" not in out
        assert "markdown" in out

    def test_model_dump_json_include_markdown_drops_others(self):
        """``include={'markdown'}`` emits only ``markdown`` — the regular
        top-level fields that previously leaked must be excluded."""
        cr = _result(md=_md())
        assert set(json.loads(cr.model_dump_json(include={"markdown"})).keys()) == {"markdown"}

    def test_model_dump_exclude_markdown_drops_key(self):
        """Corollary bug: the previous ``model_dump`` override injected
        ``markdown`` unconditionally and ignored ``exclude`` for the
        injected key. The wrap serializer must honor it."""
        cr = _result(md=_md())
        assert "markdown" not in cr.model_dump(exclude={"markdown"})

    def test_model_dump_include_markdown_excludes_others(self):
        """Corollary bug: ``include={'markdown'}`` used to leak the
        other top-level fields (those the previous ``super().model_dump()``
        returned). It must now emit only ``markdown``."""
        cr = _result(md=_md())
        assert set(cr.model_dump(include={"markdown"}).keys()) == {"markdown"}


class TestSequenceSpecIncludeExclude:
    """Regression for the crash from commit ``ffa868b8``: the ``_whole``
    helper in ``_serialize_markdown`` indexed ``spec["markdown"]`` for any
    non-set spec whose membership check passed, so a list/tuple ``exclude``
    naming ``markdown`` raised ``TypeError`` (wrapped as
    ``PydanticSerializationError``), aborting the entire
    ``model_dump``/``model_dump_json``. Pre-``ffa868b8`` (commit
    ``2e9c1743``) treated list/tuple specs as whole-field exclusion, matching
    pydantic's own field semantics; these tests pin that restored behavior.
    Pydantic's runtime honors sequence specs even though its ``IncEx`` type
    alias only declares set/Mapping forms."""

    def test_model_dump_exclude_list_drops_markdown(self):
        """``model_dump(exclude=["markdown"])`` omits ``markdown`` (was a crash)."""
        cr = _result(md=_md())
        out = cr.model_dump(exclude=["markdown"])
        assert "markdown" not in out and out.get("url") and out.get("html")

    def test_model_dump_exclude_tuple_drops_markdown(self):
        """``model_dump(exclude=("markdown",))`` omits ``markdown`` (was a crash)."""
        cr = _result(md=_md())
        out = cr.model_dump(exclude=("markdown",))
        assert "markdown" not in out and out.get("url") and out.get("html")

    def test_model_dump_json_exclude_list_drops_markdown(self):
        """``model_dump_json(exclude=["markdown"])`` omits ``markdown`` (was a crash)."""
        cr = _result(md=_md())
        out = json.loads(cr.model_dump_json(exclude=["markdown"]))
        assert "markdown" not in out and out.get("url")

    def test_model_dump_exclude_list_other_field_keeps_markdown(self):
        """A list exclude NOT naming ``markdown`` keeps it and drops only the named field."""
        cr = _result(md=_md())
        out = cr.model_dump(exclude=["url"])
        assert "url" not in out and "markdown" in out


class TestNestedDictIncludeExclude:
    """Pydantic v2's ``include``/``exclude`` accept a dict-valued nested
    spec (``{'markdown': {'fit_markdown'}}``) to select sub-fields of a
    nested model. A normal pydantic field of type ``MarkdownGenerationResult``
    honors this; ``CrawlResult.markdown`` (a ``PrivateAttr`` re-injected by
    the wrap serializer) must honor it too.

    Regression for the gap introduced in commit 8cfcb25c: the wrap
    serializer's ``"markdown" in info.include``/``info.exclude`` membership
    checks treated the dict-valued nested form as whole-field control and
    never forwarded the sub-field spec to the inner ``model_dump``, so
    nested-``exclude`` dropped the entire ``markdown`` key and
    nested-``include`` emitted all sub-fields.
    """

    def _normal(self):
        """A plain ``BaseModel`` with a normal ``MarkdownGenerationResult``
        field, used to pin the behavior ``CrawlResult.markdown`` must match."""
        from pydantic import BaseModel

        class Normal(BaseModel):
            n: MarkdownGenerationResult

        return Normal(n=_md())

    def test_nested_exclude_drops_only_named_subfield_model_dump(self):
        """``exclude={'markdown': {'fit_markdown'}}`` must keep the
        ``markdown`` key with only ``fit_markdown`` removed — not drop
        ``markdown`` entirely (the pre-fix behavior)."""
        cr = _result(md=_md())
        out = cr.model_dump(mode="json", exclude={"markdown": {"fit_markdown"}})
        assert "markdown" in out
        md = out["markdown"]
        assert set(md.keys()) == {
            "raw_markdown", "markdown_with_citations",
            "references_markdown", "fit_html",
        }
        assert md["raw_markdown"] == "hello"
        assert md["fit_html"] == "<p>fh</p>"

    def test_nested_exclude_drops_only_named_subfield_matches_normal_field(self):
        """The nested-``exclude`` output of ``CrawlResult.markdown`` must
        match that of a normal pydantic field of the same type."""
        cr = _result(md=_md())
        actual = cr.model_dump(mode="json", exclude={"markdown": {"fit_markdown"}})["markdown"]
        expected = self._normal().model_dump(
            mode="json", exclude={"n": {"fit_markdown"}}
        )["n"]
        assert actual == expected

    def test_nested_include_keeps_only_named_subfield_model_dump(self):
        """``include={'markdown': {'raw_markdown'}}`` must emit only
        ``raw_markdown`` under ``markdown`` — not all sub-fields (the
        pre-fix behavior)."""
        cr = _result(md=_md())
        out = cr.model_dump(mode="json", include={"markdown": {"raw_markdown"}})
        assert set(out.keys()) == {"markdown"}
        assert set(out["markdown"].keys()) == {"raw_markdown"}
        assert out["markdown"]["raw_markdown"] == "hello"

    def test_nested_include_keeps_only_named_subfield_matches_normal_field(self):
        """The nested-``include`` output of ``CrawlResult.markdown`` must
        match that of a normal pydantic field of the same type."""
        cr = _result(md=_md())
        actual = cr.model_dump(mode="json", include={"markdown": {"raw_markdown"}})["markdown"]
        expected = self._normal().model_dump(
            mode="json", include={"n": {"raw_markdown"}}
        )["n"]
        assert actual == expected

    def test_nested_exclude_on_model_dump_json(self):
        """Nested-``exclude`` must work on the JSON surface too — the
        wrap serializer applies to both ``model_dump`` and
        ``model_dump_json``."""
        cr = _result(md=_md())
        out = json.loads(cr.model_dump_json(exclude={"markdown": {"fit_markdown"}}))
        assert "markdown" in out
        assert "fit_markdown" not in out["markdown"]
        assert out["markdown"]["raw_markdown"] == "hello"

    def test_nested_include_on_model_dump_json(self):
        """Nested-``include`` must work on the JSON surface too."""
        cr = _result(md=_md())
        out = json.loads(cr.model_dump_json(include={"markdown": {"raw_markdown"}}))
        assert set(out.keys()) == {"markdown"}
        assert set(out["markdown"].keys()) == {"raw_markdown"}

    def test_no_markdown_set_nested_spec_does_not_inject_key(self):
        """When ``_markdown`` is None, a nested spec on ``markdown`` must
        not synthesize a spurious ``"markdown": null`` entry."""
        cr = _result()
        out = cr.model_dump(mode="json", include={"markdown": {"raw_markdown"}})
        assert "markdown" not in out

    def test_whole_field_markers_match_normal_pydantic_field(self):
        cr = _result(md=_md())
        assert cr.model_dump(mode="json", include={"markdown": ...}) == {
            "markdown": self._normal().model_dump(mode="json")["n"]
        }
        assert "markdown" not in cr.model_dump(
            mode="json", exclude={"markdown": ...}
        )


class TestRoundTrip:
    """``model_dump_json()`` output must let Pydantic reconstruct
    ``_markdown`` via the existing custom ``__init__`` pop-and-wrap.
    Previously the JSON never carried ``markdown``, so
    ``model_validate_json`` could not reconstruct it."""

    def test_model_validate_json_reconstructs_full_markdown(self):
        cr = _result(md=_md(
            raw="# Hi",
            citations="# Hi \u20064",
            refs="## Refs",
            fit="# Fit",
            fithtml="<a>fit</a>",
        ))
        cr2 = CrawlResult.model_validate_json(cr.model_dump_json())
        assert cr2._markdown is not None
        assert isinstance(cr2._markdown, MarkdownGenerationResult)
        assert cr2._markdown.raw_markdown == "# Hi"
        assert cr2._markdown.markdown_with_citations == "# Hi \u20064"
        assert cr2._markdown.references_markdown == "## Refs"
        assert cr2._markdown.fit_markdown == "# Fit"
        assert cr2._markdown.fit_html == "<a>fit</a>"


class TestNoRegressionOnOtherFields:
    """The wrap serializer must not disturb other top-level fields,
    particularly the tricky ones: ``pdf: Optional[bytes]`` and the
    ``SSLCertificate`` dict-subclass under ``arbitrary_types_allowed=True``.
    Also guards against re-leaking the deprecated ``@property``
    ``fit_markdown`` / ``markdown_v2`` accessors into JSON output."""

    def test_all_regular_top_level_fields_present_in_json(self):
        cr = _result(
            md=_md(),
            fit_html="<p>fit</p>",
            cleaned_html="<clean>",
            media={"images": [{"src": "http://x/y.png"}]},
            links={"internal": [{"href": "/a", "text": "a"}]},
            metadata={"title": "T"},
            response_headers={"etag": "abc"},
            status_code=200,
        )
        out = json.loads(cr.model_dump_json())
        for field in (
            "url", "html", "fit_html", "success", "cleaned_html", "media",
            "links", "metadata", "response_headers", "status_code",
        ):
            assert field in out, f"missing top-level field: {field}"
        assert out["markdown"]["raw_markdown"] == "hello"

    def test_bytes_pdf_field_serializes_in_json_without_error(self):
        """``pdf: Optional[bytes]`` is the trickiest non-markdown field
        to serialize to JSON; the wrap serializer must not interfere
        with the core schema's bytes handling."""
        from pydantic_core import PydanticSerializationError

        cr = _result(md=_md(), pdf=b"%PDF-1.4 binary")
        try:
            out = json.loads(cr.model_dump_json())
        except PydanticSerializationError as e:
            pytest.fail(f"model_dump_json() raised PydanticSerializationError: {e}")
        assert "pdf" in out

    def test_ssl_certificate_field_serializes_in_json(self):
        """The wrap serializer must not interfere with nested model
        serialization of ``SSLCertificate`` (a ``dict`` subclass under
        ``arbitrary_types_allowed=True``)."""
        from crawl4ai.ssl_certificate import SSLCertificate

        cert = SSLCertificate({
            "issuer": {"O": "Test"},
            "subject": {"O": "Test"},
            "valid_from": "2025-01-01T00:00:00",
            "valid_to": "2026-01-01T00:00:00",
            "fingerprint": "aa:bb",
        })
        cr = _result(md=_md(), ssl_certificate=cert)
        out = json.loads(cr.model_dump_json())
        assert "ssl_certificate" in out
        assert out["ssl_certificate"]["issuer"]["O"] == "Test"

    def test_deprecated_property_names_still_absent_from_json(self):
        """``fit_markdown`` / ``markdown_v2`` are deprecation
        ``@property`` accessors, not fields — they must not appear in the
        JSON output."""
        cr = _result()
        out = json.loads(cr.model_dump_json())
        assert "fit_markdown" not in out
        assert "markdown_v2" not in out


class TestSerializationOptionForwarding:
    """The ``@model_serializer(mode="wrap")`` must forward the full
    ``SerializationInfo`` option set to the nested
    ``MarkdownGenerationResult.model_dump(...)`` call — not just ``mode``.

    Before the fix the wrap serializer only forwarded ``mode`` (and the
    flat ``include``/``exclude`` membership check), silently dropping
    ``exclude_none`` / ``exclude_defaults`` / ``exclude_unset`` /
    ``by_alias`` / ``round_trip``. The injected ``markdown`` sub-object
    therefore violated those serialization options while the rest of the
    model honored them.

    The strongest guarantee here is *parity with a standard nested pydantic
    model*: serializing a ``CrawlResult`` carrying a ``MarkdownGenerationResult``
    under a given option must produce the same ``markdown`` sub-object that
    serializing a reference ``_Outer`` model carrying that same object as a
    regular nested field would produce. This contract is asserted directly
    against pydantic's own field-serialization path on both the dict and JSON
    surfaces for the three boolean exclusion flags; ``by_alias`` and
    ``round_trip`` (no-ops for this alias-less model) are exercised to pin
    the forwarding behavior and the ``by_alias=None`` default.
    """

    def test_exclude_none_matches_standard_nested_model(self):
        """Under ``exclude_none`` the injected ``markdown`` sub-object must
        equal what a standard nested pydantic field carrying the same
        ``MarkdownGenerationResult`` would emit, on both the dict and JSON
        surfaces. This is the headline regression: previously
        ``"fit_markdown": null`` / ``"fit_html": null`` survived inside the
        injected ``markdown`` sub-object."""
        md = MarkdownGenerationResult(
            raw_markdown="# Hi", markdown_with_citations="c",
            references_markdown="r", fit_markdown=None, fit_html=None,
        )
        cr = _result(md=md)
        outer = _Outer(inner=md)
        assert cr.model_dump(exclude_none=True)["markdown"] == outer.model_dump(exclude_none=True)["inner"]
        assert (
            json.loads(cr.model_dump_json(exclude_none=True))["markdown"]
            == json.loads(outer.model_dump_json(exclude_none=True))["inner"]
        )

    def test_exclude_defaults_matches_standard_nested_model(self):
        """``exclude_defaults`` drops markdown sub-fields whose values equal
        their defaults (``fit_markdown``/``fit_html`` default to ``None``).
        The injected sub-object must match a standard nested field's
        behavior on both surfaces."""
        md = _md(raw="# Hi", citations="c", refs="r", fit=None, fithtml=None)
        cr = _result(md=md)
        outer = _Outer(inner=md)
        assert cr.model_dump(exclude_defaults=True)["markdown"] == outer.model_dump(exclude_defaults=True)["inner"]

    def test_exclude_unset_matches_standard_nested_model(self):
        """``exclude_unset`` drops markdown sub-fields that were not explicitly
        provided when building the ``MarkdownGenerationResult``. The injected
        sub-object must match a standard nested field's behavior on both
        surfaces."""
        md = MarkdownGenerationResult(
            raw_markdown="# Hi", markdown_with_citations="c", references_markdown="r"
        )
        cr = _result(md=md)
        outer = _Outer(inner=md)
        assert cr.model_dump(exclude_unset=True)["markdown"] == outer.model_dump(exclude_unset=True)["inner"]

    def test_by_alias_forwarding_is_noop_without_aliases(self):
        """``MarkdownGenerationResult`` declares no aliases, so
        ``by_alias=True`` must be a no-op for the injected ``markdown``
        sub-object and must not raise. This pins that the forwarded
        ``by_alias`` flag (which is ``None`` by default, not ``False``) is
        passed through without corrupting the un-aliased path."""
        cr = _result(md=_md())
        assert cr.model_dump(by_alias=True)["markdown"] == cr.model_dump()["markdown"]
        assert (
            json.loads(cr.model_dump_json(by_alias=True))["markdown"]
            == json.loads(cr.model_dump_json())["markdown"]
        )

    def test_round_trip_forwarding_preserves_markdown(self):
        """``round_trip=True`` must serialize the ``markdown`` sub-object in
        a way that round-trips back through ``model_validate``."""
        cr = _result(md=_md(raw="# Hi", citations="c", refs="r", fit="fm", fithtml="<a>"))
        dumped = cr.model_dump(round_trip=True)
        rebuilt = CrawlResult.model_validate(dumped)
        assert rebuilt._markdown is not None
        assert rebuilt._markdown.raw_markdown == "# Hi"
        assert rebuilt._markdown.fit_markdown == "fm"
        assert rebuilt._markdown.fit_html == "<a>"
