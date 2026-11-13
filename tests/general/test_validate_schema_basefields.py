"""Tests for `JsonElementExtractionStrategy._validate_schema` baseFields handling.

Regression coverage for the bug where ``_validate_schema`` computed field
coverage and strict-mode expected-field checks exclusively from
``schema["fields"]``, ignoring ``schema["baseFields"]`` even though the
shared ``extract()`` method merges ``baseFields`` into every extracted
item.

The fix lives in ``crawl4ai/extraction_strategy.py`` and includes
``baseFields`` in: (a) the coverage computation, (b) the strict-mode
missing-fields check, and (c) the strict-mode success-criteria name set.

These tests are pure unit tests: no LLM, no network, no browser. They call
``_validate_schema`` (a ``@staticmethod``) directly and assert on the
returned diagnostic dict.
"""

from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    JsonElementExtractionStrategy,
    JsonXPathExtractionStrategy,
)


# ---------------------------------------------------------------------------
# Shared HTML fixtures
# ---------------------------------------------------------------------------

# Two product cards with a `data-id` attribute on the container (the
# canonical baseFields pattern documented in
# docs/md_v2/extraction/no-llm-strategies.md).
PRODUCTS_HTML = """\
<html><body>
<div class="product" data-id="1"><span class="name">Widget</span></div>
<div class="product" data-id="2"><span class="name">Gadget</span></div>
</body></html>
"""

# Variant where `data-id` is missing on every card — exercises the
# "baseField present in schema but never populated" path.
PRODUCTS_HTML_NO_ATTR = """\
<html><body>
<div class="product"><span class="name">Widget</span></div>
<div class="product"><span class="name">Gadget</span></div>
</body></html>
"""

# HTML where the baseSelector matches zero elements.
EMPTY_HTML = "<html><body><p>nothing here</p></body></html>"


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------

def _schema_mixed(selector_kind="css"):
    """Schema with one baseField (`id`) and one field (`name`) — both populated."""
    if selector_kind == "css":
        name_selector = ".name"
        base_selector = ".product"
    else:
        name_selector = ".//span[@class='name']"
        base_selector = "//div[@class='product']"
    return {
        "name": "Products",
        "baseSelector": base_selector,
        "baseFields": [
            {"name": "id", "type": "attribute", "attribute": "data-id"},
        ],
        "fields": [
            {"name": "name", "selector": name_selector, "type": "text"},
        ],
    }


def _schema_basefields_only(selector_kind="css"):
    """Schema whose only field comes from baseFields — the failing-fuzzy case."""
    base_selector = ".product" if selector_kind == "css" else "//div[@class='product']"
    return {
        "name": "Products",
        "baseSelector": base_selector,
        "baseFields": [
            {"name": "id", "type": "attribute", "attribute": "data-id"},
        ],
        "fields": [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate(schema, html, schema_type="CSS", expected_fields=None):
    """Thin wrapper around the staticmethod under test."""
    return JsonElementExtractionStrategy._validate_schema(
        schema, html, schema_type=schema_type, expected_fields=expected_fields,
    )


def _extract(schema, html, strategy_cls=JsonCssExtractionStrategy):
    """Run the real extractor — used to assert consistency with validation."""
    return strategy_cls(schema).extract(url="", html_content=html)


# ===========================================================================
# Strict mode (expected_fields provided)
# ===========================================================================


class TestStrictModeBaseFields:
    """Strict mode must accept fields implemented via baseFields."""

    def test_strict_basefields_expected_field_succeeds(self):
        """`id` as baseField, `name` as field, both expected & populated.

        Before the fix, the validator reported `id` as missing from the
        schema and returned ``success=False``. After the fix it must
        return ``success=True`` with no missing-field issue.
        """
        schema = _schema_mixed()
        result = _validate(schema, PRODUCTS_HTML, "CSS",
                           expected_fields=["id", "name"])
        assert result["success"] is True
        assert result["total_fields"] == 2
        assert result["populated_fields"] == 2
        assert result["issues"] == []
        # Both expected field names must appear in the coverage table.
        names = {fd["name"] for fd in result["field_details"]}
        assert {"id", "name"}.issubset(names)

    def test_strict_missing_expected_field_still_flagged(self):
        """An expected field in NEITHER `fields` nor `baseFields` is still flagged.

        Guards against an over-broad fix that would silently accept any
        expected-field set.
        """
        schema = _schema_mixed()
        result = _validate(schema, PRODUCTS_HTML, "CSS",
                           expected_fields=["id", "name", "price"])
        assert any("Expected fields missing from schema" in i and "price" in i
                    for i in result["issues"]), result["issues"]
        assert result["success"] is False

    def test_strict_basefield_unpopulated_fails(self):
        """Expected baseField present in schema but its attribute is absent.

        Before the fix, validation reported `id` as "missing from schema"
        (wrong reason). After the fix, `id` is recognised as a schema
        field but counted as unpopulated, so ``success=False`` for the
        right reason.
        """
        schema = _schema_mixed()
        result = _validate(schema, PRODUCTS_HTML_NO_ATTR, "CSS",
                           expected_fields=["id", "name"])
        assert result["success"] is False
        # `id` MUST NOT be in the "missing" issue (it IS in the schema now).
        for issue in result["issues"]:
            assert "Expected fields missing from schema" not in issue or "id" not in issue, \
                f"baseField 'id' wrongly flagged as missing: {result['issues']}"
        # `id` must appear in field_details (proving it's counted) but unpopulated.
        id_details = [fd for fd in result["field_details"] if fd["name"] == "id"]
        assert len(id_details) == 1
        assert id_details[0]["populated_count"] == 0


# ===========================================================================
# Fuzzy mode (expected_fields=None)
# ===========================================================================


class TestFuzzyModeBaseFields:
    """Fuzzy mode must accept schemas that extract data via baseFields."""

    def test_fuzzy_basefields_only_succeeds(self):
        """A schema whose data comes purely from baseFields must succeed.

        Before the fix: ``success=False``, ``total_fields=0``,
        ``populated_fields=0``, and the bogus issue ``"All fields returned
        None/empty — selectors likely wrong"`` even though `extract()`
        returned the IDs. After the fix: ``success=True``, ``total_fields
        == 1``, ``populated_fields == 1``, no issues.
        """
        schema = _schema_basefields_only()
        # Sanity: the actual extractor must pull out the IDs.
        items = _extract(schema, PRODUCTS_HTML)
        assert items == [{"id": "1"}, {"id": "2"}]

        result = _validate(schema, PRODUCTS_HTML, "CSS")
        assert result["success"] is True
        assert result["total_fields"] == 1
        assert result["populated_fields"] == 1
        assert "All fields returned None/empty — selectors likely wrong" not in \
            "\n".join(result["issues"])
        assert result["issues"] == []


# ===========================================================================
# Coverage / counts
# ===========================================================================


class TestCoverageCountsWithBaseFields:
    """`total_fields`, `populated_fields`, `field_details`."""

    def test_total_fields_counts_both_lists(self):
        """``total_fields == len(fields) + len(baseFields)`` for distinct names."""
        schema = {
            "name": "Wide",
            "baseSelector": ".product",
            "baseFields": [
                {"name": "id", "type": "attribute", "attribute": "data-id"},
                {"name": "klass", "type": "attribute", "attribute": "class"},
            ],
            "fields": [
                {"name": "name", "selector": ".name", "type": "text"},
            ],
        }
        result = _validate(schema, PRODUCTS_HTML, "CSS")
        assert result["total_fields"] == 3
        # `class` is always populated (BS4 returns the class string), so
        # all three populate here.
        assert result["populated_fields"] == 3
        assert result["field_coverage"] == 1.0

    def test_field_details_contains_basefields(self):
        """Every baseField has its own entry in ``field_details`` with correct counts."""
        schema = _schema_mixed()
        result = _validate(schema, PRODUCTS_HTML, "CSS")
        details = {fd["name"]: fd for fd in result["field_details"]}
        assert "id" in details
        assert "name" in details
        # Both base elements had `data-id`, so populated_count == 2.
        assert details["id"]["populated_count"] == 2
        assert details["id"]["total_count"] == 2
        # Sample value is the (truncated) string of the first non-empty value.
        assert details["id"]["sample_value"] == "1"
        assert details["name"]["populated_count"] == 2
        assert details["name"]["sample_value"] == "Widget"


# ===========================================================================
# XPath symmetry — the fix lives in the base class
# ===========================================================================


class TestXPathSymmetry:
    """The fix must also work for ``schema_type='XPATH'``.

    The fix is in the base ``JsonElementExtractionStrategy._validate_schema``
    method, which instantiates the right subclass based on ``schema_type``;
    a CSS-only regression test would miss future base-class regressions
    that break XPath.
    """

    def test_xpath_strict_basefields_succeeds(self):
        schema = _schema_mixed(selector_kind="xpath")
        result = _validate(schema, PRODUCTS_HTML, "XPATH",
                           expected_fields=["id", "name"])
        assert result["success"] is True
        assert result["total_fields"] == 2
        assert result["populated_fields"] == 2
        assert result["issues"] == []

    def test_xpath_fuzzy_basefields_only_succeeds(self):
        schema = _schema_basefields_only(selector_kind="xpath")
        items = _extract(schema, PRODUCTS_HTML, strategy_cls=JsonXPathExtractionStrategy)
        assert items == [{"id": "1"}, {"id": "2"}]
        result = _validate(schema, PRODUCTS_HTML, "XPATH")
        assert result["success"] is True
        assert result["total_fields"] == 1
        assert result["populated_fields"] == 1
        assert result["issues"] == []

    def test_xpath_strict_missing_expected_field_flagged(self):
        """XPath strict mode still flags genuinely-missing expected fields."""
        schema = _schema_mixed(selector_kind="xpath")
        result = _validate(schema, PRODUCTS_HTML, "XPATH",
                           expected_fields=["id", "name", "price"])
        assert result["success"] is False
        assert any("price" in i for i in result["issues"])


# ===========================================================================
# Feedback message integration (no LLM, just consuming _validate_schema output)
# ===========================================================================


class TestFeedbackMessageBaseFields:
    """Feedback to the LLM must not be the bogus "selectors likely wrong".

    A stated impact of the bug was that `_build_feedback_message` turned
    the bogus "All fields returned None/empty" verdict into a prompt that
    told the LLM its selectors were wrong and asked it to find correct
    child selectors — degrading an initially-correct schema across
    refinement rounds. These tests guard against that regression.
    """

    def test_feedback_no_bogus_selectors_wrong(self):
        """A valid baseFields-only schema must not generate 'selectors wrong' feedback."""
        schema = _schema_basefields_only()
        result = _validate(schema, PRODUCTS_HTML, "CSS")
        feedback = JsonElementExtractionStrategy._build_feedback_message(
            result, schema, attempt=1, is_repeated=False
        )
        assert "All fields returned None/empty" not in feedback
        # No "Here is the innerHTML of the first base element" advice block either.
        assert "use it to find correct child selectors" not in feedback

    def test_feedback_table_includes_basefields(self):
        """The coverage table in the feedback must list the baseField row."""
        schema = _schema_mixed()
        result = _validate(schema, PRODUCTS_HTML, "CSS",
                           expected_fields=["id", "name"])
        feedback = JsonElementExtractionStrategy._build_feedback_message(
            result, schema, attempt=1, is_repeated=False
        )
        assert "| Field | Populated | Sample |" in feedback
        # Rows for both `id` (the baseField) and `name` (the regular field) exist.
        assert "| id |" in feedback
        assert "| name |" in feedback


# ===========================================================================
# Consistency: extract() vs _validate_schema
# ===========================================================================


class TestExtractValidationConsistency:
    """The validator's verdict must match what `extract()` actually produced.

    The whole purpose of `_validate_schema` is to diagnose whether the
    extractor works; if the two ever drift, the diagnostic is worse than
    useless (it actively degrades correct schemas in the LLM refinement
    loop).
    """

    def test_extract_success_implies_fuzzy_success(self):
        """If extract() returns items with at least one populated value, fuzzy==True."""
        for selector_kind, strategy_cls in (("css", JsonCssExtractionStrategy),
                                             ("xpath", JsonXPathExtractionStrategy)):
            schema = _schema_mixed(selector_kind=selector_kind)
            items = _extract(schema, PRODUCTS_HTML, strategy_cls=strategy_cls)
            assert items, f"[{selector_kind}] extract produced no items"
            assert any(any(v not in (None, "") for v in item.values()) for item in items), \
                f"[{selector_kind}] no item had a populated value"
            result = _validate(schema, PRODUCTS_HTML, selector_kind.upper())
            assert result["success"] is True, \
                f"[{selector_kind}] validator said fail but extract produced data: {result}"

    def test_strict_success_when_all_expected_populated(self):
        """If extract items populate every expected field, strict==True."""
        schema = _schema_mixed()
        items = _extract(schema, PRODUCTS_HTML)
        for item in items:
            assert item.get("id") and item.get("name")
        result = _validate(schema, PRODUCTS_HTML, "CSS",
                           expected_fields=["id", "name"])
        assert result["success"] is True

    def test_strict_fail_when_extract_missing_an_expected_basefield(self):
        """If extract items lack an expected baseField value, strict==False (right reason)."""
        schema = _schema_mixed()
        items = _extract(schema, PRODUCTS_HTML_NO_ATTR)
        # `id` is missing on every item now.
        assert all("id" not in item for item in items)
        result = _validate(schema, PRODUCTS_HTML_NO_ATTR, "CSS",
                           expected_fields=["id", "name"])
        assert result["success"] is False


# ===========================================================================
# Edge cases that prevent specific footguns
# ===========================================================================


class TestEdgeCases:
    """Pathological inputs must not crash and must report correct coverage."""

    def test_duplicate_name_in_both_lists(self):
        """A name appearing in BOTH `fields` and `baseFields` must not crash.

        ``extract()`` lets `fields` override `baseFields` for the same
        name. ``_validate_schema`` MUST not crash; the verdict in strict
        mode must still be correct. This is defensive — schemas in
        practice do not define the same name in both lists, but the
        validator must not blow up if one does.
        """
        schema = {
            "name": "Dup",
            "baseSelector": ".product",
            "baseFields": [
                # `name` also exists as a regular field — different definition.
                {"name": "name", "type": "attribute", "attribute": "data-id"},
            ],
            "fields": [
                {"name": "name", "selector": ".name", "type": "text"},
            ],
        }
        # Must not raise.
        result = _validate(schema, PRODUCTS_HTML, "CSS",
                           expected_fields=["name"])
        # The `name` field is populated (via the regular field selector).
        assert result["success"] is True
        details = {fd["name"]: fd for fd in result["field_details"]}
        # `name` shows up at least once and reports at least one populated item.
        assert "name" in details
        assert details["name"]["populated_count"] > 0

    def test_basefields_outliving_regular_fields(self):
        """A baseField defined in schema, but its attribute is absent on every item.

        The validator must count the baseField in ``total_fields``, mark
        its ``populated_count`` as 0, and report it in the
        ``"Fields always empty"`` issue. Fuzzy mode succeeds because the
        regular `name` field populates.
        """
        schema = _schema_mixed()  # baseField `id`, field `name`
        result = _validate(schema, PRODUCTS_HTML_NO_ATTR, "CSS")
        details = {fd["name"]: fd for fd in result["field_details"]}
        # The baseField `id` is counted (total_fields includes it).
        assert result["total_fields"] == 2
        assert details["id"]["populated_count"] == 0
        # `name` populates → fuzzy success, but `id` flagged in always-empty issue.
        assert result["success"] is True
        assert any("id" in i and "Fields always empty" in i for i in result["issues"])


# ===========================================================================
# baseSelector matches zero elements — short-circuit unchanged
# ===========================================================================


class TestBaseSelectorNoMatchStillShortCircuits:
    """The 0-base-elements path MUST short-circuit before baseFields handling."""

    def test_zero_base_elements_skips_basefields_logic(self):
        """When the baseSelector matches nothing, the validator must NOT enter
        the field-coverage code path; it should report ``total_fields == 0``
        and a top-level-structure hint so the LLM can pick a valid selector.
        """
        schema = _schema_basefields_only()
        result = _validate(schema, EMPTY_HTML, "CSS")
        assert result["base_elements_found"] == 0
        assert result["success"] is False
        assert any("matched 0 elements" in i for i in result["issues"])
        # Short-circuit means total_fields == 0 and field_details == [].
        assert result["total_fields"] == 0
        assert result["field_details"] == []
        # The top_level_structure hint is populated so the LLM can pick a
        # valid selector.
        assert result["top_level_structure"] != ""


# ===========================================================================
# Smoke: re-run the exact scenario from the bug report
# ===========================================================================


class TestBugReportReproduction:
    """The two exact scenarios from the bug report, verbatim.

   aving these as named regression tests means any future change that
    reintroduces the bug fails at the most-precise possible reproduction.
    """

    def test_strict_mixed_schema_from_bug_report(self):
        """Strict mode on a schema with `id` as baseField and `name` as field.

        Bug-report expected (BEFORE fix): ``success=False``,
        ``total_fields=1``, ``populated_fields=1``,
        ``issues=['Expected fields missing from schema: id']``.
        After fix: ``success=True``, ``total_fields=2``,
        ``populated_fields=2``, ``issues=[]``.
        """
        schema = {
            "baseSelector": ".product",
            "baseFields": [
                {"name": "id", "type": "attribute", "attribute": "data-id"}
            ],
            "fields": [
                {"name": "name", "selector": ".name", "type": "text"}
            ],
        }
        result = _validate(schema, PRODUCTS_HTML, "CSS",
                           expected_fields=["id", "name"])
        # Verdict — the headline change.
        assert result["success"] is True
        assert result["base_elements_found"] == 2
        assert result["total_fields"] == 2
        assert result["populated_fields"] == 2
        assert result["field_coverage"] == 1.0
        assert result["issues"] == []
        # Coverage details — both fields are reported and populated.
        details = {fd["name"]: fd for fd in result["field_details"]}
        assert set(details.keys()) == {"id", "name"}
        assert details["name"]["populated_count"] == 2
        assert details["name"]["total_count"] == 2
        assert details["name"]["sample_value"] == "Widget"
        assert details["id"]["populated_count"] == 2
        assert details["id"]["total_count"] == 2
        assert details["id"]["sample_value"] == "1"
        # Sample HTML of the first base element is preserved (used for feedback).
        assert result["sample_base_html"].startswith("<div class=\"product\"")
        assert "data-id=\"1\"" in result["sample_base_html"]
        assert result["top_level_structure"] == ""

    def test_fuzzy_basefields_only_schema_from_bug_report(self):
        """Fuzzy mode on a baseFields-only schema.

        Bug-report expected (BEFORE fix): ``success=False``,
        ``total_fields=0``, ``populated_fields=0``,
        ``issues=['All fields returned None/empty — selectors likely wrong']``.
        After fix: ``success=True``, ``total_fields=1``,
        ``populated_fields=1``, ``issues=[]``.
        """
        schema = {
            "baseSelector": ".product",
            "baseFields": [
                {"name": "id", "type": "attribute", "attribute": "data-id"}
            ],
            "fields": [],
        }
        result = _validate(schema, PRODUCTS_HTML, "CSS")
        assert result["success"] is True
        assert result["total_fields"] == 1
        assert result["populated_fields"] == 1
        assert result["field_coverage"] == 1.0
        assert result["issues"] == []
        assert {fd["name"] for fd in result["field_details"]} == {"id"}
