"""Caller-supplied regex extraction steps are refused at the config boundary."""

import pytest

from crawl4ai.async_configs import (
    CrawlerRunConfig,
    Provenance,
    UntrustedConfigError,
    to_serializable_dict,
)


SCHEMA_STRATEGIES = [
    "JsonCssExtractionStrategy",
    "JsonXPathExtractionStrategy",
    "JsonLxmlExtractionStrategy",
]


def _load(params, provenance=Provenance.UNTRUSTED):
    return CrawlerRunConfig.load(
        {"type": "CrawlerRunConfig", "params": params}, provenance=provenance
    )


def _schema(field):
    return {"name": "x", "baseSelector": "p", "fields": [field]}


REGEX_FIELD = {"name": "f", "selector": "span", "type": "regex", "pattern": "(a+)+$"}
REGEX_PIPELINE_FIELD = {**REGEX_FIELD, "type": ["text", "regex"]}


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(_schema(REGEX_FIELD), id="field"),
        pytest.param(_schema(REGEX_PIPELINE_FIELD), id="pipeline"),
        *[
            pytest.param(
                _schema({
                    "name": "n", "selector": "div", "type": field_type,
                    "fields": [REGEX_FIELD],
                }),
                id=field_type,
            )
            for field_type in ("nested", "list", "nested_list")
        ],
        pytest.param(
            _schema({
                "name": "outer", "selector": "div", "type": "nested_list",
                "fields": [{
                    "name": "inner", "selector": "div", "type": "nested",
                    "fields": [REGEX_PIPELINE_FIELD],
                }],
            }),
            id="recursive-pipeline",
        ),
        pytest.param(
            {"baseSelector": "p", "baseFields": [REGEX_FIELD], "fields": []},
            id="base-field",
        ),
        pytest.param(
            {"baseSelector": "p", "baseFields": [REGEX_PIPELINE_FIELD], "fields": []},
            id="base-pipeline",
        ),
    ],
)
@pytest.mark.parametrize("strategy", SCHEMA_STRATEGIES)
@pytest.mark.parametrize("enveloped", [False, True], ids=["plain", "enveloped"])
def test_untrusted_schema_regex_field_refused(strategy, schema, enveloped):
    encoded_schema = to_serializable_dict(schema) if enveloped else schema

    with pytest.raises(UntrustedConfigError, match="schema fields of type 'regex'"):
        _load({"extraction_strategy": {"type": strategy, "params": {"schema": encoded_schema}}})


@pytest.mark.parametrize("strategy", SCHEMA_STRATEGIES)
@pytest.mark.parametrize("enveloped", [False, True], ids=["plain", "enveloped"])
@pytest.mark.parametrize("field_type", ["text", ["text"]], ids=["scalar", "pipeline"])
def test_untrusted_schema_without_regex_still_allowed(strategy, enveloped, field_type):
    schema = _schema({"name": "t", "selector": "h1", "type": field_type})
    encoded_schema = to_serializable_dict(schema) if enveloped else schema

    cfg = _load({"extraction_strategy": {"type": strategy, "params": {"schema": encoded_schema}}})

    assert type(cfg.extraction_strategy).__name__ == strategy
    assert cfg.extraction_strategy.schema == schema


@pytest.mark.parametrize("strategy", SCHEMA_STRATEGIES)
@pytest.mark.parametrize("enveloped", [False, True], ids=["plain", "enveloped"])
@pytest.mark.parametrize("field", [REGEX_FIELD, REGEX_PIPELINE_FIELD], ids=["scalar", "pipeline"])
def test_trusted_schema_regex_unchanged(strategy, enveloped, field):
    schema = _schema(field)
    encoded_schema = to_serializable_dict(schema) if enveloped else schema

    cfg = _load(
        {"extraction_strategy": {"type": strategy, "params": {"schema": encoded_schema}}},
        Provenance.TRUSTED,
    )

    assert cfg.extraction_strategy.schema == schema


def test_trusted_regex_unchanged():
    cfg = _load({"chunking_strategy": {
        "type": "RegexChunking", "params": {"patterns": [r"\n"]},
    }}, Provenance.TRUSTED)

    assert cfg.chunking_strategy.patterns == [r"\n"]
