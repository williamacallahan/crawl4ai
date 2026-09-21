"""Regression tests for the ``_UNWANTED_PROPS`` deprecation guard shared by
``CrawlerRunConfig``, ``LLMExtractionStrategy`` and ``LLMContentFilter``.

Background (commit ``2af958e1``, PR #724): each class overrides ``__setattr__``
to reject deprecated constructor properties, raising ``AttributeError`` only
when the supplied value differs from the parameter's bound default. The guard
used identity (``value is not all_params[name].default``) instead of equality
(``value != all_params[name].default``), producing false positives whenever the
caller passed a value that **equals** the default but is a **different object**.

Primary trigger — ``DEFAULT_PROVIDER = "openai/gpt-4o"`` (a non-interned
``str``): a caller-side literal ``LLMExtractionStrategy(provider="openai/gpt-4o")``
raised even though the value equals the default. Secondary trigger — boolean
cache defaults: ``CrawlerRunConfig(disable_cache=0)`` raised because ``0 == False``
but ``0 is not False``.

The fix replaces ``is not`` with ``!=`` in all three guards. These tests pin
both the no-raise (equals-default) and raise (genuine-change) behavior so the
identity comparison cannot silently return.
"""

import json

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.config import DEFAULT_PROVIDER
from crawl4ai.content_filter_strategy import LLMContentFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# A caller-side literal equal to the module constant but a distinct object
# (not interned across compilation units because it contains '/').
DEFAULT_PROVIDER_LITERAL = "openai/gpt-4o"

CACHE_FLAGS = ["disable_cache", "bypass_cache", "no_cache_read", "no_cache_write"]

LLM_CLASSES = [LLMExtractionStrategy, LLMContentFilter]


@pytest.fixture(autouse=True)
def _reset_crawler_run_config_defaults():
    """Keep CrawlerRunConfig class-level defaults clean across tests."""
    CrawlerRunConfig.reset_defaults()
    yield
    CrawlerRunConfig.reset_defaults()


def test_default_provider_literal_is_distinct_but_equal():
    """The bug's premise: a caller literal equals the module constant but is
    a different object, so identity comparison is the wrong tool."""
    assert DEFAULT_PROVIDER_LITERAL is not DEFAULT_PROVIDER
    assert DEFAULT_PROVIDER_LITERAL == DEFAULT_PROVIDER


@pytest.mark.parametrize("cls", LLM_CLASSES, ids=lambda c: c.__name__)
def test_provider_equal_to_default_does_not_raise(cls):
    # Literal and module constant both equal the default; neither should raise.
    cls(provider=DEFAULT_PROVIDER)
    cls(provider=DEFAULT_PROVIDER_LITERAL)


@pytest.mark.parametrize("cls", LLM_CLASSES, ids=lambda c: c.__name__)
def test_provider_changed_from_default_raises(cls):
    with pytest.raises(AttributeError, match=r"Setting 'provider' is deprecated"):
        cls(provider="anthropic/claude")


@pytest.mark.parametrize("cls", LLM_CLASSES, ids=lambda c: c.__name__)
def test_no_arg_construction_sets_default_provider(cls):
    instance = cls()
    assert instance.provider == DEFAULT_PROVIDER


@pytest.mark.parametrize("cls", LLM_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kwarg", ["api_token", "base_url", "api_base"])
def test_none_default_deprecated_params_do_not_raise(cls, kwarg):
    # None is the bound default; passing it explicitly must look like omitting it.
    cls(**{kwarg: None})


@pytest.mark.parametrize("cls", LLM_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kwarg", ["api_token", "base_url", "api_base"])
def test_none_default_deprecated_params_changed_raise(cls, kwarg):
    # ``!= None`` agrees with ``is not None`` for str/None values, so the fix
    # preserves the genuine-change raise for these params.
    with pytest.raises(AttributeError, match=rf"Setting '(?:{kwarg}|base_url)' is deprecated"):
        cls(**{kwarg: "non-default-value"})


@pytest.mark.parametrize("flag", CACHE_FLAGS)
@pytest.mark.parametrize("value", [False, 0], ids=["False", "int-0"])
def test_cache_flag_equal_to_default_does_not_raise(flag, value):
    # ``0 == False`` (caller did not change the option) but ``0 is not False``;
    # the guard must use equality.
    cfg = CrawlerRunConfig(**{flag: value})
    assert getattr(cfg, flag) == value


@pytest.mark.parametrize("flag", CACHE_FLAGS)
def test_cache_flag_changed_from_default_raises(flag):
    with pytest.raises(AttributeError, match=rf"Setting '{flag}' is deprecated"):
        CrawlerRunConfig(**{flag: True})


@pytest.mark.parametrize("flag", CACHE_FLAGS)
def test_post_construction_change_raises(flag):
    # The guard is on ``__setattr__``; post-construction assignment must reject
    # genuine changes too.
    cfg = CrawlerRunConfig()
    with pytest.raises(AttributeError, match=rf"Setting '{flag}' is deprecated"):
        setattr(cfg, flag, True)


def test_disable_cache_zero_from_kwargs_does_not_raise():
    # Realistic config-file trigger: JSON stores the bool as integer 0.
    cfg = CrawlerRunConfig.from_kwargs(json.loads('{"disable_cache": 0}'))
    assert cfg.disable_cache == 0


def test_disable_cache_true_from_kwargs_raises():
    with pytest.raises(AttributeError, match=r"Setting 'disable_cache' is deprecated"):
        CrawlerRunConfig.from_kwargs(json.loads('{"disable_cache": true}'))


def test_provider_default_string_from_config_does_not_raise():
    parsed = json.loads('{"provider": "openai/gpt-4o"}')
    assert parsed["provider"] is not DEFAULT_PROVIDER  # distinct object
    strategy = LLMExtractionStrategy(**parsed)
    assert strategy.provider == DEFAULT_PROVIDER


def test_provider_changed_string_from_config_raises():
    parsed = json.loads('{"provider": "anthropic/claude"}')
    with pytest.raises(AttributeError, match=r"Setting 'provider' is deprecated"):
        LLMExtractionStrategy(**parsed)
