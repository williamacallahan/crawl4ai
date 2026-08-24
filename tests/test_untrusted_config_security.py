import math

import pytest

from crawl4ai.async_configs import (
    UNTRUSTED_ALLOWED_TYPES,
    UNTRUSTED_FIELD_ALLOWLIST,
    BrowserConfig,
    CrawlerRunConfig,
    Provenance,
    UntrustedConfigError,
    from_serializable_dict,
)

UNTRUSTED = Provenance.UNTRUSTED


def _unique_strings(count, length, prefix="x"):
    values = []
    for index in range(count):
        stem = f"{prefix}{index:02d}"
        values.append(stem + "x" * (length - len(stem)))
    return values


def test_every_untrusted_constructor_has_an_explicit_field_policy():
    enum_types = {"CacheMode", "MatchMode", "DisplayMode"}

    assert UNTRUSTED_ALLOWED_TYPES - enum_types <= UNTRUSTED_FIELD_ALLOWLIST.keys()


@pytest.mark.parametrize(
    "type_name",
    [
        "LinkPreviewConfig",
        "PDFContentScrapingStrategy",
        "CosineStrategy",
        "RegexExtractionStrategy",
        "RegexChunking",
        "HTTPCrawlerConfig",
    ],
)
def test_untrusted_type_gate_rejects_egress_and_resource_constructors(type_name):
    with pytest.raises(UntrustedConfigError):
        from_serializable_dict(
            {"type": type_name, "params": {}},
            provenance=UNTRUSTED,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("check_robots_txt", True),
        ("fetch_ssl_certificate", True),
        ("check_cache_freshness", True),
        ("cache_validation_timeout", 1),
        ("link_preview_config", {}),
        ("method", "DELETE"),
        ("capture_network_requests", True),
        ("capture_console_messages", True),
    ],
)
def test_untrusted_crawler_rejects_independent_network_and_capture_fields(field, value):
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load({field: value}, provenance=UNTRUSTED)


@pytest.mark.parametrize(
    "field,value",
    [
        ("accept_downloads", True),
        ("downloads_path", "/tmp/untrusted"),
        ("ignore_https_errors", True),
        ("use_managed_browser", True),
        ("use_persistent_context", True),
    ],
)
def test_untrusted_browser_rejects_file_and_launch_policy_fields(field, value):
    with pytest.raises(UntrustedConfigError):
        BrowserConfig.load({field: value}, provenance=UNTRUSTED)


@pytest.mark.parametrize(
    "wait_for",
    [
        "js:() => true",
        "() => true",
        "function ready() { return true; }",
        "css:js:document.body",
    ],
)
def test_untrusted_wait_for_rejects_javascript(wait_for):
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load({"wait_for": wait_for}, provenance=UNTRUSTED)


@pytest.mark.parametrize("wait_for", [".ready", "css:#content", "[data-loaded='true']"])
def test_untrusted_wait_for_preserves_css_without_javascript_fallback(wait_for):
    config = CrawlerRunConfig.load({"wait_for": wait_for}, provenance=UNTRUSTED)

    assert config.wait_for.startswith("css:")
    assert config.wait_for.removeprefix("css:")


def test_untrusted_target_elements_accepts_exact_limits():
    target_elements = _unique_strings(64, 2048, prefix=".item-")

    config = CrawlerRunConfig.load(
        {"target_elements": target_elements}, provenance=UNTRUSTED
    )

    assert config.target_elements == target_elements


def test_untrusted_target_elements_accepts_near_limits():
    target_elements = _unique_strings(63, 2047, prefix=".item-")

    config = CrawlerRunConfig.load(
        {"target_elements": target_elements}, provenance=UNTRUSTED
    )

    assert config.target_elements == target_elements


@pytest.mark.parametrize(
    "target_elements",
    [
        ".item",
        [".item"] * 65,
        [123],
        [""],
        ["   "],
        ["." + "x" * 2048],
    ],
)
def test_untrusted_target_elements_rejects_invalid_shapes_and_limits(target_elements):
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load(
            {"target_elements": target_elements}, provenance=UNTRUSTED
        )


def test_trusted_target_elements_remains_unbounded():
    target_elements = ["." + "x" * 4095] * 65

    config = CrawlerRunConfig(target_elements=target_elements)

    assert config.target_elements == target_elements


def test_untrusted_excluded_tags_accepts_exact_limits():
    excluded_tags = _unique_strings(64, 64, prefix="tag-")

    config = CrawlerRunConfig.load(
        {"excluded_tags": excluded_tags}, provenance=UNTRUSTED
    )

    assert config.excluded_tags == excluded_tags


def test_untrusted_excluded_tags_accepts_near_limits():
    excluded_tags = _unique_strings(63, 63, prefix="tag-")

    config = CrawlerRunConfig.load(
        {"excluded_tags": excluded_tags}, provenance=UNTRUSTED
    )

    assert config.excluded_tags == excluded_tags


def test_untrusted_string_lists_collapse_duplicates_after_count_check():
    config = CrawlerRunConfig.load(
        {"excluded_tags": ["script", " script ", "style"]},
        provenance=UNTRUSTED,
    )

    assert config.excluded_tags == ["script", "style"]


@pytest.mark.parametrize(
    "excluded_tags",
    [
        "script",
        ["script"] * 65,
        [123],
        [""],
        ["   "],
        ["x" * 65],
    ],
)
def test_untrusted_excluded_tags_rejects_invalid_shapes_and_limits(excluded_tags):
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load({"excluded_tags": excluded_tags}, provenance=UNTRUSTED)


def test_trusted_excluded_tags_remains_unbounded():
    excluded_tags = ["x" * 65] * 65

    config = CrawlerRunConfig(excluded_tags=excluded_tags)

    assert config.excluded_tags == excluded_tags


@pytest.mark.parametrize(
    "field,maximum_length",
    [
        ("keep_attrs", 64),
        ("exclude_social_media_domains", 253),
        ("exclude_domains", 253),
        ("url_matcher", 2048),
    ],
)
def test_equivalent_untrusted_string_lists_are_bounded(field, maximum_length):
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load({field: ["x"] * 65}, provenance=UNTRUSTED)
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load(
            {field: ["x" * (maximum_length + 1)]}, provenance=UNTRUSTED
        )
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load({field: [1]}, provenance=UNTRUSTED)


def test_untrusted_url_matcher_scalar_is_bounded_too():
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load({"url_matcher": "x" * 2049}, provenance=UNTRUSTED)


def test_untrusted_pruning_whitelists_are_bounded():
    for field in ("preserve_classes", "preserve_tags"):
        with pytest.raises(UntrustedConfigError):
            from_serializable_dict(
                {
                    "type": "PruningContentFilter",
                    "params": {field: ["x"] * 65},
                },
                provenance=UNTRUSTED,
            )


def test_untrusted_delays_loops_and_retries_are_capped():
    config = CrawlerRunConfig.load(
        {
            "page_timeout": 10**9,
            "wait_for_timeout": 0,
            "delay_before_return_html": 10**6,
            "mean_delay": 10**6,
            "max_range": 10**6,
            "scroll_delay": 10**6,
            "screenshot_wait_for": 10**6,
            "max_scroll_steps": 10**9,
            "max_retries": 10**9,
            "screenshot_height_threshold": 10**9,
            "word_count_threshold": 10**9,
            "image_description_min_word_threshold": 10**9,
            "image_score_threshold": 10**9,
            "table_score_threshold": 10**9,
        },
        provenance=UNTRUSTED,
    )

    assert config.page_timeout == 60_000
    assert config.wait_for_timeout == 60_000
    assert config.delay_before_return_html == 5
    assert config.mean_delay == 5
    assert config.max_range == 5
    assert config.scroll_delay == 2
    assert config.screenshot_wait_for == 5
    assert config.max_scroll_steps == 1000
    assert config.max_retries == 3
    assert config.screenshot_height_threshold == 10_000
    assert config.word_count_threshold == 100_000
    assert config.image_description_min_word_threshold == 100_000
    assert config.image_score_threshold == 100
    assert config.table_score_threshold == 100


@pytest.mark.parametrize(
    "field,value",
    [
        ("page_timeout", -1),
        ("wait_for_timeout", "60000"),
        ("delay_before_return_html", -0.1),
        ("scroll_delay", math.inf),
        ("screenshot_wait_for", True),
        ("max_scroll_steps", 1.5),
        ("max_retries", -1),
        ("screenshot_height_threshold", 0),
        ("word_count_threshold", -1),
        ("image_score_threshold", "3"),
        ("table_score_threshold", -1),
    ],
)
def test_untrusted_resource_quantities_reject_invalid_values(field, value):
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load({field: value}, provenance=UNTRUSTED)


@pytest.mark.parametrize(
    "viewport",
    [
        {"width": 100_000, "height": 100_000},
        {
            "type": "dict",
            "value": {"width": 100_000, "height": 100_000},
        },
    ],
)
def test_untrusted_composite_viewport_and_device_scale_are_capped(viewport):
    config = BrowserConfig.load(
        {"viewport": viewport, "device_scale_factor": 100},
        provenance=UNTRUSTED,
    )

    assert config.viewport_width * config.viewport_height <= 3840 * 2160
    assert 0.1 <= config.device_scale_factor <= 2
    assert (
        config.viewport_width * config.viewport_height * config.device_scale_factor**2
        <= 16_000_000
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"viewport_width": -1},
        {"viewport_height": "600"},
        {"viewport": []},
        {"device_scale_factor": 0},
        {"device_scale_factor": math.nan},
    ],
)
def test_untrusted_viewport_rejects_invalid_quantities(payload):
    with pytest.raises(UntrustedConfigError):
        BrowserConfig.load(payload, provenance=UNTRUSTED)


def test_plain_virtual_scroll_config_is_capped_before_constructor_conversion():
    config = CrawlerRunConfig.load(
        {
            "virtual_scroll_config": {
                "container_selector": "#feed",
                "scroll_count": 10**6,
                "scroll_by": 10**9,
                "wait_after_scroll": 10**6,
            }
        },
        provenance=UNTRUSTED,
    )

    virtual_scroll_config = config.virtual_scroll_config
    assert virtual_scroll_config is not None
    assert virtual_scroll_config.scroll_count == 100
    assert virtual_scroll_config.scroll_by == 100_000
    assert virtual_scroll_config.wait_after_scroll == 2


@pytest.mark.parametrize(
    "virtual_scroll",
    [
        {"container_selector": "", "scroll_count": 1},
        {"container_selector": "#feed", "scroll_count": -1},
        {"container_selector": "#feed", "wait_after_scroll": "1"},
        {"container_selector": "#feed", "scroll_by": "javascript"},
    ],
)
def test_virtual_scroll_rejects_invalid_selectors_and_quantities(virtual_scroll):
    with pytest.raises(UntrustedConfigError):
        CrawlerRunConfig.load(
            {"virtual_scroll_config": virtual_scroll},
            provenance=UNTRUSTED,
        )


def test_side_effect_free_json_extraction_strategy_remains_available():
    config = CrawlerRunConfig.load(
        {
            "type": "CrawlerRunConfig",
            "params": {
                "extraction_strategy": {
                    "type": "JsonCssExtractionStrategy",
                    "params": {
                        "schema": {
                            "name": "items",
                            "baseSelector": ".item",
                            "fields": [],
                        }
                    },
                }
            },
        },
        provenance=UNTRUSTED,
    )

    assert type(config.extraction_strategy).__name__ == "JsonCssExtractionStrategy"
