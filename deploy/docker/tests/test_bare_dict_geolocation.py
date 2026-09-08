"""
Regression tests for the bare-dict nested-config trust gate.

The untrusted-config gate in `crawl4ai/async_configs.py::_clamp_untrusted`
previously accepted a bare-dict value for five allowlisted nested-object
fields of `CrawlerRunConfig` (`geolocation`, `table_extraction`,
`scraping_strategy`, `extraction_strategy`, and `markdown_generator`).
`CrawlerRunConfig.__init__` stored the bare dict
verbatim (no dict->object conversion, no isinstance guard), so the
downstream crawl pipeline crashed with `AttributeError` on typed-config
attributes (`geo.latitude`, `table_extraction.logger`,
`scraping_strategy.logger`), surfacing as HTTP 502 instead of the
validation-time HTTP 400 the trust gate is supposed to produce.

These tests lock in the fixed behavior:
    - a bare dict for any of the five fields is rejected at the gate with
      UntrustedConfigError -> HTTP 400 (regardless of the dict contents),
    - the typed `{"type": "<ClassName>", "params": {...}}` envelope still
      constructs correctly, and the typed-form clamp diagnostic is
      preserved (no regression),
    - the TRUSTED (SDK / in-process) path is unchanged.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" \
        .venv/bin/python -m pytest deploy/docker/tests/test_bare_dict_geolocation.py -v
"""

import pytest

from crawl4ai.async_configs import (
    CrawlerRunConfig,
    DefaultTableExtraction,
    GeolocationConfig,
    LXMLWebScrapingStrategy,
    Provenance,
    UntrustedConfigError,
)

U = Provenance.UNTRUSTED
T = Provenance.TRUSTED


@pytest.fixture
def auth(server_module):
    from auth import create_access_token

    return {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'})}"}


# ─────────────────────── HTTP trust-boundary reproduction ───────────────────────


class TestBareDictGeolocationGate:
    """The /crawl endpoint must reject bare-dict geolocation with HTTP 400,
    not accept it and crash with 502."""

    def test_typed_out_of_range_rejected_400(self, stock_client, auth):
        """Typed wrapper: out-of-range -> 400 (gate's informative clamp)."""
        r = stock_client.post(
            "/crawl",
            json={
                "urls": ["https://example.com"],
                "crawler_config": {
                    "geolocation": {
                        "type": "GeolocationConfig",
                        "params": {"latitude": 999, "longitude": 9999, "accuracy": -50},
                    }
                },
            },
            headers=auth,
        )
        assert r.status_code == 400
        assert "latitude" in r.text and "90" in r.text

    def test_bare_dict_out_of_range_now_rejected_400(self, stock_client, auth):
        """Bare-dict, out-of-range: previously 502 (crash); now 400 at the gate."""
        r = stock_client.post(
            "/crawl",
            json={
                "urls": ["https://example.com"],
                "crawler_config": {
                    "geolocation": {"latitude": 999, "longitude": 9999, "accuracy": -50}
                },
            },
            headers=auth,
        )
        assert r.status_code == 400
        assert "geolocation" in r.text

    def test_bare_dict_in_range_now_rejected_400(self, stock_client, auth):
        """Bare-dict, in-range: previously 502 (crash on valid values); now 400.

        The fix rejects the *shape* (dict where a GeolocationConfig is
        expected), independent of the values — even valid coordinates crashed
        `.latitude` before the gate rejection landed.
        """
        r = stock_client.post(
            "/crawl",
            json={
                "urls": ["https://example.com"],
                "crawler_config": {
                    "geolocation": {
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "accuracy": 10.0,
                    }
                },
            },
            headers=auth,
        )
        assert r.status_code == 400
        assert "geolocation" in r.text


# ───────────────────── Library-level gate (no HTTP/browser needed) ─────────────


class TestBareDictGateAtLibraryLevel:
    """Exercises `_clamp_untrusted`'s `CrawlerRunConfig` branch directly. This
    covers `table_extraction` and `scraping_strategy`, whose crash sites sit
    behind page navigation and so cannot be http-reproduced without a real
    browser."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("geolocation", {"latitude": 999, "longitude": 9999, "accuracy": -50}),
            ("geolocation", {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 10.0}),
            ("geolocation", {"latitude": 0, "longitude": 0}),
            ("table_extraction", {"table_score_threshold": 5}),
            ("table_extraction", {"min_rows": 2, "min_cols": 3}),
            ("table_extraction", {}),
            ("scraping_strategy", {"foo": "bar"}),
            ("scraping_strategy", {"logger": "x"}),
            ("scraping_strategy", {}),
            ("extraction_strategy", {}),
            ("markdown_generator", {}),
        ],
    )
    def test_bare_dict_rejected_at_gate(self, field, value):
        with pytest.raises(UntrustedConfigError) as exc:
            CrawlerRunConfig.load({field: value}, provenance=U)
        assert field in str(exc.value)
        assert "bare dict" in str(exc.value)

    def test_bare_dict_does_not_reach_constructor(self):
        """The gate must raise BEFORE `__init__` stores the bare dict. If the
        gate were bypassed, `__init__` would store a dict and the downstream
        `.latitude` / `.logger` access would raise `AttributeError`."""
        for field in (
            "geolocation",
            "table_extraction",
            "scraping_strategy",
            "extraction_strategy",
            "markdown_generator",
        ):
            with pytest.raises(UntrustedConfigError):
                CrawlerRunConfig.load({field: {"_sentinel": True}}, provenance=U)


# ─────────────────────── Typed-form regression (no regression) ──────────────────


class TestTypedFormStillAccepted:
    """The typed `{"type": "<ClassName>", "params": {...}}` envelope (the only
    supported wire shape for these fields) must still construct correctly.
    These payloads reach `from_serializable_dict`'s typed-object path BEFORE
    the `CrawlerRunConfig` clamp branch runs, so they must not match the
    bare-dict re-gate."""

    def test_typed_geolocation_constructs_object(self):
        c = CrawlerRunConfig.load(
            {
                "geolocation": {
                    "type": "GeolocationConfig",
                    "params": {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 10.0},
                }
            },
            provenance=U,
        )
        assert isinstance(c.geolocation, GeolocationConfig)
        assert c.geolocation.latitude == 37.7749
        assert c.geolocation.longitude == -122.4194
        assert c.geolocation.accuracy == 10.0

    def test_typed_table_extraction_constructs_object(self):
        c = CrawlerRunConfig.load(
            {
                "table_extraction": {
                    "type": "DefaultTableExtraction",
                    "params": {"table_score_threshold": 5},
                }
            },
            provenance=U,
        )
        assert isinstance(c.table_extraction, DefaultTableExtraction)
        assert c.table_extraction.table_score_threshold == 5

    def test_typed_scraping_strategy_constructs_object(self):
        c = CrawlerRunConfig.load(
            {"scraping_strategy": {"type": "LXMLWebScrapingStrategy", "params": {}}},
            provenance=U,
        )
        assert isinstance(c.scraping_strategy, LXMLWebScrapingStrategy)

    def test_typed_out_of_range_geolocation_still_informative_400(self):
        """The clamp diagnostic for the typed form is preserved: the bare-dict
        shape check must not short-circuit the typed-object clamp path."""
        with pytest.raises(UntrustedConfigError) as exc:
            CrawlerRunConfig.load(
                {
                    "geolocation": {
                        "type": "GeolocationConfig",
                        "params": {"latitude": 999, "longitude": 9999, "accuracy": -50},
                    }
                },
                provenance=U,
            )
        msg = str(exc.value)
        assert "latitude" in msg and "GeolocationConfig" in msg
        assert "bare dict" not in msg


# ───────────────────── TRUSTED path unchanged (no regression) ───────────────────


class TestTrustedPathUnchanged:
    """SDK / in-process callers (default TRUSTED) keep the existing behavior.
    The gate only fires under `Provenance.UNTRUSTED`."""

    def test_trusted_constructor_accepts_bare_dict_geolocation(self):
        # SDK callers historically passed bare dicts; behavior is unchanged.
        c = CrawlerRunConfig(geolocation={"latitude": 37.7749, "longitude": -122.4194})
        assert c.geolocation == {"latitude": 37.7749, "longitude": -122.4194}

    def test_trusted_constructor_accepts_typed_geolocation(self):
        c = CrawlerRunConfig(geolocation=GeolocationConfig(37.7749, -122.4194, 10.0))
        assert isinstance(c.geolocation, GeolocationConfig)
        assert c.geolocation.latitude == 37.7749


@pytest.mark.parametrize("field", ["extraction_strategy", "markdown_generator", "geolocation", "table_extraction", "scraping_strategy", "virtual_scroll_config"])
@pytest.mark.parametrize("value", [[], [1], "bad", True, False, 0, 1.5])
def test_untrusted_object_fields_reject_other_invalid_shapes(field, value):
    with pytest.raises(UntrustedConfigError, match=field):
        CrawlerRunConfig.load(
            {"type": "CrawlerRunConfig", "params": {field: value}}, provenance=U
        )


@pytest.mark.parametrize("field", ["geolocation", "virtual_scroll_config"])
def test_nested_config_list_is_rejected_by_http_gate(stock_client, auth, field):
    response = stock_client.post("/crawl", headers=auth, json={
        "urls": ["https://example.com"],
        "crawler_config": {"type": "CrawlerRunConfig", "params": {field: []}},
    })
    assert response.status_code == 400
    assert field in response.json()["detail"]


def test_wrong_typed_object_is_rejected_for_geolocation():
    with pytest.raises(UntrustedConfigError, match="geolocation"):
        CrawlerRunConfig.load({"type": "CrawlerRunConfig", "params": {
            "geolocation": {"type": "DefaultMarkdownGenerator", "params": {}}
        }}, provenance=U)


def test_no_table_extraction_is_a_valid_untrusted_strategy():
    from crawl4ai.table_extraction import NoTableExtraction
    config = CrawlerRunConfig.load({"type": "CrawlerRunConfig", "params": {
        "table_extraction": {"type": "NoTableExtraction", "params": {}}
    }}, provenance=U)
    assert isinstance(config.table_extraction, NoTableExtraction)


@pytest.mark.parametrize("field,type_name", [("geolocation", "GeolocationConfig"), ("virtual_scroll_config", "VirtualScrollConfig")])
def test_incomplete_nested_constructor_is_rejected_by_http_gate(stock_client, auth, field, type_name):
    response = stock_client.post("/crawl", headers=auth, json={
        "urls": ["https://example.com"],
        "crawler_config": {"type": "CrawlerRunConfig", "params": {
            field: {"type": type_name, "params": {}}
        }},
    })
    assert response.status_code == 400
    assert type_name in response.json()["detail"]
