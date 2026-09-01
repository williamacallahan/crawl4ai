"""Regression tests: CrawlerRunConfig.to_dict() must serialize virtual_scroll_config
and force_viewport_screenshot so that clone() preserves them.

Background: ``CrawlerRunConfig.clone()`` is built on ``to_dict()`` -> ``from_kwargs()``.
``dump()``/``load()`` uses the introspection-based ``to_serializable_dict()`` which
already preserves these fields. The manual ``to_dict()`` previously omitted them,
so ``clone()`` silently dropped allowlisted configuration (e.g. deep-crawled pages
stopped using virtual scrolling).
"""

import pytest

from crawl4ai.async_configs import (
    CrawlerRunConfig,
    VirtualScrollConfig,
    UNTRUSTED_FIELD_ALLOWLIST,
)


@pytest.fixture(autouse=True)
def _reset_crawler_run_config_defaults():
    """Ensure class-level set_defaults() state from other tests does not leak in."""
    CrawlerRunConfig.reset_defaults()
    yield
    CrawlerRunConfig.reset_defaults()


class TestForceViewportScreenshotSerialization:
    """force_viewport_screenshot must round-trip through to_dict()/clone()."""

    def test_to_dict_includes_field(self):
        config = CrawlerRunConfig(force_viewport_screenshot=True)
        d = config.to_dict()
        assert "force_viewport_screenshot" in d
        assert d["force_viewport_screenshot"] is True

    def test_clone_preserves(self):
        config = CrawlerRunConfig(force_viewport_screenshot=True)
        cloned = config.clone()
        assert cloned.force_viewport_screenshot is True


class TestVirtualScrollConfigSerialization:
    """virtual_scroll_config must round-trip through to_dict()/clone()."""

    def test_to_dict_includes_field(self):
        vsc = VirtualScrollConfig(
            container_selector="#feed", scroll_count=5, scroll_by=200, wait_after_scroll=1.5
        )
        config = CrawlerRunConfig(virtual_scroll_config=vsc)
        d = config.to_dict()
        assert "virtual_scroll_config" in d
        assert d["virtual_scroll_config"] == vsc.to_dict()

    def test_clone_preserves(self):
        vsc = VirtualScrollConfig(container_selector="#feed", scroll_count=7)
        config = CrawlerRunConfig(virtual_scroll_config=vsc)
        cloned = config.clone()
        assert isinstance(cloned.virtual_scroll_config, VirtualScrollConfig)
        assert cloned.virtual_scroll_config.container_selector == "#feed"
        assert cloned.virtual_scroll_config.scroll_count == 7
        # Original untouched
        assert config.virtual_scroll_config is vsc


class TestCloneConsistencyWithDumpLoad:
    """clone() (via to_dict) must agree with the deep-crawl clone contract."""

    def test_clone_with_deep_crawl_style_overrides_keeps_both_fields(self):
        """The BFS deep-crawl clone pattern must not drop these allowlisted fields.

        ``bfs_strategy.py`` clones with ``config.clone(deep_crawl_strategy=None, stream=False)``;
        virtual scroll and viewport screenshot settings must survive this clone so that
        deep-crawl-discovered pages keep the parent's capture configuration.
        """
        vsc = VirtualScrollConfig(container_selector="main", scroll_count=12)
        original = CrawlerRunConfig(
            virtual_scroll_config=vsc,
            force_viewport_screenshot=True,
        )

        cloned = original.clone(deep_crawl_strategy=None, stream=False)

        assert cloned.stream is False
        assert cloned.deep_crawl_strategy is None
        # The bug-carrying fields must be preserved:
        assert cloned.force_viewport_screenshot is True
        assert cloned.virtual_scroll_config is not None
        assert cloned.virtual_scroll_config.container_selector == "main"
        assert cloned.virtual_scroll_config.scroll_count == 12


class TestSecurityAllowlistConsistency:
    """Fields on the UNTRUSTED allowlist must be serializable by to_dict().

    This guards the whole class of bug: any allowlisted field that a future change
    forgets to add to ``to_dict()`` would be silently dropped by ``clone()``.
    """

    def test_both_fields_are_allowlisted(self):
        allowlist = UNTRUSTED_FIELD_ALLOWLIST.get("CrawlerRunConfig", set())
        assert "virtual_scroll_config" in allowlist
        assert "force_viewport_screenshot" in allowlist

    def test_allowlisted_fields_are_serialized(self):
        allowlist = UNTRUSTED_FIELD_ALLOWLIST.get("CrawlerRunConfig", set())
        d = CrawlerRunConfig(
            virtual_scroll_config=VirtualScrollConfig(container_selector="#x"),
            force_viewport_screenshot=True,
        ).to_dict()
        missing_allowlisted = sorted(f for f in allowlist if f not in d)
        assert missing_allowlisted == [], (
            f"Allowlisted fields missing from to_dict(): {missing_allowlisted}"
        )
