"""
Regression tests for proxy authentication in managed/CDP browser modes.

Bug: BrowserConfig.proxy_config (server + username + password) was silently
dropped in managed/CDP mode. Only the server URL was forwarded to Chromium via
--proxy-server; credentials never reached any Playwright context, producing
407 Proxy Authentication Required for authenticated proxies.

The default browser_mode="dedicated" (direct Playwright launch) and the
use_persistent_context=True path were unaffected; only the managed/CDP path
(connect_over_cdp / use_managed_browser) dropped BrowserConfig.proxy_config.

These tests verify both fix paths:
  1. create_browser_context() builds ProxySettings from BrowserConfig.proxy_config
     (replacing the dead `{"server": self.config.proxy}` branch).
  2. start() creates a new context (not reuses contexts[0]) when proxy_config is
     set, so credentials can be applied.
  3. crawlerRunConfig.proxy_config still overrides BrowserConfig.proxy_config.
  4. close() closes the context created by start() so it does not leak on
     shared/cached CDP connections.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, ProxyConfig
from crawl4ai.browser_manager import BrowserManager


# ── Helpers ─────────────────────────────────────────────────────────────


def _proxy_config():
    return ProxyConfig(
        server="http://proxy.example.com:8080",
        username="user",
        password="pass",
    )


def _other_proxy_config():
    return ProxyConfig(
        server="http://other-proxy.example.com:9090",
        username="other",
        password="secret",
    )


def _make_manager(browser_config=None, **kwargs):
    """Build a BrowserManager with a mocked browser attached."""
    config = browser_config or BrowserConfig(**kwargs)
    mgr = BrowserManager(config)
    # Attach a mocked browser so create_browser_context() can run.
    mock_context = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mgr.browser = mock_browser
    return mgr, mock_browser, mock_context


# ── create_browser_context: BrowserConfig.proxy_config is honored ──────


class TestCreateBrowserContextBrowserProxyConfig:
    """Verify create_browser_context applies BrowserConfig.proxy_config credentials."""

    @pytest.mark.asyncio
    async def test_browser_proxy_config_produces_proxy_settings_with_credentials(self):
        mgr, mock_browser, _ = _make_manager(
            proxy_config=_proxy_config(), headless=True
        )
        await mgr.create_browser_context()

        mock_browser.new_context.assert_called_once()
        _, kwargs = mock_browser.new_context.call_args
        proxy = kwargs.get("proxy")
        assert proxy is not None, "proxy must be set when BrowserConfig.proxy_config is provided"
        assert proxy["server"] == "http://proxy.example.com:8080"
        assert proxy["username"] == "user"
        assert proxy["password"] == "pass"

    @pytest.mark.asyncio
    async def test_no_proxy_config_yields_none_proxy(self):
        mgr, mock_browser, _ = _make_manager(headless=True)
        await mgr.create_browser_context()

        mock_browser.new_context.assert_called_once()
        _, kwargs = mock_browser.new_context.call_args
        assert kwargs.get("proxy") is None

    @pytest.mark.asyncio
    async def test_deprecated_proxy_field_does_not_resurrect_dead_branch(self):
        """BrowserConfig converts `proxy` into `proxy_config` and nullifies `proxy`,
        so the old `{"server": self.config.proxy}` branch stays dead."""
        with pytest.warns(UserWarning):
            cfg = BrowserConfig(
                proxy="1.2.3.4:8080:user:pass", headless=True
            )
        assert cfg.proxy is None
        assert cfg.proxy_config is not None
        mgr, mock_browser, _ = _make_manager(browser_config=cfg)
        await mgr.create_browser_context()

        _, kwargs = mock_browser.new_context.call_args
        proxy = kwargs.get("proxy")
        assert proxy is not None
        assert proxy["server"] == "http://1.2.3.4:8080"
        assert proxy["username"] == "user"
        assert proxy["password"] == "pass"


# ── create_browser_context: crawlerRunConfig.proxy_config overrides ─────


class TestCrawlerRunConfigOverridesBrowserProxyConfig:
    """Verify per-crawl proxy_config takes precedence over browser-level config."""

    @pytest.mark.asyncio
    async def test_crawler_run_config_overrides_browser_proxy_config(self):
        mgr, mock_browser, _ = _make_manager(
            proxy_config=_proxy_config(), headless=True
        )
        run_cfg = CrawlerRunConfig(proxy_config=_other_proxy_config())
        await mgr.create_browser_context(run_cfg)

        _, kwargs = mock_browser.new_context.call_args
        proxy = kwargs.get("proxy")
        assert proxy is not None
        assert proxy["server"] == "http://other-proxy.example.com:9090"
        assert proxy["username"] == "other"
        assert proxy["password"] == "secret"

    @pytest.mark.asyncio
    async def test_crawler_run_config_provides_proxy_when_browser_has_none(self):
        mgr, mock_browser, _ = _make_manager(headless=True)
        run_cfg = CrawlerRunConfig(proxy_config=_proxy_config())
        await mgr.create_browser_context(run_cfg)

        _, kwargs = mock_browser.new_context.call_args
        proxy = kwargs.get("proxy")
        assert proxy is not None
        assert proxy["server"] == "http://proxy.example.com:8080"
        assert proxy["username"] == "user"
        assert proxy["password"] == "pass"


# ── start(): context selection in managed/CDP mode ─────────────────────


class TestStartContextSelection:
    """Verify start() creates a new context when proxy_config is set in
    managed/CDP mode, and reuses contexts[0] otherwise."""

    def _build_start_mocks(self, existing_contexts):
        """Wire up mocks for playwright + CDP connection used by start()."""
        new_context = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.contexts = list(existing_contexts)
        mock_browser.new_context = AsyncMock(return_value=new_context)
        mock_browser.close = AsyncMock()

        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_playwright.stop = AsyncMock()

        async_playwright_mock = MagicMock()
        async_playwright_mock.start = AsyncMock(return_value=mock_playwright)
        return mock_browser, new_context, mock_playwright, async_playwright_mock, mock_playwright

    @pytest.mark.asyncio
    async def test_creates_new_context_when_proxy_config_set(self):
        cfg = BrowserConfig(
            proxy_config=_proxy_config(),
            cdp_url="http://localhost:9222",
            headless=True,
        )
        mgr = BrowserManager(cfg)
        existing = AsyncMock()
        existing.close = AsyncMock()
        mock_browser, new_ctx, mock_pw, ap_mock, _ = self._build_start_mocks([existing])

        with patch("playwright.async_api.async_playwright", return_value=ap_mock), \
             patch.object(BrowserManager, "_verify_cdp_ready", new=AsyncMock(return_value=True)):
            await mgr.start()

        # default_context is the NEW context, not the pre-existing contexts[0]
        assert mgr.default_context is new_ctx
        assert mgr.default_context is not existing
        assert mgr._default_context_owned is True
        # The new context received proxy credentials from BrowserConfig.proxy_config
        mock_browser.new_context.assert_called_once()
        _, kwargs = mock_browser.new_context.call_args
        proxy = kwargs.get("proxy")
        assert proxy is not None
        assert proxy["server"] == "http://proxy.example.com:8080"
        assert proxy["username"] == "user"
        assert proxy["password"] == "pass"

    @pytest.mark.asyncio
    async def test_reuses_existing_context_when_no_proxy_config(self):
        cfg = BrowserConfig(
            cdp_url="http://localhost:9222",
            headless=True,
        )
        mgr = BrowserManager(cfg)
        existing = AsyncMock()
        existing.close = AsyncMock()
        mock_browser, new_ctx, mock_pw, ap_mock, _ = self._build_start_mocks([existing])

        with patch("playwright.async_api.async_playwright", return_value=ap_mock), \
             patch.object(BrowserManager, "_verify_cdp_ready", new=AsyncMock(return_value=True)):
            await mgr.start()

        assert mgr.default_context is existing
        assert mgr._default_context_owned is False
        mock_browser.new_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_context_when_no_existing_contexts(self):
        cfg = BrowserConfig(
            proxy_config=_proxy_config(),
            cdp_url="http://localhost:9222",
            headless=True,
        )
        mgr = BrowserManager(cfg)
        mock_browser, new_ctx, mock_pw, ap_mock, _ = self._build_start_mocks([])

        with patch("playwright.async_api.async_playwright", return_value=ap_mock), \
             patch.object(BrowserManager, "_verify_cdp_ready", new=AsyncMock(return_value=True)):
            await mgr.start()

        assert mgr.default_context is new_ctx
        assert mgr._default_context_owned is True

    @pytest.mark.asyncio
    async def test_browser_context_id_reuses_existing_without_creating(self):
        """browser_context_id path reuses the pre-created context even with proxy_config;
        proxy credentials are the caller's responsibility for a pre-created context."""
        cfg = BrowserConfig(
            proxy_config=_proxy_config(),
            cdp_url="http://localhost:9222",
            browser_context_id="some-id",
            headless=True,
        )
        mgr = BrowserManager(cfg)
        existing = AsyncMock()
        existing.close = AsyncMock()
        mock_browser, new_ctx, mock_pw, ap_mock, _ = self._build_start_mocks([existing])

        with patch("playwright.async_api.async_playwright", return_value=ap_mock), \
             patch.object(BrowserManager, "_verify_cdp_ready", new=AsyncMock(return_value=True)):
            await mgr.start()

        assert mgr.default_context is existing
        assert mgr._default_context_owned is False
        mock_browser.new_context.assert_not_called()


# ── close(): owned default context is cleaned up ───────────────────────


class TestCloseOwnedDefaultContext:
    """Verify close() closes a default context created by start() so it does
    not leak on shared/cached CDP connections."""

    @pytest.mark.asyncio
    async def test_close_closes_owned_default_context(self):
        cfg = BrowserConfig(
            proxy_config=_proxy_config(),
            cdp_url="http://localhost:9222",
            headless=True,
        )
        mgr = BrowserManager(cfg)
        existing = AsyncMock()
        existing.close = AsyncMock()
        mock_browser, new_ctx, mock_pw, ap_mock, _ = (
            self._build_mocks_for_close([existing])
        )

        with patch("playwright.async_api.async_playwright", return_value=ap_mock), \
             patch.object(BrowserManager, "_verify_cdp_ready", new=AsyncMock(return_value=True)):
            await mgr.start()

        assert mgr._default_context_owned is True
        new_ctx_close = new_ctx.close = AsyncMock()

        # Explicit cdp_url with cdp_cleanup_on_close defaults to False → no close.
        # Force cleanup to exercise the owned-context close path:
        mgr.config.cdp_cleanup_on_close = True
        await mgr.close()

        new_ctx_close.assert_awaited()
        assert mgr._default_context_owned is False
        assert mgr.default_context is None

    @pytest.mark.asyncio
    async def test_close_does_not_close_reused_default_context(self):
        cfg = BrowserConfig(
            cdp_url="http://localhost:9222",
            headless=True,
        )
        mgr = BrowserManager(cfg)
        existing = AsyncMock()
        existing.close = AsyncMock()
        mock_browser, new_ctx, mock_pw, ap_mock, _ = (
            self._build_mocks_for_close([existing])
        )

        with patch("playwright.async_api.async_playwright", return_value=ap_mock), \
             patch.object(BrowserManager, "_verify_cdp_ready", new=AsyncMock(return_value=True)):
            await mgr.start()

        assert mgr._default_context_owned is False
        mgr.config.cdp_cleanup_on_close = True
        await mgr.close()

        # contexts[0] is Chromium's default context; must not be closed by us.
        existing.close.assert_not_awaited()

    @staticmethod
    def _build_mocks_for_close(existing_contexts):
        new_context = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.contexts = list(existing_contexts)
        mock_browser.new_context = AsyncMock(return_value=new_context)
        mock_browser.close = AsyncMock()
        mock_playwright = MagicMock()
        mock_playwright.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_playwright.stop = AsyncMock()
        ap_mock = MagicMock()
        ap_mock.start = AsyncMock(return_value=mock_playwright)
        return mock_browser, new_context, mock_playwright, ap_mock, None


# ── Non-regression: dedicated mode unaffected ──────────────────────────


class TestDedicatedModeUnaffected:
    """The default browser_mode='dedicated' (direct Playwright launch) already
    applied BrowserConfig.proxy_config via _build_browser_args; it must keep
    working and remain unaffected by the managed-path fix."""

    def test_dedicated_launch_args_include_proxy_credentials(self):
        cfg = BrowserConfig(
            proxy_config=_proxy_config(),
            headless=True,
        )
        # Root conftest may inject --no-sandbox; not relevant here.
        mgr = BrowserManager(cfg)
        args = mgr._build_browser_args()
        proxy = args.get("proxy")
        assert proxy is not None
        assert proxy["server"] == "http://proxy.example.com:8080"
        assert proxy["username"] == "user"
        assert proxy["password"] == "pass"
