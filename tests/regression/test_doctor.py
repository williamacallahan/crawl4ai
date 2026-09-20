"""Tests for crawl4ai.install.run_doctor (the ``crawl4ai-doctor`` health check).

Pins the root-context ``--no-sandbox`` behavior. ``BrowserManager`` treats
``--no-sandbox`` as an operator policy the caller must supply via
``BrowserConfig.extra_args`` when the runtime cannot use Chromium's sandbox;
running as root (e.g. the Docker build before the ``USER appuser`` switch) is
the canonical case where the sandbox is unavailable. These mock-based tests
run in the default CI lane (``-m "not network and not browser"``); no real
browser launches and no network is touched.
"""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crawl4ai.install import run_doctor


@pytest.fixture
def captured_extra_args(monkeypatch):
    """Record the raw ``extra_args`` kwarg passed to ``BrowserConfig``.

    The root ``conftest.py`` auto-appends ``--no-sandbox`` to every
    ``BrowserConfig`` when the test process runs as root. To assert what
    ``run_doctor`` itself sets (independent of that global patch), the kwarg
    is captured before delegating to the real ``__init__``; the conftest's
    later mutation of the instance list cannot corrupt that copy.
    """
    from crawl4ai.async_configs import BrowserConfig

    real_init = BrowserConfig.__init__
    captured = []

    def spy(self, *args, **kwargs):
        raw = kwargs.get("extra_args")
        captured.append(list(raw) if raw is not None else None)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(BrowserConfig, "__init__", spy)
    return captured


def _mock_crawler(markdown="# content"):
    """A mock usable as ``async with AsyncWebCrawler(...) as c`` returning markdown."""
    instance = AsyncMock()
    instance.__aenter__.return_value = instance
    instance.__aexit__.return_value = None
    result = MagicMock()
    result.markdown = markdown
    instance.arun = AsyncMock(return_value=result)
    return instance


class TestNoSandboxGating:
    @pytest.mark.asyncio
    async def test_adds_no_sandbox_when_root(self, captured_extra_args, monkeypatch):
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        with patch("crawl4ai.async_webcrawler.AsyncWebCrawler", return_value=_mock_crawler()):
            await run_doctor()
        assert captured_extra_args == [["--no-sandbox"]]

    @pytest.mark.asyncio
    async def test_omits_no_sandbox_when_non_root(self, captured_extra_args, monkeypatch):
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        with patch("crawl4ai.async_webcrawler.AsyncWebCrawler", return_value=_mock_crawler()):
            await run_doctor()
        assert captured_extra_args == [[]]
        assert "--no-sandbox" not in captured_extra_args[0]

    @pytest.mark.asyncio
    async def test_empty_extra_args_when_geteuid_unavailable(self, captured_extra_args, monkeypatch):
        monkeypatch.delattr(os, "geteuid", raising=False)
        with patch("crawl4ai.async_webcrawler.AsyncWebCrawler", return_value=_mock_crawler()):
            await run_doctor()
        assert captured_extra_args == [[]]
        assert "--no-sandbox" not in captured_extra_args[0]
