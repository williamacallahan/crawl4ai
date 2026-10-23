"""Caller browser configs get the server's ad blocking but keep their images."""

import pytest
from crawl4ai import BrowserConfig

import api
from utils import load_config


@pytest.mark.parametrize("text_mode", [False, True])
def test_policy_forces_configured_avoid_ads_and_leaves_text_mode_to_the_caller(monkeypatch, text_mode):
    monkeypatch.delenv("CRAWL4AI_CHROMIUM_SANDBOX", raising=False)
    shipped = load_config()
    assert shipped["crawler"]["browser"]["kwargs"]["text_mode"] is True

    result = api._apply_server_browser_policy(
        BrowserConfig(avoid_ads=False, text_mode=text_mode), shipped
    )

    assert result.avoid_ads is True
    assert result.text_mode is text_mode
