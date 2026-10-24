"""Client screenshot delays cannot consume the crawl deadline by sleeping."""

import pytest
from pydantic import ValidationError

from schemas import ScreenshotRequest


@pytest.mark.parametrize("delay", [-1, 5.01, 1e9, float("inf"), float("nan")])
def test_screenshot_rejects_delays_outside_client_budget(delay):
    with pytest.raises(ValidationError):
        ScreenshotRequest(url="https://example.com/", screenshot_wait_for=delay)


@pytest.mark.parametrize("delay", [None, 0, 2, 5])
def test_screenshot_accepts_supported_client_delays(delay):
    request = ScreenshotRequest(url="https://example.com/", screenshot_wait_for=delay)
    assert request.screenshot_wait_for == delay
