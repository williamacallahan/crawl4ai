"""A /crawl browser config that omits the viewport or user agent gets the server's.

The SDK otherwise supplied 1080x600 and a Chrome 116 user agent to every caller
that sent only {"headless": true}, while the browser itself reports Chrome 156.
"""

import pytest
from crawl4ai import BrowserConfig
from crawl4ai.async_configs import Provenance

import api
from utils import load_config


def _load(request):
    shipped = load_config()
    return BrowserConfig.load(
        api.with_server_browser_defaults(request, shipped), provenance=Provenance.UNTRUSTED
    ), shipped["crawler"]["browser"]["kwargs"]


@pytest.mark.parametrize(
    "request_config",
    [{"type": "BrowserConfig", "params": {"headless": True}}, {"headless": True}, {}, None],
)
def test_an_omitted_viewport_and_user_agent_come_from_the_server(request_config):
    loaded, shipped = _load(request_config)

    assert (loaded.viewport_width, loaded.viewport_height) == (
        shipped["viewport_width"],
        shipped["viewport_height"],
    )
    assert loaded.user_agent == shipped["user_agent"]
    assert '"Google Chrome";v="156"' in loaded.headers["sec-ch-ua"]


def test_a_requested_viewport_and_user_agent_are_kept():
    loaded, _ = _load(
        {
            "type": "BrowserConfig",
            "params": {"viewport_width": 800, "viewport_height": 1000, "user_agent": "probe/1.0"},
        }
    )

    assert (loaded.viewport_width, loaded.viewport_height, loaded.user_agent) == (
        800,
        1000,
        "probe/1.0",
    )


def test_a_requested_viewport_object_wins_over_the_server_dimensions():
    loaded, _ = _load(
        {"type": "BrowserConfig", "params": {"viewport": {"type": "dict", "value": {"width": 1024}}}}
    )

    assert loaded.viewport_width == 1024


def test_a_random_user_agent_request_is_left_to_its_generator():
    loaded, shipped = _load({"type": "BrowserConfig", "params": {"user_agent_mode": "random"}})

    assert loaded.user_agent != shipped["user_agent"]
