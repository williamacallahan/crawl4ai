import pytest

from crawl4ai.antibot_detector import is_blocked


def test_cloudflare_interstitial_title_blocks_large_http_200_page():
    html = (
        "<html><head><title> \nJust  a\tmoment...</title></head><body><p>"
        + "Please wait while your browser is checked. " * 300
        + "</p></body></html>"
    )
    assert len(html) > 10000

    assert is_blocked(200, html) == (True, "Cloudflare interstitial")


def test_normal_http_200_body_can_mention_just_a_moment():
    html = (
        "<html><head><title>Customer service guide</title></head><body><article><p>"
        + "Please wait just a moment while checking your browser settings. " * 200
        + "</p></article></body></html>"
    )
    assert len(html) > 10000

    assert is_blocked(200, html) == (False, "")


@pytest.mark.parametrize("padding_size", [11000, 16000])
def test_cloudflare_challenge_options_block_large_http_200_page(padding_size):
    html = (
        "<html><head><title>Browser verification</title></head><body><p>"
        + "x" * padding_size
        + "</p><script>window._cf_chl_opt = {cType: 'managed'};</script>"
        + "</body></html>"
    )

    blocked, reason = is_blocked(200, html)

    assert blocked is True
    assert reason in (
        "Cloudflare challenge options",
        "Anti-bot script marker in <script>",
    )
