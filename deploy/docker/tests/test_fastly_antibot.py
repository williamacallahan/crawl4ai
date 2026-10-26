import pytest

from crawl4ai.antibot_detector import is_blocked


@pytest.mark.parametrize("padding_size", [0, 20000, 60000])
def test_fastly_client_challenge_is_blocked_at_http_200(padding_size):
    # Structure retained from OREF's HTTP 200 Client Challenge capture; omit
    # its per-request CAPTCHA data. Padding puts the scripts past fast scans.
    html = (
        "<html><head><title>Client Challenge</title></head><body><p>"
        + "Please enable JavaScript to proceed. " * 5
        + "x" * padding_size
        + '</p><script src="/_fs-ch-example/errors.js"></script>'
        + '<script src="/_fs-ch-example/script.js?reload=true"></script>'
        + "</body></html>"
    )

    assert is_blocked(200, html) == (True, "Fastly client challenge")


@pytest.mark.parametrize(
    "title, extra",
    [
        ("Client Challenge", ""),
        ("Public article", '<script src="/_fs-ch-example/script.js"></script>'),
        (
            "Client Challenge",
            '<pre>&lt;script src="/_fs-ch-example/script.js"&gt;&lt;/script&gt;</pre>',
        ),
        (
            "Public article",
            '<!-- <title>Client Challenge</title>'
            '<script src="/_fs-ch-example/script.js"></script> -->',
        ),
    ],
)
def test_public_content_is_not_a_fastly_challenge(title, extra):
    html = (
        f"<html><head><title>{title}</title></head><body><article><p>"
        + "Public information about client challenges and website delivery. " * 10
        + f"</p>{extra}</article></body></html>"
    )

    assert is_blocked(200, html) == (False, "")
