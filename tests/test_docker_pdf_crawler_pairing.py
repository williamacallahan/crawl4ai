"""Docker PDF strategies remain rejected; SDK PDF redirects remain supported."""

import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4ai.processors.pdf import PDFContentScrapingStrategy
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent

def _crawler_config_payload(with_pdf_strategy):
    params = {"cache_mode": "bypass"}
    if with_pdf_strategy:
        params["scraping_strategy"] = {
            "type": "PDFContentScrapingStrategy",
            "params": {},
        }
    return {"type": "CrawlerRunConfig", "params": params}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "deploy" / "docker"))
    return importlib.import_module("api")


@pytest.fixture
def pool_mock(api, monkeypatch):
    crawler_pool = importlib.import_module("crawler_pool")
    egress_broker = importlib.import_module("egress_broker")
    governor = importlib.import_module("governor")

    pooled = MagicMock()
    pooled.arun = AsyncMock(return_value=[{"success": True, "url": "https://example.com/"}])
    pooled.arun_many = AsyncMock(return_value=[])
    pooled._docker_admission_released = False
    pooled._docker_request_owned = False
    pooled.active_requests = 1  # release_crawler decrements this int
    mock = AsyncMock(return_value=pooled)
    monkeypatch.setattr(crawler_pool, "get_crawler", mock)
    monkeypatch.setattr(api, "_normalize_and_validate_seeds", AsyncMock(side_effect=lambda urls: urls))
    monkeypatch.setattr(api, "_track_request_start", AsyncMock(return_value="pdf-strategy"))
    monkeypatch.setattr(api, "_close_aborted_request", AsyncMock())
    monkeypatch.setattr(crawler_pool, "release_crawler", AsyncMock())
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _: None)
    monkeypatch.setattr(governor, "clamp_deep_crawl", lambda _: None)
    return mock


@pytest.fixture
def config(api):
    return importlib.import_module("utils").load_config()


@pytest.mark.asyncio
async def test_pdf_scraping_strategy_rejected_before_admission(api, pool_mock, config):
    with pytest.raises(HTTPException) as rejected:
        await api.handle_crawl_request(
            urls=["https://example.com/document.pdf"],
            browser_config={},
            crawler_config=_crawler_config_payload(with_pdf_strategy=True),
            config=config,
        )

    assert rejected.value.status_code == 400
    assert "PDFContentScrapingStrategy" in rejected.value.detail
    pool_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_pdf_strategy_with_hooks_rejected(api, pool_mock, config):
    # Valid hooks must not provide a second route to an untrusted PDF strategy.
    hooks = {"hooks": [{"action": "block_resources", "params": {"resource_types": ["image"]}}]}
    with pytest.raises(HTTPException) as exc_info:
        await api.handle_crawl_request(
            urls=["https://example.com/document.pdf"],
            browser_config={"type": "BrowserConfig", "params": {}},
            crawler_config=_crawler_config_payload(with_pdf_strategy=True),
            config=config,
            hooks_config=hooks,
        )

    assert exc_info.value.status_code == 400
    assert "PDFContentScrapingStrategy" in exc_info.value.detail
    pool_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_strategy_still_uses_pool(api, pool_mock, config):
    response = await api.handle_crawl_request(
        urls=["https://example.com/"],
        browser_config={"type": "BrowserConfig", "params": {}},
        crawler_config=_crawler_config_payload(with_pdf_strategy=False),
        config=config,
    )

    assert response["success"] is True
    pool_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_per_url_pdf_strategy_rejected_before_execution(api, pool_mock, config):
    pdf_config = _crawler_config_payload(with_pdf_strategy=True)
    plain_config = _crawler_config_payload(with_pdf_strategy=False)

    with pytest.raises(HTTPException) as rejected:
        await api.handle_crawl_request(
            urls=["https://example.com/document.pdf", "https://example.com/page.html"],
            browser_config={},
            crawler_config=plain_config,
            crawler_configs=[pdf_config, plain_config],
            config=config,
        )

    assert rejected.value.status_code == 400
    assert "PDFContentScrapingStrategy" in rejected.value.detail
    pool_mock.return_value.arun.assert_not_awaited()
    pool_mock.return_value.arun_many.assert_not_awaited()


# ---------------------------------------------------------------------------
# PDF download redirect handling (url_validator SSRF guard)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def redirect_server():
    """Real HTTP server: /hop redirects to /doc.pdf, which serves a tiny PDF."""
    import http.server
    import socket
    import threading

    pdf_bytes = (b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                 b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
                 b"xref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n"
                 b"0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\n"
                 b"startxref\n110\n%%EOF\n")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/hop":
                self.send_response(302)
                self.send_header("Location", "/doc.pdf")
                self.end_headers()
            elif self.path == "/loop":
                self.send_response(302)
                self.send_header("Location", "/loop")
                self.end_headers()
            elif self.path == "/gated":
                # Sets a cookie, then redirects to a target that demands it.
                self.send_response(302)
                self.send_header("Set-Cookie", "sess=abc123; Path=/")
                self.send_header("Location", "/gated-doc.pdf")
                self.end_headers()
            elif self.path == "/gated-doc.pdf":
                if "sess=abc123" not in self.headers.get("Cookie", ""):
                    self.send_response(403)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.end_headers()
                self.wfile.write(pdf_bytes)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.end_headers()
                self.wfile.write(pdf_bytes)

        def log_message(self, *args):
            pass

    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    httpd = http.server.ThreadingHTTPServer(("localhost", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://localhost:{port}"
    httpd.shutdown()


def test_download_validator_vets_every_redirect_hop(redirect_server):
    """The validator must see BOTH the original URL and the redirect target,
    and a raising validator must abort the download before the hop is fetched."""
    seen = []

    def validator(u):
        seen.append(u)
        if u.endswith("/doc.pdf"):
            raise ValueError("blocked hop")

    strategy = PDFContentScrapingStrategy(url_validator=validator)
    with pytest.raises(RuntimeError, match="Failed to download"):
        strategy._get_pdf_path(f"{redirect_server}/hop")

    assert seen == [f"{redirect_server}/hop", f"{redirect_server}/doc.pdf"]


def test_download_redirects_still_followed_without_validator(redirect_server):
    """Back-compat: with no validator, redirects are followed as before."""
    strategy = PDFContentScrapingStrategy()
    path = strategy._get_pdf_path(f"{redirect_server}/hop")
    try:
        assert Path(path).read_bytes().startswith(b"%PDF")
    finally:
        Path(path).unlink(missing_ok=True)


def test_download_carries_cookies_across_redirect_hops(redirect_server):
    """Cookies set by a redirecting host must reach the next hop.

    Following redirects by hand (needed so url_validator can vet each hop)
    means a bare requests.get() per hop starts with an empty cookie jar, so
    a host that sets a cookie and then redirects never gets it back. That is
    the normal shape for gated or CDN-signed PDFs, and allow_redirects=True
    used to handle it implicitly.
    """
    strategy = PDFContentScrapingStrategy()
    path = strategy._get_pdf_path(f"{redirect_server}/gated")
    try:
        assert Path(path).read_bytes().startswith(b"%PDF")
    finally:
        Path(path).unlink(missing_ok=True)


def test_download_redirect_loop_aborts(redirect_server):
    """An endless redirect chain must abort after the cap, not hang."""
    strategy = PDFContentScrapingStrategy()
    with pytest.raises(RuntimeError, match="[Tt]oo many redirects"):
        strategy._get_pdf_path(f"{redirect_server}/loop")
