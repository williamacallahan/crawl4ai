"""Regression tests for DomainMapper host-scheme handling.

Validates that the scheme which successfully reached a host during
_validate_hosts is recorded and reused by the downstream scanning methods,
so HTTP-only hosts yield discovered URLs instead of silently failing over
HTTPS. HTTPS remains the default when no scheme is recorded.
"""
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from crawl4ai import DomainMapper, DomainMapperConfig


def _make_mapper():
    """Build a mapper without running __init__ (no real httpx client)."""
    mapper = DomainMapper.__new__(DomainMapper)
    mapper.logger = None
    mapper.client = AsyncMock()
    mapper._host_schemes = {}
    mapper._rate_sem = None
    return mapper


# ════════════════════════════════════════════════════════════════════════
#  _validate_hosts records the working scheme
# ════════════════════════════════════════════════════════════════════════

class TestValidateHostsRecordsScheme:

    @pytest.mark.asyncio
    async def test_http_only_host_recorded_as_http(self):
        """HTTPS throws (SSL), HTTP returns 2xx -> scheme recorded as http."""
        mapper = _make_mapper()

        async def head(url, **kwargs):
            if url.startswith("https://"):
                raise Exception("SSL: SSLV3_ALERT_HANDSHAKE_FAILURE")
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {}
            return resp

        mapper.client.head = AsyncMock(side_effect=head)
        config = DomainMapperConfig()

        hosts = await mapper._validate_hosts({"httponly.example"}, config)

        assert "httponly.example" in hosts
        assert mapper._host_schemes["httponly.example"] == "http"

    @pytest.mark.asyncio
    async def test_https_capable_host_recorded_as_https(self):
        """HTTPS returns 2xx -> scheme recorded as https (no HTTP fallback)."""
        mapper = _make_mapper()

        async def head(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {}
            return resp

        mapper.client.head = AsyncMock(side_effect=head)
        config = DomainMapperConfig()

        hosts = await mapper._validate_hosts({"secure.example"}, config)

        assert "secure.example" in hosts
        assert mapper._host_schemes["secure.example"] == "https"
        # HTTP fallback must NOT have been attempted for HTTPS-capable host
        requested = [c.args[0] for c in mapper.client.head.call_args_list]
        assert len(requested) == 1
        assert requested[0].startswith("https://")

    @pytest.mark.asyncio
    async def test_unreachable_host_has_no_scheme_entry(self):
        """Host unreachable on both schemes -> not validated, no scheme entry."""
        mapper = _make_mapper()

        async def head(url, **kwargs):
            raise Exception("connect EHOSTUNREACH")

        mapper.client.head = AsyncMock(side_effect=head)
        config = DomainMapperConfig()

        hosts = await mapper._validate_hosts({"dead.example"}, config)

        assert "dead.example" not in hosts
        assert "dead.example" not in mapper._host_schemes

    @pytest.mark.asyncio
    async def test_host_schemes_reset_per_validate_call(self):
        """_validate_hosts must clear stale schemes from previous scans."""
        mapper = _make_mapper()
        mapper._host_schemes = {"stale.example": "http"}

        async def head(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {}
            return resp

        mapper.client.head = AsyncMock(side_effect=head)
        config = DomainMapperConfig()

        await mapper._validate_hosts({"fresh.example"}, config)

        assert "stale.example" not in mapper._host_schemes
        assert "fresh.example" in mapper._host_schemes

    @pytest.mark.asyncio
    async def test_redirect_target_still_records_scheme(self):
        """A 3xx redirect counts as a reachable scheme and is recorded."""
        mapper = _make_mapper()

        async def head(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 301
            resp.headers = {"location": "https://canonical.example/"}
            return resp

        mapper.client.head = AsyncMock(side_effect=head)
        config = DomainMapperConfig()

        hosts = await mapper._validate_hosts({"old.example"}, config)

        assert "old.example" in hosts
        assert mapper._host_schemes["old.example"] == "https"
        assert mapper._host_redirects["old.example"] == "canonical.example"


# ════════════════════════════════════════════════════════════════════════
#  Four scanning methods honor the recorded scheme (and default to https)
# ════════════════════════════════════════════════════════════════════════

class TestScanningMethodsUseRecordedScheme:

    @pytest.mark.asyncio
    async def test_fingerprint_soft_404_uses_recorded_http_scheme(self):
        mapper = _make_mapper()
        mapper._host_schemes = {"httponly.example": "http"}
        resp = MagicMock()
        resp.status_code = 404
        resp.content = b"<html><title>Not Found</title></html>"
        resp.headers = {"content-length": "40"}
        mapper.client.get = AsyncMock(return_value=resp)

        await mapper._fingerprint_soft_404("httponly.example", DomainMapperConfig())

        requested_url = mapper.client.get.call_args.args[0]
        assert requested_url.startswith("http://httponly.example/c4ai-probe-")

    @pytest.mark.asyncio
    async def test_fingerprint_soft_404_defaults_to_https(self):
        mapper = _make_mapper()
        mapper._host_schemes = {}
        resp = MagicMock()
        resp.status_code = 404
        resp.content = b""
        resp.headers = {"content-length": "0"}
        mapper.client.get = AsyncMock(return_value=resp)

        await mapper._fingerprint_soft_404("secure.example", DomainMapperConfig())

        requested_url = mapper.client.get.call_args.args[0]
        assert requested_url.startswith("https://secure.example/c4ai-probe-")

    @pytest.mark.asyncio
    async def test_probe_paths_uses_recorded_http_scheme(self):
        mapper = _make_mapper()
        mapper._host_schemes = {"httponly.example": "http"}
        head_resp = MagicMock()
        head_resp.status_code = 200
        head_resp.url = "http://httponly.example/login"
        mapper.client.head = AsyncMock(return_value=head_resp)

        urls = await mapper._probe_paths(
            "httponly.example", ["/login"], None, DomainMapperConfig()
        )

        requested = [c.args[0] for c in mapper.client.head.call_args_list]
        assert all(u.startswith("http://httponly.example/") for u in requested)
        assert urls == ["http://httponly.example/login"]

    @pytest.mark.asyncio
    async def test_probe_paths_defaults_to_https(self):
        mapper = _make_mapper()
        mapper._host_schemes = {}
        head_resp = MagicMock()
        head_resp.status_code = 200
        head_resp.url = "https://secure.example/login"
        mapper.client.head = AsyncMock(return_value=head_resp)

        await mapper._probe_paths(
            "secure.example", ["/login"], None, DomainMapperConfig()
        )

        requested = [c.args[0] for c in mapper.client.head.call_args_list]
        assert all(u.startswith("https://secure.example/") for u in requested)

    @pytest.mark.asyncio
    async def test_discover_feeds_uses_recorded_http_scheme(self):
        mapper = _make_mapper()
        mapper._host_schemes = {"httponly.example": "http"}
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "application/rss+xml"}
        resp.text = (
            '<?xml version="1.0"?><rss version="2.0"><channel>'
            '<item><link>http://httponly.example/post-1</link></item>'
            '</channel></rss>'
        )
        resp.url = "http://httponly.example/feed"
        mapper.client.get = AsyncMock(return_value=resp)

        urls = await mapper._discover_feeds("httponly.example", DomainMapperConfig())

        requested = [c.args[0] for c in mapper.client.get.call_args_list]
        assert all(u.startswith("http://httponly.example/") for u in requested)
        assert any("post-1" in u for u in urls)

    @pytest.mark.asyncio
    async def test_discover_feeds_defaults_to_https(self):
        mapper = _make_mapper()
        mapper._host_schemes = {}
        resp = MagicMock()
        resp.status_code = 404
        resp.headers = {"content-type": "text/html"}
        resp.text = ""
        resp.url = "https://secure.example/feed"
        mapper.client.get = AsyncMock(return_value=resp)

        await mapper._discover_feeds("secure.example", DomainMapperConfig())

        requested = [c.args[0] for c in mapper.client.get.call_args_list]
        assert all(u.startswith("https://secure.example/") for u in requested)

    @pytest.mark.asyncio
    async def test_scan_homepage_uses_recorded_http_scheme(self):
        mapper = _make_mapper()
        mapper._host_schemes = {"httponly.example": "http"}
        html = (
            "<html><head><title>Home</title></head><body>"
            '<a href="/about">About</a>'
            '<a href="/blog">Blog</a>'
            "</body></html>"
        )
        resp = MagicMock()
        resp.status_code = 200
        resp.text = html
        resp.url = "http://httponly.example/"
        mapper.client.get = AsyncMock(return_value=resp)

        urls = await mapper._scan_homepage(
            "httponly.example", "httponly.example", DomainMapperConfig()
        )

        requested_url = mapper.client.get.call_args.args[0]
        assert requested_url == "http://httponly.example/"
        assert any("about" in u for u in urls)
        assert all(u.startswith("http://httponly.example/") for u in urls)

    @pytest.mark.asyncio
    async def test_scan_homepage_defaults_to_https(self):
        mapper = _make_mapper()
        mapper._host_schemes = {}
        resp = MagicMock()
        resp.status_code = 404
        resp.text = ""
        resp.url = "https://secure.example/"
        mapper.client.get = AsyncMock(return_value=resp)

        await mapper._scan_homepage(
            "secure.example", "secure.example", DomainMapperConfig()
        )

        requested_url = mapper.client.get.call_args.args[0]
        assert requested_url == "https://secure.example/"


# ════════════════════════════════════════════════════════════════════════
#  End-to-end: HTTP-only local server (no TLS) -> yields URLs over HTTP
# ════════════════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def mapper():
    async with DomainMapper() as m:
        yield m


@pytest.mark.asyncio
async def test_http_only_host_yields_urls(local_server, mapper):
    """An HTTP-only host (the local_server fixture serves plain HTTP on
    localhost with no TLS) must yield discovered URLs over HTTP.

    Before the scheme-tracking fix the scanning methods hardcoded HTTPS and
    silently returned 0 URLs for such hosts despite validating as live.
    """
    parsed = urlparse(local_server)
    host = f"{parsed.hostname}:{parsed.port}"

    config = DomainMapperConfig(
        source="probe+homepage",
        include_subdomains=False,
        extract_head=False,
        soft_404_detection=False,
        filter_nonsense_urls=False,
        verbose=False,
        force=True,
    )

    results = await mapper.scan(host, config)

    assert len(results) > 0, (
        "HTTP-only host should yield discovered URLs after the fix; "
        "got 0 (scanning methods likely still using HTTPS)"
    )
    assert mapper._host_schemes.get(host) == "http"
    assert all(r["url"].startswith("http://") for r in results)
    sources = {r["source"] for r in results}
    assert "probe" in sources
    assert "homepage" in sources
