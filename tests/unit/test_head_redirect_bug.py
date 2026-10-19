"""Regression tests for the DomainMapper Phase 3 head-extraction
off-host redirect leak (the analog of ``test_probe_redirect_bug.py``).

``_do_fetch_head`` calls ``seeder._fetch_head`` which follows up to
``max_redirects`` hops with no host filtering and returns the
post-redirect final URL. ``_scan_host`` stamps every record with
``host=<scanned host>`` and the pipeline never re-derives ``host`` from
the URL, so a head-GET that redirects off-site (e.g.
``example.com/login`` -> ``auth.external-sso.com/sso``, parked domain,
auth wall) ended up with ``url`` on the foreign domain while ``host``
still said the scanned host — the same misattribution bug class already
fixed in ``_do_probe`` (d2910997), feeds, and sitemaps (57deb0dd). The
foreign host's ``<title>``/meta tags also leaked into ``head_data``.

The fix mirrors ``_do_probe``: adopt ``final_url`` only when its host
matches ``r["host"]`` (lowercased, port-stripped, anchored ``www.``-strip
via ``removeprefix``), and reject the record (``status="not_valid"``,
no ``head_data``) when it does not.
"""
from unittest.mock import AsyncMock, patch
from urllib.parse import urlparse

import httpx
import pytest

from crawl4ai.domain_mapper import DomainMapper
from crawl4ai.async_configs import DomainMapperConfig


ONHOST_BODY = (
    b"<html><head><title>About</title>"
    b"<meta name='description' content='about us'></head></html>"
)
FOREIGN_BODY = (
    b"<html><head><title>External SSO</title>"
    b"<meta name='description' content='sso portal'></head></html>"
)


def _offhost_handler(scanned="example.com", foreign="auth.external-sso.com"):
    """``<scanned>/login`` 302-> foreign; foreign returns 200 with
    FOREIGN_BODY. Records ``(method, host, path)`` per request."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = (request.url.host or "").lower()
        path = request.url.path or ""
        requests.append((request.method, host, path))
        if host == scanned and path == "/login":
            return httpx.Response(
                302, headers={"Location": f"https://{foreign}/sso"},
                request=request)
        if host == foreign:
            return httpx.Response(200, content=FOREIGN_BODY, request=request)
        return httpx.Response(404, request=request)

    return handler, requests


def _mapper(handler):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=15, http2=False
    )
    return DomainMapper(client=client), client


def _bare_mapper():
    """Mapper without __init__ — for tests that drive ``_do_fetch_head``
    directly with a mock seeder."""
    mapper = DomainMapper.__new__(DomainMapper)
    mapper.logger = None
    return mapper


def _record(url, host, source="probe"):
    return {"url": url, "host": host, "source": source,
            "status": "valid", "head_data": {}}


def _norm(h):
    return h.lower().split(":")[0].removeprefix("www.")


#  Direct _do_fetch_head host guard (mock seeder) ────────────────────────


class TestDoFetchHeadHostGuard:
    """Direct ``_do_fetch_head`` via a mock seeder — the bug report's
    reproduction shape (seeder returns ``(ok, html, final_url)``)."""

    @pytest.mark.asyncio
    async def test_offhost_redirect_dropped_no_foreign_head(self):
        """Headline bug: foreign ``final_url`` must NOT overwrite
        ``r["url"]``, foreign HTML must NOT populate ``head_data``,
        record flagged ``not_valid``, ``netloc(url) == host`` holds."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        seeder._fetch_head = AsyncMock(return_value=(
            True, "<html><head><title>SSO Login</title></head></html>",
            "https://sso.some-idp.com/login",
        ))
        r = _record("https://example.com/login", "example.com")
        await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["url"] == "https://example.com/login", f"url leaked: {r}"
        assert r["status"] == "not_valid", f"should be not_valid: {r}"
        assert r["head_data"] == {}, f"foreign head_data leaked: {r}"
        assert _norm(urlparse(r["url"]).netloc) == _norm(r["host"])

    @pytest.mark.asyncio
    async def test_onhost_redirect_url_adopted(self):
        """Same-host redirect (trailing slash, http->https) admitted;
        ``head_data`` parsed from the response; status stays ``valid``."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        seeder._fetch_head = AsyncMock(return_value=(
            True, "<html><head><title>About</title></head></html>",
            "https://example.com/about/",
        ))
        r = _record("https://example.com/about", "example.com")
        await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["url"] == "https://example.com/about/", f"adopt: {r}"
        assert r["status"] == "valid"
        assert r["head_data"].get("title") == "About"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("final_url", [
        "https://example.com/",   # final == initial (no redirect)
        "",                        # empty final_url
        None,                      # defensive None final_url
    ])
    async def test_no_redirect_or_empty_final_keeps_url_populates_head(
        self, final_url
    ):
        """``r["url"]`` unchanged and ``head_data`` populated when no
        actual redirect occurred (preserves the skip-when-empty
        behavior)."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        seeder._fetch_head = AsyncMock(return_value=(
            True, "<html><head><title>Home</title></head></html>",
            final_url,
        ))
        r = _record("https://example.com/", "example.com")
        await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["url"] == "https://example.com/", f"unchanged: {r}"
        assert r["status"] == "valid"
        assert r["head_data"].get("title") == "Home"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("initial,final,host", [
        ("https://example.com/login", "https://www.example.com/login",
         "example.com"),
        ("https://www.example.com/login", "https://example.com/login",
         "www.example.com"),
    ])
    async def test_www_alias_redirect_adopted(self, initial, final, host):
        """Canonical ``www.``-aliasing admitted in both directions."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        seeder._fetch_head = AsyncMock(return_value=(
            True, "<html><head><title>Login</title></head></html>", final,
        ))
        r = _record(initial, host)
        await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["url"] == final, f"www-alias should adopt: {r}"
        assert r["status"] == "valid"
        assert r["head_data"].get("title") == "Login"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("host,final_url", [
        ("example.com:443", "https://example.com:443/path"),
        ("example.com", "https://example.com:443/login"),
    ])
    async def test_port_suffix_normalized_for_comparison(self, host,
                                                         final_url):
        """Port suffixes stripped via ``split(':')[0]`` before
        comparison (mirrors ``_do_probe``)."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        seeder._fetch_head = AsyncMock(return_value=(
            True, "<html><head><title>Home</title></head></html>", final_url,
        ))
        r = _record("https://example.com/path", host)
        await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["url"] == final_url, f"port-tolerant match: {r}"
        assert r["status"] == "valid"

    @pytest.mark.asyncio
    async def test_unparseable_final_url_dropped(self):
        """A ``final_url`` with empty ``netloc`` cannot equal the host;
        record marked ``not_valid``."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        seeder._fetch_head = AsyncMock(return_value=(
            True, "<html><head><title>Foreign</title></head></html>",
            "not-a-valid-url",
        ))
        r = _record("https://example.com/login", "example.com")
        await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["url"] == "https://example.com/login", f"unparseable: {r}"
        assert r["status"] == "not_valid"
        assert r["head_data"] == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fail_mode", ["ok_false", "raise_connect"])
    async def test_fetch_failure_keeps_url_marks_invalid(self, fail_mode):
        """``ok=False`` OR seeder exception marks ``not_valid`` and leaves
        ``url``/``head_data`` unchanged (pre-fix behavior preserved)."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        if fail_mode == "ok_false":
            seeder._fetch_head = AsyncMock(return_value=(False, "", ""))
        else:
            seeder._fetch_head = AsyncMock(side_effect=httpx.ConnectError("x"))
        r = _record("https://example.com/login", "example.com")
        await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["status"] == "not_valid"
        assert r["url"] == "https://example.com/login"
        assert r["head_data"] == {}

    @pytest.mark.asyncio
    async def test_parse_head_exception_marks_invalid_keeps_adopted_url(self):
        """``_parse_head`` raising marks ``not_valid`` but retains the
        on-host adopted ``final_url``; ``head_data`` stays empty."""
        mapper = _bare_mapper()
        seeder = AsyncMock()
        seeder._fetch_head = AsyncMock(return_value=(
            True, "<html><head><title>About</title></head></html>",
            "https://example.com/about/",
        ))
        r = _record("https://example.com/about", "example.com")
        with patch("crawl4ai.domain_mapper._parse_head",
                   side_effect=Exception("boom")):
            await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        assert r["status"] == "not_valid", f"parse fail -> not_valid: {r}"
        assert r["url"] == "https://example.com/about/", f"retained: {r}"
        assert r["head_data"] == {}


#  _do_fetch_head via real seeder redirect loop (MockTransport) ─────────


class TestExtractHeadsOffHostRedirect:
    """Drive ``_do_fetch_head`` via the real httpx redirect machinery
    used by ``seeder._fetch_head`` (follows redirects manually,
    re-issuing the GET per hop)."""

    @pytest.mark.asyncio
    async def test_offhost_redirect_marked_invalid_no_foreign_url(self):
        """Headline bug via the production redirect mechanism."""
        handler, requests = _offhost_handler()
        mapper, client = _mapper(handler)
        try:
            seeder = await mapper._get_seeder()
            r = _record("https://example.com/login", "example.com")
            await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
            assert r["url"] == "https://example.com/login", f"leaked: {r}"
            assert r["status"] == "not_valid"
            assert r["head_data"] == {}
            # Sanity: the seeder DID follow the redirect to the foreign
            # host (the rejection runs after the GET, before _parse_head).
            foreign = "auth.external-sso.com"
            assert any(req[1] == foreign for req in requests), \
                "seeder must reach the foreign host"
        finally:
            await mapper.close()

    @pytest.mark.asyncio
    async def test_onhost_redirect_adopted_via_real_seeder(self):
        scanned = "example.com"

        def handler(request: httpx.Request) -> httpx.Response:
            host = (request.url.host or "").lower()
            path = request.url.path or ""
            if host == scanned and path == "/about":
                return httpx.Response(
                    302, headers={"Location": f"https://{scanned}/about/"},
                    request=request)
            if host == scanned and path == "/about/":
                return httpx.Response(200, content=ONHOST_BODY, request=request)
            return httpx.Response(404, request=request)

        mapper, client = _mapper(handler)
        try:
            seeder = await mapper._get_seeder()
            r = _record(f"https://{scanned}/about", scanned)
            await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
            assert r["url"] == f"https://{scanned}/about/", f"adopt: {r}"
            assert r["status"] == "valid"
            assert r["head_data"].get("title") == "About"
        finally:
            await mapper.close()

    @pytest.mark.asyncio
    async def test_multihop_redirect_landing_offhost_dropped(self):
        """Two-hop redirect ending off-host (intermediate hop on the
        scanned host) must still be rejected — only the final
        ``final_url`` matters."""
        scanned = "example.com"
        foreign = "auth.external-sso.com"

        def handler(request: httpx.Request) -> httpx.Response:
            host = (request.url.host or "").lower()
            path = request.url.path or ""
            if host == scanned and path == "/login":
                return httpx.Response(
                    302, headers={"Location": f"https://{scanned}/sso-redirect"},
                    request=request)
            if host == scanned and path == "/sso-redirect":
                return httpx.Response(
                    302, headers={"Location": f"https://{foreign}/sso"},
                    request=request)
            if host == foreign:
                return httpx.Response(200, content=FOREIGN_BODY, request=request)
            return httpx.Response(404, request=request)

        mapper, client = _mapper(handler)
        try:
            seeder = await mapper._get_seeder()
            r = _record("https://example.com/login", "example.com")
            await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
            assert r["url"] == "https://example.com/login", f"leaked: {r}"
            assert r["status"] == "not_valid"
            assert r["head_data"] == {}
        finally:
            await mapper.close()

    @pytest.mark.asyncio
    async def test_www_prefix_alias_admitted_via_real_seeder(self):
        """``example.com`` -> ``www.example.com`` admitted via real-URL
        normalization (not mock-seeder shortcut)."""
        scanned = "example.com"

        def handler(request: httpx.Request) -> httpx.Response:
            host = (request.url.host or "").lower()
            path = request.url.path or ""
            if host == scanned and path == "/login":
                return httpx.Response(
                    302, headers={"Location": f"https://www.{scanned}/login"},
                    request=request)
            if host == f"www.{scanned}":
                return httpx.Response(200, content=ONHOST_BODY, request=request)
            return httpx.Response(404, request=request)

        mapper, client = _mapper(handler)
        try:
            seeder = await mapper._get_seeder()
            r = _record(f"https://{scanned}/login", scanned)
            await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
            assert r["url"] == f"https://www.{scanned}/login", f"adopt: {r}"
            assert r["status"] == "valid"
            assert r["head_data"].get("title") == "About"
        finally:
            await mapper.close()


#  Anchored ``www.`` normalization ──────────────────────────────────────


class TestAnchoredWwwNormalization:
    """The fix uses ``removeprefix("www.")`` (anchored), not
    ``replace("www.", "")`` (unanchored). Unanchored would conflate
    ``mywww.example.com`` with ``myexample.com``, admitting an off-host
    redirect."""

    @pytest.mark.asyncio
    async def test_host_containing_www_substring_not_conflated(self):
        scanned = "mywww.example.com"
        foreign = "myexample.com"

        def handler(request: httpx.Request) -> httpx.Response:
            host = (request.url.host or "").lower()
            path = request.url.path or ""
            if host == scanned and path == "/login":
                return httpx.Response(
                    302, headers={"Location": f"https://{foreign}/login"},
                    request=request)
            if host == foreign:
                return httpx.Response(200, content=FOREIGN_BODY, request=request)
            return httpx.Response(404, request=request)

        mapper, client = _mapper(handler)
        try:
            seeder = await mapper._get_seeder()
            r = _record(f"https://{scanned}/login", scanned)
            await mapper._do_fetch_head(r, seeder, DomainMapperConfig())
        finally:
            await mapper.close()
        # Anchored removeprefix: "mywww.example.com" has no "www." prefix ->
        # distinct from "myexample.com" -> dropped.
        assert r["url"] == f"https://{scanned}/login", f"leaked: {r}"
        assert r["status"] == "not_valid"


#  scan()-level invariant ───────────────────────────────────────────────


def _scan_handler():
    """Phase 2 probe HEADs ``/login`` and ``/about`` -> 200 on scanned
    host (so they enter the result set); Phase 3 head GET ``/login`` ->
    302 off-host (bug trigger); GET ``/about`` -> 200 on-host (sanity).
    ``_validate_hosts`` HEAD ``/`` -> 200. Foreign: FOREIGN_BODY."""
    scanned = "example.com"
    foreign = "auth.external-sso.com"

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.method
        host = (request.url.host or "").lower()
        path = request.url.path or ""
        if method == "HEAD" and host == scanned and path == "/":
            return httpx.Response(200, request=request)
        if method == "HEAD" and host == scanned and path in ("/login",
                                                              "/about"):
            return httpx.Response(200, request=request)
        if method == "GET" and host == scanned and path == "/login":
            return httpx.Response(
                302, headers={"Location": f"https://{foreign}/sso"},
                request=request)
        if method == "GET" and host == scanned and path == "/about":
            return httpx.Response(200, content=ONHOST_BODY, request=request)
        if host == foreign:
            return httpx.Response(200, content=FOREIGN_BODY, request=request)
        return httpx.Response(404, request=request)

    return handler


async def _scan_with_probe_then_head(scanned="example.com", **cfg_kwargs):
    """Run ``scan("example.com")`` with the shared scan handler and the
    ``_discover_hosts`` patch; close the mapper before returning."""
    handler = _scan_handler()
    mapper, client = _mapper(handler)
    mapper._host_schemes = {scanned: "https"}
    try:
        with patch.object(
            DomainMapper, "_discover_hosts",
            AsyncMock(return_value={scanned}),
        ):
            results = await mapper.scan(scanned, DomainMapperConfig(
                source="probe", include_subdomains=False, max_urls=-1,
                soft_404_detection=False, filter_nonsense_urls=False,
                hits_per_sec=0, http_timeout=5, force=True, **cfg_kwargs,
            ))
    finally:
        await mapper.close()
    return results


class TestScanNeverMisattributesHeadRedirect:
    """``scan()``-level invariant: every record's URL netloc must equal
    its ``host`` after canonical ``www.``-strip. Mirrors the probe-source
    invariant in ``test_domain_mapper_unit.py``."""

    @pytest.mark.asyncio
    async def test_scan_never_misattributes_offhost_head_redirect(self):
        """A scanned-host URL that 200s in Phase 2 probe but 302s off-host
        during Phase 3 head extraction must NOT have ``url`` overwritten
        with the foreign destination; record ``not_valid`` with no foreign
        ``head_data``; sibling ``/about`` stays ``valid`` with head_data."""
        scanned = "example.com"
        foreign = "auth.external-sso.com"
        results = await _scan_with_probe_then_head(scanned, extract_head=True)

        misattributed = [
            r for r in results
            if _norm(urlparse(r["url"]).netloc) != _norm(r["host"])
        ]
        assert not misattributed, f"misattributed: {misattributed}; all: {results}"
        assert not any(urlparse(r["url"]).netloc.lower() == foreign
                       for r in results), f"foreign URL leaked: {results}"

        login = [r for r in results if r["url"] == f"https://{scanned}/login"]
        assert login, f"/login should survive Phase 2: {results}"
        for r in login:
            assert r["status"] == "not_valid", f"not_valid: {r}"
            assert r["head_data"] == {}, f"foreign head_data: {r}"

        about = [r for r in results if r["url"] == f"https://{scanned}/about"]
        assert about, f"/about missing: {results}"
        for r in about:
            assert r["status"] == "valid"
            assert r["head_data"].get("title") == "About"

    @pytest.mark.asyncio
    async def test_scan_default_config_runs_head_extraction_guarded(self):
        """``extract_head`` defaults to True — the buggy path runs by
        default. The guard rejects off-host head redirects even when the
        caller does not explicitly set ``extract_head``."""
        scanned = "example.com"
        foreign = "auth.external-sso.com"
        results = await _scan_with_probe_then_head(scanned)

        assert not any(urlparse(r["url"]).netloc.lower() == foreign
                       for r in results), f"default leaked foreign URL: {results}"
        for r in results:
            if r["url"] == f"https://{scanned}/login":
                assert r["status"] == "not_valid", f"not_valid default: {r}"
                assert r["head_data"] == {}
