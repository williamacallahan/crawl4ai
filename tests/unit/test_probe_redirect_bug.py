"""Regression tests for the DomainMapper probe-source off-host-redirect leak.

``_do_probe`` issues the probe ``HEAD`` with ``follow_redirects=True`` and
returns ``str(resp.url)`` — the *final* URL after redirects. Pre-fix it had
no host check, so a probe path that redirected off-site (e.g.
``example.com/login`` -> ``auth.external-sso.com/sso``) injected a foreign
URL into the scanned host's result set, misattributed as
``host=example.com``. Under the default config the same root cause also
made the scanner issue out-of-scope soft-404 GET (Phase 2) and
head-extraction GET (Phase 3) requests to the foreign host.

The fix threads the scanned ``host`` into ``_do_probe`` and applies an
exact-host predicate (anchored ``www.``-prefix stripping via
``removeprefix``) to ``str(resp.url)`` *before* the soft-404 GET.

These tests use ``httpx.MockTransport`` so httpx's own redirect-following
logic produces a foreign ``resp.url`` (the production mechanism) rather
than stubbing ``resp.url`` by hand. They cover facets of the bug NOT
covered by the ``scan()``-level invariant test in
``test_domain_mapper_unit.py`` (which uses ``unittest.mock``): the real
httpx redirect mechanism, the non-200-`<400` edge case, out-of-scope
request issuance, ``www.``-aliasing, on-host soft-404 non-regression, and
anchored ``www.`` normalization.
"""
from urllib.parse import urlparse

import httpx
import pytest

from crawl4ai.domain_mapper import DomainMapper, Soft404Fingerprint
from crawl4ai.async_configs import DomainMapperConfig

import hashlib

SOFT404_BODY = (
    b"<html><head><title>Page Not Found</title></head>"
    b"<body>not found content xyzzy</body></html>"
)
ONHOST_BODY = b"<html><head><title>About</title></head><body>about us</body></html>"
FOREIGN_BODY = (
    b"<html><head><title>External SSO</title></head><body>sso portal</body></html>"
)


def _soft404_fp() -> Soft404Fingerprint:
    return Soft404Fingerprint(
        status_code=200, title="Page Not Found",
        content_length=len(SOFT404_BODY),
        body_hash=hashlib.md5(SOFT404_BODY[:2048]).hexdigest(),
    )


def _offhost_handler(scanned="example.com", foreign="auth.external-sso.com"):
    """Build a MockTransport handler emulating ``<scanned>/login`` -> foreign.

    ``/login`` redirects off-host; ``/about`` returns a real on-host page;
    ``/soft404-page`` returns the soft-404 body; ``/c4ai-probe-*`` returns the
    soft-404 fingerprint body; the foreign host returns 200. Records every
    request as ``(method, host)`` for out-of-scope-request assertions.
    """
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.method
        host = (request.url.host or "").lower()
        path = request.url.path or ""
        requests.append((method, host))
        if host == scanned and path == "/login":
            return httpx.Response(
                302, headers={"Location": f"https://{foreign}/sso"}, request=request
            )
        if host == foreign:
            return httpx.Response(200, content=FOREIGN_BODY, request=request)
        if host == scanned and path == "/about":
            return httpx.Response(200, content=ONHOST_BODY, request=request)
        if host == scanned and path == "/soft404-page":
            return httpx.Response(200, content=SOFT404_BODY, request=request)
        if host == scanned and path.startswith("/c4ai-probe-"):
            return httpx.Response(200, content=SOFT404_BODY, request=request)
        return httpx.Response(404, request=request)

    return handler, requests


def _mapper(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=15, http2=False)
    return DomainMapper(client=client, **kw), client


class TestDoProbeHostGuard:
    """Direct ``_do_probe`` via real httpx redirect machinery."""

    @pytest.mark.asyncio
    async def test_offhost_redirect_dropped(self):
        """The headline bug: ``/login`` redirects off-host; the post-redirect
        final URL must NOT be returned (pre-fix it returned the foreign URL)."""
        handler, _ = _offhost_handler()
        mapper, client = _mapper(handler)
        try:
            res = await mapper._do_probe(
                "https://example.com/login", "example.com", None,
                DomainMapperConfig(soft_404_detection=False),
            )
            assert res is None, f"foreign URL leaked from _do_probe: {res}"
        finally:
            await mapper.close()

    @pytest.mark.asyncio
    async def test_non200_final_status_dropped(self):
        """A non-200 but ``<400`` final status (e.g. 201) skips the soft-404
        GET branch entirely; pre-fix this fell through to ``return
        str(resp.url)``. The host guard drops it."""
        foreign = "auth.external-sso.com"

        def handler(request: httpx.Request) -> httpx.Response:
            host = (request.url.host or "").lower()
            path = request.url.path or ""
            if host == "example.com" and path == "/login":
                return httpx.Response(302, headers={"Location": f"https://{foreign}/sso"}, request=request)
            if host == foreign:
                return httpx.Response(201, content=FOREIGN_BODY, request=request)
            return httpx.Response(404, request=request)

        mapper, client = _mapper(handler)
        try:
            res = await mapper._do_probe(
                "https://example.com/login", "example.com", _soft404_fp(),
                DomainMapperConfig(soft_404_detection=True),
            )
            assert res is None, f"foreign URL leaked (201 final): {res}"
        finally:
            await mapper.close()

    @pytest.mark.asyncio
    async def test_www_alias_redirect_survives(self):
        """Canonical ``www.``-aliasing: ``example.com`` -> ``www.example.com``
        must be admitted (both normalize to ``example.com``)."""
        scanned = "example.com"

        def handler(request: httpx.Request) -> httpx.Response:
            host = (request.url.host or "").lower()
            path = request.url.path or ""
            if host == scanned and path == "/login":
                return httpx.Response(301, headers={"Location": f"https://www.{scanned}/login"}, request=request)
            if host == f"www.{scanned}":
                return httpx.Response(200, content=ONHOST_BODY, request=request)
            return httpx.Response(404, request=request)

        mapper, client = _mapper(handler)
        try:
            res = await mapper._do_probe(
                f"https://{scanned}/login", scanned, None,
                DomainMapperConfig(soft_404_detection=False),
            )
            assert res == f"https://www.{scanned}/login", f"www-alias should survive: {res}"
        finally:
            await mapper.close()


class TestProbePathsOffHostRedirect:

    @pytest.mark.asyncio
    async def test_no_outofscope_get_to_foreign_host(self):
        """With ``soft_404_detection=True`` and a 200 final on the foreign
        host, the soft-404 GET branch would (pre-fix) issue a GET to the
        foreign host. The guard fires *before* that GET, so no GET is issued
        to the foreign host during probing."""
        handler, requests = _offhost_handler()
        mapper, client = _mapper(handler)
        mapper._host_schemes = {"example.com": "https"}
        try:
            urls = await mapper._probe_paths(
                "example.com", ["/login"], _soft404_fp(),
                DomainMapperConfig(soft_404_detection=True),
            )
        finally:
            await mapper.close()
        assert urls == [], f"foreign URL leaked: {urls}"
        foreign = "auth.external-sso.com"
        foreign_gets = sum(1 for m, h in requests if m == "GET" and h == foreign)
        assert foreign_gets == 0, (
            f"scanner issued out-of-scope GET(s) to the foreign host: "
            f"{[r for r in requests if r[1] == foreign]}"
        )

    @pytest.mark.asyncio
    async def test_soft404_detection_still_works_onhost(self):
        """Regression guard: the host guard must NOT break legitimate on-host
        soft-404 detection. An on-host path returning the soft-404 body is
        dropped; a distinct on-host body is kept; the foreign URL is dropped."""
        handler, _ = _offhost_handler()
        mapper, client = _mapper(handler)
        mapper._host_schemes = {"example.com": "https"}
        try:
            urls = await mapper._probe_paths(
                "example.com", ["/about", "/soft404-page", "/login"],
                _soft404_fp(),
                DomainMapperConfig(soft_404_detection=True),
            )
        finally:
            await mapper.close()
        assert "https://example.com/about" in urls, f"on-host page should survive: {urls}"
        assert "https://example.com/soft404-page" not in urls, f"on-host soft-404 should be dropped: {urls}"
        assert not any(urlparse(u).netloc.lower() == "auth.external-sso.com" for u in urls), (
            f"foreign URL leaked: {urls}"
        )


class TestAnchoredWwwNormalization:
    """The fix uses ``host.removeprefix("www.")`` (anchored), NOT
    ``host.replace("www.", "")`` (unanchored). An unanchored strip would
    conflate ``mywww.example.com`` with ``myexample.com``, admitting an
    off-host redirect. Guards against re-introducing the pre-existing
    ``_on_host`` normalization defect (flagged in the bug report)."""

    @pytest.mark.asyncio
    async def test_host_containing_www_substring_not_conflated(self):
        scanned = "mywww.example.com"
        foreign = "myexample.com"

        def handler(request: httpx.Request) -> httpx.Response:
            host = (request.url.host or "").lower()
            path = request.url.path or ""
            if host == scanned and path == "/login":
                return httpx.Response(302, headers={"Location": f"https://{foreign}/login"}, request=request)
            if host == foreign:
                return httpx.Response(200, content=FOREIGN_BODY, request=request)
            return httpx.Response(404, request=request)

        mapper, client = _mapper(handler)
        mapper._host_schemes = {scanned: "https"}
        try:
            urls = await mapper._probe_paths(
                scanned, ["/login"], None,
                DomainMapperConfig(soft_404_detection=False),
            )
        finally:
            await mapper.close()
        # Anchored removeprefix: "mywww.example.com" has no "www." prefix →
        # distinct from "myexample.com" → dropped.
        # Unanchored replace: "mywww.example.com" → "myexample.com" would
        # match the foreign host and leak.
        assert all(urlparse(u).netloc.lower() == scanned for u in urls), (
            f"foreign host leaked via www-substring conflation: {urls}"
        )
        assert urls == [], f"expected empty, got {urls}"
