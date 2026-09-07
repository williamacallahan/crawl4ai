"""Regression tests for DomainMapper concurrency bounding.

Validates that ``DomainMapperConfig.concurrency`` — documented as "Max
concurrent requests across all hosts" — bounds in-flight HTTP requests across
*every* phase of a scan, not just Phase 3 head extraction:

  * Phase 1 host validation (``_validate_hosts``)
  * Phase 2 per-host source fetchers (robots/sitemap/probe/feed/homepage) and
    the cross-host ``asyncio.gather`` fan-out
  * Phase 3 head extraction (``_extract_heads`` -> ``AsyncUrlSeeder._fetch_head``)

The fix wraps ``self.client`` in a ``_BoundedClient`` whose ``get``/``head``/
``stream`` acquire an ``asyncio.Semaphore(config.concurrency)`` configured
per-scan. These tests use an ``httpx.AsyncClient`` backed by a counting
transport (no network) and run under ``-m "not network"``.
"""
import asyncio

import httpx
import pytest

from crawl4ai.domain_mapper import DomainMapper, _BoundedClient, _BoundedStream
from crawl4ai.async_configs import DomainMapperConfig


# ─────────────────────────────────────────────────────────────────────────
#  Counting transport — records peak concurrent in-flight requests
# ─────────────────────────────────────────────────────────────────────────


class _CountingTransport(httpx.AsyncBaseTransport):
    """Returns a flat 200 HTML page for every request and records the peak
    number of concurrently in-flight ``handle_async_request`` invocations."""

    def __init__(self, latency: float = 0.01, status: int = 200,
                 body: bytes = b"<html><head><title>x</title></head><body></body></html>"):
        self.in_flight = 0
        self.peak = 0
        self.count = 0
        self.latency = latency
        self.status = status
        self.body = body

    async def handle_async_request(self, request):
        self.in_flight += 1
        self.count += 1
        if self.in_flight > self.peak:
            self.peak = self.in_flight
        try:
            await asyncio.sleep(self.latency)
            return httpx.Response(
                self.status,
                content=self.body,
                headers={"content-type": "text/html"},
                request=request,
            )
        finally:
            self.in_flight -= 1


def _make_mapper(latency: float = 0.01, status: int = 200,
                 body: bytes = b"<html><head><title>x</title></head><body></body></html>"):
    """Build a DomainMapper whose client uses a fresh counting transport."""
    transport = _CountingTransport(latency=latency, status=status, body=body)
    client = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
    mapper = DomainMapper(client=client)
    return mapper, transport


def _patch_discover_hosts(mapper, hosts):
    """Make _discover_hosts return a fixed host set (skip real DNS/network).

    Stored as an instance attribute (a plain function, not a bound method),
    so the call ``self._discover_hosts(base_domain, sources, config)``
    invokes it without forwarding ``self``.
    """
    fixed = set(hosts)
    async def _fake(base_domain, sources, config):
        return fixed
    mapper._discover_hosts = _fake


# ─────────────────────────────────────────────────────────────────────────
#  Phase 2 — cross-host fan-out is bounded by config.concurrency
# ─────────────────────────────────────────────────────────────────────────


class TestPhase2Concurrency:

    @pytest.mark.asyncio
    async def test_concurrency_one_caps_phase2_to_one_inflight(self):
        """concurrency=1 with multiple hosts + all sources must never exceed
        1 in-flight request (the headline contract from the bug report)."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["h1.example", "h2.example", "h3.example"])
        config = DomainMapperConfig(
            source="sitemap+probe+feed+homepage+robots",
            concurrency=1,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        await mapper.scan("example.com", config)
        await mapper.close()
        assert transport.peak <= 1, f"concurrency=1 breached: peak={transport.peak}"
        # Sanity: the scan actually issued many requests, so the cap is
        # meaningful (not vacuously satisfied by zero activity).
        assert transport.count > 20, f"expected many requests, got {transport.count}"

    @pytest.mark.asyncio
    async def test_concurrency_two_caps_phase2_to_two_inflight(self):
        """concurrency=2 bounds in-flight to <= 2 even with a wide fan-out."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["h1.example", "h2.example", "h3.example"])
        config = DomainMapperConfig(
            source="sitemap+probe+feed+homepage+robots",
            concurrency=2,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        await mapper.scan("example.com", config)
        await mapper.close()
        assert transport.peak <= 2, f"concurrency=2 breached: peak={transport.peak}"
        assert transport.peak >= 2, (
            f"concurrency=2 should allow 2 concurrent; peak={transport.peak} "
            "suggests the scan did not fan out"
        )

    @pytest.mark.asyncio
    async def test_concurrency_default_does_not_exceed_cap(self):
        """Default concurrency=50 must bound in-flight to <= 50, even with a
        large host set that would otherwise exceed it (the pre-fix code
        reached ~1-per-host, unbounded)."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, [f"h{i}.example" for i in range(80)])
        config = DomainMapperConfig(
            source="sitemap+probe+feed+homepage+robots",
            concurrency=50,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        await mapper.scan("example.com", config)
        await mapper.close()
        assert transport.peak <= 50, (
            f"concurrency=50 breached: peak={transport.peak}"
        )

    @pytest.mark.asyncio
    async def test_single_host_concurrency_one_honoured(self):
        """Even with a single host, concurrency=1 must serialize all Phase 2
        fetchers (the conservative polite-user case from the bug report)."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["only.example"])
        config = DomainMapperConfig(
            source="sitemap+probe+feed+homepage+robots",
            concurrency=1,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        await mapper.scan("example.com", config)
        await mapper.close()
        assert transport.peak <= 1, f"single-host concurrency=1 breached: {transport.peak}"


# ─────────────────────────────────────────────────────────────────────────
#  Phase 1 — _validate_hosts is bounded
# ─────────────────────────────────────────────────────────────────────────


class TestValidateHostsConcurrency:

    @pytest.mark.asyncio
    async def test_validate_hosts_bounded_by_concurrency(self):
        """_validate_hosts HEADs every candidate host in parallel via
        asyncio.gather; concurrency must bound that fan-out."""
        mapper, transport = _make_mapper()
        # Configure the per-scan semaphore the way scan() does.
        mapper.client.set_concurrency(1)
        config = DomainMapperConfig(http_timeout=5.0)
        hosts = {f"h{i}.example" for i in range(10)}
        validated = await mapper._validate_hosts(hosts, config)
        await mapper.close()
        assert validated == hosts, "all hosts should validate against the flat 200 transport"
        assert transport.peak <= 1, (
            f"_validate_hosts breached concurrency=1: peak={transport.peak}"
        )
        assert transport.count == 10, (
            f"expected 10 HEADs (one per host, https succeeds first), "
            f"got {transport.count}"
        )

    @pytest.mark.asyncio
    async def test_validate_hosts_concurrency_two(self):
        mapper, transport = _make_mapper()
        mapper.client.set_concurrency(2)
        config = DomainMapperConfig(http_timeout=5.0)
        hosts = {f"h{i}.example" for i in range(10)}
        await mapper._validate_hosts(hosts, config)
        await mapper.close()
        assert transport.peak <= 2, f"_validate_hosts breached concurrency=2: {transport.peak}"


# ─────────────────────────────────────────────────────────────────────────
#  Phase 3 — head extraction is still bounded (no local semaphore regression)
# ─────────────────────────────────────────────────────────────────────────


class TestPhase3Concurrency:

    @pytest.mark.asyncio
    async def test_phase3_head_extraction_bounded(self):
        """extract_head=True triggers _extract_heads -> seeder._fetch_head ->
        client.stream. After removing the redundant local semaphore in
        _extract_heads, the _BoundedClient wrapper must still bound Phase 3."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["h1.example", "h2.example"])
        config = DomainMapperConfig(
            source="probe",
            concurrency=2,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=True,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        results = await mapper.scan("example.com", config)
        await mapper.close()
        # Phase 2 should have produced probe URLs for Phase 3 to fetch heads.
        assert len(results) > 0, "expected probe URLs before head extraction"
        assert transport.peak <= 2, (
            f"Phase 3 head extraction breached concurrency=2: peak={transport.peak}"
        )

    @pytest.mark.asyncio
    async def test_phase3_concurrency_one_serializes_streams(self):
        """concurrency=1 must serialize even the streaming head fetches."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["h1.example"])
        config = DomainMapperConfig(
            source="probe",
            concurrency=1,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=True,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        await mapper.scan("example.com", config)
        await mapper.close()
        assert transport.peak <= 1, (
            f"Phase 3 streaming breached concurrency=1: peak={transport.peak}"
        )


# ─────────────────────────────────────────────────────────────────────────
#  Edge cases: concurrency=0 (unbounded, must not deadlock) & rate-sem interplay
# ─────────────────────────────────────────────────────────────────────────


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_concurrency_zero_does_not_deadlock(self):
        """concurrency=0/<=0 is treated as unbounded (no Semaphore(0), which
        would deadlock). The scan must complete and fan out > 1."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["h1.example", "h2.example", "h3.example"])
        config = DomainMapperConfig(
            source="probe",
            concurrency=0,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        results = await mapper.scan("example.com", config)
        await mapper.close()
        assert len(results) > 0
        # Unbounded => the parallel fan-out should actually overlap.
        assert transport.peak > 1, (
            f"concurrency=0 should be unbounded (peak>1); got peak={transport.peak}"
        )

    @pytest.mark.asyncio
    async def test_negative_concurrency_treated_as_unbounded(self):
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["h1.example", "h2.example", "h3.example"])
        config = DomainMapperConfig(
            source="probe",
            concurrency=-1,
            hits_per_sec=0,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        await mapper.scan("example.com", config)
        await mapper.close()
        assert transport.peak > 1, (
            f"concurrency=-1 should be unbounded; got peak={transport.peak}"
        )

    @pytest.mark.asyncio
    async def test_rate_sem_composes_with_concurrency(self):
        """When both hits_per_sec and concurrency are set, in-flight must be
        <= min(concurrency, rate-window). With concurrency=1 the wrapper
        dominates regardless of hits_per_sec."""
        mapper, transport = _make_mapper()
        _patch_discover_hosts(mapper, ["h1.example", "h2.example", "h3.example"])
        config = DomainMapperConfig(
            source="probe",
            concurrency=1,
            hits_per_sec=10,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )
        await mapper.scan("example.com", config)
        await mapper.close()
        assert transport.peak <= 1, (
            f"concurrency=1 must dominate hits_per_sec=10; peak={transport.peak}"
        )


# ─────────────────────────────────────────────────────────────────────────
#  _BoundedClient / _BoundedStream unit tests (wrapper contract)
# ─────────────────────────────────────────────────────────────────────────


class TestBoundedClientWrapper:

    @pytest.mark.asyncio
    async def test_get_acquires_semaphore(self):
        transport = _CountingTransport()
        real = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        wrapped = _BoundedClient(real)
        wrapped.set_concurrency(1)

        async def fetch(i):
            return await wrapped.get(f"http://h{i}.example/")

        await asyncio.gather(*[fetch(i) for i in range(5)])
        await real.aclose()
        assert transport.peak <= 1, f"wrapper get breached: peak={transport.peak}"

    @pytest.mark.asyncio
    async def test_head_acquires_semaphore(self):
        transport = _CountingTransport()
        real = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        wrapped = _BoundedClient(real)
        wrapped.set_concurrency(2)

        async def fetch(i):
            return await wrapped.head(f"http://h{i}.example/")

        await asyncio.gather(*[fetch(i) for i in range(6)])
        await real.aclose()
        assert transport.peak <= 2, f"wrapper head breached: peak={transport.peak}"

    @pytest.mark.asyncio
    async def test_stream_acquires_semaphore(self):
        """stream() must hold the semaphore for the full body duration."""
        transport = _CountingTransport()
        real = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        wrapped = _BoundedClient(real)
        wrapped.set_concurrency(1)

        async def fetch(i):
            async with wrapped.stream("GET", f"http://h{i}.example/") as r:
                async for _ in r.aiter_bytes(4096):
                    pass

        await asyncio.gather(*[fetch(i) for i in range(4)])
        await real.aclose()
        assert transport.peak <= 1, f"wrapper stream breached: peak={transport.peak}"

    @pytest.mark.asyncio
    async def test_no_semaphore_when_unconfigured(self):
        """Before set_concurrency (or with <=0), requests pass through
        unbounded — preserving the original behaviour outside scan()."""
        transport = _CountingTransport()
        real = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        wrapped = _BoundedClient(real)
        # set_concurrency not called; default _sem is None
        assert wrapped._sem is None

        async def fetch(i):
            return await wrapped.get(f"http://h{i}.example/")

        await asyncio.gather(*[fetch(i) for i in range(8)])
        await real.aclose()
        assert transport.peak > 1, (
            f"unconfigured wrapper should be unbounded; peak={transport.peak}"
        )

    @pytest.mark.asyncio
    async def test_zero_concurrency_is_unbounded_not_deadlock(self):
        transport = _CountingTransport()
        real = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        wrapped = _BoundedClient(real)
        wrapped.set_concurrency(0)  # must NOT create Semaphore(0) (deadlock)

        async def fetch(i):
            return await wrapped.get(f"http://h{i}.example/")

        await asyncio.gather(*[fetch(i) for i in range(4)])
        await real.aclose()
        assert transport.peak > 1, (
            f"concurrency=0 should be unbounded; peak={transport.peak}"
        )

    @pytest.mark.asyncio
    async def test_stream_releases_semaphore_on_enter_error(self):
        """If the real stream's __aenter__ raises, the semaphore must be
        released so subsequent requests are not blocked forever."""
        real = httpx.AsyncClient(http2=False, timeout=15)
        wrapped = _BoundedClient(real)
        wrapped.set_concurrency(1)

        class _FailCM:
            async def __aenter__(self):
                raise RuntimeError("boom")
            async def __aexit__(self, *exc):
                return False

        cm = _BoundedStream(_FailCM(), wrapped._sem)
        with pytest.raises(RuntimeError):
            await cm.__aenter__()
        # The semaphore must have been released — a subsequent acquire
        # should not block.
        await asyncio.wait_for(wrapped._sem.acquire(), timeout=1.0)
        wrapped._sem.release()
        await real.aclose()

    @pytest.mark.asyncio
    async def test_proxy_attributes_delegated_to_real(self):
        """Non-HTTP attributes (headers, base_url, is_closed) must proxy to
        the underlying client via __getattr__."""
        real = httpx.AsyncClient(http2=False, timeout=15)
        wrapped = _BoundedClient(real)
        assert wrapped.is_closed is False
        assert wrapped.base_url == real.base_url
        assert wrapped.headers is real.headers
        await wrapped.aclose()
        assert wrapped.is_closed is True
        assert real.is_closed is True

    @pytest.mark.asyncio
    async def test_reconfigure_per_scan(self):
        """set_concurrency replaces the semaphore, so back-to-back scans with
        different concurrency values are honoured independently."""
        transport = _CountingTransport()
        real = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        wrapped = _BoundedClient(real)

        wrapped.set_concurrency(1)
        sem_a = wrapped._sem

        async def fetch(i):
            return await wrapped.get(f"http://h{i}.example/")

        await asyncio.gather(*[fetch(i) for i in range(4)])
        assert transport.peak <= 1
        transport.peak = 0

        wrapped.set_concurrency(3)
        sem_b = wrapped._sem
        assert sem_a is not sem_b, "set_concurrency must create a fresh semaphore"
        await asyncio.gather(*[fetch(i) for i in range(6)])
        assert transport.peak <= 3
        await real.aclose()
