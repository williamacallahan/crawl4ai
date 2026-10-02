"""Tests for the ``_QpsLimiter`` rate limiter and ``hits_per_sec`` enforcement.

These tests pin the bug where ``hits_per_sec`` was implemented as
``asyncio.Semaphore(hits_per_sec)`` — a *concurrency* cap, not a *rate* cap.
A counting semaphore bounds in-flight operations, so the effective QPS was
``permits / request_latency``: ~18x over the configured cap for fast (50ms)
targets and ~0.5x under cap for slow (2s) targets. The fix replaces the
semaphore with a leaky-bucket ``_QpsLimiter`` that paces request *start times*
at ``1 / rate`` seconds apart, restoring the documented "requests per second"
guarantee on the gated surfaces:

  * ``AsyncUrlSeeder.urls()`` worker (``live_check``/``extract_head``)
  * ``AsyncUrlSeeder.extract_head_for_urls()`` worker
  * ``DomainMapper._probe_paths`` and ``DomainMapper._extract_heads``

The tests use an ``httpx.AsyncBaseTransport`` that records the start time of
every ``handle_async_request`` invocation (no network) and run under
``-m "not network"``. Observed QPS is the inter-arrival rate of request
starts: ``(count - 1) / (last_start - first_start)``.
"""
import asyncio
import time

import httpx
import pytest

from crawl4ai.async_url_seeder import AsyncUrlSeeder, _QpsLimiter
from crawl4ai.async_configs import SeedingConfig
from crawl4ai.domain_mapper import DomainMapper
from crawl4ai.async_configs import DomainMapperConfig


# ─────────────────────────────────────────────────────────────────────────
#  Counting transport — records peak in-flight AND request start-times
# ─────────────────────────────────────────────────────────────────────────


class _CountingTransport(httpx.AsyncBaseTransport):
    """Flat 200 HTML response for every request.

    Records the start time (``time.monotonic()``) of each
    ``handle_async_request`` call so the QPS can be measured as the inter-
    arrival rate of request starts. Also tracks ``in_flight``/``peak`` for
    concurrency assertions. No network.
    """

    def __init__(
        self,
        latency: float = 0.05,
        status: int = 200,
        body: bytes = b"<html><head><title>x</title>"
        b'<meta name="description" content="d"></head>'
        b"<body></body></html>",
    ):
        self.latency = latency
        self.status = status
        self.body = body
        self.in_flight = 0
        self.peak = 0
        self.count = 0
        self.start_times: list[float] = []

    async def handle_async_request(self, request):
        # No await between these mutations => atomic w.r.t. the event loop.
        now = time.monotonic()
        self.in_flight += 1
        self.count += 1
        self.start_times.append(now)
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


def _observed_qps(transport: "_CountingTransport") -> float:
    """Inter-arrival rate of request starts: (count-1) / elapsed.

    For a perfect limiter at ``rate`` r, N starts are spaced ``1/r`` apart so
    the inter-arrival rate is exactly ``r``. A counting semaphore with
    instantaneous bodies finishes in ~0s, yielding ~inf QPS.
    """
    if transport.count < 2:
        return 0.0
    elapsed = transport.start_times[-1] - transport.start_times[0]
    if elapsed <= 0:
        return float("inf")
    return (transport.count - 1) / elapsed


def _patch_discover_hosts(mapper, hosts):
    """Skip real DNS/network in Phase 1; return a fixed host set."""
    fixed = set(hosts)

    async def _fake(base_domain, sources, config):
        return fixed

    mapper._discover_hosts = _fake


# ─────────────────────────────────────────────────────────────────────────
#  _QpsLimiter unit tests (no HTTP)
# ─────────────────────────────────────────────────────────────────────────


class TestQpsLimiterUnit:
    """Direct tests of the leaky-bucket primitive, no HTTP involved."""

    @pytest.mark.asyncio
    async def test_enforces_rate_on_burst(self):
        """A burst of N coroutines entering ``_QpsLimiter(rate)`` must start at
        ~rate/s, not all at once (which is what ``asyncio.Semaphore(rate)``
        would do for instantaneous bodies)."""
        rate = 20.0
        n = 40
        lim = _QpsLimiter(rate)
        starts: list[float] = []

        async def hit():
            async with lim:
                starts.append(time.monotonic())

        t0 = time.monotonic()
        await asyncio.gather(*[hit() for _ in range(n)])
        wall = time.monotonic() - t0

        observed = (n - 1) / (starts[-1] - starts[0]) if starts[-1] != starts[0] else float("inf")
        # Must NOT overshoot the configured rate.
        assert observed <= rate * 1.5, (
            f"overshoot: observed={observed:.1f} req/s, cap={rate}, "
            f"tolerance<={rate*1.5:.1f}"
        )
        # Must NOT drastically undershoot either.
        assert observed >= rate * 0.6, (
            f"undershoot: observed={observed:.1f} req/s, cap={rate}, "
            f"tolerance>={rate*0.6:.1f}"
        )
        # Strong lower bound: a Semaphore would finish ~instantly; the limiter
        # must take at least 60% of the theoretical (n-1)/rate seconds.
        assert wall >= (n - 1) / rate * 0.6, (
            f"finished too fast: wall={wall:.3f}s, expected >= {(n-1)/rate*0.6:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_no_overshoot_vs_counting_semaphore(self):
        """Characterize the bug: ``asyncio.Semaphore(rate)`` does NOT enforce a
        per-second rate — with instantaneous bodies all N admissions finish in
        ~0s. ``_QpsLimiter(rate)`` takes ~N/rate seconds. This test exists so the
        contrast is documented and any regression to a semaphore-based limiter
        fails it loudly."""
        rate = 25.0
        n = 25

        # Semaphore: instantaneous bodies => all admissions finish immediately.
        sem = asyncio.Semaphore(int(rate))
        sem_wall = {"t": None}

        async def sem_hit():
            async with sem:
                pass

        t0 = time.monotonic()
        await asyncio.gather(*[sem_hit() for _ in range(n)])
        sem_wall["t"] = time.monotonic() - t0

        # QpsLimiter: starts paced at 1/rate seconds apart.
        lim = _QpsLimiter(rate)

        async def lim_hit():
            async with lim:
                pass

        t0 = time.monotonic()
        await asyncio.gather(*[lim_hit() for _ in range(n)])
        lim_wall = time.monotonic() - t0

        # The semaphore "rate limiter" finishes instantly; the real limiter
        # takes ~(n-1)/rate seconds. The ratio must be large — that's the bug
        # in a single number.
        sem_time = sem_wall["t"]
        assert lim_wall > sem_time * 5, (
            f"limiter should be far slower than semaphore: "
            f"lim={lim_wall:.3f}s sem={sem_time:.3f}s"
        )
        # And the limiter must respect the configured rate (not overshoot).
        assert lim_wall >= (n - 1) / rate * 0.6, (
            f"limiter too fast: {lim_wall:.3f}s, expected >= {(n-1)/rate*0.6:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_idle_reset_no_burst_debt(self):
        """An idle period must reset the baseline so the first request after the
        gap starts immediately (no accumulated "burst debt" that would stall
        the limiter after being unused)."""
        rate = 10.0
        lim = _QpsLimiter(rate)

        # First admission: immediate.
        t0 = time.monotonic()
        async with lim:
            first = time.monotonic()
        assert (first - t0) < 0.05, "first admission must be immediate"

        # Idle for 0.3s — longer than 1/rate (0.1s).
        await asyncio.sleep(0.3)

        # Second admission: must also be immediate (baseline reset to now).
        t1 = time.monotonic()
        async with lim:
            second = time.monotonic()
        assert (second - t1) < 0.05, (
            f"second admission after idle must be immediate, took {second - t1:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_fractional_rate(self):
        """rate < 1 (e.g. 2 req/s) must pace starts 0.5s apart."""
        rate = 2.0
        n = 5
        lim = _QpsLimiter(rate)
        starts: list[float] = []

        async def hit():
            async with lim:
                starts.append(time.monotonic())

        await asyncio.gather(*[hit() for _ in range(n)])
        elapsed = starts[-1] - starts[0]
        observed = (n - 1) / elapsed if elapsed > 0 else float("inf")
        assert observed <= rate * 1.5, f"overshoot: {observed:.2f} > {rate*1.5:.2f}"
        assert observed >= rate * 0.6, f"undershoot: {observed:.2f} < {rate*0.6:.2f}"

    @pytest.mark.asyncio
    async def test_aexit_returns_false_no_swallow(self):
        """``__aexit__`` must not swallow exceptions raised inside the body."""
        lim = _QpsLimiter(50.0)
        with pytest.raises(ValueError, match="boom"):
            async with lim:
                raise ValueError("boom")


# ─────────────────────────────────────────────────────────────────────────
#  AsyncUrlSeeder — extract_head_for_urls enforces hits_per_sec end-to-end
# ─────────────────────────────────────────────────────────────────────────


class TestSeederQpsEnforcement:
    """The gated surface (extract_head=True) must enforce hits_per_sec as a
    true per-second rate, not a concurrency cap."""

    @pytest.mark.asyncio
    async def test_extract_head_enforces_hits_per_sec_fast_target(self, tmp_path):
        """Reproduction of the bug report's §2: hits_per_sec=25 with 50ms
        targets previously yielded ~500 req/s (Semaphore(25) / 0.05s); the fix
        yields ~25 req/s regardless of latency."""
        rate = 25
        n_urls = 50
        latency = 0.05
        transport = _CountingTransport(latency=latency)
        client = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        seeder = AsyncUrlSeeder(client=client, cache_root=str(tmp_path / "cache"))
        seeder.force = True  # bypass on-disk cache so every URL hits the network
        config = SeedingConfig(
            extract_head=True, concurrency=n_urls + 10, hits_per_sec=rate,
            verbose=False, filter_nonsense_urls=False,
        )
        urls = [f"http://h{i}.example/p{i}" for i in range(n_urls)]

        t0 = time.monotonic()
        await seeder.extract_head_for_urls(
            urls, config=config, concurrency=n_urls + 10, timeout=10
        )
        wall = time.monotonic() - t0
        await client.aclose()

        # Every URL must have issued a request (proves the rate sem actually
        # gates the extract_head_for_urls path — before the wiring fix it was
        # dead code and requests were unthrottled).
        assert transport.count == n_urls, (
            f"expected {n_urls} requests, got {transport.count}; "
            "rate sem may not be gating extract_head_for_urls"
        )
        observed = _observed_qps(transport)
        # The limiter must NOT overshoot the configured rate.
        assert observed <= rate * 1.5, (
            f"QPS overshoot: observed={observed:.1f} req/s, cap={rate}, "
            f"tolerance<={rate*1.5:.1f}"
        )
        # And must NOT drastically undershoot.
        assert observed >= rate * 0.5, (
            f"QPS undershoot: observed={observed:.1f} req/s, cap={rate}, "
            f"tolerance>={rate*0.5:.1f}"
        )
        # Wall-clock must reflect the pacing: ~2s for 50 reqs at 25/s.
        # A semaphore(25) would finish ~0.1s.
        assert wall >= (n_urls - 1) / rate * 0.6, (
            f"finished too fast: wall={wall:.3f}s, "
            f"expected >= {(n_urls-1)/rate*0.6:.3f}s (rate limiter not pacing)"
        )

    @pytest.mark.asyncio
    async def test_extract_head_rate_invariant_to_target_latency(self, tmp_path):
        """The documented guarantee is a *rate*, so QPS must stay at the
        configured value regardless of how fast or slow the target responds.
        Before the fix, fast targets overshot and slow targets undershoot."""
        rate = 20
        n_urls = 30
        config = SeedingConfig(
            extract_head=True, concurrency=n_urls + 10, hits_per_sec=rate,
            verbose=False, filter_nonsense_urls=False,
        )
        urls = [f"http://h{i}.example/p{i}" for i in range(n_urls)]

        for latency in (0.02, 0.10, 0.50):
            transport = _CountingTransport(latency=latency)
            client = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
            seeder = AsyncUrlSeeder(client=client, cache_root=str(tmp_path / f"cache_{latency}"))
            seeder.force = True
            await seeder.extract_head_for_urls(
                urls, config=config, concurrency=n_urls + 10, timeout=10
            )
            await client.aclose()

            assert transport.count == n_urls, (
                f"latency={latency}: expected {n_urls} reqs, got {transport.count}"
            )
            observed = _observed_qps(transport)
            # The QPS must stay within [0.5x, 1.5x] of the configured rate for
            # EVERY latency. Before the fix: 20/0.02 = 1000 qps at 20ms and
            # 20/0.5 = 40 qps at 500ms — both violate these bounds.
            assert observed <= rate * 1.5, (
                f"latency={latency}s: overshoot {observed:.1f} > {rate*1.5:.1f}"
            )
            assert observed >= rate * 0.5, (
                f"latency={latency}s: undershoot {observed:.1f} < {rate*0.5:.1f}"
            )

    @pytest.mark.asyncio
    async def test_rate_sem_disabled_when_hits_per_sec_zero(self, tmp_path):
        """hits_per_sec=0 must disable rate limiting entirely — requests
        proceed unthrottled (the documented 'disabling' path)."""
        transport = _CountingTransport(latency=0.01)
        client = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        seeder = AsyncUrlSeeder(client=client, cache_root=str(tmp_path / "cache"))
        seeder.force = True
        config = SeedingConfig(
            extract_head=True, concurrency=20, hits_per_sec=0,
            verbose=False, filter_nonsense_urls=False,
        )
        urls = [f"http://h{i}.example/p{i}" for i in range(20)]

        t0 = time.monotonic()
        await seeder.extract_head_for_urls(urls, config=config, concurrency=20, timeout=10)
        wall = time.monotonic() - t0
        await client.aclose()

        assert transport.count == 20
        assert seeder._rate_sem is None, (
            "hits_per_sec=0 must leave _rate_sem disabled"
        )
        # With no rate limiting and 20ms latency over 20 concurrent workers,
        # all 20 should finish in well under a second.
        assert wall < 1.0, (
            f"disabled limiter should not pace: wall={wall:.3f}s"
        )


# ─────────────────────────────────────────────────────────────────────────
#  DomainMapper — _probe_paths and _extract_heads enforce hits_per_sec
# ─────────────────────────────────────────────────────────────────────────


class TestDomainMapperQpsEnforcement:
    """DomainMapper's gated surfaces (_probe_paths on by default, _extract_heads
    when extract_head=True) must enforce hits_per_sec as a per-second rate."""

    @pytest.mark.asyncio
    async def test_probe_paths_enforce_hits_per_sec_fast_target(self, tmp_path):
        """The strongest real-world instance from the bug report: with default
        hits_per_sec=10 and 50ms targets, _probe_paths previously issued ~200
        req/s (Semaphore(10) / 0.05s). The fix yields ~10 req/s."""
        rate = 20
        latency = 0.05
        transport = _CountingTransport(latency=latency)
        client = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        mapper = DomainMapper(client=client, base_directory=str(tmp_path / "base"))
        _patch_discover_hosts(mapper, ["h1.example"])  # skip Phase 1 network
        config = DomainMapperConfig(
            source="probe",
            concurrency=50,
            hits_per_sec=rate,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )

        t0 = time.monotonic()
        await mapper.scan("example.com", config)
        wall = time.monotonic() - t0
        await mapper.close()

        # Only the rate-limited probe HEADs should have hit the transport
        # (Phase 1 host discovery is patched out, soft_404_detection is off,
        # source is "probe" only — so no robots/sitemap/feed/homepage fetches).
        n_probes = len(config.probe_paths) if config.probe_paths else 0
        # DEFAULT_PROBE_PATHS has 25 entries.
        from crawl4ai.domain_mapper import DEFAULT_PROBE_PATHS
        expected_probes = len(DEFAULT_PROBE_PATHS) + n_probes
        assert transport.count == expected_probes, (
            f"expected {expected_probes} probe HEADs, got {transport.count}"
        )
        observed = _observed_qps(transport)
        # Must NOT overshoot — Semaphore would yield ~200-400 req/s here.
        assert observed <= rate * 1.5, (
            f"QPS overshoot on _probe_paths: observed={observed:.1f} req/s, "
            f"cap={rate}, tolerance<={rate*1.5:.1f}"
        )
        assert observed >= rate * 0.5, (
            f"QPS undershoot on _probe_paths: observed={observed:.1f} req/s, "
            f"cap={rate}, tolerance>={rate*0.5:.1f}"
        )
        # Wall-clock must reflect pacing: ~24/20 = 1.2s for 25 probes at 20/s.
        assert wall >= (expected_probes - 1) / rate * 0.6, (
            f"finished too fast: wall={wall:.3f}s, expected "
            f">= {(expected_probes-1)/rate*0.6:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_rate_composes_with_concurrency_cap(self, tmp_path):
        """hits_per_sec (rate) and concurrency (in-flight) must compose:
        rate bounds QPS and concurrency bounds in-flight, independently. With
        concurrency=2 the in-flight must never exceed 2 even though
        hits_per_sec=20 would allow far more starts; and the rate must still
        be honoured."""
        rate = 20
        transport = _CountingTransport(latency=0.05)
        client = httpx.AsyncClient(http2=False, transport=transport, timeout=15)
        mapper = DomainMapper(client=client, base_directory=str(tmp_path / "base"))
        _patch_discover_hosts(mapper, ["h1.example"])
        config = DomainMapperConfig(
            source="probe",
            concurrency=2,
            hits_per_sec=rate,
            include_subdomains=False,
            extract_head=False,
            soft_404_detection=False,
            filter_nonsense_urls=False,
            verbose=False,
            force=True,
        )

        await mapper.scan("example.com", config)
        await mapper.close()

        # The _BoundedClient concurrency cap must bound in-flight to <= 2,
        # even though _QpsLimiter doesn't bound in-flight itself.
        assert transport.peak <= 2, (
            f"concurrency=2 breached: peak={transport.peak}"
        )
        # And the rate limiter must still keep QPS near the configured rate.
        observed = _observed_qps(transport)
        assert observed <= rate * 1.5, (
            f"rate not honoured under concurrency=2: {observed:.1f} > {rate*1.5:.1f}"
        )
