"""Regression tests for ``AsyncWebCrawler.amap_domain`` single-flight serialization.

``DomainMapper._validate_hosts`` resets the per-scan ``_host_schemes`` map at the
start of every call (so a fresh ``scan()`` does not see stale schemes), and the
four scanning methods (``_fingerprint_soft_404``, ``_probe_paths``,
``_discover_feeds``, ``_scan_homepage``) read it later. That encodes a
*single-flight* contract for the cached mapper: only one ``scan()`` may use the
mapper's instance state at a time.

``AsyncWebCrawler.amap_domain`` caches one ``DomainMapper`` and reuses it for
every call. Without serialization, two overlapping ``amap_domain()`` calls share
and clobber ``_host_schemes``: scan B's reset wipes scan A's already-populated
scheme entries while A's per-host scanning tasks are still in flight, so A's
scanning methods fall back to the ``"https"`` default and silently drop the
HTTP-only host's URLs. ``amap_domain`` now serializes against an always-present
``_domain_mapper_lock``, turning the implicit single-flight contract into a
guaranteed one. These tests guard that contract against a future revert.
"""
import asyncio
import tempfile
from unittest.mock import MagicMock

import pytest

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import DomainMapperConfig


def _crawler_with_mocked_mapper(scan):
    """Build an AsyncWebCrawler shell with a pre-cached mocked DomainMapper.

    Only the attributes ``amap_domain`` touches are populated, so the real
    ``amap_domain`` code path (lock acquisition + scan dispatch) is exercised
    without launching a browser or making network requests.
    """
    crawler = AsyncWebCrawler.__new__(AsyncWebCrawler)
    crawler._domain_mapper_lock = asyncio.Lock()
    mapper = MagicMock()
    mapper.scan = scan
    crawler._domain_mapper = mapper
    return crawler


@pytest.mark.asyncio
async def test_amap_domain_never_runs_scan_concurrently():
    """Overlapping amap_domain() calls must never have two scan() calls in
    flight at once. Without the serialization lock the mock scan sees
    ``max_in_flight == 3``; with the lock it must see ``1``.
    """
    in_flight = 0
    max_in_flight = 0

    async def scan(domain, config):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return [{"url": f"http://{domain}/", "source": "test"}]

    crawler = _crawler_with_mocked_mapper(scan)

    results = await asyncio.gather(
        crawler.amap_domain("a.example"),
        crawler.amap_domain("b.example"),
        crawler.amap_domain("c.example"),
    )

    assert max_in_flight == 1, "scan() calls overlapped; mapper lock not serializing"
    assert len(results) == 3
    assert [r[0]["url"] for r in results] == [
        "http://a.example/",
        "http://b.example/",
        "http://c.example/",
    ]


@pytest.mark.asyncio
async def test_amap_domain_blocks_second_call_until_first_releases():
    """The second amap_domain() call must not enter scan() until the first has
    fully released the lock — i.e. the lock is held for the entire scan()
    duration, not just the mapper-creation step. This is the deterministic
    guard that fails if the lock is removed or narrowed to the init only.
    """
    a_entered = asyncio.Event()
    a_release = asyncio.Event()
    b_entered = asyncio.Event()

    async def scan(domain, config):
        if domain == "a.example":
            a_entered.set()
            await a_release.wait()
            return [{"url": "http://a.example/"}]
        # domain == "b.example"
        b_entered.set()
        return [{"url": "http://b.example/"}]

    crawler = _crawler_with_mocked_mapper(scan)

    task_a = asyncio.ensure_future(crawler.amap_domain("a.example"))
    await a_entered.wait()  # A is inside scan(), holding the lock

    task_b = asyncio.ensure_future(crawler.amap_domain("b.example"))
    # Give B every opportunity to run; it must remain blocked on the lock.
    for _ in range(10):
        await asyncio.sleep(0.01)
        if b_entered.is_set():
            break
    assert not b_entered.is_set(), "scan() B started while A still held the lock"

    a_release.set()
    res_a = await task_a
    res_b = await task_b

    assert b_entered.is_set(), "scan() B never started after A released the lock"
    assert len(res_a) == 1 and len(res_b) == 1


@pytest.mark.asyncio
async def test_amap_domain_lazy_creates_mapper_at_most_once(monkeypatch):
    """When the cached mapper is absent, concurrent amap_domain() calls must
    construct exactly one DomainMapper — the lock guards the lazy init too, so
    a refactor that moves the init outside the lock is caught here.
    """
    import crawl4ai.async_webcrawler as awc

    creations = {"count": 0}

    async def fake_scan(domain, config):
        return [{"url": f"http://{domain}/"}]

    def fake_domain_mapper(*args, **kwargs):
        creations["count"] += 1
        m = MagicMock()
        m.scan = fake_scan
        return m

    monkeypatch.setattr(awc, "DomainMapper", fake_domain_mapper)

    crawler = AsyncWebCrawler.__new__(AsyncWebCrawler)
    crawler._domain_mapper = None
    crawler._domain_mapper_lock = asyncio.Lock()
    crawler.logger = None
    crawler.crawl4ai_folder = tempfile.mkdtemp(prefix="c4ai_amap_")

    results = await asyncio.gather(
        crawler.amap_domain("a.example"),
        crawler.amap_domain("b.example"),
    )

    assert creations["count"] == 1, "mapper was created more than once"
    assert len(results) == 2


@pytest.mark.asyncio
async def test_amap_domain_preserves_supplied_config_without_overrides():
    """A config object reaches the scan unchanged when kwargs are empty."""
    seen = {}

    async def scan(domain, config):
        seen["config"] = config
        return []

    crawler = _crawler_with_mocked_mapper(scan)
    config = DomainMapperConfig(source="sitemap", concurrency=2)

    assert await crawler.amap_domain("example.com", config) == []
    assert seen["config"] is config
