"""Regression tests for AsyncUrlSeeder early-close task cleanup.

When ``AsyncUrlSeeder`` discovers URLs from a sitemap *index* it fans out one
``process_subsitemap`` task per sub-sitemap and streams results through a bounded
``asyncio.Queue`` back to the caller. Before the fix, the fan-out tasks were not
cancelled if the consuming async generator was closed early (e.g. by ``max_urls``
or a ``break``): ``GeneratorExit`` landed at ``yield item`` and the only cleanup
line (``await asyncio.gather(*tasks, ...)``) was after the loop and skipped.

On a full result queue the orphaned tasks suspend inside ``await
result_queue.put(...)`` and resist a single ``Task.cancel()``: the first
``CancelledError`` bypasses their ``except Exception`` (``CancelledError`` is a
``BaseException`` since 3.8) and their ``finally: await put(None)`` re-blocks on
the still-full queue. ``asyncio.run`` shutdown (``_cancel_all_tasks`` issues a
single cancel then gathers) then hangs.

The fix cancels and awaits owned tasks on close. Producers send completion
sentinels only after normal completion or caught errors, never after cancellation.
These tests pin that behaviour for both ``_iter_sitemap_content`` and the
recursive ``_iter_sitemap``, mirroring the existing
``test_memory_adaptive_stream_closure_cleans_up_tasks`` for the dispatcher.
"""

import asyncio
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from crawl4ai.async_url_seeder import AsyncUrlSeeder
from crawl4ai.async_configs import SeedingConfig


SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


# ---------------------------------------------------------------------------
# Sitemap XML builders + fake httpx client
# ---------------------------------------------------------------------------

def _index_xml(sub_urls):
    root = ET.Element("sitemapindex", xmlns=SITEMAP_NS)
    for u in sub_urls:
        sm = ET.SubElement(root, "sitemap")
        ET.SubElement(sm, "loc").text = u
    return ET.tostring(root, encoding="utf-8")


def _urlset_xml(urls):
    root = ET.Element("urlset", xmlns=SITEMAP_NS)
    for u in urls:
        e = ET.SubElement(root, "url")
        ET.SubElement(e, "loc").text = u
    return ET.tostring(root, encoding="utf-8")


class FakeResponse:
    def __init__(self, content, status_code=200, url=""):
        self.content = content
        self.status_code = status_code
        self.url = url
        self.text = (
            content.decode("utf-8", "replace")
            if isinstance(content, (bytes, bytearray)) else content
        )
        self.headers = {}

    def raise_for_status(self):
        if not 200 <= self.status_code < 400:
            import httpx
            raise httpx.HTTPStatusError("", request=None, response=self)


class FakeClient:
    """httpx.AsyncClient stand-in that serves canned sitemap XML, no network."""

    def __init__(self, sitemap_url, index_xml, urlset_map):
        self.sitemap_url = sitemap_url
        self._index = index_xml
        self._urlsets = urlset_map  # full sub-sitemap url -> bytes
        self.aclose = AsyncMock()

    async def get(self, url, **kw):
        if url == self.sitemap_url:
            return FakeResponse(self._index, url=url)
        if url in self._urlsets:
            return FakeResponse(self._urlsets[url], url=url)
        return FakeResponse(b"", 404, url=url)

    async def head(self, url, **kw):
        if url == self.sitemap_url:
            return FakeResponse(b"", 200, url=url)
        return FakeResponse(b"", 404, url=url)


def _make_fixture(domain, n_subs, urls_per_sub, tmp_path):
    """Build a seeder wired to a fake sitemap index of n_subs * urls_per_sub URLs."""
    sitemap_url = f"{domain}/sitemap.xml"
    sub_urls = [f"{domain}/sm{i}.xml" for i in range(n_subs)]
    index = _index_xml(sub_urls)
    urlsets = {
        su: _urlset_xml([f"{domain}/sm{i}/u{j}" for j in range(urls_per_sub)])
        for i, su in enumerate(sub_urls)
    }
    client = FakeClient(sitemap_url, index, urlsets)
    seeder = AsyncUrlSeeder(
        client=client,
        base_directory=tmp_path,
        cache_root=tmp_path / "cache",
    )
    return seeder, sitemap_url, index


def _sub_tasks():
    """Live (non-done) process_subsitemap tasks belonging to this seeder call."""
    return [
        t for t in asyncio.all_tasks()
        if not t.done() and t is not asyncio.current_task()
        and "process_sub" in str(t.get_coro())
    ]


async def _force_reclaim(tasks):
    """Best-effort reclaim of orphaned process_subsitemap tasks for test hygiene.

    Pre-fix orphans block on ``result_queue.put(...)`` and resist a single
    ``cancel()`` (CancelledError bypasses ``except Exception`` and their
    ``finally: put(None)`` re-blocks). Re-issuing ``cancel()`` after a tick
    delivers a fresh cancellation at the re-blocked ``put`` and reclaims them,
    so a failing (pre-fix) run does not leave tasks hung on the per-test loop.
    After the fix this is a no-op — every task is already done by ``aclose``.
    """
    for _ in range(10):
        pending = [t for t in tasks if not t.done()]
        if not pending:
            return
        for t in pending:
            t.cancel()
        await asyncio.sleep(0)
    await asyncio.gather(*tasks, return_exceptions=True)


def _capture_sub_tasks(monkeypatch):
    """Capture every process_subsitemap task object by wrapping asyncio.create_task.

    The wrapper is transparent (it calls the real factory), so framework tasks
    created via the module function are unaffected; we keep only those whose
    coroutine is a ``process_subsitemap`` closure.
    """
    captured = []
    real = asyncio.create_task

    def tracking_create_task(coro, **kw):
        t = real(coro, **kw)
        if "process_sub" in str(coro):
            captured.append(t)
        return t

    monkeypatch.setattr(asyncio, "create_task", tracking_create_task)
    return captured


# ---------------------------------------------------------------------------
# _iter_sitemap_content — the sitemap-index path used by _from_sitemaps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_iter_sitemap_content_cancels_subtasks_on_early_close(tmp_path, monkeypatch):
    n_subs, urls_per_sub = 2, 1010  # Exceeds the 2,000-entry queue after five reads.
    seeder, sitemap_url, index = _make_fixture(
        "https://example.com", n_subs, urls_per_sub, tmp_path)
    captured = _capture_sub_tasks(monkeypatch)

    gen = seeder._iter_sitemap_content(sitemap_url, index)
    try:
        got = []
        async for u in gen:
            got.append(u)
            if len(got) >= 5:
                break

        # All fan-out tasks were created before the first yield.
        assert len(captured) == n_subs
        # ...and at least some are still alive, blocked on the full result queue —
        # i.e. this is the early-close-with-orphans scenario the fix targets.
        assert not all(t.done() for t in captured), "scenario precondition: some orphans alive"

        # Closing the generator must cancel + drain + gather every sub-sitemap task.
        await gen.aclose()

        assert all(t.done() for t in captured), "early close left a sub-sitemap task alive"
        assert not _sub_tasks(), "orphaned process_subsitemap tasks survived aclose()"

        # Sanity: the consumer did receive items before closing.
        assert len(got) == 5
    finally:
        await _force_reclaim(captured)
        await _force_reclaim(_sub_tasks())
        try:
            await gen.aclose()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_iter_sitemap_content_completes_normally(tmp_path):
    n_subs, urls_per_sub = 5, 4
    seeder, sitemap_url, index = _make_fixture(
        "https://example.com", n_subs, urls_per_sub, tmp_path)

    urls = []
    async for u in seeder._iter_sitemap_content(sitemap_url, index):
        urls.append(u)

    expected = sorted(
        f"https://example.com/sm{i}/u{j}"
        for i in range(n_subs) for j in range(urls_per_sub)
    )
    assert sorted(urls) == expected
    assert not _sub_tasks()


@pytest.mark.asyncio
async def test_iter_sitemap_content_completes_under_backpressure(tmp_path, monkeypatch):
    """Normal completion must not deadlock when the result queue is tiny.

    This guards the fix's design choice (blocking ``put(None)`` sentinel, not
    ``put_nowait``): with ``put_nowait`` the sentinel is dropped when the queue
    is full and ``completed_count`` never reaches ``total_sitemaps`` (deadlock).
    """
    real_queue = asyncio.Queue

    def tiny_queue(maxsize=0, **kw):
        return real_queue(maxsize=1)

    monkeypatch.setattr(asyncio, "Queue", tiny_queue)

    n_subs, urls_per_sub = 5, 4
    seeder, sitemap_url, index = _make_fixture(
        "https://example.com", n_subs, urls_per_sub, tmp_path)

    urls = []
    async for u in seeder._iter_sitemap_content(sitemap_url, index):
        urls.append(u)

    expected = sorted(
        f"https://example.com/sm{i}/u{j}"
        for i in range(n_subs) for j in range(urls_per_sub)
    )
    assert sorted(urls) == expected, "sentinel was dropped — completion deadline broke"
    assert not _sub_tasks()


# ---------------------------------------------------------------------------
# _iter_sitemap — the recursive sitemap-index path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_iter_sitemap_recursive_cancels_subtasks_on_early_close(tmp_path, monkeypatch):
    n_subs, urls_per_sub = 2, 1010  # Exceeds the 2,000-entry queue after five reads.
    seeder, sitemap_url, _index = _make_fixture(
        "https://example.com", n_subs, urls_per_sub, tmp_path)
    captured = _capture_sub_tasks(monkeypatch)

    gen = seeder._iter_sitemap(sitemap_url)
    try:
        got = []
        async for u in gen:
            got.append(u)
            if len(got) >= 5:
                break

        assert len(captured) == n_subs
        assert not all(t.done() for t in captured), "scenario precondition: some orphans alive"

        await gen.aclose()

        assert all(t.done() for t in captured), "early close left a sub-sitemap task alive"
        assert not _sub_tasks(), "orphaned process_subsitemap tasks survived aclose()"
        assert len(got) == 5
    finally:
        await _force_reclaim(captured)
        await _force_reclaim(_sub_tasks())
        try:
            await gen.aclose()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_iter_sitemap_recursive_completes_normally(tmp_path):
    n_subs, urls_per_sub = 4, 3
    seeder, sitemap_url, _index = _make_fixture(
        "https://example.com", n_subs, urls_per_sub, tmp_path)

    urls = []
    async for u in seeder._iter_sitemap(sitemap_url):
        urls.append(u)

    expected = sorted(
        f"https://example.com/sm{i}/u{j}"
        for i in range(n_subs) for j in range(urls_per_sub)
    )
    assert sorted(urls) == expected
    assert not _sub_tasks()


# ---------------------------------------------------------------------------
# End-to-end through AsyncUrlSeeder.urls() — the real trigger (max_urls)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_urls_with_max_urls_leaves_no_orphan_tasks(tmp_path, nested):
    n_subs, urls_per_sub = 2, 1010  # Exceeds the 2,000-entry queue after five reads.
    seeder, sitemap_url, _index = _make_fixture(
        "https://example.com", n_subs, urls_per_sub, tmp_path)
    if nested:
        nested_url = "https://example.com/nested.xml"
        seeder.client._urlsets[nested_url] = seeder.client._index
        seeder.client._index = _index_xml([nested_url])
    config = SeedingConfig(
        source="sitemap", max_urls=5, concurrency=2, hits_per_sec=None, force=True,
    )

    try:
        results = await seeder.urls("https://example.com", config)
        assert len(results) == 5

        assert not _sub_tasks(), "process_subsitemap tasks leaked after urls() returned"
    finally:
        await _force_reclaim(_sub_tasks())


@pytest.mark.asyncio
async def test_urls_unlimited_completes_normally(tmp_path):
    """Default config (max_urls=-1) consumes the generator to completion."""
    n_subs, urls_per_sub = 3, 2
    seeder, sitemap_url, _index = _make_fixture(
        "https://example.com", n_subs, urls_per_sub, tmp_path)
    config = SeedingConfig(
        source="sitemap", max_urls=-1, concurrency=2, hits_per_sec=None, force=True,
    )

    results = await seeder.urls("https://example.com", config)
    assert len(results) == n_subs * urls_per_sub
    assert not _sub_tasks()


async def _e2e_main(domain, sitemap_url, index_xml, urlsets, tmp_path):
    """End-to-end ``asyncio.run`` body: urls() with max_urls, no manual cancel."""
    client = FakeClient(sitemap_url, index_xml, urlsets)
    seeder = AsyncUrlSeeder(
        client=client,
        base_directory=tmp_path,
        cache_root=tmp_path / "cache",
    )
    config = SeedingConfig(
        source="sitemap", max_urls=5, concurrency=2, hits_per_sec=None, force=True,
    )
    results = await seeder.urls(domain, config)
    assert len(results) == 5
    assert not _sub_tasks(), "urls() returned before sitemap workers stopped"


def test_urls_does_not_hang_on_run_shutdown(tmp_path):
    """asyncio.run must complete after a max_urls-limited urls() call.

    Before the fix, orphaned ``process_subsitemap`` tasks resisted the single
    ``cancel()`` issued by ``_cancel_all_tasks`` (their ``finally: put(None)``
    re-blocked on the full queue), so ``asyncio.run`` hung on shutdown. We run
    the whole scenario in a worker thread with a hard timeout so a regression
    fails the test instead of hanging the suite.
    """
    domain = "https://example.com"
    n_subs, urls_per_sub = 2, 1010  # Exceeds the 2,000-entry queue after five reads.
    sub_urls = [f"{domain}/sm{i}.xml" for i in range(n_subs)]
    index = _index_xml(sub_urls)
    urlsets = {
        su: _urlset_xml([f"{domain}/sm{i}/u{j}" for j in range(urls_per_sub)])
        for i, su in enumerate(sub_urls)
    }
    sitemap_url = f"{domain}/sitemap.xml"
    tmp = Path(tmp_path)

    outcome = {}

    def runner():
        try:
            asyncio.run(_e2e_main(domain, sitemap_url, index, urlsets, tmp))
            outcome["ok"] = True
        except BaseException as e:  # noqa: BLE001 — record any failure for the assert
            outcome["err"] = repr(e)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "asyncio.run hung on shutdown (orphan tasks resist single cancel)"
    assert outcome.get("ok"), f"asyncio.run body failed: {outcome.get('err')}"


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"]
    ))
