"""Cache writes bound real files without breaking shared hashes or refreshes."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from crawl4ai.async_database import AsyncDatabaseManager
from crawl4ai.models import CrawlResult
from crawl4ai.utils import ensure_content_dirs, generate_content_hash


@pytest.mark.asyncio
async def test_cache_retention_collects_orphans_without_racing_writers(tmp_path):
    async def manager():
        mgr = AsyncDatabaseManager()
        mgr.db_path = str(tmp_path / "cache.db")
        mgr.content_paths = ensure_content_dirs(str(tmp_path))
        mgr.cache_settings = SimpleNamespace(max_bytes=8192, max_age_seconds=60, prune_interval_seconds=0)
        await mgr.ainit_db()
        await mgr.update_db_schema()
        mgr._initialized = True
        return mgr

    mgr = await manager()
    peer = await manager()

    def result(url, html, screenshot="shared screenshot"):
        return CrawlResult(url=url, html=html, screenshot=screenshot, success=True)

    def content_path(kind, content):
        return Path(mgr.content_paths[kind]) / generate_content_hash(content)

    await mgr.acache_url(result("https://example.com/a", "old html"))
    await mgr.acache_url(result("https://example.com/b", "old html"))
    await mgr.acache_url(result("https://example.com/a", "refreshed html", "new screenshot"))
    assert content_path("html", "old html").read_text() == "old html"
    assert (await mgr.aget_cached_url("https://example.com/a")).html == "refreshed html"
    await mgr.acache_url(result("https://example.com/b", "replacement html"))
    assert not content_path("html", "old html").exists()
    assert content_path("screenshots", "shared screenshot").is_file()

    # Pause after file creation but before its row is published, then run a
    # second manager's real write/collection against the same database.
    created, release = asyncio.Event(), asyncio.Event()
    store = mgr._store_content

    async def paused_store(content, kind):
        digest = await store(content, kind)
        if content == "in flight html":
            created.set()
            await release.wait()
        return digest

    mgr._store_content = paused_store
    writer = asyncio.create_task(mgr.acache_url(result("https://example.com/flight", "in flight html")))
    await asyncio.wait_for(created.wait(), 5)
    collector = asyncio.create_task(peer.acache_url(result("https://example.com/peer", "peer html")))
    try:
        await asyncio.wait({collector}, timeout=0.05)
        assert content_path("html", "in flight html").read_text() == "in flight html"
    finally:
        release.set()
        await asyncio.gather(writer, collector)
    assert (await peer.aget_cached_url("https://example.com/flight")).html == "in flight html"

    async with mgr.get_connection() as db:
        await db.execute("UPDATE crawled_data SET cached_at = ? WHERE url = ?",
                         (time.time() - 120, "https://example.com/flight"))
        await db.commit()
    assert await mgr.aget_cached_url("https://example.com/flight") is None
    await mgr.acache_url(result("https://example.com/trigger", "trigger"))
    assert not content_path("html", "in flight html").exists()

    mgr.cache_settings.max_bytes = 4096
    for index in range(12):
        await mgr.acache_url(result(f"https://example.com/size/{index}", str(index) + "x" * 1024))
    files = [path for directory in set(mgr.content_paths.values()) for path in Path(directory).iterdir() if path.is_file()]
    assert sum(path.stat().st_size for path in files) <= 4096
    assert await mgr.aget_cached_url("https://example.com/size/0") is None
    assert (await mgr.aget_cached_url("https://example.com/size/11")).html == "11" + "x" * 1024
    await mgr.cleanup()
    await peer.cleanup()


@pytest.mark.asyncio
async def test_cache_census_runs_at_most_once_per_interval(tmp_path):
    """Writes inside the interval skip the census; the next one after it collects."""
    mgr = AsyncDatabaseManager()
    mgr.db_path = str(tmp_path / "cache.db")
    mgr.content_paths = ensure_content_dirs(str(tmp_path))
    mgr.cache_settings = SimpleNamespace(max_bytes=1 << 30, max_age_seconds=3600, prune_interval_seconds=1)
    await mgr.ainit_db()
    await mgr.update_db_schema()
    mgr._initialized = True
    old = Path(mgr.content_paths["html"]) / generate_content_hash("old html")

    await mgr.acache_url(CrawlResult(url="https://example.com/a", html="old html", success=True))
    await mgr.acache_url(CrawlResult(url="https://example.com/a", html="new html", success=True))
    assert old.is_file()  # Orphaned, but no census ran inside the interval.
    await asyncio.sleep(1.1)
    await mgr.acache_url(CrawlResult(url="https://example.com/b", html="other html", success=True))
    assert not old.exists()
    assert (await mgr.aget_cached_url("https://example.com/a")).html == "new html"
