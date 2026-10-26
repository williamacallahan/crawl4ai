"""
Regression tests for RobotsParser SQLite cache TTL refresh behavior.

Bug history: _cache_rules previously skipped ALL DB writes when the fetched
robots.txt content hash matched the cached hash. That left fetch_time frozen,
so once the TTL elapsed the entry stayed stale forever (until robots.txt
changed or the cache was manually cleared), and every can_fetch call paid a
full network round-trip for an unchanged robots.txt.

The fix: on an unchanged hash, _cache_rules now runs
`UPDATE robots_cache SET fetch_time = ? WHERE domain = ?` to re-arm the TTL
window, while still avoiding rewriting identical content.

These tests drive RobotsParser directly against a local aiohttp server with a
robots.txt request counter. No browser, no real network, no mocking of
RobotsParser internals. They are intentionally not marked `browser` or
`network` so they run under the CI Unit gate's `-m "not network and not browser"`.
"""

import socket
import sqlite3
import time

import pytest
from aiohttp import web

from crawl4ai.utils import RobotsParser

_ROBOTS_ALLOW = "User-agent: *\nAllow: /\n"
_ROBOTS_BLOCK_PRIVATE = "User-agent: *\nDisallow: /private/\n"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _row(parser: RobotsParser, domain: str):
    """Return the cached row (rules, fetch_time, hash) or None."""
    with sqlite3.connect(parser.db_path) as conn:
        return conn.execute(
            "SELECT rules, fetch_time, hash FROM robots_cache WHERE domain = ?",
            (domain,),
        ).fetchone()


def _backdate(parser: RobotsParser, domain: str, seconds_ago: int = 7200) -> None:
    """Force a cache row to look expired by backdating its fetch_time."""
    with sqlite3.connect(parser.db_path) as conn:
        conn.execute(
            "UPDATE robots_cache SET fetch_time = ? WHERE domain = ?",
            (int(time.time()) - seconds_ago, domain),
        )


async def _start_counted_server(port: int, robots_body: str, counter: list):
    """Start an aiohttp server that serves a fixed robots.txt and counts
    how many times /robots.txt is requested."""
    app = web.Application()

    async def robots(_req):
        counter[0] += 1
        return web.Response(text=robots_body, content_type="text/plain")

    async def index(_req):
        return web.Response(
            text="<html><body><p>page</p></body></html>",
            content_type="text/html",
        )

    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", index)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


# ---------------------------------------------------------------------------
# Core fix: refresh fetch_time on unchanged-content refetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unchanged_robots_refetch_refreshes_fetch_time(tmp_path):
    """Core fix: a successful refetch of unchanged robots.txt must advance
    fetch_time so the TTL window is re-armed. Before the fix fetch_time stayed
    frozen and every subsequent can_fetch hit the network."""
    port = _free_port()
    counter = [0]
    runner = await _start_counted_server(port, _ROBOTS_ALLOW, counter)
    try:
        parser = RobotsParser(cache_dir=str(tmp_path), cache_ttl=3600)
        base = f"http://127.0.0.1:{port}"
        domain = f"127.0.0.1:{port}"

        # 1) initial fetch
        await parser.can_fetch(f"{base}/a", "bot")
        assert counter[0] == 1
        _, t1, h1 = _row(parser, domain)

        # 2) cache hit within TTL — no network
        await parser.can_fetch(f"{base}/b", "bot")
        assert counter[0] == 1

        # 3) force expiry by backdating fetch_time
        _backdate(parser, domain)
        backdated_t = _row(parser, domain)[1]
        assert backdated_t < t1  # confirm the row is now expired

        # 4) refetch of unchanged robots.txt
        await parser.can_fetch(f"{base}/c", "bot")
        assert counter[0] == 2  # legitimate refetch
        _, t2, h2 = _row(parser, domain)

        # fetch_time must have been refreshed from the backdated value back to
        # ~now. fetch_time is stored as int(time.time()), so compare against the
        # backdated value (hours in the past) rather than t1 (same second).
        # Before the fix this row stayed at backdated_t and the cache was
        # permanently stale.
        assert (
            t2 > backdated_t + 3600
        ), f"fetch_time not refreshed: backdated={backdated_t}, now={t2}"
        # hash unchanged (content is identical)
        assert h2 == h1, "hash changed for identical content"

        # 5) the next call within the new TTL must NOT refetch
        await parser.can_fetch(f"{base}/d", "bot")
        assert (
            counter[0] == 2
        ), f"TTL not re-armed: extra robots.txt fetch ({counter[0]})"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_repeated_calls_after_expiry_refetch_only_once(tmp_path):
    """After one TTL expiry + refetch of unchanged content, every subsequent
    can_fetch within the new TTL must hit the cache (1 network fetch per TTL
    period). This is the fast-path the cache documents and the regression the
    bug report's evidence table pins (calls /page4, /page5, /page6 must NOT
    each refetch)."""
    port = _free_port()
    counter = [0]
    runner = await _start_counted_server(port, _ROBOTS_ALLOW, counter)
    try:
        parser = RobotsParser(cache_dir=str(tmp_path), cache_ttl=3600)
        base = f"http://127.0.0.1:{port}"
        domain = f"127.0.0.1:{port}"

        await parser.can_fetch(f"{base}/page1", "bot")
        assert counter[0] == 1

        _backdate(parser, domain)

        # one refetch re-arms the TTL
        await parser.can_fetch(f"{base}/page3", "bot")
        assert counter[0] == 2

        # all subsequent calls stay in cache
        for path in ["page4", "page5", "page6"]:
            await parser.can_fetch(f"{base}/{path}", "bot")
            assert counter[0] == 2, f"extra fetch at {path}: {counter[0]}"
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# No regression: changed content still does a full INSERT OR REPLACE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_changed_robots_content_full_rewrite(tmp_path):
    """Regression: when the robots.txt content hash changes, _cache_rules must
    still do a full INSERT OR REPLACE (rules + fetch_time + hash), not just an
    UPDATE fetch_time. Also, the newly-fetched rules must be enforced."""
    port = _free_port()
    state = {"body": _ROBOTS_ALLOW}
    counter = [0]
    app = web.Application()

    async def robots(_req):
        counter[0] += 1
        return web.Response(text=state["body"], content_type="text/plain")

    async def index(_req):
        return web.Response(text="<html></html>", content_type="text/html")

    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", index)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        parser = RobotsParser(cache_dir=str(tmp_path), cache_ttl=3600)
        base = f"http://127.0.0.1:{port}"
        domain = f"127.0.0.1:{port}"

        await parser.can_fetch(f"{base}/a", "bot")
        assert counter[0] == 1
        r1, t1, h1 = _row(parser, domain)

        _backdate(parser, domain)
        backdated_t = _row(parser, domain)[1]

        # change the served robots content
        state["body"] = _ROBOTS_BLOCK_PRIVATE
        await parser.can_fetch(f"{base}/b", "bot")
        assert counter[0] == 2
        r2, t2, h2 = _row(parser, domain)

        assert r2 != r1, "rules content not updated after change"
        assert h2 != h1, "hash not updated after change"
        # fetch_time refreshed from the backdated value back to ~now
        assert (
            t2 > backdated_t + 3600
        ), f"fetch_time not rewritten: backdated={backdated_t}, now={t2}"

        # the new rules are enforced from the cache (no extra fetch)
        before = counter[0]
        assert await parser.can_fetch(f"{base}/public", "bot") is True
        assert await parser.can_fetch(f"{base}/private/secret", "bot") is False
        assert counter[0] == before, "enforcement should not refetch robots.txt"
    finally:
        await runner.cleanup()
