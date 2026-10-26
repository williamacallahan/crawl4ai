"""Behavioral tests for the per-Swarm-node archive.org budget.

Tests run a real Redis (REDIS_URL, else a temporary redis-server — the same
pattern as test_crawl_job_queue_redis.py) and assert on observable results:
waits, exception contracts, prefetched crawl input, and HTTP surfaces.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types

import pytest
import pytest_asyncio
import redis
from redis import asyncio as aioredis

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import archive_budget
from archive_budget import ArchiveBudgetSpent
from fastapi import HTTPException

# crawler_pool calls the glibc-only mallopt() at import (absent on macOS);
# stub it only when the real module cannot import.
try:
    import crawler_pool  # noqa: F401
except Exception:
    _pool_stub = types.ModuleType("crawler_pool")

    async def _pool_noop(*args, **kwargs):
        return None

    for _name in (
        "close_all", "get_crawler", "get_dedicated_crawler", "get_pool_snapshot",
        "janitor", "release_crawler", "release_dedicated_crawler",
        "retire_pool_crawlers",
    ):
        setattr(_pool_stub, _name, _pool_noop)
    _pool_stub.LAST_JANITOR_PASS = 0.0
    sys.modules["crawler_pool"] = _pool_stub


def _wait_ready(client):
    for _attempt in range(100):
        try:
            if client.ping():
                return True
        except redis.ConnectionError:
            time.sleep(0.01)
    return False


@pytest.fixture
def redis_url():
    configured_url = os.getenv("REDIS_URL")
    if configured_url:
        client = redis.from_url(configured_url)
        try:
            if _wait_ready(client):
                yield configured_url
                return
            pytest.fail("configured Redis did not become ready")
        finally:
            client.close()
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is not installed")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    with tempfile.TemporaryDirectory(prefix="archive-budget-redis-") as data_dir:
        process = subprocess.Popen(
            [executable, "--bind", "127.0.0.1", "--port", str(port),
             "--save", "", "--appendonly", "no", "--dir", data_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        url = f"redis://127.0.0.1:{port}/0"
        client = redis.from_url(url)
        try:
            if not _wait_ready(client):
                pytest.fail("temporary redis-server did not become ready")
            yield url
        finally:
            client.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest_asyncio.fixture
async def redis_client(redis_url):
    client = aioredis.from_url(redis_url)
    try:
        yield client
    finally:
        await client.aclose()


def budget_config(**overrides):
    """Fast budget timings; production values live in config.yml."""
    archive = {
        "spacing_seconds": 0.25,
        "max_wait_seconds": 1.0,
        "throttle_pause_seconds": 6.0,
        "max_redirect_hops": 3,
    }
    archive.update(overrides)
    return {
        "archive": archive,
        "limits": {"wall_clock_s": 0, "max_body_bytes": 4096},
        "crawler": {
            "base_config": {},
            "browser": {"kwargs": {}, "extra_args": []},
            "memory_threshold_percent": 80,
            "recovery_threshold_percent": 65,
            "rate_limiter": {"enabled": False, "base_delay": [0, 0]},
            "pool": {"max_pages": 4},
        },
    }


# ───────────────────────── reserve / spacing ─────────────────────────
@pytest.mark.asyncio
async def test_reserve_refuses_past_max_wait(redis_client, monkeypatch):
    node = "refuse-node"
    monkeypatch.setenv("CRAWL4AI_NODE", node)
    cfg = budget_config(spacing_seconds=5.0, max_wait_seconds=0.5)
    assert await archive_budget.reserve(redis_client, cfg) < 0.1
    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.reserve(redis_client, cfg)
    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.detail == "archive.org budget for this node is spent"
    assert exc.node == node
    assert 4 <= exc.retry_after <= 5
    assert exc.headers["Retry-After"] == str(exc.retry_after)


@pytest.mark.asyncio
async def test_missing_node_fails_loud_503(redis_client, monkeypatch):
    monkeypatch.delenv("CRAWL4AI_NODE", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        await archive_budget.reserve(redis_client, budget_config())
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "archive node identity is not configured"


def test_is_archive_url_hosts_and_subdomains():
    assert archive_budget.is_archive_url("https://archive.org/details/x")
    assert archive_budget.is_archive_url("https://web.archive.org/web/2020id_/https://a.b/")
    assert archive_budget.is_archive_url("https://wayback.archive.org/x")
    assert archive_budget.is_archive_url("https://WEB.ARCHIVE.ORG/web/x")
    assert not archive_budget.is_archive_url("https://example.com/")
    assert not archive_budget.is_archive_url("https://archive.org.evil.com/")
    assert not archive_budget.is_archive_url("https://notarchive.org/")
    assert not archive_budget.is_archive_url("https://archive.org@evil.com/")
    assert not archive_budget.is_archive_url("not a url")


# ───────────────────────── fetch ─────────────────────────
ARCHIVE_URL = "https://web.archive.org/web/2020id_/https://example.com/"


@pytest.mark.asyncio
async def test_fetch_throttle_pauses_and_refuses(redis_client, monkeypatch):
    """Archive 429/503 and a refused connection pause the node, then raise."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-throttle")
    cfg = budget_config()

    async def throttled(url, max_body):
        return 429, {"Content-Type": "text/plain"}, b""

    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=throttled)
    assert exc_info.value.retry_after >= 4
    # The pause is real: the node's next start is the throttle window out,
    # and a second pause never moves the stored start earlier.
    with pytest.raises(ArchiveBudgetSpent):
        await archive_budget.reserve(redis_client, cfg)
    assert await archive_budget.pause(redis_client, cfg) > 4

    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-refused")

    async def refused(url, max_body):
        raise archive_budget._ArchiveUpstreamRefused("Connection refused")

    with pytest.raises(ArchiveBudgetSpent) as exc_info:
        await archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=refused)
    assert exc_info.value.retry_after >= 4


@pytest.mark.asyncio
async def test_fetch_follows_archive_redirects(redis_client, monkeypatch):
    """Each hop is its own reservation; relative Locations stay on archive."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-redirect")
    calls = []

    async def opener(url, max_body):
        calls.append(url)
        if len(calls) == 1:
            return 302, {"Location": "/web/2021id_/https://example.com/"}, b""
        return 200, {"Content-Type": "text/plain"}, b"redirected-body"

    out = await archive_budget.fetch(redis_client, budget_config(), ARCHIVE_URL, opener=opener)
    assert calls == [
        ARCHIVE_URL,
        "https://web.archive.org/web/2021id_/https://example.com/",
    ]
    assert out == {
        "url": calls[-1],
        "status": 200,
        "contentType": "text/plain",
        "body": "redirected-body",
    }


@pytest.mark.asyncio
async def test_fetch_redirect_off_archive_is_502(redis_client, monkeypatch):
    """A Location that leaves the archive hosts is never followed."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-off-archive")
    calls = []

    async def opener(url, max_body):
        calls.append(url)
        return 302, {"Location": "https://example.com/escape"}, b""

    with pytest.raises(HTTPException) as exc_info:
        await archive_budget.fetch(redis_client, budget_config(), ARCHIVE_URL, opener=opener)
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "archive redirect left archive.org"
    assert calls == [ARCHIVE_URL]  # the foreign target was never fetched


@pytest.mark.asyncio
async def test_fetch_redirect_hop_limit_is_502(redis_client, monkeypatch):
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-hop-limit")
    cfg = budget_config()
    calls = []

    async def opener(url, max_body):
        calls.append(url)
        return 302, {"Location": "https://web.archive.org/web/next/"}, b""

    with pytest.raises(HTTPException) as exc_info:
        await archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=opener)
    assert exc_info.value.status_code == 502
    assert len(calls) == cfg["archive"]["max_redirect_hops"]


@pytest.mark.asyncio
async def test_fetch_body_cap_is_502(redis_client, monkeypatch):
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-cap")
    cfg = budget_config()
    cap = cfg["limits"]["max_body_bytes"]

    async def opener(url, max_body):
        assert max_body == cap
        return 200, {"Content-Type": "text/html"}, b"x" * (cap + 1)

    with pytest.raises(HTTPException) as exc_info:
        await archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=opener)
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "archive body exceeds max_body_bytes"


@pytest.mark.asyncio
async def test_fetch_text_gate(redis_client, monkeypatch):
    """Binary bodies answer 415; text/*, JSON, and XML families pass."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-text")
    cfg = budget_config()

    async def binary(url, max_body):
        return 200, {"Content-Type": "application/octet-stream"}, b"\x00\x01\x02"

    with pytest.raises(HTTPException) as exc_info:
        await archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=binary)
    assert exc_info.value.status_code == 415
    assert exc_info.value.detail == "archive body is not text"

    for content_type in (
        "text/csv", "application/json",
        "application/ld+json; charset=utf-8", "application/atom+xml",
    ):
        async def ok(url, max_body, ct=content_type):
            return 200, {"Content-Type": ct}, b"body"

        out = await archive_budget.fetch(redis_client, cfg, ARCHIVE_URL, opener=ok)
        assert out["body"] == "body"


# ───────────────────────── charset decode ─────────────────────────
# Regression coverage for the bug where ``fetch`` unconditionally decoded
# every archive body as UTF-8 with ``errors="replace"``, replacing each
# non-ASCII byte of a non-UTF-8 snapshot (Latin-1/Shift-JIS/GBK — the norm on
# older archive.org replays) with U+FFFD before the browser or API client
# ever saw it. The right order is HTTP charset > <meta charset> > strict
# UTF-8 > chardet > UTF-8-with-replace, mirroring AsyncHTTPCrawlerStrategy.

def test_lookup_encoding_normalizes_and_validates():
    assert archive_budget._lookup_encoding("ISO-8859-1") == "iso-8859-1"
    assert archive_budget._lookup_encoding(' "utf-8" ') == "utf-8"
    assert archive_budget._lookup_encoding("") is None
    assert archive_budget._lookup_encoding(None) is None
    # Unknown codec names are rejected (decode would otherwise raise LookupError).
    assert archive_budget._lookup_encoding("x-not-a-real-codec") is None


def test_content_type_charset_extracts_header_charset():
    assert archive_budget._content_type_charset(
        "text/html; charset=iso-8859-1") == "iso-8859-1"
    # Case-insensitive name, surrounding whitespace, single/double quotes.
    assert archive_budget._content_type_charset(
        'text/html; CHARSET = "Shift_JIS"') == "shift_jis"
    assert archive_budget._content_type_charset(
        "text/html; charset=utf-8; boundary=foo") == "utf-8"
    # Missing or empty charset parameter -> None.
    assert archive_budget._content_type_charset("text/html") is None
    assert archive_budget._content_type_charset("") is None
    assert archive_budget._content_type_charset(None) is None
    assert archive_budget._content_type_charset("text/html; charset=") is None
    # Unknown codec -> None so a later fallback can run instead of crashing.
    assert archive_budget._content_type_charset(
        "text/html; charset=bogus-xyz") is None


def test_meta_charset_sniffs_html5_and_html4_tags():
    html5 = b'<html><head><meta charset="iso-8859-1"></head>'
    assert archive_budget._meta_charset(html5) == "iso-8859-1"
    html4 = (b'<html><head><meta http-equiv="content-type" '
             b'content="text/html; charset=Shift_JIS"></head>')
    assert archive_budget._meta_charset(html4) == "shift_jis"
    # Unquoted attribute value still matches.
    assert archive_budget._meta_charset(b'<meta charset=utf-8>') == "utf-8"
    # No meta charset -> None.
    assert archive_budget._meta_charset(b'<html><head><title>x</title></head>') is None
    assert archive_budget._meta_charset(b'') is None
    # Unknown codec in meta -> None (fall through to the next signal).
    assert archive_budget._meta_charset(b'<meta charset="x-not-real">') is None
    # Only the <head> region matters: a charset buried past 4 KB is ignored,
    # which is fine because the <head> is always near the top of a real page.
    assert archive_budget._meta_charset(b'<html>' * 2049 + b'<meta charset="utf-8">') is None


@pytest.mark.asyncio
async def test_decode_body_precedence_and_fallbacks():
    """Header > meta > strict-UTF-8 > chardet > UTF-8-with-replace."""
    # 1. HTTP header charset wins over any other signal.
    body = b'<meta charset="utf-8">caf\xe9'
    assert (await archive_budget._decode_body("text/html; charset=iso-8859-1", body)) \
        == "<meta charset=\"utf-8\">café"
    # 2. <meta charset> wins when the header has no charset.
    body = b'<meta charset="iso-8859-1">caf\xe9 na\xefve'
    assert (await archive_budget._decode_body("text/html", body)) \
        == "<meta charset=\"iso-8859-1\">café naïve"
    # 3. Valid UTF-8 with no declared charset decodes as UTF-8 (not misread
    #    by chardet as Windows-1252 — the regression the strict-UTF-8 step
    #    guards against for short multibyte sequences).
    assert (await archive_budget._decode_body("text/html", "café ✓".encode("utf-8"))) \
        == "café ✓"
    # 4. Non-UTF-8 bytes with no declared charset fall through to chardet,
    #    which detects the Latin-1 family for the live elmundo.es-style body.
    body = b"<html><body>tecnolog\xeda Catalu\xf1a a\xf1os</body></html>"
    assert (await archive_budget._decode_body("text/html", body)) \
        == "<html><body>tecnología Cataluña años</body></html>"
    # 5. Unknown header charset does not crash: lookup rejects it and the
    #    next signal decides.
    assert (await archive_budget._decode_body("text/html; charset=bogus-xyz", b"caf\xe9")) \
        == "café"


@pytest.mark.asyncio
async def test_fetch_decodes_header_charset_non_utf8(redis_client, monkeypatch):
    """A body served as text/html; charset=iso-8859-1 decodes with that
    charset instead of every non-ASCII byte becoming U+FFFD."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-charset-header")
    body = b"<html><body><p>caf\xe9 na\xefve</p></body></html>"

    async def opener(url, max_body):
        return 200, {"Content-Type": "text/html; charset=iso-8859-1"}, body

    out = await archive_budget.fetch(redis_client, budget_config(), ARCHIVE_URL,
                                     opener=opener)
    assert out["contentType"] == "text/html; charset=iso-8859-1"
    assert out["body"] == "<html><body><p>café naïve</p></body></html>"
    # The old behavior left a trail of replacement characters here.
    assert "\ufffd" not in out["body"]


@pytest.mark.asyncio
async def test_fetch_meta_charset_fallback_latin1(redis_client, monkeypatch):
    """No header charset, but the page declares iso-8859-1 in a meta tag —
    a common pattern for older snapshots the header-only fix would miss."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-charset-meta")
    body = (b'<html><head><meta charset="iso-8859-1">'
            b'</head><body><p>caf\xe9 na\xefve</p></body></html>')

    async def opener(url, max_body):
        return 200, {"Content-Type": "text/html"}, body

    out = await archive_budget.fetch(redis_client, budget_config(), ARCHIVE_URL,
                                     opener=opener)
    assert out["body"] == \
        "<html><head><meta charset=\"iso-8859-1\"></head><body><p>café naïve</p></body></html>"
    assert "\ufffd" not in out["body"]


@pytest.mark.asyncio
async def test_fetch_chardet_fallback_latin1_no_signals(redis_client, monkeypatch):
    """The live elmundo.es scenario: a header with no charset parameter and
    a body of Latin-1 bytes that are not valid UTF-8. The chardet fallback
    detects the Latin-1 family and decodes the snapshot's real text."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-charset-chardet")
    body = b"<html><body>tecnolog\xeda Catalu\xf1a a\xf1os</body></html>"

    async def opener(url, max_body):
        return 200, {"Content-Type": "text/html"}, body

    out = await archive_budget.fetch(redis_client, budget_config(), ARCHIVE_URL,
                                     opener=opener)
    assert out["body"] == "<html><body>tecnología Cataluña años</body></html>"
    assert "\ufffd" not in out["body"]


@pytest.mark.asyncio
async def test_fetch_unknown_header_charset_does_not_crash(redis_client, monkeypatch):
    """A bogus header charset falls through to the next signal, never 502."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-charset-unknown")
    body = b"caf\xe9"

    async def opener(url, max_body):
        return 200, {"Content-Type": "text/html; charset=not-a-real-codec"}, body

    out = await archive_budget.fetch(redis_client, budget_config(), ARCHIVE_URL,
                                     opener=opener)
    assert out["body"] == "café"


@pytest.mark.asyncio
async def test_fetch_utf8_body_without_charset_is_unchanged(redis_client, monkeypatch):
    """Regression: a valid UTF-8 body with no declared charset still decodes
    as UTF-8 — the strict-UTF-8 step, not chardet, decides it (so a short
    multibyte sequence is not misread as Windows-1252)."""
    monkeypatch.setenv("CRAWL4AI_NODE", "fetch-charset-utf8")
    body = "<html><body>café naïve ✓</body></html>".encode("utf-8")

    async def opener(url, max_body):
        return 200, {"Content-Type": "text/html"}, body

    out = await archive_budget.fetch(redis_client, budget_config(), ARCHIVE_URL,
                                     opener=opener)
    assert out["body"] == "<html><body>café naïve ✓</body></html>"


@pytest.mark.asyncio
async def test_prefetch_carries_correctly_decoded_body(redis_client, monkeypatch):
    """prefetch_archive feeds the renderer via ``raw:`` + the prefetched body
    string, so a non-UTF-8 snapshot must be decoded before it reaches the
    browser — otherwise the mojibake is baked in unrecoverably."""
    from archive_transport import prefetch_archive
    monkeypatch.setenv("CRAWL4AI_NODE", "prefetch-charset")

    async def opener(url, max_body):
        return 200, {"Content-Type": "text/html; charset=iso-8859-1"}, \
            b"<html><body>caf\xe9 na\xefve</body></html>"

    monkeypatch.setattr(archive_budget, "_default_opener", opener)
    out = await prefetch_archive(redis_client, budget_config(), [ARCHIVE_URL])
    assert out[ARCHIVE_URL]["body"] == "<html><body>café naïve</body></html>"
    assert "\ufffd" not in out[ARCHIVE_URL]["body"]
    assert out[ARCHIVE_URL]["responseHeaders"]["Content-Type"] == \
        "text/html; charset=iso-8859-1"


def test_archive_fetch_route_decodes_non_utf8_body(
    app_client, monkeypatch, offline_dns,
):
    """POST /archive/fetch returns the decoded (not U+FFFD-replaced) body to
    API clients in its JSONResponse — the corruption was previously shipped
    straight through this route."""
    monkeypatch.setenv("CRAWL4AI_NODE", "route-charset")

    async def opener(url, max_body):
        return 200, {"Content-Type": "text/html; charset=iso-8859-1"}, \
            b"<html><body>caf\xe9 na\xefve</body></html>"

    monkeypatch.setattr(archive_budget, "_default_opener", opener)
    r = app_client.post("/archive/fetch", json={"url": ARCHIVE_URL},
                        headers=_bearer())
    assert r.status_code == 200
    body = r.json()
    assert body["contentType"] == "text/html; charset=iso-8859-1"
    assert body["body"] == "<html><body>café naïve</body></html>"
    assert "\ufffd" not in body["body"]


# ───────────────────────── prefetch ─────────────────────────
@pytest.mark.asyncio
async def test_prefetch_preserves_archive_responses(redis_client, monkeypatch):
    """Prefetch only archive seeds, preserving their status, body and origin."""
    from archive_transport import prefetch_archive
    monkeypatch.setenv("CRAWL4AI_NODE", "prefetch-node")
    fetched = []

    async def opener(url, max_body):
        fetched.append(url)
        if url.endswith("/missing"):
            return 404, {"Content-Type": "text/html"}, b"<html>not found</html>"
        return 200, {"Content-Type": "text/html"}, f"<html>{url}</html>".encode()

    monkeypatch.setattr(archive_budget, "_default_opener", opener)
    urls = [
        "https://archive.org/details/a",
        "https://example.com/",
        "https://archive.org/details/missing",
    ]
    out = await prefetch_archive(redis_client, budget_config(), urls)
    assert fetched == [urls[0], urls[2]]
    assert list(out) == [urls[0], urls[2]]
    assert out[urls[0]]["body"] == f"<html>{urls[0]}</html>"
    assert out[urls[2]]["status"] == 404
    assert out[urls[2]]["body"] == "<html>not found</html>"
    assert out[urls[2]]["url"] == urls[2]


# ─────────────────── /crawl handler and the job worker ───────────────────
@pytest.mark.asyncio
async def test_crawl_fetches_archive_seeds_sequentially(
    redis_client, monkeypatch, offline_dns,
):
    """A 3-archive-URL /crawl fetches serially at archive.spacing_seconds
    apart and each result carries the caller's original URL."""
    import api
    import crawler_pool

    monkeypatch.setenv("CRAWL4AI_NODE", "crawl-seq-node")
    cfg = budget_config()
    fetch_starts = []

    async def opener(url, max_body):
        fetch_starts.append(time.monotonic())
        return 200, {"Content-Type": "text/html"}, b"<html>Identical archived content</html>"

    monkeypatch.setattr(archive_budget, "_default_opener", opener)
    crawled_urls = []

    class _FakeCrawler:
        async def arun_many(self, urls, config=None, dispatcher=None):
            crawled_urls.extend(urls)
            # Dispatcher results arrive in completion order, not input order.
            return [
                {"url": u, "success": True, "html": u, "error_message": None}
                for u in reversed(list(urls))
            ]

    async def fake_get_crawler(*args, **kwargs):
        return _FakeCrawler()

    monkeypatch.setattr(crawler_pool, "get_crawler", fake_get_crawler)
    monkeypatch.setattr(crawler_pool, "get_dedicated_crawler", fake_get_crawler)

    urls = [
        "https://archive.org/details/a",
        "https://web.archive.org/web/2020id_/https://b.example/",
        "https://archive.org/details/c",
    ]
    response = await api.handle_crawl_request(
        urls=list(urls), browser_config={}, crawler_config={},
        config=cfg, redis=redis_client,
    )

    assert response["success"] is True
    # Each result carries the caller's original URL even though the results
    # came back out of order, including when every body is identical.
    assert [r["url"] for r in response["results"]] == list(reversed(urls))
    # Dispatch preserves identity; the strategy owns offline HTML rendering.
    assert crawled_urls == urls

    # Three fetches, one reservation each, serialized and spaced.
    assert len(fetch_starts) == 3
    spacing = cfg["archive"]["spacing_seconds"]
    for earlier, later in zip(fetch_starts, fetch_starts[1:]):
        assert later - earlier >= spacing - 0.05, (
            f"archive fetches not spaced: {fetch_starts}"
        )


@pytest.mark.asyncio
async def test_invalid_browser_config_reserves_nothing(
    redis_client, monkeypatch, offline_dns,
):
    """A rejected request must not consume the node's archive budget."""
    import api

    monkeypatch.setenv("CRAWL4AI_NODE", "no-reserve-node")
    cfg = budget_config()
    with pytest.raises(HTTPException) as exc_info:
        await api.handle_crawl_request(
            urls=[ARCHIVE_URL],
            # proxy is a forbidden power-field under untrusted provenance —
            # BrowserConfig.load fails before any reservation can happen.
            browser_config={"proxy": "http://evil.example:8080"},
            crawler_config={}, config=cfg, redis=redis_client,
        )
    assert exc_info.value.status_code == 400
    # The budget key was never touched — a valid request still gets a free slot.
    key = archive_budget.BUDGET_KEY_PREFIX + "no-reserve-node"
    assert await redis_client.get(key) is None
    assert await archive_budget.reserve(redis_client, cfg) < 0.1


@pytest.mark.asyncio
async def test_job_worker_marks_budget_spent_terminal(monkeypatch):
    """429 from the worker's crawl is terminal — never retried — and the stored
    error carries the archive_budget_spent code + retry seconds."""
    import api
    import crawl_job_worker

    async def spent(**kwargs):
        raise ArchiveBudgetSpent(42.4, "worker-node")

    monkeypatch.setattr(api, "handle_crawl_request", spent)
    calls = {}

    async def complete(entry, payload, attempt, result=None, error=None):
        calls["complete"] = {"result": result, "error": error}
        return ("failed" if error else "completed"), result, error

    async def mark_retry(*args, **kwargs):
        calls["mark_retry"] = True

    queue = types.SimpleNamespace(
        redis=None,
        settings=types.SimpleNamespace(max_attempts=3),
        complete=complete, mark_retry=mark_retry,
    )
    worker = crawl_job_worker.CrawlJobWorker(queue, {}, "consumer-test")
    outcome = await worker._process_attempt(
        types.SimpleNamespace(task_id="job-budget", stream_id="1-0"),
        {"urls": [ARCHIVE_URL], "browser_config": {}, "crawler_config": {},
         "result_fields": None},
        types.SimpleNamespace(number=1),
    )
    error = calls["complete"]["error"]
    assert outcome == ("failed", None, error)
    assert "archive_budget_spent" in error
    assert "retry after 43s" in error
    assert "mark_retry" not in calls


# ───────────────────────── routes ─────────────────────────
def _bearer():
    from auth import create_access_token

    token = create_access_token({"sub": "tester@x.com"}, scope="data")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def app_client(redis_url, monkeypatch):
    """The real FastAPI app with the test Redis behind the archive budget."""
    import server
    from starlette.testclient import TestClient

    client = aioredis.from_url(redis_url)
    monkeypatch.setattr(server, "redis", client)
    return TestClient(server.app, raise_server_exceptions=False)


@pytest.mark.parametrize("endpoint,payload", [
    ("/archive/fetch", {"url": "https://archive.org/"}),
    ("/crawl", {"urls": [ARCHIVE_URL], "browser_config": {},
                "crawler_config": {}}),
])
def test_throttled_routes_answer_429(
    endpoint, payload, app_client, monkeypatch, offline_dns,
):
    """A throttle on any route answers the typed 429 archive_budget_spent."""
    node = "route-429"
    monkeypatch.setenv("CRAWL4AI_NODE", node)

    async def throttled(url, max_body):
        return 503, {"Content-Type": "text/plain"}, b""

    monkeypatch.setattr(archive_budget, "_default_opener", throttled)
    r = app_client.post(endpoint, json=payload, headers=_bearer())
    assert r.status_code == 429
    body = r.json()
    assert body["code"] == "archive_budget_spent"
    assert body["detail"] == "archive.org budget for this node is spent"
    assert body["node"] == node
    assert isinstance(body["retryAfterSeconds"], int)
    assert body["retryAfterSeconds"] >= 1
