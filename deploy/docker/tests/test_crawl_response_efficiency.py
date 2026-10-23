"""HTTP transport, projection, and event-loop regressions without a browser."""

import asyncio
import gzip
import json
import threading
import time
from unittest.mock import AsyncMock

import api
import crawler_pool
import pytest
from auth import create_access_token
from crawl4ai import CrawlResult
from crawl4ai.models import MarkdownGenerationResult


@pytest.fixture
def crawl_result(monkeypatch):
    result = CrawlResult(
        url="https://example.com", success=True, html="<p>café 🌍</p>" * 50_000,
        markdown=MarkdownGenerationResult(
            raw_markdown="selected markdown", markdown_with_citations="selected markdown",
            references_markdown="",
        ), metadata={"title": "Example"},
    )

    class Crawler:
        async def arun(self, *args, **kwargs):
            return result

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", AsyncMock(return_value=Crawler()))
    monkeypatch.setattr(crawler_pool, "release_crawler", AsyncMock())
    return result


def auth_headers():
    return {"Authorization": "Bearer " + create_access_token({"sub": "efficiency-test"})}


def test_crawl_gzip_and_plain_json(stock_client, crawl_result):
    headers = auth_headers()
    payload = {"urls": [crawl_result.url]}
    # TestClient otherwise supplies its own Accept-Encoding default.
    stock_client.headers.pop("accept-encoding", None)
    plain = stock_client.post("/crawl", json=payload, headers=headers)
    assert plain.status_code == 200
    assert "content-encoding" not in plain.headers
    assert plain.json()["results"][0]["html"] == crawl_result.html
    with stock_client.stream(
        "POST", "/crawl", json=payload, headers={**headers, "Accept-Encoding": "gzip"}
    ) as compressed:
        wire = b"".join(compressed.iter_raw())
        assert compressed.status_code == 200
        assert compressed.headers.get("content-encoding") == "gzip"
        decoded = json.loads(gzip.decompress(wire))
    assert decoded["results"] == plain.json()["results"]
    assert len(wire) < len(plain.content) / 4


@pytest.mark.parametrize("size,compressed", [(1023, False), (1024, True)])
def test_gzip_one_kib_boundary(stock_client, server_module, monkeypatch, size, compressed):
    result = {"success": True, "results": [{"html": ""}]}
    overhead = len(json.dumps(result, separators=(",", ":")).encode())
    result["results"][0]["html"] = "x" * (size - overhead)
    monkeypatch.setattr(server_module, "handle_crawl_request", AsyncMock(return_value=result))
    response = stock_client.post(
        "/crawl", json={"urls": ["https://example.com"]},
        headers={**auth_headers(), "Accept-Encoding": "gzip"},
    )
    assert response.status_code == 200
    assert (response.headers.get("content-encoding") == "gzip") is compressed
    assert len(response.content) == size
    assert response.json() == result


def test_mcp_schema_is_not_compressed(stock_client):
    response = stock_client.get(
        "/mcp/schema", headers={**auth_headers(), "Accept-Encoding": "gzip"}
    )
    assert response.status_code == 200
    assert len(response.content) > 1024
    assert "content-encoding" not in response.headers


@pytest.mark.parametrize("fields", [["markdown"], [], None])
def test_crawl_result_fields_match_job(stock_client, crawl_result, fields):
    response = stock_client.post(
        "/crawl", json={"urls": [crawl_result.url], "result_fields": fields},
        headers=auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    expected = crawl_result.model_dump()
    if fields:
        expected = {"markdown": expected["markdown"]}
    assert body["results"] == [expected]


def test_invalid_result_field_matches_job_error(stock_client, crawl_result):
    payload = {"urls": ["https://example.com"], "result_fields": ["not_a_field"]}
    headers = auth_headers()
    sync = stock_client.post("/crawl", json=payload, headers=headers)
    job = stock_client.post("/crawl/job", json=payload, headers=headers)
    assert sync.status_code == job.status_code == 422
    assert sync.json()["detail"] == job.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_large_crawl_serialization_keeps_heartbeat(server_module, monkeypatch, stream):
    import httpx

    class LargeResult:
        def model_dump(self):
            # A deliberately slow result serializer, not a browser/network wait.
            time.sleep(0.12)
            return {"url": "https://example.com", "success": True, "html": "x" * 5_000_000}

    class Crawler:
        async def arun(self, *args, **kwargs):
            return LargeResult()

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", AsyncMock(return_value=Crawler()))
    monkeypatch.setattr(crawler_pool, "release_crawler", AsyncMock())
    if stream:
        async def results():
            yield LargeResult()

        monkeypatch.setattr(
            server_module, "handle_stream_crawl_request",
            AsyncMock(return_value=(None, results(), None, None)),
        )
    gaps = []
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    encoding_threads = []
    dumps = json.dumps

    def record_encoding(value, *args, **kwargs):
        if isinstance(value, dict) and (
            "server_processing_time_s" in value or "server_memory_mb" in value
        ):
            encoding_threads.append(threading.get_ident())
        return dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", record_encoding)

    async def heartbeat():
        previous = loop.time()
        while True:
            await asyncio.sleep(0.005)
            now = loop.time()
            gaps.append(now - previous)
            previous = now

    ticker = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server_module.app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/crawl/stream" if stream else "/crawl", json={"urls": ["https://example.com"]},
                headers={**auth_headers(), "Accept-Encoding": "identity"},
            )
        await asyncio.sleep(0.01)
        assert response.status_code == 200
        result = json.loads(response.content.splitlines()[0]) if stream else response.json()["results"][0]
        assert len(result["html"]) == 5_000_000
        # The encoder runs on a worker thread. A wall-clock bound on the loop
        # gap is not asserted: CPython's C encoder holds the GIL for the whole
        # dumps call and the in-process transport copies 5 MB on the loop, so
        # the gap measures the runner's load (26.7 ms local, 84 ms in CI).
        assert encoding_threads and all(t != loop_thread for t in encoding_threads)
        print(f"5 MB serialization (stream={stream}) heartbeat maximum gap: {max(gaps) * 1000:.1f} ms")
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path,payload", [
    ("/crawl/stream", {}), ("/crawl", {"crawler_config": {"stream": True}}),
])
async def test_ndjson_arrives_before_next_result(server_module, monkeypatch, path, payload):
    # ASGI send observes actual chunks; httpx ASGITransport buffers streams.
    next_result = asyncio.Event()
    first_line = asyncio.Event()
    sent = []

    async def results():
        yield CrawlResult(url="https://example.com/1", success=True, html="one")
        await next_result.wait()
        yield CrawlResult(url="https://example.com/2", success=True, html="two")

    monkeypatch.setattr(
        server_module, "handle_stream_crawl_request",
        AsyncMock(return_value=(None, results(), None, None)),
    )
    request_body = json.dumps({"urls": ["https://example.com"], **payload}).encode()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": request_body, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and b'example.com/1' in message.get("body", b""):
            first_line.set()

    headers = {**auth_headers(), "Accept-Encoding": "gzip", "Content-Type": "application/json"}
    task = asyncio.create_task(server_module.app({
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 12345), "server": ("testserver", 80),
    }, receive, send))
    try:
        await asyncio.wait_for(first_line.wait(), 1)
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert start["status"] == 200
        assert b"content-encoding" not in dict(start["headers"])
        assert not task.done()
        next_result.set()
        await asyncio.wait_for(task, 1)
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        assert json.loads(body.splitlines()[1])["url"] == "https://example.com/2"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
