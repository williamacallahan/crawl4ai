"""
R3 webhook connection-pinning tests (real loopback server, offline).

The webhook sender pins the connection to the validated IP (aiohttp resolver),
follows redirects manually, and re-validates every hop. We stub the egress
broker so a "good" host pins to the loopback test server and "internal" hosts /
redirects are refused.
"""

import asyncio
from threading import Event, get_ident

import egress_broker
import pytest
from egress_broker import EgressBlocked, PinnedTarget
from fastapi import BackgroundTasks, Request
from starlette.responses import JSONResponse
from webhook import WebhookDeliveryService

pytestmark = pytest.mark.posture


async def _raw_server(response: bytes):
    async def handle(reader, writer):
        await reader.read(65536)
        writer.write(response)
        await writer.drain()
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _svc():
    return WebhookDeliveryService({"webhooks": {"retry": {"max_attempts": 2,
                                   "initial_delay_ms": 1, "timeout_ms": 5000}}})


@pytest.mark.asyncio
@pytest.mark.parametrize("job_kind", ["llm", "crawl"])
async def test_job_webhook_validation_runs_off_the_event_loop(monkeypatch, job_kind):
    import api
    import job
    import utils

    loop = asyncio.get_running_loop()
    loop_thread = get_ident()
    started = loop.create_future()
    release = Event()
    calls = []
    webhook_url = "https://hooks.example/callback"
    webhook = {"webhook_url": webhook_url}
    llm_response = JSONResponse({"task_id": "llm_test"}, status_code=202)

    def blocking_validate(url):
        calls.append(url)
        assert get_ident() != loop_thread
        if len(calls) == 1:
            loop.call_soon_threadsafe(started.set_result, None)
        release.wait()

    async def accepted(*_args, **kwargs):
        assert kwargs["webhook_config"]["webhook_url"] == webhook_url
        return llm_response if job_kind == "llm" else "crawl_test"

    monkeypatch.setattr(
        utils if job_kind == "llm" else api, "validate_webhook_url", blocking_validate
    )
    if job_kind == "llm":
        payload = job.LlmJobPayload(
            url="https://example.com", q="summary", webhook_config=webhook
        )
        monkeypatch.setattr(job, "handle_llm_request", accepted)
        enqueue = job.llm_job_enqueue(
            payload,
            BackgroundTasks(),
            Request({"type": "http", "scheme": "http", "server": ("test", 80)}),
            None,
        )
    else:
        payload = job.CrawlJobPayload(urls=["https://example.com"], webhook_config=webhook)
        monkeypatch.setattr(job, "_config", {})
        monkeypatch.setattr(api.CrawlJobQueue, "enqueue", accepted)
        enqueue = job.crawl_job_enqueue(payload, BackgroundTasks(), None)

    task = asyncio.create_task(enqueue)
    try:
        done, _ = await asyncio.wait(
            {task, started}, timeout=30, return_when=asyncio.FIRST_COMPLETED
        )
        if task.done():
            task.result()
        assert started in done
        assert not task.done()
    finally:
        release.set()
        if not started.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert task.result() == (llm_response if job_kind == "llm" else {"task_id": "crawl_test"})
    assert calls == [webhook_url]


@pytest.mark.asyncio
class TestWebhookPinning:
    async def test_delivered_to_pinned_host(self, monkeypatch):
        server, port = await _raw_server(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        monkeypatch.setattr(
            egress_broker, "resolve_and_pin",
            lambda url: PinnedTarget("http", "good.example", port, "127.0.0.1"),
        )
        try:
            ok = await _svc().send_webhook(f"http://good.example:{port}/cb", {"x": 1})
            assert ok is True
        finally:
            server.close()

    async def test_internal_target_blocked_no_retry(self, monkeypatch):
        def boom(url):
            raise EgressBlocked()
        monkeypatch.setattr(egress_broker, "resolve_and_pin", boom)
        ok = await _svc().send_webhook("http://169.254.169.254/cb", {"x": 1})
        assert ok is False

    async def test_redirect_to_internal_blocked(self, monkeypatch):
        server, port = await _raw_server(
            b"HTTP/1.1 302 Found\r\nLocation: http://evil.internal/\r\nContent-Length: 0\r\n\r\n"
        )
        def pin(url):
            if url == "http://evil.internal/":
                raise EgressBlocked()
            return PinnedTarget("http", "good.example", port, "127.0.0.1")

        monkeypatch.setattr(egress_broker, "resolve_and_pin", pin)
        try:
            ok = await _svc().send_webhook(f"http://good.example:{port}/cb", {"x": 1})
            assert ok is False  # the next pinned dial refused the redirect
        finally:
            server.close()

    async def test_uses_pinned_resolver(self):
        # The resolver returns the pinned IP for any host (TLS still verifies the
        # hostname); confirm the wiring.
        from webhook import _PinnedResolver
        r = _PinnedResolver("good.example", "203.0.113.7")
        out = await r.resolve("good.example", 443)
        assert out[0]["host"] == "203.0.113.7"
        assert out[0]["hostname"] == "good.example"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_name,expected_attempts", [("EAI_AGAIN", 2), ("EAI_NONAME", 1), ("EAI_FAIL", 1)])
async def test_dns_delivery_uses_existing_bounded_retry_owner(monkeypatch, error_name, expected_attempts):
    import socket

    monkeypatch.setattr(egress_broker, "ALLOW_INTERNAL", False)
    calls = []
    reservations = []

    def unavailable(host, port, *args, **kwargs):
        calls.append(host)
        raise socket.gaierror(getattr(socket, error_name), "private resolver information")

    async def reserve():
        reservations.append(len(reservations) + 1)
        return reservations[-1]

    monkeypatch.setattr(socket, "getaddrinfo", unavailable)
    assert await _svc().send_webhook("https://unavailable.example/cb", {}, before_attempt=reserve) is False
    assert calls == ["unavailable.example"] * expected_attempts
    assert reservations == list(range(1, expected_attempts + 1))


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect", [False, True])
async def test_temporary_dns_then_normal_resolution_delivers_webhook(monkeypatch, redirect):
    import socket

    # Exercise real DNS classification, pinning and aiohttp against loopback
    # using the documented trusted-internal deployment configuration.
    monkeypatch.setattr(egress_broker, "ALLOW_INTERNAL", True)
    posts = []

    async def handle(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        for line in request.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                await reader.readexactly(int(line.split(b":", 1)[1]))
        posts.append(request)
        if redirect and b"Host: origin.example:" in request:
            writer.write(f"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://hook.example:{port}/cb\r\nContent-Length: 0\r\n\r\n".encode())
        else:
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    calls = []
    failed_once = False
    reservations = []

    def resolve(host, requested_port, *args, **kwargs):
        nonlocal failed_once
        calls.append(host)
        if host == "hook.example" and not failed_once:
            failed_once = True
            raise socket.gaierror(socket.EAI_AGAIN, "private resolver information")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", requested_port))]

    async def reserve():
        reservations.append(len(reservations) + 1)
        return reservations[-1]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    host = "origin.example" if redirect else "hook.example"
    try:
        assert await _svc().send_webhook(f"http://{host}:{port}/cb", {}, before_attempt=reserve) is True
    finally:
        server.close()
        await server.wait_closed()
    assert reservations == [1, 2]
    assert calls == (["origin.example", "hook.example", "origin.example", "hook.example"] if redirect else ["hook.example", "hook.example"])
    assert len(posts) == (3 if redirect else 1)
