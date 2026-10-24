"""
governor.py - server-enforced resource governance (R5).

The deep-crawl DoS via a client-supplied unbounded max_pages is already closed
upstream: R2 forbids `deep_crawl_strategy` on an untrusted request body, and the
`urls` list is capped at 100 by the request schema. This module adds the two
remaining verifiable chokepoints:

  * a request body-size limit (ASGI middleware) so a giant body / inline `raw:`
    HTML cannot be buffered and processed in-process -> 413;
  * clamp_deep_crawl(): defense in depth for any *trusted* / server-built config
    that still carries a deep_crawl strategy with an unbounded page/depth count.

Heavier governance (bounded work queue replacing BackgroundTasks, per-principal
Redis quotas, wall-clock deadlines, stream decoupling) is left for the
integration-tested pass; gunicorn --limit-request-* covers the transport layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math

DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_MAX_PAGES = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_PER_PRINCIPAL_JOBS = 100

logger = logging.getLogger(__name__)


class BodySizeLimitMiddleware:
    """Reject HTTP requests whose received body exceeds the configured limit.

    ``Content-Length`` is only an early rejection hint: clients can omit it or
    stream with chunked transfer encoding.  Count the actual ASGI
    ``http.request`` bytes and replay the bounded body downstream so every
    transport gets the same limit.  A configured value of ``0`` is the
    documented unbounded mode and leaves ``receive`` untouched.
    """

    def __init__(self, app, max_bytes: int = DEFAULT_MAX_BODY_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name != b"content-length":
                continue
            try:
                if int(value) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                # An invalid declaration is not trusted; the actual received
                # bytes below remain authoritative.
                pass
            break

        messages = []
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    await self._reject(send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        replay_index = 0

        async def replay_receive():
            nonlocal replay_index
            if replay_index < len(messages):
                message = messages[replay_index]
                replay_index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, send):
        body = json.dumps({"detail": "Request body too large"}).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class CancelOnDisconnectMiddleware:
    """Cancel a browser-bound handler whose client hangs up before its response.

    Uvicorn only marks the request cycle disconnected and answers the next
    ``receive()`` with ``http.disconnect``; it never cancels the handler, and a
    non-stream handler stops reading once its body is parsed. Without this, a
    caller that gives up keeps its browser page, admission permit and replica
    page-budget slot until the crawl ends or hits limits.wall_clock_s. This
    waits on ``receive()`` exactly as Starlette's StreamingResponse does
    (listen_for_disconnect), once the body is consumed so it never takes body
    bytes from the handler. A started response is left to finish: a stream
    handles its own disconnect, and nothing remains to free.
    """

    def __init__(self, app, paths: frozenset[str]):
        self.app = app
        self.paths = paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return
        body_consumed = asyncio.Event()
        response_started = False
        disconnected = False

        async def handler_receive():
            message = await receive()
            if message["type"] == "http.disconnect" or not message.get("more_body", False):
                body_consumed.set()
            return message

        async def handler_send(message):
            nonlocal response_started
            response_started |= message["type"] == "http.response.start"
            await send(message)

        handler = asyncio.create_task(self.app(scope, handler_receive, handler_send))

        async def cancel_on_disconnect():
            nonlocal disconnected
            await body_consumed.wait()
            while (await receive())["type"] != "http.disconnect":
                pass
            if not response_started:
                disconnected = True
                handler.cancel()

        watcher = asyncio.create_task(cancel_on_disconnect())
        try:
            await handler
        except asyncio.CancelledError:
            # Only the cancellation this middleware caused ends here; a server
            # shutdown cancelling this task still propagates.
            if not disconnected or asyncio.current_task().cancelling():
                raise
            logger.info("Client disconnected before the %s response; cancelled its work",
                        scope["path"])
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)


def clamp_deep_crawl(crawler_config, *, max_pages: int = DEFAULT_MAX_PAGES,
                     max_depth: int = DEFAULT_MAX_DEPTH) -> None:
    """Clamp an attached deep-crawl strategy's page/depth budget in place.

    Defense in depth: untrusted bodies cannot set deep_crawl_strategy at all
    (R2), but a server/base config might, and the library default for max_pages
    is infinity.
    """
    dc = getattr(crawler_config, "deep_crawl_strategy", None)
    if dc is None:
        return
    mp = getattr(dc, "max_pages", None)
    if mp is None or (isinstance(mp, float) and math.isinf(mp)) or mp > max_pages:
        dc.max_pages = max_pages
    md = getattr(dc, "max_depth", None)
    if md is None or md > max_depth:
        dc.max_depth = max_depth


def max_body_bytes_from_config(config: dict) -> int:
    return int((config.get("limits", {}) or {}).get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))


def _limits(config: dict) -> dict:
    return config.get("limits", {}) or {}


def wall_clock_seconds(config: dict) -> float:
    """Per-crawl wall-clock deadline in seconds; 0 (default) => no deadline."""
    return float(_limits(config).get("wall_clock_s", 0) or 0)


def job_queue_caps(config: dict) -> dict:
    """Bounded-job-queue settings; 0 => unbounded/unlimited (current behavior)."""
    q = _limits(config).get("queue", {}) or {}
    per_principal = q.get("per_principal", DEFAULT_PER_PRINCIPAL_JOBS)
    return {
        "maxsize": int(q.get("maxsize", 1) or 0),
        "workers": int(q.get("workers", 1) or 1),
        "per_principal": int(per_principal) if per_principal is not None else 0,
    }
