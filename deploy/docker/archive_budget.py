"""Per-Swarm-node archive.org request budget.

archive.org rate-limits per client IP: a burst of roughly ten requests in a few
seconds gets TCP connection refused, often for an hour, and 429/503 responses
also appear. Each Swarm node has its own public IP, so the budget is keyed by
node — two replicas co-located on one node MUST share one budget or they would
double the request rate one IP sees.

``reserve`` atomically claims this node's next request start in Redis (single
Lua script, clock from Redis ``TIME`` so co-located replicas share one clock):
the next start is pushed ``archive.spacing_seconds`` past the previous one. When
the free budget is further out than ``archive.max_wait_seconds`` the caller gets
a typed ``ArchiveBudgetSpent`` refusal (HTTP 429 + Retry-After) instead of a
sleep, so the back-end can route the overflow elsewhere. When archive.org
itself throttles (429/503/connection refused) ``pause`` pushes the node's next
start out by ``archive.throttle_pause_seconds``.

Every archive.org URL a request names is fetched through ``fetch`` (one
reservation per hop, with manual archive-only redirects). archive_transport
renders the supplied HTML offline while dispatch retains the caller's URL.

Node identity comes from ``CRAWL4AI_NODE`` (the Swarm ``{{.Node.Hostname}}``
service template). Falling back to the container hostname would split one
node's budget across two keys, so an unset variable fails loudly with 503.
"""

from __future__ import annotations

import asyncio
import errno
import math
import os
from urllib.parse import urljoin, urlparse

from fastapi import HTTPException, status

FETCH_USER_AGENT = "aVentureBot/1.0 (+https://aventure.vc) crawl4ai"
FETCH_TIMEOUT_SECONDS = 60.0
FETCH_CHUNK_BYTES = 65536

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_TEXT_CONTENT_TYPES = frozenset({"application/json", "application/xml"})

BUDGET_KEY_PREFIX = "crawl4ai:archive:next:"

# KEYS[1]: the node's next-start key (ms). ARGV: spacing_ms, max_wait_ms.
# Returns {1, wait_ms} on success, {0, wait_ms} when the free budget is beyond
# max_wait. The value is a timestamp, so PX runs to that timestamp: the key
# disappears exactly when the budget it carried has elapsed anyway.
_RESERVE_LUA = """
local spacing_ms = tonumber(ARGV[1])
local max_wait_ms = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local stored = tonumber(redis.call('GET', KEYS[1]) or '0') or 0
local next_start = math.max(stored, now)
local wait_ms = next_start - now
if wait_ms > max_wait_ms then
  return {0, math.floor(wait_ms)}
end
local new_next = next_start + spacing_ms
redis.call('SET', KEYS[1], new_next, 'PX', math.max(math.floor(new_next - now), 1))
return {1, math.floor(wait_ms)}
"""

# KEYS[1]: the node's next-start key; KEYS[2]: its pause marker. ARGV: pause_ms.
# Publish both atomically so already-queued starts see a newly imposed pause.
# Never moves the stored
# start earlier — an in-flight slow request must not shorten a pause another
# caller already recorded. Returns the pushed-back wait in ms.
_PAUSE_LUA = """
local pause_ms = tonumber(ARGV[1])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local stored = tonumber(redis.call('GET', KEYS[1]) or '0') or 0
local target = math.max(stored, now + pause_ms)
redis.call('SET', KEYS[1], target, 'PX', math.max(math.floor(target - now), 1))
redis.call('SET', KEYS[2], '1', 'PX', math.max(math.floor(target - now), 1))
return math.floor(target - now)
"""


class ArchiveBudgetSpent(HTTPException):
    """This node's archive.org budget is spent for ``retry_after`` seconds.

    Answered as HTTP 429 by the exception handler in server.py; the durable
    crawl-job worker records it terminally (no lease retry).
    """

    def __init__(self, retry_after: float, node_name: str):
        seconds = max(1, math.ceil(retry_after))
        self.retry_after = seconds
        self.node = node_name
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="archive.org budget for this node is spent",
            headers={"Retry-After": str(seconds)},
        )


class _ArchiveUpstreamRefused(Exception):
    """The archive host refused the TCP connection (fetch opener sentinel)."""


def node() -> str:
    """This Swarm node's identity, from the CRAWL4AI_NODE service template.

    Never falls back to the container hostname: that would give each replica
    on a co-located node its own key, splitting one public IP's budget.
    """
    name = os.environ.get("CRAWL4AI_NODE")
    if not name:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="archive node identity is not configured",
        )
    return name


# The Internet Archive's own domain; web.archive.org and every other host it serves are subdomains.
ARCHIVE_HOST = "archive.org"


def is_archive_url(url) -> bool:
    """True when ``url``'s host is archive.org or one of its subdomains."""
    hostname = (urlparse(str(url)).hostname or "").rstrip(".")
    return hostname == ARCHIVE_HOST or hostname.endswith("." + ARCHIVE_HOST)


def _ms(config: dict, key: str) -> float:
    return float(config["archive"][key]) * 1000


async def reserve(redis, config: dict) -> float:
    """Reserve this node's next archive.org request start.

    Returns the seconds the caller must sleep before fetching. Raises
    ``ArchiveBudgetSpent`` when the next free start is further out than
    ``archive.max_wait_seconds``.
    """
    node_name = node()
    ok, wait_ms = await redis.eval(
        _RESERVE_LUA, 1, BUDGET_KEY_PREFIX + node_name,
        _ms(config, "spacing_seconds"), _ms(config, "max_wait_seconds"),
    )
    if int(ok) != 1:
        raise ArchiveBudgetSpent(int(wait_ms) / 1000.0, node_name)
    return int(wait_ms) / 1000.0


async def pause(redis, config: dict) -> float:
    """Push this node's next start out by ``archive.throttle_pause_seconds``.

    Never moves the stored start earlier. Returns the pushed-back wait in
    seconds, suitable for ArchiveBudgetSpent's retry_after.
    """
    key = BUDGET_KEY_PREFIX + node()
    wait_ms = await redis.eval(
        _PAUSE_LUA, 2, key, key + ":pause", _ms(config, "throttle_pause_seconds"),
    )
    return int(wait_ms) / 1000.0


def _is_connection_refused(exc: BaseException) -> bool:
    err = getattr(exc, "os_error", None) or exc
    return getattr(err, "errno", None) == errno.ECONNREFUSED or "connection refused" in str(exc).lower()


def _header(headers, name: str):
    for key, value in (headers or {}).items():
        if key.lower() == name:
            return value
    return None


def _is_text_content_type(content_type) -> bool:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    return (
        mime.startswith("text/")
        or mime in _TEXT_CONTENT_TYPES
        or mime.endswith("+json")
        or mime.endswith("+xml")
    )


async def _default_opener(url: str, max_body_bytes: int):
    """One GET with redirect following off; returns (status, headers, body).

    The body is read in chunks and reading stops once it passes
    ``max_body_bytes`` (0 = unbounded), so an oversized page never buffers in
    full — ``fetch`` still answers the cap violation itself.
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(
            timeout=timeout, headers={"User-Agent": FETCH_USER_AGENT},
        ) as session:
            async with session.get(url, allow_redirects=False) as response:
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.content.iter_chunked(FETCH_CHUNK_BYTES):
                    chunks.append(chunk)
                    size += len(chunk)
                    if 0 < max_body_bytes < size:
                        break
                return response.status, dict(response.headers), b"".join(chunks)
    except aiohttp.ClientError as exc:
        if _is_connection_refused(exc):
            raise _ArchiveUpstreamRefused(str(exc)) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"archive fetch failed: {type(exc).__name__}",
        ) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="archive fetch timed out",
        ) from exc


async def fetch(redis, config: dict, url: str, *, opener=None, include_headers=False) -> dict:
    """Reserve a budget slot, GET an archive.org URL, and return its body.

    Redirects are followed manually — one reservation per hop, at most
    ``archive.max_redirect_hops`` hops — and a Location that leaves the archive
    hosts stops the fetch with 502. The body is capped at
    ``limits.max_body_bytes`` (0 = unbounded) and must be text (text/*, JSON,
    or XML); violations answer 502 and 415. Archive 429/503 and refused
    connections pause the node budget and raise ArchiveBudgetSpent; any other
    archive status keeps its body for the caller to report.

    ``opener`` is the transport seam (defaults to aiohttp): called as
    ``opener(url, max_body_bytes)`` it returns ``(status, headers, body_bytes)``
    and raises _ArchiveUpstreamRefused on a refused connection.
    """
    if not is_archive_url(url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="url host is not an archive.org host",
        )
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="archive budget store is not configured",
        )
    max_body = int(config["limits"]["max_body_bytes"])
    open_url = opener or _default_opener
    current = str(url)
    for _hop in range(int(config["archive"]["max_redirect_hops"])):
        wait = await reserve(redis, config)
        if wait > 0:
            await asyncio.sleep(wait)
        # Queue allocation is not a throttle: only the separately published
        # pause invalidates a slot that was reserved before a 429/503/refusal.
        node_name = node()
        pause_ms = await redis.pttl(BUDGET_KEY_PREFIX + node_name + ":pause")
        if pause_ms > 0:
            raise ArchiveBudgetSpent(pause_ms / 1000.0, node_name)
        try:
            status_code, headers, body = await open_url(current, max_body)
        except _ArchiveUpstreamRefused:
            retry_after = await pause(redis, config)
            raise ArchiveBudgetSpent(retry_after, node())
        if status_code in (429, 503):
            retry_after = await pause(redis, config)
            raise ArchiveBudgetSpent(retry_after, node())
        location = _header(headers, "location")
        if status_code in _REDIRECT_STATUSES and location:
            target = urljoin(current, location)
            if not is_archive_url(target):
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="archive redirect left archive.org",
                )
            current = target
            continue
        break
    else:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="archive redirect exceeded max_redirect_hops",
        )
    if 0 < max_body < len(body):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="archive body exceeds max_body_bytes",
        )
    content_type = _header(headers, "content-type") or ""
    if not _is_text_content_type(content_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="archive body is not text",
        )
    result = {
        "url": current,
        "status": status_code,
        "contentType": content_type,
        "body": body.decode("utf-8", errors="replace"),
    }
    if include_headers:
        result["responseHeaders"] = dict(headers)
    return result
