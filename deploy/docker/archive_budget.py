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

Node identity comes from ``CRAWL4AI_NODE`` (the Swarm ``{{.Node.Hostname}}``
service template). Falling back to the container hostname would split one
node's budget across two keys, so an unset variable fails loudly with 503.
"""

from __future__ import annotations

import asyncio
import errno
import math
import os
from urllib.parse import urlparse

from fastapi import HTTPException, status

FETCH_USER_AGENT = "aVentureBot/1.0 (+https://aventure.vc) crawl4ai"
FETCH_TIMEOUT_SECONDS = 60.0

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

# KEYS[1]: the node's next-start key. ARGV: pause_ms. Never moves the stored
# start earlier — an in-flight slow request must not shorten a pause another
# caller already recorded. Returns the pushed-back wait in ms.
_PAUSE_LUA = """
local pause_ms = tonumber(ARGV[1])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local stored = tonumber(redis.call('GET', KEYS[1]) or '0') or 0
local target = math.max(stored, now + pause_ms)
redis.call('SET', KEYS[1], target, 'PX', math.max(math.floor(target - now), 1))
return math.floor(target - now)
"""


class ArchiveBudgetSpent(HTTPException):
    """This node's archive.org budget is spent for ``retry_after`` seconds.

    Answered as HTTP 429 by the exception handler in server.py; the worker
    treats it like any deliberate 4xx/5xx HTTPException.
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


def is_archive_url(url, config: dict) -> bool:
    """True when ``url``'s host is an ``archive.hosts`` host or a subdomain of one."""
    hostname = (urlparse(str(url)).hostname or "").rstrip(".")
    return any(hostname == host or hostname.endswith("." + host) for host in config["archive"]["hosts"])


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
    wait_ms = await redis.eval(
        _PAUSE_LUA, 1, BUDGET_KEY_PREFIX + node(), _ms(config, "throttle_pause_seconds"),
    )
    return int(wait_ms) / 1000.0


async def gate_url(redis, url, config: dict) -> None:
    """Reserve one archive.org slot for ``url`` and sleep the returned wait.

    Non-archive URLs pass through without touching Redis.
    """
    if not is_archive_url(url, config):
        return
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="archive budget store is not configured",
        )
    wait = await reserve(redis, config)
    if wait > 0:
        await asyncio.sleep(wait)


async def gate_urls(redis, urls, config: dict) -> None:
    """Reserve one slot per archive URL in a request, in order."""
    for url in urls or []:
        await gate_url(redis, url, config)


def _names_connection_refused(error_message) -> bool:
    lowered = (error_message or "").lower()
    return "connection refused" in lowered or "connection_refused" in lowered


async def maybe_pause_for_result(redis, config: dict, url,
                                 status_code, error_message) -> bool:
    """Pause the node budget when a crawl result shows archive.org throttling.

    Covers HTTP 429/503 statuses and errors naming connection refused
    (ERR_CONNECTION_REFUSED / Connection refused).
    """
    if redis is None or not is_archive_url(url, config):
        return False
    if status_code not in (429, 503) and not _names_connection_refused(error_message):
        return False
    await pause(redis, config)
    return True


def _is_connection_refused(exc: BaseException) -> bool:
    err = getattr(exc, "os_error", None) or exc
    return getattr(err, "errno", None) == errno.ECONNREFUSED or "connection refused" in str(exc).lower()


async def _default_opener(url: str):
    """GET ``url`` with aiohttp; returns (status, content_type, body_text)."""
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(
            timeout=timeout, headers={"User-Agent": FETCH_USER_AGENT},
        ) as session:
            async with session.get(url) as response:
                body = (await response.read()).decode("utf-8", errors="replace")
                return response.status, response.headers.get("Content-Type", ""), body
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


async def fetch(redis, config: dict, url: str, *, opener=None) -> dict:
    """Reserve a budget slot, GET an archive.org URL, and return its body.

    ``opener`` is the transport seam (defaults to aiohttp); it returns
    (status, content_type, body_text) and raises _ArchiveUpstreamRefused on a
    refused connection. Archive 429/503 and refused connections pause the node
    budget and re-raise ArchiveBudgetSpent.
    """
    if not is_archive_url(url, config):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="url host is not an archive.org host",
        )
    wait = await reserve(redis, config)
    if wait > 0:
        await asyncio.sleep(wait)
    try:
        status_code, content_type, body = await (opener or _default_opener)(url)
    except _ArchiveUpstreamRefused:
        retry_after = await pause(redis, config)
        raise ArchiveBudgetSpent(retry_after, node())
    if status_code in (429, 503):
        retry_after = await pause(redis, config)
        raise ArchiveBudgetSpent(retry_after, node())
    return {
        "url": url,
        "status": status_code,
        "contentType": content_type,
        "body": body,
    }
