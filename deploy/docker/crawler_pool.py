# crawler_pool.py - Smart browser pool with tiered management
import asyncio
import ctypes
import ctypes.util
import fcntl
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional

import psutil
from fastapi import HTTPException
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from prometheus_client import Counter
from utils import (
    BrowserWatchdogSettings,
    PoolAdmissionSettings,
    get_container_memory_percent,
    load_config,
)

from crawl4ai import AsyncWebCrawler, BrowserConfig

logger = logging.getLogger(__name__)
CONFIG = load_config()

# Pool tiers
PERMANENT: Optional[AsyncWebCrawler] = None  # Always-ready default browser
HOT_POOL: Dict[str, AsyncWebCrawler] = {}    # Frequent configs
COLD_POOL: Dict[str, AsyncWebCrawler] = {}   # Rare configs
LAST_USED: Dict[str, float] = {}
USAGE_COUNT: Dict[str, int] = {}
LOCK = asyncio.Lock()

# Config
MEM_LIMIT = CONFIG.get("crawler", {}).get("memory_threshold_percent", 95.0)
# Above this container-memory share the janitor recycles the busiest browser
# instead of waiting for MEM_LIMIT to refuse new ones.
RECYCLE_LIMIT = CONFIG.get("crawler", {}).get("recovery_threshold_percent", MEM_LIMIT)
# All three processes read the same container-wide percent, so each restarts at
# most one browser per this interval: ten passes at the 30s pressure interval.
MEMORY_RECYCLE_INTERVAL_SECONDS = 300
_LAST_MEMORY_RECYCLE = float("-inf")
BASE_IDLE_TTL = CONFIG.get("crawler", {}).get("pool", {}).get("idle_ttl_sec", 300)
MAX_ACTIVE_REQUESTS = CONFIG.get("crawler", {}).get("pool", {}).get("max_pages", 30)
MAX_BROWSER_INSTANCES = CONFIG.get("crawler", {}).get("pool", {}).get(
    "max_browser_instances", MAX_ACTIVE_REQUESTS
)
# Keep the optional tighter per-process max_pages cap, sharing its wait deadline
# with the replica budget. It cannot increase the replica's total allowance.
ADMISSION_SEM = asyncio.Semaphore(MAX_ACTIVE_REQUESTS)
ADMISSION = PoolAdmissionSettings.from_config(CONFIG)
DEFAULT_CONFIG_SIG = None  # Cached sig for default config
_CLOSE_TASKS = set()  # In-flight background closes; close_all() drains them at shutdown
WATCHDOG = BrowserWatchdogSettings.from_config(CONFIG)


def _require_schedstat() -> None:
    """Refuse to start where the renderer CPU watchdog cannot read thread load.

    A kernel without CONFIG_SCHED_INFO has no schedstat file or reports
    "0 0 0", which would hide every runaway renderer from the watchdog.
    """
    try:
        on_cpu_ns = int(Path("/proc/self/schedstat").read_text().split()[0])
    except FileNotFoundError as error:
        raise RuntimeError(
            "/proc/self/schedstat is missing; the renderer CPU watchdog needs Linux "
            "with CONFIG_SCHED_INFO"
        ) from error
    if on_cpu_ns == 0:
        raise RuntimeError(
            "/proc/self/schedstat reports no CPU time; the renderer CPU watchdog "
            "needs a kernel with CONFIG_SCHED_INFO"
        )


_require_schedstat()
RENDERER_KILLS = {"cpu": 0, "rss": 0}
# Preserve the process-wide collector when the pool configuration is reloaded.
if "RENDERER_KILL_METRIC" not in globals():
    RENDERER_KILL_METRIC = Counter(
        "crawl4ai_renderer_kills_total",
        "Renderer processes terminated by the browser resource watchdog",
        ["reason"],
    )


@dataclass
class _RendererSample:
    cpu_seconds: float
    runnable_ns_by_tid: dict[int, int]
    sampled_at: float
    over_cpu_since: Optional[float] = None


_RENDERER_SAMPLES: dict[tuple[int, float], _RendererSample] = {}


def _pos(v):
    """Config values can be null, a bool, or a string; only a positive number counts."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0


# Leak-backstop ceiling: a "busy" browser untouched past it is force-closed.
# 6h floor: streams have no deadline and nothing refreshes LAST_USED mid-crawl.
_WALL_CLOCK = _pos(CONFIG.get("limits", {}).get("wall_clock_s", 0))
STALE_CEILING = _pos(CONFIG.get("crawler", {}).get("pool", {}).get("stale_lease_s", 0)) or max(2 * _WALL_CLOCK, 21600)

# Timestamp of the last completed janitor pass. The task is retained on
# app.state, so a mid-life raise parks its exception until shutdown and prints
# nothing: reclamation stops silently and the container stays "healthy" while
# refusing every new browser. /monitor/health reports this as
# janitor.seconds_since_last_pass; the container healthcheck does NOT read it
# and still returns 200 while get_crawler refuses, so recovering a wedged
# replica is an operator/scheduled-restart action, not an automatic one.
LAST_JANITOR_PASS = 0.0

# glibc gives every thread that mallocs its own arena (up to 8 x cores) and
# never returns their freed interior pages. Scraping and markdown generation run
# on asyncio.to_thread workers, so each crawl process grew 16-26 such arenas
# holding 118-290 MiB, against 10-43 MiB of live Python objects (measured from
# /proc/<pid>/smaps on production replicas). Two arenas let freed blocks be
# reused across threads, and the janitor's malloc_trim hands free pages back.
# Set at import, before the event loop starts any executor thread.
_M_ARENA_MAX = -8
_LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
_LIBC.mallopt(_M_ARENA_MAX, 2)


class BrowserCapacityError(HTTPException, RuntimeError):
    """Retryable capacity refusal, preserved by the API's HTTPException paths."""

    def __init__(self, detail: str = "Crawler browser pool is at capacity"):
        super().__init__(503, detail=detail, headers={"Retry-After": "2"})

    def __str__(self) -> str:
        return self.detail


class MemoryCapacityError(BrowserCapacityError, MemoryError):
    """Memory pressure is the same retryable admission refusal."""


class ReplicaPageBudget:
    """One independent flock per slot, shared by every container process.

    Files are never unlinked: every contender must lock the same inode. Each
    lease owns a fresh, non-inheritable descriptor; close or process death
    releases the kernel lock. No PID files or stale-slot cleanup are needed.
    """

    def __init__(self, max_pages: int, directory: str | Path = "/tmp/crawl4ai-page-budget"):
        self.max_pages = max_pages
        self.directory = Path(directory)

    async def acquire(self, wait_seconds: float) -> BinaryIO:
        deadline = time.monotonic() + wait_seconds
        self.directory.mkdir(parents=True, exist_ok=True)
        while True:
            for slot in range(self.max_pages):
                lease = (self.directory / f"{slot}.lock").open("a+b")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lease
                except BlockingIOError:
                    lease.close()
                except BaseException:
                    lease.close()
                    raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BrowserCapacityError()
            await asyncio.sleep(min(0.05, remaining))


PAGE_BUDGET = ReplicaPageBudget(ADMISSION.replica_max_pages)
_WORKER_PROCESS = False


def _admission_wait_seconds() -> float:
    if _WORKER_PROCESS:
        return ADMISSION.worker_admission_wait_seconds
    return ADMISSION.admission_wait_seconds


def use_worker_admission_wait() -> None:
    """Durable job workers wait out a busy replica inside their attempt.

    An HTTP caller gets a bounded 503; a worker refused after the HTTP bound
    would spend a retry on contention and repeat its archive prefetch.
    """
    global _WORKER_PROCESS
    _WORKER_PROCESS = True


def _check_admission_memory() -> None:
    # Only a new browser is refused: refusing hits on running browsers would
    # turn a replica whose heap sits above the limit into a 503 sink.
    mem_pct = get_container_memory_percent()
    if mem_pct >= MEM_LIMIT:
        raise MemoryCapacityError(f"Memory at {mem_pct:.1f}%, refusing a new browser")


async def _acquire_admission() -> BinaryIO:
    wait_seconds = _admission_wait_seconds()
    deadline = time.monotonic() + wait_seconds
    try:
        if ADMISSION_SEM.locked():
            await asyncio.wait_for(ADMISSION_SEM.acquire(), wait_seconds)
        else:
            await ADMISSION_SEM.acquire()
    except asyncio.TimeoutError as error:
        raise BrowserCapacityError() from error
    try:
        return await PAGE_BUDGET.acquire(max(0, deadline - time.monotonic()))
    except BaseException:
        ADMISSION_SEM.release()
        raise


def _attach_admission(crawler: AsyncWebCrawler, lease: BinaryIO) -> None:
    leases = getattr(crawler, "_docker_page_leases", None)
    if not isinstance(leases, list):
        leases = []
        setattr(crawler, "_docker_page_leases", leases)
    leases.append(lease)


def get_pool_snapshot() -> dict:
    """Return a point-in-time snapshot of pool state for monitoring.

    This is intentionally lock-free. Under CPython's GIL, reading
    ``len(dict)``, ``dict.copy()``, and ``x is not None`` are atomic
    operations, so the monitor can safely call this without contending
    on the pool LOCK that is held during slow browser start/close ops.
    The worst case is a slightly stale count, which is acceptable for
    dashboard display purposes.
    """
    return {
        "permanent": PERMANENT,
        "permanent_sig": DEFAULT_CONFIG_SIG,
        "hot_pool": HOT_POOL.copy(),
        "cold_pool": COLD_POOL.copy(),
        "last_used": LAST_USED.copy(),
        "usage_count": USAGE_COUNT.copy(),
        "renderer_kills": RENDERER_KILLS.copy(),
    }


def _sig(cfg: BrowserConfig) -> str:
    """Generate config signature."""
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",",":"))
    return hashlib.sha1(payload.encode()).hexdigest()

def _is_default_config(sig: str) -> bool:
    """Check if config matches default."""
    return sig == DEFAULT_CONFIG_SIG


def _active_requests(crawler: AsyncWebCrawler) -> int:
    active = getattr(crawler, "active_requests", 0)
    return active if isinstance(active, int) else 0


def _set_active_requests(crawler: AsyncWebCrawler, active: int) -> None:
    setattr(crawler, "active_requests", active)


async def _start_or_retire(crawler: AsyncWebCrawler) -> None:
    try:
        await crawler.start()
    except BaseException:
        _retire_crawler(crawler)
        _close_in_background(crawler)
        raise


def _retire_crawler(crawler: AsyncWebCrawler) -> int:
    """Retire local bookkeeping; shared leases stay held until browser close."""
    if getattr(crawler, "_docker_admission_released", False) is True:
        return 0

    outstanding = max(0, _active_requests(crawler))
    setattr(crawler, "_docker_admission_released", True)
    _set_active_requests(crawler, 0)
    for _ in range(outstanding):
        ADMISSION_SEM.release()
    return outstanding


def _detach_pool_crawler(
    pool: Dict[str, AsyncWebCrawler], sig: str
) -> AsyncWebCrawler:
    """Remove a pooled crawler before its close can run."""
    crawler = pool.pop(sig)
    LAST_USED.pop(sig, None)
    USAGE_COUNT.pop(sig, None)
    _retire_crawler(crawler)
    return crawler


def _detach_permanent() -> Optional[AsyncWebCrawler]:
    """Remove the permanent crawler before its close can run."""
    global PERMANENT, DEFAULT_CONFIG_SIG
    crawler = PERMANENT
    if not crawler:
        return None

    sig = DEFAULT_CONFIG_SIG
    PERMANENT = None
    # DEFAULT_CONFIG_SIG deliberately survives: while PERMANENT is None the
    # signature is how a concurrent request recognises that this config belongs
    # to the permanent browser and must wait for its replacement instead of
    # creating a pool browser under the same signature.
    if sig:
        LAST_USED.pop(sig, None)
        USAGE_COUNT.pop(sig, None)
    _retire_crawler(crawler)
    return crawler


def _is_live(crawler: AsyncWebCrawler) -> bool:
    """Return whether the crawler still owns a usable Playwright browser.

    A recycling manager is live while its browser references are temporarily
    clear; page acquisition already waits on its bounded restart barrier.
    """
    try:
        manager = crawler.crawler_strategy.browser_manager
        if getattr(manager, "_recycling", False) is True:
            return True
        if getattr(manager, "pool_recycle_requested", False) is True:
            return False
        # A failed recycle or a close leaves _closing set, and a closed manager
        # refuses every page even while its browser object still reports
        # connected; only a replacement brings this config back.
        if getattr(manager, "_closing", False) is True:
            return False
        if manager.browser is not None:
            live = manager.browser.is_connected()
        else:
            live = (
                manager.default_context is not None
                and not manager.default_context.is_closed()
            )
        if not live:
            return False
        failures = getattr(manager, "consecutive_failures", 0)
        if isinstance(failures, int) and failures >= WATCHDOG.consecutive_failure_limit:
            # Drain, then restart: closing now would fail its in-flight crawls.
            manager.consecutive_failures = 0
            manager.request_pool_recycle("consecutive browser faults")
        return True
    except Exception:
        return False


def _is_recycling(crawler: AsyncWebCrawler) -> bool:
    """Return whether the crawler's browser manager is mid-recycle.

    Mirrors the recycler invariant the admission path (``_is_live``) honors: a
    recycling manager is live while its browser references are temporarily
    clear and ``start()`` races against the ``_closing`` TOCTOU. The janitor's
    eviction loop must skip a recycling manager too, or - with the manager lying
    fallow at ``active_requests==0`` while its shielded ``_recycle_browser`` runs
    ``close(_for_recycle=True)`` -> ``start()`` - the janitor's 60s close cap
    cancels ``close()`` at the shielded recycle await, and the keeps-running
    recycle's ``start()`` finishes a fresh driver + Chromium on an already
    detached manager with ``_closing=True`` and no future cleanup path: a
    monotonic Chromium process leak. ``_recycle_browser``'s ``finally``
    clears ``_recycling`` within ``RECYCLE_CLOSE_SECONDS + RECYCLE_START_SECONDS``
    (= 180s) of the bounded wait caps ``browser_manager.py`` defines, so this
    gate only delays eviction by a bounded one-shot, never forever - the next
    pass re-evaluates and evicts normally.

    Gating on ``_recycling`` (not ``_is_live``) is intentional: ``_is_live``
    returns True for any connected idle browser, which would disable the
    janitor's primary purpose of idle eviction past TTL.
    """
    try:
        manager = crawler.crawler_strategy.browser_manager
        return bool(getattr(manager, "_recycling", False))
    except Exception:
        return False


def _discard_if_unavailable(
    pool: Dict[str, AsyncWebCrawler], sig: str, tier: str
) -> Optional[asyncio.Task]:
    crawler = pool[sig]
    if _is_live(crawler):
        return None
    logger.warning(f"{tier} pool browser is unavailable; replacing it (sig={sig[:8]})")
    return _close_in_background(_detach_pool_crawler(pool, sig))


def _browser_count() -> int:
    return (
        (1 if PERMANENT else 0)
        + len(HOT_POOL)
        + len(COLD_POOL)
        + sum(not task.done() for task in _CLOSE_TASKS)
    )


def _make_browser_capacity(*, refuse_when_full: bool = True) -> Optional[asyncio.Task]:
    """Schedule an idle eviction, or wait for an existing close, before growth.

    ``refuse_when_full=False`` is the permanent browser's exemption: it still
    evicts or waits when it can, but a pool of busy browsers never keeps the
    default browser down, because nothing else would bring it back.
    """
    if _browser_count() < MAX_BROWSER_INSTANCES:
        return None

    for task in _CLOSE_TASKS:
        if not task.done():
            return task

    # A mid-recycle entry lies fallow at active_requests == 0 and - because
    # its recycle fires at the end of an aged crawl - is typically the oldest
    # LAST_USED candidate, so without this filter min() picks exactly it;
    # evicting it orphans Chromium (see _is_recycling). Skipping it can only
    # surface as the capacity RuntimeError below until the bounded recycle
    # window clears; the janitor's next pass then evicts it normally.
    idle_browser = [
        (LAST_USED.get(sig, 0), sig, pool)
        for pool in (COLD_POOL, HOT_POOL)
        for sig, crawler in pool.items()
        if _active_requests(crawler) == 0 and not _is_recycling(crawler)
    ]
    if idle_browser:
        _, idle_sig, idle_pool = min(idle_browser, key=lambda candidate: candidate[0])
        crawler = _detach_pool_crawler(idle_pool, idle_sig)
        logger.info(f"🧹 Replaced idle browser at pool capacity (sig={idle_sig[:8]})")
        return _close_in_background(crawler)

    if not refuse_when_full:
        return None
    raise BrowserCapacityError()

async def get_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    """Get crawler from pool with tiered strategy."""
    lease = await _acquire_admission()
    try:
        crawler = await _get_admitted_crawler(cfg)
        _attach_admission(crawler, lease)
        return crawler
    except BaseException:
        lease.close()
        ADMISSION_SEM.release()
        raise


async def get_dedicated_crawler(cfg: BrowserConfig, *, crawler_strategy=None) -> AsyncWebCrawler:
    """Create one request-owned crawler under the canonical pool admission.

    Dedicated hook crawlers are never returned by ``get_crawler``. They are
    temporarily registered in the existing cold-pool capacity registry under
    an unreachable signature so every browser instance is counted while its
    admission lease remains held from creation through disposal.
    """
    lease = await _acquire_admission()
    try:
        while True:
            close_task = None
            async with LOCK:
                _check_admission_memory()
                close_task = _make_browser_capacity()
                if close_task is None:
                    crawler = AsyncWebCrawler(
                        config=cfg, thread_safe=False, crawler_strategy=crawler_strategy,
                    )
                    await _start_or_retire(crawler)

                    sig = f"dedicated:{uuid.uuid4().hex}"
                    setattr(crawler, "_docker_request_owned", True)
                    setattr(crawler, "_docker_pool_sig", sig)
                    _set_active_requests(crawler, 1)
                    COLD_POOL[sig] = crawler
                    LAST_USED[sig] = time.time()
                    USAGE_COUNT[sig] = 1
                    _attach_admission(crawler, lease)
                    return crawler
            await asyncio.shield(close_task)
    except BaseException:
        lease.close()
        ADMISSION_SEM.release()
        raise


async def release_dedicated_crawler(crawler: AsyncWebCrawler) -> None:
    """Close a request-owned crawler, then release its admission lease.

    Cleanup runs in a shielded task so caller cancellation cannot leave a live
    browser registered or leak its admission leases.
    """
    existing_cleanup = getattr(crawler, "_docker_release_task", None)
    if isinstance(existing_cleanup, asyncio.Task):
        try:
            await asyncio.shield(existing_cleanup)
        except asyncio.CancelledError:
            with suppress(Exception):
                await asyncio.shield(existing_cleanup)
            raise
        return

    if getattr(crawler, "_docker_admission_released", False) is True:
        return

    sig = getattr(crawler, "_docker_pool_sig", None)
    if not isinstance(sig, str) or not sig.startswith("dedicated:"):
        raise RuntimeError("Crawler does not own a dedicated admission lease")

    async def detach_and_close() -> None:
        async with LOCK:
            if getattr(crawler, "_docker_admission_released", False) is True:
                return
            if COLD_POOL.get(sig) is crawler:
                detached = _detach_pool_crawler(COLD_POOL, sig)
            else:
                LAST_USED.pop(sig, None)
                USAGE_COUNT.pop(sig, None)
                detached = crawler
                _retire_crawler(crawler)
            setattr(crawler, "_docker_request_owned", False)
            setattr(crawler, "_docker_pool_sig", None)
            close_task = _close_in_background(detached)
        await asyncio.shield(close_task)

    cleanup_task = asyncio.create_task(detach_and_close())
    setattr(crawler, "_docker_release_task", cleanup_task)
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.shield(cleanup_task)
        raise
    finally:
        if getattr(crawler, "_docker_release_task", None) is cleanup_task:
            setattr(crawler, "_docker_release_task", None)


async def _get_admitted_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    """Resolve a crawler after request admission has bounded pool growth."""
    sig = _sig(cfg)
    while True:
        close_task = None
        replace_permanent = False
        async with LOCK:
            # Check permanent browser for default config
            if _is_default_config(sig):
                if PERMANENT is None:
                    # A replacement is already in flight. Creating a pool
                    # browser under the default signature here would leave a
                    # clone nothing can reach - _get_admitted_crawler returns
                    # PERMANENT first - and nothing can reclaim, because every
                    # permanent hit refreshes LAST_USED for that same signature.
                    replace_permanent = True
                elif not _is_live(PERMANENT):
                    logger.warning("Permanent browser is unavailable; replacing it")
                    close_task, _done = await _init_permanent_locked(
                        cfg, force=True, target=PERMANENT
                    )
                    replace_permanent = close_task is not None
                else:
                    LAST_USED[sig] = time.time()
                    USAGE_COUNT[sig] = USAGE_COUNT.get(sig, 0) + 1
                    _set_active_requests(PERMANENT, _active_requests(PERMANENT) + 1)
                    logger.info("🔥 Using permanent browser")
                    return PERMANENT

            # Check hot pool
            if not replace_permanent and close_task is None and sig in HOT_POOL:
                close_task = _discard_if_unavailable(HOT_POOL, sig, "Hot")
                if close_task is None:
                    crawler = HOT_POOL[sig]
                    LAST_USED[sig] = time.time()
                    USAGE_COUNT[sig] = USAGE_COUNT.get(sig, 0) + 1
                    active_requests = _active_requests(crawler) + 1
                    _set_active_requests(crawler, active_requests)
                    logger.info(f"♨️  Using hot pool browser (sig={sig[:8]}, active={active_requests})")
                    return crawler

            # Check cold pool (promote to hot if used 3+ times)
            if not replace_permanent and close_task is None and sig in COLD_POOL:
                close_task = _discard_if_unavailable(COLD_POOL, sig, "Cold")
                if close_task is None:
                    LAST_USED[sig] = time.time()
                    USAGE_COUNT[sig] = USAGE_COUNT.get(sig, 0) + 1
                    crawler = COLD_POOL[sig]

                    if USAGE_COUNT[sig] >= 3:
                        logger.info(f"⬆️  Promoting to hot pool (sig={sig[:8]}, count={USAGE_COUNT[sig]})")
                        HOT_POOL[sig] = COLD_POOL.pop(sig)
                        # Under LOCK and before this request's lease is counted:
                        # no retire can return a lease the request has not yet
                        # handed over, so a cancellation here leaves the permit
                        # with get_crawler's single release.
                        try:
                            from monitor import get_monitor

                            await get_monitor().track_janitor_event(
                                "promote", sig, {"count": USAGE_COUNT[sig]}
                            )
                        except Exception:
                            pass
                    else:
                        logger.info(f"❄️  Using cold pool browser (sig={sig[:8]})")
                    _set_active_requests(crawler, _active_requests(crawler) + 1)
                    return crawler

            if not replace_permanent and close_task is None:
                # Create new in cold pool
                _check_admission_memory()
                close_task = _make_browser_capacity()
                if close_task is None:
                    logger.info(f"🆕 Creating new browser in cold pool (sig={sig[:8]})")
                    crawler = AsyncWebCrawler(config=cfg, thread_safe=False)
                    await _start_or_retire(crawler)
                    _set_active_requests(crawler, 1)
                    COLD_POOL[sig] = crawler
                    LAST_USED[sig] = time.time()
                    USAGE_COUNT[sig] = 1
                    return crawler
        if close_task is not None:
            await asyncio.shield(close_task)
        if replace_permanent:
            await init_permanent(cfg)

async def release_crawler(crawler: AsyncWebCrawler):
    """Decrement active request count for a pooled crawler.

    Call this in a finally block after finishing work with a crawler
    obtained via get_crawler() so the janitor knows when it's safe
    to close idle browsers.
    """
    # No lock and no await: a disconnect cancels the caller at its next
    # suspension point, so anything awaited here is skipped and the browser
    # stays pinned against the janitor's active_requests check forever.
    # The event loop is single-threaded and neither call below suspends, so
    # the read-modify-write needs no lock.
    if getattr(crawler, "_docker_admission_released", False) is True:
        return
    active = _active_requests(crawler)
    if active > 0:
        _set_active_requests(crawler, active - 1)
        leases = getattr(crawler, "_docker_page_leases", None)
        if isinstance(leases, list) and leases:
            leases.pop().close()
        ADMISSION_SEM.release()


async def _init_permanent_locked(
    cfg: BrowserConfig,
    *,
    force: bool = False,
    target: Optional[AsyncWebCrawler] = None,
) -> tuple[Optional[asyncio.Task], bool]:
    """Bring up the permanent browser, or detach the current one for replacement.

    Returns ``(close_task, done)`` where ``close_task`` is a background close to
    await *outside* the pool lock (or ``None`` to stop looping), and ``done``
    records whether *this call* owns the (re)placement:

    * ``done=True``  - no work was needed (a live permanent is current), the
      current permanent was detached and will be rebuilt on the next loop, or a
      fresh permanent was created.
    * ``done=False`` - a force restart was *superseded*: a concurrent non-force
      caller already rebuilt ``PERMANENT`` into a crawler that is not the
      ``target`` the restart captured. The fresh permanent - and any in-flight
      crawl admitted on it - is left untouched; the original the restart meant
      to retire is already gone.

    ``target`` is the permanent instance a force restart intends to replace.
    A force restart detaches *only* that instance, never a permanent a
    concurrent request placed during the lock-free close wait - the race that
    killed in-flight default-config crawls. A ``None`` target means no permanent
    existed at capture time; a subsequently created permanent is also protected.
    Non-force callers ignore the target.
    """
    global PERMANENT, DEFAULT_CONFIG_SIG
    # Non-force callers are idempotent: a live permanent already serves this config.
    if PERMANENT and not force and _is_live(PERMANENT):
        return None, True
    if force and PERMANENT is not None and PERMANENT is not target:
        return None, False
    if PERMANENT and force and _is_recycling(PERMANENT):
        return None, True
    if PERMANENT:
        return _close_in_background(_detach_permanent()), True
    close_task = _make_browser_capacity(refuse_when_full=False)
    if close_task is not None:
        return close_task, True

    sig = _sig(cfg)
    logger.info("🔥 Creating permanent default browser")
    crawler = AsyncWebCrawler(config=cfg, thread_safe=False)
    await _start_or_retire(crawler)
    PERMANENT = crawler
    DEFAULT_CONFIG_SIG = sig
    LAST_USED[sig] = time.time()
    USAGE_COUNT[sig] = 0
    return None, True


async def init_permanent(cfg: BrowserConfig, *, force: bool = False) -> bool:
    """Initialize or atomically replace the permanent default browser.

    Returns ``True`` if this call performed the (re)placement, or ``False`` if a
    force restart was *superseded* - a concurrent non-force request already
    rebuilt the permanent into a fresh crawler that is not the one the restart
    captured, so the restart leaves it (and any in-flight crawl on it) alone.
    Non-force callers always return ``True``; the return value is ignored by
    the admission path and the server lifespan, so widening it from ``None`` to
    ``bool`` is backward-compatible.
    """
    original = PERMANENT
    while True:
        async with LOCK:
            close_task, done = await _init_permanent_locked(
                cfg, force=force, target=original
            )
            if close_task is None:
                return done
        await asyncio.shield(close_task)


async def close_all():
    """Close all browsers."""
    async with LOCK:
        if PERMANENT:
            _close_in_background(_detach_permanent())
        for pool in (HOT_POOL, COLD_POOL):
            for sig in list(pool):
                _close_in_background(_detach_pool_crawler(pool, sig))
    # Drain all closes (janitor's included) so shutdown doesn't destroy live tasks.
    if _CLOSE_TASKS:
        with suppress(Exception):
            await asyncio.wait_for(
                asyncio.gather(
                    *(asyncio.shield(task) for task in _CLOSE_TASKS),
                    return_exceptions=True,
                ),
                timeout=65,
            )


def _close_in_background(crawler: AsyncWebCrawler) -> asyncio.Task:
    """Close a browser without holding the pool LOCK.

    close() on a wedged browser can hang; awaiting it under LOCK would freeze
    get_crawler() and every future janitor pass for the whole server.
    """
    existing = getattr(crawler, "_docker_close_task", None)
    if isinstance(existing, asyncio.Task):
        return existing

    async def _close():
        try:
            # 60s gives Playwright's own graceful-then-SIGKILL cycle room to finish
            await asyncio.wait_for(crawler.close(), timeout=60)
        except asyncio.TimeoutError:
            # close() hung partway through browser_manager.close(), which runs
            # playwright.stop() last - so the node driver and its Chromium
            # process group are still alive with nothing referencing them.
            # Stopping the driver transport makes its exit handler SIGKILL the
            # whole group; without this the abandoned browser's memory stays
            # charged to the container for its life.
            logger.warning("⚠️ Browser close timed out after 60s - stopping its Playwright driver")
            with suppress(Exception):
                await asyncio.wait_for(
                    crawler.crawler_strategy.browser_manager.playwright.stop(), timeout=5
                )
        except Exception:
            pass
        finally:
            # Detachment is not page teardown. Other processes cannot see our
            # _CLOSE_TASKS, so keep charging retired pages through their close.
            leases = getattr(crawler, "_docker_page_leases", None)
            if isinstance(leases, list):
                while leases:
                    leases.pop().close()
    task = asyncio.create_task(_close())
    setattr(crawler, "_docker_close_task", task)
    _CLOSE_TASKS.add(task)
    task.add_done_callback(_CLOSE_TASKS.discard)
    return task


async def retire_pool_crawlers(
    sig_prefix: Optional[str] = None, *, cold_only: bool = False
) -> list[tuple[str, str]]:
    """Detach matching pool crawlers and finish their bounded closes."""
    retired = []
    close_tasks = []
    pools = (("cold", COLD_POOL),) if cold_only else (("hot", HOT_POOL), ("cold", COLD_POOL))

    async with LOCK:
        for pool_type, pool in pools:
            signatures = [
                sig
                for sig in pool
                if (sig_prefix is None or sig.startswith(sig_prefix))
                # Dedicated hook crawlers live in COLD_POOL under
                # "dedicated:<uuid>" signatures and are request-owned: only
                # release_dedicated_crawler may tear them down. Admin pool
                # retirement (kill_browser / restart_browser / force_cleanup)
                # reaches this function via prefix or blanket match and would
                # otherwise close an in-flight crawler mid-crawl, mirroring the
                # active-request guard _janitor_pass already enforces.
                and not sig.startswith("dedicated:")
            ]
            if sig_prefix is not None:
                signatures = signatures[:1]
            for sig in signatures:
                if _is_recycling(pool[sig]):
                    # Mirroring d15fdd6/881307d: detaching a mid-recycle entry
                    # lets the 60s close cap cancel its shielded start() and
                    # orphan the fresh driver + Chromium on a detached,
                    # _closing=True manager. Skip it; the recycle clears within
                    # the bounded close/start caps and the janitor's next pass
                    # re-evaluates and evicts normally.
                    continue
                crawler = _detach_pool_crawler(pool, sig)
                close_tasks.append(_close_in_background(crawler))
                retired.append((sig, pool_type))
            if retired and sig_prefix is not None:
                break

    if close_tasks:
        await asyncio.gather(
            *(asyncio.shield(task) for task in close_tasks),
            return_exceptions=True,
        )
    return retired


async def _recycle_busiest_browser() -> None:
    """Drain and restart the browser that served the most pages.

    Renderer and network-service memory grows with pages served and is only
    returned by restarting the browser process. A replica under steady traffic
    keeps its browsers busy, so idle eviction never reaches them and the page
    budget alone left them running for the whole six-hour recycle window. A
    browser must have spent half its page budget since its last start, and the
    process waits MEMORY_RECYCLE_INTERVAL_SECONDS between restarts, so pressure
    the browsers cannot relieve does not restart them every pass.
    """
    global _LAST_MEMORY_RECYCLE
    if time.monotonic() - _LAST_MEMORY_RECYCLE < MEMORY_RECYCLE_INTERVAL_SECONDS:
        return
    candidates = [
        crawler.crawler_strategy.browser_manager
        for crawler in (PERMANENT, *HOT_POOL.values(), *COLD_POOL.values())
        if crawler is not None
        and not getattr(crawler, "_docker_request_owned", False)
        and _is_live(crawler)
        and not _is_recycling(crawler)
    ]
    worn = [
        manager
        for manager in candidates
        if manager._pages_served >= max(1, manager.config.max_pages_before_recycle // 2)
    ]
    if not worn:
        return
    busiest = max(worn, key=lambda manager: manager._pages_served)
    pages_served = busiest._pages_served
    if await busiest.request_recycle():
        _LAST_MEMORY_RECYCLE = time.monotonic()
        logger.warning(f"Memory pressure: recycling a browser after {pages_served} pages")


def _runnable_ns_by_tid(pid: int) -> dict[int, int]:
    """Nanoseconds each live thread spent on a CPU or waiting in its run queue.

    CPU usage alone understates a spinning renderer on a loaded host, where it
    gets only its share of the CPUs; a spinning thread is runnable the whole
    time whatever the load. Fields one and two of
    /proc/<pid>/task/<tid>/schedstat are those two times.
    """
    task_dir = f"/proc/{pid}/task"
    try:
        tids = os.listdir(task_dir)
    except FileNotFoundError as error:
        raise psutil.NoSuchProcess(pid) from error
    runnable_ns_by_tid = {}
    for tid in tids:
        try:
            with open(f"{task_dir}/{tid}/schedstat") as schedstat:
                on_cpu_ns, run_queue_ns, _ = schedstat.read().split()
        except FileNotFoundError:
            continue  # The thread exited between listing and reading.
        runnable_ns_by_tid[int(tid)] = int(on_cpu_ns) + int(run_queue_ns)
    return runnable_ns_by_tid


def _busiest_thread_cores(
    previous: dict[int, int], current: dict[int, int], elapsed_seconds: float
) -> float:
    """The largest share of the interval any one live thread was runnable.

    Per-thread increases keep an exited thread's lifetime from cancelling the
    others' growth, and taking the busiest thread rather than the sum keeps
    many briefly waking threads on a saturated host from adding up to a spin.
    """
    increases = (
        runnable_ns - previous.get(tid, 0) for tid, runnable_ns in current.items()
    )
    return max(increases, default=0) / 1e9 / elapsed_seconds


def _watch_renderers(crawler: AsyncWebCrawler, now: float) -> set[tuple[int, float]]:
    """Sample only renderers descended from this manager's owned browser PID."""
    manager = crawler.crawler_strategy.browser_manager
    browser_pid = getattr(manager, "browser_pid", None)
    if isinstance(browser_pid, bool) or not isinstance(browser_pid, int):
        return set()
    try:
        descendants = psutil.Process(browser_pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return set()

    seen = set()
    for process in descendants:
        try:
            if "--type=renderer" not in process.cmdline():
                continue
            identity = (process.pid, process.create_time())
            seen.add(identity)
            cpu_times = process.cpu_times()
            cpu_seconds = cpu_times.user + cpu_times.system
            runnable_ns_by_tid = _runnable_ns_by_tid(process.pid)
            previous = _RENDERER_SAMPLES.get(identity)
            over_cpu_since = None
            if previous is not None and now > previous.sampled_at:
                elapsed_seconds = now - previous.sampled_at
                cores = max(
                    (cpu_seconds - previous.cpu_seconds) / elapsed_seconds,
                    _busiest_thread_cores(
                        previous.runnable_ns_by_tid, runnable_ns_by_tid, elapsed_seconds
                    ),
                )
                if cores > WATCHDOG.renderer_cpu_cores:
                    over_cpu_since = (
                        previous.over_cpu_since
                        if previous.over_cpu_since is not None
                        else previous.sampled_at
                    )
            _RENDERER_SAMPLES[identity] = _RendererSample(
                cpu_seconds=cpu_seconds,
                runnable_ns_by_tid=runnable_ns_by_tid,
                sampled_at=now,
                over_cpu_since=over_cpu_since,
            )
            reason = None
            rss_bytes = process.memory_info().rss
            rss_limit_bytes = WATCHDOG.renderer_rss_mb * 1024 * 1024
            if rss_bytes > rss_limit_bytes:
                reason = "rss"
            elif (
                over_cpu_since is not None
                and now - over_cpu_since >= WATCHDOG.renderer_cpu_seconds
            ):
                reason = "cpu"
            if reason is not None:
                process.kill()  # psutil sends SIGKILL on the supported Linux image.
                RENDERER_KILLS[reason] += 1
                RENDERER_KILL_METRIC.labels(reason=reason).inc()
                _RENDERER_SAMPLES.pop(identity, None)
                logger.warning(
                    "Killed browser renderer pid=%s reason=%s browser_pid=%s "
                    "created_at=%s rss_bytes=%s rss_limit_bytes=%s",
                    process.pid, reason, browser_pid, identity[1], rss_bytes, rss_limit_bytes,
                )
                manager.renderer_kills.append((time.monotonic(), reason))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Renderers exit while pages navigate; losing one during a sample
            # must not stop sampling the other browsers.
            continue
    return seen


def _probe_failure(error: Exception) -> bool:
    if isinstance(error, (asyncio.TimeoutError, PlaywrightTimeoutError)):
        return True
    if not isinstance(error, PlaywrightError):
        return False
    message = str(error).lower()
    return "target closed" in message or "has been closed" in message


async def _probe_browser(crawler: AsyncWebCrawler) -> bool:
    """Return whether the manager is closed or its CDP failure budget is spent."""
    manager = crawler.crawler_strategy.browser_manager
    if _is_recycling(crawler):
        return False
    if (
        getattr(manager, "_closing", False) is True
        or getattr(manager, "pool_recycle_requested", False) is True
    ):
        return True
    failures = getattr(manager, "consecutive_failures", 0)
    if isinstance(failures, int) and failures >= WATCHDOG.consecutive_failure_limit:
        # Keep probing while pages drain: an unresponsive browser can exhaust
        # its CDP failure budget even after ordinary faults request a restart.
        manager.consecutive_failures = 0
        manager.request_pool_recycle("consecutive browser faults")
    if _is_recycling(crawler):
        return False
    browser = manager.browser
    if browser is None:
        return False
    if getattr(manager, "_docker_health_browser", None) is not browser:
        manager._docker_health_browser = browser
        manager._docker_health_failures = 0
        manager._docker_health_session = None

    async def get_version():
        # One session belongs to this browser and closes with it. A detach in
        # timeout cleanup would send another command through the same wedged
        # connection and hold the janitor past its five-second probe budget.
        if manager._docker_health_session is None:
            manager._docker_health_session = await browser.new_browser_cdp_session()
        await manager._docker_health_session.send("Browser.getVersion")

    try:
        await asyncio.wait_for(get_version(), WATCHDOG.browser_probe_timeout_seconds)
    except Exception as error:
        if manager.browser is not browser or _is_recycling(crawler):
            return False
        if not _probe_failure(error):
            # Browser-level CDP is Chromium-only. Other browser engines retain
            # their existing lifecycle; a protocol error is not a failed target.
            manager._docker_health_failures = 0
            return False
        manager._docker_health_failures += 1
        logger.warning(
            "Browser health probe failed (%s/%s): %s",
            manager._docker_health_failures, WATCHDOG.consecutive_failure_limit,
            type(error).__name__,
        )
    else:
        if manager.browser is not browser or _is_recycling(crawler):
            return False
        manager._docker_health_failures = 0
    return manager._docker_health_failures >= WATCHDOG.consecutive_failure_limit


async def _dispose_unhealthy(crawler: AsyncWebCrawler, browser) -> None:
    """Detach the exact failed instance and use the pool's bounded close path."""
    async with LOCK:
        manager = crawler.crawler_strategy.browser_manager
        if _is_recycling(crawler) or manager.browser is not browser:
            return
        if PERMANENT is crawler:
            _close_in_background(_detach_permanent())
            return
        for pool in (HOT_POOL, COLD_POOL):
            for sig, current in list(pool.items()):
                if current is crawler:
                    _close_in_background(_detach_pool_crawler(pool, sig))
                    return


async def _watchdog_pass() -> None:
    """Bound resource and CDP failures for busy and idle browsers alike."""
    async with LOCK:
        crawlers = [
            crawler
            for crawler in (PERMANENT, *HOT_POOL.values(), *COLD_POOL.values())
            if crawler is not None
        ]
    now = time.monotonic()
    seen = set()
    for crawler in crawlers:
        if not _is_recycling(crawler):
            seen.update(_watch_renderers(crawler, now))
    for identity in list(_RENDERER_SAMPLES):
        if identity not in seen:
            _RENDERER_SAMPLES.pop(identity, None)

    # A busy target must not hold the pool lock, or one wedged browser prevents
    # admission and reclamation for all the healthy browsers in this process.
    browsers = [crawler.crawler_strategy.browser_manager.browser for crawler in crawlers]
    results = await asyncio.gather(
        *(_probe_browser(crawler) for crawler in crawlers), return_exceptions=True
    )
    for crawler, browser, failed in zip(crawlers, browsers, results):
        if isinstance(failed, Exception):
            logger.warning("Browser health probe raised unexpectedly", exc_info=failed)
        elif failed is True:
            await _dispose_unhealthy(crawler, browser)


async def janitor():
    """Adaptive cleanup based on memory pressure."""
    global LAST_JANITOR_PASS
    while True:
        try:
            await _janitor_pass()
        except Exception:  # CancelledError is a BaseException, so shutdown still propagates
            # Never let one bad pass end reclamation. logger.error, not info:
            # the deployed log level drops INFO, which is why the previous
            # silent stall was invisible for 28h.
            logger.error("Janitor pass failed; continuing", exc_info=True)
            await asyncio.sleep(10)
        else:
            LAST_JANITOR_PASS = time.time()
        _LIBC.malloc_trim(0)


async def _janitor_pass():
    """One adaptive cleanup pass."""
    mem_pct = get_container_memory_percent()

    # Adaptive intervals and TTLs
    if mem_pct > 80:
        interval, cold_ttl, hot_ttl = 10, 30, 120
    elif mem_pct > 60:
        interval, cold_ttl, hot_ttl = 30, 60, 300
    else:
        interval, cold_ttl, hot_ttl = 60, BASE_IDLE_TTL, BASE_IDLE_TTL * 2

    # The resource watchdog needs repeated samples inside its sustained-CPU
    # window even when low container memory permits a longer idle TTL.
    await asyncio.sleep(min(interval, WATCHDOG.interval_seconds))
    await _watchdog_pass()

    now = time.time()
    async with LOCK:
        # Both pools, same rule: cold first (less valuable), hot gets the longer TTL.
        for tier, pool, ttl in (("cold", COLD_POOL, cold_ttl), ("hot", HOT_POOL, hot_ttl)):
            for sig in list(pool.keys()):
                if now - LAST_USED.get(sig, now) <= ttl:
                    continue
                crawler = pool[sig]
                idle_time = now - LAST_USED[sig]
                active = getattr(crawler, 'active_requests', 0)

                # Mirror the admission path (_is_live): a browser mid-recycle
                # lies fallow at active_requests==0 while close(_for_recycle=True)
                # -> start() runs in a shielded background task. Evicting it now
                # cancels close() at the shielded recycle await, but the recycle
                # keeps running and start() finishes a fresh driver+Chromium on
                # an already-detached manager with _closing=True stuck - nothing
                # will close it again (a monotonic process leak, the exact class
                # bdccf62/d0f57c0 were written to stop). _recycle_browser's
                # finally clears _recycling within the close/start wait caps
                # (<= 180s), so the next pass re-evaluates and evicts normally.
                if _is_recycling(crawler):
                    LAST_USED[sig] = now
                    continue

                if active > 0:
                    if idle_time <= STALE_CEILING:
                        continue  # still serving requests, skip
                    logger.error(f"🚨 Leaked request counter (sig={sig[:8]}, active={active}, idle={idle_time:.0f}s > {STALE_CEILING}s) - force-closing")
                else:
                    logger.info(f"🧹 Closing {tier} browser (sig={sig[:8]}, idle={idle_time:.0f}s)")
                _close_in_background(_detach_pool_crawler(pool, sig))

                # Track in monitor
                try:
                    from monitor import get_monitor
                    await get_monitor().track_janitor_event(f"close_{tier}", sig, {"idle_seconds": int(idle_time), "ttl": ttl})
                except Exception:  # not bare: that would swallow CancelledError and outlive shutdown
                    pass

        # The permanent browser follows the hot tier's idle rule. Otherwise its
        # only reset is the page-count recycle, which a lightly used replica
        # never reaches, so one Chromium and its renderers live from container
        # start until the memory ceiling refuses new browsers. The next
        # default-config request rebuilds it (see _get_admitted_crawler).
        permanent_idle = now - LAST_USED.get(DEFAULT_CONFIG_SIG, now)
        if (
            PERMANENT is not None
            and permanent_idle > hot_ttl
            and _active_requests(PERMANENT) == 0
            and not _is_recycling(PERMANENT)
        ):
            logger.info(f"🧹 Closing permanent browser (idle={permanent_idle:.0f}s)")
            _close_in_background(_detach_permanent())

        if mem_pct >= RECYCLE_LIMIT:
            await _recycle_busiest_browser()

        # Log pool stats
        if mem_pct > 60:
            logger.info(f"📊 Pool: hot={len(HOT_POOL)}, cold={len(COLD_POOL)}, mem={mem_pct:.1f}%")
