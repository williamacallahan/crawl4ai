# crawler_pool.py - Smart browser pool with tiered management
import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import suppress
from typing import Dict, Optional

from utils import get_container_memory_percent, load_config

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
BASE_IDLE_TTL = CONFIG.get("crawler", {}).get("pool", {}).get("idle_ttl_sec", 300)
MAX_ACTIVE_REQUESTS = CONFIG.get("crawler", {}).get("pool", {}).get("max_pages", 30)
MAX_BROWSER_INSTANCES = CONFIG.get("crawler", {}).get("pool", {}).get(
    "max_browser_instances", MAX_ACTIVE_REQUESTS
)
ADMISSION_SEM = asyncio.Semaphore(MAX_ACTIVE_REQUESTS)
DEFAULT_CONFIG_SIG = None  # Cached sig for default config
_CLOSE_TASKS = set()  # In-flight background closes; close_all() drains them at shutdown


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


def _is_live(crawler: AsyncWebCrawler) -> bool:
    """Return whether the crawler still owns a usable Playwright browser."""
    try:
        manager = crawler.crawler_strategy.browser_manager
        if manager.browser is not None:
            return manager.browser.is_connected()
        return (
            manager.default_context is not None
            and not manager.default_context.is_closed()
        )
    except Exception:
        return False


async def _close_even_if_cancelled(crawler: AsyncWebCrawler) -> None:
    """Finish browser close before propagating caller cancellation."""
    close_task = asyncio.create_task(crawler.close())
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await close_task
        raise


async def _start_or_close(crawler: AsyncWebCrawler, tier: str) -> None:
    try:
        await crawler.start()
    except BaseException:
        try:
            await _close_even_if_cancelled(crawler)
        except Exception:
            logger.warning(
                "%s browser cleanup failed after start error", tier, exc_info=True
            )
        raise


async def _discard_if_unavailable(
    pool: Dict[str, AsyncWebCrawler], sig: str, tier: str
) -> bool:
    crawler = pool[sig]
    if _is_live(crawler):
        return False
    logger.warning(f"{tier} pool browser is unavailable; replacing it (sig={sig[:8]})")
    pool.pop(sig)
    with suppress(Exception):
        await _close_even_if_cancelled(crawler)
    LAST_USED.pop(sig, None)
    USAGE_COUNT.pop(sig, None)
    return True


async def _make_browser_capacity() -> None:
    """Evict one idle browser before admitting a new configuration."""
    browser_count = (1 if PERMANENT else 0) + len(HOT_POOL) + len(COLD_POOL)
    if browser_count < MAX_BROWSER_INSTANCES:
        return

    idle_browser = [
        (LAST_USED.get(sig, 0), sig, pool)
        for pool in (COLD_POOL, HOT_POOL)
        for sig, crawler in pool.items()
        if getattr(crawler, "active_requests", 0) == 0
    ]
    if not idle_browser:
        raise RuntimeError("Crawler browser pool is at capacity")

    _, idle_sig, idle_pool = min(idle_browser, key=lambda candidate: candidate[0])
    crawler = idle_pool.pop(idle_sig)
    try:
        await _close_even_if_cancelled(crawler)
    except Exception:
        logger.warning("Idle browser cleanup failed during replacement", exc_info=True)
    finally:
        LAST_USED.pop(idle_sig, None)
        USAGE_COUNT.pop(idle_sig, None)
    logger.info(f"🧹 Replaced idle browser at pool capacity (sig={idle_sig[:8]})")

async def get_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    """Get crawler from pool with tiered strategy."""
    await ADMISSION_SEM.acquire()
    try:
        return await _get_admitted_crawler(cfg)
    except BaseException:
        ADMISSION_SEM.release()
        raise


async def get_dedicated_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    """Create one request-owned crawler under the canonical pool admission.

    Dedicated hook crawlers are never returned by ``get_crawler``. They are
    temporarily registered in the existing cold-pool capacity registry under
    an unreachable signature so every browser instance is counted while its
    admission lease remains held from creation through disposal.
    """
    await ADMISSION_SEM.acquire()
    crawler: Optional[AsyncWebCrawler] = None
    try:
        async with LOCK:
            mem_pct = get_container_memory_percent()
            if mem_pct >= MEM_LIMIT:
                raise MemoryError(
                    f"Memory at {mem_pct:.1f}%, refusing dedicated browser"
                )
            await _make_browser_capacity()
            crawler = AsyncWebCrawler(config=cfg, thread_safe=False)
            await _start_or_close(crawler, "Dedicated")

            sig = f"dedicated:{uuid.uuid4().hex}"
            setattr(crawler, "_docker_request_owned", True)
            setattr(crawler, "_docker_pool_sig", sig)
            _set_active_requests(crawler, 1)
            COLD_POOL[sig] = crawler
            LAST_USED[sig] = time.time()
            USAGE_COUNT[sig] = 1
            return crawler
    except BaseException:
        ADMISSION_SEM.release()
        raise


async def release_dedicated_crawler(crawler: AsyncWebCrawler) -> None:
    """Close a request-owned crawler, then release its admission lease.

    Cleanup runs in a shielded task so caller cancellation cannot leave a live
    browser registered or leak the shared semaphore permit.
    """
    if getattr(crawler, "_docker_admission_released", False):
        return

    existing_cleanup = getattr(crawler, "_docker_release_task", None)
    if isinstance(existing_cleanup, asyncio.Task):
        try:
            await asyncio.shield(existing_cleanup)
        except asyncio.CancelledError:
            with suppress(Exception):
                await existing_cleanup
            raise
        return

    sig = getattr(crawler, "_docker_pool_sig", None)
    if not isinstance(sig, str) or not sig.startswith("dedicated:"):
        raise RuntimeError("Crawler does not own a dedicated admission lease")

    async def close_and_unregister() -> None:
        async with LOCK:
            try:
                await crawler.close()
            finally:
                if sig and COLD_POOL.get(sig) is crawler:
                    COLD_POOL.pop(sig, None)
                    LAST_USED.pop(sig, None)
                    USAGE_COUNT.pop(sig, None)
                _set_active_requests(crawler, 0)

    cleanup_task = asyncio.create_task(close_and_unregister())
    setattr(crawler, "_docker_release_task", cleanup_task)
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await cleanup_task
        raise
    finally:
        setattr(crawler, "_docker_request_owned", False)
        setattr(crawler, "_docker_pool_sig", None)
        setattr(crawler, "_docker_release_task", None)
        setattr(crawler, "_docker_admission_released", True)
        ADMISSION_SEM.release()


async def _get_admitted_crawler(cfg: BrowserConfig) -> AsyncWebCrawler:
    """Resolve a crawler after request admission has bounded pool growth."""
    sig = _sig(cfg)
    async with LOCK:
        # Check permanent browser for default config
        if PERMANENT and _is_default_config(sig):
            if not _is_live(PERMANENT):
                logger.warning("Permanent browser is unavailable; replacing it")
                await _init_permanent_locked(cfg, force=True)
            LAST_USED[sig] = time.time()
            USAGE_COUNT[sig] = USAGE_COUNT.get(sig, 0) + 1
            _set_active_requests(PERMANENT, _active_requests(PERMANENT) + 1)
            logger.info("🔥 Using permanent browser")
            return PERMANENT

        # Check hot pool
        if sig in HOT_POOL and not await _discard_if_unavailable(HOT_POOL, sig, "Hot"):
            crawler = HOT_POOL[sig]
            LAST_USED[sig] = time.time()
            USAGE_COUNT[sig] = USAGE_COUNT.get(sig, 0) + 1
            active_requests = _active_requests(crawler) + 1
            _set_active_requests(crawler, active_requests)
            logger.info(f"♨️  Using hot pool browser (sig={sig[:8]}, active={active_requests})")
            return crawler

        # Check cold pool (promote to hot if used 3+ times)
        if sig in COLD_POOL and not await _discard_if_unavailable(COLD_POOL, sig, "Cold"):
            LAST_USED[sig] = time.time()
            USAGE_COUNT[sig] = USAGE_COUNT.get(sig, 0) + 1
            crawler = COLD_POOL[sig]
            _set_active_requests(crawler, _active_requests(crawler) + 1)

            if USAGE_COUNT[sig] >= 3:
                logger.info(f"⬆️  Promoting to hot pool (sig={sig[:8]}, count={USAGE_COUNT[sig]})")
                HOT_POOL[sig] = COLD_POOL.pop(sig)

                # Track promotion in monitor
                try:
                    from monitor import get_monitor
                    await get_monitor().track_janitor_event("promote", sig, {"count": USAGE_COUNT[sig]})
                except Exception:
                    pass

                return HOT_POOL[sig]

            logger.info(f"❄️  Using cold pool browser (sig={sig[:8]})")
            return crawler

        # Memory check before creating new
        mem_pct = get_container_memory_percent()
        if mem_pct >= MEM_LIMIT:
            logger.error(f"💥 Memory pressure: {mem_pct:.1f}% >= {MEM_LIMIT}%")
            raise MemoryError(f"Memory at {mem_pct:.1f}%, refusing new browser")

        # Create new in cold pool
        await _make_browser_capacity()
        logger.info(f"🆕 Creating new browser in cold pool (sig={sig[:8]}, mem={mem_pct:.1f}%)")
        crawler = AsyncWebCrawler(config=cfg, thread_safe=False)
        await _start_or_close(crawler, "Pooled")
        _set_active_requests(crawler, 1)
        COLD_POOL[sig] = crawler
        LAST_USED[sig] = time.time()
        USAGE_COUNT[sig] = 1
        return crawler

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
    try:
        _set_active_requests(crawler, max(0, _active_requests(crawler) - 1))
    finally:
        ADMISSION_SEM.release()


async def _init_permanent_locked(cfg: BrowserConfig, *, force: bool = False) -> None:
    global PERMANENT, DEFAULT_CONFIG_SIG
    if PERMANENT and not force and _is_live(PERMANENT):
        return
    if PERMANENT:
        with suppress(Exception):
            await _close_even_if_cancelled(PERMANENT)
        PERMANENT = None
        if DEFAULT_CONFIG_SIG:
            LAST_USED.pop(DEFAULT_CONFIG_SIG, None)
            USAGE_COUNT.pop(DEFAULT_CONFIG_SIG, None)

    sig = _sig(cfg)
    logger.info("🔥 Creating permanent default browser")
    crawler = AsyncWebCrawler(config=cfg, thread_safe=False)
    await _start_or_close(crawler, "Permanent")
    PERMANENT = crawler
    DEFAULT_CONFIG_SIG = sig
    LAST_USED[sig] = time.time()
    USAGE_COUNT[sig] = 0


async def init_permanent(cfg: BrowserConfig, *, force: bool = False) -> None:
    """Initialize or atomically replace the permanent default browser."""
    async with LOCK:
        await _init_permanent_locked(cfg, force=force)


async def close_all():
    """Close all browsers."""
    async with LOCK:
        # Through _close_in_background so one wedged browser can't hang shutdown
        # while holding LOCK; the drain below is the single bounded wait point.
        if PERMANENT:
            _close_in_background(PERMANENT)
        for c in list(HOT_POOL.values()) + list(COLD_POOL.values()):
            _close_in_background(c)
        HOT_POOL.clear()
        COLD_POOL.clear()
        LAST_USED.clear()
        USAGE_COUNT.clear()
    # Drain all closes (janitor's included) so shutdown doesn't destroy live tasks.
    if _CLOSE_TASKS:
        with suppress(Exception):
            await asyncio.wait_for(
                asyncio.gather(*_CLOSE_TASKS, return_exceptions=True), timeout=65
            )

def _close_in_background(crawler: AsyncWebCrawler):
    """Close a browser without holding the pool LOCK.

    close() on a wedged browser can hang; awaiting it under LOCK would freeze
    get_crawler() and every future janitor pass for the whole server.
    """
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
    task = asyncio.create_task(_close())
    _CLOSE_TASKS.add(task)
    task.add_done_callback(_CLOSE_TASKS.discard)

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

    await asyncio.sleep(interval)

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
                if active > 0:
                    if idle_time <= STALE_CEILING:
                        continue  # still serving requests, skip
                    logger.error(f"🚨 Leaked request counter (sig={sig[:8]}, active={active}, idle={idle_time:.0f}s > {STALE_CEILING}s) - force-closing")
                else:
                    logger.info(f"🧹 Closing {tier} browser (sig={sig[:8]}, idle={idle_time:.0f}s)")
                _close_in_background(crawler)  # close() can hang; never await it under LOCK
                pool.pop(sig, None)
                LAST_USED.pop(sig, None)
                USAGE_COUNT.pop(sig, None)

                # Track in monitor
                try:
                    from monitor import get_monitor
                    await get_monitor().track_janitor_event(f"close_{tier}", sig, {"idle_seconds": int(idle_time), "ttl": ttl})
                except Exception:  # not bare: that would swallow CancelledError and outlive shutdown
                    pass

        # Log pool stats
        if mem_pct > 60:
            logger.info(f"📊 Pool: hot={len(HOT_POOL)}, cold={len(COLD_POOL)}, mem={mem_pct:.1f}%")
