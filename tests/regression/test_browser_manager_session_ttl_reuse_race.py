"""Regression: TTL cleanup of an idle-but-present session must not race a
concurrent ``get_page`` reuse of the same session.

Before the fix, ``_kill_session`` popped ``self.sessions[session_id]`` outside
the per-session lock that ``get_page`` holds, so the pop could land between
``_get_page_impl``'s ``session_id in self.sessions`` membership check and its
bare ``self.sessions[session_id]`` lookup (separated by a yield on the contended
``_recycle_lock``) and raise ``KeyError``, or land during the post-lookup
``page.evaluate("window.stop()")`` yield and trigger the
``RuntimeError("Browser manager restarted during session acquisition")`` guard.

The fix has the kill path take the same per-session lock and re-check expiry
under it, so a pop can never interleave with the reuse window of the same
session, and a session reused after the cleanup snapshot is left alive for the
next idle reap. As defense in depth the reuse path uses ``.get()`` and falls
through to fresh allocation (re-admitting for a consistent token) instead of
raising ``KeyError``.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.browser_manager import BrowserManager

from tests.regression.test_browser_lifecycle_residuals import (  # noqa: F401
    _Context,
    _Page,
    _manager,
)


@pytest.fixture(autouse=True)
def _reset_global_pages():
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None
    BrowserManager._global_pages_lock_loop = None
    yield
    BrowserManager._global_pages_in_use.clear()
    BrowserManager._global_pages_lock = None
    BrowserManager._global_pages_lock_loop = None


async def _drain_background_tasks(manager: BrowserManager) -> None:
    """Run the manager's tracked background cleanup tasks to completion."""
    await manager._drain_background_tasks()


@pytest.mark.asyncio
async def test_ttl_cleanup_does_not_pop_during_get_page_reuse_window():
    """An expired-but-present session must not be reaped while ``get_page`` is
    reusing it; the kill task serializes on the per-session lock."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    # Expired but still present (idle sessions are reaped lazily on the next
    # get_page, not in the background).
    manager.sessions["s1"] = (context, page, time.time() - 3600)
    manager.session_ttl = 1800

    released = []

    async def record_release(p):
        released.append(p)

    manager._release_page_with_context = record_release

    async def admit_stub(page_timeout_ms=None, *, session_id=None):
        return None

    manager._admit_page_acquisition = admit_stub

    # Contend the recycle lock: this is the yield point inside the reuse path
    # where the original bg _kill_session landed its pop.
    await manager._recycle_lock.acquire()
    try:
        task = asyncio.create_task(manager.get_page(CrawlerRunConfig(session_id="s1")))
        # Let _cleanup_expired_sessions spawn the bg _kill_session("s1"); with
        # the fix it must block on the per-session lock held by get_page.
        await asyncio.sleep(0.05)
        assert "s1" in manager.sessions  # not popped during the reuse window
        assert released == []
        manager._recycle_lock.release()
        returned = await asyncio.wait_for(task, timeout=5)
    finally:
        if manager._recycle_lock.locked():
            manager._recycle_lock.release()

    returned_page, returned_context = returned
    assert returned_page is page
    assert returned_context is context

    # The kill task now acquires the per-session lock (released by get_page)
    # and re-checks expiry. last_used was refreshed by the reuse, so it skips.
    await _drain_background_tasks(manager)
    assert "s1" in manager.sessions  # not killed: no longer idle
    assert released == []


@pytest.mark.asyncio
async def test_ttl_reap_still_kills_a_genuinely_idle_session():
    """A session idle past its TTL is still reaped when
    ``_cleanup_expired_sessions`` runs (and the page is released exactly once)."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    manager.sessions["s1"] = (context, page, time.time() - 3600)
    manager.session_ttl = 1800

    released = []

    async def record_release(p):
        released.append(p)

    manager._release_page_with_context = record_release

    manager._cleanup_expired_sessions()
    await _drain_background_tasks(manager)

    assert "s1" not in manager.sessions
    assert released == [page]


@pytest.mark.asyncio
async def test_ttl_reap_skips_a_session_reused_after_snapshot():
    """only_if_expired re-check: a session reused (last_used refreshed) between
    _cleanup_expired_sessions's snapshot and the kill running is left alive."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    manager.sessions["s1"] = (context, page, time.time() - 3600)
    manager.session_ttl = 1800

    released = []

    async def record_release(p):
        released.append(p)

    manager._release_page_with_context = record_release

    # Hold the per-session lock so the bg kill task blocks until we release.
    lock = manager._session_locks.setdefault("s1", asyncio.Lock())
    await lock.acquire()
    try:
        manager._cleanup_expired_sessions()  # spawns _kill_session(only_if_expired=True)
        await asyncio.sleep(0.05)  # let the kill task queue up on the lock
        # Simulate a concurrent get_page refreshing last_used (entry replaced
        # with a fresh context) AFTER the snapshot but BEFORE the kill runs.
        refreshed_context = _Context()
        manager.sessions["s1"] = (refreshed_context, page, time.time())
        lock.release()
        await _drain_background_tasks(manager)
    finally:
        if lock.locked():
            lock.release()

    assert "s1" in manager.sessions
    assert released == []
    assert manager.sessions["s1"][0] is refreshed_context  # refreshed entry kept


@pytest.mark.asyncio
async def test_explicit_kill_session_kills_even_a_freshly_used_session():
    """The public kill_session is unconditional: a session with fresh last_used
    is still killed when the user asks (only_if_expired does not apply)."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    manager.sessions["s1"] = (context, page, time.time())  # not expired
    manager.session_ttl = 1800

    released = []

    async def record_release(p):
        released.append(p)

    manager._release_page_with_context = record_release

    await manager.kill_session("s1")

    assert "s1" not in manager.sessions
    assert released == [page]


@pytest.mark.asyncio
async def test_explicit_kill_session_serializes_with_concurrent_get_page():
    """A kill_session for a session whose get_page is in flight must wait for
    get_page to finish rather than pop the entry under the in-flight request."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    manager.sessions["s1"] = (context, page, time.time())
    manager.session_ttl = 1800

    released = []

    async def record_release(p):
        released.append(p)

    manager._release_page_with_context = record_release

    # Park get_page inside the reuse window (just after the membership check,
    # before page.evaluate) so the kill task gets a window in which to Mis-pop.
    parked, resume = asyncio.Event(), asyncio.Event()

    async def admit_stub(page_timeout_ms=None, *, session_id=None):
        return None

    manager._admit_page_acquisition = admit_stub

    async def parking_note_page_served(*, session_id=None):
        # Real _note_page_served takes the recycle lock; called between the
        # session-last_used refresh and page.evaluate. Park here so the kill
        # task gets a chance to (incorrectly) pop while get_page is in flight.
        async with manager._recycle_lock:
            if manager._closing or manager.pool_recycle_requested:
                raise RuntimeError("Browser manager is closed")
            manager._pages_served += 1
        parked.set()
        await resume.wait()

    manager._note_page_served = parking_note_page_served

    get_task = asyncio.create_task(manager.get_page(CrawlerRunConfig(session_id="s1")))
    await parked.wait()

    kill_task = asyncio.create_task(manager.kill_session("s1"))
    await asyncio.sleep(0.05)

    # get_page is parked inside the reuse body; the kill must be blocked on the
    # per-session lock held by get_page, so the entry and the page survive.
    assert "s1" in manager.sessions
    assert released == []

    resume.set()
    returned = await asyncio.wait_for(get_task, timeout=5)
    await asyncio.wait_for(kill_task, timeout=5)

    # The kill now proceeds (get_page released the per-session lock).
    assert "s1" not in manager.sessions
    assert released == [page]
    returned_page, _ = returned
    assert returned_page is page  # the in-flight get_page returned the live page


@pytest.mark.asyncio
async def test_two_concurrent_kill_session_calls_release_the_page_once():
    """Two concurrent kill_session calls for the same session must release the
    page exactly once (the loser sees the entry already gone and returns)."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    manager.sessions["s1"] = (context, page, time.time())
    manager.session_ttl = 1800

    released = []

    async def record_release(p):
        released.append(p)

    manager._release_page_with_context = record_release

    kill_a = asyncio.create_task(manager.kill_session("s1"))
    kill_b = asyncio.create_task(manager.kill_session("s1"))
    await asyncio.gather(kill_a, kill_b)

    assert "s1" not in manager.sessions
    assert released == [page]


@pytest.mark.asyncio
async def test_reuse_path_falls_through_to_fresh_allocation_when_entry_vanishes():
    """Defense in depth: if a kill path ever bypasses the per-session lock and
    pops between the membership check and the .get() lookup, the reuse path
    re-admits and falls through to a fresh allocation rather than raising
    KeyError."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    manager.sessions["s1"] = (context, page, time.time() - 3600)  # expired
    manager.session_ttl = 1800

    released = []

    async def record_release(p):
        released.append(p)

    manager._release_page_with_context = record_release

    # Simulate a future bug: a kill path that bypasses the per-session lock
    # (the pre-fix _kill_session). The real fix's lock makes this unreachable,
    # but the .get()/.re-admit defense must still hold.
    async def buggy_lockless_kill(session_id, *, only_if_expired=False):
        entry = manager.sessions.pop(session_id, None)
        if entry is None:
            return
        await manager._release_page_with_context(entry[1])

    manager._kill_session = buggy_lockless_kill

    async def admit_stub(page_timeout_ms=None, *, session_id=None):
        # Mirror the real admit: None for an in-sessions entry, real token
        # otherwise. The real admit would block on the recycle_lock the test
        # holds; the stub returns immediately so the pop lands between the
        # reuse-path's membership check and the .get() lookup (i.e. inside
        # line 2411's recycle_lock yield) instead of inside admit.
        if session_id and session_id in manager.sessions:
            return None
        token = object()
        manager._active_acquisitions.add(token)
        return token

    manager._admit_page_acquisition = admit_stub

    async def note_stub(*, session_id=None):
        pass

    manager._note_page_served = note_stub

    await manager._recycle_lock.acquire()
    try:
        task = asyncio.create_task(manager.get_page(CrawlerRunConfig(session_id="s1")))
        await asyncio.sleep(0.05)
        # The buggy kill popped the entry while get_page was suspended.
        assert "s1" not in manager.sessions
        manager._recycle_lock.release()
        returned = await asyncio.wait_for(task, timeout=5)
    finally:
        if manager._recycle_lock.locked():
            manager._recycle_lock.release()

    returned_page, returned_context = returned
    # A fresh page was allocated, not the popped one.
    assert returned_page is not page
    # The fresh session stores the new page.
    assert manager.sessions["s1"][1] is returned_page
    # The popped page was released once by the buggy kill (it is the kill-side
    # responsibility; the reuse path neither returns it nor re-releases it).
    assert released == [page]


@pytest.mark.asyncio
async def test_get_page_returns_normally_when_session_is_not_expired():
    """Regression guard: a non-expired session is reused normally; cleanup does
    not even schedule a kill for it."""
    manager = _manager(BrowserConfig(headless=True, verbose=False))
    page = _Page()
    context = _Context()
    manager.sessions["s1"] = (context, page, time.time())  # fresh
    manager.session_ttl = 1800

    spawned_kills = []
    real_kill = manager._kill_session

    async def counting_kill(session_id, *, only_if_expired=False):
        spawned_kills.append(session_id)
        return await real_kill(session_id, only_if_expired=only_if_expired)

    manager._kill_session = counting_kill
    manager._release_page_with_context = AsyncMock()

    async def admit_stub(page_timeout_ms=None, *, session_id=None):
        return None

    manager._admit_page_acquisition = admit_stub

    returned = await manager.get_page(CrawlerRunConfig(session_id="s1"))
    await _drain_background_tasks(manager)

    returned_page, returned_context = returned
    assert returned_page is page
    assert returned_context is context
    assert spawned_kills == []  # not expired -> no kill scheduled
    assert "s1" in manager.sessions
