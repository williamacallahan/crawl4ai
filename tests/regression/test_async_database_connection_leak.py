"""``AsyncDatabaseManager.get_connection`` must not leak connections on failure.

Regression coverage for the connection/thread/file-descriptor leak introduced
in commit 0982c639 (error-handling overhaul). That commit added a
schema-verification block (``PRAGMA table_info(crawled_data)`` + a
missing-columns ``ValueError``) that runs *after* ``aiosqlite.connect()`` has
succeeded but *before* the connection is inserted into ``connection_pool``.
The new inner ``except`` only logged and re-raised; it never closed the already
opened connection, and the outer ``finally`` cleanup (``if task_id in
self.connection_pool``) skipped it because pooling happens only after all
setup succeeds. Any exception raised after ``connect()`` succeeded (schema
``ValueError``, PRAGMA failure on a corrupt DB, ``sqlite3.DatabaseError``)
therefore leaked the ``aiosqlite.Connection`` plus its dedicated non-daemon
worker thread, sqlite3 handle, and WAL file descriptors -- one leak per failed
attempt, i.e. 3 per cache operation via ``execute_with_retry``, unbounded and
permanent (``aiosqlite.Connection`` runs a thread that only stops on
``close()``; ``threading`` retains running threads so they are un-GC-able and
also hang interpreter shutdown).

The fix closes the connection in the inner ``except`` before re-raising
(``conn = None`` initialized before the inner ``try`` so a ``connect()`` failure
is also covered).

These tests exercise the real ``AsyncDatabaseManager`` against isolated SQLite
DBs + filesystem content stores under ``tmp_path`` (no mocks of the DB layer).
Each test is fully isolated: nothing touches the real ``~/.crawl4ai`` and tests
can run in parallel. Leak detection is scoped per-manager by intercepting
``aiosqlite.connect`` for the manager's ``db_path`` only, so concurrent
activity by the canonical singleton (other tests) cannot perturb the counts.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_async_database_connection_leak.py
"""

import asyncio
import gc
import os

import aiosqlite
import pytest

from crawl4ai.async_database import AsyncDatabaseManager
from crawl4ai.models import CrawlResult, MarkdownGenerationResult
from crawl4ai.utils import ensure_content_dirs


# ---------------------------------------------------------------------------
# Isolated-manager setup helpers (mirror test_async_database_markdown_roundtrip)
# ---------------------------------------------------------------------------


def _new_manager(tmp_path) -> AsyncDatabaseManager:
    """An isolated, uninitialized ``AsyncDatabaseManager`` under tmp_path."""
    mgr = AsyncDatabaseManager()
    mgr.db_path = str(tmp_path / "test.db")
    mgr.content_paths = ensure_content_dirs(str(tmp_path))
    return mgr


async def _setup_healthy_manager(tmp_path) -> AsyncDatabaseManager:
    """A fully-initialized, isolated manager with a complete schema."""
    mgr = _new_manager(tmp_path)
    await mgr.ainit_db()
    await mgr.update_db_schema()
    mgr._initialized = True
    return mgr


def _manager_missing_columns(tmp_path) -> AsyncDatabaseManager:
    """A manager whose ``crawled_data`` table omits required columns.

    ``_initialized`` is set True so ``get_connection`` skips ``initialize()``
    (which would otherwise repair the schema). The schema-verification block
    in ``get_connection`` raises ``ValueError`` *after* ``aiosqlite.connect()``
    succeeds -- the exact path the bug leaks on.
    """
    import sqlite3

    mgr = _new_manager(tmp_path)
    # Only the PK column; every other expected column is missing.
    conn = sqlite3.connect(mgr.db_path)
    conn.execute("CREATE TABLE crawled_data (url TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    mgr._initialized = True
    return mgr


def _manager_corrupt_db(tmp_path) -> AsyncDatabaseManager:
    """A manager whose db_path is a non-SQLite (corrupt) file.

    ``aiosqlite.connect()`` is lazy and succeeds; the first PRAGMA then runs in
    the worker thread and raises ``sqlite3.DatabaseError`` -- a different
    post-``connect()`` failure point than the schema ``ValueError``.
    """
    mgr = _new_manager(tmp_path)
    with open(mgr.db_path, "wb") as f:
        f.write(b"\x00\x01\x02this is not a sqlite database file")
    mgr._initialized = True
    return mgr


# ---------------------------------------------------------------------------
# Leak detector (scoped to a single manager's db_path)
# ---------------------------------------------------------------------------


class _ConnectionTracker:
    """Records ``aiosqlite.Connection`` objects created for one db_path.

    Patching ``aiosqlite.connect`` with :meth:`patch` returns the *same*
    ``Connection`` object the real factory would (a ``Thread`` subclass that
    supports both ``await conn = aiosqlite.connect(...)`` and
    ``async with aiosqlite.connect(...)``), so call sites are unaffected.
    """

    def __init__(self, db_path):
        self._db_path = os.path.abspath(str(db_path))
        self.created = []
        self._real_connect = aiosqlite.connect

    def patch(self, monkeypatch):
        tracker = self

        def tracking_connect(database, *, iter_chunk_size=64, loop=None, **kwargs):
            conn = tracker._real_connect(
                database, iter_chunk_size=iter_chunk_size, loop=loop, **kwargs
            )
            try:
                if os.path.abspath(str(database)) == tracker._db_path:
                    tracker.created.append(conn)
            except Exception:
                pass
            return conn

        monkeypatch.setattr(aiosqlite, "connect", tracking_connect)

    def alive(self):
        """Connections whose worker thread is still running = leaked.

        A connection that was ``close()``-d has a finished thread
        (``is_alive()`` False); a leaked-but-never-closed connection keeps a
        live non-daemon thread (``is_alive()`` True). ``aiosqlite.connect`` is
        lazy: only connections actually ``await``-ed / ``async with``-ed start
        their thread, so an un-started proxy is ``is_alive()`` False and does
        not get miscounted.
        """
        gc.collect()
        return [c for c in self.created if c.is_alive()]


async def _assert_no_leak(mgr, tracker):
    """No live connection, pool drained, and every semaphore permit released."""
    await asyncio.sleep(0)
    assert tracker.alive() == [], "aiosqlite connection leaked"
    assert mgr.connection_pool == {}, "failed connection must not enter the pool"
    for _ in range(mgr.pool_size):
        await asyncio.wait_for(mgr.connection_semaphore.acquire(), timeout=1.0)
    for _ in range(mgr.pool_size):
        mgr.connection_semaphore.release()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_connection_closes_conn_on_missing_columns(tmp_path, monkeypatch):
    """Schema-verification ``ValueError`` (the canonical trigger from
    0982c639): the connection opened just before the failure is closed."""
    mgr = _manager_missing_columns(tmp_path)
    tracker = _ConnectionTracker(mgr.db_path)
    tracker.patch(monkeypatch)

    with pytest.raises(Exception):
        async with mgr.get_connection() as db:  # noqa: F841
            pass

    await _assert_no_leak(mgr, tracker)


@pytest.mark.asyncio
async def test_get_connection_closes_conn_on_corrupt_db(tmp_path, monkeypatch):
    """A non-SQLite file: ``aiosqlite.connect()`` succeeds, the first PRAGMA
    raises ``sqlite3.DatabaseError``. The leak is not specific to the schema
    ``ValueError`` -- *any* post-``connect`` exception must close the conn."""
    mgr = _manager_corrupt_db(tmp_path)
    tracker = _ConnectionTracker(mgr.db_path)
    tracker.patch(monkeypatch)

    with pytest.raises(Exception):
        async with mgr.get_connection() as db:  # noqa: F841
            pass

    await _assert_no_leak(mgr, tracker)


@pytest.mark.asyncio
async def test_repeated_failing_ops_do_not_accumulate_leaks(tmp_path, monkeypatch):
    """The bug report's long-running symptom: leaks accumulate per op without
    bound (3 per op via ``execute_with_retry``). After the fix, N failing ops
    must leak 0 connections (not 3*N) and must not exhaust the semaphore (which
    would eventually hang the process)."""
    mgr = _manager_missing_columns(tmp_path)
    tracker = _ConnectionTracker(mgr.db_path)
    tracker.patch(monkeypatch)

    async def _no_sleep(_d):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def _op(db):  # noqa: ARG001
        return None

    n_ops = 2 * mgr.pool_size + 3
    for _ in range(n_ops):
        with pytest.raises(Exception):
            await mgr.execute_with_retry(_op)

    # Each of the 3 retry attempts per op opens (and, post-fix, closes) its
    # own connection -- confirm the 3x amplification actually fires, then that
    # none of those connections survive.
    assert len(tracker.created) == n_ops * mgr.max_retries, (
        "each retry attempt must open a connection (the bug's 3x amplification)"
    )
    await _assert_no_leak(mgr, tracker)


@pytest.mark.asyncio
async def test_aget_cached_url_on_broken_schema_returns_none_without_leaking(
    tmp_path, monkeypatch
):
    """``aget_cached_url`` swallows the final exception and returns ``None``
    (cache miss). The bug report notes this keeps the process live while leaks
    accumulate. After the fix: still returns ``None``, but leaks nothing."""
    mgr = _manager_missing_columns(tmp_path)
    tracker = _ConnectionTracker(mgr.db_path)
    tracker.patch(monkeypatch)

    async def _no_sleep(_d):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    for i in range(4):
        result = await mgr.aget_cached_url(f"https://e.com/{i}")
        assert result is None, "broken schema must surface as a cache miss"

    await _assert_no_leak(mgr, tracker)


@pytest.mark.asyncio
async def test_healthy_schema_round_trips_without_leak(tmp_path, monkeypatch):
    """No regression in the happy path: a healthy schema still round-trips a
    row through ``acache_url`` -> ``aget_cached_url`` and the pooled
    connection is released (closed by the outer ``finally``)."""
    mgr = await _setup_healthy_manager(tmp_path)
    tracker = _ConnectionTracker(mgr.db_path)
    tracker.patch(monkeypatch)

    md = MarkdownGenerationResult(
        raw_markdown="# Heading\n\nBody.",
        markdown_with_citations="# Heading\n\nBody. \u20064\u2007",
        references_markdown="## References\n[1] https://example.com",
    )
    result = CrawlResult(
        url="https://e.com/healthy",
        html="<h1>x</h1>",
        success=True,
        markdown=md,
    )
    await mgr.acache_url(result)

    cached = await mgr.aget_cached_url("https://e.com/healthy")
    assert cached is not None, "happy path must still hit the cache"
    assert cached.html == "<h1>x</h1>"
    assert cached.markdown.raw_markdown == "# Heading\n\nBody."

    await _assert_no_leak(mgr, tracker)
