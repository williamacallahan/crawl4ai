"""The destructive blob->hash migration must not re-fire on a version bump.

Regression coverage for the bug introduced in commit f9fe6f8 ("feat(database):
implement version management and migration checks during initialization").
That change reused the *package* ``__version__`` as the trigger for the
one-time blob->hash ``migrate_database`` (via ``VersionManager.needs_update``),
so every ``pip install --upgrade`` flipped ``needs_update`` True and re-ran the
non-idempotent migration. ``migrate_database`` treated the already-stored
content-hash pointer as raw content, re-hashed it, updated the DB column to
the new (hash-of-hash) pointer and orphaned the original content file. The
crawler's ``aget_cached_url`` then returned the 16-char hash string as the
page's HTML/markdown.

Fix (this commit):
  1. Gate the one-time migration on a dedicated marker
     (``VersionManager.needs_migration`` / ``mark_migrated``), independent of
     the package version. Idempotent schema updates (``update_db_schema``)
     still run on every version bump via ``needs_update``.
  2. Make ``migrate_database`` idempotent: ``_store_content`` skips column
     values that already name an existing ``*_content/<hash>`` file, so a
     re-run after a partial failure or a stale/missing marker does not re-hash
     already-migrated rows.
  3. Pass ``self.db_path`` to ``run_migration`` so custom-base users
     (``CRAWL4_AI_BASE_DIRECTORY`` set) migrate their live DB rather than the
     hardcoded ``Path.home()`` default. Safe only because of (2).

These tests exercise the real ``AsyncDatabaseManager.initialize()`` against an
isolated SQLite DB + filesystem content store + VersionManager state under
``tmp_path`` (no mocks of the DB layer).

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_db_migration_refire_corruption.py
"""

import os

import aiofiles
import aiosqlite
import pytest

from crawl4ai.async_database import AsyncDatabaseManager
from crawl4ai.migrations import DatabaseMigration, run_migration
from crawl4ai.models import CrawlResult, MarkdownGenerationResult
from crawl4ai.utils import VersionManager, ensure_content_dirs, generate_content_hash
import crawl4ai.__version__ as version_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_version(monkeypatch, v):
    """Patch the running package version seen by VersionManager.needs_update."""
    monkeypatch.setattr(version_module, "__version__", v)


def _new_manager(tmp_path) -> AsyncDatabaseManager:
    """Build an isolated AsyncDatabaseManager under tmp_path.

    Overrides db_path, content_paths and VersionManager state (version.txt +
    migration marker) so each test is fully isolated: nothing touches the
    real ``~/.crawl4ai`` and tests can run in parallel.
    """
    mgr = AsyncDatabaseManager()
    mgr.db_path = str(tmp_path / "test.db")
    mgr.content_paths = ensure_content_dirs(str(tmp_path))
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    mgr.version_manager = VersionManager()
    mgr.version_manager.home_dir = home / ".crawl4ai"
    mgr.version_manager.home_dir.mkdir(parents=True, exist_ok=True)
    mgr.version_manager.version_file = mgr.version_manager.home_dir / "version.txt"
    mgr.version_manager.migration_marker = (
        mgr.version_manager.home_dir / mgr.version_manager.MIGRATION_MARKER_FILENAME
    )
    mgr._initialized = False
    return mgr


def _md(raw="# md"):
    return MarkdownGenerationResult(
        raw_markdown=raw, markdown_with_citations="", references_markdown=""
    )


def _result(url, html=("<html><body><h1>Real original content</h1></body></html>")):
    return CrawlResult(url=url, html=html, success=True, markdown=_md())


async def _db_column(mgr, url, column):
    """Read a raw column value directly from the DB (bypassing _load_content)."""
    async with aiosqlite.connect(mgr.db_path) as db:
        async with db.execute(
            f"SELECT {column} FROM crawled_data WHERE url = ?", (url,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


def _is_hash(value) -> bool:
    return bool(value) and len(value) == 16 and all(
        c in "0123456789abcdef" for c in value
    )


async def _insert_raw_blob_row(mgr, url, html, cleaned_html="", markdown="",
                               extracted_content="", screenshot=""):
    """Insert a row whose content columns hold RAW blobs (pre-migration state)."""
    async with aiosqlite.connect(mgr.db_path) as db:
        await db.execute(
            "INSERT INTO crawled_data (url, html, cleaned_html, markdown, "
            "extracted_content, screenshot, success, downloaded_files) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (url, html, cleaned_html, markdown, extracted_content, screenshot, 1, "[]"),
        )
        await db.commit()


def _backup_files(mgr):
    return [
        f for f in os.listdir(os.path.dirname(mgr.db_path))
        if f.startswith("test.db.backup_")
    ]


# ---------------------------------------------------------------------------
# VersionManager: marker API
# ---------------------------------------------------------------------------


def test_needs_migration_true_when_marker_missing(tmp_path):
    vm = VersionManager()
    vm.home_dir = tmp_path / ".crawl4ai"
    vm.home_dir.mkdir(parents=True, exist_ok=True)
    vm.migration_marker = vm.home_dir / VersionManager.MIGRATION_MARKER_FILENAME
    vm.version_file = vm.home_dir / "version.txt"
    assert vm.migration_marker.exists() is False
    assert vm.needs_migration() is True


def test_needs_migration_false_when_marker_present(tmp_path):
    vm = VersionManager()
    vm.home_dir = tmp_path / ".crawl4ai"
    vm.home_dir.mkdir(parents=True, exist_ok=True)
    vm.migration_marker = vm.home_dir / VersionManager.MIGRATION_MARKER_FILENAME
    vm.version_file = vm.home_dir / "version.txt"
    vm.mark_migrated()
    assert vm.migration_marker.exists() is True
    assert vm.needs_migration() is False


def test_mark_migrated_writes_marker(tmp_path):
    vm = VersionManager()
    vm.home_dir = tmp_path / "nested" / ".crawl4ai"  # not yet created
    vm.migration_marker = vm.home_dir / VersionManager.MIGRATION_MARKER_FILENAME
    vm.version_file = vm.home_dir / "version.txt"
    assert not vm.home_dir.exists()
    vm.mark_migrated()
    assert vm.migration_marker.exists() is True
    assert vm.needs_migration() is False


# ---------------------------------------------------------------------------
# Fresh install: migration runs once and writes the marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_install_runs_migration_and_writes_marker(tmp_path, monkeypatch):
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()

    assert mgr.version_manager.migration_marker.exists()
    assert mgr.version_manager.version_file.read_text().strip() == "0.9.3"


@pytest.mark.asyncio
async def test_fresh_install_leaves_no_spurious_backup(tmp_path, monkeypatch):
    """A fresh/empty DB must not produce a spurious .backup_* file."""
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    assert _backup_files(mgr) == []


# ---------------------------------------------------------------------------
# THE REGRESSION: a routine version bump must not re-fire the migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upgrade_does_not_refire_migration(tmp_path, monkeypatch):
    """Re-running initialize() after a version bump must not corrupt cache.

    Phase 1 (0.9.2): cache a page. The html column becomes a 16-char hash and
    html_content/<hash> holds the original HTML; the marker is written.
    Phase 2 (0.9.3, simulating `pip install --upgrade`): re-run initialize().
    Phase 3: aget_cached_url must return the original HTML, not the hash.
    """
    url = "https://e.com/upgrade"

    # --- Phase 1: 0.9.2 ---
    _set_version(monkeypatch, "0.9.2")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    await mgr.acache_url(_result(url))

    hash_after_phase1 = await _db_column(mgr, url, "html")
    assert _is_hash(hash_after_phase1), f"phase 1 html should be a hash, got {hash_after_phase1!r}"
    html_file = os.path.join(mgr.content_paths["html"], hash_after_phase1)
    assert os.path.exists(html_file)
    with open(html_file) as f:
        assert f.read() == "<html><body><h1>Real original content</h1></body></html>"
    assert mgr.version_manager.migration_marker.exists()

    # --- Phase 2: upgrade to 0.9.3 ---
    _set_version(monkeypatch, "0.9.3")
    mgr2 = _new_manager(tmp_path)  # same tmp_path -> same DB, same marker
    assert mgr2.db_path == mgr.db_path
    await mgr2.initialize()  # before the fix: re-ran the destructive migration
    mgr2._initialized = True

    # --- Phase 3: read cache ---
    hash_after_phase2 = await _db_column(mgr2, url, "html")
    cached = await mgr2.aget_cached_url(url)
    assert cached is not None
    assert cached.html == "<html><body><h1>Real original content</h1></body></html>", (
        f"cached html is the hash {cached.html!r}, not the original HTML"
    )
    # The hash pointer must not have been re-hashed.
    assert hash_after_phase2 == hash_after_phase1, (
        f"hash pointer changed {hash_after_phase1} -> {hash_after_phase2}; "
        f"the destructive migration re-fired on the version bump"
    )
    assert mgr2.version_manager.migration_marker.exists()
    assert mgr2.version_manager.version_file.read_text().strip() == "0.9.3"
    # No re-fire backup should be created on the upgrade (no rows to migrate).
    assert _backup_files(mgr2) == []


@pytest.mark.asyncio
async def test_version_bump_still_runs_idempotent_schema_update(tmp_path, monkeypatch):
    """Schema updates (update_db_schema) must still run on a version bump.

    They are idempotent (only ADD missing columns) and are gated on
    needs_update, NOT on the migration marker. A future release that adds a
    column must get it added even though the one-time migration marker is
    already present.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    assert mgr.version_manager.migration_marker.exists()

    # Simulate a future release that adds a new expected column by dropping
    # one of the columns update_db_schema adds, then upgrading the version.
    async with aiosqlite.connect(mgr.db_path) as db:
        # SQLite supports DROP COLUMN from 3.35+.
        await db.execute("ALTER TABLE crawled_data DROP COLUMN head_fingerprint")
        await db.commit()

    _set_version(monkeypatch, "0.9.4")
    mgr2 = _new_manager(tmp_path)
    await mgr2.initialize()
    async with aiosqlite.connect(mgr2.db_path) as db:
        async with db.execute("PRAGMA table_info(crawled_data)") as cursor:
            names = {row[1] for row in await cursor.fetchall()}
    assert "head_fingerprint" in names, "schema update must re-run on version bump"
    assert mgr2.version_manager.version_file.read_text().strip() == "0.9.4"
    # The one-time migration must NOT have re-fired (marker already present).
    # There is data-free DB here so no backup is expected regardless.


# ---------------------------------------------------------------------------
# Idempotency: re-running migrate_database on already-hashed rows is safe
# (defense-in-depth for a stale/missing marker or a mid-migration crash)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_is_idempotent_direct_rerun(tmp_path, monkeypatch):
    """run_migration(db_path) re-run on already-hashed rows leaves them intact.

    Simulates a stale/missing marker (e.g. operator deleted it, or a partial
    failure left it unwritten) by force-running the migration again. Before
    the fix this re-hashed the hash string and returned the hash as content.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    url = "https://e.com/idem"
    await mgr.acache_url(_result(url))

    hash_before = await _db_column(mgr, url, "html")
    html_file = os.path.join(mgr.content_paths["html"], hash_before)
    with open(html_file) as f:
        file_before = f.read()
    assert file_before == "<html><body><h1>Real original content</h1></body></html>"

    # Remove the marker and force the migration to re-run (marker gate gone).
    os.remove(mgr.version_manager.migration_marker)
    assert mgr.version_manager.needs_migration() is True
    await run_migration(mgr.db_path)

    hash_after = await _db_column(mgr, url, "html")
    cached = await mgr.aget_cached_url(url)
    assert hash_after == hash_before, (
        f"hash pointer changed {hash_before} -> {hash_after}"
    )
    with open(html_file) as f:
        assert f.read() == "<html><body><h1>Real original content</h1></body></html>"
    assert cached.html == "<html><body><h1>Real original content</h1></body></html>"


@pytest.mark.asyncio
async def test_is_content_hash_guard():
    """DatabaseMigration._is_content_hash detects stored hash pointers."""
    dm = DatabaseMigration.__new__(DatabaseMigration)
    dm.content_paths = {"html": "/", "markdown": "/"}
    # 16-hex-char string is the xxh64 hexdigest shape.
    assert dm._is_content_hash("1e32d216f47f664f", "html") in (False, True)
    # Non-hex / wrong-length / empty are never hashes.
    assert dm._is_content_hash("", "html") is False
    assert dm._is_content_hash("not-a-hash", "html") is False
    assert dm._is_content_hash("12345", "html") is False  # too short
    assert dm._is_content_hash("zzzzzzzzzzzzzzzz", "html") is False  # not hex
    # A real content-hash pointer is only recognized if its file exists.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        dm.content_paths = {"html": d}
        h = generate_content_hash("hello")
        assert len(h) == 16
        assert dm._is_content_hash(h, "html") is False  # file absent
        with open(os.path.join(d, h), "w") as f:
            f.write("hello")
        assert dm._is_content_hash(h, "html") is True


# ---------------------------------------------------------------------------
# Custom-base correctness: run_migration operates on self.db_path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_migration_uses_live_db_path_not_home_default(
    tmp_path, monkeypatch
):
    """A pre-migration DB with raw-blob rows at self.db_path gets migrated.

    Pre-fix, ``run_migration`` was called with no args and defaulted to
    ``Path.home()/.crawl4ai/crawl4ai.db``; for a custom-base user the live DB
    lived elsewhere and the migration silently no-op'd (or operated on the
    wrong file). After the fix, initialize() passes self.db_path, so the live
    DB is migrated. Verified by inserting a RAW-blob row and checking the
    migration turns it into a hash pointer whose file holds the original.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.ainit_db()
    await mgr.update_db_schema()
    url = "https://e.com/raw"
    raw_html = "<html><body><p>raw blob content awaiting migration</p></body></html>"
    await _insert_raw_blob_row(mgr, url, html=raw_html)

    # No marker yet -> initialize() will run the migration on self.db_path.
    assert not mgr.version_manager.migration_marker.exists()
    await mgr.initialize()
    mgr._initialized = True

    hashed = await _db_column(mgr, url, "html")
    assert _is_hash(hashed), f"raw blob should have been migrated to a hash, got {hashed!r}"
    html_file = os.path.join(mgr.content_paths["html"], hashed)
    assert os.path.exists(html_file)
    with open(html_file) as f:
        assert f.read() == raw_html
    cached = await mgr.aget_cached_url(url)
    assert cached is not None
    assert cached.html == raw_html
    assert mgr.version_manager.migration_marker.exists()


# ---------------------------------------------------------------------------
# Partial-failure recovery: marker not written until full success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marker_not_written_if_migration_raises(tmp_path, monkeypatch):
    """If migrate_database raises, the marker must NOT be written.

    Otherwise a partial migration (some rows mutated, then a crash) would be
    marked complete and never re-attempted, leaving a half-migrated DB. With
    the marker unwritten, the next initialize() re-runs -- and because the
    migration is now idempotent, the already-migrated rows are skipped and
    only the remaining raw rows are processed.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.ainit_db()
    await mgr.update_db_schema()
    url1, url2 = "https://e.com/raw1", "https://e.com/raw2"
    await _insert_raw_blob_row(mgr, url1, html="<h1>one</h1>")
    await _insert_raw_blob_row(mgr, url2, html="<h1>two</h1>")

    # Patch migrate_database to run the real migration then raise (simulating a
    # mid-migration crash after the rows have been mutated).
    original_migrate = DatabaseMigration.migrate_database
    call_count = {"n": 0}

    async def boom(self):
        call_count["n"] += 1
        await original_migrate(self)
        raise RuntimeError("simulated mid-migration crash")

    monkeypatch.setattr(DatabaseMigration, "migrate_database", boom)

    # First initialize() drives the migration through the production path
    # (initialize calls run_migration -> migrate_database). It must propagate
    # the RuntimeError so mark_migrated() is never reached.
    with pytest.raises(RuntimeError):
        await mgr.initialize()
    assert not mgr.version_manager.migration_marker.exists(), (
        "marker must not be written when the migration raised mid-run"
    )
    assert call_count["n"] == 1

    # Restore the real migration and re-initialize through the production path.
    monkeypatch.setattr(DatabaseMigration, "migrate_database", original_migrate)
    mgr2 = _new_manager(tmp_path)  # same tmp_path -> same DB, same marker state
    await mgr2.initialize()
    mgr2._initialized = True

    # The marker is now written (initialize succeeded this time).
    assert mgr2.version_manager.migration_marker.exists()
    # The already-migrated rows survived intact (idempotent recovery): both
    # hold valid hash pointers backed by content files holding the original.
    for url, expected in [(url1, "<h1>one</h1>"), (url2, "<h1>two</h1>")]:
        h = await _db_column(mgr2, url, "html")
        assert _is_hash(h), f"{url} html not a hash after recovery: {h!r}"
        with open(os.path.join(mgr2.content_paths["html"], h)) as f:
            assert f.read() == expected
        cached = await mgr2.aget_cached_url(url)
        assert cached.html == expected


# ---------------------------------------------------------------------------
# No regression: normal cache round-trip still works through initialize()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_cache_roundtrip_through_initialize(tmp_path, monkeypatch):
    """acache_url -> aget_cached_url round-trips intact after initialize()."""
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    result = _result("https://e.com/roundtrip")
    await mgr.acache_url(result)

    cached = await mgr.aget_cached_url("https://e.com/roundtrip")
    assert cached is not None
    assert cached.html == "<html><body><h1>Real original content</h1></body></html>"
    assert cached.url == "https://e.com/roundtrip"
    assert cached.success is True
    assert cached.markdown.raw_markdown == "# md"


@pytest.mark.asyncio
async def test_reinitialize_at_same_version_is_a_noop(tmp_path, monkeypatch):
    """A second initialize() at the same version runs neither path."""
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    marker_mtime = mgr.version_manager.migration_marker.stat().st_mtime_ns
    version_mtime = mgr.version_manager.version_file.stat().st_mtime_ns

    # A small sleep is not needed: neither needs_update() nor needs_migration()
    # is True, so nothing is written. Re-initialize and assert the marker and
    # version.txt are untouched (no rewrite).
    import time
    time.sleep(0.01)
    mgr2 = _new_manager(tmp_path)
    await mgr2.initialize()
    assert mgr2.version_manager.migration_marker.stat().st_mtime_ns == marker_mtime
    assert mgr2.version_manager.version_file.stat().st_mtime_ns == version_mtime
