"""Runtime-cached content-hash pointers survive a re-fired migration.

Regression coverage for the bug introduced in commit aa481c8c ("fix(migrations):
persist content pointer provenance", merged via daf5dc5a). That change added a
``crawl4ai_content_pointers`` provenance ledger so a re-run of the one-time
blob->hash ``migrate_database`` could distinguish stored 16-hex hash pointers
from raw legacy 16-char hex content, and to protect against unledgered raw-hex
values it re-introduced a backing-file existence gate
(``legacy_pointer = self._is_content_hash(...) and os.path.exists(backing_file)``)
plus a ``force=not (known or legacy_pointer)`` flag that bypasses
``_store_content``'s idempotency guard. Two gaps made that harmful:

1. The runtime write path (``AsyncDatabaseManager.acache_url``) never inserted
   rows into ``crawl4ai_content_pointers``, so any row cached *after* the
   one-time migration completed was unledgered.
2. With the migration marker lost (ephemeral storage, container rebuild,
   operator cleanup) the one-time migration re-fires; for an unledgered
   16-hex pointer whose backing file is also missing, ``known=False`` and
   ``legacy_pointer=False`` (existence gate), so ``force=True`` bypassed the
   idempotency guard and ``_store_content`` re-hashed the 16-char hash
   *string*, writing a NEW file whose body is that string and updating the
   column to the hash-of-hash. ``aget_cached_url`` then served the 16-char
   hash string as the page's HTML on a cache HIT -- silent, unrecoverable
   cache corruption (exactly the outcome commit 36fd610a had fixed).

Fix (this commit):
  1. ``migrate_database`` classifies pointer shape purely by ``_is_content_hash``
     (no backing-file existence gate), so an unledgered 16-hex pointer with a
     missing backing file is treated as an intact pointer and left alone; the
     missing file self-heals as a cache miss on read.
  2. ``acache_url`` records ``(url, content_type)`` rows in
     ``crawl4ai_content_pointers`` (also created idempotently in ``ainit_db``
     so a fresh install whose one-time migration was a 0-row no-op has the
     table), making the provenance ledger the migration relies on complete.

The existing regression test ``test_migration_rerun_with_missing_content_files_
does_not_corrupt`` masked the bug by running the migration once while the
backing file still existed -- seeding the ledger, so the later re-migration
saw ``known=True`` and short-circuited ``force``. The modified companion file
removes that seeding; this file adds focused coverage for both fixes,
including the exact production reproduction (acache_url -> lost ledger -> lost
file+marker -> re-migrate).

These tests exercise the real ``AsyncDatabaseManager.initialize()`` /
``acache_url`` / ``aget_cached_url`` and ``run_migration`` paths against an
isolated SQLite DB + filesystem content store under ``tmp_path`` (no mocks).

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_db_migration_runtime_pointer_protection.py
"""

import os

import aiosqlite
import pytest

from crawl4ai.async_database import AsyncDatabaseManager, _POINTER_TABLE
from crawl4ai.migrations import run_migration
from crawl4ai.models import CrawlResult, MarkdownGenerationResult
from crawl4ai.utils import VersionManager, ensure_content_dirs, generate_content_hash
import crawl4ai.__version__ as version_module


# ---------------------------------------------------------------------------
# Helpers (mirror tests/regression/test_db_migration_refire_corruption.py)
# ---------------------------------------------------------------------------


def _set_version(monkeypatch, v):
    monkeypatch.setattr(version_module, "__version__", v)


def _new_manager(tmp_path) -> AsyncDatabaseManager:
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


def _is_hash(value) -> bool:
    return bool(value) and len(value) == 16 and all(
        c in "0123456789abcdef" for c in value
    )


def _md(raw="# md"):
    return MarkdownGenerationResult(
        raw_markdown=raw, markdown_with_citations="", references_markdown=""
    )


def _result(url, html="<html><body><h1>Real original content</h1></body></html>"):
    return CrawlResult(url=url, html=html, success=True, markdown=_md())


async def _db_column(mgr, url, column):
    async with aiosqlite.connect(mgr.db_path) as db:
        async with db.execute(
            f"SELECT {column} FROM crawled_data WHERE url = ?", (url,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def _ledger_rows(mgr, url):
    async with aiosqlite.connect(mgr.db_path) as db:
        async with db.execute(
            f"SELECT content_type FROM {_POINTER_TABLE} WHERE url = ? ORDER BY content_type",
            (url,),
        ) as cursor:
            return sorted(r[0] for r in await cursor.fetchall())


async def _table_exists(mgr, name):
    async with aiosqlite.connect(mgr.db_path) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ) as cursor:
            return bool(await cursor.fetchone())


# ---------------------------------------------------------------------------
# Fix 2: acache_url records provenance; ledger table exists from ainit_db
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ledger_table_exists_after_initialize_on_fresh_empty_db(
    tmp_path, monkeypatch
):
    """The ledger table must exist even when the one-time migration is a no-op.

    Pre-fix, the table was created only inside ``migrate_database``, which is
    skipped on a 0-row DB (``run_migration`` short-circuits "No rows to
    migrate"). A fresh install then had no ledger table, so ``acache_url``'s
    provenance insert would fail. Post-fix, ``ainit_db`` creates it
    idempotently, so it always exists.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    assert await _table_exists(mgr, _POINTER_TABLE), (
        "ledger table must exist after initialize() on a fresh/empty DB so "
        "acache_url can record provenance"
    )
    assert mgr.version_manager.migration_marker.exists()


@pytest.mark.asyncio
async def test_acache_url_records_pointer_provenance_for_each_nonempty_field(
    tmp_path, monkeypatch
):
    """acache_url must insert a ledger row per non-empty content field.

    Cover for fix 2: without these rows, a re-migration after a lost marker
    would see ``known=False`` for every runtime-cached pointer. Per non-empty
    field, exactly one (url, content_type) row is inserted; empty fields
    (e.g. a CrawlResult with no screenshot) insert nothing. The ledger insert
    is idempotent across re-crawls (INSERT OR IGNORE).
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    url = "https://e.com/provenance"
    # html + cleaned_html(=result.html by default? no: cleaned empty) + markdown
    # are non-empty; screenshot/extracted empty here.
    result = CrawlResult(
        url=url,
        html="<html><body><h1>x</h1></body></html>",
        cleaned_html="<h1>x</h1>",
        success=True,
        markdown=_md("# md"),
        extracted_content="",
        screenshot="",
    )
    await mgr.acache_url(result)

    rows = await _ledger_rows(mgr, url)
    # html, cleaned, markdown stored -> ledger; extracted + screenshots empty
    # -> no rows.
    assert rows == ["cleaned", "html", "markdown"], (
        f"expected ledger rows for non-empty content fields, got {rows!r}"
    )

    # Re-crawl: idempotent (no duplicate rows; INSERT OR IGNORE).
    await mgr.acache_url(result)
    assert await _ledger_rows(mgr, url) == ["cleaned", "html", "markdown"]


@pytest.mark.asyncio
async def test_acache_url_ledger_row_survives_remigrate_with_missing_file(
    tmp_path, monkeypatch
):
    """A ledger-backed pointer + missing file is preserved on re-migration.

    Fix 2 path: ``acache_url`` seeded the ledger, so on a later re-migration
    (marker lost) ``known=True`` short-circuits ``force`` and the pointer is
    left intact, with the missing file self-healing as a cache miss.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    url = "https://e.com/ledger-preserved"
    original_html = "<html><body><h1>Real original content</h1></body></html>"
    await mgr.acache_url(_result(url, html=original_html))

    hash_before = await _db_column(mgr, url, "html")
    assert _is_hash(hash_before)
    assert await _ledger_rows(mgr, url)  # fix 2 seeded the ledger

    html_file = os.path.join(mgr.content_paths["html"], hash_before)
    os.remove(html_file)
    os.remove(mgr.version_manager.migration_marker)
    assert mgr.version_manager.needs_migration() is True

    await run_migration(mgr.db_path)

    assert await _db_column(mgr, url, "html") == hash_before, (
        "ledger-backed pointer was re-hashed when its backing file was missing"
    )
    assert not os.path.exists(html_file), "a hash-of-hash file was (re)created"
    assert await mgr.aget_cached_url(url) is None, (
        "cache served content instead of a self-healing miss"
    )


# ---------------------------------------------------------------------------
# Fix 1: unledgered 16-hex pointer with missing file is NOT re-hashed
# (the exact production reproduction from the bug report)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_cached_pointer_missing_file_no_ledger_not_corrupted(
    tmp_path, monkeypatch
):
    """The exact production reproduction: lost ledger + lost file + re-migrate.

    1. ``acache_url`` writes a 16-hex pointer + backing file.
    2. Simulate a PRE-fix runtime-cached row (or a ledger that was lost with
       the content dir): delete the ledger row for this url (known=False).
    3. Lose the backing file + the marker.
    4. Re-run the migration (production entry point ``run_migration``).

    Before the fix: ``known=False`` + ``legacy_pointer=False`` (existence
    gate) -> ``force=True`` -> re-hash -> column becomes a hash-of-hash, a new
    file whose body is the old hash string is written, and ``aget_cached_url``
    serves that hash string as the page's HTML. After the fix: shape-based
    classification leaves the pointer intact; the missing file self-heals as a
    cache miss.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    url = "https://e.com/unledgered-repro"
    original_html = "<html><body><h1>Real original content</h1></body></html>"
    await mgr.acache_url(_result(url, html=original_html))

    hash_before = await _db_column(mgr, url, "html")
    assert _is_hash(hash_before)
    html_file = os.path.join(mgr.content_paths["html"], hash_before)
    assert os.path.exists(html_file)

    # Strip provenance: simulate a pre-fix runtime-cached row (acache_url
    # never recorded it) or a ledger lost with the content dir.
    async with aiosqlite.connect(mgr.db_path) as db:
        await db.execute(f"DELETE FROM {_POINTER_TABLE} WHERE url = ?", (url,))
        await db.commit()
    assert await _ledger_rows(mgr, url) == []

    # Lose the backing file + the marker (DB survives).
    os.remove(html_file)
    os.remove(mgr.version_manager.migration_marker)
    assert mgr.version_manager.needs_migration() is True

    await run_migration(mgr.db_path)

    # 1. Pointer NOT re-hashed into a hash-of-hash.
    hash_after = await _db_column(mgr, url, "html")
    assert hash_after == hash_before, (
        f"unledgered pointer re-hashed {hash_before} -> {hash_after} when its "
        f"backing file was missing (the silent-corruption bug)"
    )
    # 2. No hash-of-hash backing file whose body is the old hash string.
    assert not os.path.exists(os.path.join(mgr.content_paths["html"], hash_after)), (
        "a backing file was (re)created for an intact pointer; the migration "
        "re-hashed the 16-char hash string"
    )
    # 3. Cache self-heals as a MISS, not a HIT serving the hash string.
    cached = await mgr.aget_cached_url(url)
    assert cached is None, (
        f"cache returned {getattr(cached, 'html', cached)!r} instead of a "
        f"self-healing miss; the hash string was served as page HTML"
    )
    # 4. Re-crawling restores a healthy row (self-heal completes end-to-end).
    await mgr.acache_url(_result(url, html=original_html))
    cached_healed = await mgr.aget_cached_url(url)
    assert cached_healed is not None and cached_healed.html == original_html


@pytest.mark.asyncio
async def test_unledgered_pointer_with_file_present_preserved_on_remigrate(
    tmp_path, monkeypatch
):
    """Control case: unledgered 16-hex pointer WITH its backing file is preserved.

    The marker is lost but the content dir survived -> re-migration runs. The
    pointer (unledgered or not) must stay intact and the cache stays a HIT.
    Confirms the fix does not regress the happy "lost marker only" path.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    url = "https://e.com/file-present"
    original_html = "<html><body><h1>Real original content</h1></body></html>"
    await mgr.acache_url(_result(url, html=original_html))
    hash_before = await _db_column(mgr, url, "html")
    html_file = os.path.join(mgr.content_paths["html"], hash_before)

    # Strip the ledger so the result depends on shape, not known=True.
    async with aiosqlite.connect(mgr.db_path) as db:
        await db.execute(f"DELETE FROM {_POINTER_TABLE} WHERE url = ?", (url,))
        await db.commit()

    os.remove(mgr.version_manager.migration_marker)
    assert mgr.version_manager.needs_migration() is True

    await run_migration(mgr.db_path)

    assert await _db_column(mgr, url, "html") == hash_before
    with open(html_file) as f:
        assert f.read() == original_html
    cached = await mgr.aget_cached_url(url)
    assert cached is not None and cached.html == original_html


# ---------------------------------------------------------------------------
# No regression: normal cache round-trip through the production paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_cache_roundtrip_still_works(tmp_path, monkeypatch):
    """acache_url -> aget_cached_url round-trips intact after the fix."""
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.initialize()
    mgr._initialized = True
    result = _result("https://e.com/roundtrip")
    await mgr.acache_url(result)
    cached = await mgr.aget_cached_url("https://e.com/roundtrip")
    assert cached is not None
    assert (
        cached.html == "<html><body><h1>Real original content</h1></body></html>"
    )
    assert cached.url == "https://e.com/roundtrip"
    assert cached.success is True
    assert cached.markdown.raw_markdown == "# md"
    # Fix 2 also recorded provenance for this row.
    assert "html" in await _ledger_rows(mgr, "https://e.com/roundtrip")


@pytest.mark.asyncio
async def test_raw_non_hex_legacy_content_still_migrated(tmp_path, monkeypatch):
    """Raw legacy content that is NOT 16-hex is still migrated (re-hashed).

    Confirms the fix narrowly targets the 16-hex-pointer-shaped case and does
    not turn the one-time migration into a no-op for ordinary raw blobs.
    """
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.ainit_db()
    await mgr.update_db_schema()
    url = "https://e.com/raw-blob"
    raw_html = "<html><body><p>raw blob content awaiting migration</p></body></html>"
    async with aiosqlite.connect(mgr.db_path) as db:
        await db.execute(
            "INSERT INTO crawled_data (url, html, cleaned_html, markdown, "
            "extracted_content, screenshot, success, downloaded_files) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (url, raw_html, "", "", "", "", 1, "[]"),
        )
        await db.commit()

    await run_migration(mgr.db_path)

    stored = await _db_column(mgr, url, "html")
    assert _is_hash(stored), f"raw blob should be migrated to a hash, got {stored!r}"
    assert stored == generate_content_hash(raw_html)
    with open(os.path.join(mgr.content_paths["html"], stored)) as f:
        assert f.read() == raw_html
    cached = await mgr.aget_cached_url(url)
    assert cached is not None and cached.html == raw_html
