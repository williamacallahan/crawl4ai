"""Atomic content-file writes during the blob->hash migration.

Regression coverage for the bug introduced in commit d0014c67 ("New async
database manager and migration support"). ``DatabaseMigration._store_content``
in ``crawl4ai/migrations.py`` wrote content files with the non-atomic pattern::

    if not os.path.exists(file_path):
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

``aiofiles.open(file_path, "w")`` creates/truncates ``file_path`` *before* the
awaited ``f.write(content)`` completes. A failed/interrupted write (ENOSPC,
OOM-kill, SIGKILL, power loss) therefore leaves an EMPTY or PARTIAL file at
``file_path``. The migration run that hit the failure rolls back its DB
transaction (the column stays a raw blob), but the partial file persists on
disk. On the documented partial-failure recovery re-run -- which the docstring
of ``run_migration`` and the comments at ``crawl4ai/async_database.py:77-88``
explicitly promise is "safe to re-run after a partial failure or a stale
marker" -- the same blob re-hashes to the SAME hash; ``_store_content`` sees
``os.path.exists(file_path)`` is True and *skips* re-writing, then commits the
column as a hash pointer to that empty/partial file. ``aget_cached_url`` then
loads ``""`` or a truncated string (not ``None``), so the cache is treated as a
hit and the loss is silent. The ``html`` field is re-validated by ``arun`` (a
fresh crawl is triggered only when ``html`` is empty, per bug_033), but
``markdown`` / ``extracted_content`` / ``screenshot`` are served as-is.

Fix: write atomically. Stage the content to a ``tempfile.mkstemp`` temp file in
the same directory, then ``os.replace(tmp, file_path)`` only after the full
write succeeds. A failed write leaves the temp file behind but never
creates/truncates ``file_path``, so the ``os.path.exists`` dedup check stays
sound: a present ``file_path`` always holds complete content, and a recovery
re-run re-attempts the write instead of skipping it and committing a pointer
to an empty/partial file.

These tests exercise the real ``AsyncDatabaseManager.initialize()`` ->
``run_migration()`` -> ``DatabaseMigration._store_content()`` path against an
isolated SQLite DB + filesystem content store under ``tmp_path`` (no mocks of
the DB layer). I/O failures are injected by wrapping ``aiofiles.open`` so that
the first matching content WRITE opens the destination (creating/truncating
it, exactly as a real ``open('w')`` does) but makes the subsequent ``await
f.write(content)`` raise ``OSError`` -- the same shape as a real ENOSPC during
write. This lets a single test demonstrate BOTH the buggy behavior (a 0-byte
file left at ``file_path``) AND the fixed behavior (no file left at
``file_path``, recovery re-run writes correctly), without depending on a real
full disk.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \\
        tests/regression/test_migration_failed_write_data_loss.py
"""

import os

import aiofiles
import aiosqlite
import pytest

from crawl4ai.async_database import AsyncDatabaseManager
from crawl4ai.migrations import DatabaseMigration
from crawl4ai.models import CrawlResult, MarkdownGenerationResult
from crawl4ai.utils import VersionManager, ensure_content_dirs, generate_content_hash
import crawl4ai.__version__ as version_module


# ---------------------------------------------------------------------------
# Helpers (mirror tests/regression/test_db_migration_refire_corruption.py)
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


def _is_hash(value) -> bool:
    return bool(value) and len(value) == 16 and all(
        c in "0123456789abcdef" for c in value
    )


async def _db_column(mgr, url, column):
    """Read a raw column value directly from the DB (bypassing _load_content)."""
    async with aiosqlite.connect(mgr.db_path) as db:
        async with db.execute(
            f"SELECT {column} FROM crawled_data WHERE url = ?", (url,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


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


def _files_in(directory):
    """List all files in ``directory`` (non-recursive)."""
    return sorted(
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    )


# ---------------------------------------------------------------------------
# I/O-failure injection: wrap aiofiles.open so the first matching content WRITE
# opens the destination (creating/truncating it, exactly like a real open('w'))
# but makes the subsequent await f.write(content) raise OSError. This is the
# same failure shape as a real ENOSPC during write -- the file at the open
# target path is created/truncated to 0 bytes before the write raises -- and
# works against BOTH the buggy code (which opens ``file_path`` directly) and
# the fixed code (which opens a temp file then os.replace's it onto
# ``file_path``).
# ---------------------------------------------------------------------------


class _FailingWriteHandle:
    """Async context manager that opens the real file (creating/truncating it)
    then raises OSError from ``write`` -- mimicking a real ENOSPC mid-write."""

    def __init__(self, cm):
        self._cm = cm
        self._fh = None

    async def __aenter__(self):
        # Opening with mode "w" creates/truncates the target path BEFORE write
        # -- this is exactly the behavior the bug relies on.
        self._fh = await self._cm.__aenter__()
        return self

    async def write(self, data):
        raise OSError("simulated disk-full / interrupted write")

    async def __aexit__(self, exc_type, exc, tb):
        return await self._cm.__aexit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._fh, name)


def _install_failing_write(monkeypatch, target_dir, *, fire_on=1):
    """Patch ``aiofiles.open`` (as seen by ``crawl4ai.migrations``) via the
    ``monkeypatch`` fixture (auto-restored at test teardown) so the
    ``fire_on``-th write-mode open whose target lives under ``target_dir``
    opens the real file then fails the write with OSError.

    Returns ``(state, disable)`` where ``state["fired"]`` records whether the
    injection matched and ``disable()`` flips the wrapper to a passthrough for
    a subsequent recovery re-run -- WITHOUT calling ``monkeypatch.undo()``
    (which would also undo any ``_set_version`` patch on the same fixture).
    """
    import crawl4ai.migrations as migrations_module

    real_open = migrations_module.aiofiles.open
    state = {"fired": False, "calls": 0, "enabled": True}

    def wrapped(file, mode="r", *args, **kwargs):
        matches = (
            state["enabled"]
            and "w" in mode
            and isinstance(file, (str, os.PathLike))
            and os.path.abspath(str(file)).startswith(
                os.path.abspath(str(target_dir)) + os.sep
            )
        )
        cm = real_open(file, mode, *args, **kwargs)
        if matches:
            state["calls"] += 1
            if state["calls"] == fire_on and not state["fired"]:
                state["fired"] = True
                return _FailingWriteHandle(cm)
        return cm

    monkeypatch.setattr(migrations_module.aiofiles, "open", wrapped)

    def disable():
        state["enabled"] = False

    return state, disable


# ---------------------------------------------------------------------------
# Unit-level: a failed write leaves NO file at the hash path (the core fix).
# Footgun this guards: someone reverts the atomic write back to
# ``aiofiles.open(file_path, "w") + f.write``, which truncates file_path before
# the awaited write and leaves a 0-byte file on failure -- the recovery re-run
# then skips re-writing (os.path.exists is True) and commits a pointer to it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_write_leaves_no_file_at_hash_path(tmp_path, monkeypatch):
    dm = DatabaseMigration(tmp_path / "test.db")
    markdown_dir = dm.content_paths["markdown"]

    payload = "ORIGINAL MARKDOWN CONTENT THAT MUST SURVIVE"
    expected_hash = generate_content_hash(payload)
    file_path = os.path.join(markdown_dir, expected_hash)

    assert not os.path.exists(file_path), "precondition: hash path clean"

    state, disable = _install_failing_write(monkeypatch, markdown_dir)

    with pytest.raises(OSError):
        await dm._store_content(payload, "markdown")
    assert state["fired"], "injection never fired -- test is vacuous"

    # THE FIX GUARANTEE: no file at the hash path (the temp staging file was
    # unlinked by the except clause; file_path was never created/truncated).
    assert not os.path.exists(file_path), (
        "BUG: a failed write left a file at the hash path; the recovery "
        "re-run would skip re-writing and commit a pointer to it"
    )
    # And no leftover temp files in the markdown content dir.
    assert _files_in(markdown_dir) == [], (
        f"leftover temp files in markdown dir: {_files_in(markdown_dir)}"
    )

    # After disabling the injection, a normal write produces the file intact.
    disable()
    h = await dm._store_content(payload, "markdown")
    assert h == expected_hash
    with open(file_path, "r", encoding="utf-8") as f:
        assert f.read() == payload
    assert _files_in(markdown_dir) == [expected_hash]


# ---------------------------------------------------------------------------
# No-regression: the ``os.path.exists`` dedup-skip optimization must still
# work when a complete file already exists.
# Footgun this guards: someone "simplifies" the atomic write by always writing
# (removing the ``if not os.path.exists(file_path)`` guard), which would break
# the space-dedup that the original guard was added for.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_skip_still_works_when_complete_file_exists(tmp_path):
    dm = DatabaseMigration(tmp_path / "test.db")
    markdown_dir = dm.content_paths["markdown"]
    payload = "duplicate-eligible content"
    h = await dm._store_content(payload, "markdown")
    file_path = os.path.join(markdown_dir, h)
    first_mtime = os.path.getmtime(file_path)

    # Second call with the same content: must skip writing (dedup).
    h2 = await dm._store_content(payload, "markdown")
    assert h2 == h, "same content must hash to the same key"
    assert os.path.getmtime(file_path) == first_mtime, (
        "dedup-skip regressed: the existing file was re-written"
    )
    with open(file_path, "r", encoding="utf-8") as f:
        assert f.read() == payload
    assert _files_in(markdown_dir) == [h]


# ---------------------------------------------------------------------------
# End-to-end via AsyncDatabaseManager.initialize() -- the bug report's
# failing scenario. Footgun this guards: the exact bug regresses in the full
# initialize() -> run_migration() -> migrate_database() -> _store_content()
# path (e.g. someone re-introduces a non-atomic write, or breaks the rollback
# interaction so the partial file survives the recovery re-run).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_rerun_preserves_content_after_failed_write(
    tmp_path, monkeypatch
):
    """A failed markdown write during migration must NOT leave a file at the
    hash path; the documented partial-failure recovery re-run (marker missing
    -> re-migrate) must then write the file correctly and the markdown must
    survive end-to-end via ``aget_cached_url`` (not be served as ``""``)."""
    _set_version(monkeypatch, "0.9.3")
    mgr = _new_manager(tmp_path)
    await mgr.ainit_db()
    await mgr.update_db_schema()
    url = "https://e.com/recovery"
    raw_html = "<html><body><h1>real html</h1></body></html>"
    raw_markdown = "ORIGINAL MARKDOWN CONTENT THAT MUST SURVIVE"
    await _insert_raw_blob_row(mgr, url, html=raw_html, markdown=raw_markdown)

    # --- Phase 1: fail the first markdown content write ---
    markdown_dir = mgr.content_paths["markdown"]
    state, disable = _install_failing_write(monkeypatch, markdown_dir)

    with pytest.raises(OSError, match="simulated disk-full"):
        await mgr.initialize()
    assert state["fired"], "injection never fired -- test is vacuous"
    assert not mgr.version_manager.migration_marker.exists(), (
        "marker must not be written when the migration raised mid-run"
    )

    # The DB transaction rolled back: markdown is still the raw blob.
    md_col = await _db_column(mgr, url, "markdown")
    assert md_col == raw_markdown, (
        f"markdown column should still be the raw blob after rollback, got {md_col!r}"
    )

    # THE FIX: no file at the markdown hash path (the buggy code left a 0-byte
    # file here, which corrupted the recovery re-run).
    expected_md_hash = generate_content_hash(raw_markdown)
    md_hash_path = os.path.join(markdown_dir, expected_md_hash)
    assert not os.path.exists(md_hash_path), (
        "BUG: failed migration left a file at the markdown hash path; the "
        "recovery re-run will skip re-writing and commit a pointer to it"
    )

    # --- Phase 2: documented partial-failure recovery re-run ---
    # Disable ONLY the I/O injection (keep the version monkeypatch in place)
    # so the recovery re-run uses the real aiofiles.open and writes normally.
    disable()

    mgr2 = _new_manager(tmp_path)  # same tmp_path -> same DB, same marker state
    await mgr2.initialize()
    mgr2._initialized = True

    assert mgr2.version_manager.migration_marker.exists(), (
        "marker must be written after the successful recovery re-run"
    )

    md_col = await _db_column(mgr2, url, "markdown")
    assert _is_hash(md_col), (
        f"markdown column should be a hash pointer after recovery, got {md_col!r}"
    )
    assert md_col == expected_md_hash, (
        f"same blob must re-hash to the same key: {md_col} != {expected_md_hash}"
    )

    # THE FIX GUARANTEE: the file the hash points at holds the ORIGINAL
    # content, not empty/partial.
    with open(os.path.join(markdown_dir, md_col), "r", encoding="utf-8") as f:
        file_content = f.read()
    assert file_content == raw_markdown, (
        f"DATA LOSS: markdown file content is {file_content!r}, expected {raw_markdown!r}"
    )

    # End-to-end: aget_cached_url returns the ORIGINAL markdown, not "".
    cached = await mgr2.aget_cached_url(url)
    assert cached is not None, "recovered row must be a cache hit"
    assert cached.html == raw_html
    assert cached.markdown.raw_markdown == raw_markdown, (
        f"DATA LOSS: cached.markdown.raw_markdown is "
        f"{cached.markdown.raw_markdown!r}, expected {raw_markdown!r}"
    )
