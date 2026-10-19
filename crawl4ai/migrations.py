import os
import asyncio
import tempfile
from pathlib import Path
import aiosqlite
from typing import Optional
import xxhash
import aiofiles
import shutil
from datetime import datetime
from .async_logger import AsyncLogger, LogLevel

# Initialize logger
logger = AsyncLogger(log_level=LogLevel.DEBUG, verbose=True)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class DatabaseMigration:
    _POINTER_TABLE = "crawl4ai_content_pointers"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.content_paths = self._ensure_content_dirs(os.path.dirname(db_path))

    def _ensure_content_dirs(self, base_path: str) -> dict:
        dirs = {
            "html": "html_content",
            "cleaned": "cleaned_html",
            "markdown": "markdown_content",
            "extracted": "extracted_content",
            "screenshots": "screenshots",
        }
        content_paths = {}
        for key, dirname in dirs.items():
            path = os.path.join(base_path, dirname)
            os.makedirs(path, exist_ok=True)
            content_paths[key] = path
        return content_paths

    def _generate_content_hash(self, content: str) -> str:
        x = xxhash.xxh64()
        x.update(content.encode())
        content_hash = x.hexdigest()
        return content_hash
        # return hashlib.sha256(content.encode()).hexdigest()

    def _is_content_hash(self, content: str, content_type: str) -> bool:
        """Return True if ``content`` is already a stored content-hash pointer.

        After the blob->hash migration has run on a row, the DB column holds an
        xxh64 hex string (16 chars) naming a file under
        ``content_paths[content_type]``. Re-running the migration must detect
        this and leave the pointer intact, otherwise it re-hashes the hash
        string, updates the column to the new (hash-of-hash) pointer and
        orphans the original content file on disk.

        Classification is based purely on the value's shape (a 16-char hex
        string matching the xxh64 hexdigest format) and deliberately does NOT
        gate on whether the backing ``*_content/<hash>`` file currently
        exists. A row whose file is missing (e.g. the DB survived but the
        ``*_content/`` directory was lost) is still an already-migrated row;
        re-hashing its pointer would write a new file whose body is the old
        hash string and leave the cache serving that hash string as content on
        a hit. The correct recovery for a missing backing file is a
        self-healing cache miss (``aget_cached_url`` returns ``None`` when
        ``_load_content`` cannot find the file), not a destructive re-hash.
        """
        if not content or len(content) != 16:
            return False
        # xxh64 hexdigest is 16 hex chars; reject non-hex strings of that
        # length so a 16-char raw blob is not misclassified as a pointer.
        try:
            int(content, 16)
        except (TypeError, ValueError):
            return False
        # A 16-hex-char value is a hash pointer regardless of whether the
        # backing file currently exists; a missing file self-heals as a cache
        # miss, never via a destructive re-hash into a hash-of-hash pointer.
        return True

    async def _store_content(
        self, content: str, content_type: str, *, force: bool = False
    ) -> str:
        if not content:
            return ""

        # Idempotency guard: if the column already holds a content-hash
        # pointer (the migration has already run on this row), keep it as-is
        # instead of re-hashing the hash string. This makes ``migrate_database``
        # safe to re-run after a partial failure or a stale marker, including
        # when the backing ``*_content/`` file is missing (the row self-heals
        # as a cache miss on read rather than being corrupted into a
        # hash-of-hash pointer whose new file body is the old hash string).
        if not force and self._is_content_hash(content, content_type):
            return content

        content_hash = self._generate_content_hash(content)
        file_path = os.path.join(self.content_paths[content_type], content_hash)

        if not os.path.exists(file_path):
            # Write atomically: stage the content into a temp file in the same
            # directory, then ``os.replace`` it onto ``file_path`` only after
            # the full write succeeds. ``aiofiles.open(file_path, "w")``
            # truncates/creates ``file_path`` *before* the awaited write
            # completes, so a failed/interrupted write (ENOSPC, OOM-kill,
            # power loss) leaves an empty/partial file at ``file_path``; the
            # migration then rolls back, but the partial file persists on
            # disk. On the documented partial-failure recovery re-run, the
            # same blob re-hashes to the same key, the ``os.path.exists``
            # dedup check above sees the leftover file and *skips* writing,
            # and the column is committed as a pointer to an empty/partial
            # file -- silent data loss for cached content. Staging to a temp
            # file and replacing atomically means a failed write leaves the
            # temp file behind but never creates/truncates ``file_path``, so
            # the dedup check stays sound: a present ``file_path`` always
            # holds complete content, and a recovery re-run re-attempts the
            # write instead of skipping it.
            fd, tmp_path = tempfile.mkstemp(dir=self.content_paths[content_type])
            os.close(fd)
            try:
                async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                    await f.write(content)
                os.replace(tmp_path, file_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                raise

        return content_hash

    async def migrate_database(self):
        """Migrate existing database to file-based storage"""
        # logger.info("Starting database migration...")
        logger.info("Starting database migration...", tag="INIT")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get all rows
                async with db.execute(
                    """SELECT url, html, cleaned_html, markdown, 
                       extracted_content, screenshot FROM crawled_data"""
                ) as cursor:
                    rows = await cursor.fetchall()

                await db.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self._POINTER_TABLE} (
                        url TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        PRIMARY KEY (url, content_type)
                    )"""
                )
                async with db.execute(
                    f"SELECT url, content_type FROM {self._POINTER_TABLE}"
                ) as cursor:
                    known_pointers = set(await cursor.fetchall())

                migrated_count = 0
                for row in rows:
                    (
                        url,
                        html,
                        cleaned_html,
                        markdown,
                        extracted_content,
                        screenshot,
                    ) = row

                    # Store content in files and get hashes
                    content_values = {
                        "html": html,
                        "cleaned": cleaned_html,
                        "markdown": markdown,
                        "extracted": extracted_content,
                        "screenshots": screenshot,
                    }
                    stored_hashes = {}
                    for content_type, value in content_values.items():
                        pointer_key = (url, content_type)
                        known = pointer_key in known_pointers
                        # A 16-hex value is an already-migrated content-hash
                        # pointer purely by its shape (see ``_is_content_hash``),
                        # NOT by whether its backing ``*_content/<hash>`` file
                        # currently exists. Re-gating pointer classification on
                        # ``os.path.exists(backing_file)`` would leave any
                        # unledgered 16-hex pointer whose backing file is missing
                        # (a runtime-cached row written by ``acache_url`` that
                        # predates the provenance ledger -- or whose ledger row
                        # was lost -- with its content dir gone) classified as
                        # raw legacy content, force it through ``_store_content``
                        # with ``force=True``, bypass the idempotency guard, and
                        # re-hash the 16-char hash *string* into a destructive
                        # "hash-of-hash" pointer whose new file body is the old
                        # hash string. The cache then serves that hash string as
                        # the page's HTML on a HIT (silent, unrecoverable
                        # corruption). The sound recovery for a missing backing
                        # file is a self-healing cache miss on read
                        # (``aget_cached_url`` returns ``None`` when
                        # ``_load_content`` cannot find the file), not a re-hash.
                        legacy_pointer = self._is_content_hash(value, content_type)
                        stored_hashes[content_type] = await self._store_content(
                            value,
                            content_type,
                            force=not (known or legacy_pointer),
                        )
                        if value and pointer_key not in known_pointers:
                            await db.execute(
                                f"INSERT OR IGNORE INTO {self._POINTER_TABLE} "
                                "(url, content_type) VALUES (?, ?)",
                                pointer_key,
                            )
                            known_pointers.add(pointer_key)

                    html_hash = stored_hashes["html"]
                    cleaned_hash = stored_hashes["cleaned"]
                    markdown_hash = stored_hashes["markdown"]
                    extracted_hash = stored_hashes["extracted"]
                    screenshot_hash = stored_hashes["screenshots"]

                    # Update database with hashes
                    await db.execute(
                        """
                        UPDATE crawled_data 
                        SET html = ?, 
                            cleaned_html = ?,
                            markdown = ?,
                            extracted_content = ?,
                            screenshot = ?
                        WHERE url = ?
                    """,
                        (
                            html_hash,
                            cleaned_hash,
                            markdown_hash,
                            extracted_hash,
                            screenshot_hash,
                            url,
                        ),
                    )

                    migrated_count += 1
                    if migrated_count % 100 == 0:
                        logger.info(f"Migrated {migrated_count} records...", tag="INIT")

                await db.commit()
                logger.success(
                    f"Migration completed. {migrated_count} records processed.",
                    tag="COMPLETE",
                )

        except Exception as e:
            # logger.error(f"Migration failed: {e}")
            logger.error(
                message="Migration failed: {error}",
                tag="ERROR",
                params={"error": str(e)},
            )
            raise e


async def backup_database(db_path: str) -> str:
    """Create backup of existing database"""
    if not os.path.exists(db_path):
        logger.info("No existing database found. Skipping backup.", tag="INIT")
        return None

    # Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"

    try:
        # Wait for any potential write operations to finish
        await asyncio.sleep(1)

        # Create backup
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backup created at: {backup_path}", tag="COMPLETE")
        return backup_path
    except Exception as e:
        # logger.error(f"Backup failed: {e}")
        logger.error(
            message="Migration failed: {error}", tag="ERROR", params={"error": str(e)}
        )
        raise e


async def run_migration(db_path: Optional[str] = None):
    """Run the one-time blob->hash database migration.

    The migration is idempotent: rows whose content columns already hold a
    content-hash pointer (a 16-char xxh64 hex string naming a
    ``*_content/<hash>`` file, whether or not that file currently exists on
    disk) are skipped by ``DatabaseMigration._store_content``, so a re-run
    after a partial failure or a stale marker does not re-hash
    already-migrated content.
    """
    if db_path is None:
        db_path = os.path.join(Path.home(), ".crawl4ai", "crawl4ai.db")

    if not os.path.exists(db_path):
        logger.info("No existing database found. Skipping migration.", tag="INIT")
        return

    # Nothing to migrate on a fresh/empty DB; skip the backup so a first
    # launch (which now passes the live ``db_path`` and therefore always finds
    # the just-created empty DB) does not leave a spurious empty ``.backup_*``
    # file behind. Also short-circuits foreign/legacy DBs without the table.
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='crawled_data'"
            ) as cursor:
                if not await cursor.fetchone():
                    logger.info(
                        "No crawled_data table. Skipping migration.", tag="INIT"
                    )
                    return
            async with db.execute("SELECT COUNT(*) FROM crawled_data") as cursor:
                (row_count,) = await cursor.fetchone()
    except Exception as e:
        logger.error(
            message="Migration pre-check failed: {error}",
            tag="ERROR",
            params={"error": str(e)},
        )
        raise
    if row_count == 0:
        logger.info("No rows to migrate. Skipping migration.", tag="INIT")
        return

    # Create backup first
    backup_path = await backup_database(db_path)
    if not backup_path:
        return

    migration = DatabaseMigration(db_path)
    await migration.migrate_database()


def main():
    """CLI entry point for migration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate Crawl4AI database to file-based storage"
    )
    parser.add_argument("--db-path", help="Custom database path")
    args = parser.parse_args()

    asyncio.run(run_migration(args.db_path))


if __name__ == "__main__":
    main()
