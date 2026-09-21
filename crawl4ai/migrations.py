import os
import asyncio
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
        xxh64 hex string (16 chars) naming an existing file under
        ``content_paths[content_type]``. Re-running the migration must detect
        this and leave the pointer intact, otherwise it re-hashes the hash
        string, updates the column to the new (hash-of-hash) pointer and
        orphans the original content file on disk. Implementation of the
        recommended fix: "skip rows whose column value already names an
        existing ``*_content/<hash>`` file".
        """
        if not content or len(content) != 16:
            return False
        # xxh64 hexdigest is 16 hex chars; reject non-hex strings of that
        # length so a 16-char raw blob that happens to exist as a file name
        # under a different content dir is not misclassified.
        try:
            int(content, 16)
        except (TypeError, ValueError):
            return False
        return os.path.exists(os.path.join(self.content_paths[content_type], content))

    async def _store_content(self, content: str, content_type: str) -> str:
        if not content:
            return ""

        # Idempotency guard: if the column already holds a content-hash
        # pointer (the migration has already run on this row), keep it as-is
        # instead of re-hashing the hash string. This makes ``migrate_database``
        # safe to re-run after a partial failure or a stale marker.
        if self._is_content_hash(content, content_type):
            return content

        content_hash = self._generate_content_hash(content)
        file_path = os.path.join(self.content_paths[content_type], content_hash)

        if not os.path.exists(file_path):
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

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
                    html_hash = await self._store_content(html, "html")
                    cleaned_hash = await self._store_content(cleaned_html, "cleaned")
                    markdown_hash = await self._store_content(markdown, "markdown")
                    extracted_hash = await self._store_content(
                        extracted_content, "extracted"
                    )
                    screenshot_hash = await self._store_content(
                        screenshot, "screenshots"
                    )

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
    content-hash pointer (naming an existing ``*_content/<hash>`` file) are
    skipped by ``DatabaseMigration._store_content``, so a re-run after a
    partial failure or a stale marker does not re-hash already-migrated
    content.
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
