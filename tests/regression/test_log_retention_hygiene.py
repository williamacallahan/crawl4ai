"""The Docker defaults stream; opt-in SDK file sinks have hard byte bounds."""

from types import SimpleNamespace

from crawl4ai import AsyncWebCrawler, BrowserConfig
import crawl4ai.async_database as database
from crawl4ai.async_logger import AsyncLogger, AsyncFileLogger


def test_server_logs_stream_and_sdk_files_rotate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PYTHON_ENV", "production")
    monkeypatch.setattr(database, "base_directory", str(tmp_path / ".crawl4ai"))
    mgr = database.AsyncDatabaseManager()
    crawler = AsyncWebCrawler(base_directory=str(tmp_path), config=BrowserConfig(verbose=True),
                              crawler_strategy=SimpleNamespace())
    mgr.logger.info("database stream probe")
    crawler.logger.info("crawler stream probe")
    assert not list(tmp_path.rglob("*.log"))
    stderr = capsys.readouterr().err
    assert "database stream probe" in stderr and "crawler stream probe" in stderr

    monkeypatch.delenv("PYTHON_ENV")
    for name, logger_type in (("sdk", AsyncLogger), ("file", AsyncFileLogger)):
        path = tmp_path / f"{name}.log"
        options = {"verbose": False} if logger_type is AsyncLogger else {}
        logger = logger_type(log_file=str(path), max_log_bytes=512, log_backup_count=2, **options)
        for index in range(20):
            logger.info(f"record {index}: " + "é" * 120)
        logger.info("oversized " + "é" * 2048)
        logger.info("latest marker")
        files = list(tmp_path.glob(f"{name}.log*"))
        assert 1 < len(files) <= 3
        assert all(file.stat().st_size <= 512 for file in files)
        assert "latest marker" in path.read_text()
        assert sum(file.stat().st_size for file in files) <= 1536
