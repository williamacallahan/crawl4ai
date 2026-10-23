"""A real permanent Chromium whose process tree outgrows its RSS cap is recycled."""

import asyncio
import dataclasses
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil
import pytest

import crawler_pool
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

pytestmark = pytest.mark.browser


class Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<html><body><p>tree rss recycle</p>" + b"<p>A paragraph of real page text.</p>" * 40 + b"</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def page_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("browser_rss_mb,recycled", [(1, True), (1_000_000, False)])
def test_permanent_browser_over_tree_rss_cap_restarts_and_serves(
    monkeypatch, page_url, browser_rss_mb, recycled
):
    monkeypatch.setattr(crawler_pool, "WATCHDOG",
                        dataclasses.replace(crawler_pool.WATCHDOG, browser_rss_mb=browser_rss_mb))
    for name, empty in (("HOT_POOL", {}), ("COLD_POOL", {}), ("_RENDERER_SAMPLES", {})):
        monkeypatch.setattr(crawler_pool, name, empty)
    monkeypatch.setattr(crawler_pool, "LOCK", asyncio.Lock())

    async def exercise():
        crawler = AsyncWebCrawler(config=BrowserConfig(headless=True, extra_args=["--no-sandbox"]))
        await crawler.start()
        monkeypatch.setattr(crawler_pool, "PERMANENT", crawler)
        manager = crawler.crawler_strategy.browser_manager
        try:
            assert (await crawler.arun(page_url, config=CrawlerRunConfig())).success
            first_pid = manager.browser_pid
            await crawler_pool._watchdog_pass()
            for _attempt in range(600 if recycled else 0):
                if manager.browser_pid != first_pid and not manager._recycling:
                    break
                await asyncio.sleep(0.1)
            assert manager._recycle_due is False
            assert (manager.browser_pid != first_pid) is recycled
            assert psutil.pid_exists(first_pid) is not recycled
            result = await crawler.arun(page_url, config=CrawlerRunConfig())
            assert result.success and "tree rss recycle" in result.html
        finally:
            await crawler.close()

    asyncio.run(exercise())
