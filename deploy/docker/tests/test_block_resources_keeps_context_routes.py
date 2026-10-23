"""The block_resources hook leaves the context's own abort routes in force."""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from hook_registry import build_declarative_hooks

pytestmark = pytest.mark.browser


def test_block_resources_hook_keeps_avoid_ads_blocking():
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(f"{self.headers['Host'].split(':')[0]}{self.path}")
            port = self.server.server_address[1]
            body = (
                f'<script src="http://www.googletagmanager.com:{port}/gtm.js"></script>'
                f'<script src="http://site.test:{port}/app.js"></script>'
                f'<img src="http://site.test:{port}/photo.png">'
                + "<p>Real page text.</p>" * 40
            ).encode() if self.path == "/" else b""
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    hooks = build_declarative_hooks(
        [{"action": "block_resources", "params": {"resource_types": ["image"]}}])

    async def crawl():
        browser = BrowserConfig(headless=True, avoid_ads=True, extra_args=[
            "--no-sandbox", "--host-resolver-rules=MAP * 127.0.0.1"])
        async with AsyncWebCrawler(config=browser) as crawler:
            for hook_point, hook in hooks.items():
                crawler.crawler_strategy.set_hook(hook_point, hook)
            return await crawler.arun(f"http://site.test:{httpd.server_address[1]}/",
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, wait_until="load"))

    try:
        assert asyncio.run(crawl()).success
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert "site.test/app.js" in paths
    assert "site.test/photo.png" not in paths  # the hook's own block
    assert "www.googletagmanager.com/gtm.js" not in paths  # the context's avoid_ads block
