"""avoid_ads blocks tracker hosts with their subdomains, and nothing else."""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

pytestmark = pytest.mark.browser

TRACKERS = [
    "www.googletagmanager.com", "pagead2.googlesyndication.com",
    "securepubads.g.doubleclick.net", "connect.facebook.net", "static.hotjar.com",
    "cdn.segment.com", "www.google-analytics.com", "c.amazon-adsystem.com",
    "sb.scorecardresearch.com", "doubleclick.net",
]
# A lookalike host and a tracker name in a path are not trackers.
OTHERS = ["cdn.example.com", "notdoubleclick.net", "site.test"]


@pytest.fixture
def server():
    hosts = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hosts.append(self.headers["Host"].split(":")[0])
            port = self.server.server_address[1]
            if self.path == "/":
                tags = "".join(f'<script src="http://{host}:{port}/t.js"></script>'
                               for host in TRACKERS + OTHERS[:2])
                tags += f'<script src="http://site.test:{port}/doubleclick.net/t.js"></script>'
                body = f"<html><body>{tags}{'<p>Real page text.</p>' * 40}</body></html>".encode()
                content_type = "text/html"
            else:
                body, content_type = b"", "application/javascript"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], hosts
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("avoid_ads", [True, False])
def test_avoid_ads_aborts_tracker_subdomains_only(server, avoid_ads):
    port, hosts = server
    browser = BrowserConfig(headless=True, avoid_ads=avoid_ads, extra_args=[
        "--no-sandbox", "--host-resolver-rules=MAP * 127.0.0.1"])

    async def crawl():
        async with AsyncWebCrawler(config=browser) as crawler:
            return await crawler.arun(f"http://site.test:{port}/", config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS, wait_until="load"))

    assert asyncio.run(crawl()).success
    assert hosts.count("site.test") == 2  # the page and the tracker-named path
    requested = set(hosts)
    assert set(OTHERS) <= requested
    assert requested.isdisjoint(TRACKERS) is avoid_ads
    if not avoid_ads:
        # Headless Chromium's own incognito script blocking drops a few
        # (connect.facebook.net, www.google-analytics.com); the rest load.
        assert {"www.googletagmanager.com", "securepubads.g.doubleclick.net",
                "static.hotjar.com", "c.amazon-adsystem.com"} <= requested
