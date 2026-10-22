"""Native crawler/dispatcher contracts for offline archive HTML."""
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from archive_transport import ArchiveHTMLStrategy, grouped_results
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher


def dispatcher():
    return MemoryAdaptiveDispatcher(memory_threshold_percent=100,
                                    critical_threshold_percent=100,
                                    recovery_threshold_percent=99,
                                    max_session_permit=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_equal_bodies_use_original_matchers_and_response_metadata(stream):
    urls = ["https://archive.org/a", "https://archive.org/b"]
    html = '<html><body><p>Identical archived content.</p><a href="child">Child</a></body></html>'
    responses = {
        u: {"url": u + "/final/", "status": 404 if i else 200,
            "body": html, "responseHeaders": {"X-Seed": str(i)}}
        for i, u in enumerate(urls)
    }
    strategy = ArchiveHTMLStrategy(responses, browser_config=BrowserConfig())
    crawler = AsyncWebCrawler(crawler_strategy=strategy)
    # Native raw processing requires no browser; avoid arun's automatic launch.
    crawler.ready = True
    configs = [CrawlerRunConfig(url_matcher=u, cache_mode=CacheMode.BYPASS,
                               stream=stream, word_count_threshold=0) for u in urls]
    results = await crawler.arun_many(urls, config=configs,
                                     dispatcher=dispatcher())
    if stream:
        results = [r async for r in results]
    by_url = {r.url: r for r in results}
    assert set(by_url) == set(urls)
    assert by_url[urls[0]].status_code == 200
    assert by_url[urls[1]].status_code == 404
    assert by_url[urls[1]].response_headers["X-Seed"] == "1"
    assert by_url[urls[1]].redirected_url == urls[1] + "/final/"
    assert all(c.base_url is None for c in configs)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_mixed_groups_keep_both_results_and_release_on_close(monkeypatch, stream):
    import archive_transport as transport
    urls = ["https://archive.org/a", "raw:<p>ordinary content</p>"]
    active = 0

    @asynccontextmanager
    async def owned(browser, responses, hooks):
        nonlocal active
        assert active == 0
        active += 1
        strategy = ArchiveHTMLStrategy(responses, browser_config=browser) if responses else None
        crawler = AsyncWebCrawler(config=browser, crawler_strategy=strategy)
        crawler.ready = True
        try:
            yield crawler
        finally:
            active -= 1

    monkeypatch.setattr(transport, "owned_crawler", owned)
    responses = {urls[0]: {"url": urls[0], "status": 200,
                          "body": "<p>archive content</p>", "responseHeaders": {}}}
    cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, word_count_threshold=0, stream=stream)
    results = [r async for r in grouped_results(urls, BrowserConfig(), cfg,
                                               responses, {}, dispatcher)]
    assert {r.url for r in results} == set(urls)
    assert active == 0
    if stream:
        generator = grouped_results(urls, BrowserConfig(), cfg, responses, {}, dispatcher)
        await anext(generator)
        await generator.aclose()
        assert active == 0


@pytest.mark.browser
@pytest.mark.asyncio
async def test_raw_screenshot_never_fetches_remote_assets(effective_browser_args):
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"body { color: red }")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = "https://archive.org/snapshot"
    asset = f"http://127.0.0.1:{server.server_port}/style.css"
    html = f'<html><head><link rel="stylesheet" href="{asset}"></head><body>Snapshot</body></html>'
    browser = BrowserConfig(headless=True, text_mode=False, extra_args=effective_browser_args)
    strategy = ArchiveHTMLStrategy(
        {original: {"url": original, "status": 200, "body": html,
                    "responseHeaders": {"Content-Type": "text/html"}}},
        browser_config=browser,
    )
    try:
        async with AsyncWebCrawler(config=browser, crawler_strategy=strategy) as crawler:
            result = await crawler.arun(original, config=CrawlerRunConfig(
                screenshot=True, cache_mode=CacheMode.BYPASS, word_count_threshold=0,
            ))
        assert result.url == original
        assert result.screenshot
        assert hits == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
