"""Local hostile documents against the deployed Chromium configuration.

These are real browser tests; no network outside loopback and no mocked page
methods. The allocation attack touches a fixed 896 MiB in 32 MiB chunks,
never unbounded host memory. A failed outer deadline kills only this fixture's
own Chromium tree and fails, so cleanup cannot make a missing deadline pass.
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import psutil
import pytest
import pytest_asyncio

from crawl4ai import BrowserConfig, CacheMode, CrawlerRunConfig

pytestmark = pytest.mark.browser
MIB = 1024 * 1024
PAGE_TIMEOUT_MS = 4000
NORMAL_TEXT = "Normal crawl remains responsive and returns this complete article. " * 12


def _document(script="", body=None):
    return (
        "<!doctype html><html><head><title>Isolation fixture</title></head>"
        f"<body><h1>Isolation fixture</h1><p>{body or NORMAL_TEXT}</p>"
        f"<script>{script}</script></body></html>"
    )


@dataclass
class HostileSite:
    base: str
    events: dict

    async def armed(self, kind, timeout=5):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if self.events.get(kind, threading.Event()).is_set():
                return
            await asyncio.sleep(0.02)
        pytest.fail(f"hostile document never armed: {kind}")


@pytest.fixture
def hostile_site():
    events = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/armed":
                events.setdefault(query["kind"][0], threading.Event()).set()
                payload = "armed"
            elif parsed.path == "/popups":
                payload = _document("""
                    addEventListener('load', () => {
                        let opened = 0;
                        for (let i = 0; i < 50; i++) {
                            if (window.open('/normal?popup=' + i, '_blank')) opened++;
                        }
                        document.body.dataset.attempts = '50';
                        document.body.dataset.opened = String(opened);
                    });
                """)
            elif parsed.path in ("/spin", "/memory", "/history"):
                kind = parsed.path[1:]
                loop = (
                    "for (let i = 0; ; i++) history.pushState({}, '', '#p' + i);"
                    if kind == "history" else "for (;;) {}"
                )
                payload = _document(f"""
                    addEventListener('load', () => {{
                        fetch('/armed?kind={kind}').then(() => {{ {loop} }});
                    }});
                """)
                if kind == "memory":
                    payload = _document("""
                        addEventListener('load', () => {
                            window.buffers = [];
                            function grow() {
                                const b = new Uint8Array(new ArrayBuffer(32 * 1024 * 1024));
                                for (let j = 0; j < b.length; j += 4096) b[j] = 1;
                                window.buffers.push(b);
                                if (window.buffers.length === 4) fetch('/armed?kind=memory');
                                if (window.buffers.length < 28) setTimeout(grow, 20);
                                else for (;;) {}
                            }
                            grow();
                        });
                    """)
            elif parsed.path == "/frames":
                frames = "".join(f'<iframe src="/normal?frame={i}"></iframe>' for i in range(200))
                payload = _document(body=NORMAL_TEXT + frames)
            elif parsed.path == "/idb":
                mode = query.get("mode", ["read"])[0]
                payload = _document(f"""
                    const opening = indexedDB.open('crawl-isolation-boundary', 1);
                    opening.onupgradeneeded = () => opening.result.createObjectStore('state');
                    opening.onsuccess = () => {{
                        const db = opening.result;
                        const tx = db.transaction('state', '{'readwrite' if mode == 'seed' else 'readonly'}');
                        const store = tx.objectStore('state');
                        const op = {'store.put("first-request-private", "marker")' if mode == 'seed' else 'store.get("marker")'};
                        op.onsuccess = () => {{
                            document.body.dataset.idb = {'"seeded"' if mode == 'seed' else 'op.result || "empty"'};
                        }};
                        tx.oncomplete = () => db.close();
                    }};
                """)
            else:
                payload = _document()
            encoded = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield HostileSite(f"http://127.0.0.1:{server.server_port}", events)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _browser_config():
    # Exercise the same server-authored flags and isolation policy as /crawl.
    from api import _apply_server_browser_policy
    from utils import load_config

    config = load_config()
    browser = BrowserConfig(**config["crawler"]["browser"].get("kwargs", {}))
    browser.verbose = False
    return _apply_server_browser_policy(browser, config)


class BrowserProbe:
    def __init__(self, crawler, browser_process):
        self.crawler = crawler
        self.process = browser_process
        self.tasks = set()
        self.owned = {browser_process.pid: browser_process}
        self.roots = [browser_process]

    async def adopt(self, crawler):
        browser = crawler.crawler_strategy.browser_manager.browser
        session = await browser.new_browser_cdp_session()
        try:
            info = await session.send("SystemInfo.getProcessInfo")
        finally:
            await session.detach()
        pid = next(int(p["id"]) for p in info["processInfo"] if p["type"] == "browser")
        self.crawler, self.process = crawler, psutil.Process(pid)
        self.roots.append(self.process)
        self.owned[pid] = self.process

    def processes(self):
        for root in self.roots:
            try:
                for process in root.children(recursive=True):
                    self.owned[process.pid] = process
            except psutil.NoSuchProcess:
                pass
        # Retain observed children after their parent exits, so orphaned
        # renderers still count against RSS and teardown assertions.
        return [process for process in self.owned.values() if process.is_running()]

    def rss(self):
        total = 0
        for process in self.processes():
            try:
                total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total

    def kill(self):
        # psutil.Process retains create_time, avoiding unrelated recycled PIDs.
        for process in reversed(self.processes()):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass

    async def crawl(self, url, *, timeout_ms=PAGE_TIMEOUT_MS, **kwargs):
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=timeout_ms,
            wait_until="load",
            delay_before_return_html=kwargs.pop("delay_before_return_html", 0.5),
            verbose=False,
            **kwargs,
        )
        task = asyncio.create_task(self.crawler.arun(url, config=config))
        self.tasks.add(task)
        started = time.monotonic()
        done, _ = await asyncio.wait({task}, timeout=timeout_ms / 1000 + 10)
        if not done:
            self.kill()
            task.cancel()
            pytest.fail(f"crawl exceeded page_timeout + 10s: {url}")
        self.tasks.discard(task)
        return task.result(), time.monotonic() - started


@pytest_asyncio.fixture
async def browser_probe(monkeypatch):
    import crawler_pool

    # This test process owns a separate real pool, without Redis or a server.
    for name, value in {
        "PERMANENT": None, "DEFAULT_CONFIG_SIG": None,
        "HOT_POOL": {}, "COLD_POOL": {}, "LAST_USED": {}, "USAGE_COUNT": {},
        "LOCK": asyncio.Lock(), "ADMISSION_SEM": asyncio.Semaphore(12),
        "_CLOSE_TASKS": set(),
    }.items():
        monkeypatch.setattr(crawler_pool, name, value)
    crawler = await crawler_pool.get_crawler(_browser_config())
    browser = crawler.crawler_strategy.browser_manager.browser
    session = await browser.new_browser_cdp_session()
    try:
        info = await session.send("SystemInfo.getProcessInfo")
    finally:
        await session.detach()
    browser_pid = next(int(p["id"]) for p in info["processInfo"] if p["type"] == "browser")
    probe = BrowserProbe(crawler, psutil.Process(browser_pid))

    async def watch():
        # origin/main has no watchdog. Running that baseline must expose the
        # real missing protection, rather than fail to import the test suite.
        watchdog = getattr(crawler_pool, "_watchdog_pass", None)
        if watchdog is None:
            return
        while True:
            await watchdog()
            await asyncio.sleep(crawler_pool.WATCHDOG.interval_seconds)

    watcher = asyncio.create_task(watch())
    try:
        yield probe
        if watcher.done():
            watcher.result()
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        # Kill only our browser if a regression left a runaway renderer. The
        # assertions all precede this backstop; this is never success evidence.
        probe.kill()
        for task in probe.tasks:
            task.cancel()
        if probe.tasks:
            await asyncio.wait(probe.tasks, timeout=5)
        await crawler_pool.release_crawler(probe.crawler)
        closing = asyncio.create_task(crawler_pool.close_all())
        done, _ = await asyncio.wait({closing}, timeout=10)
        if done:
            closing.result()
        else:
            closing.cancel()


async def _normal(probe, site, *, same_browser=False):
    import crawler_pool

    if not same_browser and not crawler_pool._is_live(probe.crawler):
        # Match a subsequent /crawl request: the pool replaces a retired browser
        # instead of retrying a stale crawler object that rejects admission.
        await crawler_pool.release_crawler(probe.crawler)
        await probe.adopt(await crawler_pool.get_crawler(_browser_config()))
    result, elapsed = await probe.crawl(site.base + "/normal")
    assert result.success, result.error_message
    assert "Normal crawl remains responsive" in result.html
    return elapsed


@pytest.mark.asyncio
async def test_fifty_popups_leave_no_extra_pages_and_later_crawl_works(browser_probe, hostile_site):
    probe = browser_probe
    observed_popups = []

    async def observe(page, context, **_kwargs):
        context.on("page", lambda popup: observed_popups.append(popup))
        return page

    probe.crawler.crawler_strategy.set_hook("on_page_context_created", observe)
    result, _ = await probe.crawl(hostile_site.base + "/popups")
    assert result.success, result.error_message
    assert 'data-attempts="50"' in result.html, "popup payload did not run"
    manager = probe.crawler.crawler_strategy.browser_manager
    assert not [p for p in observed_popups if not p.is_closed()]
    assert sum(len(c.pages) for c in manager.browser.contexts) == 0
    await _normal(probe, hostile_site)


@pytest.mark.asyncio
async def test_infinite_javascript_fails_by_deadline_and_renderer_stops(browser_probe, hostile_site):
    probe = browser_probe
    task = asyncio.create_task(probe.crawl(hostile_site.base + "/spin"))
    await hostile_site.armed("spin")
    result, elapsed = await task
    assert not result.success, "an infinite-JS document unexpectedly succeeded"
    assert elapsed < PAGE_TIMEOUT_MS / 1000 + 10
    await asyncio.sleep(5)
    renderers = []
    for process in probe.processes():
        try:
            if "--type=renderer" in process.cmdline():
                process.cpu_percent()
                renderers.append(process)
        except psutil.NoSuchProcess:
            pass
    await asyncio.sleep(0.5)
    for process in renderers:
        try:
            assert process.cpu_percent() <= 50, "hostile renderer is still consuming CPU"
        except psutil.NoSuchProcess:
            pass
    await _normal(probe, hostile_site)


@pytest.mark.asyncio
async def test_bounded_arraybuffer_attack_fails_and_releases_rss(browser_probe, hostile_site):
    probe = browser_probe
    await _normal(probe, hostile_site)
    import crawler_pool

    baseline = probe.rss()
    kills_before = getattr(crawler_pool, "RENDERER_KILLS", {}).get("rss", 0)
    task = asyncio.create_task(probe.crawl(
        hostile_site.base + "/memory", timeout_ms=8000,
        delay_before_return_html=2,
    ))
    await hostile_site.armed("memory")
    peak = probe.rss()
    while not task.done():
        peak = max(peak, probe.rss())
        await asyncio.sleep(0.02)
    result, elapsed = await task
    assert not result.success, "allocation-plus-spin document unexpectedly succeeded"
    assert elapsed < 8, f"memory watchdog failed to abort quickly: {elapsed:.2f}s"
    assert getattr(crawler_pool, "RENDERER_KILLS", {}).get("rss", 0) > kills_before, (
        "real renderer RSS never triggered the default 768 MiB watchdog"
    )
    await asyncio.sleep(5)
    recovered = probe.rss()
    assert recovered <= baseline + 100 * MIB, json.dumps({
        "baseline_mib": baseline / MIB,
        "peak_mib": peak / MIB,
        "recovered_mib": recovered / MIB,
    })
    await _normal(probe, hostile_site)


@pytest.mark.asyncio
async def test_pushstate_flood_does_not_starve_normal_crawl(browser_probe, hostile_site):
    probe = browser_probe
    attack = asyncio.create_task(probe.crawl(hostile_site.base + "/history"))
    await hostile_site.armed("history")
    normal = asyncio.create_task(_normal(probe, hostile_site, same_browser=True))
    try:
        done, _ = await asyncio.wait({normal}, timeout=15)
        assert done, "pushState flood blocked a concurrent normal crawl for 15s"
        assert normal.result() < 15
        result, _ = await attack
        assert not result.success
    finally:
        if not normal.done():
            normal.cancel()
        if not attack.done():
            attack.cancel()


@pytest.mark.asyncio
async def test_two_hundred_iframes_finish_within_page_timeout(browser_probe, hostile_site):
    result, elapsed = await browser_probe.crawl(
        hostile_site.base + "/frames", timeout_ms=10000, process_iframes=True,
    )
    assert elapsed < 10, f"200 iframe crawl exceeded page_timeout: {elapsed:.2f}s"
    assert result.success, result.error_message
    inlined = result.html.count('class="extracted-iframe-content-')
    assert 0 < inlined <= 20, f"inlined {inlined} of 200 frames; cap is 20"
    assert result.html.count("<iframe ") == 200 - inlined
    await _normal(browser_probe, hostile_site)


@pytest.mark.asyncio
async def test_indexeddb_does_not_survive_cross_site_crawl_workflow(browser_probe, hostile_site):
    probe = browser_probe
    seeded, _ = await probe.crawl(
        hostile_site.base + "/idb?mode=seed", wait_for="js:() => document.body.dataset.idb === 'seeded'",
    )
    assert seeded.success, seeded.error_message
    assert 'data-idb="seeded"' in seeded.html
    # localhost and 127.0.0.1 are different sites, despite one loopback server.
    other_site = hostile_site.base.replace("127.0.0.1", "localhost")
    between, _ = await probe.crawl(other_site + "/normal")
    assert between.success, between.error_message
    read, _ = await probe.crawl(
        hostile_site.base + "/idb?mode=read", wait_for="js:() => !!document.body.dataset.idb",
    )
    assert read.success, read.error_message
    assert 'data-idb="empty"' in read.html, "a later request inherited an earlier request's IndexedDB"


@pytest.mark.asyncio
async def test_sustained_cpu_watchdog_kills_renderer_before_long_page_deadline(
    browser_probe, hostile_site, monkeypatch
):
    import dataclasses

    import crawler_pool

    # Production waits 65 s (page_timeout plus termination) so it only catches
    # renderers script termination missed; the mechanism is the same with a
    # 20 s window, which keeps this test inside its 35 s bound.
    monkeypatch.setattr(
        crawler_pool, "WATCHDOG",
        dataclasses.replace(crawler_pool.WATCHDOG, renderer_cpu_seconds=20.0),
    )
    probe = browser_probe
    kills_before = getattr(crawler_pool, "RENDERER_KILLS", {}).get("cpu", 0)
    attack = asyncio.create_task(probe.crawl(hostile_site.base + "/spin", timeout_ms=60000))
    try:
        await hostile_site.armed("spin")
        # >0.9 cores sustained for the 20 s window, sampled every 5 seconds:
        # the kill must precede the 60 s deadline.
        done, _ = await asyncio.wait({attack}, timeout=35)
        assert done, "sustained-CPU renderer survived 35s without watchdog intervention"
        result, elapsed = attack.result()
        assert not result.success, "infinite-JS document unexpectedly succeeded"
        assert elapsed < 35
        assert getattr(crawler_pool, "RENDERER_KILLS", {}).get("cpu", 0) > kills_before, (
            "real CPU consumption never triggered the sustained-CPU watchdog"
        )
        await _normal(probe, hostile_site)
    finally:
        if not attack.done():
            attack.cancel()


@pytest.mark.asyncio
async def test_effective_chromium_flags_preserve_protections(browser_probe):
    from crawl4ai.browser_manager import PLAYWRIGHT_DISABLED_FEATURES

    browser = browser_probe.crawler.crawler_strategy.browser_manager.browser
    session = await browser.new_browser_cdp_session()
    try:
        command = await session.send("Browser.getBrowserCommandLine")
    finally:
        await session.detach()
    args = command["arguments"]
    disabled = [arg for arg in args if arg.startswith("--disable-features=")]
    assert len(disabled) == 1, disabled
    assert set(PLAYWRIGHT_DISABLED_FEATURES) <= set(disabled[0].split("=", 1)[1].split(","))
    assert "--disable-ipc-flooding-protection" not in args
    assert not any(arg.startswith("--renderer-process-limit=") for arg in args)


@pytest.mark.asyncio
async def test_three_crawl_timeouts_keep_the_browser(browser_probe, hostile_site):
    """Slow or dead target sites are not browser faults: a batch of them timing
    out together must not retire the browser under other crawls."""
    import crawler_pool

    original = browser_probe.crawler
    manager = original.crawler_strategy.browser_manager
    for _ in range(3):
        result, _ = await browser_probe.crawl(hostile_site.base + "/spin", timeout_ms=1500)
        assert not result.success
    assert manager.consecutive_failures == 0
    assert crawler_pool._is_live(original)
    await _normal(browser_probe, hostile_site)
    assert browser_probe.crawler is original
