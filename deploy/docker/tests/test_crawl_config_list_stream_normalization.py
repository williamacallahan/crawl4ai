"""Regression tests for non-streaming /crawl per-URL `crawler_configs` stream
normalization (the companion of `test_stream_crawl_config_list.py`).

Bug: On the non-streaming ``/crawl`` endpoint, when a caller put ``"stream":
true`` in the first entry of ``crawler_configs``, the per-URL configs were
loaded (the untrusted gate explicitly allowlists ``stream``) and passed to
``arun_many``, which honors ``config[0].stream`` and returns an unconsumed
async generator. ``handle_crawl_request`` never normalized ``stream`` on the
per-URL path, so the generator was wrapped as a single "result" via
``results = [results]`` and classified as an unexpected type
(``url="unknown"``, ``success=False``, error_message=``"unexpected result
type: <class 'async_generator'>"``), failing the whole request with a redacted
502 -- no crawl ever ran. ``stream: true`` in any non-first per-URL config was
silently ignored, making the behavior position-dependent.

Fix: ``handle_crawl_request`` now forces ``cfg.stream = False`` on every
per-URL config, mirroring the streaming handler's
``cfg.stream = True`` normalization. The endpoint owns the batch/stream
contract.
"""

import asyncio

import api
import crawler_pool
import egress_broker

from crawl4ai import CrawlerRunConfig


def _server_config() -> dict:
    """Minimal server config for handle_crawl_request (no rate limiter, no
    wall-clock deadline). Mirrors _stream_config() in
    test_stream_crawl_config_list.py."""
    return {
        "crawler": {
            "base_config": {"delay_before_return_html": 1.0},
            "pool": {"max_pages": 2},
            "memory_threshold_percent": 95,
            "recovery_threshold_percent": 80,
            "rate_limiter": {"enabled": False, "base_delay": [0, 0]},
            "browser": {"kwargs": {}},
        },
        "limits": {"wall_clock_s": 0},
    }


class _CrawlResult:
    """Minimal CrawlResult stand-in exposing model_dump (the only shape the
    handler's result loop consumes for the success/failure verdict)."""

    def __init__(self, url, success=True, error_message=None):
        self.url = url
        self.success = success
        self.error_message = error_message

    def model_dump(self):
        return {
            "url": self.url,
            "success": self.success,
            "error_message": self.error_message,
            "fit_html": None,
            "pdf": None,
        }


class _RecordingCrawler:
    """Minimal AsyncWebCrawler stand-in for the non-streaming path.

    Records ``arun_many`` / ``arun`` calls and returns a list of
    CrawlResult-shaped objects (the batch-mode return shape), so
    handle_crawl_request can complete without a real browser.
    """

    def __init__(self):
        self.arun_many_calls: list = []
        self.arun_calls: list = []

    async def arun_many(self, urls, config, dispatcher=None, **_kwargs):
        self.arun_many_calls.append(
            {"urls": list(urls), "config": config, "dispatcher": dispatcher}
        )
        return [_CrawlResult(u, success=True) for u in urls]

    async def arun(self, url, config, **_kwargs):
        self.arun_calls.append({"url": url, "config": config})
        return _CrawlResult(url, success=True)


class _StreamingBranchCrawler:
    """A crawler that mimics ``AsyncWebCrawler.arun_many``'s stream
    branching exactly: if ``config[0].stream`` is true (or the single config's
    ``stream``), it returns an async generator (the streaming return shape);
    otherwise it returns a list of CrawlResult (the batch shape).

    Used to prove the fix actually prevents the buggy async-generator return:
    without the normalization, the per-URL config keeps the caller's
    stream=true and this crawler yields a generator, which the handler then
    misclassifies. With the normalization, stream is False on every per-URL
    config and this crawler always takes the batch branch.
    """

    def __init__(self):
        self.arun_many_calls: list = []

    async def arun_many(self, urls, config, dispatcher=None, **_kwargs):
        self.arun_many_calls.append(
            {"urls": list(urls), "config": config, "dispatcher": dispatcher}
        )
        primary = config[0] if isinstance(config, list) else config
        if getattr(primary, "stream", False):
            async def _gen():
                if False:  # pragma: no cover - bug shape, never iterated
                    yield None
            return _gen()
        return [_CrawlResult(u, success=True) for u in urls]


def _install_recording_crawler(monkeypatch, recording_crawler):
    """Patch the surface handle_crawl_request touches, mirroring the pattern
    in test_stream_crawl_config_list.py."""
    async def get_crawler(_browser_config):
        return recording_crawler

    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)
    monkeypatch.setattr(api, "_track_request_start", _no_op_track)


async def _no_op_track(_endpoint, _urls, _browser_config):
    return "req-test"


# ─── bug regression: stream in per-URL crawler_configs ────────────────────


def test_non_stream_crawl_never_wraps_async_generator_as_result(monkeypatch):
    """Direct guard for the bug's symptom. The crawler simulates the real
    arun_many branch: it returns an async generator when ``config[0].stream``
    is true (mirroring crawl4ai/async_webcrawler.py). Without the fix, the
    un-normalized per-URL stream=true causes the generator to be wrapped via
    ``results = [results]`` and then classified as an unexpected type
    (success=False, url="unknown", "unexpected result type: <class
    'async_generator'>"). With the fix, every per-URL config has stream=False
    and the batch branch is always taken.
    """
    crawler = _StreamingBranchCrawler()
    _install_recording_crawler(monkeypatch, crawler)

    response = asyncio.run(
        api.handle_crawl_request(
            urls=["https://site-a.example/x", "https://site-b.example/x"],
            browser_config={},
            crawler_config={},
            config=_server_config(),
            crawler_configs=[{"stream": True}, {"stream": True}],
        )
    )

    assert response["success"] is True, response
    assert len(response["results"]) == 2, response["results"]
    assert {r["url"] for r in response["results"]} == {
        "https://site-a.example/x",
        "https://site-b.example/x",
    }
    assert all(r["success"] for r in response["results"]), response["results"]
    assert not any(
        "unexpected result type" in (r.get("error_message") or "")
        for r in response["results"]
    ), response["results"]
    assert len(crawler.arun_many_calls) == 1
    config_arg = crawler.arun_many_calls[0]["config"]
    assert all(c.stream is False for c in config_arg), config_arg


def test_non_stream_crawl_normalizes_stream_false_on_every_per_url_config(monkeypatch):
    """The fix must force ``stream=False`` on EVERY per-URL config handed to
    arun_many -- not just the first -- so position no longer decides whether
    streaming kicks in. The streaming handler does the mirror (forces True).
    """
    crawler = _RecordingCrawler()
    _install_recording_crawler(monkeypatch, crawler)

    asyncio.run(
        api.handle_crawl_request(
            urls=[
                "https://site-a.example/x",
                "https://site-b.example/x",
                "https://site-c.example/x",
            ],
            browser_config={},
            crawler_config={},
            config=_server_config(),
            # stream=true in first, middle, and last positions.
            crawler_configs=[
                {"stream": True},
                {"stream": True},
                {"stream": True},
            ],
        )
    )

    assert len(crawler.arun_many_calls) == 1
    config_arg = crawler.arun_many_calls[0]["config"]
    assert isinstance(config_arg, list)
    assert len(config_arg) == 3
    assert all(isinstance(c, CrawlerRunConfig) for c in config_arg)
    streams = [c.stream for c in config_arg]
    assert streams == [False, False, False], (
        "every per-URL config must have stream=False on /crawl; got " + repr(streams)
    )


def test_non_stream_crawl_stream_in_non_first_per_url_config_normalized(monkeypatch):
    """Position independence: ``crawler_configs=[{}, {"stream": True}]`` must
    be normalized identically to ``[{"stream": True}, {}]`` -- both per-URL
    configs end up with stream=False, so the non-first stream=true cannot
    combine with a future first-position stream=true to re-introduce the bug.
    """
    crawler = _RecordingCrawler()
    _install_recording_crawler(monkeypatch, crawler)

    response = asyncio.run(
        api.handle_crawl_request(
            urls=["https://site-a.example/x", "https://site-b.example/x"],
            browser_config={},
            crawler_config={},
            config=_server_config(),
            crawler_configs=[{}, {"stream": True}],
        )
    )

    assert response["success"] is True, response
    config_arg = crawler.arun_many_calls[0]["config"]
    assert [c.stream for c in config_arg] == [False, False]


# ─── server.py top-level stream redirect unaffected ──────────────────────


def test_server_crawl_redirects_to_stream_when_top_level_stream_true(
    server_module, monkeypatch
):
    """No regression on the server.py top-level ``crawler_config.stream=True``
    redirect: the per-URL normalization in handle_crawl_request must not
    interfere with the existing redirect to stream_process. The /crawl
    endpoint still routes a top-level streaming request to the streaming
    handler (which then normalizes per-URL stream to True), and the
    crawler_configs list is forwarded intact.
    """
    captured: dict = {}

    async def fake_stream_process(*, crawl_request):
        captured["redirected"] = True
        captured["crawler_configs"] = crawl_request.crawler_configs
        captured["crawler_config"] = crawl_request.crawler_config
        return object()

    monkeypatch.setattr(server_module, "stream_process", fake_stream_process)

    crawler_configs = [
        {"url_matcher": "*a*", "stream": True},
        {"url_matcher": "*b*", "stream": True},
    ]
    crawl_request = server_module.CrawlRequestWithHooks(
        urls=["https://a.example/x", "https://b.example/x"],
        crawler_config={"stream": True},
        crawler_configs=crawler_configs,
    )

    # Bypass the @limiter / @mcp_tool decorators by reaching through __wrapped__.
    crawl_endpoint = server_module.crawl
    unwrapped = getattr(crawl_endpoint, "__wrapped__", crawl_endpoint)
    asyncio.run(unwrapped(request=object(), crawl_request=crawl_request, _td={}))

    assert captured.get("redirected") is True
    # The redirect preserved crawler_configs for the streaming handler to
    # normalize (it will force stream=True on every per-URL config itself).
    assert captured["crawler_configs"] == crawler_configs
    assert captured["crawler_config"] == {"stream": True}
