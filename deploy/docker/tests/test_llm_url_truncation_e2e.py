"""End-to-end regression test for the ``/llm/{url:path}`` URL-truncation bug.

Companion to ``test_llm_url_truncation_repro.py``.  Where the repro file
drives the FastAPI app with stubbed crawl/LLM boundaries and proves the
URL survives the handler at the *API* layer, this file completes the
proof at the *browser* layer: a real headless Chromium (via
``AsyncWebCrawler``) actually navigates to the URL the user supplied,
and the body returned by the *user's* URL — not a truncated stand-in —
is what reaches the (stubbed) LLM prompt.

Mechanism: a local ``http.server.ThreadingHTTPServer`` returns distinct
HTML bodies for ``/search?q=apple`` and ``/search``.  The real
``AsyncWebCrawler`` (unstubbed Playwright/Chromium) is what
``handle_llm_qa`` ends up driving: ``crawler_pool.get_crawler`` is
patched to return this real crawler, ``_crawler_arun`` is left real, and
only the support boundaries that need external services are stubbed
(``validate_url_destination`` for the SSRF guard, ``enforce_egress``
for the pinning proxy, ``resolve_llm`` / ``aperform_completion_with_backoff``
to capture the prompt and return a stub answer, and ``redis`` is a
``PermitRedis`` so the real ``llm_permit`` admission runs offline).  The
buggy heuristic previously truncated ``/search?q=apple`` to ``/search``,
so Chromium fetched the wrong page and the LLM saw
``NO_QUERY_STRING_LANDING``; after the fix Chromium fetches
``/search?q=apple`` and the LLM sees ``QUERY_PARAM_APPLE_PRESENT``.

Preconditions:
* A real headless Chromium installed for Playwright
  (``playwright install chromium``).  When running as root, the root
  ``conftest.py`` injects ``--no-sandbox`` into every ``BrowserConfig``.
"""
import asyncio
import socket
import sys
import threading
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import api
import crawler_pool
import egress_broker
import llm_broker
import pytest
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from utils import load_config

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

pytestmark = pytest.mark.browser


_APPLE_BODY = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Apple Search Results</title>
  <meta name="description" content="Search results for apple">
</head>
<body>
  <header><h1>Apple Search Results</h1><nav>Home | About | Contact</nav></header>
  <main>
    <article>
      <h2>QUERY_PARAM_APPLE_PRESENT apple results page</h2>
      <p>This is a search results page rendered specifically for the query
         string <code>?q=apple</code>. The page is served only when the
         request's own query string's first parameter is named <code>q</code>
         and its value is <code>apple</code>. Any other shape (no query,
         or a different value) returns a different body.</p>
      <p>The body is intentionally long enough to clear the crawler's
         <em>structural</em> anti-bot heuristic (which would otherwise
         reject minimal pages). The distinguishing token is
         QUERY_PARAM_APPLE_PRESENT, embedded once in the article and
         once in the page footer.</p>
      <ul>
        <li>Result 1: Apple Inc.</li>
        <li>Result 2: Apple pie recipe</li>
        <li>Result 3: Apple orchards near me</li>
        <li>Result 4: Apple MacBook Pro</li>
        <li>Result 5: Apple iPhone</li>
      </ul>
      <p>These results exist only because the server received the
         <code>?q=apple</code> query parameter. They would not appear on
         the no-query landing page.</p>
    </article>
  </main>
  <footer><p>Marker: QUERY_PARAM_APPLE_PRESENT. Copyright Example Search.</p></footer>
</body>
</html>"""

_NO_QUERY_BODY = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Landing Page</title>
  <meta name="description" content="Generic search landing page">
</head>
<body>
  <header><h1>Search Landing Page</h1><nav>Home | About | Contact</nav></header>
  <main>
    <article>
      <h2>NO_QUERY_STRING_LANDING generic search page</h2>
      <p>This is the generic search landing page. It is served for any
         <code>/search</code> request that does <em>not</em> include a
         <code>?q=apple</code> query string. Pages with the query string
         present are served a different body.</p>
      <p>The body is intentionally long enough to clear the crawler's
         <em>structural</em> anti-bot heuristic. The distinguishing token
         is NO_QUERY_STRING_LANDING, embedded once in the article and
         once in the page footer.</p>
      <ul>
        <li>Popular search: apples</li>
        <li>Popular search: bananas</li>
        <li>Popular search: cherries</li>
        <li>Popular search: dragonfruit</li>
        <li>Popular search: elderberry</li>
      </ul>
      <p>To actually search, append <code>?q=your+term</code> to the URL.</p>
    </article>
  </main>
  <footer><p>Marker: NO_QUERY_STRING_LANDING. Copyright Example Search.</p></footer>
</body>
</html>"""


class _QuerySensitiveHandler(BaseHTTPRequestHandler):
    """Returns ``_APPLE_BODY`` when ``?q=apple`` is present in the URL,
    ``_NO_QUERY_BODY`` otherwise.

    Differentiating on the query string is the defining behaviour of a
    search endpoint and is the precise semantic the bug violated: any
    real server that inspects the query string serves different content
    for ``/search?q=apple`` vs ``/search``.
    """

    def _serve(self):
        if self.path.startswith("/search") and "q=apple" in self.path:
            body = _APPLE_BODY
        elif self.path.startswith("/search"):
            body = _NO_QUERY_BODY
        else:
            body = "<html><body>NOT_FOUND</body></html>"
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve()

    def log_message(self, *_a, **_kw):  # silence noisy stderr in CI logs
        pass


@pytest.fixture
def local_search_server():
    """Start a local ``ThreadingHTTPServer`` returning query-sensitive bodies."""
    # Bind to loopback so the SSRF guard (patched anyway) sees a private host.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = ThreadingHTTPServer(("127.0.0.1", port), _QuerySensitiveHandler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class PermitRedis:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.hset = AsyncMock()
        self.expire = AsyncMock()
        self.eval = AsyncMock(return_value=1)

    async def set(self, *_args, **_kwargs):
        return self.acquired


def _llm_config():
    return {
        **load_config(),
        "llm": {"provider": "openai/qwen3-32b", "api_key": "test-only"},
    }


def _patch_llm_boundaries(monkeypatch, captured_prompt):
    """Stub only what requires external services; leave the real crawler."""
    monkeypatch.setattr(api, "validate_url_destination", lambda _url: None)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _config: None)
    monkeypatch.setattr(
        llm_broker,
        "resolve_llm",
        lambda *_a, **_kw: {
            "provider": "openai/qwen3-32b",
            "api_token": "test-only",
            "temperature": 0.0,
            "base_url": None,
            "extra_args": {
                "timeout": 300,
                "num_retries": 0,
                "reasoning_effort": "low",
                "max_tokens": 4096,
            },
        },
    )

    async def fake_completion(**kwargs):
        # Capture the *prompt* that carries the crawled content so we can
        # assert which page the browser actually fetched.
        captured_prompt["text"] = kwargs.get("prompt_with_variables", "")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )

    monkeypatch.setattr(api, "aperform_completion_with_backoff", fake_completion)


@pytest.mark.asyncio
async def test_e2e_q_param_url_fetches_full_query_page(local_search_server, monkeypatch):
    """The endpoint fetches the user's URL verbatim, not a truncated one.

    With the buggy heuristic, ``handle_llm_qa('/search?q=apple')`` would
    slice to ``/search`` and Chromium would navigate to the truncated
    URL.  The content reaching the LLM would be ``NO_QUERY_STRING_LANDING``.
    After the fix, Chromium fetches ``/search?q=apple`` and the LLM sees
    ``QUERY_PARAM_APPLE_PRESENT``.  This test drives the *real*
    ``AsyncWebCrawler`` (unstubbed Playwright/Chromium) through the real
    ``handle_llm_qa`` flow so the wire-level fetch is observed.
    """
    page_url = f"{local_search_server}/search?q=apple"
    captured_prompt: dict = {}
    _patch_llm_boundaries(monkeypatch, captured_prompt)

    # Real headless Chromium crawler.  The async-with owns its lifecycle;
    # ``crawler_pool.get_crawler``/``release_crawler`` are patched to
    # surface this crawler to ``handle_llm_qa`` without the pool's
    # admission/recycling machinery touching it.
    browser_config = BrowserConfig(headless=True, verbose=False)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        monkeypatch.setattr(crawler_pool, "get_crawler", AsyncMock(return_value=crawler))
        monkeypatch.setattr(
            crawler_pool, "release_crawler", AsyncMock(return_value=None)
        )

        answer = await api.handle_llm_qa(
            page_url,
            "What is on this page?",
            _llm_config(),
            redis=PermitRedis(),
        )

    assert answer == "answer"
    prompt = captured_prompt["text"]
    # The user's URL was fetched verbatim, so its query-sensitive body is in
    # the LLM prompt...
    assert "QUERY_PARAM_APPLE_PRESENT" in prompt
    # ...and the truncated-URL body is not.
    assert "NO_QUERY_STRING_LANDING" not in prompt


@pytest.mark.asyncio
async def test_e2e_no_query_url_non_regression(local_search_server, monkeypatch):
    """Negative end-to-end control: a URL with no own query string lands
    on the no-query body, both before and after the fix.  This pins the
    non-query path so the fix can never silently regress to *stripping*
    non-existent query strings on URLs that never had one (the
    ``example.com``-style documented example).
    """
    page_url = f"{local_search_server}/search"
    captured_prompt: dict = {}
    _patch_llm_boundaries(monkeypatch, captured_prompt)

    browser_config = BrowserConfig(headless=True, verbose=False)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        monkeypatch.setattr(crawler_pool, "get_crawler", AsyncMock(return_value=crawler))
        monkeypatch.setattr(
            crawler_pool, "release_crawler", AsyncMock(return_value=None)
        )

        answer = await api.handle_llm_qa(
            page_url,
            "What is on this page?",
            _llm_config(),
            redis=PermitRedis(),
        )

    assert answer == "answer"
    prompt = captured_prompt["text"]
    assert "NO_QUERY_STRING_LANDING" in prompt
    assert "QUERY_PARAM_APPLE_PRESENT" not in prompt
