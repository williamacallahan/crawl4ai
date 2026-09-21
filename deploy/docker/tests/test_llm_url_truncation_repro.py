"""Regression suite for the ``/llm/{url:path}`` URL-truncation bug.

Historical bug
--------------
``handle_llm_qa`` in ``deploy/docker/api.py`` used to slice the user URL at
the last literal ``?q=`` substring before fetching it::

    last_q_index = url.rfind('?q=')
    if last_q_index != -1:
        url = url[:last_q_index]

Because FastAPI's ``/{url:path}`` converter percent-decodes the path
before the handler runs, the documented client convention
``urllib.parse.quote_plus(page_url, safe="")`` produces a literal ``?q=``
inside ``url`` whenever the user's own URL query begins with ``q=`` (most
notably ``https://www.google.com/search?q=apple``).  The heuristic then
stripped the query string and Chromium fetched the wrong page (the LLM
answered a question about ``/search`` instead of ``/search?q=apple``).
The documented fused form
``/llm/https://example.com?q=What is this page about?`` (convention (a)
in ``docs/md_v2/assets/llm.txt/txt/docker.txt``) was never affected:
FastAPI's ``:path`` converter stops at the first unencoded ``?`` and
hands ``https://example.com`` to ``url`` (leaving the LLM question in
``q``); the heuristic had no ``?q=`` to match inside ``url``.

Fix
---
The three slicing lines are removed from ``handle_llm_qa``.
FastAPI's ``{url:path}`` + ``Query("q")`` split already separates the
page URL from the LLM question at the HTTP layer, so the post-processing
is unnecessary and was harmful for any URL whose own query string's first
parameter is ``q``.

Coverage
--------
* ``test_percent_decoding`` — documents that ``%3Fq%3D`` in the wire
  path arrives as a literal ``?q=`` in the handler's ``url`` argument
  (proven by capturing ``url`` at ``api.validate_url_destination``).
  Passes both before and after the fix; it pins the root-cause
  mechanism (Starlette ``:path`` percent-decoding) that made the bug
  reachable on convention (b).
* ``test_llm_qa_url_preserved`` — RED on the buggy tree for any URL
  whose own query first param is ``q``; GREEN on the fixed tree.  It
  captures ``url`` at both ``validate_url_destination`` (pre-slice) and
  ``_crawler_arun`` (post-slice) while driving the real FastAPI app via
  ``starlette.testclient.TestClient``.
* ``test_convention_a_fused_form_non_regression`` — the documented
  unencoded fused form ``/llm/https://example.com?q=What is this page
  about?`` continues to split the page URL from the LLM question at the
  HTTP layer (passing both before and after the fix; the regression
  guard for over-correction).
* ``test_handle_llm_qa_direct_call_preserves_url`` — direct
  ``asyncio.run(api.handle_llm_qa(...))`` for every URL shape (with and
  without its own query; first-param-``q`` and otherwise), asserting
  ``_crawler_arun`` receives the verbatim user URL.
"""
import asyncio
import urllib.parse
from types import SimpleNamespace
from unittest.mock import AsyncMock

import api
import crawler_pool
import egress_broker
import llm_broker
import pytest
from auth import create_access_token
from utils import load_config


class PermitRedis:
    """Real ``llm_permit`` runs against this stub.

    Controls permit acquire/release so the LLM admission path is
    reachable offline without a real Redis, mirroring the convention in
    ``test_server_crawler_defaults.py``.
    """

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


def _resolve_llm():
    return lambda *_a, **_kw: {
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
    }


def _stub_completion():
    return AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
    )


def _patch_boundaries(monkeypatch, captured, *, crawl_markdown="ctx", server_module=None):
    """Stub every boundary ``handle_llm_qa`` cannot reach offline.

    Captures the URL at the two points the bug report uses to prove the
    slice: ``validate_url_destination`` (called *before* the buggy slice,
    so on the buggy tree it sees the user's verbatim URL) and
    ``_crawler_arun`` (called *after* the buggy slice, so on the buggy
    tree it sees the truncated URL).  Both are populated into
    ``captured["pre_slice_urls"]`` / ``captured["post_slice_urls"]`` for
    the test to assert on.

    When ``server_module`` is provided (HTTP-level tests via
    ``stock_client``), the server-wide ``redis`` global is also replaced
    with a ``PermitRedis`` stub so the real ``llm_permit`` admission
    context-manager runs against an in-memory stub instead of dialing
    a real Redis at 127.0.0.1:6379.  Keeping ``llm_permit`` real follows
    the suite convention (``test_server_crawler_defaults.py``); only the
    transport is stubbed, not the admission policy.
    """
    captured.setdefault("pre_slice_urls", [])
    captured.setdefault("post_slice_urls", [])

    def fake_validate(url):
        captured["pre_slice_urls"].append(url)
        return None

    async def fake_arun(_crawler, *args, **kwargs):
        captured["post_slice_urls"].append(kwargs.get("url"))
        return SimpleNamespace(
            success=True,
            markdown=SimpleNamespace(
                fit_markdown=crawl_markdown, raw_markdown=crawl_markdown
            ),
        )

    async def fake_get_crawler(_bc):
        return object()

    async def fake_release_crawler(_c):
        return None

    monkeypatch.setattr(api, "validate_url_destination", fake_validate)
    monkeypatch.setattr(api, "_crawler_arun", fake_arun)
    monkeypatch.setattr(crawler_pool, "get_crawler", fake_get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", fake_release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)
    monkeypatch.setattr(llm_broker, "resolve_llm", _resolve_llm())
    monkeypatch.setattr(api, "aperform_completion_with_backoff", _stub_completion())
    if server_module is not None:
        monkeypatch.setattr(server_module, "redis", PermitRedis())


def _auth_header():
    return {"Authorization": f"Bearer {create_access_token({'sub': 'test@example.com'})}"}


# ─────────────────────────── Unit layer ────────────────────────────


@pytest.mark.parametrize(
    ("page_url", "expected_fetched_url"),
    [
        # The bug case: query first param is `q`. Was truncated to "/search".
        (
            "https://www.google.com/search?q=apple",
            "https://www.google.com/search?q=apple",
        ),
        # Multi-param: was truncated to "/search" (lost `&hl=en` too).
        (
            "https://www.google.com/search?q=apple&hl=en",
            "https://www.google.com/search?q=apple&hl=en",
        ),
        # Reddit search uses `?q=` (path has a trailing slash before `?`).
        (
            "https://www.reddit.com/search/?q=apple",
            "https://www.reddit.com/search/?q=apple",
        ),
        # Negative control: first param is `v`. Was never truncated by
        # the bug (rfind('?q=') never matched); stays preserved.
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        # Edge case: `q` appears *after* another param (preceded by `&`,
        # not `?`). The literal `?q=` substring never appears, so the
        # buggy heuristic never matched. Stays preserved.
        (
            "https://example.com/search?v=foo&q=bar",
            "https://example.com/search?v=foo&q=bar",
        ),
        # No query at all (the documented fused-form example URL).
        ("https://example.com", "https://example.com"),
    ],
    ids=[
        "q-param",
        "q-param-multi",
        "reddit-q-param",
        "v-param-control",
        "q-not-first-param",
        "no-query",
    ],
)
def test_handle_llm_qa_direct_call_preserves_url(
    monkeypatch, page_url, expected_fetched_url
):
    """Direct call: `_crawler_arun` receives the verbatim user URL.

    This is the regression guard. On the buggy tree the `?q=...` cases
    failed because the heuristic stripped the query string before
    `_crawler_arun` was called.
    """
    captured = {}
    _patch_boundaries(monkeypatch, captured)

    answer = asyncio.run(
        api.handle_llm_qa(
            page_url,
            "What is on this page?",
            _llm_config(),
            redis=PermitRedis(),
        )
    )

    assert answer == "answer"
    assert captured["pre_slice_urls"] == [page_url]
    assert captured["post_slice_urls"] == [expected_fetched_url], (
        f"expected {expected_fetched_url!r} preserved, "
        f"got {captured['post_slice_urls'][-1]!r}"
    )


def test_handle_llm_qa_does_not_strip_url_when_protocol_is_added(monkeypatch):
    """Bare-host URLs receiving an `https://` prefix must also retain
    their query string. The bug stripped `example.com/search?q=apple`
    (no scheme) to `example.com/search` after the https:// prepend.
    """
    captured = {}
    _patch_boundaries(monkeypatch, captured)

    answer = asyncio.run(
        api.handle_llm_qa(
            "www.google.com/search?q=apple",
            "question?",
            _llm_config(),
            redis=PermitRedis(),
        )
    )

    assert answer == "answer"
    assert captured["pre_slice_urls"] == ["https://www.google.com/search?q=apple"]
    assert captured["post_slice_urls"] == [
        "https://www.google.com/search?q=apple"
    ]


# ────────────────────────── HTTP layer ──────────────────────────────


@pytest.mark.parametrize(
    ("page_url", "expected_fetched_url", "test_id"),
    [
        (
            "https://www.google.com/search?q=apple",
            "https://www.google.com/search?q=apple",
            "q-param",
        ),
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "v-param",
        ),
    ],
    ids=["q-param", "v-param"],
)
def test_llm_qa_url_preserved(
    stock_client, server_module, monkeypatch, page_url, expected_fetched_url, test_id
):
    """HTTP-level: driving the real FastAPI app via TestClient, the URL
    reaching `_crawler_arun` (post-slice) is the verbatim user URL.

    The page URL is sent percent-encoded per convention (b) so its own
    `?q=...` arrives literally inside `url` after Starlette's `:path`
    decoder runs.  The `q-param` case was RED on the buggy tree
    (`post_slice_url == 'https://www.google.com/search'`) and is GREEN
    on the fixed tree.
    """
    captured = {}
    _patch_boundaries(monkeypatch, captured, server_module=server_module)

    encoded = urllib.parse.quote_plus(page_url, safe="")
    response = stock_client.get(
        f"/llm/{encoded}",
        params={"q": "What is on this page?"},
        headers=_auth_header(),
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"answer": "answer"}
    assert captured["pre_slice_urls"] == [page_url], (
        f"[{test_id}] percent-decoding dropped/transformed the URL: "
        f"expected {page_url!r}, got {captured['pre_slice_urls']!r}"
    )
    assert captured["post_slice_urls"] == [expected_fetched_url], (
        f"[{test_id}] expected preserved url={expected_fetched_url!r}, "
        f"got {captured['post_slice_urls'][-1]!r}"
    )


def test_percent_decoding(stock_client, server_module, monkeypatch):
    """Documents the root-cause mechanism.

    ``%3Fq%3D`` in the wire path is decoded to a literal ``?q=`` by
    Starlette's ``:path`` converter *before* ``handle_llm_qa`` runs.
    Capturing at ``validate_url_destination`` (which the handler calls
    *before* the historical slice) shows the literal ``?q=`` is present
    in ``url``.  This passes both before and after the fix and pins the
    fact that the heuristic saw a literal ``?q=`` on convention (b).
    """
    captured = {}
    _patch_boundaries(monkeypatch, captured, server_module=server_module)

    page_url = "https://www.google.com/search?q=apple"
    encoded = urllib.parse.quote_plus(page_url, safe="")
    # Sanity: the encoded path contains the %-encoded ?= pair, so the
    # wire URL is unambiguous about what Starlette must decode.
    assert "%3Fq%3D" in encoded

    response = stock_client.get(
        f"/llm/{encoded}",
        params={"q": "What is on this page?"},
        headers=_auth_header(),
    )

    assert response.status_code == 200, response.text
    assert captured["pre_slice_urls"] == [page_url]


# Traefik path sanitizing merges "//" before the app sees the path.
@pytest.mark.parametrize("path", ["/llm/https://example.com", "/llm/https:/example.com"])
def test_convention_a_fused_form_non_regression(stock_client, server_module, monkeypatch, path):
    """The documented unencoded fused form continues to work.

    Per ``docs/md_v2/assets/llm.txt/txt/docker.txt:172-174``::

        response = requests.get(
            "http://localhost:11235/llm/https://example.com?q=What is this page about?"
        )

    Here the page URL is sent *unencoded* in the path, and the LLM
    question is the literal `?q=` query string. FastAPI's `:path`
    converter stops at the first unencoded `?`, so `url` arrives as
    `https://example.com` and `q` arrives as the question. The
    (now-removed) heuristic never saw a `?q=` *inside* `url` for this
    form, so this test passes both before and after the fix; we keep
    it as the regression guard for over-correction of the removal.
    """
    captured = {}
    # Override _patch_boundaries's completion stub to capture the question
    # text reaching the LLM, so we can assert the HTTP-layer split.
    completion = _stub_completion()

    def fake_validate(url):
        captured.setdefault("pre_slice_urls", []).append(url)
        return None

    async def fake_arun(_crawler, *args, **kwargs):
        captured.setdefault("post_slice_urls", []).append(kwargs.get("url"))
        return SimpleNamespace(
            success=True,
            markdown=SimpleNamespace(fit_markdown="ctx", raw_markdown="ctx"),
        )

    async def fake_get_crawler(_bc):
        return object()

    async def fake_release_crawler(_c):
        return None

    monkeypatch.setattr(api, "validate_url_destination", fake_validate)
    monkeypatch.setattr(api, "_crawler_arun", fake_arun)
    monkeypatch.setattr(crawler_pool, "get_crawler", fake_get_crawler)
    monkeypatch.setattr(crawler_pool, "release_crawler", fake_release_crawler)
    monkeypatch.setattr(egress_broker, "enforce_egress", lambda _c: None)
    monkeypatch.setattr(llm_broker, "resolve_llm", _resolve_llm())
    monkeypatch.setattr(api, "aperform_completion_with_backoff", completion)
    monkeypatch.setattr(server_module, "redis", PermitRedis())

    response = stock_client.get(
        path,
        params={"q": "What is this page about?"},
        headers=_auth_header(),
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"answer": "answer"}
    # FastAPI's :path converter + Query("q") split: url is the path
    # only (stops at the first unencoded `?`); q is the question.
    assert captured["pre_slice_urls"] == ["https://example.com"]
    assert captured["post_slice_urls"] == ["https://example.com"]
    # The LLM question must reach the prompt intact.
    completion.assert_awaited_once()
    _, prompt_kwargs = completion.await_args
    assert "Question: What is this page about?" in prompt_kwargs["prompt_with_variables"]


def test_llm_qa_missing_q_query_is_422(stock_client, server_module, monkeypatch):
    """Sanity guard: FastAPI's `q: str = Query(...)` rejects a missing
    `q` with 422 before the handler runs. The bug report documents
    that the in-handler `if not q: raise HTTPException(400, ...)` is
    dead code; this pins that contract so nobody re-introduces a
    permissive default that would let `q` reach the handler empty.
    """
    _patch_boundaries(monkeypatch, {}, server_module=server_module)

    encoded = urllib.parse.quote_plus("https://example.com", safe="")
    response = stock_client.get(f"/llm/{encoded}", headers=_auth_header())

    assert response.status_code == 422


# ──────────── Defensive: the buggy heuristic must stay gone ──────────


def test_handle_llm_qa_source_does_not_slice_on_q_substring():
    """Source-level guard: ``handle_llm_qa`` must not contain the
    ``rfind('?q=')`` substring-slicing heuristic.  The repo's security
    suite uses source-text greps as a convenience double-check
    (``test_security_2026_04_b2.py``), and we mirror that here for the
    narrower correctness rule this file guards.  This is the early
    warning if the heuristic is ever re-introduced:
    ``test_handle_llm_qa_direct_call_preserves_url`` is the behavioral
    guard, this is the attention-grabber.
    """
    import inspect

    source = inspect.getsource(api.handle_llm_qa)
    assert "rfind('?q=')" not in source, (
        "The buggy rfind('?q=') heuristic was re-introduced into "
        "handle_llm_qa; the URL's query string would be silently "
        "stripped when the first param is `q`. See the module docstring."
    )
    assert 'rfind("?q=")' not in source, (
        "The buggy rfind(\"?q=\") heuristic was re-introduced into "
        "handle_llm_qa; the URL's query string would be silently "
        "stripped when the first param is `q`. See the module docstring."
    )
