"""Regression coverage for the Docker client PDF base64-decode contract.

The Docker server base64-encodes ``CrawlResult.pdf`` bytes into a JSON string
for transport on both ``/crawl`` and ``/crawl/stream`` (``deploy/docker/api.py``
calls ``b64encode(result_dict['pdf']).decode('utf-8')``), introduced in commit
``1b6a31f`` ("fix: encode PDF results to base64 in /crawl endpoint. ref #1301").
That commit added no matching client-side ``b64decode``, so
``Crawl4aiDockerClient.crawl()`` reconstructed ``CrawlResult(**server_dict)``
feeding the base64 *string* straight into the ``pdf: Optional[bytes]`` field.
pydantic v2 silently UTF-8-coerces a ``str`` to ``bytes``, so callers got
corrupt data (``b'JVBE...'`` — the base64 of ``%PDF`` — instead of ``b'%PDF...'``)
with no exception. These tests pin the decode on both crawl paths. No browser,
network, or Redis required.
"""

import asyncio
import json
import sys
from base64 import b64encode
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

# deploy/docker holds the server encoding contract this client must undo; put
# it on sys.path so a future server schema change is caught here without
# starting the server.
_DEPLOY_DOCKER = Path(__file__).resolve().parents[2] / "deploy" / "docker"
if str(_DEPLOY_DOCKER) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DOCKER))

from crawl4ai.docker_client import Crawl4aiDockerClient  # noqa: E402
from crawl4ai.async_configs import CrawlerRunConfig  # noqa: E402
from crawl4ai.models import CrawlResult  # noqa: E402


# Realistic PDF payload: starts with the %PDF magic the contract keys on.
ORIGINAL_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< >>\n%%EOF"
# The exact server transform from deploy/docker/api.py:1003 / 1177.
B64_PDF = b64encode(ORIGINAL_PDF).decode("utf-8")


@pytest.fixture
def client():
    """A client whose HTTP transport never touches the network.

    Mirrors the fixture in test_docker_client_hooks_unit.py. The constructor's
    ``httpx.AsyncClient`` is harmless without I/O and is closed to avoid
    resource warnings.
    """
    c = Crawl4aiDockerClient(base_url="http://localhost:0", verbose=False)
    try:
        yield c
    finally:
        asyncio.run(c.close())


def _install_non_streaming(monkeypatch, results_payload, success=True):
    async def fake_check_server(_self):
        return None

    async def fake_request(_self, method, endpoint, **kwargs):
        class FakeResponse:
            def json(self_inner):
                return {"success": success, "results": results_payload}

        return FakeResponse()

    monkeypatch.setattr(Crawl4aiDockerClient, "_check_server", fake_check_server)
    monkeypatch.setattr(Crawl4aiDockerClient, "_request", fake_request)


@pytest_asyncio.fixture
async def streaming_client(monkeypatch):
    async def fake_check_server(_self):
        return None

    monkeypatch.setattr(Crawl4aiDockerClient, "_check_server", fake_check_server)
    c = Crawl4aiDockerClient(base_url="http://test", verbose=False)
    await c._http_client.aclose()
    try:
        yield c
    finally:
        if c._http_client is not None and not c._http_client.is_closed:
            await c._http_client.aclose()


def _install_asgi(client, app):
    client._http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        headers={"Content-Type": "application/json"},
    )


def _asgi_ndjson(lines):
    async def app(scope, receive, send):
        body = ("".join(line + "\n" for line in lines)).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/x-ndjson")]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
    return app


# --- non-streaming path: crawl() -> _request -> CrawlResult(**_decode_pdf(r)) ---

@pytest.mark.asyncio
async def test_non_streaming_crawl_decodes_base64_pdf(client, monkeypatch):
    """A base64 ``pdf`` string from the server yields raw PDF bytes.

    Before the fix, pydantic v2 silently UTF-8-coerced the base64 text to
    ``bytes`` (``b'JVBE...'``), so ``result.pdf[:4]`` was ``b'JVBE'`` (the
    base64 of ``%PDF``) instead of ``b'%PDF'``.
    """
    _install_non_streaming(monkeypatch, [
        {"url": "https://x", "html": "", "success": True, "pdf": B64_PDF},
    ])
    result = await client.crawl(["https://x"], crawler_config=CrawlerRunConfig())
    assert isinstance(result, CrawlResult)
    assert isinstance(result.pdf, bytes)
    assert result.pdf == ORIGINAL_PDF
    assert result.pdf[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_non_streaming_crawl_preserves_none_pdf(client, monkeypatch):
    """A result with ``pdf: None`` (the common case — pdf not requested) is
    not decoded and stays None, so the decode guard never breaks the no-PDF
    path."""
    _install_non_streaming(monkeypatch, [
        {"url": "https://x", "html": "", "success": True, "pdf": None},
    ])
    result = await client.crawl(["https://x"], crawler_config=CrawlerRunConfig())
    assert result.pdf is None


# --- streaming path: aiter_lines -> json.loads -> CrawlResult(**_decode_pdf(r)) ---

@pytest.mark.asyncio
async def test_streaming_crawl_decodes_base64_pdf(streaming_client):
    """The streaming path decodes the base64 ``pdf`` just like the
    non-streaming path."""
    lines = [json.dumps({"url": "https://x", "html": "", "success": True, "pdf": B64_PDF})]
    _install_asgi(streaming_client, _asgi_ndjson(lines))

    gen = await streaming_client.crawl(
        ["https://x"], crawler_config=CrawlerRunConfig(stream=True)
    )
    yielded = [r async for r in gen]

    assert len(yielded) == 1
    assert isinstance(yielded[0].pdf, bytes)
    assert yielded[0].pdf == ORIGINAL_PDF
    assert yielded[0].pdf[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_streaming_crawl_skips_control_lines_and_decodes_results(streaming_client):
    """Completion/error control lines are still skipped, and only result lines
    are decoded — guards against a regression that yields control lines or
    drops the decode on the streaming branch."""
    lines = [
        json.dumps({"status": "completed"}),                                   # skipped
        json.dumps({"error": "boom", "url": "https://failed.example.com"}),     # skipped (logged)
        json.dumps({"url": "https://ok.example.org", "html": "", "success": True, "pdf": B64_PDF}),
        json.dumps({"url": "https://none.example.net", "html": "", "success": True, "pdf": None}),
        json.dumps({"status": "completed"}),                                   # skipped
    ]
    _install_asgi(streaming_client, _asgi_ndjson(lines))

    gen = await streaming_client.crawl(
        ["https://ok.example.org", "https://none.example.net"],
        crawler_config=CrawlerRunConfig(stream=True),
    )
    yielded = [r async for r in gen]

    assert [r.url for r in yielded] == ["https://ok.example.org", "https://none.example.net"]
    assert yielded[0].pdf == ORIGINAL_PDF
    assert yielded[0].pdf[:4] == b"%PDF"
    assert yielded[1].pdf is None
