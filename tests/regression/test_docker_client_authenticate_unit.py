"""Regression coverage for the Docker client ``/token`` issuance contract.

The hardened (0.9.0+) ``deploy/docker/server.py`` ``/token`` handler requires
the server's operator ``CRAWL4AI_API_TOKEN`` in the request body (it mints a
JWT over it rather than accepting it as a Bearer credential). The shipped
client's ``authenticate(email)`` previously POSTed only ``{"email": email}``,
so the server's ``constant_time_eq`` guard returned ``401 "Invalid or missing
api_token"`` before a token was ever issued, leaving the documented JWT
issuance path unusable on every JWT-enabled 0.9.0+ deployment (the bug was
introduced in commit 60886d1a, the 0.9.0 secure-by-default hardening).

These tests pin the fixed body contract (``email`` + ``api_token``) against
the server's ``TokenRequest`` schema and the end-to-end JWT issuance against
the real ``server.app``, plus the server's fail-closed guard chain
(401/403) surfaced on the client as ``ConnectionError``. Mirrors
``tests/regression/test_docker_client_hooks_unit.py`` for the ``/token``
body, the contract that had no test coverage when the server was hardened.
No browser, network, Redis, or live server required — the real
``server.app`` is driven in-process via ``httpx.ASGITransport`` (which does
not run the FastAPI lifespan, so no Chromium is launched and no Redis
connection is opened at startup).
"""

import json
import sys
from pathlib import Path

import httpx
import jwt
import pytest
import pytest_asyncio

# deploy/docker holds ``auth`` (the /token request schema) and ``server`` (the
# real handler); put it on sys.path so we can cross-validate the client body
# against the genuine ``TokenRequest`` and drive the real ``/token`` handler
# without starting the server — mirroring tests/regression/test_docker_client_hooks_unit.py.
_DEPLOY_DOCKER = Path(__file__).resolve().parents[2] / "deploy" / "docker"
if str(_DEPLOY_DOCKER) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DOCKER))

import auth  # noqa: E402
import server  # noqa: E402

from crawl4ai.docker_client import (  # noqa: E402
    Crawl4aiClientError,
    Crawl4aiDockerClient,
    ConnectionError,
)

OPERATOR_TOKEN = "operator-secret-token-0123456789"
TEST_SECRET = "test-only-secret-key-value-000000"


# --- helpers ---------------------------------------------------------------
def _jwt_enabled_env(monkeypatch, operator_token=OPERATOR_TOKEN):
    """Configure the real server env for a JWT-enabled posture, per-request.

    ``_current_jwt_enabled`` / ``_current_api_token`` / ``resolve_secret_key``
    all re-read the environment at request time, so monkeypatching these before
    the request hits the handler is sufficient (no re-import needed).
    """
    monkeypatch.setenv("CRAWL4AI_JWT_ENABLED", "true")
    monkeypatch.setenv("CRAWL4AI_API_TOKEN", operator_token)
    monkeypatch.setenv("SECRET_KEY", TEST_SECRET)


def _stub_email_ok(monkeypatch):
    """Stub verify_email_domain so the /token handler runs fully offline."""
    monkeypatch.setattr(server, "verify_email_domain", lambda _email: True)


def _install_mock(client, handler):
    """Swap a client's transport for a MockTransport, closing the default first.

    Mirrors the ``mock_client``/``streaming_client`` fixtures in
    test_docker_client_hooks_unit.py so the constructor's default
    ``httpx.AsyncClient`` is not orphaned.
    """
    client._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
        headers={"Content-Type": "application/json"},
    )


# --- /token request-schema cross-contract ---------------------------------
# Mirrors test_docker_client_hooks_unit.py:55 (the server hook schema pin).
# The client body must round-trip through the server's TokenRequest with the
# api_token populated; before the fix, the body lacked the field the handler's
# constant_time_eq guard requires.

def test_token_request_schema_has_email_and_api_token_fields():
    assert set(auth.TokenRequest.model_fields) == {"email", "api_token"}


# --- wire contract: authenticate() posts both email and api_token ---------
@pytest.mark.asyncio
async def test_authenticate_posts_email_and_api_token_in_body():
    """The /token POST body carries both ``email`` and ``api_token``.

    Cross-contract: the body round-trips through the server's ``TokenRequest``
    with both fields populated, so the handler's ``constant_time_eq`` guard has
    a non-empty ``req.api_token`` to compare. Before the fix only
    ``{"email": email}`` was sent (the exact regression this test pins).
    """
    captured = {}

    def handler(request):
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "email": "user@example.com",
                "access_token": "jwt-abc",
                "token_type": "bearer",
            },
        )

    client = Crawl4aiDockerClient(base_url="http://testserver", verbose=False)
    await client._http_client.aclose()
    _install_mock(client, handler)
    try:
        await client.authenticate("user@example.com", api_token=OPERATOR_TOKEN)
    finally:
        if not client._http_client.is_closed:
            await client._http_client.aclose()

    body = json.loads(captured["body"])
    assert set(body) == {"email", "api_token"}
    req = auth.TokenRequest.model_validate(body)  # round-trips server schema
    assert req.email == "user@example.com"
    assert req.api_token == OPERATOR_TOKEN


@pytest.mark.asyncio
async def test_authenticate_reuses_constructor_api_token_in_body():
    """A constructor ``api_token=`` is reused as the /token body credential.

    Pins the ``api_token or self._token`` fallback so the documented flow
    "construct with ``api_token=`` then call ``authenticate(email)``" sends the
    constructor's operator token in the body, matching the explicit-arg path.
    """
    captured = {}

    def handler(request):
        captured["body"] = request.read().decode()
        return httpx.Response(
            200, json={"email": "u@x.com", "access_token": "jwt", "token_type": "bearer"}
        )

    client = Crawl4aiDockerClient(
        base_url="http://testserver", verbose=False, api_token="constructor-token-0123456789"
    )
    await client._http_client.aclose()
    _install_mock(client, handler)
    try:
        await client.authenticate("u@x.com")
        body = json.loads(captured["body"])
        assert body["email"] == "u@x.com"
        assert body["api_token"] == "constructor-token-0123456789"
    finally:
        if not client._http_client.is_closed:
            await client._http_client.aclose()


# --- client-side guard: authenticate() without an operator token ----------
@pytest.mark.asyncio
async def test_authenticate_without_operator_token_raises_connection_error_before_request():
    """Bare-constructor ``authenticate(email)`` raises *before* any request.

    The documented bare ``Crawl4aiDockerClient(base_url=...).authenticate(email)``
    call has no operator token to send; surfacing a clear ``ConnectionError``
    up front prevents the cryptic 401 ("Invalid or missing api_token") the
    pre-fix body produced against a JWT-enabled server. The request is never
    dispatched (the MockTransport handler is never hit).
    """
    sent = {}

    def handler(_request):
        sent["hit"] = True
        return httpx.Response(401, json={"detail": "Invalid or missing api_token"})

    client = Crawl4aiDockerClient(base_url="http://testserver", verbose=False)
    await client._http_client.aclose()
    _install_mock(client, handler)
    try:
        with pytest.raises(ConnectionError) as exc_info:
            await client.authenticate("user@example.com")
        assert isinstance(exc_info.value, Crawl4aiClientError)
        assert "operator API token" in str(exc_info.value)
        assert "api_token" in str(exc_info.value)
        assert sent.get("hit") is not True  # no request was dispatched
    finally:
        if not client._http_client.is_closed:
            await client._http_client.aclose()


# --- end-to-end against the real server.app (in-process ASGI, no lifespan) -
@pytest_asyncio.fixture
async def asgi_client(monkeypatch):
    """A client whose transport is the real ``server.app`` (no lifespan).

    ``verify_email_domain`` is stubbed so the ``/token`` handler passes the
    MX-record guard offline. The ``AuthGateMiddleware`` admits ``/token`` as a
    public path, so no Bearer credential is required to reach the handler; the
    handler validates the body ``api_token`` itself. No browser or Redis is
    started: ``httpx.ASGITransport`` does not run the FastAPI lifespan. The
    constructor's default ``httpx.AsyncClient`` is closed up front (it is never
    used) and whatever client is installed at teardown is closed in the same
    event loop.
    """
    _stub_email_ok(monkeypatch)
    c = Crawl4aiDockerClient(base_url="http://testserver", verbose=False)
    await c._http_client.aclose()
    c._http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url="http://testserver",
        headers={"Content-Type": "application/json"},
    )
    try:
        yield c
    finally:
        if c._http_client is not None and not c._http_client.is_closed:
            await c._http_client.aclose()


@pytest.mark.asyncio
async def test_authenticate_against_real_server_mints_real_jwt(asgi_client, monkeypatch):
    """The documented JWT flow mints a real HS256 JWT against the real handler.

    The exact call shape the constructor docstring describes — pass the
    operator ``api_token`` and call ``authenticate(email)`` — must return 200
    from the real ``/token`` handler (driven in-process via ASGITransport),
    store the JWT, and set ``Authorization: Bearer <jwt>``. Before the fix the
    same handler returned 401 ("Invalid or missing api_token") and the client
    raised ``ConnectionError``.
    """
    _jwt_enabled_env(monkeypatch)

    await asgi_client.authenticate("user@example.com", api_token=OPERATOR_TOKEN)

    token = asgi_client._token
    assert token is not None
    assert token.count(".") == 2  # header.payload.signature (HS256)
    assert asgi_client._http_client.headers["Authorization"] == f"Bearer {token}"

    claims = jwt.decode(token, TEST_SECRET, algorithms=["HS256"])
    assert claims["sub"] == "user@example.com"
    assert claims["scope"] == "data"


# --- fail-closed guard chain: every server mode that isn't a successful JWT -
# Walking the /token handler (server.py:663-679) shows there is no branch in
# which the pre-fix {"email": email} body could succeed. These tests pin each
# branch so the client surfaces the server's fail-closed responses as
# ConnectionError (with the branch detail, not a leaked httpx.HTTPStatusError)
# and the body contract can never silently regress.

@pytest.mark.asyncio
async def test_authenticate_with_wrong_token_surfaces_server_401_as_connection_error(
    asgi_client, monkeypatch
):
    """A wrong body ``api_token`` surfaces the server's 401 as ``ConnectionError``.

    This is the original bug's symptom: the operator-token guard
    (``constant_time_eq``) returns 401 when the body token doesn't match. The
    fixed client now reaches a *real* comparison (a non-empty body token) instead
    of the pre-fix missing-token 401, and the 4xx is normalized to the client's
    own ``ConnectionError`` with the server's JSON ``detail`` (never a leaked
    ``httpx.HTTPStatusError``).
    """
    _jwt_enabled_env(monkeypatch)

    with pytest.raises(ConnectionError) as exc_info:
        await asgi_client.authenticate(
            "user@example.com", api_token="wrong-token-not-the-operator"
        )
    assert isinstance(exc_info.value, Crawl4aiClientError)
    assert not isinstance(exc_info.value, httpx.HTTPStatusError)
    assert "401" in str(exc_info.value)
    assert "Invalid or missing api_token" in str(exc_info.value)


@pytest.mark.asyncio
async def test_authenticate_on_jwt_disabled_server_surfaces_403_as_connection_error(
    asgi_client, monkeypatch
):
    """JWT disabled -> server 403 "JWT issuance is disabled" -> ``ConnectionError``.

    Default config ships with ``jwt_enabled: false``; the handler short-circuits
    with 403 before the api_token guard, and the client surfaces it as
    ``ConnectionError`` (the bug's "no branch succeeds" analysis, branch 1).
    """
    monkeypatch.setenv("CRAWL4AI_JWT_ENABLED", "false")
    monkeypatch.setenv("CRAWL4AI_API_TOKEN", OPERATOR_TOKEN)
    monkeypatch.setenv("SECRET_KEY", TEST_SECRET)

    with pytest.raises(ConnectionError) as exc_info:
        await asgi_client.authenticate("user@example.com", api_token=OPERATOR_TOKEN)
    assert "403" in str(exc_info.value)
    assert "JWT issuance is disabled" in str(exc_info.value)


@pytest.mark.asyncio
async def test_authenticate_with_no_operator_token_configured_surfaces_403_as_connection_error(
    asgi_client, monkeypatch
):
    """JWT enabled but no operator api_token -> 403 "Token issuance is disabled".

    The 0.9.0 fail-closed guard: the pre-0.9.0 permissive check was tightened
    to refuse issuance when the server has no operator token configured, so a
    missing body api_token can never mint a JWT. The client surfaces the
    server's 403 as ``ConnectionError`` (the bug's "no branch succeeds"
    analysis, branch 2).
    """
    monkeypatch.setenv("CRAWL4AI_JWT_ENABLED", "true")
    monkeypatch.delenv("CRAWL4AI_API_TOKEN", raising=False)
    monkeypatch.setenv("SECRET_KEY", TEST_SECRET)

    with pytest.raises(ConnectionError) as exc_info:
        await asgi_client.authenticate(
            "user@example.com", api_token="should-not-help-without-server-token"
        )
    assert "403" in str(exc_info.value)
    assert "no operator API token is configured" in str(exc_info.value)
