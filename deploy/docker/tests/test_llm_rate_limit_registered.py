"""Regression: ``GET /llm/{url:path}`` must carry the slowapi per-IP limiter.

Historical bug (commit a58c800, regressed from 33a21d6a7)
--------------------------------------------------------
The Docker server wires a slowapi per-IP rate limiter (``limiter =
Limiter(...)`` in ``deploy/docker/server.py``) and decorates every
resource-intensive endpoint with ``@limiter.limit(config["rate_limiting"]
["default_limit"])`` (``"1000/minute"`` per ``deploy/docker/config.yml``).
Eight endpoints carried it: ``/md``, ``/html``, ``/screenshot``, ``/pdf``,
``/execute_js``, ``/crawl``, ``/crawl/stream``, ``/ask``.

A server rewrite (commit a58c800, "refactor(server): migrate to pool-based
crawler management") re-added ``GET /llm/{url:path}`` -- renaming the path
param ``{input:path}`` -> ``{url:path}`` -- *without* the
``@limiter.limit`` decorator the route had carried since the original
rate-limit setup (33a21d6a7), and that the same commit retained on the
sibling ``GET /md/{url:path}`` route. The decorator was simply dropped
during the copy-forward.

Because the app does **not** register ``SlowAPIMiddleware`` /
``app.state.limiter`` / ``_rate_limit_exceeded_handler`` and does not set
application-wide ``default_limits`` on the running app
(``limiter._application_limits == []``; see ``slowapi/extension.py``
``_check_request_limit``), the ``@limiter.limit`` decorator is the *only*
enforcement path. An undecorated route bypasses slowapi entirely: no
per-IP counter increment, no ``429``, no defense-in-depth layer, and a
per-IP accounting blind spot for any operator monitoring keyed on slowapi
counters. ``/llm`` was the sole resource-intensive endpoint in that state.

The limiter *is* otherwise wired correctly end-to-end:
``slowapi.errors.RateLimitExceeded`` subclasses
``starlette.exceptions.HTTPException`` and carries ``status_code=429``, so
the server's central ``@app.exception_handler(_StarletteHTTPException)``
(``server.py``) converts it into a ``429`` JSONResponse without needing a
dedicated handler. This is exactly why the sibling ``/md`` route returns
``429`` past the cap, and what ``/llm`` must now do too.

Coverage
--------
* ``test_llm_endpoint_is_registered_with_slowapi`` -- the structural guard,
  mirroring the conclusive evidence in the bug report: before the fix,
  ``any(k.endswith("llm_endpoint") for k in limiter._route_limits)`` was
  ``False``. After the fix it must be ``True``.
* ``test_every_resource_endpoint_carries_slowapi`` -- the operator-intent
  guard: every resource-intensive route function name must be present in
  ``limiter._route_limits``. Stops the same drop from recurring on ``/llm``
  or any sibling.
* ``test_llm_endpoint_limit_matches_configured_default`` -- the decorator
  must be applied with the configured ``default_limit`` value, not some
  other (e.g. tighter or absent) limit.
* ``test_llm_endpoint_returns_429_past_per_ip_limit`` -- the behavioral
  guard: driving the real FastAPI app past the per-IP cap returns ``429``,
  the same behavior as ``/md``.
* ``test_llm_endpoint_unauthenticated_still_401`` -- auth-posture guard:
  the added decorator must sit *after* ``AuthGateMiddleware`` (the
  outermost ASGI layer), so an anonymous ``/llm`` call still ``401``s
  before slowapi runs -- the decorator must not move rate-limiting ahead
  of authentication.
"""
import pytest

pytestmark = pytest.mark.posture

# Function names that slowapi keys route limits on (``"{module}.{__name__}"``
# of the decorated coroutine, per ``slowapi/extension.py``). Every
# resource-intensive endpoint the server exposes must appear here so a
# dropped ``@limiter.limit`` on any one of them fails this test.
RESOURCE_LIMITED_ROUTE_FUNCS = [
    "server.get_markdown",        # POST /md
    "server.generate_html",       # POST /html
    "server.generate_screenshot", # POST /screenshot
    "server.generate_pdf",        # POST /pdf
    "server.execute_js",          # POST /execute_js
    "server.crawl",               # POST /crawl
    "server.crawl_stream",        # POST /crawl/stream
    "server.get_context",        # GET  /ask
    "server.llm_endpoint",        # GET  /llm/{url:path}  <-- the bug
]


def _auth_header():
    from auth import create_access_token

    return {"Authorization": f"Bearer {create_access_token({'sub': 'u@x.com'})}"}


# ─────────────────────── structural guards ──────────────────────────


def test_llm_endpoint_is_registered_with_slowapi(server_module):
    """``/llm``'s handler must be registered with the slowapi limiter.

    This is the direct regression guard for the reported bug. Before the
    fix, ``server.llm_endpoint`` was absent from ``limiter._route_limits``
    and slowapi enforcement was structurally impossible for ``/llm``
    regardless of request rate.
    """
    keys = server_module.limiter._route_limits.keys()
    assert "server.llm_endpoint" in keys, (
        "GET /llm is not registered with slowapi -- the @limiter.limit "
        "decorator is missing on llm_endpoint; the route bypasses the "
        "per-IP rate limiter that every other resource endpoint enforces."
    )


def test_every_resource_endpoint_carries_slowapi(server_module):
    """Every resource-intensive route function must be in ``_route_limits``.

    Guards the operator intent ("decorate every resource endpoint") so the
    same copy-forward drop that lost the decorator on ``/llm`` cannot
    recur on ``/llm`` or any sibling without this test going red.
    """
    registered = set(server_module.limiter._route_limits.keys())
    missing = [name for name in RESOURCE_LIMITED_ROUTE_FUNCS if name not in registered]
    assert not missing, (
        f"Resource endpoint(s) missing the @limiter.limit decorator: {missing}. "
        f"Registered route limits: {sorted(registered)}"
    )


def test_llm_endpoint_limit_matches_configured_default(server_module):
    """The ``/llm`` decorator must use the configured ``default_limit``.

    Catches a regression where the decorator is added but with the wrong
    limit string (e.g. a per-route override, or a hard-coded value that
    diverges from the operator-tunable ``rate_limiting.default_limit``).
    """
    from limits import parse

    expected = server_module.config["rate_limiting"]["default_limit"]
    expected_item = parse(expected)
    limits_for_llm = server_module.limiter._route_limits["server.llm_endpoint"]
    assert limits_for_llm, "server.llm_endpoint registered with no limit items"
    applied = limits_for_llm[0].limit
    assert applied.amount == expected_item.amount, (
        f"/llm limiter amount {applied.amount} != configured default "
        f"{expected!r} (amount {expected_item.amount})"
    )


# ─────────────────────── behavioral guards ─────────────────────────


def test_llm_endpoint_returns_429_past_per_ip_limit(
    stock_client, server_module, monkeypatch
):
    """Driving ``/llm`` past the per-IP cap must return ``429``.

    Proves the end-to-end chain: ``@limiter.limit`` -> ``RateLimitExceeded``
    (``status_code=429``, subclass of ``starlette.exceptions.HTTPException``)
    -> the central ``_http_exception_handler`` -> ``429`` JSONResponse.
    The route's limit is temporarily swapped to a tiny value so the test
    is fast, deterministic and offline; the handler is stubbed so each
    under-limit call is cheap (slowapi runs *before* the handler).
    """
    from limits import parse_many
    from slowapi.wrappers import Limit

    # Stub the handler so under-limit calls return 200 without touching
    # DNS, the browser pool, Redis or any LLM provider. slowapi's check
    # runs before the handler, so the 429 path is exercised regardless.
    async def fake_handle_llm_qa(*_args, **_kwargs):
        return "answer"

    monkeypatch.setattr(server_module, "handle_llm_qa", fake_handle_llm_qa)

    # Swap the route limit to a small, deterministic value (3/minute) and
    # reset the in-memory storage so the test starts from a clean counter.
    # We reuse the registered limit's own key_func/scope so the counter
    # key matches what the decorator will look up at request time.
    src = server_module.limiter._route_limits["server.llm_endpoint"][0]
    small = next(iter(parse_many("3/minute")))
    monkeypatch.setattr(
        server_module.limiter,
        "_route_limits",
        {
            **server_module.limiter._route_limits,
            "server.llm_endpoint": [
                Limit(
                    limit=small,
                    key_func=src.key_func,
                    scope=None,
                    per_method=src.per_method,
                    methods=src.methods,
                    error_message=src.error_message,
                    exempt_when=None,
                    cost=src.cost,
                    override_defaults=src.override_defaults,
                )
            ],
        },
    )
    server_module.limiter.reset()

    headers = _auth_header()
    statuses = []
    for _ in range(5):
        r = stock_client.get(
            "/llm/https://example.com", params={"q": "what"}, headers=headers
        )
        statuses.append(r.status_code)

    assert statuses[:3] == [200, 200, 200], (
        f"under-limit /llm calls were not 200 (statuses={statuses}); "
        "the stubbed handler should have served them normally"
    )
    assert statuses[3] == 429 and statuses[4] == 429, (
        f"/llm did not return 429 past the per-IP cap (statuses={statuses}); "
        "slowapi rate limiting is not engaged on /llm -- the "
        "@limiter.limit decorator is missing or not wired."
    )
    # The 429 body comes from the central handler's {"detail": ...} shape.
    assert "detail" in stock_client.get(
        "/llm/https://example.com", params={"q": "what"}, headers=headers
    ).json()


def test_llm_endpoint_unauthenticated_still_401(stock_client, server_module, monkeypatch):
    """The added limiter must not move rate-limiting ahead of auth.

    ``AuthGateMiddleware`` is the outermost ASGI layer and must ``401`` an
    anonymous ``/llm`` call *before* slowapi's decorator ever runs (the
    decorator only fires once the route is dispatched, which is past auth).
    This is the threat-model guarantee for the bug: the marginal fix is a
    second defense-in-depth throttle for *authenticated* callers, not a
    change to the authentication order.
    """
    async def fake_handle_llm_qa(*_args, **_kwargs):
        return "answer"

    monkeypatch.setattr(server_module, "handle_llm_qa", fake_handle_llm_qa)
    server_module.limiter.reset()

    # No Authorization header -> AuthGateMiddleware short-circuits with 401
    # before the route (and thus before @limiter.limit) is reached.
    r = stock_client.get("/llm/https://example.com", params={"q": "what"})
    assert r.status_code == 401, (
        f"anonymous /llm returned {r.status_code} (expected 401); the "
        "limiter decorator must not run before AuthGateMiddleware"
    )
