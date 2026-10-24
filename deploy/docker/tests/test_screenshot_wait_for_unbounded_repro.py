"""Regression suite for the unbounded ``screenshot_wait_for`` on the
``/screenshot`` endpoint's ``ScreenshotRequest`` body (introduced in 5297e362).

Bug
---
Commit 5297e362 added ``/screenshot`` with ``screenshot_wait_for: float | None = 2``
(no ``ge``/``le``) and passed ``body.screenshot_wait_for`` straight into
``CrawlerRunConfig(**kwargs)`` via ``api.server_crawler_config`` -- the *trusted*
constructor path, so ``crawl4ai.async_configs._clamp_untrusted`` (which caps
``screenshot_wait_for`` at ``_MAX_DELAY_SECONDS = 5.0``) never ran.  The
unbounded value flowed into a bare ``await asyncio.sleep(config.screenshot_wait_for)``
in ``async_crawler_strategy._crawl_web`` with no deadline, and the handler holds a
``GLOBAL_SEM`` crawl slot (default ``max_pages: 12``) for the whole sleep.  A single
authenticated ``POST /screenshot`` with ``{"screenshot_wait_for": 1e9}`` parked a
slot for ~31 years; ``max_pages`` concurrent such requests pinned the entire
browser/crawl capacity indefinitely -- a trivial DoS that also starved
``/crawl``, ``/pdf``, ``/html`` and ``/md``.  None of the compensating layers
(governor ``wall_clock_seconds`` is scoped to ``/crawl`` only; the janitor floor
is ~6h and never cancels the awaiting handler; gunicorn ``--timeout`` does not
reap hung async handlers) were scoped to this endpoint.  The project's *intent*
that client-influenced ``screenshot_wait_for`` must never exceed 5s is pinned by
``tests/regression/test_untrusted_config_security.py`` -- but only for the
``/crawl`` deserialization path.  ``/screenshot`` defeated that intent by
constructing the config server-side through the trusted constructor.

Fix
---
Bound the field at the request schema to match the existing clamp behavior::

    screenshot_wait_for: float | None = Field(default=2, ge=0, le=5)

This matches ``_MAX_DELAY_SECONDS = 5.0`` and mirrors the ``ge``/``le`` precedent
already used by ``MarkdownRequest.temperature`` (``Field(None, ge=0.0, le=2.0)``)
and ``server.py``'s ``score_ratio`` query param (``Query(0.5, ge=0.0, le=1.0)``).
An out-of-range or non-finite value now fails schema validation (HTTP 422,
``RequestValidationError``) before the route handler runs, so neither
``crawler.arun`` nor the bare ``asyncio.sleep`` is ever invoked with the
attacker's value.

Coverage
--------
* Schema-level rejection (direct pydantic) for finite out-of-range and
  non-finite inputs; acceptance of in-range bounds (``0.0`` + ``5.0``), ``None``,
  ints, and omission (default ``2``).
* The non-finite JSON-parser arm: stdlib ``json`` accepts ``NaN`` literals, so
  the only thing standing between a ``NaN`` body value and the bare sleep is the
  new ``ge``/``le`` bound.
* End-to-end via Starlette ``TestClient``: finite out-of-range values are
  rejected with **422** at the schema layer (not a handler 500, not a hang).
* End-to-end "handler never reached" for the DoS payload (``1e9``) and for
  non-finite values: ``RequestValidationError`` fires before the route handler,
  so ``server_crawler_config``/``get_crawler``/``crawler.arun``/``asyncio.sleep``
  are never invoked -- the bug report's actual safety guarantee.
* Happy-path: an in-range value is NOT over-rejected -- ``server_crawler_config``
  receives the value intact.
* The trusted-path constructor (``server_crawler_config``) is intentionally
  unbounded by design; the schema bound is what gates the client-reachable
  surface (pinned so a future change to clamp it is a conscious decision).
* The OpenAPI contract advertised at ``/schema`` carries ``minimum: 0`` /
  ``maximum: 5`` on ``screenshot_wait_for``; the bound matches
  ``_MAX_DELAY_SECONDS`` so drift between the two surfaces surfaces here.
"""
import json

import pytest
from auth import create_access_token
from pydantic import ValidationError


def _bearer() -> dict:
    """A valid bearer header for the in-process test principal."""
    return {"Authorization": f"Bearer {create_access_token({'sub': 'test@example.com'})}"}


# Matches the bug report's exploit payload and the project's existing untrusted
# clamp regression test (tests/regression/test_untrusted_config_security.py).
_OUT_OF_RANGE = [-0.1, 5.1, 10.0, 99999.0, 1e9, -5.0]
_NON_FINITE = [float("nan"), float("inf"), float("-inf")]
_IN_RANGE = [0.0, 0.7, 2.0, 5.0]


# --------------------------------------------------------------------------- #
# Section 1 -- Schema-level rejection on the POST /screenshot body surface
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", _OUT_OF_RANGE, ids=[repr(v) for v in _OUT_OF_RANGE])
def test_screenshot_request_rejects_out_of_range_wait(value):
    """A finite out-of-range ``screenshot_wait_for`` must fail schema validation.

    This is the exact DoS payload the bug report cites (``1e9``): with the fix
    it is rejected at the schema layer instead of flowing into
    ``await asyncio.sleep(1e9)``.
    """
    from schemas import ScreenshotRequest

    with pytest.raises(ValidationError) as exc:
        ScreenshotRequest(url="https://example.com", screenshot_wait_for=value)
    errors = exc.value.errors()
    assert any(err["loc"][-1] == "screenshot_wait_for" for err in errors), errors
    # The constraint message must be the bound, not a type error: this pins
    # that the rejection is because of ge/le (the fix), not a side effect.
    msg = json.dumps(errors, default=str)
    assert (
        "less than or equal to 5" in msg or "greater than or equal to 0" in msg
    ), msg


@pytest.mark.parametrize("value", _NON_FINITE, ids=[repr(v) for v in _NON_FINITE])
def test_screenshot_request_rejects_non_finite_wait(value):
    """Non-finite ``screenshot_wait_for`` (nan/inf/-inf) must fail schema validation.

    The point of bounding the field is precisely to keep these from reaching
    the bare ``asyncio.sleep``: ``asyncio.sleep(float('inf'))`` would never
    return, and the only guard is the schema bound.
    """
    from schemas import ScreenshotRequest

    with pytest.raises(ValidationError) as exc:
        ScreenshotRequest(url="https://example.com", screenshot_wait_for=value)
    assert any(err["loc"][-1] == "screenshot_wait_for" for err in exc.value.errors())


@pytest.mark.parametrize("value", _IN_RANGE, ids=[repr(v) for v in _IN_RANGE])
def test_screenshot_request_accepts_in_range_wait(value):
    """In-range values (incl. both bounds 0.0 and 5.0) must be accepted."""
    from schemas import ScreenshotRequest

    req = ScreenshotRequest(url="https://example.com", screenshot_wait_for=value)
    assert req.screenshot_wait_for == value


def test_screenshot_request_accepts_int_in_range():
    """Integer in-range values must coerce to float and be accepted
    (regression guard: a strict-mode bound would reject legitimate ints)."""
    from schemas import ScreenshotRequest

    assert ScreenshotRequest(
        url="https://example.com", screenshot_wait_for=3
    ).screenshot_wait_for == 3.0
    assert ScreenshotRequest(
        url="https://example.com", screenshot_wait_for=0
    ).screenshot_wait_for == 0.0


def test_screenshot_request_accepts_omitted_wait():
    """Omitting ``screenshot_wait_for`` must default to ``2`` (no regression)."""
    from schemas import ScreenshotRequest

    req = ScreenshotRequest(url="https://example.com")
    assert req.screenshot_wait_for == 2


def test_screenshot_request_accepts_none_wait():
    """An explicit ``None`` is accepted (the crawler treats it as no-extra-wait)."""
    from schemas import ScreenshotRequest

    req = ScreenshotRequest(url="https://example.com", screenshot_wait_for=None)
    assert req.screenshot_wait_for is None


# --------------------------------------------------------------------------- #
# Section 2 -- Non-finite JSON-parser arm (bug report Evidence: stdlib json)
# --------------------------------------------------------------------------- #

def test_non_finite_wait_rejected_even_when_json_parser_accepts_it():
    """The venv has no ``orjson``/``ujson``; Starlette falls back to stdlib
    ``json``, which accepts the ``NaN`` / ``Infinity`` / ``-Infinity`` literals
    by default.  Without the ``ge``/``le`` bound this would flow straight into
    ``asyncio.sleep``.  This test pins that the *only* thing stopping it is the
    schema bound, by parsing a raw ``NaN`` literal the way the server-side
    parser does and asserting pydantic then rejects it.
    """
    from schemas import ScreenshotRequest

    # Sanity: confirm the parser in use accepts the literal.  If a future
    # environment installs orjson (which rejects these), this assertion flips
    # and the test below still holds vacuously.
    parsed = json.loads('{"url":"https://example.com","screenshot_wait_for":NaN}')
    assert parsed["screenshot_wait_for"] != parsed["screenshot_wait_for"]  # NaN != NaN

    with pytest.raises(ValidationError) as exc:
        ScreenshotRequest(**parsed)
    assert any(err["loc"][-1] == "screenshot_wait_for" for err in exc.value.errors())


# --------------------------------------------------------------------------- #
# Section 3 -- End-to-end: finite out-of-range -> 422 at the schema layer
# --------------------------------------------------------------------------- #

def _assert_422_for_wait(response, surface):
    """Shared assertions for the 422-everywhere guarantee on the fix.

    The bug report's headline: the DoS payload used to surface as a hang (the
    request worker parked in ``asyncio.sleep(1e9)``).  After the fix it is a
    422 at the schema layer, with a pydantic ``detail`` that references
    ``screenshot_wait_for``, and the crawl/leak signature is absent (the
    handler was never reached, so ``crawler.arun`` was never invoked).
    """
    assert response.status_code == 422, (
        f"{surface}: expected 422, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert "detail" in body, body
    locs = [
        tuple(err.get("loc", ()))
        for err in body["detail"]
        if isinstance(err, dict)
    ]
    assert any("screenshot_wait_for" in loc for loc in locs), body
    # The handler was never reached, so no crawl / arun / sleep signature can
    # appear anywhere in the response regardless of the input value.
    assert "arun" not in response.text, response.text
    assert "asyncio" not in response.text, response.text


def test_screenshot_out_of_range_wait_rejected_at_schema_422(stock_client):
    """POST /screenshot with the literal DoS payload -> 422, not a hang.

    No stubs are needed: pydantic rejects the body during request validation,
    so the route handler never runs (no crawler, no browser, no Redis, no
    sleep, no crawl slot held).
    """
    response = stock_client.post(
        "/screenshot",
        json={"url": "https://example.com", "screenshot_wait_for": 1e9},
        headers=_bearer(),
    )
    _assert_422_for_wait(response, "POST /screenshot (1e9 DoS payload)")


def test_screenshot_negative_wait_rejected_at_schema_422(stock_client):
    """POST /screenshot with a negative wait -> 422 (the ge=0 arm)."""
    response = stock_client.post(
        "/screenshot",
        json={"url": "https://example.com", "screenshot_wait_for": -0.1},
        headers=_bearer(),
    )
    _assert_422_for_wait(response, "POST /screenshot (negative)")


def test_screenshot_just_over_bound_rejected_at_schema_422(stock_client):
    """POST /screenshot with 5.1 -> 422; the le=5 arm fires just past the bound."""
    response = stock_client.post(
        "/screenshot",
        json={"url": "https://example.com", "screenshot_wait_for": 5.1},
        headers=_bearer(),
    )
    _assert_422_for_wait(response, "POST /screenshot (just over bound)")


# --------------------------------------------------------------------------- #
# Section 4 -- End-to-end: handler never reached for the DoS payload + NaN
# --------------------------------------------------------------------------- #
# ``RequestValidationError`` fires before the route handler runs, so
# ``generate_screenshot`` (and therefore ``server_crawler_config``,
# ``crawler.arun`` and the bare ``asyncio.sleep``) is never invoked.  This is
# the bug report's actual *safety* guarantee: the attacker's value must never
# reach the sleep.  We prove it by installing spies that raise if entered and
# asserting they are not called.  (For the NaN body the wire status is 500
# only because Starlette's ``allow_nan=False`` JSONResponse cannot echo the
# non-finite ``input`` back in the validation error; that serializer quirk is
# outside this fix's scope.  The handler-not-reached guarantee is what this
# fix owns.)


def _install_handler_spies(server_module, monkeypatch):
    """Spy on every step between request validation and the bare sleep.

    ``server_crawler_config`` is the first call after schema validation that
    sees ``body.screenshot_wait_for``; ``get_crawler`` acquires the
    crawl-slot/pool admission.  Either being reached for an out-of-range
    value means the schema bound failed and the DoS surface is open.
    """
    entered = {"config": False, "get_crawler": False}

    def _config_spy(config, **kwargs):
        entered["config"] = True
        entered["config_kwargs"] = kwargs
        raise AssertionError(
            "server_crawler_config must not be reached for an out-of-range "
            "screenshot_wait_for; the ge/le bound must reject it first"
        )

    async def _get_crawler_spy(*_a, **_kw):
        entered["get_crawler"] = True
        raise AssertionError(
            "get_crawler must not be reached for an out-of-range "
            "screenshot_wait_for; the ge/le bound must reject it first"
        )

    monkeypatch.setattr(server_module, "server_crawler_config", _config_spy)
    monkeypatch.setattr(server_module, "get_crawler", _get_crawler_spy)
    return entered


def test_screenshot_dos_payload_never_reaches_handler(stock_client, server_module, monkeypatch):
    """The literal DoS payload (``1e9``) must not reach the handler.

    With the schema bound, ``RequestValidationError`` rejects the body before
    ``generate_screenshot`` runs, so neither ``server_crawler_config`` nor
    ``get_crawler`` is invoked and the crawl slot is never acquired.
    """
    entered = _install_handler_spies(server_module, monkeypatch)
    response = stock_client.post(
        "/screenshot",
        json={"url": "https://example.com", "screenshot_wait_for": 1e9},
        headers=_bearer(),
    )
    assert response.status_code == 422, response.text
    assert not entered["config"], (
        "server_crawler_config was called for 1e9 -- the schema bound is "
        "missing or mis-placed; the DoS surface is OPEN"
    )
    assert not entered["get_crawler"], (
        "get_crawler was called for 1e9 -- a crawl slot would be acquired "
        "and pinned by the DoS payload"
    )


def test_screenshot_non_finite_wait_never_reaches_handler(
    stock_client, server_module, monkeypatch
):
    """A NaN body wait must not reach the handler (the bare ``asyncio.sleep``
    would hang forever on a non-finite value)."""
    entered = _install_handler_spies(server_module, monkeypatch)
    # Send raw bytes so stdlib json parses the NaN literal the way the server
    # does (no orjson here; see Section 2).
    body = b'{"url":"https://example.com","screenshot_wait_for":NaN}'
    response = stock_client.post(
        "/screenshot",
        content=body,
        headers={"Content-Type": "application/json", **_bearer()},
    )
    assert not entered["config"], (
        "server_crawler_config ran despite a non-finite wait; the ge/le "
        "bound is missing or mis-placed"
    )
    assert not entered["get_crawler"], (
        "get_crawler ran despite a non-finite wait"
    )
    # The crawl/arun leak signature must never reach the client regardless of
    # the wire status (which is 500 here only because Starlette cannot
    # serialize the NaN echoed in the validation error).
    assert "arun" not in response.text, response.text
    assert "asyncio" not in response.text, response.text


# --------------------------------------------------------------------------- #
# Section 5 -- Happy-path: in-range value is NOT over-rejected
# --------------------------------------------------------------------------- #


def test_screenshot_in_range_value_reaches_config_intact(
    stock_client, server_module, monkeypatch
):
    """An in-range ``screenshot_wait_for`` must reach ``server_crawler_config``
    with the value intact (no over-rejection) at the HTTP layer.

    ``validate_url_scheme`` is stubbed to skip the SSRF network check and
    ``get_crawler`` short-circuits before any browser/crawl, so this runs fully
    offline.  The 500 here is the short-circuit, NOT an over-rejection (422 would
    mean over-rejection)."""
    captured = {}

    async def _noop_validate(*_a, **_kw):
        return None

    def _capturing_config(config, **kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("SHORT_CIRCUIT_AFTER_CONFIG")

    async def _unreached_get_crawler(*_a, **_kw):
        raise AssertionError("get_crawler should not run; config spy short-circuits first")

    monkeypatch.setattr(server_module, "validate_url_scheme", _noop_validate)
    monkeypatch.setattr(server_module, "server_crawler_config", _capturing_config)
    monkeypatch.setattr(server_module, "get_crawler", _unreached_get_crawler)

    response = stock_client.post(
        "/screenshot",
        json={"url": "https://example.com", "screenshot_wait_for": 3.0},
        headers=_bearer(),
    )

    assert captured.get("kwargs") == {
        "screenshot": True,
        "screenshot_wait_for": 3.0,
        "wait_for_images": False,
    }, captured
    assert response.status_code == 500, response.text
    assert "SHORT_CIRCUIT_AFTER_CONFIG" in response.text or "Internal server error" in response.text


def test_screenshot_default_wait_reaches_config(stock_client, server_module, monkeypatch):
    """Omitting ``screenshot_wait_for`` (default ``2``) reaches the config unchanged."""
    captured = {}

    async def _noop_validate(*_a, **_kw):
        return None

    def _capturing_config(config, **kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("SHORT_CIRCUIT_AFTER_CONFIG")

    monkeypatch.setattr(server_module, "validate_url_scheme", _noop_validate)
    monkeypatch.setattr(server_module, "server_crawler_config", _capturing_config)

    response = stock_client.post(
        "/screenshot",
        json={"url": "https://example.com"},
        headers=_bearer(),
    )

    assert captured.get("kwargs", {}).get("screenshot_wait_for") == 2, captured
    assert response.status_code == 500, response.text


# --------------------------------------------------------------------------- #
# Section 6 -- The trusted-path bypass is now harmless (defense-in-depth check)
# --------------------------------------------------------------------------- #


def test_server_crawler_config_still_trusts_its_callers():
    """``server_crawler_config`` is a *trusted*-path constructor: it deliberately
    does NOT clamp (operators may pass any value via internal callers).  The fix
    is therefore at the ``ScreenshotRequest`` schema, the only client-reachable
    entry to this path.  Pinned so a future refactor that clamps here is a
    conscious decision: an out-of-range value passed *directly* still flows
    through (the schema is the gate)."""
    from api import server_crawler_config
    from utils import load_config

    cfg = server_crawler_config(
        load_config(),
        screenshot=True,
        screenshot_wait_for=1e9,  # would be a DoS if reached via the endpoint
        wait_for_images=False,
    )
    # The trusted path itself is unbounded by design; the schema bound is what
    # prevents 1e9 from ever reaching here from POST /screenshot.
    assert cfg.screenshot_wait_for == 1e9


# --------------------------------------------------------------------------- #
# Section 7 -- OpenAPI contract parity
# --------------------------------------------------------------------------- #


def _numeric_subschema(field_schema):
    """Pydantic 2.x serializes ``float | None`` as
    ``anyOf: [{type: number, ...}, {type: null}]``; pull the numeric member so
    the parity check reads the actual constraint-bearing sub-schema."""
    if "anyOf" in field_schema:
        for sub in field_schema["anyOf"]:
            if sub.get("type") == "number":
                return sub
    return field_schema


def test_screenshot_request_openapi_advertises_ge_le_bound(server_module):
    """The OpenAPI contract the server presents must advertise the bound, so a
    caller reading the spec sees ``minimum: 0`` / ``maximum: 5`` for
    ``screenshot_wait_for`` on POST /screenshot."""
    spec = server_module.app.openapi()
    ssw = spec["components"]["schemas"]["ScreenshotRequest"]["properties"][
        "screenshot_wait_for"
    ]
    num = _numeric_subschema(ssw)
    assert num.get("type") == "number", ssw
    assert num.get("minimum") == 0, ssw
    assert num.get("maximum") == 5, ssw
    # The default must still be 2 (no regression on the documented contract).
    assert ssw.get("default") == 2, ssw


# --------------------------------------------------------------------------- #
# Section 8 -- Bound matches the untrusted-path clamp constant
# --------------------------------------------------------------------------- #


def test_screenshot_wait_for_bound_matches_untrusted_clamp_constant():
    """The schema bound must match ``_MAX_DELAY_SECONDS`` so the two surfaces
    (the untrusted ``/crawl`` deserialization clamp and the ``/screenshot``
    schema bound) enforce the same ceiling.  If the upstream constant changes,
    this test surfaces the drift."""
    from crawl4ai.async_configs import _MAX_DELAY_SECONDS
    from schemas import ScreenshotRequest

    finfo = ScreenshotRequest.model_fields["screenshot_wait_for"]
    le = None
    for constraint in finfo.metadata:
        # annotated_types.Le carries the bound in `.le`
        if hasattr(constraint, "le") and not hasattr(constraint, "ge"):
            le = constraint.le
    assert le is not None, "screenshot_wait_for has no Le bound"
    assert le == _MAX_DELAY_SECONDS, (
        f"ScreenshotRequest.screenshot_wait_for le={le} does not match "
        f"_MAX_DELAY_SECONDS={_MAX_DELAY_SECONDS}; the /screenshot schema "
        f"bound has drifted from the untrusted-path clamp"
    )
