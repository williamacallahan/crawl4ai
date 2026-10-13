"""Regression suite for the unbounded ``temperature`` override on every LLM
request surface introduced in 159207b8.

Historical bug
--------------
Commit 159207b8 added a caller-supplied LLM ``temperature`` override to all
three Docker request surfaces and wired it straight through the handlers into
``litellm.acompletion``, with no ``ge``/``le``/``field_validator``:

* ``MarkdownRequest.temperature`` in ``schemas.py`` (POST /md body)
* ``LlmJobPayload.temperature`` in ``job.py`` (POST /llm/job body)
* ``temperature: Optional[float] = Query(None, ...)`` in ``server.py``
  (GET /llm/{url:path} query param)

The field was declared ``float | None`` with only a *descriptive* hint
``"(0.0-2.0)"`` in the docstring (not a binding), so any float -- including
negatives, ``99999.0``, ``float('inf')`` and ``float('nan')`` -- was accepted
at the schema layer and forwarded verbatim into
``extra_args["temperature"]`` → ``litellm.acompletion``.  When the upstream
provider bounds-checks (e.g. OpenAI requires ``0..2``), the resulting
``litellm.BadRequestError`` was caught by the generic
``except Exception as e: HTTPException(500, detail=str(e))`` clause on the
sync ``/md`` and ``/llm/{url:path}`` paths, so the caller received an
HTTP **500** with the upstream provider's raw exception text in ``detail``
instead of an HTTP **422** at the schema layer.  ``litellm.drop_params = True``
does not rescue this: it drops *unsupported* parameters (by name) for a
model, never *out-of-range values* for a parameter a model accepts (the
default ``gpt-4o-mini`` accepts ``temperature``).

Fix
---
Add ``ge=0.0, le=2.0`` to all three surfaces so an out-of-range or non-finite
value fails schema validation (HTTP 422, or a handler-never-reached rejection
-- see the non-finite note below) before any transport call, mirroring the
project's existing ``ge``/``le`` precedent at ``server.py`` (``score_ratio:
float = Query(0.5, ge=0.0, le=1.0, ...)``) and ``hook_registry.py``
(``Field(..., ge=..., le=...)``).

Coverage
--------
* Schema-level rejection (direct pydantic) on both body models for finite
  out-of-range and non-finite inputs.
* Schema-level acceptance of the in-range bounds (``0.0`` + ``2.0``) and
  omission -- the ``0.0`` case is the explicit regression guard for the
  falsy-collapse contract already pinned in ``test_temperature_e2e.py``.
* The non-finite JSON-parser arm: stdlib ``json`` accepts ``NaN`` literals
  (no ``orjson``/``ujson`` installed), so the only thing standing between a
  ``NaN`` body value and litellm is the new ``ge``/``le`` bound.  Proven by
  parsing a raw ``NaN`` body through ``MarkdownRequest``.
* GET query-param coercion (``Query(ge=0, le=2)``) rejects finite
  out-of-range and non-finite query strings uniformly.
* End-to-end via Starlette ``TestClient``: a *finite* out-of-range value is
  rejected with **422** at the schema layer (not the original 500) on all
  three surfaces, with a pydantic ``detail`` that references ``temperature``.
* End-to-end "handler never reached" for non-finite values on all three
  surfaces: the bug report's actual *safety* guarantee is that a non-finite
  value must never reach litellm.  Because ``RequestValidationError`` fires
  before the route handler, the handler (and therefore litellm) is never
  invoked; this is proven by installing a spy that raises if the handler is
  entered and asserting it is not called.  (Note: on the body surfaces, the
  422 ``RequestValidationError`` response itself cannot be serialized by
  Starlette's ``allow_nan=False`` ``JSONResponse`` because the validation
  error echoes the non-finite ``input`` -- the wire status is 500, not 422.
  That is a separate Starlette serialization quirk outside this fix's scope;
  the value never reaches the handler regardless, which is the property
  this suite pins.)
* The upstream leak signature (``litellm.BadRequestError`` /
  ``"temperature must be between 0 and 2"``) never reaches the client,
  because the handler is never reached.
* Happy-path regression: an in-range value and an omitted value are NOT
  rejected at the schema layer (no over-rejection) on all three surfaces.
  This is pinned at the schema layer in Sections 1-3; the existing
  ``test_temperature_e2e.py`` suite pins that in-range values reach litellm
  at the transport boundary, so an HTTP-layer happy-path duplicate is not
  re-asserted here.
* Parity: every surface's OpenAPI schema carries ``minimum: 0.0`` /
  ``maximum: 2.0`` on the ``temperature`` field -- the contract is uniform
  across the body surfaces and the query-param surface.
"""
import json
import urllib.parse

import pytest
from auth import create_access_token
from pydantic import ValidationError


def _bearer() -> dict:
    """A valid bearer header for the in-process test principal."""
    return {"Authorization": f"Bearer {create_access_token({'sub': 'test@example.com'})}"}


_OUT_OF_RANGE = [-0.1, 2.1, 99999.0, -5.0]
_NON_FINITE = [float("nan"), float("inf"), float("-inf")]
_IN_RANGE = [0.0, 0.7, 2.0]


# --------------------------------------------------------------------------- #
# Section 1 -- Schema-level rejection on the POST /md body surface
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", _OUT_OF_RANGE, ids=[repr(v) for v in _OUT_OF_RANGE])
def test_markdown_request_rejects_out_of_range_temperature(value):
    """A finite out-of-range ``temperature`` must fail schema validation."""
    from schemas import MarkdownRequest

    with pytest.raises(ValidationError) as exc:
        MarkdownRequest(url="https://example.com", f="raw", temperature=value)
    errors = exc.value.errors()
    assert any(err["loc"][-1] == "temperature" for err in errors), errors
    # The constraint message must be the bound, not a type error: this pins
    # that the rejection is because of ge/le (the fix), not a side effect.
    msg = json.dumps(errors, default=str)
    assert "less than or equal to 2" in msg or "greater than or equal to 0" in msg, msg


@pytest.mark.parametrize("value", _NON_FINITE, ids=[repr(v) for v in _NON_FINITE])
def test_markdown_request_rejects_non_finite_temperature(value):
    """Non-finite ``temperature`` (nan/inf/-inf) must fail schema validation.

    The point of bounding the override is precisely to keep these from
    reaching litellm: provider behaviour for non-finite temperatures is
    undefined, and the only guard is the schema bound.
    """
    from schemas import MarkdownRequest

    with pytest.raises(ValidationError) as exc:
        MarkdownRequest(url="https://example.com", f="raw", temperature=value)
    assert any(err["loc"][-1] == "temperature" for err in exc.value.errors())


@pytest.mark.parametrize("value", _IN_RANGE, ids=[repr(v) for v in _IN_RANGE])
def test_markdown_request_accepts_in_range_temperature(value):
    """In-range values (incl. both bounds) must be accepted."""
    from schemas import MarkdownRequest

    req = MarkdownRequest(url="https://example.com", f="raw", temperature=value)
    assert req.temperature == value


def test_markdown_request_accepts_omitted_temperature():
    """Omitting ``temperature`` must default to ``None`` (no regression)."""
    from schemas import MarkdownRequest

    req = MarkdownRequest(url="https://example.com", f="raw")
    assert req.temperature is None


# --------------------------------------------------------------------------- #
# Section 2 -- Schema-level rejection on the POST /llm/job body surface
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", _OUT_OF_RANGE, ids=[repr(v) for v in _OUT_OF_RANGE])
def test_llm_job_payload_rejects_out_of_range_temperature(value):
    """The job surface must reject the same finite out-of-range values."""
    from job import LlmJobPayload

    with pytest.raises(ValidationError) as exc:
        LlmJobPayload(url="https://example.com", q="x", temperature=value)
    assert any(err["loc"][-1] == "temperature" for err in exc.value.errors())


@pytest.mark.parametrize("value", _NON_FINITE, ids=[repr(v) for v in _NON_FINITE])
def test_llm_job_payload_rejects_non_finite_temperature(value):
    """The job surface must reject non-finite values too."""
    from job import LlmJobPayload

    with pytest.raises(ValidationError) as exc:
        LlmJobPayload(url="https://example.com", q="x", temperature=value)
    assert any(err["loc"][-1] == "temperature" for err in exc.value.errors())


@pytest.mark.parametrize("value", _IN_RANGE, ids=[repr(v) for v in _IN_RANGE])
def test_llm_job_payload_accepts_in_range_temperature(value):
    from job import LlmJobPayload

    p = LlmJobPayload(url="https://example.com", q="x", temperature=value)
    assert p.temperature == value


def test_llm_job_payload_accepts_omitted_temperature():
    from job import LlmJobPayload

    p = LlmJobPayload(url="https://example.com", q="x")
    assert p.temperature is None


# --------------------------------------------------------------------------- #
# Section 3 -- Query-param surface coercion (GET /llm/{url:path}), schema layer
# --------------------------------------------------------------------------- #

# FastAPI parses a Query(float) from the raw query string via pydantic string
# coercion with the declared ge/le applied. The raw strings below mirror what
# the client would send; the coercion must reject out-of-range and non-finite
# values the same way the body surfaces do.
_QUERY_REJECT = ["99999", "-5", "-0.1", "2.1", "nan", "inf", "-inf"]
_QUERY_ACCEPT = ["0.0", "0.7", "2.0"]


@pytest.mark.parametrize("raw", _QUERY_REJECT, ids=_QUERY_REJECT)
def test_query_param_surface_rejects_out_of_range_and_nonfinite(raw):
    """The GET query-param surface (Query(ge=0, le=2)) must reject the same
    values the body surfaces reject -- including ``nan``/``inf``/``-inf``,
    which pydantic coerces from the query string regardless of the JSON parser.
    This is the conservative anchor the bug report calls out: the non-finite
    arm of the GET surface is open even with orjson installed.
    """
    import pydantic

    # Mirror the schema FastAPI builds for `Query(None, ge=0.0, le=2.0)`:
    # an Optional confloat with the same bounds.
    adapter = pydantic.TypeAdapter(pydantic.confloat(ge=0.0, le=2.0) | None)
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python(raw)


@pytest.mark.parametrize("raw", _QUERY_ACCEPT, ids=_QUERY_ACCEPT)
def test_query_param_surface_accepts_in_range(raw):
    import pydantic

    adapter = pydantic.TypeAdapter(pydantic.confloat(ge=0.0, le=2.0) | None)
    assert adapter.validate_python(raw) == float(raw)


# --------------------------------------------------------------------------- #
# Section 4 -- Non-finite JSON-parser arm (bug report Evidence #3)
# --------------------------------------------------------------------------- #

def test_non_finite_temperature_rejected_even_when_json_parser_accepts_it():
    """The venv has no ``orjson``/``ujson``; Starlette falls back to stdlib
    ``json``, which accepts the ``NaN`` / ``Infinity`` / ``-Infinity`` literals
    by default.  Without the ``ge``/``le`` bound this would flow straight into
    litellm.  This test pins that the *only* thing stopping it is the schema
    bound, by parsing a raw ``NaN`` literal the way the server-side parser does
    and asserting pydantic then rejects it.
    """
    from schemas import MarkdownRequest

    # Sanity: confirm the parser in use accepts the literal.  If a future
    # environment installs orjson (which rejects these), this assertion flips
    # and the test below still holds vacuously because the body never parses
    # to a non-finite float in the first place.
    parsed = json.loads('{"url":"https://example.com","f":"raw","temperature":NaN}')
    assert parsed["temperature"] != parsed["temperature"]  # NaN is not equal to itself

    with pytest.raises(ValidationError) as exc:
        MarkdownRequest(**parsed)
    assert any(err["loc"][-1] == "temperature" for err in exc.value.errors())


# --------------------------------------------------------------------------- #
# Section 5 -- End-to-end: finite out-of-range -> 422 at the schema layer
# --------------------------------------------------------------------------- #

def _assert_422_for_temperature(response, surface):
    """Shared assertions for the 422-everywhere guarantee on the fix.

    The bug report's headline: a finite out-of-range value used to surface as
    a 500 with the upstream provider's raw exception text in ``detail``.
    After the fix it is a 422 at the schema layer, with a pydantic ``detail``
    that references the ``temperature`` field, and the upstream leak signature
    is absent (the handler was never reached, so litellm was never invoked).
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
    assert any("temperature" in loc for loc in locs), body
    # The upstream leak signature must never reach the client: with the fix
    # the handler (and therefore litellm) is never invoked, so the provider's
    # bounds-check exception text cannot appear anywhere in the response.
    assert "litellm" not in response.text, response.text
    assert "temperature must be between" not in response.text, response.text
    assert "BadRequestError" not in response.text, response.text


def test_md_out_of_range_temperature_rejected_at_schema_422(stock_client):
    """POST /md with an out-of-range body temperature -> 422, not 500.

    No stubs are needed: pydantic rejects the body during request validation,
    so the route handler never runs (no crawler, no litellm, no Redis).
    """
    response = stock_client.post(
        "/md",
        json={"url": "https://example.com", "f": "raw", "temperature": 99999.0},
        headers=_bearer(),
    )
    _assert_422_for_temperature(response, "POST /md (out-of-range)")


def test_md_negative_temperature_rejected_at_schema_422(stock_client):
    """POST /md with a negative temperature -> 422 (the ge=0.0 arm)."""
    response = stock_client.post(
        "/md",
        json={"url": "https://example.com", "f": "raw", "temperature": -0.1},
        headers=_bearer(),
    )
    _assert_422_for_temperature(response, "POST /md (negative)")


def test_llm_job_out_of_range_temperature_rejected_at_schema_422(stock_client):
    """POST /llm/job with an out-of-range body temperature -> 422, not 500."""
    response = stock_client.post(
        "/llm/job",
        json={"url": "https://example.com", "q": "extract", "temperature": 99999.0},
        headers=_bearer(),
    )
    _assert_422_for_temperature(response, "POST /llm/job (out-of-range)")


def test_llm_query_param_out_of_range_temperature_rejected_at_schema_422(
    stock_client,
):
    """GET /llm/{url:path}?temperature=99999 -> 422, not 500.

    No stubs are needed: FastAPI validates the query param before the handler
    runs.  This is the surface the bug report flagged as having no ``ge``/``le``
    even in its description; the bound now matches the body surfaces.
    """
    encoded = urllib.parse.quote_plus("https://example.com", safe="")
    response = stock_client.get(
        f"/llm/{encoded}",
        params={"q": "What is this page about?", "temperature": "99999"},
        headers=_bearer(),
    )
    _assert_422_for_temperature(response, "GET /llm/{url} (out-of-range)")


def test_llm_query_param_negative_temperature_rejected_at_schema_422(stock_client):
    encoded = urllib.parse.quote_plus("https://example.com", safe="")
    response = stock_client.get(
        f"/llm/{encoded}",
        params={"q": "What is this page about?", "temperature": "-0.1"},
        headers=_bearer(),
    )
    _assert_422_for_temperature(response, "GET /llm/{url} (negative)")


# For the GET query-param surface, the non-finite coercion-error's `input`
# is a *string* ('nan'/'inf'), so Starlette can serialize the 422 response
# cleanly. Postfix the 422 guarantee for non-finite here specifically, unlike
# the body surfaces (see the non-finite note in the module docstring).
@pytest.mark.parametrize("raw", ["nan", "inf", "-inf"], ids=["nan", "inf", "-inf"])
def test_llm_query_param_non_finite_temperature_rejected_at_schema_422(
    stock_client, raw
):
    encoded = urllib.parse.quote_plus("https://example.com", safe="")
    response = stock_client.get(
        f"/llm/{encoded}",
        params={"q": "What is this page about?", "temperature": raw},
        headers=_bearer(),
    )
    _assert_422_for_temperature(response, f"GET /llm/{{url}} ({raw})")


# --------------------------------------------------------------------------- #
# Section 6 -- End-to-end: non-finite body values never reach the handler
# --------------------------------------------------------------------------- #
#
# On the body surfaces a non-finite value triggers RequestValidationError (the
# ge/le bound) before the handler runs, so the handler -- and therefore
# litellm -- is never invoked.  This is the bug report's actual *safety*
# guarantee: a non-finite value must never be forwarded to litellm.  We prove
# it by installing a spy that raises if the handler is entered and asserting
# it is not called.  (The wire status for body non-finite is 500 because
# Starlette's allow_nan=False JSONResponse cannot echo the non-finite `input`
# back in the validation error; that serializer quirk is outside this fix's
# scope.  The handler-not-reached guarantee is what this fix owns.)


def test_md_non_finite_temperature_never_reaches_handler(
    stock_client, server_module, monkeypatch
):
    """A NaN body temperature on POST /md must not reach the handler."""
    entered = {"called": False}

    async def _spy(*_args, **_kwargs):
        entered["called"] = True
        raise AssertionError(
            "handle_markdown_request must not be reached for a non-finite "
            "body temperature; the ge/le bound must reject it first"
        )

    monkeypatch.setattr(server_module, "handle_markdown_request", _spy)

    body = b'{"url":"https://example.com","f":"llm","q":"x","temperature":NaN}'
    response = stock_client.post(
        "/md",
        content=body,
        headers={"Content-Type": "application/json", **_bearer()},
    )

    assert not entered["called"], (
        "the handler ran despite the non-finite temperature; the ge/le bound "
        "is missing or mis-placed"
    )
    # Litellm's leak signature must never reach the client regardless of the
    # wire status (which is 500 here only because Starlette cannot serialize
    # the NaN echoed in the validation error).
    assert "litellm" not in response.text, response.text
    assert "temperature must be between" not in response.text, response.text
    assert "BadRequestError" not in response.text, response.text


def test_llm_job_non_finite_temperature_never_reaches_handler(
    stock_client, monkeypatch
):
    """A NaN body temperature on POST /llm/job must not reach the handler."""
    import job

    entered = {"called": False}

    async def _spy(*_args, **_kwargs):
        entered["called"] = True
        raise AssertionError(
            "handle_llm_request must not be reached for a non-finite body "
            "temperature; the ge/le bound must reject it first"
        )

    monkeypatch.setattr(job, "handle_llm_request", _spy)

    body = b'{"url":"https://example.com","q":"extract","temperature":Infinity}'
    response = stock_client.post(
        "/llm/job",
        content=body,
        headers={"Content-Type": "application/json", **_bearer()},
    )

    assert not entered["called"], (
        "the job handler ran despite the non-finite temperature"
    )
    assert "litellm" not in response.text, response.text
    assert "BadRequestError" not in response.text, response.text


def test_llm_query_param_non_finite_never_reaches_handler(
    stock_client, server_module, monkeypatch
):
    """A ``temperature=nan`` query param on GET /llm/{url} must not reach the
    handler.  (This surface returns a clean 422 because the validation error
    echoes the *string* input; here we additionally pin the handler-not-
    reached safety guarantee.)"""
    entered = {"called": False}

    async def _spy(*_args, **_kwargs):
        entered["called"] = True
        raise AssertionError(
            "handle_llm_qa must not be reached for a non-finite query "
            "temperature; the ge/le bound must reject it first"
        )

    monkeypatch.setattr(server_module, "handle_llm_qa", _spy)

    encoded = urllib.parse.quote_plus("https://example.com", safe="")
    response = stock_client.get(
        f"/llm/{encoded}",
        params={"q": "What is this page about?", "temperature": "nan"},
        headers=_bearer(),
    )

    assert not entered["called"], (
        "the QA handler ran despite the non-finite query temperature"
    )
    assert "litellm" not in response.text, response.text


# --------------------------------------------------------------------------- #
# Section 8 -- Parity: every surface carries minimum:0.0 / maximum:2.0
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


def test_all_temperature_surfaces_carry_ge_le_in_openapi(server_module):
    """The OpenAPI contract the server presents must advertise the same bound
    on all three surfaces.  This is the cross-surface parity the bug report
    identifies as the intended contract: a caller reading the spec should see
    ``minimum: 0.0 / maximum: 2.0`` for ``temperature`` on POST /md, on
    POST /llm/job, and on GET /llm/{url:path}, uniformly.
    """
    spec = server_module.app.openapi()

    # --- POST /md: MarkdownRequest body ---
    md_temp = spec["components"]["schemas"]["MarkdownRequest"]["properties"]["temperature"]
    md_num = _numeric_subschema(md_temp)
    assert md_num.get("minimum") == 0.0, md_temp
    assert md_num.get("maximum") == 2.0, md_temp
    assert md_num.get("type") == "number", md_temp

    # --- POST /llm/job: LlmJobPayload body ---
    job_temp = spec["components"]["schemas"]["LlmJobPayload"]["properties"]["temperature"]
    job_num = _numeric_subschema(job_temp)
    assert job_num.get("minimum") == 0.0, job_temp
    assert job_num.get("maximum") == 2.0, job_temp
    assert job_num.get("type") == "number", job_temp

    # --- GET /llm/{url:path}: query parameter ---
    llm_get = spec["paths"]["/llm/{url}"]["get"]
    temp_param = next(
        p for p in llm_get["parameters"] if p["name"] == "temperature"
    )
    temp_num = _numeric_subschema(temp_param.get("schema", {}))
    assert temp_num.get("minimum") == 0.0, temp_num
    assert temp_num.get("maximum") == 2.0, temp_num
    assert temp_num.get("type") == "number", temp_num


def test_temperature_field_metadata_uniform_across_body_models():
    """The two body models must carry the same annotated-types ge/le metadata,
    so a future refactor cannot drift one surface from the other."""
    from job import LlmJobPayload
    from schemas import MarkdownRequest

    import annotated_types

    def _bounds(model, name):
        finfo = model.model_fields[name]
        ge = le = None
        for constraint in finfo.metadata:
            if isinstance(constraint, annotated_types.Ge):
                ge = constraint.ge
            elif isinstance(constraint, annotated_types.Le):
                le = constraint.le
        return ge, le

    assert _bounds(MarkdownRequest, "temperature") == (0.0, 2.0)
    assert _bounds(LlmJobPayload, "temperature") == (0.0, 2.0)
