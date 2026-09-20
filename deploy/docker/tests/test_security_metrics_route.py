"""
Regression tests for the Prometheus metrics route.

The shipped `deploy/docker/server.py` used to register a `metrics()` route
unconditionally at `config["observability"]["prometheus"]["endpoint"]` whose
handler returned `RedirectResponse(<that same endpoint>)` — a self-referential
307. The `prometheus_fast_api_instrumentator.Instrumentator` block (registered
separately, only when `prometheus.enabled` is true, and hardcoded to serve at
`/metrics`) shadowed the buggy route **only** under the shipped default config
(`enabled: true`, `endpoint: "/metrics"`). Under any other config the buggy
route was the sole handler at the configured path and emitted a self-redirect
that a redirect-following client (e.g. Prometheus's default `follow_redirects:
true`) looped on until its redirect cap.

The fix deletes the redundant `metrics()` route: when Prometheus is enabled
the Instrumentator serves the metrics endpoint; when disabled the path is
unregistered (framework default 404).

These tests pin the endpoint behavior across the three config states so the
self-redirect cannot be reintroduced:

  * State A — shipped default (`enabled: true`, `endpoint: "/metrics"`):
        `GET /metrics` with a credential returns 200 Prometheus exposition
        text and is NOT a redirect of any kind.
  * State B — `enabled: false`: the configured endpoint is unregistered
        (404), not a self-redirect.
  * State C — `enabled: true`, `endpoint != "/metrics"`: the configured
        endpoint does not self-redirect.

The unauthenticated-401 posture of `/metrics` is already pinned by
`test_security_default_posture.py`, so these tests cover the authenticated
routing/response shape only.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cve

from auth import create_access_token  # noqa: E402


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _load_server_with_config(monkeypatch, cfg: dict, mod_name: str):
    """Import ``deploy/docker/server.py`` fresh under a *unique* module name
    with a custom loaded config.

    This re-executes server.py's module body (route registration, middleware
    wiring, the conditional Instrumentator block) against ``cfg`` without
    touching ``config.yml`` on disk and without mutating the session-scoped
    ``server`` module used by other tests:

    * ``utils.load_config`` is patched (via ``monkeypatch``) so server.py's
      ``from utils import load_config`` binds the custom config for its
      module-level ``config = load_config()`` call.
    * The fresh module is installed under ``mod_name`` (never ``"server"``)
      so the session ``server_module`` fixture is unaffected.
    * ``monkeypatch`` reverts both the ``load_config`` patch and the
      ``sys.modules`` entry at teardown.
    """
    import utils as _utils

    monkeypatch.setattr(_utils, "load_config", lambda: cfg)
    spec = importlib.util.spec_from_file_location(
        mod_name, str(Path(__file__).resolve().parents[1] / "server.py")
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    return mod


def _client_for(mod):
    """A Starlette TestClient without lifespan (no Chromium/Redis) so the
    routing/auth/middleware stack runs but no browser or Redis is opened."""
    from starlette.testclient import TestClient

    return TestClient(mod.app, raise_server_exceptions=False)


# ─────────────────────────── State A: shipped default ───────────────────────


def test_configured_endpoint_serves_prometheus_text_not_a_redirect(
    stock_client, server_module
):
    """Shipped default (`enabled: true`, `endpoint: "/metrics"`): an
    authenticated GET /metrics must return 200 Prometheus exposition text and
    must not be a redirect of any kind. Guards the happy path against a
    reintroduced self-redirect (which the Instrumentator would shadow here,
    masking the bug) or a broken Instrumentator wiring."""
    tok = create_access_token({"sub": "ops@metrics.example"})
    r = stock_client.get("/metrics", headers=_bearer(tok), follow_redirects=False)
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]!r}"
    assert "location" not in r.headers, (
        f"GET /metrics returned a redirect (status={r.status_code}, "
        f"location={r.headers.get('location')!r}); the metrics endpoint "
        "must not redirect."
    )
    assert r.headers.get("content-type", "").startswith("text/plain"), (
        f"expected Prometheus text/plain content-type, got {r.headers.get('content-type')!r}"
    )
    assert "# HELP" in r.text or "# TYPE" in r.text, (
        f"response body does not look like Prometheus exposition text: {r.text[:200]!r}"
    )


# ────────────────────── State B: prometheus.enabled = false ─────────────────


def test_configured_endpoint_does_not_self_redirect_when_disabled(monkeypatch):
    """With prometheus disabled, the configured endpoint must be unregistered
    (404), not a self-referential 307. Before the fix, the unconditional
    `metrics()` route was the sole handler here and self-redirected to the
    same path, looping redirect-following scrapers."""
    from utils import load_config as _orig_load_config

    cfg = _orig_load_config()
    cfg["observability"]["prometheus"]["enabled"] = False
    cfg["observability"]["prometheus"]["endpoint"] = "/metrics"

    mod = _load_server_with_config(monkeypatch, cfg, "server_under_test_metrics_disabled")
    client = _client_for(mod)
    tok = create_access_token({"sub": "ops@metrics.example"})

    r = client.get("/metrics", headers=_bearer(tok), follow_redirects=False)
    assert r.status_code != 307, (
        f"GET /metrics returned 307 to {r.headers.get('location')!r} — "
        "the self-referential redirect loop is present."
    )
    assert "location" not in r.headers, (
        f"GET /metrics returned status {r.status_code} with a redirect to "
        f"{r.headers.get('location')!r}; the disabled metrics endpoint "
        "must not redirect."
    )
    assert r.status_code == 404, (
        f"expected 404 for the unregistered disabled endpoint, got {r.status_code}"
    )


# ─────────────── State C: enabled, non-default configured endpoint ───────────


def test_configured_endpoint_does_not_self_redirect_with_non_default_path(monkeypatch):
    """When the operator configures `endpoint != "/metrics"` and Prometheus is
    enabled, the configured path must not self-redirect. Before the fix it had
    only the buggy `metrics()` route and emitted a self-referential 307, while
    the Instrumentator silently served `/metrics` (its own hardcoded default —
    a separate, out-of-scope concern)."""
    from utils import load_config as _orig_load_config

    cfg = _orig_load_config()
    cfg["observability"]["prometheus"]["enabled"] = True
    cfg["observability"]["prometheus"]["endpoint"] = "/custom-metrics"

    mod = _load_server_with_config(monkeypatch, cfg, "server_under_test_metrics_custom_endpoint")
    client = _client_for(mod)
    tok = create_access_token({"sub": "ops@metrics.example"})

    r = client.get("/custom-metrics", headers=_bearer(tok), follow_redirects=False)
    assert r.status_code != 307, (
        f"GET /custom-metrics returned 307 to {r.headers.get('location')!r} — "
        "the self-referential redirect loop is present at the configured endpoint."
    )
    assert "location" not in r.headers, (
        f"GET /custom-metrics returned status {r.status_code} with a redirect "
        f"to {r.headers.get('location')!r}; the configured metrics endpoint "
        "must not redirect."
    )
