# ───────────────────────── server.py ─────────────────────────
"""
Crawl4AI FastAPI entry‑point
• Browser pool + global page cap
• Rate‑limiting, security, metrics
• /crawl, /crawl/stream, /md, /llm endpoints
"""

# ── stdlib & 3rd‑party imports ───────────────────────────────
import ast
import asyncio
import base64
import logging
import os
import pathlib
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from api import (
    handle_crawl_request,
    handle_llm_qa,
    handle_markdown_request,
    handle_stream_crawl_request,
    server_crawler_config,
    stream_results,
)
from auth import (
    TokenRequest,
    constant_time_eq,
    create_access_token,
    get_token_dependency,
    resolve_secret_key,
)
from auth_gate import AuthGateMiddleware
from crawler_pool import close_all, get_crawler, janitor, release_crawler
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from governor import BodySizeLimitMiddleware, max_body_bytes_from_config
from job import init_job_router
from mcp_bridge import attach_mcp, mcp_tool
from monitor_routes import router as monitor_router
from prometheus_fastapi_instrumentator import Instrumentator
from rank_bm25 import BM25Okapi
from redis import asyncio as aioredis
from redis_config import (
    build_rate_limit_storage_uri as _build_rate_limit_storage_uri,
)
from redis_config import (
    RESILIENT_CLIENT_KWARGS as _REDIS_RESILIENT_KWARGS,
)
from redis_config import (
    build_redis_url as _build_redis_url,
)
from schemas import (
    CrawlRequestWithHooks,
    HTMLRequest,
    JSEndpointRequest,
    MarkdownRequest,
    PDFRequest,
    ScreenshotRequest,
)
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as _StarletteHTTPException
from utils import (
    get_browser_extra_args,
    load_config,
    public_error_detail,
    public_crawl_error,
    setup_logging,
    validate_url_destination,
    validate_webhook_url,
    verify_email_domain,
)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.__version__ import __version__
from crawl4ai.async_configs import Provenance, UntrustedConfigError

# ── internal imports (after sys.path append) ─────────────────
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# ────────────────── configuration / logging ──────────────────
config = load_config()
setup_logging(config)

# Version is imported from crawl4ai package to ensure it stays in sync

# ── global page semaphore (hard cap) ─────────────────────────
MAX_PAGES = config["crawler"]["pool"].get("max_pages", 30)
GLOBAL_SEM = asyncio.Semaphore(MAX_PAGES)

# ── security feature flags ───────────────────────────────────
# Hooks are disabled by default for security (RCE risk). Set to "true" to enable.
HOOKS_ENABLED = os.environ.get("CRAWL4AI_HOOKS_ENABLED", "false").lower() == "true"

# /execute_js disabled by default (arbitrary JS + SSRF risk). Set to "true" to enable.
EXECUTE_JS_ENABLED = os.environ.get("CRAWL4AI_EXECUTE_JS_ENABLED", "false").lower() == "true"

def _current_api_token() -> str:
    """The effective static operator token (config or environment)."""
    configured = config.get("security", {}).get("api_token", "")
    if configured and not isinstance(configured, str):
        raise RuntimeError("security.api_token must be a string")
    return os.environ.get("CRAWL4AI_API_TOKEN", "") or configured


def _current_jwt_enabled() -> bool:
    """Resolve one JWT posture for socket binding and issuance.

    ``CRAWL4AI_JWT_ENABLED`` is an explicit runtime override for container
    deployments; otherwise the canonical config value applies.
    """
    raw = os.environ.get("CRAWL4AI_JWT_ENABLED")
    if raw is None or not raw.strip():
        configured = config.get("security", {}).get("jwt_enabled", False)
        if not isinstance(configured, bool):
            raise RuntimeError("security.jwt_enabled must be a YAML boolean")
        return configured
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        "CRAWL4AI_JWT_ENABLED must be one of true/false, 1/0, yes/no, on/off"
    )


def _internal_service_auth_headers() -> dict:
    """Use the effective operator token, or a JWT in JWT-only deployments."""
    token = _current_api_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {
        "Authorization": (
            "Bearer " + create_access_token({"sub": "mcp-service"}, scope="data")
        )
    }


def _effective_bind_host() -> str:
    """Return the host actually bound by Gunicorn or the direct server."""
    bind = os.environ.get("GUNICORN_BIND", "")
    if not bind or bind.startswith("unix:"):
        return config["app"]["host"]
    if bind.startswith("["):
        return bind[1:].split("]", 1)[0]
    return bind.rsplit(":", 1)[0]


if not _current_api_token() and not _current_jwt_enabled():
    logging.getLogger("crawl4ai.security").warning(
        "No API token or JWT posture is configured; startup will restrict the "
        "service to loopback."
    )

def get_default_browser_config() -> BrowserConfig:
    """Get default BrowserConfig from config.yml.

    Egress hardening (TLS-verify on, pinning proxy so Chromium never resolves
    the target itself) is applied here so every endpoint that fetches through
    the default config (/html, /screenshot, /pdf, /execute_js) gets the same
    DNS-rebinding / redirect-to-internal protection as /crawl, rather than
    relying on each handler to remember it."""
    bc = BrowserConfig(
        extra_args=get_browser_extra_args(config),
        **config["crawler"]["browser"].get("kwargs", {}),
    )
    from egress_broker import enforce_egress
    enforce_egress(bc)
    return bc

# import logging
# page_log = logging.getLogger("page_cap")
# orig_arun = AsyncWebCrawler.arun
# async def capped_arun(self, *a, **kw):
#     await GLOBAL_SEM.acquire()                        # ← take slot
#     try:
#         in_flight = MAX_PAGES - GLOBAL_SEM._value     # used permits
#         page_log.info("🕸️  pages_in_flight=%s / %s", in_flight, MAX_PAGES)
#         return await orig_arun(self, *a, **kw)
#     finally:
#         GLOBAL_SEM.release()                          # ← free slot

orig_arun = AsyncWebCrawler.arun


async def capped_arun(self, *a, **kw):
    async with GLOBAL_SEM:
        return await orig_arun(self, *a, **kw)
AsyncWebCrawler.arun = capped_arun

# ───────────────────── FastAPI lifespan ──────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    import monitor as monitor_module
    from crawler_pool import init_permanent
    from egress_proxy import start_pinning_proxy, stop_pinning_proxy
    from monitor import MonitorStats

    # Enforce auth posture before serving any traffic.
    _resolve_auth()

    app.state.egress_proxy = None
    app.state.readiness_checks_active = False
    try:
        # Initialize the sandboxed artifact store + reaper.
        from artifacts import init_store

        init_store()
        app.state.artifact_janitor = asyncio.create_task(_artifact_janitor())

        # The API and every durable worker use the same lifecycle owner.  It
        # registers the localhost pinning proxy before any browser config is
        # constructed.
        app.state.egress_proxy = await start_pinning_proxy()

        # Bounded background-job queue (per-principal quotas optional).
        from governor import job_queue_caps
        from work_queue import WorkQueue, set_job_queue

        caps = job_queue_caps(config)
        app.state.job_queue = WorkQueue(redis=redis, **caps)
        await app.state.job_queue.start()
        set_job_queue(app.state.job_queue)

        # Initialize monitor and prove the effective (possibly external,
        # authenticated) Redis is reachable before readiness becomes active.
        await asyncio.wait_for(redis.ping(), timeout=5.0)
        monitor_module.monitor_stats = MonitorStats(redis)
        await monitor_module.monitor_stats.load_from_redis()
        monitor_module.monitor_stats.start_persistence_worker()

        # Permanent browser must be initialized from the same post-proxy
        # effective config used by per-request crawlers.
        await init_permanent(get_default_browser_config())

        app.state.janitor = asyncio.create_task(janitor())
        app.state.timeline_updater = asyncio.create_task(_timeline_updater())
        app.state.readiness_checks_active = True

        yield
    finally:
        app.state.readiness_checks_active = False
        background_tasks = []
        for task_name in ("janitor", "timeline_updater", "artifact_janitor"):
            task = getattr(app.state, task_name, None)
            if task is not None:
                task.cancel()
                background_tasks.append(task)
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

        job_queue = getattr(app.state, "job_queue", None)
        if job_queue is not None:
            try:
                await job_queue.stop()
            except Exception:
                pass

        from monitor import get_monitor

        try:
            await get_monitor().cleanup()
        except Exception as e:
            logger.error(f"Monitor cleanup failed: {e}")

        await close_all()
        try:
            await stop_pinning_proxy(app.state.egress_proxy)
        except Exception:
            pass

async def _artifact_janitor():
    """Periodically reap expired / over-quota artifacts."""
    from artifacts import janitor as _reap
    while True:
        await asyncio.sleep(300)
        try:
            await asyncio.to_thread(_reap)
        except Exception as e:
            logger.warning(f"Artifact janitor error: {e}")

async def _timeline_updater():
    """Update timeline data every 5 seconds."""
    from monitor import get_monitor
    while True:
        await asyncio.sleep(5)
        try:
            await asyncio.wait_for(get_monitor().update_timeline(), timeout=4.0)
        except asyncio.TimeoutError:
            logger.warning("Timeline update timeout after 4s")
        except Exception as e:
            logger.warning(f"Timeline update error: {e}")

# ───────────────────── FastAPI instance ──────────────────────
app = FastAPI(
    title=config["app"]["title"],
    version=config["app"]["version"],
    lifespan=lifespan,
)

# ── static playground ──────────────────────────────────────
STATIC_DIR = pathlib.Path(__file__).parent / "static" / "playground"
if not STATIC_DIR.exists():
    raise RuntimeError(f"Playground assets not found at {STATIC_DIR}")
app.mount(
    "/playground",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="play",
)

# ── static monitor dashboard ────────────────────────────────
MONITOR_DIR = pathlib.Path(__file__).parent / "static" / "monitor"
if not MONITOR_DIR.exists():
    raise RuntimeError(f"Monitor assets not found at {MONITOR_DIR}")
app.mount(
    "/dashboard",
    StaticFiles(directory=MONITOR_DIR, html=True),
    name="monitor_ui",
)

# ── static assets (logo, etc) ────────────────────────────────
ASSETS_DIR = pathlib.Path(__file__).parent / "static" / "assets"
if ASSETS_DIR.exists():
    app.mount(
        "/static/assets",
        StaticFiles(directory=ASSETS_DIR),
        name="assets",
    )


@app.get("/")
async def root():
    return RedirectResponse("/playground")


# Pre-0.9 clients used /monitor for the dashboard shell, which now lives at
# /dashboard.  Only this exact path is public; /monitor/* remains protected.
@app.get("/monitor", include_in_schema=False)
async def monitor_ui_redirect():
    return RedirectResponse("/dashboard")

# ─────────────────── infra / middleware  ─────────────────────
redis = aioredis.from_url(_build_redis_url(config), **_REDIS_RESILIENT_KWARGS)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config["rate_limiting"]["default_limit"]],
    storage_uri=_build_rate_limit_storage_uri(config),
)


def _setup_security(app_: FastAPI):
    sec = config["security"]
    if sec.get("https_redirect"):
        app_.add_middleware(HTTPSRedirectMiddleware)
    # Apply the Host guard whenever real hostnames are configured, independent
    # of `security.enabled` (the old code silently skipped it when disabled).
    trusted = sec.get("trusted_hosts", ["*"])
    if trusted and trusted != ["*"]:
        app_.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted)
    elif _effective_bind_host() not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger("crawl4ai.security").warning(
            "trusted_hosts is ['*'] on a non-loopback bind (%s): the Host guard "
            "is disabled. Set security.trusted_hosts to your real hostname(s).",
            _effective_bind_host(),
        )

    # Deny-by-default CORS: only explicitly allowlisted origins; never '*' with
    # credentials.
    origins = [o for o in (sec.get("cors_allow_origins") or []) if o and o != "*"]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware
        app_.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


_setup_security(app)

if config["observability"]["prometheus"]["enabled"]:
    Instrumentator().instrument(app).expose(app)

token_dep = get_token_dependency(config)

# ── security response headers (unconditional, strict-by-default) ──────
SECURITY_BASELINE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
# Strict CSP for the API / error surface (the injection-reflection paths).
STRICT_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'none'"
)
# The dashboard/playground still ship inline scripts/styles; until they are
# externalized (CSP-compat refactor) they must not receive the strict CSP or
# they break. They still get the baseline headers (nosniff, frame DENY).
_UI_PREFIXES = ("/dashboard", "/playground", "/static")


async def add_security_headers(request: Request, call_next):
    resp = await call_next(request)
    for k, v in SECURITY_BASELINE_HEADERS.items():
        resp.headers.setdefault(k, v)
    path = request.url.path
    if not any(path.startswith(p) for p in _UI_PREFIXES):
        resp.headers.setdefault("Content-Security-Policy", STRICT_CSP)
    if config["security"].get("https_redirect"):
        resp.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
        )
    return resp


# ── authentication gate (outermost ASGI layer, fails closed) ──────────
# The single place auth is decided: covers every route, static mount, the MCP
# transports and the metrics endpoint, for HTTP and WebSocket alike. Only the
# health/token endpoints and the exact UI redirects are public.
HEALTH_PATH = config["observability"]["health_check"]["endpoint"]


# ── request body-size limit (DoS) ─────────────────────────────────────
app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes_from_config(config))

# Add auth after the body limiter so Starlette makes it the outermost layer:
# unauthorized callers are rejected before their body is read or buffered.
app.add_middleware(
    AuthGateMiddleware,
    token_provider=_current_api_token,
    public_paths={HEALTH_PATH, "/token", "/", "/monitor"},
    public_prefixes=_UI_PREFIXES,
)

# Response decoration sits outside the auth boundary so even an early 401/413
# carries the baseline headers. It does not read request bodies; AuthGate is
# still the first authorization decision and remains outermost for WebSockets.
app.middleware("http")(add_security_headers)


def _resolve_auth():
    """Runtime auth-posture guard. Runs at startup (lifespan), not import, so
    the behavioral test harness can import the app without a hard exit.

    - credential configured  -> enforce (fail fast on jwt_enabled w/o SECRET_KEY)
    - none + non-loopback bind -> refuse to start (would be open to the network)
    - none + loopback bind    -> start with protected routes unavailable
    """
    bind = os.environ.get("GUNICORN_BIND", "")
    bind_host = _effective_bind_host()
    loopback = bind.startswith("unix:") or bind_host in (
        "127.0.0.1",
        "localhost",
        "::1",
    )
    api_token = _current_api_token()
    jwt_enabled = _current_jwt_enabled()

    if api_token or jwt_enabled:
        if jwt_enabled:
            resolve_secret_key(required=True)  # fail fast: no ephemeral secret
        logger.info("Auth gate active (credential configured).")
        return

    if not loopback:
        logger.critical(
            "Refusing to start: binding %s with no CRAWL4AI_API_TOKEN and "
            "jwt_enabled=false would expose an unauthenticated API. Provide an "
            "existing operator-managed credential, enable JWT with an existing "
            "SECRET_KEY, or bind loopback.",
            bind_host,
        )
        sys.exit(1)

    logger.warning(
        "No API credential is configured; protected loopback routes will return "
        "401 until the operator supplies an existing credential."
    )

# ───────────────── URL validation helper ─────────────────
ALLOWED_URL_SCHEMES = ("http://", "https://")
ALLOWED_URL_SCHEMES_WITH_RAW = ("http://", "https://", "raw:", "raw://")


async def validate_url_scheme(
    url: str,
    allow_raw: bool = False,
    check_destination: bool = True,
) -> None:
    """Validate URL scheme (LFI) and destination (SSRF)."""
    allowed = ALLOWED_URL_SCHEMES_WITH_RAW if allow_raw else ALLOWED_URL_SCHEMES
    if not url.startswith(allowed):
        schemes = ", ".join(allowed)
        raise HTTPException(400, f"URL must start with {schemes}")
    if url.startswith(("raw:", "raw://")) or not check_destination:
        return
    await asyncio.to_thread(validate_url_destination, url)


# ───────────────── safe config‑dump helper ─────────────────
ALLOWED_TYPES = {
    "CrawlerRunConfig": CrawlerRunConfig,
    "BrowserConfig": BrowserConfig,
}


def _config_from_json(data: dict) -> dict:
    """Validate a {type, params} config under the untrusted trust boundary and
    echo the normalized result.

    This endpoint is no longer a gadget-construction oracle: only the gated,
    side-effect-free CrawlerRunConfig/BrowserConfig types may be validated, the
    untrusted gate raises on forbidden power-fields and disallowed nested types
    (LLM*, proxy, deep-crawl - which is what would read env/secrets), drops
    unknown fields, and clamps quantities."""
    config_type = data.get("type")
    if config_type == "CrawlerRunConfig":
        obj = CrawlerRunConfig.load(data, provenance=Provenance.UNTRUSTED)
    elif config_type == "BrowserConfig":
        obj = BrowserConfig.load(data, provenance=Provenance.UNTRUSTED)
    else:
        raise ValueError("type must be 'CrawlerRunConfig' or 'BrowserConfig'")
    return obj.dump()


# ── job router ──────────────────────────────────────────────
app.include_router(init_job_router(redis, config, token_dep))

# ── monitor router ──────────────────────────────────────────
# Do not attach token_dep at router level: it is HTTP Request-only and breaks
# the WebSocket upgrade on /monitor/ws (TypeError: _principal() missing 'request').
# AuthGateMiddleware already authenticates HTTP + WS; destructive monitor
# actions keep their own Depends(require_admin).
app.include_router(monitor_router)

logger = logging.getLogger(__name__)


# ── central exception handling (no internal detail leaks) ─────────────
# 16 sites used to return raw str(e) to clients, leaking paths, dependency
# versions, resolved internal IPs and sometimes secrets. Centralize: 5xx
# responses are generic + carry a correlation id; the full detail is logged
# server-side. 4xx developer messages are preserved.
@app.exception_handler(_StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: _StarletteHTTPException):
    # 500 is the raw-str(e) leak vector -> genericize with a correlation id.
    # Deliberate operational statuses (502/503/504, with their own short
    # messages + headers like Retry-After) pass through, as do 4xx.
    if exc.status_code == 500:
        cid = uuid.uuid4().hex[:12]
        logger.error("server error 500 [cid=%s]: %s", cid, exc.detail)
        return JSONResponse(
            {"error": "Internal server error", "correlation_id": cid},
            status_code=500,
        )
    return JSONResponse(
        {"detail": exc.detail},
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    cid = uuid.uuid4().hex[:12]
    logger.exception("unhandled exception [cid=%s]", cid)
    return JSONResponse(
        {"error": "Internal server error", "correlation_id": cid},
        status_code=500,
    )


# ──────────────────────── Endpoints ──────────────────────────
@app.post("/token")
async def get_token(req: TokenRequest):
    if not _current_jwt_enabled():
        raise HTTPException(403, "JWT issuance is disabled on the server.")
    expected_token = _current_api_token()
    if not expected_token:
        # Fail closed: without a configured api_token the old behavior minted a
        # JWT to anyone whose email merely had an MX record. Refuse instead.
        raise HTTPException(
            403,
            "Token issuance is disabled: no operator API token is configured.",
        )
    if not req.api_token or not constant_time_eq(req.api_token, expected_token):
        raise HTTPException(401, "Invalid or missing api_token")
    if not await asyncio.to_thread(verify_email_domain, req.email):
        raise HTTPException(400, "Invalid email domain")
    try:
        token = create_access_token({"sub": req.email})
    except RuntimeError:
        raise HTTPException(
            503,
            "JWT issuance is unavailable until an existing SECRET_KEY is configured.",
        )
    return {"email": req.email, "access_token": token, "token_type": "bearer"}


@app.post("/config/dump")
async def config_dump(
    data: dict,
    _td: Dict = Depends(token_dep),
):
    try:
        return JSONResponse(_config_from_json(data))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/md")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool("md")
async def get_markdown(
    request: Request,
    body: MarkdownRequest,
    _td: Dict = Depends(token_dep),
):
    """
    Convert a web page into Markdown format.

    Supports multiple extraction modes:
    - fit (default): Readability-based extraction for clean content
    - raw: Direct DOM to Markdown conversion
    - bm25: BM25 relevance ranking with optional query
    - llm: LLM-based summarization with optional query

    Use this tool when you need clean, readable text from web pages.
    """
    if not body.url.startswith(("http://", "https://")) and not body.url.startswith(("raw:", "raw://")):
        raise HTTPException(
            400, "Invalid URL format. Must start with http://, https://, or for raw HTML (raw:, raw://)")
    # base_url is intentionally not accepted from the request (key-exfil vector);
    # the LLM endpoint is server-derived from the provider name only.
    markdown = await handle_markdown_request(
        redis, body.url, body.f, body.q, body.c, config, body.provider,
        body.temperature
    )
    return JSONResponse({
        "url": body.url,
        "filter": body.f,
        "query": body.q,
        "cache": body.c,
        "markdown": markdown,
        "success": True
    })


@app.post("/html")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool("html")
async def generate_html(
    request: Request,
    body: HTMLRequest,
    _td: Dict = Depends(token_dep),
):
    """
    Crawls the URL, preprocesses the raw HTML for schema extraction, and returns the processed HTML.
    Use when you need sanitized HTML structures for building schemas or further processing.
    """
    await validate_url_scheme(body.url, allow_raw=True)
    cfg = server_crawler_config(config)
    crawler = None
    try:
        crawler = await get_crawler(get_default_browser_config())
        results = await crawler.arun(url=body.url, config=cfg)
        if not results[0].success:
            # Upstream fetch failed (anti-bot block, navigation refusal, DNS,
            # timeout): a gateway failure with its reason, not an internal 500
            # whose detail the central handler must genericize away.
            raise HTTPException(502, detail=public_error_detail(results[0].error_message))

        raw_html = results[0].html
        from crawl4ai.utils import preprocess_html_for_schema
        processed_html = preprocess_html_for_schema(raw_html)
        return JSONResponse({"html": processed_html, "url": body.url, "success": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        if crawler:
            await release_crawler(crawler)

# ── artifact store helpers ───────────────────────────────────
def _store_artifact(kind: str, data: bytes) -> dict:
    """Write to the sandboxed store; map quota/size errors to HTTP codes."""
    from artifacts import ArtifactTooLarge, QuotaExceeded, write_artifact
    try:
        meta = write_artifact(kind, data)
    except ArtifactTooLarge:
        raise HTTPException(413, "Artifact too large")
    except QuotaExceeded:
        raise HTTPException(507, "Artifact storage quota exceeded")
    return {
        "artifact_id": meta["artifact_id"],
        "url": f"/artifacts/{meta['artifact_id']}",
        "mime": meta["mime"],
        "size": meta["size"],
    }


@app.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str, _td: Dict = Depends(token_dep)):
    """Fetch a previously generated artifact by its opaque id (authed)."""
    from artifacts import ArtifactNotFound, resolve_artifact
    try:
        path, mime = await asyncio.to_thread(resolve_artifact, artifact_id)
    except ArtifactNotFound:
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, media_type=mime, headers={"X-Content-Type-Options": "nosniff"})


# Screenshot endpoint


@app.post("/screenshot")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool("screenshot")
async def generate_screenshot(
    request: Request,
    body: ScreenshotRequest,
    _td: Dict = Depends(token_dep),
):
    """
    Capture a full-page PNG screenshot of the specified URL, waiting an optional delay before capture.
    Use when you need an image snapshot of the rendered page. The image is also written to the
    sandboxed artifact store; the response includes an `artifact_id` and a `url` to fetch it.
    """
    await validate_url_scheme(body.url)
    crawler = None
    try:
        cfg = server_crawler_config(
            config,
            screenshot=True,
            screenshot_wait_for=body.screenshot_wait_for,
            wait_for_images=body.wait_for_images,
        )
        crawler = await get_crawler(get_default_browser_config())
        results = await crawler.arun(url=body.url, config=cfg)
        if not results[0].success:
            # Upstream fetch failed (anti-bot block, navigation refusal, DNS,
            # timeout): a gateway failure with its reason, not an internal 500
            # whose detail the central handler must genericize away.
            raise HTTPException(502, detail=public_error_detail(results[0].error_message))
        screenshot_data = results[0].screenshot
        art = await asyncio.to_thread(
            _store_artifact,
            "png",
            base64.b64decode(screenshot_data),
        )
        return {"success": True, "screenshot": screenshot_data, **art}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        if crawler:
            await release_crawler(crawler)

# PDF endpoint


@app.post("/pdf")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool("pdf")
async def generate_pdf(
    request: Request,
    body: PDFRequest,
    _td: Dict = Depends(token_dep),
):
    """
    Generate a PDF document of the specified URL.
    Use when you need a printable or archivable snapshot of the page. The PDF is also written to the
    sandboxed artifact store; the response includes an `artifact_id` and a `url` to fetch it.
    """
    await validate_url_scheme(body.url)
    crawler = None
    try:
        cfg = server_crawler_config(config, pdf=True)
        crawler = await get_crawler(get_default_browser_config())
        results = await crawler.arun(url=body.url, config=cfg)
        if not results[0].success:
            # Upstream fetch failed (anti-bot block, navigation refusal, DNS,
            # timeout): a gateway failure with its reason, not an internal 500
            # whose detail the central handler must genericize away.
            raise HTTPException(502, detail=public_error_detail(results[0].error_message))
        pdf_data = results[0].pdf
        art = await asyncio.to_thread(_store_artifact, "pdf", pdf_data)
        return {"success": True, "pdf": base64.b64encode(pdf_data).decode(), **art}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        if crawler:
            await release_crawler(crawler)


@app.post("/execute_js")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool("execute_js")
async def execute_js(
    request: Request,
    body: JSEndpointRequest,
    _td: Dict = Depends(token_dep),
):
    """
    Execute a sequence of JavaScript snippets on the specified URL.
    Return the full CrawlResult JSON (first result).
    Use this when you need to interact with dynamic pages using JS.
    REMEMBER: Scripts accept a list of separated JS snippets to execute and execute them in order.
    IMPORTANT: Each script should be an expression that returns a value. It can be an IIFE or an async function. You can think of it as such.
        Your script will replace '{script}' and execute in the browser context. So provide either an IIFE or a sync/async function that returns a value.
    Return Format:
        - The return result is an instance of CrawlResult, so you have access to markdown, links, and other stuff. If this is enough, you don't need to call again for other endpoints.

        ```python
        class CrawlResult(BaseModel):
            url: str
            html: str
            success: bool
            cleaned_html: Optional[str] = None
            media: Dict[str, List[Dict]] = {}
            links: Dict[str, List[Dict]] = {}
            downloaded_files: Optional[List[str]] = None
            js_execution_result: Optional[Dict[str, Any]] = None
            screenshot: Optional[str] = None
            pdf: Optional[bytes] = None
            mhtml: Optional[str] = None
            _markdown: Optional[MarkdownGenerationResult] = PrivateAttr(default=None)
            extracted_content: Optional[str] = None
            metadata: Optional[dict] = None
            error_message: Optional[str] = None
            session_id: Optional[str] = None
            response_headers: Optional[dict] = None
            status_code: Optional[int] = None
            ssl_certificate: Optional[SSLCertificate] = None
            dispatch_result: Optional[DispatchResult] = None
            redirected_url: Optional[str] = None
            network_requests: Optional[List[Dict[str, Any]]] = None
            console_messages: Optional[List[Dict[str, Any]]] = None

        class MarkdownGenerationResult(BaseModel):
            raw_markdown: str
            markdown_with_citations: str
            references_markdown: str
            fit_markdown: Optional[str] = None
            fit_html: Optional[str] = None
        ```

    """
    if not EXECUTE_JS_ENABLED:
        raise HTTPException(403, "execute_js endpoint is disabled. Set CRAWL4AI_EXECUTE_JS_ENABLED=true to enable.")
    await validate_url_scheme(body.url, check_destination=False)
    try:
        await asyncio.to_thread(validate_webhook_url, body.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    crawler = None
    try:
        cfg = server_crawler_config(config, js_code=body.scripts)
        crawler = await get_crawler(get_default_browser_config())
        results = await crawler.arun(url=body.url, config=cfg)
        if not results[0].success:
            # Upstream fetch failed (anti-bot block, navigation refusal, DNS,
            # timeout): a gateway failure with its reason, not an internal 500
            # whose detail the central handler must genericize away.
            raise HTTPException(502, detail=public_error_detail(results[0].error_message))
        data = results[0].model_dump()
        if data.get("error_message"):
            data["error_message"] = public_crawl_error(data["error_message"], data.get("url"))
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        if crawler:
            await release_crawler(crawler)


@app.get("/llm/{url:path}")
async def llm_endpoint(
    request: Request,
    url: str = Path(...),
    q: str = Query(...),
    provider: Optional[str] = Query(None, description="LLM provider override, e.g. 'openai/gpt-4o-mini'"),
    temperature: Optional[float] = Query(None, description="LLM temperature override"),
    _td: Dict = Depends(token_dep),
):
    # base_url is intentionally not accepted (key-exfil vector); the endpoint is
    # derived server-side from the provider name only.
    if not q:
        raise HTTPException(400, "Query parameter 'q' is required")
    if not url.startswith(("http://", "https://")) and not url.startswith(("raw:", "raw://")):
        url = "https://" + url
    answer = await handle_llm_qa(
        url,
        q,
        config,
        provider=provider,
        temperature=temperature,
        redis=redis,
    )
    return JSONResponse({"answer": answer})


@app.get("/schema")
async def get_schema():
    from crawl4ai import BrowserConfig, CrawlerRunConfig
    return {"browser": BrowserConfig().dump(),
            "crawler": CrawlerRunConfig().dump()}


@app.get("/hooks/info")
async def get_hooks_info():
    """Enumerate the available declarative hook actions and their parameter schemas.

    Arbitrary hook code is no longer accepted (it was an exec()-based RCE
    surface). Each action maps to a server-authored, single-purpose Playwright
    operation; clients select an action and supply schema-validated params.
    """
    from hook_registry import describe_registry

    return JSONResponse({
        "available_actions": describe_registry(),
        "usage": {
            "field": "hooks",
            "shape": [{"action": "<action>", "params": {"...": "..."}}],
            "max_hooks": 10,
        },
    })


@app.get(HEALTH_PATH)
async def health():
    headers = {"Connection": "close"}
    payload = {
        "status": "unhealthy",
        "timestamp": time.time(),
        "version": __version__,
        "revision": os.environ.get("C4AI_GIT_SHA", ""),
        "instance": os.environ.get("HOSTNAME", ""),
        "components": {"api": "unavailable"},
    }
    if not getattr(app.state, "readiness_checks_active", False):
        return JSONResponse(payload, status_code=503, headers=headers)
    try:
        await asyncio.wait_for(redis.ping(), timeout=2.0)
        payload["status"] = "ok"
        payload["components"] = {"api": "ready", "redis": "ready"}
        return JSONResponse(payload, headers=headers)
    except Exception as exc:
        # Log the failure class: a fast ConnectionError points at the overlay /
        # stale pooled connection, a TimeoutError at a stalled Redis.
        logger.warning("health: redis ping failed: %r", exc)
        payload["components"]["redis"] = "unavailable"
        return JSONResponse(payload, status_code=503, headers=headers)


@app.get(config["observability"]["prometheus"]["endpoint"])
async def metrics():
    return RedirectResponse(config["observability"]["prometheus"]["endpoint"])


@app.post("/crawl")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool("crawl")
async def crawl(
    request: Request,
    crawl_request: CrawlRequestWithHooks,
    _td: Dict = Depends(token_dep),
):
    """
    Crawl a list of URLs and return the results as JSON.
    For streaming responses, use /crawl/stream endpoint.
    Supports optional user-provided hook functions for customization.
    """
    if not crawl_request.urls:
        raise HTTPException(400, "At least one URL required")
    if crawl_request.hooks and crawl_request.hooks.hooks and not HOOKS_ENABLED:
        raise HTTPException(403, "Hooks are disabled. Set CRAWL4AI_HOOKS_ENABLED=true to enable.")
    # Check whether it is a redirection for a streaming request
    try:
        crawler_config = CrawlerRunConfig.load(
            crawl_request.crawler_config, provenance=Provenance.UNTRUSTED
        )
    except UntrustedConfigError as e:
        raise HTTPException(400, f"Rejected config: {e}")
    if crawler_config.stream:
        return await stream_process(crawl_request=crawl_request)
    
    # Prepare hooks config if provided
    hooks_config = None
    if crawl_request.hooks and crawl_request.hooks.hooks:
        hooks_config = {'hooks': crawl_request.hooks.hooks}
    
    results = await handle_crawl_request(
        urls=crawl_request.urls,
        browser_config=crawl_request.browser_config,
        crawler_config=crawl_request.crawler_config,
        config=config,
        hooks_config=hooks_config,
        crawler_configs=crawl_request.crawler_configs,
    )
    if not results["success"]:
        # Every URL failed upstream: surface the first reason as a gateway
        # error instead of a genericized internal 500. handle_crawl_request
        # already sanitized each error_message, so it passes through as-is.
        first = results["results"][0] if results["results"] else {}
        raise HTTPException(
            502,
            f"Crawl request failed: {first.get('error_message') or 'Crawl failed'}",
        )
    return JSONResponse(results)


@app.post("/crawl/stream")
@limiter.limit(config["rate_limiting"]["default_limit"])
async def crawl_stream(
    request: Request,
    crawl_request: CrawlRequestWithHooks,
    _td: Dict = Depends(token_dep),
):
    if not crawl_request.urls:
        raise HTTPException(400, "At least one URL required")
    if crawl_request.hooks and crawl_request.hooks.hooks and not HOOKS_ENABLED:
        raise HTTPException(403, "Hooks are disabled. Set CRAWL4AI_HOOKS_ENABLED=true to enable.")

    return await stream_process(crawl_request=crawl_request)

async def stream_process(crawl_request: CrawlRequestWithHooks):
    
    # Prepare hooks config if provided# Prepare hooks config if provided
    hooks_config = None
    if crawl_request.hooks and crawl_request.hooks.hooks:
        hooks_config = {'hooks': crawl_request.hooks.hooks}
    
    crawler, gen, hooks_info = await handle_stream_crawl_request(
        urls=crawl_request.urls,
        browser_config=crawl_request.browser_config,
        crawler_config=crawl_request.crawler_config,
        config=config,
        hooks_config=hooks_config
    )
    
    # Add hooks info to response headers if available
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Stream-Status": "active",
    }
    if hooks_info:
        import json
        headers["X-Hooks-Status"] = json.dumps(hooks_info['status']['status'])
    
    return StreamingResponse(
        stream_results(crawler, gen),
        media_type="application/x-ndjson",
        headers=headers,
    )


def chunk_code_functions(code_md: str) -> List[str]:
    """Extract each function/class from markdown code blocks per file."""
    pattern = re.compile(
        # match "## File: <path>" then a ```py fence, then capture until the closing ```
        r'##\s*File:\s*(?P<path>.+?)\s*?\r?\n'      # file header
        r'```py\s*?\r?\n'                         # opening fence
        r'(?P<code>.*?)(?=\r?\n```)',             # code block
        re.DOTALL
    )
    chunks: List[str] = []
    for m in pattern.finditer(code_md):
        file_path = m.group("path").strip()
        code_blk = m.group("code")
        tree = ast.parse(code_blk)
        lines = code_blk.splitlines()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 1)
                snippet = "\n".join(lines[start:end])
                chunks.append(f"# File: {file_path}\n{snippet}")
    return chunks


def chunk_doc_sections(doc: str) -> List[str]:
    lines = doc.splitlines(keepends=True)
    sections = []
    current: List[str] = []
    for line in lines:
        if re.match(r"^#{1,6}\s", line):
            if current:
                sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current))
    return sections


@app.get("/ask")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool("ask")
async def get_context(
    request: Request,
    _td: Dict = Depends(token_dep),
    context_type: str = Query("all", pattern="^(code|doc|all)$"),
    query: Optional[str] = Query(
        None, description="search query to filter chunks"),
    score_ratio: float = Query(
        0.5, ge=0.0, le=1.0, description="min score as fraction of max_score"),
    max_results: int = Query(
        20, ge=1, description="absolute cap on returned chunks"),
):
    """
    This end point is design for any questions about Crawl4ai library. It returns a plain text markdown with extensive information about Crawl4ai. 
    You can use this as a context for any AI assistant. Use this endpoint for AI assistants to retrieve library context for decision making or code generation tasks.
    Alway is BEST practice you provide a query to filter the context. Otherwise the lenght of the response will be very long.

    Parameters:
    - context_type: Specify "code" for code context, "doc" for documentation context, or "all" for both.
    - query: RECOMMENDED search query to filter paragraphs using BM25. You can leave this empty to get all the context.
    - score_ratio: Minimum score as a fraction of the maximum score for filtering results.
    - max_results: Maximum number of results to return. Default is 20.

    Returns:
    - JSON response with the requested context.
    - If "code" is specified, returns the code context.
    - If "doc" is specified, returns the documentation context.
    - If "all" is specified, returns both code and documentation contexts.
    - When a query is provided but matches no terms in the corpus, the
      corresponding result lists ("code_results" and/or "doc_results") are
      empty rather than populated with irrelevant score=0.0 chunks. Use
      query=None to retrieve the full, unfiltered context.
    """
    # load contexts
    base = os.path.dirname(__file__)
    code_path = os.path.join(base, "c4ai-code-context.md")
    doc_path = os.path.join(base, "c4ai-doc-context.md")
    if not os.path.exists(code_path) or not os.path.exists(doc_path):
        raise HTTPException(404, "Context files not found")

    with open(code_path, "r") as f:
        code_content = f.read()
    with open(doc_path, "r") as f:
        doc_content = f.read()

    # if no query, just return raw contexts
    if not query:
        if context_type == "code":
            return JSONResponse({"code_context": code_content})
        if context_type == "doc":
            return JSONResponse({"doc_context": doc_content})
        return JSONResponse({
            "code_context": code_content,
            "doc_context": doc_content,
        })

    tokens = query.split()
    results: Dict[str, List[Dict[str, float]]] = {}

    # code BM25 over functions/classes
    if context_type in ("code", "all"):
        code_chunks = chunk_code_functions(code_content)
        bm25 = BM25Okapi([c.split() for c in code_chunks])
        scores = bm25.get_scores(tokens)
        max_sc = float(scores.max()) if scores.size > 0 else 0.0
        if max_sc <= 0:
            results["code_results"] = []
        else:
            cutoff = max_sc * score_ratio
            picked = [(c, s) for c, s in zip(code_chunks, scores) if s >= cutoff]
            picked = sorted(picked, key=lambda x: x[1], reverse=True)[:max_results]
            results["code_results"] = [{"text": c, "score": s} for c, s in picked]

    # doc BM25 over markdown sections
    if context_type in ("doc", "all"):
        sections = chunk_doc_sections(doc_content)
        bm25d = BM25Okapi([sec.split() for sec in sections])
        scores_d = bm25d.get_scores(tokens)
        max_sd = float(scores_d.max()) if scores_d.size > 0 else 0.0
        if max_sd <= 0:
            results["doc_results"] = []
        else:
            cutoff_d = max_sd * score_ratio
            idxs = [i for i, s in enumerate(scores_d) if s >= cutoff_d]
            neighbors = set(i for idx in idxs for i in (idx-1, idx, idx+1))
            valid = [i for i in sorted(neighbors) if 0 <= i < len(sections)]
            valid = valid[:max_results]
            results["doc_results"] = [
                {"text": sections[i], "score": scores_d[i]} for i in valid
            ]

    return JSONResponse(results)


# attach MCP layer (adds /mcp/ws, /mcp/sse, /mcp/schema)
print(f"MCP server running on {config['app']['host']}:{config['app']['port']}")
attach_mcp(
    app,
    # Internal MCP tool calls go over loopback to our own gated endpoints,
    # carrying a service token. Pin to 127.0.0.1 (config host may be 0.0.0.0,
    # which is a bind address, not a valid connect target).
    base_url=f"http://127.0.0.1:{config['app']['port']}",
    auth_headers_provider=_internal_service_auth_headers,
)

# ────────────────────────── cli ──────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config["app"]["host"],
        port=config["app"]["port"],
        reload=config["app"]["reload"],
        timeout_keep_alive=config["app"]["timeout_keep_alive"],
    )
# ─────────────────────────────────────────────────────────────
