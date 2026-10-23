import json
import logging
import os
import re
import uuid
import zlib
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Optional, assert_never

import dns.resolver
import yaml
from error_reporting import init_error_reporting
from fastapi import Request

from egress_proxy import tunnel_failure_code


class TaskStatus(str, Enum):
    PROCESSING = "processing"
    FAILED = "failed"
    COMPLETED = "completed"


class ResultEncoding(str, Enum):
    # Deploy these readers everywhere before enabling compressed writers.
    IDENTITY = "identity"
    ZLIB = "zlib"


MAX_SERIALIZED_RESULT_BYTES = 16 * 1024 * 1024
MAX_STORED_RESULT_BYTES = 4 * 1024 * 1024


class ResultTooLarge(ValueError):
    """A result exceeded its serialized or stored byte budget."""

    def __init__(self, dimension: str, actual: int, limit: int):
        self.dimension = dimension
        self.actual = actual
        self.limit = limit
        super().__init__(f"result {dimension} {actual} exceeds limit {limit}")


def encode_task_result(
    result: Any,
    encoding: ResultEncoding,
    *,
    max_serialized_bytes: int = MAX_SERIALIZED_RESULT_BYTES,
    max_stored_bytes: int = MAX_STORED_RESULT_BYTES,
) -> str | bytes:
    """Serialize task results at the hash-format owner."""
    serialized = json.dumps(result)
    serialized_bytes = serialized.encode("utf-8")
    if len(serialized_bytes) > max_serialized_bytes:
        raise ResultTooLarge("serialized_bytes", len(serialized_bytes), max_serialized_bytes)
    match encoding:
        case ResultEncoding.IDENTITY:
            stored = serialized
            stored_size = len(serialized_bytes)
        case ResultEncoding.ZLIB:
            stored = zlib.compress(serialized_bytes, level=6)
            stored_size = len(stored)
        case _:
            assert_never(encoding)
    if stored_size > max_stored_bytes:
        raise ResultTooLarge("stored_bytes", stored_size, max_stored_bytes)
    return stored


def decode_task_result(
    task: dict,
    task_id: str,
    *,
    max_serialized_bytes: int = MAX_SERIALIZED_RESULT_BYTES,
) -> Any:
    """Missing markers are legacy JSON; unknown markers must never be guessed."""
    if max_serialized_bytes < 0:
        raise ValueError("max_serialized_bytes must be nonnegative")
    marker = task.get("result_encoding", ResultEncoding.IDENTITY.value)
    try:
        if isinstance(marker, bytes):
            marker = marker.decode("utf-8")
        encoding = ResultEncoding(marker)
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"task {task_id}: unknown result_encoding {marker!r}") from error
    result = task.get("result")
    if not result:
        return None
    match encoding:
        case ResultEncoding.IDENTITY:
            serialized_size = len(result.encode("utf-8") if isinstance(result, str) else result)
            if serialized_size > max_serialized_bytes:
                raise ResultTooLarge("serialized_bytes", serialized_size, max_serialized_bytes)
            return json.loads(result)
        case ResultEncoding.ZLIB:
            decompressor = zlib.decompressobj()
            # The extra byte detects overflow without expanding the entire result.
            serialized = decompressor.decompress(result, max_serialized_bytes + 1)
            if len(serialized) > max_serialized_bytes:
                raise ResultTooLarge("serialized_bytes", len(serialized), max_serialized_bytes)
            if not decompressor.eof:
                raise zlib.error("incomplete or truncated result stream")
            return json.loads(serialized.decode("utf-8"))
        case _:
            assert_never(encoding)


class FilterType(str, Enum):
    RAW = "raw"
    FIT = "fit"
    BM25 = "bm25"
    LLM = "llm"

DEFAULT_CONFIG = {
    "app": {
        "title": "Crawl4AI API",
        "version": "1.0.0",
        "host": "0.0.0.0",
        "port": 11235,
        "reload": False,
        "workers": 1,
        "timeout_keep_alive": 300,
    },
    "llm": {
        "provider": "openai/gpt-4o-mini",
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": "",
        "task_ttl_seconds": 3600,
        "ssl": False,
    },
    "crawl_jobs": {
        "protocol_version": 3,
        "stream": "crawl-jobs",
        "group": "crawl-workers",
        "lease_seconds": 20,
        "heartbeat_seconds": 5,
        "read_block_ms": 1000,
        "max_attempts": 3,
        "max_pending_jobs": 1000,
        "max_attempt_seconds": 600,
        "max_payload_bytes": 262144,
        "redis_headroom_bytes": 100663296,
        "consumer_reap_batch": 100,
    },
    "rate_limiting": {
        "enabled": True,
        "default_limit": "1000/minute",
        "trusted_proxies": [],
        "storage_uri": "memory://",
    },
    "security": {
        "enabled": False,
        "jwt_enabled": False,
        "api_token": "",
        "https_redirect": False,
        "trusted_hosts": ["*"],
        "headers": {
            "x_content_type_options": "nosniff",
            "x_frame_options": "DENY",
            "content_security_policy": "default-src 'self'",
            "strict_transport_security": "max-age=63072000; includeSubDomains",
        },
    },
    "crawler": {
        "base_config": {"delay_before_return_html": 1.0},
        "memory_threshold_percent": 95.0,
        "rate_limiter": {"enabled": True, "base_delay": [1.0, 2.0]},
        "timeouts": {"stream_init": 30.0, "batch_process": 300.0},
        "pool": {
            "max_pages": 40,
            "idle_ttl_sec": 300,
            "replica_max_pages": 8,
            "request_max_pages": 4,
            "admission_wait_seconds": 25.0,
            "worker_admission_wait_seconds": 300.0,
        },
        "browser": {
            "kwargs": {"headless": True, "text_mode": True, "avoid_ads": True},
            "extra_args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--allow-insecure-localhost",
                "--ignore-certificate-errors",
            ],
        },
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "observability": {
        "prometheus": {"enabled": True, "endpoint": "/metrics"},
        "health_check": {"endpoint": "/health", "liveness_endpoint": "/health/live"},
    },
    "webhooks": {
        "enabled": True,
        "default_url": None,
        "data_in_payload": False,
        "retry": {
            "max_attempts": 5,
            "initial_delay_ms": 1000,
            "max_delay_ms": 32000,
            "timeout_ms": 30000,
        },
        "headers": {"User-Agent": "Crawl4AI-Webhook/1.0"},
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class PoolAdmissionSettings:
    replica_max_pages: int = 8
    request_max_pages: int = 4
    admission_wait_seconds: float = 25.0
    worker_admission_wait_seconds: float = 300.0

    @classmethod
    def from_config(cls, config: dict) -> "PoolAdmissionSettings":
        configured = config.get("crawler", {}).get("pool", {})
        pages = {}
        for name in ("replica_max_pages", "request_max_pages"):
            count = configured.get(name, getattr(cls, name))
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"crawler.pool.{name} must be a positive integer")
            pages[name] = count
        waits = {}
        for name in ("admission_wait_seconds", "worker_admission_wait_seconds"):
            wait = configured.get(name, getattr(cls, name))
            if (
                isinstance(wait, bool)
                or not isinstance(wait, (int, float))
                or not isfinite(wait)
                or wait < 0
            ):
                raise ValueError(f"crawler.pool.{name} must be a finite nonnegative number")
            waits[name] = float(wait)
        return cls(**pages, **waits)


@dataclass(frozen=True)
class BrowserWatchdogSettings:
    interval_seconds: float = 5.0
    renderer_cpu_cores: float = 0.9
    renderer_cpu_seconds: float = 65.0
    renderer_rss_mb: int = 768
    browser_rss_mb: int = 600
    browser_probe_timeout_seconds: float = 5.0
    consecutive_failure_limit: int = 3

    @classmethod
    def from_config(cls, config: dict) -> "BrowserWatchdogSettings":
        configured = config.get("crawler", {}).get("pool", {}).get("watchdog", {})
        values = {}
        for field in fields(cls):
            value = configured.get(field.name, field.default)
            expected = int if field.type is int else (int, float)
            if (
                isinstance(value, bool)
                or not isinstance(value, expected)
                or not isfinite(value)
                or value <= 0
            ):
                kind = "integer" if expected is int else "number"
                raise ValueError(
                    f"crawler.pool.watchdog.{field.name} must be a positive {kind}"
                )
            values[field.name] = value
        return cls(**values)


@dataclass(frozen=True)
class CrawlResourceSettings:
    max_iframes: int = 20
    page_termination_timeout: float = 5.0

    @classmethod
    def from_config(cls, config: dict) -> "CrawlResourceSettings":
        configured = config.get("crawler", {}).get("base_config", {})
        max_iframes = configured.get("max_iframes", cls.max_iframes)
        if (
            isinstance(max_iframes, bool)
            or not isinstance(max_iframes, int)
            or max_iframes <= 0
        ):
            raise ValueError("crawler.base_config.max_iframes must be a positive integer")
        timeout = configured.get("page_termination_timeout", cls.page_termination_timeout)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "crawler.base_config.page_termination_timeout must be a positive number"
            )
        return cls(max_iframes=max_iframes, page_termination_timeout=timeout)


def load_config() -> Dict:
    """Load and return application configuration with environment variable overrides."""
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, "r") as config_file:
        user_config = yaml.safe_load(config_file) or {}

    # Deep-merge user config on top of defaults so missing keys get safe values
    config = _deep_merge(DEFAULT_CONFIG, user_config)

    for section in DEFAULT_CONFIG:
        if section not in user_config:
            logging.warning(
                f"Config section '{section}' missing from config.yml, using defaults"
            )
    
    # Override LLM provider from environment if set
    llm_provider = os.environ.get("LLM_PROVIDER")
    if llm_provider:
        config["llm"]["provider"] = llm_provider
        logging.info(f"LLM provider overridden from environment: {llm_provider}")
    
    # Also support direct API key from environment if the provider-specific key isn't set
    llm_api_key = os.environ.get("LLM_API_KEY")
    if llm_api_key and "api_key" not in config["llm"]:
        config["llm"]["api_key"] = llm_api_key
        logging.info("LLM API key loaded from LLM_API_KEY environment variable")

    # Override Redis task TTL from environment if set
    redis_task_ttl = os.environ.get("REDIS_TASK_TTL")
    if redis_task_ttl:
        try:
            config["redis"]["task_ttl_seconds"] = int(redis_task_ttl)
            logging.info(f"Redis task TTL overridden from REDIS_TASK_TTL: {redis_task_ttl}s")
        except ValueError:
            logging.warning(f"Invalid REDIS_TASK_TTL value: {redis_task_ttl}, using default")

    return config


def get_browser_extra_args(config: Dict) -> list[str]:
    """Resolve operator Chromium flags against the sandbox opt-in."""
    configured_args = config["crawler"]["browser"].get("extra_args", [])
    if os.environ.get("CRAWL4AI_CHROMIUM_SANDBOX", "false").lower() == "true":
        return [arg for arg in configured_args if arg != "--no-sandbox"]
    return list(configured_args)


class CRLFSafeFilter(logging.Filter):
    """Strip CR/LF/control chars from log records (log-injection / forging).

    A crawl URL or error reflected into a log line could otherwise inject
    newlines and forge additional log entries.
    """

    _BAD = {ord(c): None for c in "\r\n"} | {i: None for i in range(0, 32) if i not in (9,)}

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            cleaned = msg.translate(self._BAD)
            if cleaned != msg:
                record.msg = cleaned
                record.args = ()
        except Exception:
            pass
        return True


def setup_logging(config: Dict) -> None:
    """Configure application logging with CRLF-safe records."""
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"]
    )
    # Configuration loading may have already installed the default WARNING handler.
    logging.getLogger().setLevel(config["logging"]["level"])
    crlf = CRLFSafeFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(crlf)
    init_error_reporting()

def get_base_url(request: Request) -> str:
    """Get base URL including scheme and host."""
    return f"{request.url.scheme}://{request.url.netloc}"


def is_task_id(value: str) -> bool:
    """Check if the value matches task ID pattern."""
    return value.startswith("llm_") and "_" in value

logger = logging.getLogger(__name__)

_INTERNAL_ERROR_MARKERS = ("Unexpected error in", "Code context", "Traceback", 'File "', ".py")
# The markers above only recognize the library's get_error_context shape. Its
# dispatcher reports a bare str(e) instead (crawl4ai/async_dispatcher.py:357
# and :769), so a path-bearing OSError carries no marker at all. Match any
# absolute container path directly, after removing the crawled URL — a URL
# legitimately has path segments, but whatever is left should not. A fixed
# prefix allowlist would miss operator overrides of standard env vars (e.g.
# PLAYWRIGHT_BROWSERS_PATH=/data/..., TMPDIR=/mnt/...) under unlisted
# top-level dirs, leaking them to the client, so any /word/ segment is matched.
# Directory names may begin with an uppercase letter or a digit too (e.g.
# /DATA/..., /123/...), so all three are accepted; the URL is stripped first
# to keep URL path segments from being mistaken for a container path.
_CONTAINER_PATH = re.compile(r"/(?:[^/\s][^/\s]*)/(?:\S)")
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
# Chromium's navigation error code. A withheld message keeps it because it is the
# one fact a client needs to tell a target-side fault from a transient one. Only
# Playwright's own goto error line counts: the call log echoes the crawled URL.
_NAVIGATION_ERROR_CODE = re.compile(r"Page\.goto: (net::ERR_[A-Z0-9_]{1,64})(?: at (\S+))?")
_TUNNEL_CONNECTION_FAILED = "net::ERR_TUNNEL_CONNECTION_FAILED"
# Playwright's own goto deadline line. A withheld timeout keeps it for the same
# reason as a Chromium code: without it a client cannot tell a page that never
# answered from an unexplained server fault.
_NAVIGATION_TIMEOUT = re.compile(r"Page\.goto: (Timeout \d{1,9}ms exceeded)")


def public_error_detail(error_message: Optional[str]) -> str:
    """Client-safe form of a failed CrawlResult's error_message.

    The library's catch-all error_message (get_error_context) embeds container
    file paths, source snippets and raw exception text; those must never reach
    a client, so only its bare Chromium `net::ERR_*` code or Playwright goto
    timeout survives. Upstream-failure messages (anti-bot, navigation refusal,
    DNS, timeouts) pass through, capped.
    """
    message = (error_message or "").strip()
    if (
        message
        and not any(marker in message for marker in _INTERNAL_ERROR_MARKERS)
        and not _CONTAINER_PATH.search(_URL.sub("", message))
    ):
        # A clean message — but a tunnel failure is not really clean: it is the
        # pinning proxy masking a dead target with a generic 403. The "All
        # proxies failed: ..." shape produced when CrawlerRunConfig.max_retries
        # >= 1 carries no internal marker or container path, so it takes this
        # early return. Restore the recorded target-side code for it alone;
        # other clean upstream-failure messages (anti-bot, cert, DNS, timeouts)
        # still pass through verbatim per the documented pass-through contract.
        navigation = _NAVIGATION_ERROR_CODE.search(message)
        if (
            navigation is not None
            and navigation.group(1) == _TUNNEL_CONNECTION_FAILED
            and navigation.group(2)
        ):
            swapped = tunnel_failure_code(navigation.group(2))
            if swapped:
                return f"Crawl failed: {swapped}"
        return message[:500]
    navigation = _NAVIGATION_ERROR_CODE.search(message)
    if navigation is None:
        timeout = _NAVIGATION_TIMEOUT.search(message)
        return f"Crawl failed: {timeout.group(1)}" if timeout else "Crawl failed"
    code, navigated_url = navigation.groups()
    if code == _TUNNEL_CONNECTION_FAILED and navigated_url:
        # The pinning proxy dials for the browser, so a dead target surfaces as a
        # tunnel failure; restore the code a direct dial would have produced.
        code = tunnel_failure_code(navigated_url) or code
    return f"Crawl failed: {code}"


class CorrelatedError(str):
    """Client-safe detail paired with its private log correlation id."""

    def __new__(cls, detail: str, correlation_id: Optional[str] = None):
        value = super().__new__(cls, detail)
        value.correlation_id = correlation_id
        return value


def correlated_error(
    public: str,
    detail: object,
    context: str = "crawl failure",
    level: int = logging.ERROR,
) -> CorrelatedError:
    """Stamp a client-safe message with the correlation id its full text is logged under.

    Nothing is discarded: the caller gets a message that cannot leak internals
    and an id an operator can grep the server log for. The logged text is
    capped because this runs once per failed URL and a batch can be large.
    """
    cid = uuid.uuid4().hex[:12]
    logger.log(
        level,
        "%s [cid=%s]: %s",
        context,
        cid,
        str(detail)[:2000],
        exc_info=detail if isinstance(detail, BaseException) else False,
    )
    return CorrelatedError(f"{public} (correlation_id={cid})", cid)


def public_crawl_error(
    raw: Optional[str], url: Optional[str] = None
) -> CorrelatedError:
    """Client-safe form of a failed result's error_message.

    Every crawl surface serializes CrawlResult verbatim, so this runs at each
    serialization owner. A correlation id is minted only when something was
    actually withheld: an upstream reason that passes through, one that is
    merely capped, and an empty message all leave the log alone.
    """
    message = (raw or "").strip()
    public = public_error_detail(raw)
    if not message or public == message[:500]:
        return CorrelatedError(public)
    # A recovered Chromium code or goto timeout means the target site failed, so
    # it stays below the ERROR level that Sentry reports. A bare "Crawl failed"
    # withheld a traceback or container path: a server-side fault.
    level = logging.ERROR if public == "Crawl failed" else logging.WARNING
    return correlated_error(public, raw, f"crawl failure url={url or 'unknown'}", level)


def datetime_handler(obj: any) -> Optional[str]:
    """Handle datetime serialization for JSON."""
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def should_cleanup_task(created_at: str, ttl_seconds: int = 3600) -> bool:
    """Return whether ``created_at`` is older than ``ttl_seconds``.

    A ``ttl_seconds`` of 0 (or any negative value) returns False, mirroring the
    "0 == disabled" sentinel of ``redis.task_ttl_seconds``. The status handler
    does not use this age check: polling leaves producer-set expiry unchanged.
    """
    if ttl_seconds <= 0:
        return False
    created = datetime.fromisoformat(created_at)
    return (datetime.now() - created).total_seconds() > ttl_seconds

def decode_redis_hash(hash_data: dict) -> dict[str, str | bytes]:
    """Decode metadata, preserving result bytes for the encoding-aware reader."""
    return {
        k.decode("utf-8") if isinstance(k, bytes) else k:
        v if k in (b"result", "result", b"result_encoding", "result_encoding")
        or not isinstance(v, bytes) else v.decode("utf-8")
        for k, v in hash_data.items()
    }


def get_redis_task_ttl(config: Dict) -> int:
    """Get Redis task TTL in seconds from config.

    Args:
        config: The application configuration dictionary

    Returns:
        TTL in seconds (default 3600). Returns 0 if TTL is disabled.
    """
    return config.get("redis", {}).get("task_ttl_seconds", 3600)


def get_llm_api_key(config: Dict, provider: Optional[str] = None) -> Optional[str]:
    """Get the appropriate API key based on the LLM provider.
    
    Args:
        config: The application configuration dictionary
        provider: Optional provider override (e.g., "openai/gpt-4")
    
    Returns:
        The API key if directly configured, otherwise None to let litellm handle it
    """
    # Check if direct API key is configured (for backward compatibility)
    if "api_key" in config["llm"]:
        return config["llm"]["api_key"]
    
    # Return None - litellm will automatically find the right environment variable
    return None


def validate_llm_provider(config: Dict, provider: Optional[str] = None) -> tuple[bool, str]:
    """Validate that the LLM provider has an associated API key.
    
    Args:
        config: The application configuration dictionary
        provider: Optional provider override (e.g., "openai/gpt-4")
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # If a direct API key is configured, validation passes
    if "api_key" in config["llm"]:
        return True, ""
    
    # Otherwise, trust that litellm will find the appropriate environment variable
    # We can't easily validate this without reimplementing litellm's logic
    return True, ""


def get_llm_temperature(config: Dict, provider: Optional[str] = None) -> Optional[float]:
    """Get temperature setting based on the LLM provider.
    
    Priority order:
    1. Provider-specific environment variable (e.g., OPENAI_TEMPERATURE)
    2. Global LLM_TEMPERATURE environment variable
    3. None (to use litellm/provider defaults)
    
    Args:
        config: The application configuration dictionary
        provider: Optional provider override (e.g., "openai/gpt-4")
    
    Returns:
        The temperature setting if configured, otherwise None
    """
    # Check provider-specific temperature first
    if provider:
        provider_name = provider.split('/')[0].upper()
        provider_temp = os.environ.get(f"{provider_name}_TEMPERATURE")
        if provider_temp:
            try:
                return float(provider_temp)
            except ValueError:
                logging.warning(f"Invalid temperature value for {provider_name}: {provider_temp}")
    
    # Check global LLM_TEMPERATURE
    global_temp = os.environ.get("LLM_TEMPERATURE")
    if global_temp:
        try:
            return float(global_temp)
        except ValueError:
            logging.warning(f"Invalid global temperature value: {global_temp}")
    
    # Return None to use litellm/provider defaults
    return None


def get_llm_base_url(config: Dict, provider: Optional[str] = None) -> Optional[str]:
    """Get base URL setting based on the LLM provider.
    
    Priority order:
    1. Provider-specific environment variable (e.g., OPENAI_BASE_URL)
    2. Global LLM_BASE_URL environment variable
    3. None (to use default endpoints)
    
    Args:
        config: The application configuration dictionary
        provider: Optional provider override (e.g., "openai/gpt-4")
    
    Returns:
        The base URL if configured, otherwise None
    """
    # Check provider-specific base URL first
    if provider:
        provider_name = provider.split('/')[0].upper()
        provider_url = os.environ.get(f"{provider_name}_BASE_URL")
        if provider_url:
            return provider_url
    
    # Check global LLM_BASE_URL
    return os.environ.get("LLM_BASE_URL")


# ── Security utilities ──────────────────────────────────────
# validate_output_path / ALLOWED_OUTPUT_DIR were removed: caller-supplied
# output paths are no longer accepted (string-only validation was bypassable via
# symlink/TOCTOU and sibling-prefix names -> arbitrary write -> RCE). The server
# owns all artifact paths now (see artifacts.py).


import ipaddress
from urllib.parse import urlparse

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

_BLOCKED_HOSTNAMES = {
    "localhost", "metadata.google.internal", "metadata",
    "kubernetes.default", "kubernetes.default.svc",
}


ALLOW_INTERNAL_URLS = os.environ.get("CRAWL4AI_ALLOW_INTERNAL_URLS", "false").lower() == "true"


def validate_url_destination(url: str) -> None:
    """Block crawl URLs targeting internal/private networks (SSRF protection).
    Skipped when CRAWL4AI_ALLOW_INTERNAL_URLS=true.
    Skipped for raw: URLs (inline HTML, no network fetch)."""
    if ALLOW_INTERNAL_URLS:
        return
    if str(url).startswith(("raw:", "raw://")):
        return
    try:
        validate_webhook_url(url)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"URL blocked (SSRF protection): {e}")


def _expand_ip_candidates(ip):
    """Return [ip] plus any IPv4 form wrapped inside the IPv6 address.
    SSRF guards must check the unwrapped form because ::ffff:127.0.0.1 and
    ::127.0.0.1 route to 127.0.0.1 but would not match IPv4 blocklists directly."""
    candidates = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            candidates.append(ip.ipv4_mapped)
        else:
            as_int = int(ip)
            if 0 < as_int < 2**32:
                candidates.append(ipaddress.IPv4Address(as_int))
    return candidates


def validate_webhook_url(url: str) -> None:
    """Reject webhook/crawl URLs targeting non-global networks (SSRF protection).

    Delegates to the single egress rule (egress_broker: reject any resolved IP
    where not ip.is_global, including v4-mapped/NAT64/6to4/v4-compat embedded
    forms). The raised message is intentionally opaque - it never echoes the
    resolved IP or hostname, so this is not a DNS/oracle leak.
    """
    from egress_broker import EgressBlocked, resolve_and_pin
    parsed = urlparse(str(url))
    if not parsed.hostname:
        raise ValueError("URL must have a valid hostname")
    try:
        resolve_and_pin(url)
    except EgressBlocked:
        raise ValueError("URL blocked")


def verify_email_domain(email: str) -> bool:
    try:
        domain = email.split('@')[1]
        # Try to resolve MX records for the domain.
        records = dns.resolver.resolve(domain, 'MX')
        return True if records else False
    except Exception:
        return False

def get_container_memory_percent() -> float:
    """Get actual container memory usage vs limit (cgroup v1/v2 aware)."""
    import psutil

    try:
        # Try cgroup v2 first
        usage_path = Path("/sys/fs/cgroup/memory.current")
        limit_path = Path("/sys/fs/cgroup/memory.max")
        if not usage_path.exists():
            # Fall back to cgroup v1
            usage_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

        usage = int(usage_path.read_text())
        # Working set, the share the container memory alert measures: the
        # kernel reclaims inactive page cache before the limit, so counting it
        # recycled and refused browsers on replicas holding 1.5 GiB of cache.
        stat_path = usage_path.with_name("memory.stat")
        if stat_path.exists():
            stat = dict(line.split() for line in stat_path.read_text().splitlines())
            cgroup_v2 = usage_path.name == "memory.current"
            inactive = stat.get("inactive_file" if cgroup_v2 else "total_inactive_file", 0)
            usage = max(0, usage - int(inactive))
        limit_raw = limit_path.read_text().strip()
        if limit_raw == "max":
            limit = psutil.virtual_memory().total
        else:
            limit = int(limit_raw)
        # Handle unlimited v1 sentinel (> 1e18)
        if limit > 1e18:
            limit = psutil.virtual_memory().total

        return (usage / limit) * 100
    except (FileNotFoundError, OSError, ValueError, ZeroDivisionError):
        # Non-container or unsupported: fallback to host
        return psutil.virtual_memory().percent
