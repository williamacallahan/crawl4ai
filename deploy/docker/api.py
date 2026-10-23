import asyncio
import json
import logging
import sys
import time
from base64 import b64encode
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast
from urllib.parse import unquote
from uuid import uuid4

import psutil
from crawl_job_queue import (
    CrawlJobPayloadRejected,
    CrawlJobPrincipalQuotaExceeded,
    CrawlJobQueue,
    CrawlJobQueueFull,
)
from fastapi import HTTPException, Request, status
from fastapi.background import BackgroundTasks
from fastapi.responses import JSONResponse
from hook_registry import HookValidationError, build_declarative_hooks
from llm_broker import LLMProviderNotAllowed
from redis import asyncio as aioredis
from redis.exceptions import RedisError
from utils import (
    CrawlResourceSettings,
    FilterType,
    ResultEncoding,
    TaskStatus,
    decode_redis_hash,
    decode_task_result,
    ResultTooLarge,
    encode_task_result,
    get_base_url,
    get_browser_extra_args,
    get_redis_task_ttl,
    is_task_id,
    public_error_detail,
    correlated_error,
    public_crawl_error,
    validate_llm_provider,
    validate_url_destination,
)
from webhook import WebhookDeliveryService
from archive_transport import acquire_crawler, collect_results, grouped_results, prefetch_archive
from governor import wall_clock_seconds

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
    LLMExtractionStrategy,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from crawl4ai.async_configs import Provenance, UntrustedConfigError
from crawl4ai.content_filter_strategy import (
    BM25ContentFilter,
    PruningContentFilter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.prompts import PROMPT_FILTER_CONTENT
from crawl4ai.utils import (
    aperform_completion_with_backoff,
    escape_json_string,
    extract_xml_data,
    sanitize_html,
)

logger = logging.getLogger(__name__)
LLM_PERMIT_KEY = "crawl4ai:llm:permit:v1"
_RELEASE_LLM_PERMIT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
_RENEW_LLM_PERMIT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return 0
"""


class LlmCapacityBusy(HTTPException):
    """Expected shared-permit backpressure, not a server failure."""


@asynccontextmanager
async def llm_permit(redis, config):
    token = uuid4().hex
    ttl = config["llm"].get("permit_ttl_seconds", 300)
    # Start before the command: network latency must not extend our local
    # view of a lease Redis may already be counting down.
    confirmed_expiry = time.monotonic() + ttl
    acquired = await redis.set(
        LLM_PERMIT_KEY,
        token,
        nx=True,
        ex=ttl,
    )
    if not acquired:
        raise LlmCapacityBusy(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Crawl4AI LLM capacity is busy",
            headers={"Retry-After": "30"},
        )
    owner = asyncio.current_task()
    lease_lost = asyncio.Event()

    async def renew_permit():
        nonlocal confirmed_expiry
        safety_margin = min(1.0, ttl / 10)
        delay = ttl / 3
        try:
            while True:
                retry_deadline = confirmed_expiry - safety_margin
                await asyncio.sleep(min(delay, max(0, retry_deadline - time.monotonic())))
                if time.monotonic() >= retry_deadline:
                    # Leave room for the last in-flight renewal, but do not
                    # cancel protected work while the confirmed lease lives.
                    await asyncio.sleep(max(0, confirmed_expiry - time.monotonic()))
                    break
                renewal_started = time.monotonic()
                try:
                    renewed = await asyncio.wait_for(
                        redis.eval(_RENEW_LLM_PERMIT, 1, LLM_PERMIT_KEY, token, ttl),
                        timeout=confirmed_expiry - renewal_started,
                    )
                except (RedisError, asyncio.TimeoutError):
                    # An unacknowledged renewal does not invalidate the last
                    # confirmed lease. Keep trying within that lease only.
                    logger.warning("LLM permit renewal failed; retrying within lease", exc_info=True)
                    delay = min(1.0, ttl / 10)
                    continue
                if not renewed or time.monotonic() >= confirmed_expiry:
                    break
                confirmed_expiry = renewal_started + ttl
                delay = ttl / 3
        except Exception:
            logger.exception("LLM permit renewal failed unexpectedly")
        lease_lost.set()
        if owner is not None:
            owner.cancel()

    heartbeat = asyncio.create_task(renew_permit())
    try:
        if time.monotonic() >= confirmed_expiry:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Crawl4AI LLM capacity lease was lost",
            )
        yield
    except asyncio.CancelledError:
        if lease_lost.is_set():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Crawl4AI LLM capacity lease was lost",
            )
        raise
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        try:
            await redis.eval(_RELEASE_LLM_PERMIT, 1, LLM_PERMIT_KEY, token)
        except Exception:
            # Cleanup must never mask the result of the protected work; the
            # permit self-releases via TTL when the Lua compare-and-del fails.
            logger.exception("Failed to release LLM permit during cleanup")


async def _enqueue_job(background_tasks, factory, principal=None, on_cancel=None):
    """Submit a background job to the bounded work queue (per-principal quota).

    Falls back to FastAPI BackgroundTasks when the queue isn't running (tests /
    no lifespan). Maps queue/quota limits to HTTP 503 / 429.
    """
    from work_queue import QueueFull, QuotaExceeded, get_job_queue

    queue = get_job_queue()
    if queue is None or not queue.started:
        background_tasks.add_task(factory)
        return
    try:
        await queue.submit(factory, principal, on_cancel)
    except QuotaExceeded:
        raise HTTPException(status_code=429, detail="Too many concurrent jobs for this caller")
    except QueueFull:
        raise HTTPException(
            status_code=503,
            detail="Server busy, retry later",
            headers={"Retry-After": "5"},
        )


def _attach_declarative_hooks(crawler, hooks: dict) -> dict:
    """Attach the server-authored functions already validated for this request."""
    for hook_point, fn in hooks.items():
        crawler.crawler_strategy.set_hook(hook_point, fn)
    return {"status": "success", "attached": list(hooks.keys())}


async def _crawler_arun(crawler: AsyncWebCrawler, *args, **kwargs):
    """Call the instance-decorated arun installed by AsyncWebCrawler.__init__."""
    return await getattr(crawler, "arun")(*args, **kwargs)


def _apply_server_browser_policy(browser_config, config):
    """Apply server-owned launch flags and limits to caller browser config.

    Kept from the fork alongside upstream's egress_broker/governor guards, which
    cover SSRF and deep-crawl budgets but NOT Chromium's own memory growth. The
    proxy-safety helper this used to sit beside is gone: enforce_egress()
    supersedes it. Guarded by tests/test_resource_policy.py.
    """
    browser_defaults = config["crawler"]["browser"].get("kwargs", {})
    browser_config.extra_args = get_browser_extra_args(config)

    if browser_defaults.get("memory_saving_mode", False):
        browser_config.memory_saving_mode = True

    recycle_limit = browser_defaults.get("max_pages_before_recycle", 0)
    requested_recycle_limit = browser_config.max_pages_before_recycle
    if recycle_limit > 0 and (
        not isinstance(requested_recycle_limit, int)
        or isinstance(requested_recycle_limit, bool)
        or requested_recycle_limit <= 0
        or requested_recycle_limit > recycle_limit
    ):
        browser_config.max_pages_before_recycle = recycle_limit

    return browser_config


def apply_server_crawler_defaults(loaded_config, request_config, config):
    """Apply server defaults only when the wire request omitted the field."""
    request_config = request_config or {}
    request_params = (
        request_config.get("params", {})
        if request_config.get("type") == "CrawlerRunConfig"
        else request_config
    )
    if not isinstance(request_params, dict):
        request_params = {}
    for key, value in config["crawler"]["base_config"].items():
        if key not in request_params and hasattr(loaded_config, key):
            setattr(loaded_config, key, value)
    limits = CrawlResourceSettings.from_config(config)
    loaded_config.max_iframes = min(loaded_config.max_iframes, limits.max_iframes)
    loaded_config.page_termination_timeout = min(
        loaded_config.page_termination_timeout, limits.page_termination_timeout
    )
    return loaded_config


def server_crawler_config(config, **kwargs):
    """Construct a server-owned crawler config with canonical defaults."""
    return apply_server_crawler_defaults(CrawlerRunConfig(**kwargs), {}, config)


def _memory_dispatcher(config):
    """Build the shared dispatcher policy for both crawl response modes."""
    rate_config = config["crawler"]["rate_limiter"]
    return MemoryAdaptiveDispatcher(
        max_session_permit=config["crawler"]["pool"]["max_pages"],
        memory_threshold_percent=config["crawler"]["memory_threshold_percent"],
        recovery_threshold_percent=config["crawler"]["recovery_threshold_percent"],
        rate_limiter=(
            RateLimiter(base_delay=tuple(rate_config["base_delay"]))
            if rate_config["enabled"]
            else None
        ),
    )


def _project_crawl_result(result, result_fields):
    """Narrow a crawl result to the caller's requested fields.

    Kept from the fork: the durable CrawlJobQueue path honours `result_fields`,
    and upstream dropped this helper along with the fire-and-forget job runner
    it no longer needed. Still used by the streaming path below.
    """
    if not result_fields:
        return result
    return {field: result[field] for field in result_fields if field in result}


class CorrelatedCrawlFailure(HTTPException):
    """A redacted failed result that retains its private error-record id."""

    def __init__(self, detail: str, correlation_id: Optional[str]):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)
        self.correlation_id = correlation_id


def _raise_for_crawl_failure(result, *, correlate_error: bool = False):
    if not result.success:
        if correlate_error:
            raw_error = result.error_message
            detail = public_crawl_error(raw_error)
            raise CorrelatedCrawlFailure(detail, detail.correlation_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=public_error_detail(result.error_message),
        )


# --- Helper to get memory ---
def _get_memory_mb():
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return None


def _terminal_task_status(status: Any) -> bool:
    """True when a task write records completed or failed."""
    if isinstance(status, TaskStatus):
        status = status.value
    return status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value)


async def hset_with_ttl(redis, key: str, mapping: dict, config: dict):
    """Write a task hash, expiring it only once the write is terminal.

    In-flight writes leave the key with no TTL, matching the crawl-job queue
    until its completion script. ``volatile-lru`` evicts only keys that have a
    TTL, so a live task hash is not an eviction candidate. A ``completed`` or
    ``failed`` status applies ``redis.task_ttl_seconds``. A TTL of 0 still
    disables that expiry.

    Args:
        redis: Redis client instance
        key: Redis key (e.g., "task:abc123")
        mapping: Hash field-value mapping
        config: Application config containing redis.task_ttl_seconds
    """
    await redis.hset(key, mapping=mapping)
    if not _terminal_task_status(mapping.get("status")):
        return
    ttl = get_redis_task_ttl(config)
    if ttl > 0:
        await redis.expire(key, ttl)


# Redis TIME stamps and judges heartbeats, so clock skew between replica hosts
# cannot fail a live job.
_LLM_TASK_HEARTBEAT_SCRIPT = """
-- crawl4ai:llm-task-heartbeat
if redis.call('HGET', KEYS[1], 'status') ~= 'processing' then
  return 0
end
redis.call('HSET', KEYS[1], 'heartbeat_at', redis.call('TIME')[1])
return 1
"""

_FINISH_LLM_TASK_SCRIPT = """
-- crawl4ai:finish-llm-task
if redis.call('HGET', KEYS[1], 'status') ~= 'processing' then
  return 0
end
redis.call('HSET', KEYS[1], unpack(ARGV, 2))
if tonumber(ARGV[1]) > 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return 1
"""

# A hash with no heartbeat_at came from a release that never heartbeats; during
# a rolling deploy its worker may still be running on an old replica, so it is
# never judged lost.
_RECOVER_LLM_TASK_SCRIPT = """
-- crawl4ai:recover-llm-task
local recovered = 0
if redis.call('HGET', KEYS[1], 'status') == 'processing' then
  local last_seen = tonumber(redis.call('HGET', KEYS[1], 'heartbeat_at'))
  local now = tonumber(redis.call('TIME')[1])
  if last_seen and last_seen < now - tonumber(ARGV[1]) then
    redis.call('HSET', KEYS[1], 'status', 'failed',
      'error', 'worker lost before completion')
    if tonumber(ARGV[2]) > 0 then
      redis.call('EXPIRE', KEYS[1], ARGV[2])
    end
    recovered = 1
  end
end
return {recovered, redis.call('HGETALL', KEYS[1])}
"""


def llm_task_heartbeat_seconds(config: dict) -> int:
    interval = config.get("llm", {}).get("task_heartbeat_seconds", 30)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("llm.task_heartbeat_seconds must be a positive integer")
    return interval


def llm_orphan_scan_settings(config: dict) -> Tuple[int, int]:
    llm = config.get("llm", {})
    interval = llm.get("orphan_scan_seconds", 60)
    count = llm.get("orphan_scan_count", 100)
    for name, value in (("orphan_scan_seconds", interval), ("orphan_scan_count", count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"llm.{name} must be a positive integer")
    return interval, count


async def _recover_llm_task(redis, task_id: str, config: dict) -> dict:
    recovered, fields = await redis.eval(
        _RECOVER_LLM_TASK_SCRIPT, 1, f"task:{task_id}",
        3 * llm_task_heartbeat_seconds(config), get_redis_task_ttl(config),
    )
    task = decode_redis_hash(dict(zip(fields[::2], fields[1::2])))
    if recovered:
        # Only the atomic transition winner notifies, even when a status read
        # and several replicas' reapers race each other or worker completion.
        # Sent in the background: a poll or reaper page must not wait through
        # a slow receiver's whole retry budget.
        webhook_config = task.get("webhook_config")
        _notify_llm_failure_later(
            config, task_id, task["url"],
            json.loads(webhook_config) if webhook_config else None, task["error"],
        )
    return task


_LLM_NOTIFICATIONS: set[asyncio.Task] = set()


def _notify_llm_failure_later(config, task_id, url, webhook_config, error) -> None:
    notification = asyncio.create_task(WebhookDeliveryService(config).notify_job_completion(
        task_id=task_id, task_type="llm_extraction", status="failed",
        urls=[url], webhook_config=webhook_config, error=error,
    ))
    _LLM_NOTIFICATIONS.add(notification)
    notification.add_done_callback(_LLM_NOTIFICATIONS.discard)


async def scan_llm_orphans(redis, config: dict, cursor: int = 0) -> int:
    """Recover one SCAN page; the caller carries the cursor between passes."""
    _, count = llm_orphan_scan_settings(config)
    cursor, keys = await redis.scan(cursor=cursor, match="task:llm_*", count=count)
    for key in keys:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        try:
            # The script checks processing/heartbeat atomically and leaves
            # live, terminal, and pre-heartbeat-release hashes untouched.
            await _recover_llm_task(redis, key.removeprefix("task:"), config)
        except RedisError:
            raise
        except Exception:
            logger.exception("Failed to recover LLM task %s", key)
    return cursor


async def reap_llm_tasks(redis, config: dict) -> None:
    interval, _ = llm_orphan_scan_settings(config)
    cursor = 0
    while True:
        try:
            cursor = await scan_llm_orphans(redis, config, cursor)
        except RedisError:
            logger.warning("LLM orphan scan failed; retrying next interval", exc_info=True)
        await asyncio.sleep(interval)


async def _heartbeat_llm_task(redis, task_id: str, config: dict) -> None:
    interval = llm_task_heartbeat_seconds(config)
    # The non-expiring task hash cannot be evicted by volatile-lru.
    delay = interval
    while True:
        await asyncio.sleep(delay)
        try:
            if not await redis.eval(_LLM_TASK_HEARTBEAT_SCRIPT, 1, f"task:{task_id}"):
                return
            delay = interval
        except RedisError:
            # A Redis restart lasts seconds; ending the loop here would let a
            # later poll fail this live job once three intervals pass, and
            # waiting a full interval after Redis returns risks the same.
            logger.warning("LLM task %s heartbeat failed; retrying", task_id, exc_info=True)
            delay = 1.0


async def _stop_llm_heartbeat(heartbeat: asyncio.Task) -> None:
    heartbeat.cancel()
    with suppress(asyncio.CancelledError):
        await heartbeat


async def _finish_llm_task(redis, key: str, mapping: dict, config: dict) -> bool:
    fields = [item for pair in mapping.items() for item in pair]
    return bool(await redis.eval(
        _FINISH_LLM_TASK_SCRIPT, 1, key, get_redis_task_ttl(config), *fields,
    ))


async def handle_llm_qa(
    url: str,
    query: str,
    config: dict,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    *,
    redis=None,
) -> str:
    """Process QA using LLM with crawled content as context."""
    from crawler_pool import get_crawler, release_crawler
    crawler: Optional[AsyncWebCrawler] = None
    try:
        if not url.startswith(('http://', 'https://')) and not url.startswith(("raw:", "raw://")):
            url = 'https://' + url
        await asyncio.to_thread(validate_url_destination, url)

        # Get markdown content (use default config)
        from utils import load_config
        cfg = load_config()
        browser_cfg = BrowserConfig(
            extra_args=get_browser_extra_args(cfg),
            **cfg["crawler"]["browser"].get("kwargs", {}),
        )
        from egress_broker import enforce_egress
        enforce_egress(browser_cfg)
        crawler = await get_crawler(browser_cfg)
        crawler_config = server_crawler_config(cfg)
        result = await _crawler_arun(
            crawler,
            url=url,
            config=crawler_config,
        )
        _raise_for_crawl_failure(result)
        content = result.markdown.fit_markdown or result.markdown.raw_markdown

        # The crawler is no longer needed: only the materialised `content`
        # string feeds the LLM call below. Release the pooled browser and its
        # ADMISSION_SEM permit now, before the slow LLM roundtrip, so a
        # browser-pool slot is not held idle for the whole permit window.
        # `crawler = None` is load-bearing: release_crawler is not
        # idempotent for healthy pooled crawlers (it never sets
        # _docker_admission_released), so the finally block must not
        # release it a second time -- that would leak a semaphore permit
        # and over-report pool capacity.
        await release_crawler(crawler)
        crawler = None

        # Create prompt and get LLM response
        prompt = f"""Use the following content as context to answer the question.
    Content:
    {content}

    Question: {query}

    Answer:"""

        # Provider by name only; base_url/api_token are server-derived. A
        # request-supplied base_url is ignored (it was the key-exfil vector).
        from llm_broker import LLMWorkload, resolve_llm
        llm = resolve_llm(config, provider, workload=LLMWorkload.PAGE_QUESTION)
        if redis is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Crawl4AI LLM admission is unavailable",
            )
        effective_temperature = (
            temperature if temperature is not None else llm["temperature"]
        )
        qa_extra_args = dict(llm["extra_args"])
        if effective_temperature is not None:
            qa_extra_args["temperature"] = effective_temperature
        async with llm_permit(redis, config):
            response = await aperform_completion_with_backoff(
                provider=llm["provider"],
                prompt_with_variables=prompt,
                api_token=llm["api_token"],
                base_url=llm["base_url"],
                extra_args=qa_extra_args,
                base_delay=config["llm"].get("backoff_base_delay", 2),
                max_attempts=config["llm"].get("backoff_max_attempts", 1),
                exponential_factor=config["llm"].get("backoff_exponential_factor", 2)
            )

        return response.choices[0].message.content
    except HTTPException:
        raise
    except LLMProviderNotAllowed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"QA processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if crawler:
            await release_crawler(crawler)

class LlmExtractionRejected(RuntimeError):
    """Provider-reported extraction errors.

    Unlike everything else the catch-all below sees, this text comes from the
    LLM response, not from the server, so it is already client-facing and must
    reach the caller intact.
    """


async def process_llm_extraction(
    redis: aioredis.Redis,
    config: dict,
    task_id: str,
    url: str,
    instruction: str,
    schema: Optional[str] = None,
    cache: str = "0",
    provider: Optional[str] = None,
    webhook_config: Optional[Dict] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    heartbeat: Optional[asyncio.Task] = None,
) -> None:
    """Process LLM extraction in background."""
    heartbeat = heartbeat or asyncio.create_task(_heartbeat_llm_task(redis, task_id, config))
    try:
        webhook_service = WebhookDeliveryService(config)
        # Recovery may have won while this factory was waiting in the queue.
        if not await redis.eval(_LLM_TASK_HEARTBEAT_SCRIPT, 1, f"task:{task_id}"):
            return
        # Validate provider
        is_valid, error_msg = validate_llm_provider(config, provider)
        if not is_valid:
            if not await _finish_llm_task(redis, f"task:{task_id}", {
                "status": TaskStatus.FAILED,
                "error": error_msg
            }, config):
                return

            # Send webhook notification on failure
            await webhook_service.notify_job_completion(
                task_id=task_id,
                task_type="llm_extraction",
                status="failed",
                urls=[url],
                webhook_config=webhook_config,
                error=error_msg
            )
            return
        # Provider by name only; base_url/api_token server-derived (no exfil).
        from llm_broker import LLMWorkload, resolve_llm
        _llm = resolve_llm(config, provider, workload=LLMWorkload.JOB_EXTRACTION)
        job_extra_args = {
            **_llm["extra_args"],
            "timeout": config["llm"].get("job_request_timeout_seconds", 120),
            "num_retries": config["llm"].get("job_request_retries", 0),
            "max_tokens": config["llm"].get("job_max_output_tokens", 4096),
        }
        effective_temperature = (
            temperature if temperature is not None else _llm["temperature"]
        )
        if effective_temperature is not None:
            job_extra_args["temperature"] = effective_temperature
        llm_strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider=_llm["provider"],
                api_token=_llm["api_token"],
                temperature=effective_temperature,
                base_url=_llm["base_url"],
                backoff_max_attempts=config["llm"].get("backoff_max_attempts", 1),
            ),
            instruction=instruction,
            schema=cast(Dict, json.loads(schema) if schema else None),
            apply_chunking=False,
            extra_args=job_extra_args,
        )

        cache_mode = CacheMode.ENABLED if cache == "1" else CacheMode.WRITE_ONLY

        # Re-validate the destination at fetch time (the enqueue-time check is a
        # TOCTOU seed-only guard) and pin egress so the background fetch cannot
        # be rebound/redirected to an internal target.
        await asyncio.to_thread(validate_url_destination, url)
        from utils import load_config as _load_config
        _wcfg = await asyncio.to_thread(_load_config)
        worker_browser_cfg = BrowserConfig(
            extra_args=get_browser_extra_args(_wcfg),
            **_wcfg["crawler"]["browser"].get("kwargs", {}),
        )
        from egress_broker import enforce_egress
        enforce_egress(worker_browser_cfg)
        async with llm_permit(redis, config):
            async with AsyncWebCrawler(config=worker_browser_cfg) as crawler:
                crawler_config = server_crawler_config(
                    _wcfg,
                    extraction_strategy=llm_strategy,
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    cache_mode=cache_mode,
                )
                result = await crawler.arun(
                    url=url,
                    config=crawler_config,
                )

        if not result.success:
            error_message = public_crawl_error(result.error_message, url)
            if not await _finish_llm_task(redis, f"task:{task_id}", {
                "status": TaskStatus.FAILED,
                "error": error_message
            }, config):
                return

            # Send webhook notification on failure
            await webhook_service.notify_job_completion(
                task_id=task_id,
                task_type="llm_extraction",
                status="failed",
                urls=[url],
                webhook_config=webhook_config,
                error=error_message
            )
            return

        try:
            content = json.loads(result.extracted_content)
        except json.JSONDecodeError:
            content = result.extracted_content
        if isinstance(content, list):
            extraction_errors = [
                str(block.get("content") or "LLM extraction failed")
                for block in content
                if isinstance(block, dict) and block.get("error")
            ]
            if extraction_errors:
                raise LlmExtractionRejected("; ".join(extraction_errors))

        result_data = {"extracted_content": content}

        if not await _finish_llm_task(redis, f"task:{task_id}", {
            "status": TaskStatus.COMPLETED,
            "result": encode_task_result(content, ResultEncoding.IDENTITY)
        }, config):
            return

        # Send webhook notification on successful completion
        await webhook_service.notify_job_completion(
            task_id=task_id,
            task_type="llm_extraction",
            status="completed",
            urls=[url],
            webhook_config=webhook_config,
            result=result_data
        )

    except Exception as e:
        # This runs as a background task, so the central 500 handler never sees
        # it: str(e) would land verbatim in the task hash that /llm/job/{id}
        # returns and in the caller-supplied webhook.
        level = logging.WARNING if isinstance(e, LlmCapacityBusy) else logging.ERROR
        error_message = (
            str(e)
            if isinstance(e, LlmExtractionRejected)
            else correlated_error(
                "LLM extraction failed", e, f"llm extraction task={task_id}", level=level
            )
        )
        logger.log(level, "LLM extraction error: %s", e, exc_info=True)
        if not await _finish_llm_task(redis, f"task:{task_id}", {
            "status": TaskStatus.FAILED,
            "error": error_message
        }, config):
            return

        # Send webhook notification on failure
        await webhook_service.notify_job_completion(
            task_id=task_id,
            task_type="llm_extraction",
            status="failed",
            urls=[url],
            webhook_config=webhook_config,
            error=error_message
        )
    finally:
        await _stop_llm_heartbeat(heartbeat)

async def handle_markdown_request(
    redis,
    url: str,
    filter_type: FilterType,
    query: Optional[str] = None,
    cache: str = "0",
    config: Optional[dict] = None,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None
) -> str:
    """Handle markdown generation requests."""
    config = config or {}
    crawler: Optional[AsyncWebCrawler] = None
    llm = None
    try:
        # Validate provider if using LLM filter
        if filter_type == FilterType.LLM:
            is_valid, error_msg = validate_llm_provider(config, provider)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
            from llm_broker import LLMWorkload, resolve_llm

            llm = resolve_llm(config, provider, workload=LLMWorkload.MARKDOWN_FILTER)
        decoded_url = unquote(url)
        if not decoded_url.startswith(('http://', 'https://')) and not decoded_url.startswith(("raw:", "raw://")):
            decoded_url = 'https://' + decoded_url
        await asyncio.to_thread(validate_url_destination, decoded_url)

        if filter_type in {FilterType.RAW, FilterType.LLM}:
            md_generator = DefaultMarkdownGenerator()
        else:
            content_filter = {
                FilterType.FIT: PruningContentFilter(),
                FilterType.BM25: BM25ContentFilter(user_query=query or ""),
            }[filter_type]
            md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

        cache_mode = CacheMode.ENABLED if cache == "1" else CacheMode.WRITE_ONLY

        from utils import load_config as _load_config
        _cfg = config or _load_config()
        browser_cfg = BrowserConfig(
            extra_args=get_browser_extra_args(_cfg),
            **_cfg["crawler"]["browser"].get("kwargs", {}),
        )
        from egress_broker import enforce_egress
        enforce_egress(browser_cfg)
        crawler_config = server_crawler_config(
            _cfg,
            markdown_generator=md_generator,
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=cache_mode,
        )
        # Fetch archive URLs through the node budget before taking a pooled
        # browser: the crawl runs on the raw body, so no browser touches
        # archive.org. A spent budget raises ArchiveBudgetSpent (429) here.
        seconds = wall_clock_seconds(_cfg)
        deadline = asyncio.get_running_loop().time() + seconds if seconds > 0 else None
        responses = await _await_before_deadline(
            prefetch_archive(redis, _cfg, [decoded_url]), deadline
        )
        crawler = await _await_before_deadline(acquire_crawler(browser_cfg, responses), deadline)
        result = await _await_before_deadline(_crawler_arun(
            crawler,
            url=decoded_url,
            config=crawler_config,
        ), deadline)

        _raise_for_crawl_failure(result, correlate_error=True)

        if filter_type == FilterType.LLM:
            prompt = PROMPT_FILTER_CONTENT.replace(
                "{HTML}",
                escape_json_string(sanitize_html(result.cleaned_html)),
            ).replace(
                "{REQUEST}",
                query or "Extract main content",
            )
            # The crawler is no longer needed: only the materialised `prompt`
            # string feeds the LLM call below. Release the pooled browser and
            # its ADMISSION_SEM permit now, before the slow LLM roundtrip, so
            # a browser-pool slot is not held idle for the whole permit window.
            # `crawler = None` is load-bearing: release_crawler is not
            # idempotent for healthy pooled crawlers (it never sets
            # _docker_admission_released), so the finally block must not
            # release it a second time -- that would leak a semaphore permit
            # and over-report pool capacity.
            await _dispose_crawler(crawler)
            crawler = None
            effective_temperature = (
                temperature if temperature is not None else llm["temperature"]
            )
            md_extra_args = dict(llm["extra_args"])
            if effective_temperature is not None:
                md_extra_args["temperature"] = effective_temperature
            async with llm_permit(redis, config):
                response = await aperform_completion_with_backoff(
                    provider=llm["provider"],
                    prompt_with_variables=prompt,
                    api_token=llm["api_token"],
                    base_url=llm["base_url"],
                    extra_args=md_extra_args,
                    max_attempts=1,
                )
            markdown = extract_xml_data(
                ["content"],
                response.choices[0].message.content or "",
            )["content"]
            if not markdown:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="LLM markdown filter returned no content",
                )
            return markdown

        return (result.markdown.raw_markdown
                if filter_type == FilterType.RAW
                else result.markdown.fit_markdown)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Crawl exceeded the time limit")
    except HTTPException:
        raise
    except LLMProviderNotAllowed as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Markdown error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if crawler:
            await _dispose_crawler(crawler)

async def handle_llm_request(
    redis: aioredis.Redis,
    background_tasks: BackgroundTasks,
    request: Request,
    input_path: str,
    query: Optional[str] = None,
    schema: Optional[str] = None,
    cache: str = "0",
    config: Optional[dict] = None,
    provider: Optional[str] = None,
    webhook_config: Optional[Dict] = None,
    temperature: Optional[float] = None,
    api_base_url: Optional[str] = None,
    requester: Optional[str] = None,
    is_admin: bool = False,
) -> JSONResponse:
    """Handle LLM extraction requests."""
    config = config or {}
    base_url = get_base_url(request)

    try:
        if is_task_id(input_path):
            return await handle_task_status(
                redis, input_path, base_url,
                collection="llm/job",
                requester=requester, is_admin=is_admin,
                config=config,
            )

        if not query:
            return JSONResponse({
                "message": "Please provide an instruction",
                "_links": {
                    "example": {
                        "href": f"{base_url}/llm/{input_path}?q=Extract+main+content",
                        "title": "Try this example"
                    }
                }
            })

        return await create_new_task(
            redis,
            background_tasks,
            input_path,
            query,
            schema,
            cache,
            base_url,
            config,
            provider,
            webhook_config,
            temperature,
            api_base_url,
            owner=requester,
        )

    except HTTPException:
        raise  # 429/503 (queue/quota), 400, etc. - don't mask as 500
    except Exception:
        logger.exception("LLM endpoint failed")
        raise

async def handle_task_status(
    redis: aioredis.Redis,
    task_id: str,
    base_url: str,
    *,
    collection: str,
    keep: bool = False,
    requester: Optional[str] = None,
    is_admin: bool = False,
    config: Optional[dict] = None,
) -> JSONResponse:
    """Handle task status check requests.

    Enforces ownership: a task records the `owner` (principal sub) that created
    it; a different requester gets 404 (not 403, so task existence is not
    revealed). Admin-scope principals may read any task.

    Terminal reads preserve expiry. Recovering an orphan is a terminal write,
    so it arms the same retention as worker completion. ``keep`` never changes
    retention.
    """
    task = await redis.hgetall(f"task:{task_id}")
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    task = decode_redis_hash(cast(Dict[bytes, bytes], task))

    owner = task.get("owner")
    if owner and not is_admin and owner != requester:
        # Do not leak existence of someone else's task.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Only LLM tasks heartbeat; a crawl job read through this route keeps its
    # own lease-based recovery.
    config = config or {}
    if (
        collection == "llm/job" and task_id.startswith("llm_")
        and task["status"] == TaskStatus.PROCESSING and "heartbeat_at" in task
    ):
        # Redis TIME is the sole clock for both stamps and recovery verdicts.
        task = await _recover_llm_task(redis, task_id, config)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

    response = create_task_response(task, task_id, base_url, collection)

    return JSONResponse(response)

async def create_new_task(
    redis: aioredis.Redis,
    background_tasks: BackgroundTasks,
    input_path: str,
    query: str,
    schema: Optional[str],
    cache: str,
    base_url: str,
    config: dict,
    provider: Optional[str] = None,
    webhook_config: Optional[Dict] = None,
    temperature: Optional[float] = None,
    api_base_url: Optional[str] = None,
    owner: Optional[str] = None,
) -> JSONResponse:
    """Create and initialize a new task."""
    decoded_url = unquote(input_path)
    if not decoded_url.startswith(('http://', 'https://')) and not decoded_url.startswith(("raw:", "raw://")):
        decoded_url = 'https://' + decoded_url
    await asyncio.to_thread(validate_url_destination, decoded_url)

    llm_task_heartbeat_seconds(config)
    task_id = f"llm_{uuid4().hex}"

    task_data = {
        "status": TaskStatus.PROCESSING,
        "created_at": datetime.now().isoformat(),
        "url": decoded_url
    }
    if owner:
        task_data["owner"] = owner

    # Store webhook config if provided
    if webhook_config:
        task_data["webhook_config"] = json.dumps(webhook_config)

    # heartbeat_at marks this hash recoverable from accept, so work waiting in
    # the local queue is covered. Use the same clock as renewal and recovery.
    task_data["heartbeat_at"] = (await redis.time())[0]
    await hset_with_ttl(redis, f"task:{task_id}", task_data, config)
    heartbeat = asyncio.create_task(_heartbeat_llm_task(redis, task_id, config))

    async def cancel_task() -> None:
        error = "LLM extraction interrupted by server shutdown"
        try:
            if await _finish_llm_task(redis, f"task:{task_id}", {
                "status": TaskStatus.FAILED, "error": error,
            }, config):
                # The reaper skips terminal tasks, so this is the only notice
                # a webhook-only client gets; bounded so shutdown cannot hang.
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        WebhookDeliveryService(config).notify_job_completion(
                            task_id=task_id, task_type="llm_extraction", status="failed",
                            urls=[decoded_url], webhook_config=webhook_config, error=error,
                        ),
                        timeout=10,
                    )
        finally:
            await _stop_llm_heartbeat(heartbeat)

    try:
        await _enqueue_job(
            background_tasks,
            lambda: process_llm_extraction(
                redis, config, task_id, decoded_url, query, schema, cache,
                provider, webhook_config, temperature, api_base_url,
                heartbeat=heartbeat,
            ),
            principal=owner,
            on_cancel=cancel_task,
        )
    except BaseException:
        # Admission failures have no worker to own the heartbeat or hash.
        try:
            await redis.delete(f"task:{task_id}")
        finally:
            await _stop_llm_heartbeat(heartbeat)
        raise

    task_url = f"{base_url.rstrip('/')}/llm/job/{task_id}"
    return JSONResponse({
        "task_id": task_id,
        "status": TaskStatus.PROCESSING,
        "url": decoded_url,
        "_links": {
            "self": {"href": task_url},
            "status": {"href": task_url}
        }
    }, status_code=status.HTTP_202_ACCEPTED)

def create_task_response(task: dict, task_id: str, base_url: str, collection: str) -> dict:
    """Create response for task status check."""
    task_url = f"{base_url.rstrip('/')}/{collection}/{task_id}"
    response = {
        "task_id": task_id,
        "status": task["status"],
        "created_at": task["created_at"],
        "url": task["url"],
        "_links": {
            "self": {"href": task_url},
            "refresh": {"href": task_url}
        }
    }

    # A crawl job whose URLs all failed is terminal-failed but still carries
    # its per-URL diagnostics, so result and error are not exclusive.
    try:
        result = decode_task_result(task, task_id)
    except ResultTooLarge:
        # Stored before results were bounded: poll it as a terminal failure
        # instead of inflating it past the read budget into a 500.
        response["status"] = TaskStatus.FAILED.value
        response["error"] = "result_too_large"
        return response
    if task.get("result"):
        response["result"] = result
    if task["status"] == TaskStatus.FAILED:
        response["error"] = task["error"]

    return response


async def _new_hook_crawler(browser_config) -> AsyncWebCrawler:
    """Create an isolated crawler for request-local declarative hooks.

    A pooled crawler can serve concurrent requests, so mutating its strategy hook
    map cannot be made request-local by snapshot/restore. Hook-bearing requests
    use a dedicated crawler; ordinary requests keep the shared pool throughput.
    """
    from crawler_pool import get_dedicated_crawler

    return await get_dedicated_crawler(browser_config)


async def _dispose_crawler(crawler: Optional[AsyncWebCrawler]) -> None:
    if crawler is None:
        return
    if getattr(crawler, "_docker_admission_released", False):
        return
    if getattr(crawler, "_docker_request_owned", False):
        from crawler_pool import release_dedicated_crawler

        await release_dedicated_crawler(crawler)
        return
    from crawler_pool import release_crawler

    await release_crawler(crawler)


async def _stream_before_deadline(
    results_gen: AsyncGenerator,
    deadline_at: float,
) -> AsyncGenerator:
    """Bound the full generator lifetime using a deadline set before setup."""
    iterator = aiter(results_gen)
    try:
        while True:
            remaining = deadline_at - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                result = await asyncio.wait_for(anext(iterator), timeout=remaining)
            except StopAsyncIteration:
                return
            yield result
    finally:
        close = getattr(iterator, "aclose", None)
        if close:
            with suppress(Exception):
                await close()


async def _await_before_deadline(awaitable, deadline_at: Optional[float]):
    if deadline_at is None:
        return await awaitable
    remaining = deadline_at - asyncio.get_running_loop().time()
    if remaining <= 0:
        close = getattr(awaitable, "close", None)
        if close:
            close()
        raise asyncio.TimeoutError
    return await asyncio.wait_for(awaitable, timeout=remaining)


async def _track_request_start(endpoint: str, urls: List[str], browser_config: dict) -> str:
    """Register a request on the monitor; the monitor is never critical."""
    request_id = f"req_{uuid4().hex[:8]}"
    try:
        from monitor import get_monitor
        await get_monitor().track_request_start(
            request_id, endpoint, urls[0] if urls else "batch", browser_config
        )
    except Exception:
        pass
    return request_id


async def _close_aborted_request(request_id: str, pending: Optional[BaseException]) -> None:
    """Close a still-active monitor entry from the exception that aborted it.

    Every exit that skipped an explicit track_request_end — SSRF 400s,
    request-validation 400s, deadline 504s, client cancellation — would
    otherwise leave the entry "active" forever (issue #2). Deliberate-status
    HTTPException details are already client-safe; anything else gets a type
    name only, because /monitor/* returns this text to any data-scope principal.
    """
    try:
        from monitor import get_monitor
        monitor = get_monitor()
        # The still-active guard is load-bearing, not redundant with the one
        # inside track_request_end: when the caller already recorded (e.g. the
        # correlated, sanitized 500), this must never overwrite it with the
        # in-flight exception's detail — the generic 500 carries raw str(e).
        if request_id not in monitor.active_requests:
            return
        if isinstance(pending, HTTPException) and pending.status_code != 500:
            aborted_status = pending.status_code
            aborted_error = str(pending.detail)[:500]
        else:
            aborted_status = 500
            aborted_error = (
                f"request aborted: {type(pending).__name__}" if pending else "request aborted"
            )
        await monitor.track_request_end(
            request_id, success=False, error=aborted_error, status_code=aborted_status
        )
    except BaseException:
        # Monitor not critical — and nothing here may shadow the caller's
        # crawler disposal, cancellation included.
        pass


async def stream_results(
    crawler: AsyncWebCrawler,
    results_gen: AsyncGenerator,
    request_id: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """Stream results with heartbeats and completion markers.

    The HTTP status is already 200 once bytes flow, so the monitor record
    carries the crawl verdict: success when any URL succeeded and the stream
    reached its completion marker (issue #78).
    """
    from utils import datetime_handler
    any_url_succeeded = False
    first_error: Optional[str] = None
    # Cleared only when the completion marker is produced: a consumer that
    # closes the generator early (client gone while parked on a write) must
    # not be recorded as a success.
    terminal_error: Optional[str] = "Stream closed before completion"
    try:
        async for result in results_gen:
            try:
                server_memory_mb = _get_memory_mb()
                result_dict = await asyncio.to_thread(result.model_dump)
                if result_dict.get("error_message"):
                    result_dict["error_message"] = public_crawl_error(
                        result_dict["error_message"], result_dict.get("url")
                    )
                if result_dict.get("success"):
                    any_url_succeeded = True
                elif first_error is None:
                    first_error = result_dict.get("error_message") or "Crawl failed"
                result_dict['server_memory_mb'] = server_memory_mb
                if "fit_html" in result_dict and not (result_dict["fit_html"] is None or isinstance(result_dict["fit_html"], str)):
                    result_dict["fit_html"] = None
                if result_dict.get('pdf') is not None:
                    result_dict['pdf'] = b64encode(result_dict['pdf']).decode('utf-8')
                logger.info(f"Streaming result for {result_dict.get('url', 'unknown')}")
                data = await asyncio.to_thread(
                    lambda: (json.dumps(result_dict, default=datetime_handler) + "\n").encode("utf-8")
                )
                yield data
            except Exception as e:
                streamed_url = getattr(result, 'url', 'unknown')
                error_response = {
                    "error": correlated_error("Result serialization failed", e, f"stream url={streamed_url}"),
                    "url": streamed_url,
                }
                yield (json.dumps(error_response) + "\n").encode('utf-8')

        terminal_error = None
        yield json.dumps({"status": "completed"}).encode('utf-8')

    except asyncio.TimeoutError:
        logger.warning("Streaming crawl exceeded its wall-clock deadline")
        terminal_error = "Crawl exceeded the time limit"
        yield (json.dumps({"status": "failed", "error": terminal_error}) + "\n").encode("utf-8")
    except Exception:
        logger.exception("Streaming crawl failed")
        terminal_error = "Streaming crawl failed"
        yield (json.dumps({"status": "failed", "error": terminal_error}) + "\n").encode("utf-8")
    except asyncio.CancelledError:
        logger.warning("Client disconnected during streaming")
        terminal_error = "Client disconnected during streaming"
    finally:
        if request_id is not None:
            try:
                from monitor import get_monitor
                await get_monitor().track_request_end(
                    request_id,
                    success=any_url_succeeded and terminal_error is None,
                    error=terminal_error or (None if any_url_succeeded else first_error or "Crawl failed"),
                    status_code=200,
                )
            except BaseException:
                pass  # Monitor not critical; never shadow the disposal below.
        try:
            close = getattr(results_gen, "aclose", None)
            if close:
                await close()
        finally:
            await _dispose_crawler(crawler)


async def _normalize_and_validate_seeds(urls: List[str]) -> List[str]:
    """Prefix bare hosts with https:// and SSRF-validate every seed URL's
    destination. Shared by the streaming and non-streaming crawl handlers so a
    new entry point cannot silently skip the destination check."""
    urls = [('https://' + url) if not url.startswith(('http://', 'https://')) and not url.startswith(("raw:", "raw://")) else url for url in urls]
    for url in urls:
        await asyncio.to_thread(validate_url_destination, url)
    return urls


async def handle_crawl_request(
    urls: List[str],
    browser_config: dict,
    crawler_config: dict,
    config: dict,
    hooks_config: Optional[dict] = None,
    crawler_configs: Optional[List[dict]] = None,
    result_fields: Optional[List[str]] = None,
    redis=None,
) -> dict:
    """Handle non-streaming crawl requests with optional hooks."""
    deadline_seconds = wall_clock_seconds(config)
    deadline_at = (
        asyncio.get_running_loop().time() + deadline_seconds
        if deadline_seconds > 0
        else None
    )
    request_id = await _track_request_start("/crawl", urls, browser_config)
    crawler: Optional[AsyncWebCrawler] = None

    start_mem_mb = _get_memory_mb() # <--- Get memory before
    start_time = time.time()
    mem_delta_mb = None
    peak_mem_mb = start_mem_mb

    try:
        urls = await _normalize_and_validate_seeds(urls)
        loaded_browser_config = BrowserConfig.load(
            browser_config,
            provenance=Provenance.UNTRUSTED,
        )
        _apply_server_browser_policy(loaded_browser_config, config)
        loaded_crawler_config = CrawlerRunConfig.load(
            crawler_config,
            provenance=Provenance.UNTRUSTED,
        )
        from egress_broker import enforce_egress
        enforce_egress(loaded_browser_config)
        from governor import clamp_deep_crawl
        clamp_deep_crawl(loaded_crawler_config)

        # Every request input must validate before a reservation is spent:
        # per-URL crawler overrides and declarative hook specs both reject
        # with 400, so they are loaded ahead of the archive materialization.
        if crawler_configs and len(urls) > 1:
            # Per-URL config list: deserialize each and apply base_config
            config_list = [CrawlerRunConfig.load(cc, provenance=Provenance.UNTRUSTED) for cc in crawler_configs]
            for cfg, request_config in zip(config_list, crawler_configs):
                apply_server_crawler_defaults(cfg, request_config, config)
            effective_config = config_list
        else:
            # Single config (original behavior)
            effective_config = apply_server_crawler_defaults(
                loaded_crawler_config,
                crawler_config,
                config,
            )

        # Validate hooks before spending a reservation or acquiring a browser.
        hooks = build_declarative_hooks(
            hooks_config.get("hooks", []) or []
        ) if hooks_config else {}
        responses = await _await_before_deadline(
            prefetch_archive(redis, config, urls), deadline_at
        )

        from crawler_pool import get_crawler

        hooks_status = {}
        if responses:
            if hooks_config:
                hooks_status = {"status": "success", "attached": list(hooks)}
            results = await _await_before_deadline(collect_results(grouped_results(
                urls, loaded_browser_config, effective_config, responses, hooks,
                lambda: _memory_dispatcher(config),
            )), deadline_at)
        else:
            if hooks_config:
                crawler = await _await_before_deadline(
                    _new_hook_crawler(loaded_browser_config), deadline_at
                )
                hooks_status = _attach_declarative_hooks(crawler, hooks)
                logger.info(f"Hooks attachment status: {hooks_status['status']}")
            else:
                crawler = await _await_before_deadline(
                    get_crawler(loaded_browser_config), deadline_at
                )
            dispatcher = _memory_dispatcher(config)
            func = getattr(crawler, "arun" if len(urls) == 1 else "arun_many")
            partial_func = partial(func, urls[0] if len(urls) == 1 else urls,
                                   config=effective_config, dispatcher=dispatcher)
            results = await _await_before_deadline(partial_func(), deadline_at)
        
        # Ensure results is always a list
        if not isinstance(results, list):
            results = [results]

        end_mem_mb = _get_memory_mb() # <--- Get memory after
        end_time = time.time()
        
        if start_mem_mb is not None and end_mem_mb is not None:
            mem_delta_mb = end_mem_mb - start_mem_mb # <--- Calculate delta
            peak_mem_mb = max(peak_mem_mb if peak_mem_mb else 0, end_mem_mb) # <--- Get peak memory
        logger.info(f"Memory usage: Start: {start_mem_mb} MB, End: {end_mem_mb} MB, Delta: {mem_delta_mb} MB, Peak: {peak_mem_mb} MB")

        # Process results to handle PDF bytes
        processed_results = []
        any_url_succeeded = False
        for result in results:
            try:
                result_dict: Dict[str, Any]
                # Check if result has model_dump method (is a proper CrawlResult)
                if hasattr(result, 'model_dump'):
                    result_dict = await asyncio.to_thread(result.model_dump)
                elif isinstance(result, dict):
                    result_dict = dict(result)
                else:
                    # Handle unexpected result type. Neither the class name nor
                    # the object's repr (which carries its address) belongs in a
                    # client body; both stay in the log behind the id below.
                    result_dict = {
                        "url": "unknown",
                        "success": False,
                        "error_message": correlated_error(
                            "Crawl failed", f"unexpected result type: {type(result)!r}"
                        ),
                    }

                # if fit_html is not a string, set it to None to avoid serialization errors
                if "fit_html" in result_dict and not (result_dict["fit_html"] is None or isinstance(result_dict["fit_html"], str)):
                    result_dict["fit_html"] = None

                # If PDF exists, encode it to base64
                pdf = result_dict.get('pdf')
                if isinstance(pdf, bytes):
                    result_dict['pdf'] = b64encode(pdf).decode('utf-8')

                # Sanitize before projection so a caller-requested
                # error_message field carries the client-safe text.
                if result_dict.get("error_message"):
                    result_dict["error_message"] = public_crawl_error(
                        result_dict["error_message"], result_dict.get("url")
                    )
                # Read the per-URL verdict before projection: result_fields may
                # omit "success", and the caller-visible aggregate below must
                # not depend on what the caller asked to see.
                any_url_succeeded = any_url_succeeded or bool(result_dict.get("success"))

                if result_fields:
                    result_dict = _project_crawl_result(result_dict, result_fields)

                processed_results.append(result_dict)
            except Exception as e:
                processed_results.append({
                    "url": "unknown",
                    "success": False,
                    "error_message": correlated_error("Crawl failed", f"result serialization failed: {e}"),
                })
            
        response = {
            # Whether the crawl produced anything usable. Callers that project
            # result fields away still get an authoritative verdict here, and
            # /crawl/job stores it as the job's terminal state.
            "success": any_url_succeeded,
            "results": processed_results,
            "server_processing_time_s": end_time - start_time,
            "server_memory_delta_mb": mem_delta_mb,
            "server_peak_memory_mb": peak_mem_mb
        }

        # Track request completion. The handler owns the verdict: an
        # all-failed crawl is the caller's 502 (server.py /crawl) or a failed
        # job (crawl_job_worker), so recording it as success/200 here counted
        # every upstream failure as a success in /monitor (issue #18).
        try:
            from monitor import get_monitor
            await get_monitor().track_request_end(
                request_id,
                success=any_url_succeeded,
                error=None if any_url_succeeded else (
                    (processed_results[0].get("error_message") if processed_results else None)
                    or "Crawl failed"
                ),
                pool_hit=True,
                status_code=200 if any_url_succeeded else status.HTTP_502_BAD_GATEWAY,
            )
        except Exception:
            pass

        # Add hooks information if hooks were used
        if hooks_config:
            response["hooks"] = hooks_status

        return response

    except (UntrustedConfigError, HookValidationError) as e:
        # An untrusted request body tried to set a forbidden power-field,
        # construct a disallowed type, or specify an invalid hook. Client
        # error; the finally below records it on the monitor.
        raise HTTPException(status_code=400, detail=f"Rejected request: {e}")

    except asyncio.TimeoutError:
        # Per-crawl wall-clock deadline exceeded.
        raise HTTPException(status_code=504, detail="Crawl exceeded the time limit")

    except HTTPException:
        # Deliberate status (e.g. 400 SSRF "URL blocked") must pass through
        # rather than be genericized to 500 by the handler below.
        raise

    except Exception as e:
        logger.error(f"Crawl error: {str(e)}", exc_info=True)

        # /monitor/requests, /monitor/logs/errors and /monitor/ws return this
        # verbatim to any data-scope principal, so it cannot be str(e).
        monitor_error = correlated_error("Crawl failed", e, f"crawl request={request_id}")

        # Track request error
        try:
            from monitor import get_monitor
            await get_monitor().track_request_end(
                request_id, success=False, error=monitor_error, status_code=500
            )
        except Exception:
            pass

        # Measure memory even on error if possible
        end_mem_mb_error = _get_memory_mb()
        if start_mem_mb is not None and end_mem_mb_error is not None:
            mem_delta_mb = end_mem_mb_error - start_mem_mb

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=json.dumps({ # Send structured error
                "error": str(e),
                "server_memory_delta_mb": mem_delta_mb,
                "server_peak_memory_mb": max(peak_mem_mb if peak_mem_mb else 0, end_mem_mb_error or 0)
            })
        )
    finally:
        await _close_aborted_request(request_id, sys.exc_info()[1])
        await _dispose_crawler(crawler)

async def handle_stream_crawl_request(
    urls: List[str],
    browser_config: dict,
    crawler_config: dict,
    config: dict,
    hooks_config: Optional[dict] = None,
    crawler_configs: Optional[List[dict]] = None,
    redis=None,
) -> Tuple[AsyncWebCrawler, AsyncGenerator, Optional[Dict], str]:
    """Handle streaming crawl requests with optional hooks."""
    hooks_info = None
    crawler: Optional[AsyncWebCrawler] = None

    deadline_seconds = wall_clock_seconds(config)
    deadline_at = (
        asyncio.get_running_loop().time() + deadline_seconds
        if deadline_seconds > 0
        else None
    )
    # The entry stays active until stream_results closes it (issue #78); only
    # a setup failure below closes it here.
    request_id = await _track_request_start("/crawl/stream", urls, browser_config)
    try:
        # SSRF guard: validate every seed URL's destination before fetching,
        # mirroring handle_crawl_request. The streaming path previously skipped
        # this, leaving /crawl/stream (and /crawl with stream=true) unguarded.
        urls = await _normalize_and_validate_seeds(urls)
        loaded_browser_config = BrowserConfig.load(
            browser_config,
            provenance=Provenance.UNTRUSTED,
        )
        _apply_server_browser_policy(loaded_browser_config, config)
        # browser_config.verbose = True # Set to False or remove for production stress testing
        loaded_browser_config.verbose = False
        from egress_broker import enforce_egress
        enforce_egress(loaded_browser_config)
        loaded_crawler_config = CrawlerRunConfig.load(
            crawler_config,
            provenance=Provenance.UNTRUSTED,
        )
        apply_server_crawler_defaults(
            loaded_crawler_config,
            crawler_config,
            config,
        )
        from governor import clamp_deep_crawl
        clamp_deep_crawl(loaded_crawler_config)
        loaded_crawler_config.scraping_strategy = LXMLWebScrapingStrategy()
        loaded_crawler_config.stream = True

        # Deep crawl streaming supports exactly one start URL
        if loaded_crawler_config.deep_crawl_strategy is not None and len(urls) != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Deep crawling with stream currently supports exactly one URL per request. "
                    f"Received {len(urls)} URLs."
                ),
            )

        # Every request input must validate before a reservation is spent:
        # per-URL crawler overrides and declarative hook specs both reject
        # with 400, so they are loaded ahead of the archive materialization.
        if (
            crawler_configs
            and len(urls) > 1
            and loaded_crawler_config.deep_crawl_strategy is None
        ):
            # Match the shared streaming config's policy for every URL.
            effective_config = [
                CrawlerRunConfig.load(cc, provenance=Provenance.UNTRUSTED)
                for cc in crawler_configs
            ]
            for cfg, request_config in zip(effective_config, crawler_configs):
                apply_server_crawler_defaults(cfg, request_config, config)
                clamp_deep_crawl(cfg)
                cfg.scraping_strategy = LXMLWebScrapingStrategy()
                cfg.stream = True
        else:
            effective_config = loaded_crawler_config

        # Build hooks and prefetch under the same request deadline, before
        # either archive or ordinary dispatch acquires a browser.
        hooks = build_declarative_hooks(
            hooks_config.get("hooks", []) or []
        ) if hooks_config else {}
        responses = await _await_before_deadline(
            prefetch_archive(redis, config, urls), deadline_at
        )
        if responses:
            if hooks_config:
                hooks_info = {"status": {"status": "success", "attached": list(hooks)}}
            results_gen = grouped_results(
                urls, loaded_browser_config, effective_config, responses, hooks,
                lambda: _memory_dispatcher(config),
            )
            if deadline_at is not None:
                results_gen = _stream_before_deadline(results_gen, deadline_at)
            return None, results_gen, hooks_info, request_id

        from crawler_pool import get_crawler

        if hooks_config:
            crawler = await _await_before_deadline(
                _new_hook_crawler(loaded_browser_config),
                deadline_at,
            )
            hooks_status = _attach_declarative_hooks(crawler, hooks)
            logger.info(f"Hooks attachment status for streaming: {hooks_status['status']}")
            hooks_info = {'status': hooks_status}
        else:
            crawler = await _await_before_deadline(
                get_crawler(loaded_browser_config),
                deadline_at,
            )


        # Deep crawl with single URL: use arun() which returns an async generator
        # mirroring the Python library's streaming behavior
        if loaded_crawler_config.deep_crawl_strategy is not None and len(urls) == 1:
            results_gen = await _await_before_deadline(
                crawler.arun(
                    urls[0],
                    config=loaded_crawler_config,
                ),
                deadline_at,
            )
        else:
            # Default multi-URL streaming via arun_many
            results_gen = await _await_before_deadline(
                crawler.arun_many(
                    urls=urls,
                    config=effective_config,
                    dispatcher=_memory_dispatcher(config),
                ),
                deadline_at,
            )

        if deadline_at is not None:
            results_gen = _stream_before_deadline(results_gen, deadline_at)
        return crawler, results_gen, hooks_info, request_id

    except (UntrustedConfigError, HookValidationError) as e:
        await _dispose_crawler(crawler)
        raise HTTPException(status_code=400, detail=f"Rejected request: {e}")

    except asyncio.TimeoutError:
        await _dispose_crawler(crawler)
        raise HTTPException(status_code=504, detail="Crawl exceeded the time limit")

    except HTTPException:
        # Deliberate status (e.g. 400 SSRF "URL blocked") must pass through
        # rather than be genericized to 500 by the handler below.
        await _dispose_crawler(crawler)
        raise

    except Exception as e:
        # Release crawler on setup error (for successful streams,
        # release happens in stream_results finally block)
        await _dispose_crawler(crawler)
        logger.error(f"Stream crawl error: {str(e)}", exc_info=True)
        # Raising HTTPException here will prevent streaming response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        pending = sys.exc_info()[1]
        if pending is not None:
            await _close_aborted_request(request_id, pending)
        
async def handle_crawl_job(
    redis,
    urls: List[str],
    browser_config: Dict,
    crawler_config: Dict,
    config: Dict,
    result_fields: Optional[List[str]] = None,
    webhook_config: Optional[Dict] = None,
    owner: Optional[str] = None,
) -> Dict:
    """Persist a crawl request for the supervised Redis Stream workers.

    The durable queue owns both backlog capacity and per-principal fairness; the
    owner is persisted with the task so polling authorization remains valid after
    the API process or worker restarts.
    """
    try:
        task_id = await CrawlJobQueue(redis, config).enqueue(
            urls=urls,
            browser_config=browser_config,
            crawler_config=crawler_config,
            result_fields=result_fields,
            webhook_config=webhook_config,
            owner=owner,
        )
    except CrawlJobPayloadRejected as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except CrawlJobPrincipalQuotaExceeded as error:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(error)) from error
    except CrawlJobQueueFull as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return {"task_id": task_id}
