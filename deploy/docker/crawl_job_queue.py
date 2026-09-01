"""Durable Redis Stream queue for asynchronous crawl jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from redis.exceptions import ResponseError
from schemas import CrawlRequest
from utils import (
    DEFAULT_CONFIG,
    TaskStatus,
    get_redis_task_ttl,
    validate_url_destination,
)
from work_queue import principal_lease_key

from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.async_configs import (
    Provenance,
    UNTRUSTED_FIELD_ALLOWLIST,
    UntrustedConfigError,
    to_serializable_dict,
)

logger = logging.getLogger(__name__)

# v3: per-principal quota moved from the v1 hash pair to lease ZSETs (#76).
# The version bump keeps mixed-version replicas on disjoint streams during a
# rolling deploy, so an old worker can never complete a new job and strand its
# +inf quota member (or vice versa).
CRAWL_JOB_PROTOCOL_VERSION = 3


class CrawlJobLeaseLost(RuntimeError):
    """Raised when another consumer has taken over a pending crawl job."""


class CrawlJobQueueFull(RuntimeError):
    """Raised when the configured durable crawl-job backlog is full."""

    def __init__(self, max_pending_jobs: int):
        super().__init__(f"Crawl job queue is full (max_pending_jobs={max_pending_jobs})")


class CrawlJobPrincipalQuotaExceeded(RuntimeError):
    """Raised when one principal has filled its durable pending-job allowance."""

    def __init__(self, per_principal: int):
        super().__init__(f"Too many pending crawl jobs for this caller (limit={per_principal})")


class CrawlJobPayloadRejected(ValueError):
    """Raised before Redis mutation when a durable crawl payload is unsafe."""


_ENQUEUE_IF_CAPACITY_SCRIPT = """
-- crawl4ai:enqueue
-- KEYS[4] is the shared per-principal quota lease ZSET (see work_queue.py).
-- Durable claims carry score +inf: their release is atomic in the terminal
-- scripts, and a pending backlog job may outlive any finite lease.
if redis.call('XLEN', KEYS[1]) >= tonumber(ARGV[1]) then
  return 0
end
if tonumber(ARGV[2]) > 0 and ARGV[11] ~= '' then
  redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', ARGV[13])
  if redis.call('ZCARD', KEYS[4]) >= tonumber(ARGV[2]) then
    return -1
  end
end
redis.call('HSET', KEYS[2], 'status', ARGV[3], 'created_at', ARGV[4],
  'url', ARGV[5], 'result', ARGV[6], 'error', ARGV[7],
  'webhook_config', ARGV[8], 'owner', ARGV[11], 'protocol_version', ARGV[12])
redis.call('HSET', KEYS[3], 'payload', ARGV[9], 'attempt', '0',
  'owner', ARGV[11], 'lease_owner', '', 'fence_token', '',
  'protocol_version', ARGV[12])
redis.call(
  'XADD', KEYS[1], '*', 'task_id', ARGV[10],
  'protocol_version', ARGV[12], 'owner', ARGV[11]
)
if tonumber(ARGV[2]) > 0 and ARGV[11] ~= '' then
  redis.call('ZADD', KEYS[4], '+inf', ARGV[10])
end
return 1
"""


_CLAIM_STALE_SCRIPT = """
-- crawl4ai:claim-stale
local claimed = redis.call(
  'XAUTOCLAIM', KEYS[1], ARGV[1], ARGV[2], ARGV[3], ARGV[4], 'COUNT', 1
)
local messages = claimed[2]
if #messages > 0 then
  local fields = messages[1][2]
  local task_id = nil
  local protocol_version = nil
  for index = 1, #fields, 2 do
    if fields[index] == 'task_id' then
      task_id = fields[index + 1]
    elseif fields[index] == 'protocol_version' then
      protocol_version = fields[index + 1]
    end
  end
  if task_id and protocol_version == ARGV[7] then
    local payload_key = ARGV[6] .. task_id
    if redis.call('HEXISTS', payload_key, 'payload') == 1 then
    redis.call(
      'HSET', payload_key,
      'lease_owner', ARGV[2], 'fence_token', ARGV[5]
    )
    end
  end
end
return claimed
"""


_START_ATTEMPT_SCRIPT = """
-- crawl4ai:start-attempt
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[3], ARGV[3], 1)
if #pending == 0 or pending[1][2] ~= ARGV[2] then
  return {0, 0}
end
if redis.call('HEXISTS', KEYS[2], 'payload') == 0
  or redis.call('HGET', KEYS[2], 'protocol_version') ~= ARGV[10] then
  return {0, 0}
end
local attempt = redis.call('HINCRBY', KEYS[2], 'attempt', 1)
redis.call(
  'HSET', KEYS[2], 'attempt', attempt, 'lease_owner', ARGV[2],
  'fence_token', ARGV[4], 'owner', ARGV[8], 'protocol_version', ARGV[10]
)
redis.call(
  'HSET', KEYS[3], 'status', ARGV[5], 'created_at', ARGV[6],
  'url', ARGV[7], 'result', '', 'error', '', 'owner', ARGV[8],
  'attempt', attempt, 'lease_owner', ARGV[2], 'lease_stream_id', ARGV[3],
  'lease_heartbeat_at', ARGV[9], 'protocol_version', ARGV[10]
)
return {1, attempt}
"""


_HEARTBEAT_SCRIPT = """
-- crawl4ai:heartbeat
if redis.call('HGET', KEYS[2], 'fence_token') ~= ARGV[4]
  or redis.call('HGET', KEYS[2], 'lease_owner') ~= ARGV[2] then
  return 0
end
local claimed = redis.call(
  'XCLAIM', KEYS[1], ARGV[1], ARGV[2], 0, ARGV[3],
  'IDLE', 0, 'RETRYCOUNT', ARGV[5], 'JUSTID'
)
if #claimed == 0 then
  return 0
end
redis.call(
  'HSET', KEYS[3], 'status', ARGV[6], 'created_at', ARGV[7],
  'url', ARGV[8], 'result', '', 'error', '', 'owner', ARGV[9],
  'attempt', ARGV[5], 'lease_owner', ARGV[2], 'lease_stream_id', ARGV[3],
  'lease_heartbeat_at', ARGV[10]
)
return 1
"""


_MARK_RETRY_SCRIPT = """
-- crawl4ai:mark-retry
if redis.call('HGET', KEYS[1], 'fence_token') ~= ARGV[1]
  or redis.call('HGET', KEYS[1], 'lease_owner') ~= ARGV[2] then
  return 0
end
redis.call(
  'HSET', KEYS[2], 'status', ARGV[3], 'created_at', ARGV[4],
  'url', ARGV[5], 'result', '', 'error', '', 'owner', ARGV[6],
  'attempt', ARGV[7], 'last_error', ARGV[8], 'lease_owner', ARGV[2],
  'lease_stream_id', ARGV[9], 'lease_heartbeat_at', ARGV[10]
)
redis.call(
  'HSET', KEYS[1], 'last_error', ARGV[8], 'attempt', ARGV[7],
  'owner', ARGV[6]
)
return 1
"""


_COMPLETE_SCRIPT = """
-- crawl4ai:complete
if redis.call('HGET', KEYS[1], 'fence_token') ~= ARGV[1]
  or redis.call('HGET', KEYS[1], 'lease_owner') ~= ARGV[2] then
  return 0
end
local attempt = redis.call('HGET', KEYS[1], 'attempt') or ''
redis.call(
  'HSET', KEYS[2], 'status', ARGV[3], 'created_at', ARGV[4],
  'url', ARGV[5], 'result', ARGV[6], 'error', ARGV[7],
  'completed_at', ARGV[8], 'owner', ARGV[9], 'attempt', attempt
)
if tonumber(ARGV[10]) > 0 then
  redis.call('EXPIRE', KEYS[2], ARGV[10])
end
redis.call('ZREM', KEYS[4], ARGV[13])
redis.call('DEL', KEYS[1])
redis.call('XACK', KEYS[3], ARGV[11], ARGV[12])
redis.call('XDEL', KEYS[3], ARGV[12])
return 1
"""


_DISCARD_MISSING_PAYLOAD_SCRIPT = """
-- crawl4ai:discard-missing-payload
local pending_entry = redis.call(
  'XPENDING', KEYS[3], ARGV[1], ARGV[3], ARGV[3], 1
)
if #pending_entry == 0 or pending_entry[1][2] ~= ARGV[2] then
  return 0
end
if redis.call('HEXISTS', KEYS[2], 'payload') == 1 then
  return 0
end
local recovered_owner = ARGV[11]
if redis.call('EXISTS', KEYS[1]) == 1 then
  if redis.call('HGET', KEYS[1], 'status') ~= ARGV[4]
    or redis.call('HGET', KEYS[1], 'protocol_version') ~= ARGV[10] then
    return 0
  end
  local owner = redis.call('HGET', KEYS[1], 'owner') or ''
  redis.call(
    'HSET', KEYS[1], 'status', ARGV[5], 'result', '', 'error', ARGV[6],
    'completed_at', ARGV[7], 'owner', owner
  )
  if tonumber(ARGV[8]) > 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[8])
  end
else
  redis.call(
    'HSET', KEYS[1], 'status', ARGV[5], 'created_at', ARGV[7],
    'url', '', 'result', '', 'error', ARGV[6], 'completed_at', ARGV[7],
    'owner', recovered_owner, 'protocol_version', ARGV[10]
  )
  if tonumber(ARGV[8]) > 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[8])
  end
end
-- XAUTOCLAIM must never leave a lease-only payload hash behind.
redis.call('DEL', KEYS[2])
redis.call('ZREM', KEYS[4], ARGV[9])
redis.call('XACK', KEYS[3], ARGV[1], ARGV[3])
redis.call('XDEL', KEYS[3], ARGV[3])
return 1
"""


@dataclass(frozen=True)
class CrawlJobEntry:
    """A crawl job Redis Stream entry assigned to a consumer."""

    stream_id: str
    task_id: str


@dataclass(frozen=True)
class CrawlJobAttempt:
    """Immutable identity for one execution attempt of a Stream entry."""

    number: int
    fence_token: str
    consumer: str


@dataclass(frozen=True)
class CrawlJobQueueSettings:
    """Queue settings owned by ``utils.DEFAULT_CONFIG`` / ``config.yml``."""

    stream: str
    group: str
    lease_seconds: int
    heartbeat_seconds: int
    read_block_ms: int
    max_attempts: int
    max_pending_jobs: int
    max_attempt_seconds: int
    max_payload_bytes: int
    per_principal: int
    protocol_version: int

    @classmethod
    def from_config(cls, config: dict) -> CrawlJobQueueSettings:
        defaults = DEFAULT_CONFIG["crawl_jobs"]
        configured = {**defaults, **config.get("crawl_jobs", {})}

        def positive_int(name: str) -> int:
            candidate = configured[name]
            if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
                raise ValueError(f"crawl_jobs.{name} must be a positive integer")
            return candidate

        lease_seconds = positive_int("lease_seconds")
        heartbeat_seconds = positive_int("heartbeat_seconds")
        if heartbeat_seconds >= lease_seconds:
            raise ValueError("crawl_jobs.heartbeat_seconds must be less than lease_seconds")

        from governor import job_queue_caps

        per_principal = job_queue_caps(config)["per_principal"]
        if per_principal < 0:
            raise ValueError("limits.queue.per_principal must not be negative")

        protocol_version = positive_int("protocol_version")
        if protocol_version != CRAWL_JOB_PROTOCOL_VERSION:
            raise ValueError(
                "crawl_jobs.protocol_version is unsupported by this worker "
                f"(expected {CRAWL_JOB_PROTOCOL_VERSION})"
            )
        protocol_suffix = f":v{protocol_version}"

        def versioned_name(name: str) -> str:
            return name if name.endswith(protocol_suffix) else f"{name}{protocol_suffix}"

        return cls(
            stream=versioned_name(str(configured["stream"])),
            group=versioned_name(str(configured["group"])),
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            read_block_ms=positive_int("read_block_ms"),
            max_attempts=positive_int("max_attempts"),
            max_pending_jobs=positive_int("max_pending_jobs"),
            max_attempt_seconds=positive_int("max_attempt_seconds"),
            max_payload_bytes=positive_int("max_payload_bytes"),
            per_principal=per_principal,
            protocol_version=protocol_version,
        )


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _field(fields: Mapping[Any, Any] | list[Any], name: str) -> str | None:
    if isinstance(fields, Mapping):
        items = fields.items()
    else:
        items = zip(fields[::2], fields[1::2])
    for key, value in items:
        if _as_text(key) == name:
            return _as_text(value)
    return None


class CrawlJobQueue:
    """Owns the persistent request payload and Stream lifecycle for crawl jobs."""

    def __init__(self, redis: Any, config: dict):
        self.redis = redis
        self.config = config
        self.settings = CrawlJobQueueSettings.from_config(config)
        self._claim_cursor = "0-0"

    @staticmethod
    def task_key(task_id: str) -> str:
        return f"task:{task_id}"

    @property
    def payload_prefix(self) -> str:
        return f"crawl-job:v{self.settings.protocol_version}:"

    def payload_key(self, task_id: str) -> str:
        return f"{self.payload_prefix}{task_id}"

    @staticmethod
    def _created_at() -> str:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    @staticmethod
    async def _canonical_urls(urls: list[str]) -> list[str]:
        try:
            validated = CrawlRequest.model_validate({"urls": urls}).urls
        except (TypeError, ValueError) as error:
            raise CrawlJobPayloadRejected("Rejected crawl job URL list") from error

        normalized = [
            url
            if url.startswith(("http://", "https://", "raw:", "raw://"))
            else f"https://{url}"
            for url in validated
        ]
        for url in normalized:
            try:
                await asyncio.to_thread(validate_url_destination, url)
            except Exception as error:
                raise CrawlJobPayloadRejected("Rejected crawl job URL") from error
        return normalized

    def _canonical_payload(
        self,
        *,
        urls: list[str],
        browser_config: dict,
        crawler_config: dict,
        result_fields: list[str] | None,
        webhook_config: dict | None,
        created_at: str,
        owner: str,
    ) -> str:
        try:
            loaded_browser = BrowserConfig.load(
                browser_config,
                provenance=Provenance.UNTRUSTED,
            )
            canonical_browser = {"type": "BrowserConfig", "params": {}}
            request_browser_params = (
                browser_config.get("params", {})
                if browser_config.get("type") == "BrowserConfig"
                else browser_config
            )
            for field in request_browser_params:
                if field in UNTRUSTED_FIELD_ALLOWLIST["BrowserConfig"]:
                    canonical_browser["params"][field] = to_serializable_dict(
                        getattr(loaded_browser, field)
                    )
            loaded_crawler = CrawlerRunConfig.load(
                crawler_config,
                provenance=Provenance.UNTRUSTED,
            )
            canonical_crawler = loaded_crawler.dump()
            request_params = (
                crawler_config.get("params", {})
                if crawler_config.get("type") == "CrawlerRunConfig"
                else crawler_config
            )
            for field in request_params:
                if field in UNTRUSTED_FIELD_ALLOWLIST["CrawlerRunConfig"]:
                    canonical_crawler["params"][field] = to_serializable_dict(
                        getattr(loaded_crawler, field)
                    )
        except (TypeError, UntrustedConfigError, ValueError) as error:
            raise CrawlJobPayloadRejected(f"Rejected crawl job configuration: {error}") from error

        payload = {
            "protocol_version": self.settings.protocol_version,
            "urls": urls,
            "browser_config": canonical_browser,
            "crawler_config": canonical_crawler,
            "result_fields": result_fields,
            "webhook_config": webhook_config,
            "created_at": created_at,
            "owner": owner,
        }
        try:
            serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise CrawlJobPayloadRejected("Rejected non-serializable crawl job payload") from error
        payload_bytes = len(serialized.encode("utf-8"))
        if payload_bytes > self.settings.max_payload_bytes:
            raise CrawlJobPayloadRejected(
                "Crawl job payload exceeds the durable payload limit "
                f"({payload_bytes} > {self.settings.max_payload_bytes} bytes)"
            )
        return serialized

    async def ensure_group(self) -> None:
        """Create the consumer group without skipping work queued before startup."""
        if self.settings.protocol_version > 1:
            suffix = f":v{self.settings.protocol_version}"
            base_stream = self.settings.stream.removesuffix(suffix)
            previous = self.settings.protocol_version - 1
            # v1 streams carried no suffix; later versions carry :v{n}.
            legacy_stream = base_stream if previous == 1 else f"{base_stream}:v{previous}"
            if await self.redis.xlen(legacy_stream):
                raise RuntimeError(
                    f"legacy crawl jobs remain in {legacy_stream}; refusing v{self.settings.protocol_version} worker startup"
                )
        try:
            await self.redis.xgroup_create(
                self.settings.stream,
                self.settings.group,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    async def enqueue(
        self,
        urls: list[str],
        browser_config: dict,
        crawler_config: dict,
        result_fields: list[str] | None,
        webhook_config: dict | None,
        owner: str | None = None,
    ) -> str:
        """Atomically persist a request and make it visible to worker consumers."""
        task_id = f"crawl_{uuid4().hex}"
        created_at = self._created_at()
        canonical_urls = await self._canonical_urls(urls)
        serialized_payload = self._canonical_payload(
            urls=canonical_urls,
            browser_config=browser_config,
            crawler_config=crawler_config,
            result_fields=result_fields,
            webhook_config=webhook_config,
            created_at=created_at,
            owner=owner or "",
        )
        enqueued = await self.redis.eval(
            _ENQUEUE_IF_CAPACITY_SCRIPT,
            4,
            self.settings.stream,
            self.task_key(task_id),
            self.payload_key(task_id),
            principal_lease_key(owner or ""),
            self.settings.max_pending_jobs,
            self.settings.per_principal,
            TaskStatus.PROCESSING.value,
            created_at,
            json.dumps(canonical_urls),
            "",
            "",
            json.dumps(webhook_config) if webhook_config else "",
            serialized_payload,
            task_id,
            owner or "",
            self.settings.protocol_version,
            time.time(),
        )
        if int(enqueued) == -1:
            raise CrawlJobPrincipalQuotaExceeded(self.settings.per_principal)
        if int(enqueued) != 1:
            raise CrawlJobQueueFull(self.settings.max_pending_jobs)
        return task_id

    async def read_new(self, consumer: str) -> list[CrawlJobEntry]:
        response = await self.redis.xreadgroup(
            self.settings.group,
            consumer,
            {self.settings.stream: ">"},
            count=1,
            block=self.settings.read_block_ms,
        )
        return self._stream_entries(response)

    async def claim_stale(self, consumer: str) -> list[CrawlJobEntry]:
        """Claim one stale pending entry whose lease was not heartbeated."""
        response = await self.redis.eval(
            _CLAIM_STALE_SCRIPT,
            1,
            self.settings.stream,
            self.settings.group,
            consumer,
            self.settings.lease_seconds * 1000,
            self._claim_cursor,
            f"claim-{uuid4().hex}",
            self.payload_prefix,
            self.settings.protocol_version,
        )
        if not response:
            self._claim_cursor = "0-0"
            return []

        next_start_id, messages, _deleted = response
        self._claim_cursor = _as_text(next_start_id)
        return self._messages_to_entries(messages)

    async def load_payload(self, task_id: str) -> dict | None:
        payload = await self.redis.hget(self.payload_key(task_id), "payload")
        if payload is None:
            return None
        return json.loads(_as_text(payload))

    async def start_attempt(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        consumer: str,
    ) -> CrawlJobAttempt:
        fence_token = uuid4().hex
        owner = payload.get("owner") or ""
        result = await self.redis.eval(
            _START_ATTEMPT_SCRIPT,
            3,
            self.settings.stream,
            self.payload_key(entry.task_id),
            self.task_key(entry.task_id),
            self.settings.group,
            consumer,
            entry.stream_id,
            fence_token,
            TaskStatus.PROCESSING.value,
            payload["created_at"],
            json.dumps(payload["urls"]),
            owner,
            self._created_at(),
            self.settings.protocol_version,
        )
        if not result or int(result[0]) != 1:
            raise CrawlJobLeaseLost(f"crawl job {entry.task_id} is no longer owned by {consumer}")
        return CrawlJobAttempt(
            number=int(result[1]),
            fence_token=fence_token,
            consumer=consumer,
        )

    async def heartbeat(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        consumer: str,
        attempt: CrawlJobAttempt,
    ) -> None:
        """Renew both the pending-entry lease and its durable task metadata."""
        renewed = await self.redis.eval(
            _HEARTBEAT_SCRIPT,
            3,
            self.settings.stream,
            self.payload_key(entry.task_id),
            self.task_key(entry.task_id),
            self.settings.group,
            consumer,
            entry.stream_id,
            attempt.fence_token,
            attempt.number,
            TaskStatus.PROCESSING.value,
            payload["created_at"],
            json.dumps(payload["urls"]),
            payload.get("owner") or "",
            self._created_at(),
        )
        if int(renewed) != 1:
            raise CrawlJobLeaseLost(f"crawl job {entry.task_id} lease was claimed by another worker")

    async def mark_retry(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        consumer: str,
        attempt: CrawlJobAttempt,
        error: str,
    ) -> None:
        updated = await self.redis.eval(
            _MARK_RETRY_SCRIPT,
            2,
            self.payload_key(entry.task_id),
            self.task_key(entry.task_id),
            attempt.fence_token,
            consumer,
            TaskStatus.PROCESSING.value,
            payload["created_at"],
            json.dumps(payload["urls"]),
            payload.get("owner") or "",
            attempt.number,
            error,
            entry.stream_id,
            self._created_at(),
        )
        if int(updated) != 1:
            raise CrawlJobLeaseLost(f"crawl job {entry.task_id} lease was claimed by another worker")

    async def complete(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: CrawlJobAttempt,
        *,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Atomically publish a terminal task state and remove its Stream entry."""
        status = TaskStatus.COMPLETED.value if error is None else TaskStatus.FAILED.value
        completed = await self.redis.eval(
            _COMPLETE_SCRIPT,
            4,
            self.payload_key(entry.task_id),
            self.task_key(entry.task_id),
            self.settings.stream,
            principal_lease_key(payload.get("owner") or ""),
            attempt.fence_token,
            attempt.consumer,
            status,
            payload["created_at"],
            json.dumps(payload["urls"]),
            json.dumps(result) if result is not None else "",
            error or "",
            self._created_at(),
            payload.get("owner") or "",
            get_redis_task_ttl(self.config),
            self.settings.group,
            entry.stream_id,
            entry.task_id,
        )
        if int(completed) != 1:
            raise CrawlJobLeaseLost(
                f"crawl job {entry.task_id} lease held by {attempt.consumer} was claimed by another worker"
            )

    async def discard_missing_payload(self, entry: CrawlJobEntry, consumer: str) -> None:
        """Fenced cleanup for an entry whose durable payload is absent."""
        # The quota lease key is per-principal; recover the owner from the
        # Stream entry's immutable 'owner' field before the fenced script runs
        # and thread it through ARGV (the script must not re-read it).
        stream_owner = ""
        for _stream_id, fields in await self.redis.xrange(
            self.settings.stream, entry.stream_id, entry.stream_id, count=1
        ) or []:
            stream_owner = _as_text(_field(fields, "owner") or "")
        discarded = await self.redis.eval(
            _DISCARD_MISSING_PAYLOAD_SCRIPT,
            4,
            self.task_key(entry.task_id),
            self.payload_key(entry.task_id),
            self.settings.stream,
            principal_lease_key(stream_owner),
            self.settings.group,
            consumer,
            entry.stream_id,
            TaskStatus.PROCESSING.value,
            TaskStatus.FAILED.value,
            "Durable crawl job payload is missing",
            self._created_at(),
            get_redis_task_ttl(self.config),
            entry.task_id,
            self.settings.protocol_version,
            stream_owner,
        )
        if int(discarded) != 1:
            raise CrawlJobLeaseLost(
                f"crawl job {entry.task_id} missing-payload cleanup lost its lease"
            )

    @staticmethod
    def _stream_entries(response: Any) -> list[CrawlJobEntry]:
        entries: list[CrawlJobEntry] = []
        for _stream_name, messages in response or []:
            entries.extend(CrawlJobQueue._messages_to_entries(messages))
        return entries

    @staticmethod
    def _messages_to_entries(messages: Any) -> list[CrawlJobEntry]:
        entries: list[CrawlJobEntry] = []
        for stream_id, fields in messages or []:
            task_id = _field(fields, "task_id")
            if task_id:
                entries.append(CrawlJobEntry(_as_text(stream_id), task_id))
            else:
                logger.error("Ignoring crawl Stream entry %s without task_id", _as_text(stream_id))
        return entries
