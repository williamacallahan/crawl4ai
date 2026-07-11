"""Durable Redis Stream queue for asynchronous crawl jobs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from uuid import uuid4

from redis.exceptions import ResponseError

from utils import DEFAULT_CONFIG, TaskStatus, get_redis_task_ttl

logger = logging.getLogger(__name__)


class CrawlJobLeaseLost(RuntimeError):
    """Raised when another consumer has taken over a pending crawl job."""


class CrawlJobQueueFull(RuntimeError):
    """Raised when the configured durable crawl-job backlog is full."""

    def __init__(self, max_pending_jobs: int):
        super().__init__(f"Crawl job queue is full (max_pending_jobs={max_pending_jobs})")


_ENQUEUE_IF_CAPACITY_SCRIPT = """
if redis.call('XLEN', KEYS[1]) >= tonumber(ARGV[1]) then
  return 0
end
redis.call('HSET', KEYS[2], 'status', ARGV[2], 'created_at', ARGV[3],
  'url', ARGV[4], 'result', ARGV[5], 'error', ARGV[6], 'webhook_config', ARGV[7])
redis.call('HSET', KEYS[3], 'payload', ARGV[8], 'attempt', '0')
redis.call('XADD', KEYS[1], '*', 'task_id', ARGV[9])
return 1
"""


@dataclass(frozen=True)
class CrawlJobEntry:
    """A crawl job Redis Stream entry assigned to a consumer."""

    stream_id: str
    task_id: str


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

    @classmethod
    def from_config(cls, config: dict) -> "CrawlJobQueueSettings":
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

        return cls(
            stream=str(configured["stream"]),
            group=str(configured["group"]),
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            read_block_ms=positive_int("read_block_ms"),
            max_attempts=positive_int("max_attempts"),
            max_pending_jobs=positive_int("max_pending_jobs"),
        )


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _field(fields: Mapping[Any, Any], name: str) -> Optional[str]:
    for key, value in fields.items():
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

    @staticmethod
    def payload_key(task_id: str) -> str:
        return f"crawl-job:{task_id}"

    @staticmethod
    def _created_at() -> str:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    async def ensure_group(self) -> None:
        """Create the consumer group without skipping work queued before startup."""
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
        result_fields: Optional[list[str]],
        webhook_config: Optional[dict],
    ) -> str:
        """Atomically persist a request and make it visible to worker consumers."""
        task_id = f"crawl_{uuid4().hex}"
        created_at = self._created_at()
        payload = {
            "urls": urls,
            "browser_config": browser_config,
            "crawler_config": crawler_config,
            "result_fields": result_fields,
            "webhook_config": webhook_config,
            "created_at": created_at,
        }
        enqueued = await self.redis.eval(
            _ENQUEUE_IF_CAPACITY_SCRIPT,
            3,
            self.settings.stream,
            self.task_key(task_id),
            self.payload_key(task_id),
            self.settings.max_pending_jobs,
            TaskStatus.PROCESSING.value,
            created_at,
            json.dumps(urls),
            "",
            "",
            json.dumps(webhook_config) if webhook_config else "",
            json.dumps(payload),
            task_id,
        )
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
        response = await self.redis.xautoclaim(
            self.settings.stream,
            self.settings.group,
            consumer,
            min_idle_time=self.settings.lease_seconds * 1000,
            start_id=self._claim_cursor,
            count=1,
        )
        if not response:
            self._claim_cursor = "0-0"
            return []

        next_start_id, messages, _deleted = response
        self._claim_cursor = _as_text(next_start_id)
        return self._messages_to_entries(messages)

    async def load_payload(self, task_id: str) -> Optional[dict]:
        payload = await self.redis.hget(self.payload_key(task_id), "payload")
        if payload is None:
            return None
        return json.loads(_as_text(payload))

    async def start_attempt(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        consumer: str,
    ) -> int:
        attempt = await self.redis.hincrby(self.payload_key(entry.task_id), "attempt", 1)
        await self._store_processing(entry, payload, consumer, int(attempt))
        return int(attempt)

    async def heartbeat(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        consumer: str,
        attempt: int,
    ) -> None:
        """Renew both the pending-entry lease and its durable task metadata."""
        claimed = await self.redis.xclaim(
            self.settings.stream,
            self.settings.group,
            consumer,
            0,
            [entry.stream_id],
            idle=0,
            retrycount=attempt,
        )
        if not claimed:
            raise CrawlJobLeaseLost(f"crawl job {entry.task_id} lease was claimed by another worker")
        await self._store_processing(entry, payload, consumer, attempt)

    async def mark_retry(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        consumer: str,
        attempt: int,
        error: str,
    ) -> None:
        task = {
            "status": TaskStatus.PROCESSING.value,
            "created_at": payload["created_at"],
            "url": json.dumps(payload["urls"]),
            "result": "",
            "error": "",
            "attempt": str(attempt),
            "last_error": error,
            "lease_owner": consumer,
            "lease_stream_id": entry.stream_id,
            "lease_heartbeat_at": self._created_at(),
        }
        pipeline = self.redis.pipeline(transaction=True)
        pipeline.hset(self.task_key(entry.task_id), mapping=task)
        pipeline.hset(
            self.payload_key(entry.task_id),
            mapping={"last_error": error, "attempt": str(attempt)},
        )
        await pipeline.execute()

    async def complete(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        *,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Atomically publish a terminal task state and remove its Stream entry."""
        status = TaskStatus.COMPLETED.value if error is None else TaskStatus.FAILED.value
        task = {
            "status": status,
            "created_at": payload["created_at"],
            "url": json.dumps(payload["urls"]),
            "result": json.dumps(result) if result is not None else "",
            "error": error or "",
            "completed_at": self._created_at(),
        }
        pipeline = self.redis.pipeline(transaction=True)
        pipeline.hset(self.task_key(entry.task_id), mapping=task)
        self._expire_terminal_task(pipeline, self.task_key(entry.task_id))
        pipeline.delete(self.payload_key(entry.task_id))
        pipeline.xack(self.settings.stream, self.settings.group, entry.stream_id)
        pipeline.xdel(self.settings.stream, entry.stream_id)
        await pipeline.execute()

    async def discard_missing_payload(self, entry: CrawlJobEntry) -> None:
        """Remove an unrecoverable entry whose durable payload is absent."""
        pipeline = self.redis.pipeline(transaction=True)
        pipeline.xack(self.settings.stream, self.settings.group, entry.stream_id)
        pipeline.xdel(self.settings.stream, entry.stream_id)
        await pipeline.execute()

    async def _store_processing(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        consumer: str,
        attempt: int,
    ) -> None:
        task = {
            "status": TaskStatus.PROCESSING.value,
            "created_at": payload["created_at"],
            "url": json.dumps(payload["urls"]),
            "result": "",
            "error": "",
            "attempt": str(attempt),
            "lease_owner": consumer,
            "lease_stream_id": entry.stream_id,
            "lease_heartbeat_at": self._created_at(),
        }
        pipeline = self.redis.pipeline(transaction=True)
        pipeline.hset(self.task_key(entry.task_id), mapping=task)
        pipeline.hset(
            self.payload_key(entry.task_id),
            mapping={"attempt": str(attempt), "lease_owner": consumer},
        )
        await pipeline.execute()

    def _expire_terminal_task(self, pipeline: Any, key: str) -> None:
        task_ttl = get_redis_task_ttl(self.config)
        if task_ttl > 0:
            pipeline.expire(key, task_ttl)

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
