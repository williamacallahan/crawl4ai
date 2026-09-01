"""
work_queue.py - bounded background-job execution with per-principal quotas.

/crawl/job and /llm/job used FastAPI BackgroundTasks with no bound: a client
could enqueue unlimited background jobs and exhaust memory / browser slots, and
one caller could starve others.

This replaces that with a fixed worker pool draining an asyncio.Queue, plus an
optional per-principal concurrency cap. Everything is configurable, and any
limit set to 0 (or null) means "unbounded" - i.e. the previous behavior is fully
recoverable:

    limits.queue.maxsize        0 => unbounded queue (never 503)
    limits.queue.workers        worker pool size (>=1)
    limits.queue.per_principal  0 => no per-caller cap (never 429)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

logger = logging.getLogger("crawl4ai.workqueue")

JobFactory = Callable[[], Awaitable[None]]
CancellationCallback = Callable[[], Awaitable[None]]

PRINCIPAL_QUOTA_COUNTS_KEY = "crawl4ai:jobs:principal-counts:v1"
PRINCIPAL_QUOTA_CLAIMS_KEY = "crawl4ai:jobs:principal-claims:v1"

_ACQUIRE_PRINCIPAL_QUOTA_SCRIPT = """
-- crawl4ai:acquire-principal-quota
if tonumber(ARGV[1]) <= 0 or ARGV[2] == '' then
  return 1
end
if redis.call('HEXISTS', KEYS[2], ARGV[3]) == 1 then
  return 1
end
local active = tonumber(redis.call('HGET', KEYS[1], ARGV[2]) or '0')
if active >= tonumber(ARGV[1]) then
  return 0
end
redis.call('HSET', KEYS[2], ARGV[3], ARGV[2])
redis.call('HINCRBY', KEYS[1], ARGV[2], 1)
return 1
"""

_RELEASE_PRINCIPAL_QUOTA_SCRIPT = """
-- crawl4ai:release-principal-quota
local principal = redis.call('HGET', KEYS[2], ARGV[1])
if not principal then
  return 0
end
redis.call('HDEL', KEYS[2], ARGV[1])
local active = tonumber(redis.call('HGET', KEYS[1], principal) or '0')
if active <= 1 then
  redis.call('HDEL', KEYS[1], principal)
else
  redis.call('HINCRBY', KEYS[1], principal, -1)
end
return 1
"""


async def acquire_principal_quota(
    redis: Any,
    per_principal: int,
    principal: str | None,
    claim_token: str,
) -> bool:
    if per_principal <= 0 or not principal:
        return True
    acquired = await redis.eval(
        _ACQUIRE_PRINCIPAL_QUOTA_SCRIPT,
        2,
        PRINCIPAL_QUOTA_COUNTS_KEY,
        PRINCIPAL_QUOTA_CLAIMS_KEY,
        per_principal,
        principal,
        claim_token,
    )
    return int(acquired) == 1


async def release_principal_quota(redis: Any, claim_token: str | None) -> bool:
    if not claim_token:
        return False
    released = await redis.eval(
        _RELEASE_PRINCIPAL_QUOTA_SCRIPT,
        2,
        PRINCIPAL_QUOTA_COUNTS_KEY,
        PRINCIPAL_QUOTA_CLAIMS_KEY,
        claim_token,
    )
    return int(released) == 1


class QueueFull(Exception):
    """The bounded job queue is full -> 503 Retry-After."""


class QuotaExceeded(Exception):
    """The principal has too many concurrent jobs -> 429."""


class WorkQueue:
    def __init__(
        self,
        maxsize: int = 0,
        workers: int = 4,
        per_principal: int = 0,
        redis: Any = None,
    ):
        self.maxsize = max(0, int(maxsize))          # 0 = unbounded
        self.workers = max(1, int(workers))
        self.per_principal = max(0, int(per_principal))  # 0 = unlimited
        self.redis = redis
        self._q: asyncio.Queue[
            tuple[str | None, JobFactory, CancellationCallback | None, str | None]
        ] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._counts: dict[str, int] = {}

    @property
    def started(self) -> bool:
        return self._q is not None

    async def start(self) -> None:
        self._q = asyncio.Queue(maxsize=self.maxsize)
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]
        logger.info(
            "work queue started (maxsize=%s, workers=%s, per_principal=%s)",
            self.maxsize or "unbounded", self.workers, self.per_principal or "unlimited",
        )

    async def stop(self) -> None:
        queue = self._q
        tasks, self._tasks = self._tasks, []
        self._q = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if queue is not None:
            while True:
                try:
                    principal, _factory, on_cancel, claim_token = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    await self._cancel(on_cancel)
                    await self._release(principal, claim_token)
                finally:
                    queue.task_done()
        self._counts.clear()

    async def _worker(self) -> None:
        queue = self._q
        assert queue is not None
        while True:
            principal, factory, on_cancel, claim_token = await queue.get()
            try:
                await factory()
            except asyncio.CancelledError:
                await self._cancel(on_cancel)
                raise
            except Exception:
                logger.exception("background job failed")
            finally:
                await self._release(principal, claim_token)
                queue.task_done()

    async def _release(self, principal: str | None, claim_token: str | None) -> None:
        if self.redis is not None:
            try:
                await release_principal_quota(self.redis, claim_token)
            except Exception:
                # Same failure class as the LLM-permit release (#63): a Redis
                # blip here must not kill the worker task, skip task_done(), or
                # replace a QueueFull/QuotaExceeded with a connection error.
                # Unlike the permit key these quota hashes have no TTL, so a
                # swallowed failure leaks one slot for this principal.
                logger.exception(
                    "Failed to release principal quota (claim %s)", claim_token
                )
            return
        if not principal:
            return
        n = self._counts.get(principal, 0) - 1
        if n <= 0:
            self._counts.pop(principal, None)
        else:
            self._counts[principal] = n

    async def _cancel(self, callback: CancellationCallback | None) -> None:
        if callback is None:
            return
        try:
            await callback()
        except Exception:
            logger.exception("background job cancellation callback failed")

    async def submit(
        self,
        factory: JobFactory,
        principal: str | None = None,
        on_cancel: CancellationCallback | None = None,
    ) -> None:
        """Enqueue a job. Raises QuotaExceeded / QueueFull (mapped to 429 / 503)."""
        queue = self._q
        if queue is None:
            raise RuntimeError("work queue not started")

        claim_token = uuid4().hex if self.per_principal and principal else None
        if self.redis is not None:
            if not await acquire_principal_quota(
                self.redis,
                self.per_principal,
                principal,
                claim_token or "",
            ):
                raise QuotaExceeded()
        elif self.per_principal and principal:
            if self._counts.get(principal, 0) >= self.per_principal:
                raise QuotaExceeded()
            self._counts[principal] = self._counts.get(principal, 0) + 1

        if self._q is not queue:
            await self._release(principal, claim_token)
            raise RuntimeError("work queue stopped during admission")
        try:
            queue.put_nowait((principal, factory, on_cancel, claim_token))
        except asyncio.QueueFull:
            await self._release(principal, claim_token)
            raise QueueFull()


# Process-wide singleton, set at server boot.
_JOB_QUEUE: WorkQueue | None = None


def set_job_queue(q: WorkQueue | None) -> None:
    global _JOB_QUEUE
    _JOB_QUEUE = q


def get_job_queue() -> WorkQueue | None:
    return _JOB_QUEUE
