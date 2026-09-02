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
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

logger = logging.getLogger("crawl4ai.workqueue")

JobFactory = Callable[[], Awaitable[None]]
CancellationCallback = Callable[[], Awaitable[None]]

# One ZSET per principal (member = claim id, score = expiry deadline) is the
# sole owner of "active background jobs per principal". It is shared with the
# durable crawl queue: durable claims carry score +inf because their release is
# atomic inside the terminal-state Lua scripts and a pending backlog job may
# legitimately wait longer than any lease. In-process claims carry a finite
# deadline so a claim orphaned by a failed release or a killed process expires
# instead of consuming the slot forever (#76). Not cluster-safe key layout;
# the deployment runs a single Redis.
PRINCIPAL_QUOTA_LEASE_PREFIX = "crawl4ai:jobs:principal-leases:v2:"

# ponytail: fixed lease, no renewal. An in-process job outliving it temporarily
# frees its quota slot; add heartbeat renewal if legitimate >1h jobs appear.
QUOTA_LEASE_TTL_S = 3600

# The v1 hash pair this ZSET replaced (crawl4ai:jobs:principal-counts:v1 and
# crawl4ai:jobs:principal-claims:v1) is left untouched: old replicas keep
# enforcing on it consistently until the rollout completes, after which it is
# inert residue an operator may DEL. Deleting it at boot would erase the old
# replicas' live accounting mid-rollout and triple the admitted budget.


def principal_lease_key(principal: str) -> str:
    return PRINCIPAL_QUOTA_LEASE_PREFIX + principal


_ACQUIRE_PRINCIPAL_QUOTA_SCRIPT = """
-- crawl4ai:acquire-principal-quota
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[3])
if redis.call('ZSCORE', KEYS[1], ARGV[2]) then
  return 1
end
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then
  return 0
end
redis.call('ZADD', KEYS[1], ARGV[4], ARGV[2])
return 1
"""

_RELEASE_PRINCIPAL_QUOTA_SCRIPT = """
-- crawl4ai:release-principal-quota
local released = redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[2])
return released
"""


async def acquire_principal_quota(
    redis: Any,
    per_principal: int,
    principal: str | None,
    claim_token: str,
) -> bool:
    if per_principal <= 0 or not principal:
        return True
    now = time.time()
    acquired = await redis.eval(
        _ACQUIRE_PRINCIPAL_QUOTA_SCRIPT,
        1,
        principal_lease_key(principal),
        per_principal,
        claim_token,
        now,
        now + QUOTA_LEASE_TTL_S,
    )
    return int(acquired) == 1


async def release_principal_quota(
    redis: Any, principal: str | None, claim_token: str | None
) -> bool:
    if not claim_token or not principal:
        return False
    released = await redis.eval(
        _RELEASE_PRINCIPAL_QUOTA_SCRIPT,
        1,
        principal_lease_key(principal),
        claim_token,
        time.time(),
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
                await release_principal_quota(self.redis, principal, claim_token)
            except Exception:
                # Same failure class as the LLM-permit release (#63): a Redis
                # blip here must not kill the worker task, skip task_done(), or
                # replace a QueueFull/QuotaExceeded with a connection error.
                # The unreleased lease self-expires after QUOTA_LEASE_TTL_S.
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
