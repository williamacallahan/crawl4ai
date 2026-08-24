"""Dedicated Redis Stream consumer for durable ``/crawl/job`` execution."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from crawl_job_queue import (
    CrawlJobAttempt,
    CrawlJobEntry,
    CrawlJobLeaseLost,
    CrawlJobQueue,
)
from crawler_pool import close_all
from egress_proxy import start_pinning_proxy, stop_pinning_proxy
from fastapi import HTTPException, status
from redis import asyncio as aioredis
from redis_config import build_redis_url
from utils import load_config, setup_logging
from webhook import WebhookDeliveryService

logger = logging.getLogger(__name__)

CrawlCallable = Callable[[dict], Awaitable[dict]]


class CrawlJobWorker:
    """Processes one Stream entry at a time so browser pool limits remain effective."""

    def __init__(
        self,
        queue: CrawlJobQueue,
        config: dict,
        consumer: str,
        crawl: CrawlCallable | None = None,
        webhook_service: Any | None = None,
    ):
        self.queue = queue
        self.config = config
        self.consumer = consumer
        self.crawl = crawl or self._crawl
        self.webhook_service = webhook_service or WebhookDeliveryService(config)

    async def run(self) -> None:
        await self.queue.ensure_group()
        while True:
            try:
                entries = await self.queue.claim_stale(self.consumer)
                if not entries:
                    entries = await self.queue.read_new(self.consumer)
                for entry in entries:
                    await self.process(entry)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Crawl job worker loop failed; retaining pending work for retry")
                await asyncio.sleep(1)

    async def process(self, entry: CrawlJobEntry) -> None:
        payload = await self.queue.load_payload(entry.task_id)
        if payload is None:
            logger.error("Discarding crawl job %s because its durable payload is absent", entry.task_id)
            try:
                await self.queue.discard_missing_payload(entry, self.consumer)
            except CrawlJobLeaseLost:
                logger.warning(
                    "Crawl worker no longer owns %s during missing-payload cleanup",
                    entry.task_id,
                )
            return

        try:
            attempt = await self.queue.start_attempt(entry, payload, self.consumer)
        except CrawlJobLeaseLost:
            logger.warning("Crawl worker no longer owns %s before attempt start", entry.task_id)
            return
        if attempt.number > self.queue.settings.max_attempts:
            await self.queue.complete(
                entry,
                payload,
                attempt,
                error="Crawl job exhausted its retry budget before execution",
            )
            await self._notify(
                entry,
                payload,
                "failed",
                error="Crawl job exhausted its retry budget before execution",
            )
            return

        try:
            completion = await self._run_with_lease(entry, payload, attempt)
        except asyncio.CancelledError:
            logger.info("Crawl worker cancelled while processing %s; leaving Stream entry pending", entry.task_id)
            raise
        except CrawlJobLeaseLost:
            logger.warning("Crawl worker lost the lease for %s; leaving it pending", entry.task_id)
            return
        except Exception:
            logger.exception("Crawl job %s failed outside its retry transaction", entry.task_id)
            return

        if completion:
            status, result, error = completion
            await self._notify(entry, payload, status, result=result, error=error)

    async def _run_with_lease(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: CrawlJobAttempt,
    ) -> tuple[str, dict | None, str | None] | None:
        operation_task = asyncio.create_task(self._process_attempt(entry, payload, attempt))
        heartbeat_task = asyncio.create_task(self._heartbeat(entry, payload, attempt))
        try:
            done, _pending = await asyncio.wait(
                {operation_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self.queue.settings.max_attempt_seconds,
            )
            if not done:
                return await self._release_stalled_attempt(entry, payload, attempt)
            if operation_task in done:
                return operation_task.result()

            heartbeat_task.result()
            raise CrawlJobLeaseLost(f"heartbeat stopped for {entry.task_id}")
        finally:
            if not operation_task.done():
                operation_task.cancel()
            if not heartbeat_task.done():
                heartbeat_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await operation_task
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat_task

    async def _release_stalled_attempt(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: CrawlJobAttempt,
    ) -> tuple[str, dict | None, str | None] | None:
        """Give up on an attempt that outlived its budget so its consumer is freed.

        The heartbeat renews the lease for as long as an attempt runs, which means a crawl
        that never returns keeps proving it is alive: its entry never goes idle, XAUTOCLAIM
        can never reclaim it, and that consumer is occupied permanently. With few workers a
        couple of hung jobs starve the whole group — the stream stops being drained, XLEN
        stays at max_pending_jobs, and every new submission is rejected with a queue-full
        503 even though the service looks healthy. Bounding the attempt is what makes the
        existing reclaim path reachable.
        """
        error_message = (
            "Crawl job exceeded its "
            f"{self.queue.settings.max_attempt_seconds}s attempt budget and was released"
        )
        logger.error(
            "Crawl job %s exceeded its attempt budget on attempt %s/%s; releasing its lease",
            entry.task_id,
            attempt.number,
            self.queue.settings.max_attempts,
        )
        if attempt.number >= self.queue.settings.max_attempts:
            await self.queue.complete(entry, payload, attempt, error=error_message)
            return "failed", None, error_message
        # Leave the entry pending: cancelling the heartbeat lets its idle time pass
        # lease_seconds so another worker reclaims it through claim_stale.
        await self.queue.mark_retry(entry, payload, self.consumer, attempt, error_message)
        return None

    async def _process_attempt(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: CrawlJobAttempt,
    ) -> tuple[str, dict | None, str | None] | None:
        try:
            result = await self.crawl(payload)
        except asyncio.CancelledError:
            raise
        except (HTTPException, OSError, RuntimeError, TypeError, ValueError) as error:
            error_message = str(error.detail if isinstance(error, HTTPException) else error)
            terminal_input = isinstance(error, HTTPException) and error.status_code in {
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            }
            if terminal_input or attempt.number >= self.queue.settings.max_attempts:
                await self.queue.complete(entry, payload, attempt, error=error_message)
                return "failed", None, error_message
            else:
                logger.warning(
                    "Crawl job %s attempt %s/%s failed; retaining it for lease-based retry: %s",
                    entry.task_id,
                    attempt.number,
                    self.queue.settings.max_attempts,
                    error_message,
                )
                await self.queue.mark_retry(entry, payload, self.consumer, attempt, error_message)
            return

        await self.queue.complete(entry, payload, attempt, result=result)
        return "completed", result, None

    async def _crawl(self, payload: dict) -> dict:
        """Load the existing synchronous crawl owner lazily inside the worker."""
        from api import handle_crawl_request

        return await handle_crawl_request(
            urls=payload["urls"],
            browser_config=payload["browser_config"],
            crawler_config=payload["crawler_config"],
            config=self.config,
            result_fields=payload.get("result_fields"),
        )

    async def _heartbeat(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: CrawlJobAttempt,
    ) -> None:
        while True:
            await asyncio.sleep(self.queue.settings.heartbeat_seconds)
            await self.queue.heartbeat(entry, payload, self.consumer, attempt)

    async def _notify(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        status: str,
        *,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        try:
            await self.webhook_service.notify_job_completion(
                task_id=entry.task_id,
                task_type="crawl",
                status=status,
                urls=payload["urls"],
                webhook_config=payload.get("webhook_config"),
                result=result,
                error=error,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Webhook notification failed for crawl job %s", entry.task_id)


async def run_worker() -> None:
    config = load_config()
    setup_logging(config)
    redis = aioredis.from_url(build_redis_url(config), decode_responses=True)
    queue = CrawlJobQueue(redis, config)
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    proxy = None
    try:
        proxy = await start_pinning_proxy()
        worker = CrawlJobWorker(queue, config, consumer)
        await worker.run()
    finally:
        try:
            await close_all()
        finally:
            try:
                await stop_pinning_proxy(proxy)
            finally:
                await redis.aclose()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
