"""Dedicated Redis Stream consumer for durable ``/crawl/job`` execution."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from contextlib import suppress
from typing import Any, Awaitable, Callable, Optional

from redis import asyncio as aioredis

from crawl_job_queue import CrawlJobEntry, CrawlJobLeaseLost, CrawlJobQueue
from crawler_pool import close_all
from utils import build_redis_url, load_config, setup_logging
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
        crawl: Optional[CrawlCallable] = None,
        webhook_service: Optional[Any] = None,
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
            await self.queue.discard_missing_payload(entry)
            return

        attempt = await self.queue.start_attempt(entry, payload, self.consumer)
        if attempt > self.queue.settings.max_attempts:
            await self.queue.complete(
                entry,
                payload,
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
        attempt: int,
    ) -> Optional[tuple[str, Optional[dict], Optional[str]]]:
        operation_task = asyncio.create_task(self._process_attempt(entry, payload, attempt))
        heartbeat_task = asyncio.create_task(self._heartbeat(entry, payload, attempt))
        try:
            done, _pending = await asyncio.wait(
                {operation_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
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

    async def _process_attempt(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: int,
    ) -> Optional[tuple[str, Optional[dict], Optional[str]]]:
        try:
            result = await self.crawl(payload)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            error_message = str(error)
            if attempt >= self.queue.settings.max_attempts:
                await self.queue.complete(entry, payload, error=error_message)
                return "failed", None, error_message
            else:
                logger.warning(
                    "Crawl job %s attempt %s/%s failed; retaining it for lease-based retry: %s",
                    entry.task_id,
                    attempt,
                    self.queue.settings.max_attempts,
                    error_message,
                )
                await self.queue.mark_retry(entry, payload, self.consumer, attempt, error_message)
            return

        await self.queue.complete(entry, payload, result=result)
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

    async def _heartbeat(self, entry: CrawlJobEntry, payload: dict, attempt: int) -> None:
        while True:
            await asyncio.sleep(self.queue.settings.heartbeat_seconds)
            await self.queue.heartbeat(entry, payload, self.consumer, attempt)

    async def _notify(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        status: str,
        *,
        result: Optional[dict] = None,
        error: Optional[str] = None,
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
    worker = CrawlJobWorker(queue, config, consumer)
    try:
        await worker.run()
    finally:
        await close_all()
        await redis.aclose()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
