"""Dedicated Redis Stream consumer for durable ``/crawl/job`` execution."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from collections.abc import Awaitable, Callable
from typing import Any

from archive_budget import ArchiveBudgetSpent
from crawl_job_consumers import reap_idle_consumers
from crawl_job_queue import (
    CrawlJobAttempt,
    CrawlJobEntry,
    CrawlJobLeaseLost,
    CrawlJobQueue,
)
from crawler_pool import close_all, janitor, use_worker_admission_wait
from egress_proxy import start_pinning_proxy, stop_pinning_proxy
from fastapi import HTTPException, status
from redis import asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis_config import RESILIENT_CLIENT_KWARGS, build_redis_url
from utils import correlated_error, load_config, setup_logging
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
        self._next_consumer_reap = 0.0

    async def run(self) -> None:
        # A worker started during a Redis outage waits in-process: exiting
        # would leave supervisord's growing restart delay between it and the
        # moment Redis returns.
        delay = 1.0
        while True:
            try:
                await self.queue.ensure_group()
                break
            except (RedisConnectionError, RedisTimeoutError) as error:
                logger.warning("Crawl job worker waiting for Redis: %r", error)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        crawl_loop = asyncio.create_task(self._crawl_loop())
        outbox_loop = asyncio.create_task(self._outbox_loop())
        try:
            await asyncio.gather(crawl_loop, outbox_loop)
        finally:
            for task in (crawl_loop, outbox_loop):
                task.cancel()
            await asyncio.gather(crawl_loop, outbox_loop, return_exceptions=True)

    async def _crawl_loop(self) -> None:
        while True:
            try:
                entries = await self.queue.claim_stale(self.consumer)
                await self._deliver_pending_webhooks()
                if not entries:
                    await self._reap_idle_consumers()
                    entries = await self.queue.read_new(self.consumer)
                for entry in entries:
                    await self.process(entry)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Crawl job worker loop failed; retaining pending work for retry")
                await asyncio.sleep(1)

    async def _reap_idle_consumers(self) -> None:
        loop = asyncio.get_running_loop()
        if loop.time() < self._next_consumer_reap:
            return
        # Set first: a failing pass then costs one loop iteration per lease.
        self._next_consumer_reap = loop.time() + self.queue.settings.lease_seconds
        await reap_idle_consumers(self.queue)

    async def _outbox_loop(self) -> None:
        while True:
            try:
                entries = await self.queue.read_outbox(f"{self.consumer}:outbox")
                for stream_id, event in entries:
                    await self.process_outbox(stream_id, event, consumer=f"{self.consumer}:outbox")
                if not entries:
                    await asyncio.sleep(max(self.queue.settings.read_block_ms / 1000, 0.01))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Webhook outbox loop failed; retaining pending delivery for retry")
                await asyncio.sleep(1)

    async def process_outbox(self, stream_id: str, event: dict, *, consumer: str | None = None) -> None:
        """Deliver a committed terminal event while renewing its fenced lease."""
        consumer = consumer or self.consumer
        async def reserve_attempt() -> int:
            attempt = await self.queue.reserve_outbox_attempt(stream_id, consumer)
            if not attempt:
                raise CrawlJobLeaseLost(f"lost webhook lease for {stream_id}")
            return attempt

        async def heartbeat() -> None:
            loop = asyncio.get_running_loop()
            last_renewal = loop.time()
            while True:
                try:
                    # Both a Redis call and the sleep after an early failure
                    # must fit inside the last confirmed lease.
                    async with asyncio.timeout_at(last_renewal + self.queue.settings.lease_seconds):
                        await asyncio.sleep(self.queue.settings.heartbeat_seconds)
                        renewal_started = loop.time()
                        await self.queue.heartbeat_outbox(stream_id, consumer)
                    last_renewal = renewal_started
                except CrawlJobLeaseLost:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # An outage must not let a sender run indefinitely after
                    # its last confirmed renewal; another worker can reclaim.
                    if loop.time() - last_renewal >= self.queue.settings.lease_seconds:
                        raise CrawlJobLeaseLost(f"webhook lease renewal expired for {stream_id}")
                    logger.warning("Webhook heartbeat failed for %s", stream_id, exc_info=True)

        delivery = asyncio.create_task(self.webhook_service.notify_job_completion(
            task_id=event["task_id"],
            task_type="crawl",
            status=event["status"],
            urls=event["urls"],
            webhook_config=event.get("webhook_config"),
            result=event.get("result"),
            error=event.get("error"),
            event_id=event["event_id"],
            before_attempt=reserve_attempt,
        ))
        renewal = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait({delivery, renewal}, return_when=asyncio.FIRST_COMPLETED)
            if renewal in done:
                renewal.result()
                raise CrawlJobLeaseLost(f"webhook heartbeat stopped for {stream_id}")
            outcome = delivery.result()
            if not isinstance(outcome, bool):
                raise TypeError("webhook delivery did not return a terminal outcome")
            if outcome:
                await self.queue.finish_outbox(stream_id, consumer)
            else:
                await self.queue.fail_outbox(stream_id, consumer)
                logger.warning("Webhook %s undelivered after its retry budget; dropped", stream_id)
        except CrawlJobLeaseLost:
            logger.warning("Webhook worker lost lease for %s; leaving it pending", stream_id)
        finally:
            for task in (delivery, renewal):
                if not task.done():
                    task.cancel()
            await asyncio.gather(delivery, renewal, return_exceptions=True)

    async def _deliver_pending_webhooks(self) -> None:
        for stream_id, event in await self.queue.pending_outbox(self.consumer):
            await self.process_outbox(stream_id, event)

    async def process(self, entry: CrawlJobEntry) -> None:
        await self._process_entry(entry)
        # Completion and outbox reservation are atomic. This path also handles
        # missing payloads and exhausted retries before a possible rollback.
        await self._deliver_pending_webhooks()

    async def _process_entry(self, entry: CrawlJobEntry) -> None:
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
            return

        try:
            await self._run_with_lease(entry, payload, attempt)
        except asyncio.CancelledError:
            logger.info("Crawl worker cancelled while processing %s; leaving Stream entry pending", entry.task_id)
            raise
        except CrawlJobLeaseLost:
            logger.warning("Crawl worker lost the lease for %s; leaving it pending", entry.task_id)
            return
        except Exception:
            logger.exception("Crawl job %s failed outside its retry transaction", entry.task_id)
            return

    async def _run_with_lease(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: CrawlJobAttempt,
    ) -> tuple[str, dict | None, str | None] | None:
        operation_task = asyncio.create_task(self._process_attempt(entry, payload, attempt))
        heartbeat_task = asyncio.create_task(self._heartbeat(entry, payload, attempt))

        async def stop_attempt() -> None:
            for task in (operation_task, heartbeat_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(operation_task, heartbeat_task, return_exceptions=True)

        try:
            done, _pending = await asyncio.wait(
                {operation_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self.queue.settings.max_attempt_seconds,
            )
            if not done:
                # No attempt-owned write may race the timeout release.
                await stop_attempt()
                return await self._release_stalled_attempt(entry, payload, attempt)
            if operation_task in done:
                return operation_task.result()

            heartbeat_task.result()
            raise CrawlJobLeaseLost(f"heartbeat stopped for {entry.task_id}")
        finally:
            await stop_attempt()

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
            return await self.queue.complete(entry, payload, attempt, error=error_message)
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
        except ArchiveBudgetSpent as error:
            # Terminal, never a lease retry: the node's archive.org budget is
            # spent and the poller routes the job through the overflow lane
            # when it sees the code in the stored error.
            error_message = (
                f"archive_budget_spent: {error.detail}; retry after {error.retry_after}s"
            )
            return await self.queue.complete(entry, payload, attempt, error=error_message)
        except (HTTPException, OSError, RuntimeError, TypeError, ValueError) as error:
            # This worker is a separate process: the API's central 500 handler
            # never sees these, so nothing else genericizes them before they
            # reach the task hash /crawl/job/{id} returns and the webhook. A
            # deliberate 4xx detail is already client-facing; a 5xx detail and
            # a bare exception (which routinely names container paths) are not.
            if isinstance(error, HTTPException) and error.status_code < 500:
                error_message = str(error.detail)
            else:
                # 502/503/504 are a failed target, a busy server, or a crawl
                # deadline; a server-side fault behind a 502 was already logged
                # at ERROR by public_crawl_error.
                upstream = isinstance(error, HTTPException) and error.status_code in {
                    status.HTTP_502_BAD_GATEWAY,
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    status.HTTP_504_GATEWAY_TIMEOUT,
                }
                error_message = correlated_error(
                    "Crawl job failed",
                    error,
                    f"crawl job={entry.task_id}",
                    logging.WARNING if upstream else logging.ERROR,
                )
            terminal_input = isinstance(error, HTTPException) and error.status_code in {
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            }
            if terminal_input or attempt.number >= self.queue.settings.max_attempts:
                return await self.queue.complete(entry, payload, attempt, error=error_message)
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

        if result.get("success") is False:
            # Every URL failed. The attempt itself ran, so this is terminal
            # rather than retryable, but the per-URL diagnostics still ship:
            # complete() writes result and error independently.
            error_message = "Every crawled URL failed"
            return await self.queue.complete(entry, payload, attempt, result=result, error=error_message)

        return await self.queue.complete(entry, payload, attempt, result=result)

    async def _crawl(self, payload: dict) -> dict:
        """Load the existing synchronous crawl owner lazily inside the worker."""
        from api import handle_crawl_request

        return await handle_crawl_request(
            urls=payload["urls"],
            browser_config=payload["browser_config"],
            crawler_config=payload["crawler_config"],
            config=self.config,
            result_fields=payload.get("result_fields"),
            redis=self.queue.redis,
        )

    async def _heartbeat(
        self,
        entry: CrawlJobEntry,
        payload: dict,
        attempt: CrawlJobAttempt,
    ) -> None:
        while True:
            await asyncio.sleep(self.queue.settings.heartbeat_seconds)
            try:
                await self.queue.heartbeat(entry, payload, self.consumer, attempt)
            except CrawlJobLeaseLost:
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Heartbeat renewal failed for crawl job %s; will retry next tick",
                    entry.task_id,
                    exc_info=True,
                )

async def run_worker() -> None:
    config = load_config()
    setup_logging(config)
    redis = aioredis.from_url(
        build_redis_url(config), decode_responses=False, **RESILIENT_CLIENT_KWARGS
    )
    queue = CrawlJobQueue(redis, config)
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    use_worker_admission_wait()
    proxy = None
    # Each worker process owns a private browser pool (the pools in
    # crawler_pool are module globals, and supervisord runs numprocs=2 of
    # this program beside gunicorn). Without its own janitor nothing here
    # ever closes an idle browser, so a worker's pool only grows - up to
    # crawler.pool.max_browser_instances of them - and that memory counts
    # against the same container limit that gunicorn's get_crawler checks.
    # It is also invisible to /monitor/*, which reads the gunicorn process.
    pool_janitor = asyncio.create_task(janitor())
    try:
        proxy = await start_pinning_proxy()
        worker = CrawlJobWorker(queue, config, consumer)
        await worker.run()
    finally:
        pool_janitor.cancel()
        await asyncio.gather(pool_janitor, return_exceptions=True)
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
