"""Durable retry accounting and cancellation at the webhook service boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from crawl_job_queue import CrawlJobLeaseLost
from crawl_job_worker import CrawlJobWorker
from webhook import WebhookDeliveryService


def service(**retry):
    return WebhookDeliveryService({"webhooks": {"retry": {
        "max_attempts": 3, "initial_delay_ms": 10, **retry,
    }}})


def event():
    return {
        "task_id": "crawl_test", "event_id": "crawl_test:terminal",
        "status": "completed", "urls": ["https://example.com"],
        "webhook_config": {"webhook_url": "https://callback.example"},
        "result": {"success": True}, "error": None,
    }


def queue():
    return SimpleNamespace(
        settings=SimpleNamespace(heartbeat_seconds=0.001, lease_seconds=1),
        reserve_outbox_attempt=AsyncMock(return_value=1),
        heartbeat_outbox=AsyncMock(), finish_outbox=AsyncMock(), fail_outbox=AsyncMock(),
    )


def test_retry_budget_and_backoff_survive_sender_restart(monkeypatch):
    async def scenario():
        attempts = 0

        async def reserve():
            nonlocal attempts
            attempts += 1
            return attempts

        first = service()
        first._deliver = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await first.send_webhook("https://callback.example", {}, before_attempt=reserve)
        assert attempts == 1

        restarted = service()
        restarted._deliver = AsyncMock(side_effect=[503, 200])
        sleeps = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleeps)
        assert await restarted.send_webhook(
            "https://callback.example", {}, event_id="stable-event", before_attempt=reserve,
        ) is True
        assert attempts == 3
        assert restarted._deliver.await_count == 2
        sleeps.assert_awaited_once_with(0.02)
        for call in restarted._deliver.await_args_list:
            assert call.args[1]["event_id"] == "stable-event"
            assert call.args[2]["X-Crawl4AI-Event-ID"] == "stable-event"

    asyncio.run(scenario())


@pytest.mark.parametrize("reserved", [-1, 0, 4])
def test_lost_lease_or_spent_budget_never_sends(reserved):
    async def scenario():
        sender = service()
        sender._deliver = AsyncMock()
        assert await sender.send_webhook(
            "https://callback.example", {}, before_attempt=AsyncMock(return_value=reserved),
        ) is False
        sender._deliver.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("config", [{}, {"webhooks": {
    "enabled": False, "default_url": "https://callback.example",
}}])
@pytest.mark.parametrize("event_id", [None, "crawl_test:terminal"])
def test_notifications_without_delivery_do_not_ack_durable_events(config, event_id):
    async def scenario():
        sender = WebhookDeliveryService(config)
        sender._deliver = AsyncMock()
        outcome = await sender.notify_job_completion(
            "task", "crawl", "completed", [], None, event_id=event_id,
        )
        assert outcome is (event_id is None)
        sender._deliver.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("status, attempts", [(200, 1), (400, 1), (503, 3)])
def test_outbox_finishes_only_after_success(status, attempts):
    async def scenario():
        pending = queue()
        pending.reserve_outbox_attempt.side_effect = range(1, 4)
        sender = service()
        sender._deliver = AsyncMock(return_value=status)
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        await worker.process_outbox("1-0", event())
        assert pending.reserve_outbox_attempt.await_count == attempts
        assert sender._deliver.await_count == attempts
        if status == 200:
            pending.finish_outbox.assert_awaited_once_with("1-0", "worker")
            pending.fail_outbox.assert_not_awaited()
        else:
            pending.finish_outbox.assert_not_awaited()
            pending.fail_outbox.assert_awaited_once_with("1-0", "worker")

    asyncio.run(scenario())


def test_dns_failure_exhausts_retry_budget_before_dropping(monkeypatch):
    async def scenario():
        import egress_broker
        from egress_broker import TargetUnresolvable

        pins = []

        def boom(url):
            pins.append(url)
            raise TargetUnresolvable()

        monkeypatch.setattr(egress_broker, "resolve_and_pin", boom)
        pending = queue()
        pending.reserve_outbox_attempt.side_effect = range(1, 4)
        sender = service()
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        await worker.process_outbox("1-0", event())
        assert len(pins) == 3
        assert pending.reserve_outbox_attempt.await_count == 3
        pending.finish_outbox.assert_not_awaited()
        pending.fail_outbox.assert_awaited_once_with("1-0", "worker")

    asyncio.run(scenario())


def test_policy_block_drops_outbox_after_single_attempt(monkeypatch):
    async def scenario():
        import egress_broker
        from egress_broker import EgressBlocked

        pins = []

        def boom(url):
            pins.append(url)
            raise EgressBlocked()

        monkeypatch.setattr(egress_broker, "resolve_and_pin", boom)
        pending = queue()
        pending.reserve_outbox_attempt.side_effect = range(1, 4)
        sender = service()
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        await worker.process_outbox("1-0", event())
        assert len(pins) == 1
        assert pending.reserve_outbox_attempt.await_count == 1
        pending.finish_outbox.assert_not_awaited()
        pending.fail_outbox.assert_awaited_once_with("1-0", "worker")

    asyncio.run(scenario())


def test_outbox_heartbeat_loss_cancels_active_send():
    async def scenario():
        pending = queue()
        pending.heartbeat_outbox.side_effect = CrawlJobLeaseLost("reclaimed")
        cancelled = asyncio.Event()
        sender = service()

        async def blocked_send(*_args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        sender._deliver = blocked_send
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        await asyncio.wait_for(worker.process_outbox("1-0", event()), 1)
        assert cancelled.is_set()
        pending.finish_outbox.assert_not_awaited()
        pending.fail_outbox.assert_not_awaited()

    asyncio.run(scenario())


def test_outbox_send_stops_at_lease_expiry_while_renewal_hangs():
    # A renewal stuck in the Redis client's retry loop must not let the send
    # outlive the lease another worker can now reclaim.
    async def scenario():
        pending = queue()
        pending.settings.lease_seconds = 0.2

        async def hung_renewal(*_args):
            await asyncio.Event().wait()

        pending.heartbeat_outbox.side_effect = hung_renewal
        cancelled = asyncio.Event()
        sender = service()

        async def blocked_send(*_args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        sender._deliver = blocked_send
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        await asyncio.wait_for(worker.process_outbox("1-0", event()), 1)
        assert cancelled.is_set()
        pending.finish_outbox.assert_not_awaited()

    asyncio.run(scenario())


def test_reservation_error_keeps_outbox_pending_without_sending():
    async def scenario():
        pending = queue()
        pending.reserve_outbox_attempt.side_effect = RuntimeError("Redis unavailable")
        sender = service()
        sender._deliver = AsyncMock()
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        with pytest.raises(RuntimeError, match="Redis unavailable"):
            await worker.process_outbox("1-0", event())
        sender._deliver.assert_not_awaited()
        pending.finish_outbox.assert_not_awaited()
        pending.fail_outbox.assert_not_awaited()

    asyncio.run(scenario())


def test_outbox_retry_sleep_cannot_outlive_last_confirmed_lease():
    async def scenario():
        pending = queue()
        pending.settings.lease_seconds = 0.4
        pending.settings.heartbeat_seconds = 0.25

        async def failed_renewal(*_args):
            await asyncio.sleep(0.05)
            raise ConnectionError("Redis connection retries exhausted")

        pending.heartbeat_outbox.side_effect = failed_renewal
        cancelled = asyncio.Event()
        sender = service()

        async def blocked_send(*_args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        sender._deliver = blocked_send
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        task = asyncio.create_task(worker.process_outbox("1-0", event()))
        try:
            # Expiry is at 0.4s, before the next full heartbeat sleep at 0.55s.
            await asyncio.wait_for(cancelled.wait(), 0.5)
            await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        pending.finish_outbox.assert_not_awaited()
        pending.fail_outbox.assert_not_awaited()

    asyncio.run(scenario())


def test_outbox_reservation_fence_loss_keeps_event_pending():
    async def scenario():
        pending = queue()
        pending.reserve_outbox_attempt.return_value = 0
        sender = service()
        sender._deliver = AsyncMock()
        worker = CrawlJobWorker(pending, {}, "worker", webhook_service=sender)
        await worker.process_outbox("1-0", event())
        sender._deliver.assert_not_awaited()
        pending.finish_outbox.assert_not_awaited()
        pending.fail_outbox.assert_not_awaited()

    asyncio.run(scenario())
