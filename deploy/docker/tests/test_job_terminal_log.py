import asyncio

from redis import asyncio as aioredis

# Fixtures: a reachable Redis, test-scoped key cleanup, and the queue settings.
from test_crawl_job_queue_redis import public_seed_validation, queue_config, redis_url  # noqa: F401


def test_finished_job_logs_one_terminal_record_with_its_trace(redis_url, caplog):
    from crawl_job_queue import CrawlJobQueue

    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

    async def exercise():
        client = aioredis.from_url(redis_url)
        queue = CrawlJobQueue(client, queue_config())
        try:
            await queue.ensure_group()
            task_id = await queue.enqueue(["https://traced.example"], {}, {}, None, None,
                                          owner="alice", trace_id=trace_id)
            entry = (await queue.read_new("worker-a"))[0]
            payload = await queue.load_payload(task_id)
            attempt = await queue.start_attempt(entry, payload, "worker-a")
            with caplog.at_level("INFO", logger="crawl_job_queue"):
                await queue.complete(entry, payload, attempt, error="Every crawled URL failed")
            return task_id
        finally:
            await client.aclose()

    task_id = asyncio.run(exercise())
    terminal = [r.getMessage() for r in caplog.records if "event=job_terminal" in r.getMessage()]
    assert len(terminal) == 1
    assert "\n" not in terminal[0]
    assert f"task_id={task_id}" in terminal[0]
    assert "status=failed" in terminal[0]
    assert "attempt=1" in terminal[0]
    assert "duration_ms=" in terminal[0]
    assert f"trace_id={trace_id}" in terminal[0]
