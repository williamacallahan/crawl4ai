"""Shutdown behavior for Docker's bounded background work queue."""

import asyncio

import pytest

from deploy.docker.work_queue import WorkQueue


@pytest.mark.asyncio
async def test_stop_awaits_cancelled_worker() -> None:
    queue = WorkQueue(workers=1, per_principal=1)
    started = asyncio.Event()

    async def job() -> None:
        started.set()
        await asyncio.Event().wait()

    await queue.start()
    worker = queue._tasks[0]
    await queue.submit(job, principal="principal")
    await asyncio.wait_for(started.wait(), timeout=1)

    await queue.stop()

    assert worker.done()
    assert worker.cancelled()
    assert queue._counts == {}
