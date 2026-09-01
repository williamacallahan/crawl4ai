"""Shutdown behavior for Docker's bounded background work queue."""

import asyncio

import pytest

from deploy.docker.work_queue import WorkQueue


class _FailingReleaseRedis:
    """eval() succeeds for acquire and raises for release, like a Redis blip."""

    def __init__(self) -> None:
        self.release_attempts = 0

    async def eval(self, script: str, *args) -> int:
        if "release-principal-quota" in script:
            self.release_attempts += 1
            raise ConnectionError("redis gone")
        return 1


@pytest.mark.asyncio
async def test_worker_survives_release_failure() -> None:
    redis = _FailingReleaseRedis()
    queue = WorkQueue(workers=1, per_principal=1, redis=redis)
    done = asyncio.Event()

    async def first() -> None:
        pass

    async def second() -> None:
        done.set()

    await queue.start()
    worker = queue._tasks[0]
    await queue.submit(first, principal="principal")
    await queue.submit(second, principal="principal")
    # The second job only runs if the worker survived the failed release of
    # the first job's quota claim.
    await asyncio.wait_for(done.wait(), timeout=1)
    assert queue._q is not None
    await asyncio.wait_for(queue._q.join(), timeout=1)

    assert not worker.done()
    assert redis.release_attempts == 2
    await queue.stop()


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
