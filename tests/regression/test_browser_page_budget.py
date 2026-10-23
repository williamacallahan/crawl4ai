"""Deadline behavior without launching a browser or mocking renderer internals."""

import asyncio
import json

import pytest

from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.browser_budget import page_deadline, within_page_budget


@pytest.mark.asyncio
async def test_budget_cancels_a_blocked_operation():
    stopped = asyncio.Event()

    async def blocked():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    token = page_deadline.set(asyncio.get_running_loop().time() + 0.05)
    task = asyncio.create_task(within_page_budget(blocked()))
    try:
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "The page budget must bound work that never completes"
        with pytest.raises(asyncio.TimeoutError):
            await task
        assert stopped.is_set(), "Timed-out work must be cancelled before returning"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        page_deadline.reset(token)


@pytest.mark.asyncio
async def test_sequential_operations_share_an_absolute_deadline():
    loop = asyncio.get_running_loop()
    token = page_deadline.set(loop.time() + 0.05)
    second_started = asyncio.Event()

    async def second_operation():
        second_started.set()
        return "finished"

    try:
        assert await within_page_budget(asyncio.sleep(0, result="first")) == "first"
        # Time between renderer calls also spends the crawl's budget.
        await asyncio.sleep(0.06)
        with pytest.raises(asyncio.TimeoutError):
            await within_page_budget(second_operation())
        assert not second_started.is_set()
    finally:
        page_deadline.reset(token)


@pytest.mark.asyncio
async def test_concurrent_crawls_keep_independent_deadlines():
    short_started = asyncio.Event()
    long_started = asyncio.Event()
    release_long = asyncio.Event()

    async def crawl(seconds, started, release):
        token = page_deadline.set(asyncio.get_running_loop().time() + seconds)

        async def operation():
            started.set()
            await release.wait()
            return "completed"

        try:
            return await within_page_budget(operation())
        finally:
            page_deadline.reset(token)

    parent_deadline = page_deadline.get()
    short = asyncio.create_task(crawl(0.05, short_started, asyncio.Event()))
    long = asyncio.create_task(crawl(2, long_started, release_long))
    try:
        await asyncio.wait_for(asyncio.gather(short_started.wait(), long_started.wait()), 1)
        done, _ = await asyncio.wait({short}, timeout=1)
        assert short in done, "The shorter crawl must exhaust its own budget"
        with pytest.raises(asyncio.TimeoutError):
            await short
        assert not long.done(), "One crawl timing out must not cancel another"
        release_long.set()
        assert await asyncio.wait_for(long, 1) == "completed"
        assert page_deadline.get() == parent_deadline
    finally:
        for task in (short, long):
            task.cancel()
        await asyncio.gather(short, long, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_deadline", [False, True])
async def test_caller_cancellation_reaches_the_operation(has_deadline):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    deadline = asyncio.get_running_loop().time() + 10 if has_deadline else None
    token = page_deadline.set(deadline)
    task = asyncio.create_task(within_page_budget(operation()))
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        page_deadline.reset(token)


@pytest.mark.asyncio
async def test_operation_without_a_deadline_can_complete():
    token = page_deadline.set(None)
    try:
        assert await within_page_budget(asyncio.sleep(0, result="content")) == "content"
    finally:
        page_deadline.reset(token)


def test_page_limits_survive_serialization_and_clone():
    original = CrawlerRunConfig(max_iframes=3, page_termination_timeout=0.25)
    serialized = json.loads(json.dumps(original.dump()))
    restored = CrawlerRunConfig.load(serialized)
    for config in (original.clone(), restored):
        assert config.max_iframes == 3
        assert config.page_termination_timeout == 0.25
    updated = original.clone(max_iframes=1, page_termination_timeout=0.5)
    assert updated.max_iframes == 1
    assert updated.page_termination_timeout == 0.5
    assert original.max_iframes == 3
    assert original.page_termination_timeout == 0.25


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3", None])
def test_iframe_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_iframes"):
        CrawlerRunConfig(max_iframes=value)


@pytest.mark.parametrize(
    "value", [0, -1, True, False, "1", None, float("nan"), float("inf"), -float("inf")]
)
def test_termination_timeout_requires_a_positive_number(value):
    with pytest.raises(ValueError, match="page_termination_timeout"):
        CrawlerRunConfig(page_termination_timeout=value)
