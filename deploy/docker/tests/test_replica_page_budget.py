"""Real file leases bound page admission across independent worker processes."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time

from fastapi import HTTPException
import pytest

import crawler_pool


DOCKER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DOCKER_DIR.parents[1]
HOLDER_SCRIPT = """
import asyncio
from pathlib import Path
import sys

from crawler_pool import ReplicaPageBudget

async def main():
    budget = ReplicaPageBudget(max_pages=8, directory=Path(sys.argv[1]))
    leases = [await budget.acquire(wait_seconds=1.0) for _ in range(4)]
    Path(sys.argv[2]).write_text("ready")
    try:
        await asyncio.Event().wait()
    finally:
        for lease in leases:
            lease.close()

asyncio.run(main())
"""


def _assert_capacity_error(error):
    assert isinstance(error, HTTPException)
    assert isinstance(error, RuntimeError)
    assert error.status_code == 503
    assert float(error.headers["Retry-After"]) > 0


async def _wait_until_ready(process, marker, output):
    deadline = time.monotonic() + 20
    while not marker.exists():
        assert process.poll() is None, output.read_text()[-3000:]
        assert time.monotonic() < deadline, "page-budget holder did not become ready"
        await asyncio.sleep(0.02)


def _lease_files(directory):
    return {
        path.name: path.stat().st_ino
        for path in directory.iterdir()
        if path.is_file()
    }


@pytest.mark.asyncio
async def test_eight_slots_are_shared_across_processes_and_recover_after_kill(tmp_path):
    directory = tmp_path / "budget"
    budget = crawler_pool.ReplicaPageBudget(max_pages=8, directory=directory)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(DOCKER_DIR), str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    )
    processes = []
    recovered = []
    try:
        readiness = []
        for index in range(2):
            marker = tmp_path / f"holder-{index}.ready"
            output = tmp_path / f"holder-{index}.log"
            with output.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-c", HOLDER_SCRIPT, str(directory), str(marker)],
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            processes.append(process)
            readiness.append(_wait_until_ready(process, marker, output))
        await asyncio.gather(*readiness)
        assert all(process.poll() is None for process in processes)

        started = time.monotonic()
        with pytest.raises(crawler_pool.BrowserCapacityError) as refusal:
            await budget.acquire(wait_seconds=0.2)
        elapsed = time.monotonic() - started
        _assert_capacity_error(refusal.value)
        assert 0.15 <= elapsed < 0.5, f"admission waited {elapsed:.3f}s"

        files_before = _lease_files(directory)
        assert files_before, "the shared budget must have persistent lease files"
        processes[0].kill()
        processes[0].wait(timeout=5)
        assert processes[1].poll() is None

        for _ in range(4):
            recovered.append(await budget.acquire(wait_seconds=0.3))
        with pytest.raises(crawler_pool.BrowserCapacityError) as refusal:
            await budget.acquire(wait_seconds=0)
        _assert_capacity_error(refusal.value)
        assert _lease_files(directory) == files_before
    finally:
        for lease in recovered:
            lease.close()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


@pytest.mark.asyncio
async def test_closing_a_lease_admits_an_existing_waiter(tmp_path):
    first = crawler_pool.ReplicaPageBudget(max_pages=1, directory=tmp_path)
    second = crawler_pool.ReplicaPageBudget(max_pages=1, directory=str(tmp_path))
    lease = await first.acquire(wait_seconds=0)
    acquired = None
    waiter = asyncio.create_task(second.acquire(wait_seconds=1))
    try:
        await asyncio.sleep(0.03)
        assert not waiter.done()
        lease.close()
        acquired = await asyncio.wait_for(waiter, timeout=0.5)
        assert not acquired.closed
        with pytest.raises(crawler_pool.BrowserCapacityError):
            await first.acquire(wait_seconds=0)
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        lease.close()
        if acquired is not None:
            acquired.close()

    replacement = await first.acquire(wait_seconds=0)
    replacement.close()


@pytest.mark.asyncio
async def test_cancelling_a_waiter_does_not_consume_capacity(tmp_path):
    budget = crawler_pool.ReplicaPageBudget(max_pages=8, directory=tmp_path)
    leases = []
    waiter = None
    try:
        for _ in range(8):
            leases.append(await budget.acquire(wait_seconds=0))
        waiter = asyncio.create_task(budget.acquire(wait_seconds=5))
        await asyncio.sleep(0.03)
        assert not waiter.done()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        for lease in leases:
            lease.close()
        leases.clear()

        for _ in range(8):
            leases.append(await budget.acquire(wait_seconds=0.3))
        with pytest.raises(crawler_pool.BrowserCapacityError):
            await budget.acquire(wait_seconds=0)
    finally:
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        for lease in leases:
            lease.close()
