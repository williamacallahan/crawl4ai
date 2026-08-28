from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import verify_rollout


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = Path(__file__).resolve().parents[1]


def test_monitor_records_and_rejects_a_public_failure(monkeypatch, tmp_path):
    health = iter(
        [
            {
                "status": "ok",
                "instance": "old",
                "revision": "old",
                "components": {"api": "ready", "redis": "ready"},
            },
            TimeoutError(),
        ]
    )
    monkeypatch.setattr(verify_rollout.time, "sleep", lambda _seconds: None)
    evidence = tmp_path / "monitor.jsonl"
    armed = tmp_path / "armed"

    def read_json(_url, _api_key):
        if "application.runtimeServiceState" in _url:
            return {
                "tasks": [
                    {
                        "status": {"state": "running", "containerId": "old"},
                        "addresses": ["10.0.0.1/24"],
                    }
                ],
                "traefik": {
                    "services": []
                },
            }
        result = next(health)
        if isinstance(result, Exception):
            raise result
        return result

    with pytest.raises(RuntimeError, match="failed request"):
        verify_rollout.monitor_public_health(
            health_url="https://crawl.example/health",
            expected_replicas=1,
            evidence_path=evidence,
            armed_path=armed,
            stop_path=tmp_path / "stop",
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="application",
            read_json=read_json,
        )

    assert armed.exists()
    assert [json.loads(line)["ok"] for line in evidence.read_text().splitlines()] == [
        True,
        False,
    ]
    with pytest.raises(RuntimeError, match="did not remain successful"):
        verify_rollout.verify_monitor_evidence(
            evidence,
            frozenset({"replacement"}),
            frozenset({"replacement"}),
            frozenset({"10.0.0.2"}),
        )


def test_monitor_evidence_proves_predecessor_withdrawal_and_replacement_coverage(
    tmp_path,
):
    evidence = tmp_path / "monitor.jsonl"
    def task(container, address, state="running"):
        return {
            "status": {"state": state, "containerId": container},
            "addresses": [f"{address}/24"],
        }

    old = [task("old-a", "10.0.0.1"), task("old-b", "10.0.0.2"), task("old-c", "10.0.0.3")]
    final = [task("new-a", "10.0.0.4"), task("new-b", "10.0.0.5"), task("new-c", "10.0.0.6")]
    samples = [
        {
            "ok": True,
            "instance": "old-a",
            "tasks": old,
            "upAddresses": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        },
        {
            "ok": True,
            "instance": "old-b",
            "tasks": [*old, final[0]],
            "upAddresses": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        },
        {
            "ok": True,
            "instance": "old-c",
            "tasks": [*old, final[0]],
            "upAddresses": ["10.0.0.2", "10.0.0.3", "10.0.0.4"],
        },
        {
            "ok": True,
            "instance": "new-a",
            "tasks": [old[1], old[2], final[0], final[1]],
            "upAddresses": ["10.0.0.3", "10.0.0.4"],
        },
        {
            "ok": True,
            "instance": "new-b",
            "tasks": [old[1], old[2], final[0], final[1]],
            "upAddresses": ["10.0.0.3", "10.0.0.4", "10.0.0.5"],
        },
        {
            "ok": True,
            "instance": "new-c",
            "tasks": [old[2], *final],
            "upAddresses": ["10.0.0.4", "10.0.0.5"],
        },
        {
            "ok": True,
            "instance": "new-c",
            "tasks": [old[2], *final],
            "upAddresses": ["10.0.0.4", "10.0.0.5", "10.0.0.6"],
        },
    ]
    evidence.write_text("".join(json.dumps(sample) + "\n" for sample in samples))

    verify_rollout.verify_monitor_evidence(
        evidence,
        frozenset({"new-a", "new-b", "new-c"}),
        frozenset({"new-a", "new-b", "new-c"}),
        frozenset({"10.0.0.4", "10.0.0.5", "10.0.0.6"}),
    )


def test_deploy_workflow_uses_only_native_swarm_readiness():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-deploy.yml").read_text()
    monitor = workflow.index("verify_rollout.py monitor")
    readiness = workflow.index(
        '"readinessCheckSwarm":{"Path":"/health/route","Interval":500000000,'
        '"UnhealthyInterval":250000000,"Timeout":400000000,"Status":200}'
    )
    deploy = workflow.index("/api/application.deploy", readiness)
    final_proof = workflow.index("python3 deploy/docker/verify_rollout.py\n", deploy)

    assert monitor < readiness < deploy < final_proof
    assert '"replicas":3' in workflow
    assert '"placementSwarm":{"MaxReplicas":1}' in workflow
    assert '"stopGracePeriodSwarm":390000000000' in workflow


def test_pid1_withdrawal_delay_outlives_routing_health_removal():
    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text()
    delay_seconds = int(
        re.search(r"CRAWL4AI_DRAIN_DELAY_SECONDS:-([0-9]+)", entrypoint).group(1)
    )
    readiness = verify_rollout.CRAWL_READINESS_CHECK

    assert delay_seconds * 1_000_000_000 > readiness["Interval"] + readiness["Timeout"]


@pytest.mark.asyncio
async def test_startup_redis_failure_leaves_routing_unadmitted(
    server_module, monkeypatch
):
    class UnavailableRedis:
        async def ping(self):
            raise ConnectionError("unavailable")

    class WorkQueue:
        async def start(self):
            return None

        async def stop(self):
            return None

    class Monitor:
        async def cleanup(self):
            return None

    async def no_op(*_args, **_kwargs):
        return None

    import artifacts
    import egress_proxy
    import monitor
    import work_queue

    monkeypatch.setattr(server_module, "_resolve_auth", lambda: None)
    monkeypatch.setattr(server_module, "_artifact_janitor", no_op)
    monkeypatch.setattr(server_module, "close_all", no_op)
    monkeypatch.setattr(server_module, "redis", UnavailableRedis())
    monkeypatch.setattr(artifacts, "init_store", lambda: None)
    monkeypatch.setattr(egress_proxy, "start_pinning_proxy", no_op)
    monkeypatch.setattr(egress_proxy, "stop_pinning_proxy", no_op)
    monkeypatch.setattr(work_queue, "WorkQueue", lambda **_kwargs: WorkQueue())
    monkeypatch.setattr(work_queue, "set_job_queue", lambda _queue: None)
    monkeypatch.setattr(monitor, "get_monitor", lambda: Monitor())

    with pytest.raises(ConnectionError, match="unavailable"):
        async with server_module.lifespan(server_module.app):
            pass

    assert server_module.app.state.readiness_checks_active is False


@pytest.mark.asyncio
async def test_later_redis_loss_does_not_withdraw_an_admitted_task(
    server_module, monkeypatch
):
    class UnavailableRedis:
        calls = 0

        async def ping(self):
            self.calls += 1
            raise ConnectionError("unavailable")

    redis = UnavailableRedis()
    monkeypatch.setattr(server_module, "redis", redis)
    server_module.app.state.readiness_checks_active = True
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server_module.app),
            base_url="http://test",
        ) as client:
            routing = await client.get("/health/route")
            public = await client.get("/health")
    finally:
        server_module.app.state.readiness_checks_active = False

    assert routing.status_code == 200
    assert routing.json()["components"] == {"api": "ready"}
    assert public.status_code == 503
    assert public.json()["components"]["redis"] == "unavailable"
    assert redis.calls == 1


@pytest.mark.asyncio
async def test_routing_health_withdraws_while_public_health_keeps_draining_task_live(
    server_module, monkeypatch, tmp_path
):
    class ReadyRedis:
        async def ping(self):
            return True

    drain_path = tmp_path / "crawl4ai-draining"
    monkeypatch.setattr(server_module, "DRAIN_PATH", drain_path)
    monkeypatch.setattr(server_module, "redis", ReadyRedis())

    server_module.app.state.readiness_checks_active = False
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server_module.app),
            base_url="http://test",
        ) as client:
            assert (await client.get("/health")).status_code == 503
            assert (await client.get("/health/route")).status_code == 503

            server_module.app.state.readiness_checks_active = True
            ready = await client.get("/health")
            assert ready.status_code == 200
            assert ready.json()["components"] == {
                "api": "ready",
                "redis": "ready",
            }

            drain_path.touch()
            draining = await client.get("/health")
            assert draining.status_code == 200
            assert draining.json()["components"]["api"] == "draining"
            assert (await client.get("/health/route")).status_code == 503
    finally:
        server_module.app.state.readiness_checks_active = False


def test_entrypoint_marks_drain_before_terminating_supervisord(tmp_path):
    events = tmp_path / "events"
    child_pid = tmp_path / "child.pid"
    drain_path = tmp_path / "draining"
    supervisord = tmp_path / "supervisord"
    supervisord.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, signal\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid()))\n"
        "def stop(*_args):\n"
        f"    drain = pathlib.Path({str(drain_path)!r}).exists()\n"
        "    event = 'term-after-drain' if drain else 'term-before-drain'\n"
        f"    pathlib.Path({str(events)!r}).write_text(event)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        "while True:\n"
        "    signal.pause()\n"
    )
    supervisord.chmod(0o755)
    (tmp_path / "config.yml").write_text("security:\n  jwt_enabled: false\n")
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "REDIS_HOST": "external-redis",
        "CRAWL4AI_API_TOKEN": "test-token",
        "CRAWL4AI_DRAIN_PATH": str(drain_path),
        "CRAWL4AI_DRAIN_DELAY_SECONDS": "0.2",
    }
    process = subprocess.Popen(
        ["bash", str(DOCKER_DIR / "entrypoint.sh")], cwd=tmp_path, env=environment
    )
    try:
        for _ in range(50):
            if child_pid.exists():
                break
            time.sleep(0.02)
        assert child_pid.exists()

        process.send_signal(signal.SIGTERM)
        for _ in range(50):
            if drain_path.exists():
                break
            time.sleep(0.01)
        assert drain_path.exists()
        assert process.poll() is None
        os.kill(int(child_pid.read_text()), 0)
        process.send_signal(signal.SIGTERM)
        assert process.poll() is None

        assert process.wait(timeout=5) == 0
        assert events.read_text() == "term-after-drain"
        with pytest.raises(ProcessLookupError):
            os.kill(int(child_pid.read_text()), 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
