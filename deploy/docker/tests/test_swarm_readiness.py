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


def _health(instance="current", revision="target"):
    return {
        "status": "ok",
        "instance": instance,
        "revision": revision,
        "components": {"api": "ready", "redis": "ready"},
    }


def _crawl_service(image: str, revision: str) -> dict:
    update = {
        "Parallelism": 1,
        "Delay": verify_rollout.CRAWL_UPDATE_DELAY_NS,
        "FailureAction": "rollback",
        "Monitor": verify_rollout.CRAWL_UPDATE_MONITOR_NS,
        "MaxFailureRatio": 0,
        "Order": "start-first",
    }
    rollback = {**update, "FailureAction": "pause"}
    return {
        "image": image,
        "replicas": verify_rollout.CRAWL_REPLICAS,
        "taskLabels": verify_rollout._observability_labels(revision),
        "placement": {
            "MaxReplicas": verify_rollout.CRAWL_MAX_REPLICAS_PER_NODE
        },
        "healthCheck": {
            "Test": verify_rollout.CRAWL_HEALTHCHECK_TEST,
            **verify_rollout.CRAWL_HEALTHCHECK_TIMING,
        },
        "updateConfig": update,
        "rollbackConfig": rollback,
        "stopGracePeriod": verify_rollout.REQUIRED_STOP_GRACE_NS,
    }


def test_continuity_monitor_arms_in_mixed_state_and_rejects_a_public_failure(
    monkeypatch, tmp_path
):
    health = iter([_health("old", "old"), TimeoutError()])
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
                    "routers": [
                        {"routerId": "crawl4ai-router-7@file"}
                    ],
                    "services": [
                        {
                            "serviceId": "crawl4ai-7@swarm",
                            "serverStatus": {
                                "http://10.0.0.1:11235/": "UP"
                            },
                        },
                        {
                            "serviceId": "crawl4ai-service-7@file",
                            "serverStatus": {
                                "http://legacy-crawl4ai:11235/": "UP"
                            },
                        }
                    ],
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


def test_native_monitor_arms_only_after_task_ips_match_native_backends(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(verify_rollout.time, "sleep", lambda _seconds: None)
    evidence = tmp_path / "monitor.jsonl"
    armed = tmp_path / "armed"
    stop = tmp_path / "stop"
    service_snapshots = [
        [],
        [
            {
                "serviceId": "crawl4ai-7@swarm",
                "serverStatus": {"http://10.0.0.1:11235/": "DOWN"},
            }
        ],
        [
            {
                "serviceId": "crawl4ai-7@swarm",
                "serverStatus": {"http://10.0.0.1:11235/": "UP"},
            }
        ],
        [
            {
                "serviceId": "crawl4ai-7@swarm",
                "serverStatus": {"http://10.0.0.1:11235/": "UP"},
            }
        ],
    ]
    router_snapshots = [
        [],
        [],
        [{"routerId": "crawl4ai-router-old@file"}],
        [],
    ]

    def read_json(url, _api_key):
        if "application.runtimeServiceState" not in url:
            return _health()

        assert not armed.exists()
        services = service_snapshots.pop(0)
        routers = router_snapshots.pop(0)
        if not service_snapshots:
            stop.touch()
        return {
            "tasks": [
                {
                    "status": {"state": "running", "containerId": "current"},
                    "addresses": ["10.0.0.1/24"],
                }
            ],
            "traefik": {"routers": routers, "services": services},
        }

    verify_rollout.monitor_public_health(
        health_url="https://crawl.example/health",
        expected_replicas=1,
        evidence_path=evidence,
        armed_path=armed,
        stop_path=stop,
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="application",
        require_native_routing=True,
        read_json=read_json,
    )

    assert armed.exists()


def test_monitor_keeps_probing_while_switching_to_native_routing(
    monkeypatch, tmp_path
):
    evidence = tmp_path / "monitor.jsonl"
    armed = tmp_path / "armed"
    stop = tmp_path / "stop"
    require_native = tmp_path / "require-native"
    native_proof = tmp_path / "native-proof"
    native_proof_ready = tmp_path / "native-proof-ready"
    service_snapshots = [
        [],
        [],
        [
            {
                "serviceId": "crawl4ai-7@swarm",
                "serverStatus": {"http://10.0.0.1:11235/": "UP"},
            }
        ],
        [
            {
                "serviceId": "crawl4ai-7@swarm",
                "serverStatus": {"http://10.0.0.1:11235/": "UP"},
            }
        ],
    ]

    def advance(_seconds):
        if armed.exists() and not require_native.exists():
            require_native.touch()
            armed.unlink()
        elif armed.exists() and not native_proof.exists():
            native_proof.touch()

    def read_json(url, _api_key):
        if "application.runtimeServiceState" not in url:
            return _health()
        services = service_snapshots.pop(0)
        if not service_snapshots:
            stop.touch()
        return {
            "tasks": [
                {
                    "status": {"state": "running", "containerId": "current"},
                    "addresses": ["10.0.0.1/24"],
                }
            ],
            "traefik": {"routers": [], "services": services},
        }

    monkeypatch.setattr(verify_rollout.time, "sleep", advance)
    verify_rollout.monitor_public_health(
        health_url="https://crawl.example/health",
        expected_replicas=1,
        evidence_path=evidence,
        armed_path=armed,
        stop_path=stop,
        require_native_path=require_native,
        native_proof_path=native_proof,
        native_proof_ready_path=native_proof_ready,
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="application",
        read_json=read_json,
    )

    assert armed.exists()
    assert require_native.exists()
    assert native_proof_ready.exists()
    assert len(evidence.read_text().splitlines()) == 4


def test_native_proof_waits_for_every_predecessor_instance(monkeypatch, tmp_path):
    evidence = tmp_path / "monitor.jsonl"
    armed = tmp_path / "armed"
    stop = tmp_path / "stop"
    native_proof = tmp_path / "native-proof"
    native_proof_ready = tmp_path / "native-proof-ready"
    native_proof.touch()
    instances = iter(["old-a", "old-b", "old-c"])
    addresses = [f"10.0.0.{index}" for index in range(1, 4)]

    def read_json(url, _api_key):
        if "application.runtimeServiceState" not in url:
            assert not native_proof_ready.exists()
            return _health(next(instances), "old")
        if evidence.exists() and len(evidence.read_text().splitlines()) == 2:
            stop.touch()
        return {
            "tasks": [
                {
                    "status": {"state": "running", "containerId": f"old-{index}"},
                    "addresses": [f"{address}/24"],
                }
                for index, address in enumerate(addresses)
            ],
            "traefik": {
                "routers": [],
                "services": [
                    {
                        "serviceId": "crawl4ai-7@swarm",
                        "serverStatus": {
                            f"http://{address}:11235/": "UP" for address in addresses
                        }
                    }
                ]
            },
        }

    monkeypatch.setattr(verify_rollout.time, "sleep", lambda _seconds: None)
    verify_rollout.monitor_public_health(
        health_url="https://crawl.example/health",
        expected_replicas=3,
        evidence_path=evidence,
        armed_path=armed,
        stop_path=stop,
        native_proof_path=native_proof,
        native_proof_ready_path=native_proof_ready,
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="application",
        require_native_routing=True,
        read_json=read_json,
    )

    assert native_proof_ready.exists()
    assert len(evidence.read_text().splitlines()) == 3


def test_bootstrap_requires_three_exact_release_instances_before_native_cutover():
    image = "registry.example/crawl4ai@sha256:" + "d" * 64
    containers = [character * 64 for character in "abc"]
    health_calls = 0

    def read_json(url, _api_key):
        nonlocal health_calls
        if "deployment.all" in url:
            return [{"status": "done"}]
        if "application.runtimeServiceState" in url:
            return {
                "application": {"appName": "crawl4ai"},
                "service": _crawl_service(image, "target"),
                "tasks": [
                    {
                        "desiredState": "running",
                        "status": {"state": "running", "containerId": container},
                    }
                    for container in containers
                ],
            }
        instance = containers[health_calls % len(containers)][:12]
        health_calls += 1
        return _health(instance)

    verify_rollout.verify_bootstrap(
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="application",
        revision="target",
        expected_image=image,
        health_url="https://crawl.example/health",
        read_json=read_json,
        sleep=lambda _seconds: None,
    )

    assert health_calls == 24


def test_deployment_wait_requires_a_new_service_generation():
    versions = iter([9, 9, 10])
    now = 0

    def sleep(seconds):
        nonlocal now
        now += seconds

    verify_rollout._wait_for_service_generation(
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="application",
        read_json=lambda _url, _api_key: {
            "application": {"appName": "crawl4ai"},
            "service": {"versionIndex": next(versions)},
        },
        sleep=sleep,
        monotonic=lambda: now,
        previous_service_version=9,
    )

    with pytest.raises(StopIteration):
        next(versions)


def test_bootstrap_rejects_unsafe_rollout_policy_before_native_cutover():
    image = "registry.example/crawl4ai@sha256:" + "d" * 64
    service = _crawl_service(image, "target")
    service["placement"] = {"MaxReplicas": 2}

    def read_json(url, _api_key):
        if "deployment.all" in url:
            return [{"status": "done"}]
        return {"application": {"appName": "crawl4ai"}, "service": service}

    with pytest.raises(ValueError, match="one replica per node"):
        verify_rollout.verify_bootstrap(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="application",
            revision="target",
            expected_image=image,
            health_url="https://crawl.example/health",
            read_json=read_json,
            sleep=lambda _seconds: None,
        )


def test_native_routing_detection_requires_fail_closed_swarm_labels():
    state = {
        "application": {"appName": "crawl4ai"},
        "service": {
            "rootLabels": {
                "traefik.enable": "true",
                "traefik.swarm.network": "dokploy-network",
                "traefik.swarm.lbswarm": "false",
                "traefik.http.services.crawl.loadbalancer.healthcheck.initialstatus": "down",
            }
        },
        "traefik": {"routers": [{"routerId": "crawl4ai-7-web@swarm"}]},
    }

    assert verify_rollout._has_native_routing(state)
    state["traefik"]["routers"].append(
        {"routerId": "crawl4ai-router-old@file"}
    )
    assert not verify_rollout._has_native_routing(state)
    state["traefik"]["routers"].pop()
    state["service"]["rootLabels"].pop(
        "traefik.http.services.crawl.loadbalancer.healthcheck.initialstatus"
    )
    assert not verify_rollout._has_native_routing(state)


def test_monitor_evidence_proves_predecessor_withdrawal_and_replacement_coverage(
    tmp_path,
):
    evidence = tmp_path / "monitor.jsonl"
    def task(container, address, state="running"):
        return {
            "taskId": container,
            "status": {"state": state, "containerId": f"container-{container}"},
            "addresses": [f"{address}/24"],
        }

    old = [task("old-a", "10.0.0.1"), task("old-b", "10.0.0.2"), task("old-c", "10.0.0.3")]
    final = [task("new-a", "10.0.0.4"), task("new-b", "10.0.0.5"), task("new-c", "10.0.0.1")]
    samples = [
        {
            "ok": True,
            "instance": "old-a",
            "tasks": [*old, task("historical", "10.0.0.99", "shutdown")],
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
            "upAddresses": ["10.0.0.1", "10.0.0.4", "10.0.0.5"],
        },
        {
            "ok": True,
            "instance": "new-a",
            "tasks": final,
            "upAddresses": ["10.0.0.1", "10.0.0.4", "10.0.0.5"],
        },
    ]
    for sample in samples:
        sample["proofPhase"] = "native"
    samples.insert(
        0,
        {
            "ok": True,
            "instance": "old-a",
            "tasks": [old[0], old[1], task("historical", "10.0.0.99", "shutdown")],
            "upAddresses": ["10.0.0.1", "10.0.0.2"],
            "proofPhase": "native",
        },
    )
    samples.insert(
        0,
        {
            "ok": True,
            "instance": "bootstrap-task",
            "tasks": [task("bootstrap-task", "10.0.0.20")],
            "upAddresses": [],
            "proofPhase": "continuity",
        },
    )
    evidence.write_text("".join(json.dumps(sample) + "\n" for sample in samples))

    verify_rollout.verify_monitor_evidence(
        evidence,
        frozenset({"new-a", "new-b", "new-c"}),
        frozenset({"new-a", "new-b", "new-c"}),
        frozenset({"10.0.0.1", "10.0.0.4", "10.0.0.5"}),
    )


def test_deploy_workflow_uses_only_native_swarm_readiness():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-deploy.yml").read_text()
    routing_mode = workflow.index("verify_rollout.py routing-mode")
    native_monitor = workflow.index("start_monitor monitor-native", routing_mode)
    native_baseline = workflow.index("begin_native_proof", native_monitor)
    legacy_monitor = workflow.index("start_monitor monitor ", routing_mode)
    artifact_update = workflow.index('"dockerImage":', routing_mode)
    service_cursor = workflow.index("verify_rollout.py service-cursor", artifact_update)
    bootstrap_deploy = workflow.index("/api/application.deploy", service_cursor)
    legacy_branch = workflow.index('if [ "$routing_mode" = legacy ]; then', bootstrap_deploy)
    bootstrap = workflow.index("verify_rollout.py bootstrap", legacy_branch)
    readiness = workflow.index(
        '"readinessCheckSwarm":{"Path":"/health/route","Interval":500000000,'
        '"UnhealthyInterval":250000000,"Timeout":400000000,"Status":200}'
    )
    require_native = workflow.index(
        'touch "$ROLLOUT_MONITOR_REQUIRE_NATIVE_PATH"', bootstrap
    )
    reset_admission = workflow.index(
        'rm -f "$ROLLOUT_MONITOR_ARMED_PATH"', require_native
    )
    native_admission = workflow.index("wait_for_monitor", readiness)
    migration_baseline = workflow.index("begin_native_proof", native_admission)
    migration_cursor = workflow.index("verify_rollout.py service-cursor", migration_baseline)
    deploy = workflow.index("/api/application.deploy", migration_cursor)
    final_proof = workflow.index("python3 deploy/docker/verify_rollout.py\n", deploy)

    assert routing_mode < native_monitor < native_baseline < artifact_update
    assert routing_mode < legacy_monitor < artifact_update
    assert (
        artifact_update
        < service_cursor
        < bootstrap_deploy
        < legacy_branch
        < bootstrap
        < readiness
    )
    assert (
        bootstrap
        < require_native
        < reset_admission
        < readiness
        < native_admission
        < migration_baseline
        < migration_cursor
        < deploy
        < final_proof
    )
    assert "finish_monitor" not in workflow[bootstrap:readiness]
    assert "readinessCheckSwarm" not in workflow[artifact_update:bootstrap_deploy]
    assert workflow.count("${{ runner.temp }}/crawl4ai-rollout-monitor.jsonl") == 1
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
