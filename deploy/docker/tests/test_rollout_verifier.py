from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import verify_rollout as rollout_verifier
from verify_rollout import (
    MAX_RESPONSE_BYTES,
    REQUIRED_STOP_GRACE_NS,
    _request_json,
    _has_task_error,
    verify_monitor_evidence,
    verify_native_route,
    verify_rollout,
    verify_rollout_preflight,
)

ORIGINAL_INSPECT_TASK_RUNTIME = rollout_verifier._inspect_task_runtime
TARGET_IMAGE = "registry.example/crawl4ai@sha256:target"
HEALTH_URLS = ("https://crawl.example/health",)
DOCKER_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _task_runtime(monkeypatch):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", 1)
    monkeypatch.setattr(
        rollout_verifier,
        "_crawl_node_inventory",
        lambda: {
            node: ("Ready", "Active") for node in rollout_verifier.CRAWL_ELIGIBLE_NODES
        },
    )

    def inspect(task_id, _network_id):
        instance = ("b" if "b" in task_id else "a") * 12
        address = "10.0.1.12" if instance.startswith("b") else "10.0.1.11"
        return (
            instance,
            frozenset({address}),
            rollout_verifier._observability_labels("target"),
            "registry.example/crawl4ai@sha256:target",
        )

    monkeypatch.setattr(rollout_verifier, "_inspect_task_runtime", inspect)
    monkeypatch.setattr(rollout_verifier, "_dokploy_network_id", lambda: "network")
    monkeypatch.setattr(rollout_verifier, "verify_native_route", lambda **_kwargs: None)
    monkeypatch.setattr(rollout_verifier, "_post_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        rollout_verifier,
        "_service_update_state",
        lambda _app_name: "rollback_completed",
    )
    monkeypatch.setattr(
        rollout_verifier,
        "_service_spec",
        lambda app_name: (
            _redis_service_spec()
            if app_name == "crawl4ai-redis"
            else _crawl_service_spec(rollout_verifier.CRAWL_REPLICAS)
        ),
    )


def test_release_requires_exactly_three_replicas(monkeypatch):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", 3)
    with pytest.raises(ValueError, match="exactly three"):
        rollout_verifier._verify_release_configuration(
            _application(replicas=2), "target", TARGET_IMAGE
        )


@pytest.mark.parametrize("field", ["updateConfigSwarm", "stopGracePeriodSwarm"])
def test_current_rollout_source_rejects_timing_drift(field):
    application = _application()
    if field == "updateConfigSwarm":
        application[field] = {
            **application[field],
            "Delay": rollout_verifier.ROLLOUT_DELAY_NS - 1,
        }
    else:
        application[field] = rollout_verifier.STOP_GRACE_NS - 1
    with pytest.raises(ValueError):
        rollout_verifier._verify_current_rollout_source(application)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda spec: spec["TaskTemplate"]["ContainerSpec"]["Healthcheck"].update(
            StartPeriod=1
        ),
        lambda spec: spec["EndpointSpec"].update(
            Ports=[{"TargetPort": 11235, "PublishedPort": 11235}]
        ),
    ],
)
def test_rendered_crawl_service_rejects_healthcheck_or_published_port_drift(mutation):
    spec = copy.deepcopy(_crawl_service_spec())
    mutation(spec)

    with pytest.raises(ValueError):
        rollout_verifier._verify_current_service_spec(spec, TARGET_IMAGE)


def test_rendered_redis_service_rejects_command_drift():
    spec = copy.deepcopy(_redis_service_spec())
    spec["TaskTemplate"]["ContainerSpec"]["Command"][6] = "always"
    with pytest.raises(ValueError, match="appendfsync everysec"):
        rollout_verifier._verify_redis_service_spec(spec)


def _application(**overrides):
    application = {
        "replicas": 1,
        "appName": "crawl4ai",
        "dockerImage": "registry.example/crawl4ai@sha256:target",
        "cpuReservation": rollout_verifier.CRAWL_CPU_RESERVATION,
        "cpuLimit": rollout_verifier.CRAWL_CPU_LIMIT,
        "memoryReservation": rollout_verifier.CRAWL_MEMORY_RESERVATION,
        "memoryLimit": rollout_verifier.CRAWL_MEMORY_LIMIT,
        "labelsSwarm": rollout_verifier._observability_labels("target"),
        "env": (
            f"LLM_PROVIDER={rollout_verifier.LLM_PROVIDER}\n"
            f"LLM_BASE_URL={rollout_verifier.LLM_BASE_URL}\n"
            "LLM_API_KEY=nonempty"
        ),
        "mounts": [
            {
                "type": "volume",
                "volumeName": rollout_verifier.REDIS_VOLUME,
                "mountPath": rollout_verifier.REDIS_MOUNT_PATH,
            }
        ],
        "healthCheckSwarm": rollout_verifier.CRAWL_HEALTHCHECK_POLICY,
        "command": "redis-server --dir /data --appendonly yes --appendfsync everysec --loglevel notice",
        "placementSwarm": {
            "Constraints": [rollout_verifier.CRAWL_NODE_CONSTRAINT],
            "MaxReplicas": rollout_verifier.CRAWL_MAX_REPLICAS_PER_NODE,
        },
        "endpointSpecSwarm": rollout_verifier.CRAWL_ENDPOINT_SPEC,
        "swarmVipConnectionReuse": False,
        "updateConfigSwarm": {
            "Order": "start-first",
            "Parallelism": 1,
            "Delay": rollout_verifier.ROLLOUT_DELAY_NS,
            "Monitor": rollout_verifier.ROLLOUT_MONITOR_NS,
            "FailureAction": "rollback",
            "MaxFailureRatio": 0,
        },
        "rollbackConfigSwarm": {
            "Order": "start-first",
            "Parallelism": 1,
            "Delay": rollout_verifier.ROLLOUT_DELAY_NS,
            "Monitor": rollout_verifier.ROLLOUT_MONITOR_NS,
            "FailureAction": "pause",
            "MaxFailureRatio": 0,
        },
        "stopGracePeriodSwarm": rollout_verifier.STOP_GRACE_NS,
    }
    application.update(overrides)
    return application


def _running_task(task_id: str, node: str, seconds: int = 1) -> dict:
    return {
        "containerId": task_id,
        "state": "running",
        "currentState": f"Running {seconds}s",
        "node": {"a": "haiku-4", "b": "haiku-5"}.get(node, node),
        "error": "",
    }


def _crawl_service_spec(replicas: int = 1) -> dict:
    return {
        "TaskTemplate": {
            "ContainerSpec": {
                "Image": TARGET_IMAGE,
                "Healthcheck": rollout_verifier.CRAWL_HEALTHCHECK_POLICY,
                "StopGracePeriod": rollout_verifier.STOP_GRACE_NS,
            },
            "Placement": {
                "Constraints": [rollout_verifier.CRAWL_NODE_CONSTRAINT],
                "MaxReplicas": 1,
            },
            "Resources": {
                "Reservations": {
                    "NanoCPUs": int(rollout_verifier.CRAWL_CPU_RESERVATION),
                    "MemoryBytes": int(rollout_verifier.CRAWL_MEMORY_RESERVATION),
                },
                "Limits": {
                    "NanoCPUs": int(rollout_verifier.CRAWL_CPU_LIMIT),
                    "MemoryBytes": int(rollout_verifier.CRAWL_MEMORY_LIMIT),
                },
            },
        },
        "Mode": {"Replicated": {"Replicas": replicas}},
        "EndpointSpec": rollout_verifier.CRAWL_ENDPOINT_SPEC,
        "UpdateConfig": {
            "Order": "start-first",
            "Parallelism": 1,
            "FailureAction": "rollback",
            "MaxFailureRatio": 0,
            "Delay": rollout_verifier.ROLLOUT_DELAY_NS,
            "Monitor": rollout_verifier.ROLLOUT_MONITOR_NS,
        },
        "RollbackConfig": {
            "Order": "start-first",
            "Parallelism": 1,
            "FailureAction": "pause",
            "MaxFailureRatio": 0,
            "Delay": rollout_verifier.ROLLOUT_DELAY_NS,
            "Monitor": rollout_verifier.ROLLOUT_MONITOR_NS,
        },
    }


def _redis_service_spec() -> dict:
    return {
        "TaskTemplate": {
            "ContainerSpec": {
                "Image": rollout_verifier.REDIS_IMAGE,
                "Command": rollout_verifier.REDIS_COMMAND,
                "Healthcheck": rollout_verifier.REDIS_HEALTHCHECK_POLICY,
                "Mounts": [
                    {
                        "Type": "volume",
                        "Source": rollout_verifier.REDIS_VOLUME,
                        "Target": rollout_verifier.REDIS_MOUNT_PATH,
                    }
                ],
            },
            "Placement": {
                "Constraints": [rollout_verifier.REDIS_NODE_CONSTRAINT],
                "MaxReplicas": 1,
            },
            "Resources": {
                "Reservations": {
                    "NanoCPUs": int(rollout_verifier.REDIS_CPU_RESERVATION),
                    "MemoryBytes": int(rollout_verifier.REDIS_MEMORY_RESERVATION),
                },
                "Limits": {
                    "NanoCPUs": int(rollout_verifier.REDIS_CPU_LIMIT),
                    "MemoryBytes": int(rollout_verifier.REDIS_MEMORY_LIMIT),
                },
            },
        }
    }


def _health(instance: str, revision: str = "target") -> dict:
    return {
        "instance": instance,
        "revision": revision,
        "status": "ok",
        "components": {"api": "ready", "redis": "ready"},
    }


def _fake_read(handlers):
    def read_json(url, _api_key):
        operation = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
        if (
            operation == "docker.getServiceContainersByAppName"
            and "appName=crawl4ai-redis" in url
        ):
            return [_running_task("redis-task", "haiku-18")]
        if operation not in handlers:
            raise AssertionError(f"unexpected operation: {operation}")
        handler = handlers[operation]
        result = handler(url) if callable(handler) else handler
        if operation == "application.one" and isinstance(result, dict):
            supplied = result
            result = _application(**result)
            if "applicationId=redis" in url:
                result.update(
                    {
                        "appName": "crawl4ai-redis",
                        "dockerImage": rollout_verifier.REDIS_IMAGE,
                        "replicas": 1,
                        "cpuReservation": rollout_verifier.REDIS_CPU_RESERVATION,
                        "cpuLimit": rollout_verifier.REDIS_CPU_LIMIT,
                        "memoryReservation": rollout_verifier.REDIS_MEMORY_RESERVATION,
                        "memoryLimit": rollout_verifier.REDIS_MEMORY_LIMIT,
                    }
                )
                if "healthCheckSwarm" not in supplied:
                    result["healthCheckSwarm"] = (
                        rollout_verifier.REDIS_HEALTHCHECK_POLICY
                    )
                if "placementSwarm" not in supplied:
                    result["placementSwarm"] = {
                        "Constraints": [rollout_verifier.REDIS_NODE_CONSTRAINT],
                        "MaxReplicas": 1,
                    }
            result.setdefault("stopGracePeriodSwarm", REQUIRED_STOP_GRACE_NS)
        return result

    return read_json


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("", False),
        ("Error:", False),
        ("Error: task failed", True),
        ("task failed", True),
    ],
)
def test_task_error_ignores_only_empty_dokploy_error_boilerplate(error, expected):
    assert _has_task_error({"error": error}) is expected


def test_task_runtime_reads_only_the_dokploy_overlay(monkeypatch):
    output = (
        '"'
        + "a" * 64
        + '"\t'
        + '{"otel.service.version":"target"}\t'
        + '[{"Network":{"ID":"other"},"Addresses":["10.9.0.2/24"]},'
        + '{"Network":{"ID":"network"},"Addresses":["10.0.1.11/24"]}]\t'
        + '"registry.example/crawl4ai@sha256:target"\n'
    )
    monkeypatch.setattr(
        rollout_verifier.subprocess,
        "run",
        lambda *_args, **_kwargs: rollout_verifier.subprocess.CompletedProcess(
            [], 0, output, ""
        ),
    )

    instance, addresses, labels, image = ORIGINAL_INSPECT_TASK_RUNTIME(
        "task", "network"
    )

    assert instance == "a" * 12
    assert addresses == frozenset({"10.0.1.11"})
    assert labels == {"otel.service.version": "target"}
    assert image == "registry.example/crawl4ai@sha256:target"


def test_native_route_requires_only_vip_services_and_the_no_reuse_transport(
    monkeypatch,
):
    route = """http:
  routers:
    crawl4ai-yq1svi-router-1:
      service: crawl4ai-yq1svi-service-1
      rule: Host(`crawl4ai.haiku.host`)
    crawl4ai-yq1svi-router-2:
      service: crawl4ai-yq1svi-service-2
      rule: Host(`crawl4ai.popos-sf0.com`)
  services:
    crawl4ai-yq1svi-service-1:
      loadBalancer:
        servers:
          - url: http://crawl4ai-yq1svi:11235
        serversTransport: crawl4ai-yq1svi-swarm-vip
    crawl4ai-yq1svi-service-2:
      loadBalancer:
        servers:
          - url: http://crawl4ai-yq1svi:11235
        serversTransport: crawl4ai-yq1svi-swarm-vip
  serversTransports:
    crawl4ai-yq1svi-swarm-vip:
      maxIdleConnsPerHost: -1
"""
    monkeypatch.setattr(rollout_verifier, "_request_json", lambda *_args: route)

    verify_native_route(
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="app",
        app_name="crawl4ai-yq1svi",
    )


@pytest.mark.parametrize(
    "replacement",
    [
        "http://legacy-ingress:11235",
        "serversTransport: default",
        "maxIdleConnsPerHost: 0",
    ],
)
def test_native_route_rejects_backend_or_transport_drift(monkeypatch, replacement):
    route = """http:
  routers:
    crawl4ai-yq1svi-router-1:
      service: crawl4ai-yq1svi-service-1
      rule: Host(`crawl4ai.haiku.host`)
  services:
    crawl4ai-yq1svi-service-1:
      loadBalancer:
        servers:
          - url: http://crawl4ai-yq1svi:11235
        serversTransport: crawl4ai-yq1svi-swarm-vip
  serversTransports:
    crawl4ai-yq1svi-swarm-vip:
      maxIdleConnsPerHost: -1
"""
    if replacement.startswith("http"):
        route = route.replace("http://crawl4ai-yq1svi:11235", replacement)
    elif replacement.startswith("serversTransport"):
        route = route.replace(
            "serversTransport: crawl4ai-yq1svi-swarm-vip", replacement
        )
    else:
        route = route.replace("maxIdleConnsPerHost: -1", replacement)
    monkeypatch.setattr(rollout_verifier, "_request_json", lambda *_args: route)

    with pytest.raises(ValueError):
        verify_native_route(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            app_name="crawl4ai-yq1svi",
        )


def test_native_route_rejects_one_router_service_without_transport(monkeypatch):
    route = """http:
  routers:
    crawl4ai-yq1svi-router-1:
      service: crawl4ai-yq1svi-service-1
      rule: Host(`crawl4ai.haiku.host`)
    crawl4ai-yq1svi-router-2:
      service: crawl4ai-yq1svi-service-2
      rule: Host(`crawl4ai.popos-sf0.com`)
  services:
    crawl4ai-yq1svi-service-1:
      loadBalancer:
        servers:
          - url: http://crawl4ai-yq1svi:11235
        serversTransport: crawl4ai-yq1svi-swarm-vip
    crawl4ai-yq1svi-service-2:
      loadBalancer:
        servers:
          - url: http://crawl4ai-yq1svi:11235
  serversTransports:
    crawl4ai-yq1svi-swarm-vip:
      maxIdleConnsPerHost: -1
"""
    monkeypatch.setattr(rollout_verifier, "_request_json", lambda *_args: route)

    with pytest.raises(ValueError, match="does not use"):
        verify_native_route(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            app_name="crawl4ai-yq1svi",
        )


def test_rollback_verifier_requires_tasks_public_health_and_metadata():
    read_json = _fake_read(
        {
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": [
                _running_task("task-a", "haiku-4", 5)
            ],
        }
    )

    rollout_verifier._wait_for_verified_rollback(
        app_name="crawl4ai",
        baseline_image=TARGET_IMAGE,
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="app",
        health_urls=HEALTH_URLS,
        read_json=read_json,
        probe_health=lambda _url, count: [_health("a" * 12)] * count,
        inspect_task_runtime=rollout_verifier._inspect_task_runtime,
        sleep=lambda _seconds: None,
        monotonic=lambda: 0,
    )


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        ({"haiku-4": ("Ready", "Active")}, "admitted haiku-4/5/9/18"),
        (
            {
                **{
                    node: ("Ready", "Active")
                    for node in rollout_verifier.CRAWL_ELIGIBLE_NODES
                },
                "haiku-0": ("Down", "Drain"),
            },
            "admitted haiku-4/5/9/18",
        ),
        (
            {
                node: (("Down", "Drain") if node == "haiku-9" else ("Ready", "Active"))
                for node in rollout_verifier.CRAWL_ELIGIBLE_NODES
            },
            "Ready and Active",
        ),
        (None, None),
    ],
)
def test_rollout_preflight_requires_the_exact_capacity_admitted_pool(
    monkeypatch, inventory, message
):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", 3)
    monkeypatch.setattr(
        rollout_verifier,
        "_crawl_node_inventory",
        lambda: inventory
        or {
            node: ("Ready", "Active") for node in rollout_verifier.CRAWL_ELIGIBLE_NODES
        },
    )
    monkeypatch.setattr(
        rollout_verifier,
        "_request_json",
        _fake_read({"application.one": {"replicas": 3}}),
    )

    if inventory is not None:
        with pytest.raises(RuntimeError, match=message):
            verify_rollout_preflight(
                dokploy_url="https://dokploy.example",
                api_key="secret",
                application_id="app",
                redis_application_id="redis",
            )
    else:
        verify_rollout_preflight(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
        )


@pytest.mark.parametrize(
    "tasks",
    [
        [_running_task("task-a", "haiku-0", 5)],
        [
            _running_task("task-a", "haiku-4", 5),
            _running_task("task-b", "haiku-4", 5),
            _running_task("task-c", "haiku-5", 5),
        ],
    ],
)
def test_verifier_rejects_tasks_outside_or_duplicated_within_admitted_pool(
    monkeypatch, tasks
):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", len(tasks))
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": len(tasks), "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": tasks,
        }
    )

    with pytest.raises(ValueError, match="distinct admitted nodes"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
            sleep=lambda _seconds: None,
        )


def test_monitor_proves_admission_and_predecessor_withdrawal_before_exit(tmp_path):
    def task(
        task_id,
        container_id,
        desired,
        current,
        slot=1,
        status_timestamp="2026-08-29T00:00:00Z",
        updated_at="2026-08-29T00:00:00Z",
    ):
        return {
            "ID": task_id,
            "ContainerID": container_id,
            "DesiredState": desired,
            "CurrentState": current,
            "Slot": slot,
            "StatusTimestamp": status_timestamp,
            "UpdatedAt": updated_at,
        }

    def sample(instances, tasks):
        return {
            "ok": True,
            "health": [
                {
                    "ok": True,
                    "url": "https://crawl.example/health",
                    "instance": instance,
                    "revision": "target",
                }
                for instance in instances
            ],
            "tasks": tasks,
        }

    predecessor = task("old-task", "old-instance", "Running", "Running 1m")
    candidate_starting = task(
        "new-task",
        "new-instance",
        "Running",
        "Starting 1s",
        status_timestamp="2026-08-29T00:00:01Z",
        updated_at="2026-08-29T00:00:01Z",
    )
    candidate_running = task(
        "new-task",
        "new-instance",
        "Running",
        "Running 1s",
        status_timestamp="2026-08-29T00:00:02Z",
        updated_at="2026-08-29T00:00:02Z",
    )
    predecessor_draining = task(
        "old-task",
        "old-instance",
        "Shutdown",
        "Running 1m",
        updated_at="2026-08-29T00:00:03Z",
    )
    predecessor_exited = task(
        "old-task",
        "old-instance",
        "Shutdown",
        "Complete 1s",
        status_timestamp="2026-08-29T00:00:04Z",
        updated_at="2026-08-29T00:00:03Z",
    )
    evidence = tmp_path / "rollout.jsonl"
    evidence.write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                sample(["old-instance"], [predecessor]),
                sample(["old-instance"], [predecessor, candidate_starting]),
                sample(["new-instance"], [candidate_running, predecessor_draining]),
                sample(["new-instance"], [candidate_running, predecessor_exited]),
            )
        )
        + "\n"
    )

    verify_monitor_evidence(
        Path(evidence),
        frozenset({"new-instance"}),
        frozenset({"new-task"}),
        "target",
        frozenset({"https://crawl.example/health"}),
    )


def test_monitor_requires_each_public_domain_to_observe_the_final_task(tmp_path):
    primary = "https://crawl.example/health"
    secondary = "https://crawl-legacy.example/health"
    evidence = tmp_path / "rollout.jsonl"
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "health": [
                    {
                        "ok": True,
                        "url": primary,
                        "instance": "new-instance",
                        "revision": "target",
                    }
                ],
                "tasks": [],
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match=secondary):
        verify_monitor_evidence(
            evidence,
            frozenset({"new-instance"}),
            frozenset({"new-task"}),
            "target",
            frozenset({primary, secondary}),
        )


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


def test_request_json_rejects_streamed_responses_over_64_kib():
    class OversizedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * (MAX_RESPONSE_BYTES + 1))

        def log_message(self, format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), OversizedHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        with pytest.raises(ValueError, match="exceeds 64 KiB"):
            _request_json(f"http://127.0.0.1:{server.server_port}/")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"labelsSwarm": {}}, "observability labels"),
        (
            {
                "env": (
                    "LLM_PROVIDER=openai/gpt-4o-mini\n"
                    "LLM_BASE_URL=https://api.llm-gateway.iocloudhost.net/v1\n"
                    "LLM_API_KEY=nonempty"
                )
            },
            "LLM_PROVIDER",
        ),
        (
            {
                "env": (
                    "LLM_PROVIDER=openai/qwen3.8-27b\n"
                    "LLM_BASE_URL=https://wrong.example/v1\n"
                    "LLM_API_KEY=nonempty"
                )
            },
            "LLM_BASE_URL",
        ),
        (
            {
                "env": (
                    "LLM_PROVIDER=openai/qwen3.8-27b\n"
                    "LLM_BASE_URL=https://api.llm-gateway.iocloudhost.net/v1\n"
                    "LLM_API_KEY="
                )
            },
            "LLM_API_KEY must be nonempty",
        ),
        ({"placementSwarm": {"MaxReplicas": 2}}, "one replica per node"),
        ({"healthCheckSwarm": {"Test": ["CMD", "true"]}}, "admission healthcheck"),
        (
            {
                "healthCheckSwarm": {
                    "Test": rollout_verifier.CRAWL_HEALTHCHECK,
                    "Interval": 1_000_000_000,
                    "Timeout": 1_000_000_000,
                    "Retries": 1,
                }
            },
            "exact admission healthcheck",
        ),
        (
            {
                "updateConfigSwarm": {
                    "Order": "stop-first",
                    "Parallelism": 1,
                    "MaxFailureRatio": 0,
                }
            },
            "updateConfigSwarm must use start-first",
        ),
        (
            {
                "rollbackConfigSwarm": {
                    "Order": "stop-first",
                    "Parallelism": 1,
                    "MaxFailureRatio": 0,
                }
            },
            "rollbackConfigSwarm must use start-first",
        ),
    ],
)
def test_verifier_rejects_invalid_release_observability_configuration(
    overrides, message
):
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": overrides,
        }
    )

    with pytest.raises(ValueError, match=message):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mounts": []}, "crawl4ai-redis-data"),
        (
            {"healthCheckSwarm": {"Test": ["CMD-SHELL", "redis-cli ping"]}},
            "healthcheck",
        ),
        (
            {
                "command": "redis-server --dir /data --appendonly yes --appendfsync always"
            },
            "appendfsync everysec",
        ),
        ({"placementSwarm": {"Constraints": [], "MaxReplicas": 1}}, "haiku-18"),
        (
            {
                "placementSwarm": {
                    "Constraints": ["node.hostname==haiku-18"],
                    "MaxReplicas": 2,
                }
            },
            "MaxReplicas",
        ),
    ],
)
def test_verifier_rejects_invalid_external_redis_configuration(overrides, message):
    def application_for_id(url):
        return overrides if "applicationId=redis" in url else {}

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": application_for_id,
        }
    )

    with pytest.raises(ValueError, match=message):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
        )


def test_verifier_requires_two_identical_complete_task_and_instance_snapshots_with_runtime_labels():
    container_id = "a" * 64
    health_urls = (
        "https://crawl.example/health",
        "https://crawl-legacy.example/health",
    )
    probed_urls = []
    task_snapshots = iter(
        [
            [_running_task("task-a", "a")],
            [_running_task("task-a", "a", 6)],
        ]
    )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": lambda _url: next(task_snapshots),
        }
    )

    def probe_health(url, count):
        probed_urls.append(url)
        return [_health(container_id[:12])] * count

    verify_rollout(
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="app",
        redis_application_id="redis",
        revision="target",
        expected_image=TARGET_IMAGE,
        baseline_image=TARGET_IMAGE,
        health_urls=health_urls,
        read_json=read_json,
        probe_health=probe_health,
        sleep=lambda _seconds: None,
    )
    assert set(probed_urls) == set(health_urls)


@pytest.mark.parametrize(
    "second_tasks,second_health,second_labels,error_type,error_message",
    [
        (
            [_running_task("task-b", "b")],
            _health("b" * 12),
            None,
            TimeoutError,
            "did not stabilize",
        ),
        (
            [_running_task("task-a", "a", 6)],
            _health("a" * 12, "rolled-back"),
            None,
            TimeoutError,
            "did not stabilize",
        ),
        (
            [_running_task("task-a", "a", 6)],
            _health("a" * 12),
            rollout_verifier._observability_labels("wrong"),
            ValueError,
            "observability labels",
        ),
    ],
)
def test_verifier_rejects_membership_churn_revision_rollback_or_runtime_label_drift(
    second_tasks, second_health, second_labels, error_type, error_message
):
    clock = iter([0, 0, 0, 1, 1, rollout_verifier.ROLLOUT_PROOF_TIMEOUT_SECONDS + 1])
    task_snapshots = iter(
        [
            [_running_task("task-a", "a")],
            second_tasks,
        ]
    )
    health_snapshots = iter([_health("a" * 12), second_health])
    config_labels = iter([None, second_labels])

    def inspect_task(task_id, _network_id):
        container_id = ("a" if task_id == "task-a" else "b") * 64
        labels = next(config_labels)
        return (
            container_id[:12],
            frozenset({"10.0.1.11" if task_id == "task-a" else "10.0.1.12"}),
            rollout_verifier._observability_labels("target")
            if labels is None
            else labels,
            "registry.example/crawl4ai@sha256:target",
        )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": lambda _url: next(task_snapshots),
        }
    )

    with pytest.raises(error_type, match=error_message):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
            probe_health=lambda _url, count: [next(health_snapshots)] * count,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
            inspect_task_runtime=inspect_task,
        )


def test_verifier_rejects_replica_counts_other_than_three():
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 17, "appName": "crawl4ai"},
        }
    )

    with pytest.raises(ValueError, match="exactly three"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
            sleep=lambda _seconds: None,
        )


def test_verifier_rejects_stop_grace_shorter_than_the_process_drain():
    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {
                "replicas": 1,
                "appName": "crawl4ai",
                "stopGracePeriodSwarm": REQUIRED_STOP_GRACE_NS - 1,
            },
        }
    )

    with pytest.raises(ValueError, match="must outlive"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
            sleep=lambda _seconds: None,
        )


def test_verifier_rejects_shutdown_desired_tasks_that_are_still_running():
    clock = iter([0, 0, 0, 1, rollout_verifier.ROLLOUT_PROOF_TIMEOUT_SECONDS + 1])
    tasks = [
        _running_task("task-current", "a", 5),
        {
            "containerId": "task-stale",
            "state": "shutdown",
            "currentState": "Running 1s",
            "node": "b",
            "error": "",
        },
    ]

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": tasks,
        }
    )

    with pytest.raises(TimeoutError, match="did not stabilize"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
            probe_health=lambda _url, count: [_health("a" * 12)] * count,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
        )


def test_verifier_rejects_desired_running_tasks_that_are_still_pending():
    clock = iter([0, 0, 0, 1, rollout_verifier.ROLLOUT_PROOF_TIMEOUT_SECONDS + 1])
    tasks = [
        _running_task("task-a", "a", 5),
        _running_task("task-b", "b", 5),
        {
            "containerId": "task-pending",
            "state": "running",
            "currentState": "Pending 1s",
            "node": "",
            "error": "",
        },
    ]

    def unexpected_inspection(_task, _network):
        raise AssertionError(
            "impossible task sets must short-circuit before inspection"
        )

    read_json = _fake_read(
        {
            "deployment.all": [{"status": "done"}],
            "application.one": {"replicas": 1, "appName": "crawl4ai"},
            "docker.getServiceContainersByAppName": tasks,
        }
    )

    with pytest.raises(TimeoutError, match="did not stabilize"):
        verify_rollout(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            redis_application_id="redis",
            revision="target",
            expected_image=TARGET_IMAGE,
            baseline_image=TARGET_IMAGE,
            health_urls=HEALTH_URLS,
            read_json=read_json,
            probe_health=lambda _url, count: [_health("a" * 12), _health("b" * 12)]
            * (count // 2),
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
            inspect_task_runtime=unexpected_inspection,
        )
