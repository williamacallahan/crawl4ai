from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import verify_rollout as rollout_verifier
from verify_rollout import (
    MAX_RESPONSE_BYTES,
    REQUIRED_STOP_GRACE_NS,
    _curl_json,
    _has_task_error,
    verify_monitor_evidence,
    verify_native_route,
    verify_rollout,
    verify_rollout_preflight,
)

ORIGINAL_INSPECT_TASK_RUNTIME = rollout_verifier._inspect_task_runtime


@pytest.fixture(autouse=True)
def _task_runtime(monkeypatch):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", 1)
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


def test_release_requires_exactly_three_replicas(monkeypatch):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", 3)
    with pytest.raises(ValueError, match="exactly three"):
        rollout_verifier._verify_release_configuration(_application(replicas=2), "target")


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
        "healthCheckSwarm": {
            "Test": rollout_verifier.CRAWL_HEALTHCHECK,
            "Interval": 10_000_000_000,
            "Timeout": 5_000_000_000,
            "Retries": 5,
        },
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
        "node": node,
        "error": "",
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
                result["appName"] = "crawl4ai-redis"
                if "healthCheckSwarm" not in supplied:
                    result["healthCheckSwarm"] = {
                        "Test": rollout_verifier.REDIS_HEALTHCHECK
                    }
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
        '"' + "a" * 64 + '"\t'
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
    crawl4ai-yq1svi-router-2:
      service: crawl4ai-yq1svi-service-2
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
    monkeypatch.setattr(rollout_verifier, "_curl_json", lambda *_args: route)

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
    monkeypatch.setattr(rollout_verifier, "_curl_json", lambda *_args: route)

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
    crawl4ai-yq1svi-router-2:
      service: crawl4ai-yq1svi-service-2
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
    monkeypatch.setattr(rollout_verifier, "_curl_json", lambda *_args: route)

    with pytest.raises(ValueError, match="does not use"):
        verify_native_route(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
            app_name="crawl4ai-yq1svi",
        )


@pytest.mark.parametrize("eligible", [frozenset({"haiku-4"}), None])
def test_rollout_preflight_requires_the_exact_capacity_admitted_pool(
    monkeypatch, eligible
):
    monkeypatch.setattr(rollout_verifier, "CRAWL_REPLICAS", 3)
    nodes = eligible or rollout_verifier.CRAWL_ELIGIBLE_NODES
    monkeypatch.setattr(
        rollout_verifier.subprocess,
        "run",
        lambda *_args, **_kwargs: rollout_verifier.subprocess.CompletedProcess(
            [], 0, "".join(f"{node} Active\n" for node in sorted(nodes)), ""
        ),
    )
    monkeypatch.setattr(
        rollout_verifier,
        "_curl_json",
        lambda *_args: _application(replicas=3),
    )
    monkeypatch.setattr(
        rollout_verifier,
        "_service_spec",
        lambda _app_name: {
            "TaskTemplate": {
                "ContainerSpec": {
                    "Image": "registry.example/crawl4ai@sha256:target",
                    "Healthcheck": {"Test": rollout_verifier.CRAWL_HEALTHCHECK},
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
            "Mode": {"Replicated": {"Replicas": 3}},
            "EndpointSpec": {"Mode": "vip"},
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
        },
    )

    if eligible is not None:
        with pytest.raises(RuntimeError, match="admitted haiku-4/5/9/18"):
            verify_rollout_preflight(
                dokploy_url="https://dokploy.example",
                api_key="secret",
                application_id="app",
            )
    else:
        verify_rollout_preflight(
            dokploy_url="https://dokploy.example",
            api_key="secret",
            application_id="app",
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
                {"ok": True, "instance": instance} for instance in instances
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
    )

    verify_monitor_evidence(
        Path(evidence),
        frozenset({"new-instance"}),
        frozenset({"new-task"}),
    )


def test_curl_json_rejects_streamed_responses_over_64_kib():
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
            _curl_json(f"http://127.0.0.1:{server.server_port}/")
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
            "tolerate transient blips",
        ),
        (
            {"updateConfigSwarm": {"Order": "stop-first", "Parallelism": 1, "MaxFailureRatio": 0}},
            "updateConfigSwarm must use start-first",
        ),
        (
            {"rollbackConfigSwarm": {"Order": "stop-first", "Parallelism": 1, "MaxFailureRatio": 0}},
            "rollbackConfigSwarm must use start-first",
        ),
    ],
)
def test_verifier_rejects_invalid_release_observability_configuration(overrides, message):
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
            health_url="https://crawl.example/health",
            read_json=read_json,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mounts": []}, "crawl4ai-redis-data"),
        ({"healthCheckSwarm": {"Test": ["CMD-SHELL", "redis-cli ping"]}}, "healthcheck"),
        (
            {"command": "redis-server --dir /data --appendonly yes --appendfsync always"},
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
            health_url="https://crawl.example/health",
            read_json=read_json,
        )


def test_verifier_requires_two_identical_complete_task_and_instance_snapshots_with_runtime_labels():
    container_id = "a" * 64
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

    verify_rollout(
        dokploy_url="https://dokploy.example",
        api_key="secret",
        application_id="app",
        redis_application_id="redis",
        revision="target",
        health_url="https://crawl.example/health",
        read_json=read_json,
        probe_health=lambda _url, count: [_health(container_id[:12])] * count,
        sleep=lambda _seconds: None,
    )


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
            health_url="https://crawl.example/health",
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
            health_url="https://crawl.example/health",
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
            health_url="https://crawl.example/health",
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
            health_url="https://crawl.example/health",
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
            health_url="https://crawl.example/health",
            read_json=read_json,
            probe_health=lambda _url, count: [_health("a" * 12), _health("b" * 12)]
            * (count // 2),
            sleep=lambda _seconds: None,
            monotonic=lambda: next(clock),
            inspect_task_runtime=unexpected_inspection,
        )
