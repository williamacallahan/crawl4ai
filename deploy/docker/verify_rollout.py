"""Prove one immutable Crawl4AI release through Dokploy's native Swarm VIP."""

from __future__ import annotations

import concurrent.futures
import configparser
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

MAX_RESPONSE_BYTES = 65_536
MAX_REPLICAS = 16
CRAWL_REPLICAS = 3
LLM_PROVIDER = "openai/qwen3.8-27b"
LLM_BASE_URL = "https://api.llm-gateway.iocloudhost.net/v1"
REDIS_VOLUME = "crawl4ai-redis-data"
REDIS_MOUNT_PATH = "/data"
REDIS_HEALTHCHECK = ["CMD", "redis-cli", "ping"]
REDIS_NODE_CONSTRAINT = "node.hostname==haiku-18"
CRAWL_HEALTHCHECK = ["CMD", "curl", "-f", "http://localhost:11235/health"]
CRAWL_HEALTHCHECK_MIN_RETRIES = 3
CRAWL_HEALTHCHECK_MIN_INTERVAL_NS = 5_000_000_000
CRAWL_MAX_REPLICAS_PER_NODE = 1
CRAWL_NODE_CONSTRAINT = "node.labels.crawl4ai-eligible==true"
CRAWL_ELIGIBLE_NODES = frozenset({"haiku-4", "haiku-5", "haiku-9", "haiku-18"})
CRAWL_CPU_RESERVATION = "500000000"
CRAWL_CPU_LIMIT = "2000000000"
CRAWL_MEMORY_RESERVATION = "1073741824"
CRAWL_MEMORY_LIMIT = "4294967296"
CRAWL_ENDPOINT_SPEC = {"Mode": "vip", "Ports": []}
ROLLOUT_DELAY_NS = 150_000_000_000
ROLLOUT_MONITOR_NS = 150_000_000_000
STOP_GRACE_NS = 390_000_000_000
ROLLOUT_PROOF_TIMEOUT_SECONDS = 900
MONITOR_INTERVAL_SECONDS = 0.5


def _observability_labels(revision: str) -> dict[str, str]:
    return {
        "otel.logs.enabled": "true",
        "otel.service.name": "crawl4ai",
        "otel.deployment.environment.name": "production",
        "otel.service.version": revision,
    }


def _environment_values(environment: str) -> dict[str, str]:
    values = {}
    for line in environment.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _verify_release_configuration(application: Any, revision: str) -> None:
    if not isinstance(application, dict):
        raise ValueError("application configuration response is invalid")
    if application.get("labelsSwarm") != _observability_labels(revision):
        raise ValueError("observability labels do not match the release")
    if application.get("replicas") != CRAWL_REPLICAS:
        raise ValueError("Crawl4AI must keep exactly three replicas")
    image = os.environ.get("IMAGE")
    digest = os.environ.get("IMAGE_DIGEST")
    if image and digest and application.get("dockerImage") != f"{image}@{digest}":
        raise ValueError("Crawl4AI image does not match the immutable release")
    environment = application.get("env")
    if not isinstance(environment, str):
        raise ValueError("LLM environment is not configured")
    values = _environment_values(environment)
    if values.get("LLM_PROVIDER") != LLM_PROVIDER:
        raise ValueError("LLM_PROVIDER does not match the release")
    if values.get("LLM_BASE_URL") != LLM_BASE_URL:
        raise ValueError("LLM_BASE_URL does not match the release")
    if not values.get("LLM_API_KEY", "").strip():
        raise ValueError("LLM_API_KEY must be nonempty")
    placement = application.get("placementSwarm")
    if (
        not isinstance(placement, dict)
        or placement.get("MaxReplicas") != CRAWL_MAX_REPLICAS_PER_NODE
    ):
        raise ValueError("Crawl4AI placement must keep one replica per node")
    if placement.get("Constraints") != [CRAWL_NODE_CONSTRAINT]:
        raise ValueError("Crawl4AI placement must use capacity-admitted nodes")
    for field, expected in (
        ("cpuReservation", CRAWL_CPU_RESERVATION),
        ("cpuLimit", CRAWL_CPU_LIMIT),
        ("memoryReservation", CRAWL_MEMORY_RESERVATION),
        ("memoryLimit", CRAWL_MEMORY_LIMIT),
    ):
        if application.get(field) != expected:
            raise ValueError(f"Crawl4AI {field} does not match the admitted envelope")
    if application.get("endpointSpecSwarm") != CRAWL_ENDPOINT_SPEC:
        raise ValueError("Crawl4AI must use the native Swarm VIP endpoint mode")
    if application.get("swarmVipConnectionReuse") is not False:
        raise ValueError("Crawl4AI must disable Traefik-to-VIP connection reuse")
    healthcheck = application.get("healthCheckSwarm")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != CRAWL_HEALTHCHECK:
        raise ValueError("Crawl4AI must use the routing admission healthcheck")
    # A 1s/1-retry probe kills tasks on a single transient /health 503;
    # require tolerance for short blips.
    if (
        healthcheck.get("Retries", 0) < CRAWL_HEALTHCHECK_MIN_RETRIES
        or healthcheck.get("Interval", 0) < CRAWL_HEALTHCHECK_MIN_INTERVAL_NS
    ):
        raise ValueError(
            "Crawl4AI healthcheck must tolerate transient blips (Retries>=3, Interval>=5s)"
        )
    for field, failure_action in (
        ("updateConfigSwarm", "rollback"),
        ("rollbackConfigSwarm", "pause"),
    ):
        config = application.get(field)
        if not isinstance(config, dict) or config.get("Order") != "start-first":
            raise ValueError(f"{field} must use start-first order")
        if config.get("Parallelism") != 1 or config.get("MaxFailureRatio") != 0:
            raise ValueError(f"{field} must replace one task at a time and fail closed")
        if config.get("FailureAction") != failure_action:
            raise ValueError(f"{field} has the wrong failure action")
        if (
            config.get("Delay") != ROLLOUT_DELAY_NS
            or config.get("Monitor") != ROLLOUT_MONITOR_NS
        ):
            raise ValueError(
                f"{field} must keep the qualified rollout delay and monitor"
            )
    if int(application.get("stopGracePeriodSwarm") or 0) < STOP_GRACE_NS:
        raise ValueError("Crawl4AI stop grace must outlive the complete drain window")


def _verify_redis_configuration(application: Any) -> None:
    if not isinstance(application, dict):
        raise ValueError("external Redis configuration response is invalid")
    mounts = application.get("mounts")
    if not isinstance(mounts, list) or not any(
        isinstance(mount, dict)
        and mount.get("type") == "volume"
        and mount.get("volumeName") == REDIS_VOLUME
        and mount.get("mountPath") == REDIS_MOUNT_PATH
        for mount in mounts
    ):
        raise ValueError("external Redis must mount crawl4ai-redis-data at /data")
    healthcheck = application.get("healthCheckSwarm")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != REDIS_HEALTHCHECK:
        raise ValueError("external Redis must use redis-cli ping healthcheck")
    command = application.get("command")
    if not isinstance(command, str) or "--appendfsync everysec" not in command:
        # fsync-per-write blocks the single Redis thread on disk stalls, which
        # 503s every replica's /health at once.
        raise ValueError("external Redis must persist with --appendfsync everysec")
    placement = application.get("placementSwarm")
    if not isinstance(placement, dict):
        raise ValueError("external Redis placement is not configured")
    constraints = placement.get("Constraints")
    if not isinstance(constraints, list) or REDIS_NODE_CONSTRAINT not in constraints:
        raise ValueError("external Redis must be placed on haiku-18")
    if placement.get("MaxReplicas") != 1:
        raise ValueError("external Redis MaxReplicas must be 1")


def _verify_current_rollout_source(application: Any) -> None:
    if (
        not isinstance(application, dict)
        or "@sha256:" not in str(application.get("dockerImage", ""))
    ):
        raise ValueError("current Crawl4AI artifact is not configured")
    if application.get("replicas") != CRAWL_REPLICAS:
        raise ValueError("current Crawl4AI service must have three replicas")
    healthcheck = application.get("healthCheckSwarm")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != CRAWL_HEALTHCHECK:
        raise ValueError("current Crawl4AI service lacks the admission healthcheck")
    placement = application.get("placementSwarm")
    if (
        not isinstance(placement, dict)
        or placement.get("MaxReplicas") != 1
        or placement.get("Constraints") != [CRAWL_NODE_CONSTRAINT]
    ):
        raise ValueError("current Crawl4AI service lacks one-task-per-node placement")
    if application.get("endpointSpecSwarm") != CRAWL_ENDPOINT_SPEC:
        raise ValueError("current Crawl4AI service does not use the native VIP")
    for field, expected in (
        ("cpuReservation", CRAWL_CPU_RESERVATION),
        ("cpuLimit", CRAWL_CPU_LIMIT),
        ("memoryReservation", CRAWL_MEMORY_RESERVATION),
        ("memoryLimit", CRAWL_MEMORY_LIMIT),
    ):
        if application.get(field) != expected:
            raise ValueError(f"current Crawl4AI {field} is not admission-safe")
    for field, failure_action in (
        ("updateConfigSwarm", "rollback"),
        ("rollbackConfigSwarm", "pause"),
    ):
        config = application.get(field)
        if (
            not isinstance(config, dict)
            or config.get("Order") != "start-first"
            or config.get("Parallelism") != 1
            or config.get("MaxFailureRatio") != 0
            or config.get("FailureAction") != failure_action
        ):
            raise ValueError(f"current {field} is not a safe rollback source")
    if int(application.get("stopGracePeriodSwarm") or 0) < REQUIRED_STOP_GRACE_NS:
        raise ValueError("current Crawl4AI stop grace is not drain-safe")


def _service_spec(app_name: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "service", "inspect", app_name, "--format", "{{json .Spec}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    spec = json.loads(result.stdout)
    if not isinstance(spec, dict):
        raise ValueError("current Crawl4AI service spec is invalid")
    return spec


def _verify_current_service_spec(
    spec: dict[str, Any], application: dict[str, Any]
) -> None:
    task = spec.get("TaskTemplate")
    container = task.get("ContainerSpec") if isinstance(task, dict) else None
    if not isinstance(container, dict) or container.get("Image") != application["dockerImage"]:
        raise ValueError("current Docker service artifact differs from Dokploy")
    healthcheck = container.get("Healthcheck")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != CRAWL_HEALTHCHECK:
        raise ValueError("current Docker service lacks the admission healthcheck")
    placement = task.get("Placement")
    if (
        not isinstance(placement, dict)
        or placement.get("MaxReplicas") != 1
        or placement.get("Constraints") != [CRAWL_NODE_CONSTRAINT]
    ):
        raise ValueError("current Docker service lacks one-task-per-node placement")
    resources = task.get("Resources")
    if not isinstance(resources, dict) or resources.get("Reservations") != {
        "NanoCPUs": int(CRAWL_CPU_RESERVATION),
        "MemoryBytes": int(CRAWL_MEMORY_RESERVATION),
    } or resources.get("Limits") != {
        "NanoCPUs": int(CRAWL_CPU_LIMIT),
        "MemoryBytes": int(CRAWL_MEMORY_LIMIT),
    }:
        raise ValueError("current Docker service resources differ from admission")
    if int(container.get("StopGracePeriod") or 0) < REQUIRED_STOP_GRACE_NS:
        raise ValueError("current Docker service stop grace is not drain-safe")
    mode = spec.get("Mode")
    if not isinstance(mode, dict) or mode.get("Replicated", {}).get("Replicas") != CRAWL_REPLICAS:
        raise ValueError("current Docker service does not have three replicas")
    endpoint = spec.get("EndpointSpec")
    if not isinstance(endpoint, dict) or endpoint.get("Mode") != "vip":
        raise ValueError("current Docker service does not use the native VIP")
    for field, failure_action in (
        ("UpdateConfig", "rollback"),
        ("RollbackConfig", "pause"),
    ):
        config = spec.get(field)
        if (
            not isinstance(config, dict)
            or config.get("Order") != "start-first"
            or config.get("Parallelism") != 1
            or config.get("FailureAction") != failure_action
            or config.get("MaxFailureRatio") != 0
            or int(config.get("Delay") or 0) < ROLLOUT_DELAY_NS
            or int(config.get("Monitor") or 0) < ROLLOUT_MONITOR_NS
        ):
            raise ValueError(f"current Docker {field} is not a safe rollback source")


def verify_rollout_preflight(
    *, dokploy_url: str, api_key: str, application_id: str
) -> None:
    nodes = subprocess.run(
        [
            "docker",
            "node",
            "ls",
            "--filter",
            "node.label=crawl4ai-eligible=true",
            "--filter",
            "status=ready",
            "--format",
            "{{.Hostname}} {{.Availability}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    eligible = {
        line.removesuffix(" Active") for line in nodes if line.endswith(" Active")
    }
    if eligible != CRAWL_ELIGIBLE_NODES:
        raise RuntimeError(
            "crawl4ai-eligible must match the admitted haiku-4/5/9/18 pool"
        )
    application = _curl_json(
        _dokploy_url(
            dokploy_url,
            "application.one",
            applicationId=application_id,
        ),
        api_key,
    )
    _verify_current_rollout_source(application)
    _verify_current_service_spec(_service_spec(application["appName"]), application)


def _required_stop_grace_ns() -> int:
    parser = configparser.RawConfigParser()
    parser.read(Path(__file__).with_name("supervisord.conf"))
    entrypoint = Path(__file__).with_name("entrypoint.sh").read_text()
    match = re.search(r"CRAWL4AI_DRAIN_DELAY_SECONDS:-([0-9]+)", entrypoint)
    if match is None:
        raise ValueError("entrypoint drain delay is not configured")
    return (
        parser.getint("program:gunicorn", "stopwaitsecs")
        + int(match.group(1))
        + 1
    ) * 1_000_000_000


REQUIRED_STOP_GRACE_NS = _required_stop_grace_ns()


def _curl_json(url: str, api_key: str | None = None) -> Any:
    """Fetch bounded JSON with a true transport deadline and no secret in argv."""
    config = ""
    if api_key:
        escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
        config = f'header = "x-api-key: {escaped}"\n'
    curl = subprocess.Popen(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--connect-timeout",
            "5",
            "--max-time",
            "15",
            "--config",
            "-",
            url,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert curl.stdin is not None and curl.stdout is not None
    curl.stdin.write(config.encode())
    curl.stdin.close()
    try:
        response = subprocess.run(
            ["head", "-c", str(MAX_RESPONSE_BYTES + 1)],
            stdin=curl.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=True,
        )
    finally:
        curl.stdout.close()
        if curl.poll() is None:
            curl.kill()
        curl.wait()
    if len(response.stdout) > MAX_RESPONSE_BYTES:
        raise ValueError("HTTP response exceeds 64 KiB")
    if curl.returncode != 0:
        raise RuntimeError("HTTP request failed")
    return json.loads(response.stdout)


def _dokploy_url(base_url: str, operation: str, **params: str) -> str:
    return f"{base_url.rstrip('/')}/api/{operation}?{urllib.parse.urlencode(params)}"


def verify_native_route(
    *, dokploy_url: str, api_key: str, application_id: str, app_name: str
) -> None:
    route = _curl_json(
        _dokploy_url(
            dokploy_url,
            "application.readTraefikConfig",
            applicationId=application_id,
        ),
        api_key,
    )
    if not isinstance(route, str):
        raise ValueError("Dokploy Traefik configuration response is invalid")
    config = yaml.safe_load(route)
    if not isinstance(config, dict) or not isinstance(config.get("http"), dict):
        raise ValueError("Dokploy Traefik configuration is malformed")
    http = config["http"]
    routers = http.get("routers")
    services = http.get("services")
    transports = http.get("serversTransports")
    if not all(isinstance(value, dict) for value in (routers, services, transports)):
        raise ValueError("Dokploy route is missing routers, services, or transports")
    expected_url = f"http://{app_name}:11235"
    transport_name = f"{app_name}-swarm-vip"
    if transports.get(transport_name, {}).get("maxIdleConnsPerHost") != -1:
        raise ValueError("Dokploy Swarm VIP transport must disable idle connection reuse")
    if not routers:
        raise ValueError("Dokploy route has no routers")
    for router in routers.values():
        if not isinstance(router, dict) or not isinstance(router.get("service"), str):
            raise ValueError("Dokploy router has no service target")
        service = services.get(router["service"])
        load_balancer = service.get("loadBalancer") if isinstance(service, dict) else None
        servers = load_balancer.get("servers") if isinstance(load_balancer, dict) else None
        urls = {
            server.get("url")
            for server in servers or []
            if isinstance(server, dict)
        }
        if urls != {expected_url}:
            raise ValueError("Dokploy router does not target the Crawl4AI Swarm VIP")
        if load_balancer.get("serversTransport") != transport_name:
            raise ValueError("Dokploy router does not use the Swarm VIP transport")
def _public_health_sample(url: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        health = _curl_json(f"{url}?rollout={uuid.uuid4()}")
        if not _is_exact_health_shape(health):
            raise ValueError("public health response is malformed")
        return {
            "ok": True,
            "timestamp": time.time(),
            "latencySeconds": time.monotonic() - started,
            "instance": health["instance"],
            "revision": health["revision"],
        }
    except Exception as error:
        return {
            "ok": False,
            "timestamp": time.time(),
            "latencySeconds": time.monotonic() - started,
            "error": type(error).__name__,
        }


def _is_exact_health_shape(health: Any) -> bool:
    return (
        isinstance(health, dict)
        and bool(health.get("instance"))
        and bool(health.get("revision"))
        and health.get("status") == "ok"
    )


def _swarm_tasks(
    app_name: str, tracked_task_ids: set[str]
) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "docker",
            "service",
            "ps",
            "--no-trunc",
            "--format",
            "{{json .}}",
            app_name,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tasks = [json.loads(line) for line in result.stdout.splitlines() if line]
    task_ids = [
        _task_id(task)
        for task in tasks
        if _task_id(task) in tracked_task_ids
        or str(task.get("DesiredState", "")).lower() == "running"
    ]
    if task_ids:
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .ID}}\t{{json .Slot}}\t"
                "{{json .Status.ContainerStatus.ContainerID}}\t"
                "{{json .Status.Timestamp}}\t{{json .Meta.UpdatedAt}}",
                *task_ids,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        containers = {}
        for task_id, slot, container_id, status_timestamp, updated_at in (
            line.split("\t", 4) for line in inspected.stdout.splitlines()
        ):
            container = json.loads(container_id)
            containers[json.loads(task_id)[:12]] = (
                json.loads(slot),
                container[:12] if container else "",
                json.loads(status_timestamp),
                json.loads(updated_at),
            )
        for task in tasks:
            slot, container, status_timestamp, updated_at = containers.get(
                _task_id(task), (None, "", None, None)
            )
            task["Slot"] = slot
            task["ContainerID"] = container
            task["StatusTimestamp"] = status_timestamp
            task["UpdatedAt"] = updated_at
    return tasks


def monitor_public_health(
    *,
    health_urls: tuple[str, ...],
    app_name: str,
    expected_replicas: int,
    evidence_path: Path,
    armed_path: Path,
    stop_path: Path,
) -> None:
    baseline_instances: set[str] = set()
    tracked_task_ids: set[str] = set()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("w") as evidence, concurrent.futures.ThreadPoolExecutor(
        max_workers=len(health_urls) * 6
    ) as executor:
        while not stop_path.exists():
            tasks = _swarm_tasks(app_name, tracked_task_ids)
            tracked_task_ids.update(
                _task_id(task)
                for task in tasks
                if str(task.get("DesiredState", "")).lower() == "running"
            )
            health = list(
                executor.map(
                    _public_health_sample,
                    health_urls * 6,
                )
            )
            sample = {
                "ok": all(probe["ok"] for probe in health),
                "timestamp": time.time(),
                "health": health,
                "tasks": [
                    task for task in tasks if _task_id(task) in tracked_task_ids
                ],
            }
            evidence.write(json.dumps(sample, separators=(",", ":")) + "\n")
            evidence.flush()
            os.fsync(evidence.fileno())
            if not sample["ok"]:
                raise RuntimeError("public health monitor observed a failed request")
            baseline_instances.update(probe["instance"] for probe in health)
            if len(baseline_instances) >= expected_replicas and not armed_path.exists():
                armed_path.touch()
            time.sleep(MONITOR_INTERVAL_SECONDS)


def _task_id(task: dict[str, Any]) -> str:
    return str(task.get("ID", task.get("Id", "")))[:12]


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify_monitor_evidence(
    evidence_path: Path | None,
    final_instances: frozenset[str],
    final_tasks: frozenset[str],
) -> None:
    if evidence_path is None:
        return
    samples = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    if not samples or any(not sample.get("ok") for sample in samples):
        raise RuntimeError("public health monitor did not remain successful")
    observed_instances = {
        probe.get("instance")
        for sample in samples
        for probe in sample["health"]
    }
    if not final_instances <= observed_instances:
        raise RuntimeError("public monitor did not observe every replacement task")
    baseline_instances: set[str] = set()
    for sample in samples:
        baseline_instances.update(
            probe["instance"] for probe in sample["health"]
        )
        if len(baseline_instances) == len(final_instances):
            break
    if len(baseline_instances) != len(final_instances):
        raise RuntimeError("public monitor did not capture the complete predecessor set")
    if not baseline_instances.isdisjoint(final_instances):
        raise RuntimeError("public monitor did not prove full predecessor withdrawal")

    first_tasks_by_slot = {
        task.get("Slot"): _task_id(task)
        for task in samples[0]["tasks"]
        if str(task.get("DesiredState", "")).lower() == "running"
    }
    first_tasks = set(first_tasks_by_slot.values())
    if first_tasks & final_tasks:
        raise RuntimeError("Swarm task identities did not fully roll over")
    for task_id in final_tasks:
        observed = [
            (index, task)
            for index, sample in enumerate(samples)
            for task in sample["tasks"]
            if _task_id(task) == task_id
        ]
        lifecycle = {
            str(task.get("CurrentState", "")).split(" ", 1)[0].lower()
            for _, task in observed
        }
        slots = {task.get("Slot") for _, task in observed}
        if "starting" not in lifecycle or "running" not in lifecycle:
            raise RuntimeError("monitor did not prove candidate STARTING-to-RUNNING admission")
        if len(slots) != 1:
            raise RuntimeError("monitor did not prove one stable slot for a candidate")
        slot = slots.pop()
        predecessor = first_tasks_by_slot.get(slot)
        if not predecessor:
            raise RuntimeError("monitor did not correlate a candidate with its predecessor")
        candidate_running = next(
            (index, task)
            for index, task in observed
            if str(task.get("CurrentState", "")).startswith("Running ")
        )
        predecessor_shutdown = next(
            (
                (index, task)
                for index, sample in enumerate(samples)
                for task in sample["tasks"]
                if _task_id(task) == predecessor
                and str(task.get("DesiredState", "")).lower() == "shutdown"
            ),
            None,
        )
        if predecessor_shutdown is None:
            raise RuntimeError("monitor did not observe predecessor shutdown")
        candidate_running_at = candidate_running[1].get("StatusTimestamp")
        predecessor_shutdown_at = predecessor_shutdown[1].get("UpdatedAt")
        if not candidate_running_at or not predecessor_shutdown_at:
            raise RuntimeError("monitor task timestamps are incomplete")
        if _timestamp(candidate_running_at) >= _timestamp(predecessor_shutdown_at):
            raise RuntimeError("monitor did not prove per-slot admission before shutdown")

    predecessor_instances = {
        task.get("ContainerID")
        for task in samples[0]["tasks"]
        if _task_id(task) in first_tasks
    }
    for predecessor in predecessor_instances:
        withdrawal = next(
            (
                index
                for index, sample in enumerate(samples)
                if any(
                    task.get("ContainerID") == predecessor
                    and str(task.get("DesiredState", "")).lower() == "shutdown"
                    for task in sample["tasks"]
                )
            ),
            None,
        )
        exit_index = next(
            (
                index
                for index, sample in enumerate(samples)
                if any(
                    task.get("ContainerID") == predecessor
                    and str(task.get("CurrentState", "")).split(" ", 1)[0].lower()
                    in {"complete", "failed", "shutdown"}
                    for task in sample["tasks"]
                )
            ),
            None,
        )
        if withdrawal is None or exit_index is None or withdrawal >= exit_index:
            raise RuntimeError("monitor did not observe predecessor withdrawal before exit")
        withdrawn_task = next(
            task
            for task in samples[withdrawal]["tasks"]
            if task.get("ContainerID") == predecessor
        )
        if not str(withdrawn_task.get("CurrentState", "")).startswith("Running "):
            raise RuntimeError("monitor did not observe a withdrawn predecessor still running")
        served_while_draining = {
            probe.get("instance")
            for sample in samples[withdrawal : exit_index + 1]
            for probe in sample["health"]
        }
        if predecessor in served_while_draining:
            raise RuntimeError("public routing still selected a withdrawn predecessor")


def _is_exact_health(health: Any, revision: str) -> bool:
    return (
        isinstance(health, dict)
        and bool(health.get("instance"))
        and health.get("revision") == revision
        and health.get("status") == "ok"
        and health.get("components", {}).get("api") == "ready"
        and health.get("components", {}).get("redis") == "ready"
    )


def _has_task_error(task: dict[str, Any]) -> bool:
    error = str(task.get("error", "")).strip()
    return bool(error.removeprefix("Error:").strip())


def _inspect_task_runtime(
    task_id: str, network_id: str
) -> tuple[str, frozenset[str], dict[str, str], str]:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            task_id,
            "--format",
            "{{json .Status.ContainerStatus.ContainerID}}\t"
            "{{json .Spec.ContainerSpec.Labels}}\t{{json .NetworksAttachments}}\t"
            "{{json .Spec.ContainerSpec.Image}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    container, labels, attachments, image = result.stdout.rstrip("\n").split("\t", 3)
    addresses = frozenset(
        address.split("/", 1)[0]
        for attachment in json.loads(attachments) or []
        if attachment.get("Network", {}).get("ID") == network_id
        for address in attachment.get("Addresses", [])
    )
    return json.loads(container)[:12], addresses, json.loads(labels) or {}, json.loads(image)


def _dokploy_network_id() -> str:
    network = subprocess.run(
        ["docker", "network", "inspect", "dokploy-network", "--format", "{{.ID}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not network:
        raise ValueError("dokploy-network could not be resolved")
    return network


def _service_force_update(app_name: str) -> int:
    result = subprocess.run(
        [
            "docker",
            "service",
            "inspect",
            app_name,
            "--format",
            "{{.Spec.TaskTemplate.ForceUpdate}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def verify_rollout(
    *,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    redis_application_id: str,
    revision: str,
    health_url: str,
    read_json: Callable[[str, str | None], Any] = _curl_json,
    probe_health: Callable[[str, int], list[Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    baseline_force_update: int | None = None,
    monitor_evidence_path: Path | None = None,
    inspect_task_runtime: Callable[
        [str, str], tuple[str, frozenset[str], dict[str, str], str]
    ]
    | None = None,
) -> None:
    """Require a stable task set and matching public instances at one revision."""
    inspect_task_runtime = inspect_task_runtime or _inspect_task_runtime
    if baseline_force_update is not None:
        generation_deadline = monotonic() + 600
        while _service_force_update(os.environ["APPLICATION_APP_NAME"]) <= baseline_force_update:
            if monotonic() >= generation_deadline:
                raise TimeoutError("Dokploy did not create a new service generation")
            sleep(5)
    deployment_deadline = monotonic() + 600
    while monotonic() < deployment_deadline:
        deployments = read_json(
            _dokploy_url(dokploy_url, "deployment.all", applicationId=application_id),
            api_key,
        )
        status = deployments[0]["status"] if deployments else "none"
        if status == "done":
            break
        if status == "error":
            raise RuntimeError("Dokploy reported a failed deployment")
        sleep(15)
    else:
        raise TimeoutError("Dokploy deployment did not finish within 10 minutes")

    expected_labels = _observability_labels(revision)
    network = _dokploy_network_id()

    if probe_health is None:

        def probe_health(url: str, count: int) -> list[Any]:
            def probe(_: int) -> Any:
                try:
                    return read_json(f"{url}?rollout={uuid.uuid4()}", None)
                except Exception:
                    return None

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(16, count)
            ) as executor:
                return list(executor.map(probe, range(count)))

    proof_deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
    candidate: tuple[frozenset[str], frozenset[str], frozenset[str]] | None = None
    stable_rounds = 0
    while monotonic() < proof_deadline:
        application = read_json(
            _dokploy_url(dokploy_url, "application.one", applicationId=application_id),
            api_key,
        )
        _verify_release_configuration(application, revision)
        redis_application = read_json(
            _dokploy_url(
                dokploy_url, "application.one", applicationId=redis_application_id
            ),
            api_key,
        )
        _verify_redis_configuration(redis_application)
        redis_tasks = read_json(
            _dokploy_url(
                dokploy_url,
                "docker.getServiceContainersByAppName",
                appName=redis_application["appName"],
            ),
            api_key,
        )
        redis_running = frozenset(
            task["containerId"]
            for task in redis_tasks
            if task.get("state") == "running"
            and str(task.get("currentState", "")).startswith("Running ")
            and task.get("node")
            and not _has_task_error(task)
        )
        if len(redis_running) != 1:
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue
        replicas = int(application["replicas"])
        if not 1 <= replicas <= MAX_REPLICAS:
            raise ValueError(
                f"configured replicas must be between 1 and {MAX_REPLICAS}"
            )
        stop_grace_ns = int(application.get("stopGracePeriodSwarm") or 0)
        if stop_grace_ns < REQUIRED_STOP_GRACE_NS:
            raise ValueError(
                "configured Swarm stop grace must outlive the supervised process drain"
            )
        app_name = application["appName"]
        tasks = read_json(
            _dokploy_url(
                dokploy_url, "docker.getServiceContainersByAppName", appName=app_name
            ),
            api_key,
        )
        actual_running_tasks = frozenset(
            task["containerId"]
            for task in tasks
            if str(task.get("currentState", "")).startswith("Running ")
            and task.get("node")
            and not _has_task_error(task)
        )
        desired_running_tasks = frozenset(
            task["containerId"] for task in tasks if task.get("state") == "running"
        )
        if (
            len(actual_running_tasks) != replicas
            or actual_running_tasks != desired_running_tasks
        ):
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue

        def inspect_task(task_id: str) -> tuple[str, frozenset[str]]:
            instance, addresses, labels, image = inspect_task_runtime(task_id, network)
            if not isinstance(labels, dict) or any(
                labels.get(key) != value for key, value in expected_labels.items()
            ):
                raise ValueError("container observability labels do not match release")
            if not addresses:
                raise ValueError("Crawl4AI task has no overlay address")
            if image != application["dockerImage"]:
                raise ValueError("running task image does not match the immutable release")
            return instance, addresses

        with concurrent.futures.ThreadPoolExecutor(max_workers=replicas) as executor:
            inspected_tasks = list(executor.map(inspect_task, actual_running_tasks))
        authoritative_instances = frozenset(instance for instance, _ in inspected_tasks)
        authoritative_addresses = frozenset(
            address for _, addresses in inspected_tasks for address in addresses
        )

        responses = probe_health(health_url, max(4, replicas * 4))
        matches = [_is_exact_health(health, revision) for health in responses]
        public_instances = frozenset(
            health["instance"]
            for health, matches_revision in zip(responses, matches)
            if matches_revision
        )
        complete = public_instances == authoritative_instances and all(matches)
        snapshot = (actual_running_tasks, public_instances, redis_running)
        if complete and snapshot == candidate:
            stable_rounds += 1
        elif complete:
            candidate = snapshot
            stable_rounds = 1
        else:
            candidate = None
            stable_rounds = 0
        if stable_rounds >= 2:
            verify_monitor_evidence(
                monitor_evidence_path,
                authoritative_instances,
                frozenset(task[:12] for task in actual_running_tasks),
            )
            print(
                json.dumps(
                    {
                        "revision": revision,
                        "tasks": sorted(actual_running_tasks),
                        "instances": sorted(authoritative_instances),
                        "vipTaskAddresses": sorted(authoritative_addresses),
                    },
                    separators=(",", ":"),
                )
            )
            return
        sleep(5)
    raise TimeoutError(
        f"exact rollout did not stabilize at {revision}; last snapshot: {candidate}"
    )


def main(arguments: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments == ["monitor"]:
        monitor_public_health(
            health_urls=tuple(
                os.environ.get(
                    "HEALTH_URLS", "https://crawl4ai.haiku.host/health"
                ).split(",")
            ),
            app_name=os.environ["APPLICATION_APP_NAME"],
            expected_replicas=CRAWL_REPLICAS,
            evidence_path=Path(os.environ["ROLLOUT_MONITOR_PATH"]),
            armed_path=Path(os.environ["ROLLOUT_MONITOR_ARMED_PATH"]),
            stop_path=Path(os.environ["ROLLOUT_MONITOR_STOP_PATH"]),
        )
        return
    if arguments == ["route"]:
        verify_native_route(
            dokploy_url=os.environ["DOKPLOY_URL"],
            api_key=os.environ["DOKPLOY_API_KEY"],
            application_id=os.environ["APPLICATION_ID"],
            app_name=os.environ["APPLICATION_APP_NAME"],
        )
        return
    if arguments == ["preflight"]:
        verify_rollout_preflight(
            dokploy_url=os.environ["DOKPLOY_URL"],
            api_key=os.environ["DOKPLOY_API_KEY"],
            application_id=os.environ["APPLICATION_ID"],
        )
        return
    if arguments:
        raise ValueError(f"Unsupported verifier arguments: {arguments}")
    verify_rollout(
        dokploy_url=os.environ["DOKPLOY_URL"],
        api_key=os.environ["DOKPLOY_API_KEY"],
        application_id=os.environ["APPLICATION_ID"],
        redis_application_id=os.environ["REDIS_APPLICATION_ID"],
        revision=os.environ["GITHUB_SHA"],
        health_url=os.environ.get("HEALTH_URL", "https://crawl4ai.haiku.host/health"),
        monitor_evidence_path=(
            Path(os.environ["ROLLOUT_MONITOR_PATH"])
            if os.environ.get("ROLLOUT_MONITOR_PATH")
            else None
        ),
        baseline_force_update=(
            int(os.environ["BASELINE_FORCE_UPDATE"])
            if os.environ.get("BASELINE_FORCE_UPDATE")
            else None
        ),
    )


if __name__ == "__main__":
    main()
