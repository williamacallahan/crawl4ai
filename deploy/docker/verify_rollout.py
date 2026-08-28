"""Fail closed until Dokploy and every public replica serve one exact revision."""

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
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 65_536
CRAWL_REPLICAS = 3
REDIS_VOLUME = "crawl4ai-redis-data"
REDIS_MOUNT_PATH = "/data"
REDIS_NODE_CONSTRAINT = "node.hostname==haiku-18"
CRAWL_HEALTHCHECK_TIMING = {
    "Interval": 1_000_000_000,
    "Timeout": 1_000_000_000,
    "StartPeriod": 120_000_000_000,
    "Retries": 1,
}
CRAWL_HEALTHCHECK_TEST = [
    "CMD",
    "curl",
    "-f",
    "http://localhost:11235/health/route",
]
CRAWL_READINESS_CHECK = {
    "Path": "/health/route",
    "Interval": 500_000_000,
    "UnhealthyInterval": 250_000_000,
    "Timeout": 400_000_000,
    "Status": 200,
}
CRAWL_MAX_REPLICAS_PER_NODE = 1
CRAWL_UPDATE_DELAY_NS = 150_000_000_000
CRAWL_UPDATE_MONITOR_NS = 150_000_000_000
ROLLOUT_PROOF_TIMEOUT_SECONDS = 900
MONITOR_INTERVAL_SECONDS = 0.5


def _observability_labels(revision: str) -> dict[str, str]:
    return {
        "otel.logs.enabled": "true",
        "otel.service.name": "crawl4ai",
        "otel.deployment.environment.name": "production",
        "otel.service.version": revision,
    }


def _runtime_service_state(state: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(state, dict):
        raise ValueError("application runtime state response is invalid")
    application = state.get("application")
    service = state.get("service")
    if not isinstance(application, dict) or not isinstance(service, dict):
        raise ValueError("application runtime state response is invalid")
    return application, service


def _has_native_routing(state: Any) -> bool:
    _, service = _runtime_service_state(state)
    root_labels = service.get("rootLabels")
    traefik = state.get("traefik") if isinstance(state, dict) else None
    routers = traefik.get("routers") if isinstance(traefik, dict) else None
    return (
        isinstance(root_labels, dict)
        and isinstance(routers, list)
        and root_labels.get("traefik.enable") == "true"
        and root_labels.get("traefik.swarm.network") == "dokploy-network"
        and root_labels.get("traefik.swarm.lbswarm") == "false"
        and any(
            key.endswith(".loadbalancer.healthcheck.initialstatus")
            and value == "down"
            for key, value in root_labels.items()
        )
        and any(
            isinstance(router, dict)
            and str(router.get("routerId", "")).endswith("@swarm")
            for router in routers
        )
        and not any(
            isinstance(router, dict)
            and str(router.get("routerId", "")).endswith("@file")
            for router in routers
        )
    )


def _verify_deployment_configuration(
    state: Any, revision: str, expected_image: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    application, service = _runtime_service_state(state)
    task_labels = service.get("taskLabels")
    expected_labels = _observability_labels(revision)
    if not isinstance(task_labels, dict) or any(
        task_labels.get(key) != value for key, value in expected_labels.items()
    ):
        raise ValueError("observability labels do not match the release")
    image = service.get("image")
    if (
        not isinstance(image, str)
        or not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image)
        or (expected_image is not None and image != expected_image)
    ):
        raise ValueError("Crawl4AI image does not match the release")
    if service.get("replicas") != CRAWL_REPLICAS:
        raise ValueError("Crawl4AI must keep exactly three replicas")
    placement = service.get("placement")
    if (
        not isinstance(placement, dict)
        or placement.get("MaxReplicas") != CRAWL_MAX_REPLICAS_PER_NODE
    ):
        raise ValueError("Crawl4AI placement must keep one replica per node")
    health_check = service.get("healthCheck")
    if (
        not isinstance(health_check, dict)
        or health_check.get("Test") != CRAWL_HEALTHCHECK_TEST
        or {
            key: health_check.get(key) for key in CRAWL_HEALTHCHECK_TIMING
        }
        != CRAWL_HEALTHCHECK_TIMING
    ):
        raise ValueError("Crawl4AI must use the routing admission healthcheck")
    for field in ("updateConfig", "rollbackConfig"):
        config = service.get(field)
        if not isinstance(config, dict) or config.get("Order") != "start-first":
            raise ValueError(f"{field} must use start-first order")
        if config.get("Parallelism") != 1 or config.get("MaxFailureRatio") != 0:
            raise ValueError(f"{field} must replace one task at a time and fail closed")
        failure_action = "rollback" if field == "updateConfig" else "pause"
        if config.get("FailureAction") != failure_action:
            raise ValueError(f"{field} must fail closed with {failure_action}")
        if int(config.get("Monitor") or 0) < CRAWL_UPDATE_MONITOR_NS:
            raise ValueError(f"{field} monitor must cover candidate startup")
        if int(config.get("Delay") or 0) < CRAWL_UPDATE_DELAY_NS:
            raise ValueError(f"{field} delay must preserve peers through candidate admission")
    if int(service.get("stopGracePeriod") or 0) < REQUIRED_STOP_GRACE_NS:
        raise ValueError(
            "configured Swarm stop grace must outlive the supervised process drain"
        )
    return application, service


def _verify_release_configuration(
    state: Any, revision: str, expected_image: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], tuple[frozenset[str], ...]]:
    application, service = _verify_deployment_configuration(
        state, revision, expected_image
    )
    root_labels = service.get("rootLabels")
    if not isinstance(root_labels, dict) or any(
        root_labels.get(key) != value
        for key, value in {
            "traefik.enable": "true",
            "traefik.swarm.network": "dokploy-network",
            "traefik.swarm.lbswarm": "false",
        }.items()
    ):
        raise ValueError("Crawl4AI must use native fail-closed Swarm readiness routing")
    readiness_services = {
        key.removesuffix(".healthcheck.path")
        for key in root_labels
        if key.endswith(".loadbalancer.healthcheck.path")
    }
    readiness_labels = {
        "server.port": "11235",
        "healthcheck.path": CRAWL_READINESS_CHECK["Path"],
        "healthcheck.interval": f'{CRAWL_READINESS_CHECK["Interval"]}ns',
        "healthcheck.unhealthyinterval": f'{CRAWL_READINESS_CHECK["UnhealthyInterval"]}ns',
        "healthcheck.timeout": f'{CRAWL_READINESS_CHECK["Timeout"]}ns',
        "healthcheck.status": str(CRAWL_READINESS_CHECK["Status"]),
        "healthcheck.initialstatus": "down",
    }
    if not readiness_services or any(
        root_labels.get(f"{prefix}.{suffix}") != value
        for prefix in readiness_services
        for suffix, value in readiness_labels.items()
    ):
        raise ValueError("Crawl4AI must use native fail-closed Swarm readiness routing")
    readiness_service_ids = {
        f'{prefix.removeprefix("traefik.http.services.").removesuffix(".loadbalancer")}@swarm'
        for prefix in readiness_services
    }
    readiness_router_ids = {
        f'{key.removeprefix("traefik.http.routers.").removesuffix(".rule")}@swarm'
        for key in root_labels
        if key.startswith("traefik.http.routers.") and key.endswith(".rule")
    }
    traefik = state.get("traefik")
    routers = traefik.get("routers") if isinstance(traefik, dict) else None
    services = traefik.get("services") if isinstance(traefik, dict) else None
    if not isinstance(routers, list) or not routers or not isinstance(services, list) or not services:
        raise ValueError("Crawl4AI Traefik runtime routing is unavailable")
    service_ids = {
        service.get("serviceId")
        for service in services
        if isinstance(service, dict)
        and service.get("status") == "enabled"
        and str(service.get("serviceId", "")).endswith("@swarm")
    }
    router_services = {
        router.get("service")
        for router in routers
        if isinstance(router, dict)
        and router.get("status") == "enabled"
        and str(router.get("routerId", "")).endswith("@swarm")
    }
    router_ids = {
        router.get("routerId") for router in routers if isinstance(router, dict)
    }
    if (
        len(service_ids) != len(services)
        or any(
            not isinstance(router, dict)
            or router.get("status") != "enabled"
            or not str(router.get("routerId", "")).endswith("@swarm")
            for router in routers
        )
        or service_ids != readiness_service_ids
        or router_ids != readiness_router_ids
        or router_services != service_ids
    ):
        raise ValueError("Crawl4AI Traefik routers and services are not enabled")
    admitted_services = []
    for traefik_service in services:
        server_status = traefik_service.get("serverStatus")
        if not isinstance(server_status, dict):
            raise ValueError("Crawl4AI Traefik server status is unavailable")
        admitted = set()
        for url, status in server_status.items():
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port != 11235:
                raise ValueError("Crawl4AI Traefik server status contains an invalid backend")
            if status == "UP":
                admitted.add(parsed.hostname)
        admitted_services.append(frozenset(admitted))
    return application, service, tuple(admitted_services)


def _verify_redis_configuration(
    state: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    application, service = _runtime_service_state(state)
    mounts = service.get("volumeMounts")
    if not isinstance(mounts, list) or not any(
        isinstance(mount, dict)
        and mount.get("Type") == "volume"
        and mount.get("Source") == REDIS_VOLUME
        and mount.get("Target") == REDIS_MOUNT_PATH
        for mount in mounts
    ):
        raise ValueError("external Redis must mount crawl4ai-redis-data at /data")
    if not isinstance(service.get("healthCheck"), dict):
        raise ValueError("external Redis must use redis-cli ping healthcheck")
    placement = service.get("placement")
    if not isinstance(placement, dict):
        raise ValueError("external Redis placement is not configured")
    constraints = placement.get("Constraints")
    if not isinstance(constraints, list) or REDIS_NODE_CONSTRAINT not in constraints:
        raise ValueError("external Redis must be placed on haiku-18")
    if placement.get("MaxReplicas") != 1:
        raise ValueError("external Redis MaxReplicas must be 1")
    if service.get("replicas") != 1:
        raise ValueError("external Redis must keep exactly one replica")
    return application, service


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


def _service_cursor(
    *,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    read_json: Callable[[str, str | None], Any],
) -> int:
    runtime = read_json(
        _dokploy_url(
            dokploy_url,
            "application.runtimeServiceState",
            applicationId=application_id,
        ),
        api_key,
    )
    _, service = _runtime_service_state(runtime)
    version = service.get("versionIndex")
    if not isinstance(version, int):
        raise ValueError("Crawl4AI service version is unavailable")
    return version


def _wait_for_service_generation(
    *,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    previous_service_version: int | None,
    read_json: Callable[[str, str | None], Any],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> None:
    if previous_service_version is None:
        return
    deadline = monotonic() + 600
    while monotonic() < deadline:
        if _service_cursor(
            dokploy_url=dokploy_url,
            api_key=api_key,
            application_id=application_id,
            read_json=read_json,
        ) > previous_service_version:
            return
        sleep(15)
    raise TimeoutError("Dokploy did not submit a new service generation within 10 minutes")


def verify_bootstrap(
    *,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    revision: str,
    expected_image: str,
    health_url: str,
    read_json: Callable[[str, str | None], Any] = _curl_json,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    previous_service_version: int | None = None,
) -> None:
    _wait_for_service_generation(
        dokploy_url=dokploy_url,
        api_key=api_key,
        application_id=application_id,
        read_json=read_json,
        sleep=sleep,
        monotonic=monotonic,
        previous_service_version=previous_service_version,
    )
    deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
    stable_snapshot: frozenset[str] | None = None
    stable_rounds = 0
    while monotonic() < deadline:
        runtime = read_json(
            _dokploy_url(
                dokploy_url,
                "application.runtimeServiceState",
                applicationId=application_id,
            ),
            api_key,
        )
        _, service = _verify_deployment_configuration(
            runtime, revision, expected_image
        )
        tasks = runtime.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("bootstrap runtime tasks are unavailable")
        instances = frozenset(
            str(task.get("status", {}).get("containerId", ""))[:12]
            for task in tasks
            if isinstance(task, dict)
            and task.get("desiredState") == "running"
            and task.get("status", {}).get("state") == "running"
            and task.get("status", {}).get("containerId")
        )
        if len(instances) != CRAWL_REPLICAS:
            stable_snapshot = None
            stable_rounds = 0
            sleep(5)
            continue
        responses = [
            read_json(f"{health_url}?bootstrap={uuid.uuid4()}", None)
            for _ in range(CRAWL_REPLICAS * 4)
        ]
        matches = [_is_exact_health(response, revision) for response in responses]
        public_instances = frozenset(
            response["instance"]
            for response, matches_revision in zip(responses, matches)
            if matches_revision
        )
        complete = public_instances == instances and all(matches)
        if complete and stable_snapshot == instances:
            stable_rounds += 1
        elif complete:
            stable_snapshot = instances
            stable_rounds = 1
        else:
            stable_snapshot = None
            stable_rounds = 0
        if stable_rounds >= 2:
            return
        sleep(5)
    raise TimeoutError("bootstrap rollout did not stabilize before native routing")


def monitor_public_health(
    *,
    health_url: str,
    expected_replicas: int,
    evidence_path: Path,
    armed_path: Path,
    stop_path: Path,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    require_native_routing: bool = False,
    require_native_path: Path | None = None,
    native_proof_path: Path | None = None,
    native_proof_ready_path: Path | None = None,
    read_json: Callable[[str, str | None], Any] = _curl_json,
) -> None:
    baseline_instances: set[str] = set()
    proof_instances: set[str] = set()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("w") as evidence:
        while not stop_path.exists():
            started = time.monotonic()
            native_proof = native_proof_path is not None and native_proof_path.exists()
            try:
                health = read_json(f"{health_url}?rollout={uuid.uuid4()}", None)
                revision = health.get("revision") if isinstance(health, dict) else None
                if not revision or not _is_exact_health(health, revision):
                    raise ValueError("public health response is malformed")
                runtime = read_json(
                    _dokploy_url(
                        dokploy_url,
                        "application.runtimeServiceState",
                        applicationId=application_id,
                    ),
                    api_key,
                )
                runtime_tasks = runtime.get("tasks") if isinstance(runtime, dict) else None
                traefik = runtime.get("traefik") if isinstance(runtime, dict) else None
                routers = traefik.get("routers") if isinstance(traefik, dict) else None
                services = traefik.get("services") if isinstance(traefik, dict) else None
                if (
                    not isinstance(runtime_tasks, list)
                    or not isinstance(routers, list)
                    or not isinstance(services, list)
                ):
                    raise ValueError("runtime routing timeline is unavailable")
                swarm_services = [
                    service
                    for service in services
                    if isinstance(service, dict)
                    and str(service.get("serviceId", "")).endswith("@swarm")
                ]
                up_addresses = sorted(
                    {
                        urllib.parse.urlsplit(url).hostname
                        for service in swarm_services
                        for url, status in (service.get("serverStatus") or {}).items()
                        if status == "UP" and urllib.parse.urlsplit(url).hostname
                    }
                )
                sample = {
                    "ok": True,
                    "timestamp": time.time(),
                    "latencySeconds": time.monotonic() - started,
                    "instance": health["instance"],
                    "revision": revision,
                    "tasks": runtime_tasks,
                    "nativeServices": len(swarm_services),
                    "legacyRouters": sum(
                        str(router.get("routerId", "")).endswith("@file")
                        for router in routers
                        if isinstance(router, dict)
                    ),
                    "upAddresses": up_addresses,
                    "proofPhase": "native" if native_proof else "continuity",
                }
            except Exception as error:
                sample = {
                    "ok": False,
                    "timestamp": time.time(),
                    "latencySeconds": time.monotonic() - started,
                    "error": type(error).__name__,
                }
            evidence.write(json.dumps(sample, separators=(",", ":")) + "\n")
            evidence.flush()
            os.fsync(evidence.fileno())
            if not sample["ok"]:
                raise RuntimeError("public health monitor observed a failed request")
            baseline_instances.add(sample["instance"])
            if native_proof:
                proof_instances.add(sample["instance"])
            task_addresses = {
                address.split("/", 1)[0]
                for task in sample["tasks"]
                if isinstance(task, dict)
                and task.get("status", {}).get("state") == "running"
                for address in task.get("addresses", [])
            }
            native_required = require_native_routing or (
                require_native_path is not None and require_native_path.exists()
            )
            native_admitted = (
                sample["nativeServices"] > 0
                and sample["legacyRouters"] == 0
                and task_addresses == set(sample["upAddresses"])
            )
            if (
                native_proof
                and len(proof_instances) >= expected_replicas
                and len(task_addresses) == expected_replicas
                and native_admitted
                and native_proof_ready_path is not None
                and not native_proof_ready_path.exists()
            ):
                native_proof_ready_path.touch()
            if (
                len(baseline_instances) >= expected_replicas
                and len(task_addresses) == expected_replicas
                and (not native_required or native_admitted)
                and not armed_path.exists()
            ):
                armed_path.touch()
            time.sleep(MONITOR_INTERVAL_SECONDS)


def verify_monitor_evidence(
    evidence_path: Path | None,
    final_instances: frozenset[str],
    final_containers: frozenset[str],
    final_addresses: frozenset[str],
) -> None:
    if evidence_path is None:
        return
    samples = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    if not samples or any(not sample.get("ok") for sample in samples):
        raise RuntimeError("public health monitor did not remain successful")
    proof_samples = [sample for sample in samples if sample.get("proofPhase") == "native"]
    if not proof_samples:
        raise RuntimeError("public monitor did not establish the native rollout baseline")
    observed_instances = list(
        dict.fromkeys(
            sample.get("instance")
            for sample in proof_samples
            if sample.get("instance")
        )
    )
    if not final_instances <= set(observed_instances):
        raise RuntimeError("public monitor did not observe every replacement task")
    baseline_instances = frozenset(observed_instances[: len(final_instances)])
    if len(baseline_instances) != len(final_instances):
        raise RuntimeError("public monitor did not capture the complete predecessor set")
    if not baseline_instances.isdisjoint(final_instances):
        raise RuntimeError("public monitor did not prove full predecessor withdrawal")

    runtime_samples = [
        sample
        for sample in proof_samples
        if isinstance(sample.get("tasks"), list)
        and isinstance(sample.get("upAddresses"), list)
    ]

    def running_task_addresses(
        sample: dict[str, Any],
    ) -> dict[str, frozenset[str]]:
        return {
            str(task["taskId"]): frozenset(
                address.split("/", 1)[0] for address in task.get("addresses", [])
            )
            for task in sample["tasks"]
            if isinstance(task, dict)
            and task.get("status", {}).get("state") == "running"
            and task.get("taskId")
            and task.get("addresses")
        }

    def task_addresses(tasks: dict[str, frozenset[str]]) -> set[str]:
        return set().union(*tasks.values()) if tasks else set()

    baseline_runtime = next(
        (
            sample
            for sample in runtime_samples
            if len(running_task_addresses(sample)) == len(final_containers)
            and task_addresses(running_task_addresses(sample))
            == set(sample["upAddresses"])
        ),
        None,
    )
    if baseline_runtime is None:
        raise RuntimeError("runtime monitor did not capture the admitted predecessor set")
    baseline_tasks = running_task_addresses(baseline_runtime)
    baseline_containers = frozenset(baseline_tasks)
    if not baseline_containers.isdisjoint(final_containers):
        raise RuntimeError("runtime monitor did not prove full predecessor replacement")
    if not any(
        frozenset(running_task_addresses(sample)) == final_containers
        and task_addresses(running_task_addresses(sample)) == final_addresses
        and final_addresses == set(sample["upAddresses"])
        for sample in runtime_samples
    ):
        raise RuntimeError("runtime monitor did not capture the admitted replacement set")

    for replacement in final_containers:
        first_seen = next(
            (
                index
                for index, sample in enumerate(runtime_samples)
                if replacement in running_task_addresses(sample)
            ),
            None,
        )
        if first_seen is None or not running_task_addresses(runtime_samples[first_seen])[
            replacement
        ].isdisjoint(runtime_samples[first_seen]["upAddresses"]):
            raise RuntimeError("runtime monitor did not prove fail-closed candidate admission")
        if not any(
            replacement in running_task_addresses(sample)
            and not running_task_addresses(sample)[replacement].isdisjoint(
                sample["upAddresses"]
            )
            for sample in runtime_samples[first_seen + 1 :]
        ):
            raise RuntimeError("runtime monitor did not observe candidate admission")

    for predecessor in baseline_containers:
        if not any(
            predecessor in running_task_addresses(sample)
            and running_task_addresses(sample)[predecessor].isdisjoint(
                sample["upAddresses"]
            )
            for sample in runtime_samples
        ):
            raise RuntimeError("runtime monitor did not prove predecessor withdrawal before exit")


def _inspect_task_runtime(
    task_id: str, network_id: str
) -> tuple[str, frozenset[str], dict[str, str]]:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            task_id,
            "--format",
            "{{json .Status.ContainerStatus.ContainerID}}\t"
            "{{json .Spec.ContainerSpec.Labels}}\t{{json .NetworksAttachments}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    container, labels, attachments = result.stdout.rstrip("\n").split("\t", 2)
    addresses = frozenset(
        address.split("/", 1)[0]
        for attachment in json.loads(attachments) or []
        if attachment.get("Network", {}).get("ID") == network_id
        for address in attachment.get("Addresses", [])
    )
    return json.loads(container)[:12], addresses, json.loads(labels) or {}


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
    monitor_evidence_path: Path | None = None,
    network_id: str | None = None,
    expected_image: str | None = None,
    previous_service_version: int | None = None,
    inspect_task_runtime: Callable[
        [str, str], tuple[str, frozenset[str], dict[str, str]]
    ]
    | None = None,
) -> None:
    """Require a stable task set and matching public instances at one revision."""
    inspect_task_runtime = inspect_task_runtime or _inspect_task_runtime
    _wait_for_service_generation(
        dokploy_url=dokploy_url,
        api_key=api_key,
        application_id=application_id,
        read_json=read_json,
        sleep=sleep,
        monotonic=monotonic,
        previous_service_version=previous_service_version,
    )

    expected_labels = _observability_labels(revision)
    if network_id is None:
        network_id = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                "dokploy-network",
                "--format",
                "{{.ID}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not network_id:
            raise ValueError("dokploy-network could not be resolved")

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
        runtime_state = read_json(
            _dokploy_url(
                dokploy_url,
                "application.runtimeServiceState",
                applicationId=application_id,
            ),
            api_key,
        )
        application, service, admitted_services = _verify_release_configuration(
            runtime_state, revision, expected_image
        )
        redis_runtime_state = read_json(
            _dokploy_url(
                dokploy_url,
                "application.runtimeServiceState",
                applicationId=redis_application_id,
            ),
            api_key,
        )
        redis_application, _ = _verify_redis_configuration(redis_runtime_state)
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
        replicas = int(service["replicas"])
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
            instance, addresses, labels = inspect_task_runtime(task_id, network_id)
            if not isinstance(labels, dict) or any(
                labels.get(key) != value for key, value in expected_labels.items()
            ):
                raise ValueError("container observability labels do not match release")
            if not addresses:
                raise ValueError("Crawl4AI task has no overlay address")
            return instance, addresses

        with concurrent.futures.ThreadPoolExecutor(max_workers=replicas) as executor:
            inspected_tasks = list(executor.map(inspect_task, actual_running_tasks))
        authoritative_instances = frozenset(instance for instance, _ in inspected_tasks)
        authoritative_addresses = frozenset(
            address for _, addresses in inspected_tasks for address in addresses
        )
        if any(admitted != authoritative_addresses for admitted in admitted_services):
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue

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
                actual_running_tasks,
                authoritative_addresses,
            )
            print(
                json.dumps(
                    {
                        "revision": revision,
                        "tasks": sorted(actual_running_tasks),
                        "instances": sorted(authoritative_instances),
                        "taskAddresses": sorted(authoritative_addresses),
                        "admittedTaskAddresses": sorted(authoritative_addresses),
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
    if arguments in (["monitor"], ["monitor-native"]):
        monitor_public_health(
            health_url=os.environ.get(
                "HEALTH_URL", "https://crawl4ai.haiku.host/health"
            ),
            expected_replicas=int(os.environ.get("EXPECTED_REPLICAS", "3")),
            evidence_path=Path(os.environ["ROLLOUT_MONITOR_PATH"]),
            armed_path=Path(os.environ["ROLLOUT_MONITOR_ARMED_PATH"]),
            stop_path=Path(os.environ["ROLLOUT_MONITOR_STOP_PATH"]),
            dokploy_url=os.environ["DOKPLOY_URL"],
            api_key=os.environ["DOKPLOY_API_KEY"],
            application_id=os.environ["APPLICATION_ID"],
            require_native_routing=arguments == ["monitor-native"],
            require_native_path=(
                Path(os.environ["ROLLOUT_MONITOR_REQUIRE_NATIVE_PATH"])
                if os.environ.get("ROLLOUT_MONITOR_REQUIRE_NATIVE_PATH")
                else None
            ),
            native_proof_path=(
                Path(os.environ["ROLLOUT_MONITOR_NATIVE_PROOF_PATH"])
                if os.environ.get("ROLLOUT_MONITOR_NATIVE_PROOF_PATH")
                else None
            ),
            native_proof_ready_path=(
                Path(os.environ["ROLLOUT_MONITOR_NATIVE_PROOF_READY_PATH"])
                if os.environ.get("ROLLOUT_MONITOR_NATIVE_PROOF_READY_PATH")
                else None
            ),
        )
        return
    if arguments == ["routing-mode"]:
        runtime = _curl_json(
            _dokploy_url(
                os.environ["DOKPLOY_URL"],
                "application.runtimeServiceState",
                applicationId=os.environ["APPLICATION_ID"],
            ),
            os.environ["DOKPLOY_API_KEY"],
        )
        print("native" if _has_native_routing(runtime) else "legacy")
        return
    if arguments == ["service-cursor"]:
        print(
            _service_cursor(
                dokploy_url=os.environ["DOKPLOY_URL"],
                api_key=os.environ["DOKPLOY_API_KEY"],
                application_id=os.environ["APPLICATION_ID"],
                read_json=_curl_json,
            )
        )
        return
    if arguments == ["bootstrap"]:
        verify_bootstrap(
            dokploy_url=os.environ["DOKPLOY_URL"],
            api_key=os.environ["DOKPLOY_API_KEY"],
            application_id=os.environ["APPLICATION_ID"],
            revision=os.environ["GITHUB_SHA"],
            expected_image=f'{os.environ["IMAGE"]}@{os.environ["IMAGE_DIGEST"]}',
            health_url=os.environ.get(
                "HEALTH_URL", "https://crawl4ai.haiku.host/health"
            ),
            previous_service_version=(
                int(os.environ["ROLLOUT_SERVICE_CURSOR"])
                if os.environ.get("ROLLOUT_SERVICE_CURSOR")
                else None
            ),
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
        expected_image=f'{os.environ["IMAGE"]}@{os.environ["IMAGE_DIGEST"]}',
        previous_service_version=(
            int(os.environ["ROLLOUT_SERVICE_CURSOR"])
            if os.environ.get("ROLLOUT_SERVICE_CURSOR")
            else None
        ),
    )


if __name__ == "__main__":
    main()
