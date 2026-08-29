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
REDIS_IMAGE = (
    "redis@sha256:ff02b58f971e7d7d156a1267e283fcbbeee91773b6aa36c49dac28ecfe28eadf"
)
REDIS_COMMAND = [
    "redis-server",
    "--dir",
    "/data",
    "--appendonly",
    "yes",
    "--appendfsync",
    "everysec",
    "--loglevel",
    "notice",
]
REDIS_HEALTHCHECK_POLICY = {
    "Test": REDIS_HEALTHCHECK,
    "Interval": 10_000_000_000,
    "Timeout": 3_000_000_000,
    "StartPeriod": 10_000_000_000,
    "Retries": 3,
}
REDIS_CPU_RESERVATION = "250000000"
REDIS_CPU_LIMIT = "1000000000"
REDIS_MEMORY_RESERVATION = "268435456"
REDIS_MEMORY_LIMIT = "1073741824"
CRAWL_HEALTHCHECK = ["CMD", "curl", "-f", "http://localhost:11235/health"]
CRAWL_MAX_REPLICAS_PER_NODE = 1
CRAWL_NODE_CONSTRAINT = "node.labels.crawl4ai-eligible==true"
CRAWL_ELIGIBLE_NODES = frozenset({"haiku-4", "haiku-5", "haiku-9", "haiku-18"})
CRAWL_HOSTS = frozenset({"crawl4ai.haiku.host", "crawl4ai.popos-sf0.com"})
CRAWL_CPU_RESERVATION = "500000000"
CRAWL_CPU_LIMIT = "2000000000"
CRAWL_MEMORY_RESERVATION = "1073741824"
CRAWL_MEMORY_LIMIT = "4294967296"
CRAWL_ENDPOINT_SPEC = {"Mode": "vip", "Ports": []}
CRAWL_HEALTHCHECK_POLICY = {
    "Test": CRAWL_HEALTHCHECK,
    "Interval": 10_000_000_000,
    "Timeout": 5_000_000_000,
    "StartPeriod": 120_000_000_000,
    "Retries": 5,
}
ROLLOUT_DELAY_NS = 400_000_000_000
ROLLOUT_MONITOR_NS = ROLLOUT_DELAY_NS
STOP_GRACE_NS = 390_000_000_000
ROLLOUT_PROOF_TIMEOUT_SECONDS = 4_000
MONITOR_INTERVAL_SECONDS = 0.5
# A task that has not reached its container yet (new/pending/assigned/accepted/
# preparing) carries no Status.ContainerStatus, and Go template evaluation of a
# missing key exits 1 with "template parsing error". Every rolling update creates
# such a task, so the field must be read defensively.
TASK_CONTAINER_ID_FORMAT = (
    "{{if .Status.ContainerStatus}}{{json .Status.ContainerStatus.ContainerID}}"
    '{{else}}""{{end}}'
)


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


def _uses_native_rollback(application: dict[str, Any]) -> bool:
    return (
        application.get("rollbackActive") is True
        and bool(application.get("registryId"))
        and application.get("rollbackRegistryId") == application.get("registryId")
    )


def _verify_release_configuration(
    application: Any, revision: str, expected_image: str
) -> str:
    if not isinstance(application, dict):
        raise ValueError("application configuration response is invalid")
    if application.get("labelsSwarm") != _observability_labels(revision):
        raise ValueError("observability labels do not match the release")
    if application.get("replicas") != CRAWL_REPLICAS:
        raise ValueError("Crawl4AI must keep exactly three replicas")
    if application.get("dockerImage") != expected_image:
        raise ValueError("Crawl4AI image does not match the immutable release")
    if not _uses_native_rollback(application):
        raise ValueError("Crawl4AI must use its Nexus registry for native rollback")
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
    if application.get("healthCheckSwarm") != CRAWL_HEALTHCHECK_POLICY:
        raise ValueError("Crawl4AI must use the exact admission healthcheck policy")
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
    if application.get("dockerImage") != REDIS_IMAGE:
        raise ValueError("external Redis image must be immutable")
    mounts = application.get("mounts")
    if not isinstance(mounts, list) or not any(
        isinstance(mount, dict)
        and mount.get("type") == "volume"
        and mount.get("volumeName") == REDIS_VOLUME
        and mount.get("mountPath") == REDIS_MOUNT_PATH
        for mount in mounts
    ):
        raise ValueError("external Redis must mount crawl4ai-redis-data at /data")
    if application.get("healthCheckSwarm") != REDIS_HEALTHCHECK_POLICY:
        raise ValueError("external Redis must use the exact healthcheck policy")
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
    if application.get("replicas") != 1:
        raise ValueError("external Redis must have exactly one replica")
    for field, expected in (
        ("cpuReservation", REDIS_CPU_RESERVATION),
        ("cpuLimit", REDIS_CPU_LIMIT),
        ("memoryReservation", REDIS_MEMORY_RESERVATION),
        ("memoryLimit", REDIS_MEMORY_LIMIT),
    ):
        if application.get(field) != expected:
            raise ValueError(f"external Redis {field} does not match admission")


def _verify_redis_service_spec(spec: dict[str, Any]) -> None:
    task = spec.get("TaskTemplate")
    container = task.get("ContainerSpec") if isinstance(task, dict) else None
    if not isinstance(container, dict) or container.get("Image") != REDIS_IMAGE:
        raise ValueError("running Redis image differs from Dokploy")
    if container.get("Command") != REDIS_COMMAND:
        raise ValueError("running Redis command is not appendfsync everysec")
    if container.get("Healthcheck") != REDIS_HEALTHCHECK_POLICY:
        raise ValueError("running Redis healthcheck differs from Dokploy")
    mounts = container.get("Mounts")
    if not isinstance(mounts, list) or not any(
        mount.get("Type") == "volume"
        and mount.get("Source") == REDIS_VOLUME
        and mount.get("Target") == REDIS_MOUNT_PATH
        for mount in mounts
        if isinstance(mount, dict)
    ):
        raise ValueError("running Redis volume differs from Dokploy")
    placement = task.get("Placement") if isinstance(task, dict) else None
    if not isinstance(placement, dict) or placement != {
        "Constraints": [REDIS_NODE_CONSTRAINT],
        "MaxReplicas": 1,
    }:
        raise ValueError("running Redis placement differs from Dokploy")
    resources = task.get("Resources") if isinstance(task, dict) else None
    if (
        not isinstance(resources, dict)
        or resources.get("Reservations")
        != {
            "NanoCPUs": int(REDIS_CPU_RESERVATION),
            "MemoryBytes": int(REDIS_MEMORY_RESERVATION),
        }
        or resources.get("Limits")
        != {
            "NanoCPUs": int(REDIS_CPU_LIMIT),
            "MemoryBytes": int(REDIS_MEMORY_LIMIT),
        }
    ):
        raise ValueError("running Redis resources differ from Dokploy")


def _verify_current_rollout_source(application: Any) -> None:
    if not isinstance(application, dict) or "@sha256:" not in str(
        application.get("dockerImage", "")
    ):
        raise ValueError("current Crawl4AI artifact is not configured")
    if application.get("replicas") != CRAWL_REPLICAS:
        raise ValueError("current Crawl4AI service must have three replicas")
    if not _uses_native_rollback(application):
        raise ValueError("current Crawl4AI service lacks native rollback")
    if application.get("healthCheckSwarm") != CRAWL_HEALTHCHECK_POLICY:
        raise ValueError(
            "current Crawl4AI service lacks the exact admission healthcheck"
        )
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
            or config.get("Delay") != ROLLOUT_DELAY_NS
            or config.get("Monitor") != ROLLOUT_MONITOR_NS
        ):
            raise ValueError(f"current {field} is not a safe rollback source")
    if int(application.get("stopGracePeriodSwarm") or 0) < STOP_GRACE_NS:
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


def _service_update_state(app_name: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "service",
            "inspect",
            app_name,
            "--format",
            "{{.UpdateStatus.State}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _service_spec_with_retry(
    app_name: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
    while monotonic() < deadline:
        try:
            return _service_spec(app_name)
        except subprocess.CalledProcessError:
            sleep(5)
    raise TimeoutError("Docker service could not be inspected")


def _wait_for_verified_rollback(
    *,
    app_name: str,
    baseline_image: str,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    health_urls: tuple[str, ...],
    candidate_image: str | None = None,
    candidate_labels: dict[str, str] | None = None,
    read_json: Callable[[str, str | None], Any],
    probe_health: Callable[[str, int], list[Any]],
    inspect_task_runtime: Callable[
        [str, str], tuple[str, frozenset[str], dict[str, str], str]
    ],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> str:
    deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
    while monotonic() < deadline:
        try:
            state = _service_update_state(app_name)
        except subprocess.CalledProcessError:
            sleep(5)
            continue
        if state in {"completed", "rollback_completed"}:
            try:
                service_spec = _service_spec(app_name)
            except subprocess.CalledProcessError:
                sleep(5)
                continue
            try:
                _verify_current_service_spec(service_spec, baseline_image)
            except ValueError:
                sleep(5)
                continue
            tasks = read_json(
                _dokploy_url(
                    dokploy_url,
                    "docker.getServiceContainersByAppName",
                    appName=app_name,
                ),
                api_key,
            )
            actual_tasks = frozenset(
                task["containerId"]
                for task in tasks
                if task.get("state") == "running"
                and str(task.get("currentState", "")).startswith("Running ")
                and task.get("node")
                and not _has_task_error(task)
            )
            desired_tasks = frozenset(
                task["containerId"] for task in tasks if task.get("state") == "running"
            )
            running_nodes = frozenset(
                task["node"]
                for task in tasks
                if task.get("containerId") in actual_tasks
            )
            if (
                len(actual_tasks) != CRAWL_REPLICAS
                or actual_tasks != desired_tasks
                or len(running_nodes) != CRAWL_REPLICAS
                or not running_nodes <= CRAWL_ELIGIBLE_NODES
            ):
                raise RuntimeError("rollback task census did not converge")
            network = _dokploy_network_id()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=CRAWL_REPLICAS
            ) as executor:
                runtimes = list(
                    executor.map(
                        lambda task_id: inspect_task_runtime(task_id, network),
                        actual_tasks,
                    )
                )
            instances = frozenset(runtime[0] for runtime in runtimes)
            revisions = frozenset(
                runtime[2].get("otel.service.version") for runtime in runtimes
            )
            if (
                len(instances) != CRAWL_REPLICAS
                or len(revisions) != 1
                or None in revisions
                or any(
                    not runtime[1] or runtime[3] != baseline_image
                    for runtime in runtimes
                )
            ):
                raise RuntimeError("rollback runtime identity did not converge")
            revision = next(iter(revisions))
            for url in health_urls:
                responses = probe_health(url, CRAWL_REPLICAS * 4)
                if not all(
                    _is_exact_health(response, revision) for response in responses
                ):
                    raise RuntimeError("rollback public health did not converge")
                if (
                    frozenset(response["instance"] for response in responses)
                    != instances
                ):
                    raise RuntimeError("rollback public task coverage did not converge")
            verify_native_route(
                dokploy_url=dokploy_url,
                api_key=api_key,
                application_id=application_id,
                app_name=app_name,
            )
            restore_payload = {
                "applicationId": application_id,
                "dockerImage": baseline_image,
                "labelsSwarm": _observability_labels(revision),
            }
            if candidate_image and candidate_labels:
                restore_payload.update(
                    {
                        "expectedDockerImage": candidate_image,
                        "expectedLabelsSwarm": candidate_labels,
                    }
                )
            _post_json(
                f"{dokploy_url.rstrip('/')}/api/application.update",
                api_key,
                restore_payload,
            )
            restored_application = read_json(
                _dokploy_url(
                    dokploy_url,
                    "application.one",
                    applicationId=application_id,
                ),
                api_key,
            )
            if (
                not isinstance(restored_application, dict)
                or restored_application.get("dockerImage") != baseline_image
                or restored_application.get("labelsSwarm")
                != _observability_labels(revision)
            ):
                raise RuntimeError("Dokploy rollback metadata did not converge")
            return
        if state in {"paused", "rollback_paused"}:
            raise RuntimeError(f"Swarm rollback stopped in {state}")
        sleep(5)
    raise TimeoutError("Swarm rollback did not reach rollback_completed")


def _verify_current_service_spec(spec: dict[str, Any], expected_image: str) -> None:
    task = spec.get("TaskTemplate")
    container = task.get("ContainerSpec") if isinstance(task, dict) else None
    if not isinstance(container, dict) or container.get("Image") != expected_image:
        raise ValueError("current Docker service artifact differs from Dokploy")
    if container.get("Healthcheck") != CRAWL_HEALTHCHECK_POLICY:
        raise ValueError("current Docker service lacks the exact admission healthcheck")
    placement = task.get("Placement")
    if (
        not isinstance(placement, dict)
        or placement.get("MaxReplicas") != 1
        or placement.get("Constraints") != [CRAWL_NODE_CONSTRAINT]
    ):
        raise ValueError("current Docker service lacks one-task-per-node placement")
    resources = task.get("Resources")
    if (
        not isinstance(resources, dict)
        or resources.get("Reservations")
        != {
            "NanoCPUs": int(CRAWL_CPU_RESERVATION),
            "MemoryBytes": int(CRAWL_MEMORY_RESERVATION),
        }
        or resources.get("Limits")
        != {
            "NanoCPUs": int(CRAWL_CPU_LIMIT),
            "MemoryBytes": int(CRAWL_MEMORY_LIMIT),
        }
    ):
        raise ValueError("current Docker service resources differ from admission")
    if int(container.get("StopGracePeriod") or 0) < REQUIRED_STOP_GRACE_NS:
        raise ValueError("current Docker service stop grace is not drain-safe")
    mode = spec.get("Mode")
    if (
        not isinstance(mode, dict)
        or mode.get("Replicated", {}).get("Replicas") != CRAWL_REPLICAS
    ):
        raise ValueError("current Docker service does not have three replicas")
    endpoint = spec.get("EndpointSpec")
    if (
        not isinstance(endpoint, dict)
        or endpoint.get("Mode") != "vip"
        or endpoint.get("Ports")
    ):
        raise ValueError("current Docker service must use an unpublished native VIP")
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
    *,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    redis_application_id: str,
) -> None:
    node_inventory = _crawl_node_inventory()
    if node_inventory.keys() != CRAWL_ELIGIBLE_NODES:
        raise RuntimeError(
            "crawl4ai-eligible must match the admitted haiku-4/5/9/18 pool"
        )
    if any(state != ("Ready", "Active") for state in node_inventory.values()):
        raise RuntimeError("every crawl4ai-eligible node must be Ready and Active")
    application = _request_json(
        _dokploy_url(
            dokploy_url,
            "application.one",
            applicationId=application_id,
        ),
        api_key,
    )
    _verify_current_rollout_source(application)
    _verify_current_service_spec(
        _service_spec(application["appName"]), application["dockerImage"]
    )
    redis_application = _request_json(
        _dokploy_url(
            dokploy_url,
            "application.one",
            applicationId=redis_application_id,
        ),
        api_key,
    )
    _verify_redis_configuration(redis_application)
    _verify_redis_service_spec(_service_spec(redis_application["appName"]))


def _crawl_node_inventory() -> dict[str, tuple[str, str]]:
    nodes = subprocess.run(
        [
            "docker",
            "node",
            "ls",
            "--filter",
            "node.label=crawl4ai-eligible=true",
            "--format",
            "{{.Hostname}}\t{{.Status}}\t{{.Availability}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        hostname: (status, availability)
        for hostname, status, availability in (line.split("\t", 2) for line in nodes)
    }


def _required_stop_grace_ns() -> int:
    parser = configparser.RawConfigParser()
    parser.read(Path(__file__).with_name("supervisord.conf"))
    entrypoint = Path(__file__).with_name("entrypoint.sh").read_text()
    match = re.search(r"CRAWL4AI_DRAIN_DELAY_SECONDS:-([0-9]+)", entrypoint)
    if match is None:
        raise ValueError("entrypoint drain delay is not configured")
    return (
        parser.getint("program:gunicorn", "stopwaitsecs") + int(match.group(1)) + 1
    ) * 1_000_000_000


REQUIRED_STOP_GRACE_NS = _required_stop_grace_ns()


def _request_json(url: str, api_key: str | None = None) -> Any:
    """Fetch bounded JSON without exposing credentials in argv."""
    config = ""
    if api_key:
        escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
        config = f'header = "x-api-key: {escaped}"\n'
    try:
        response = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "--connect-timeout",
                "5",
                "--max-time",
                "15",
                "--max-filesize",
                str(MAX_RESPONSE_BYTES),
                "--config",
                "-",
                url,
            ],
            input=config.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("HTTP request failed") from error
    if response.returncode == 63:
        raise ValueError("HTTP response exceeds 64 KiB")
    if response.returncode != 0:
        raise RuntimeError("HTTP request failed")
    if len(response.stdout) > MAX_RESPONSE_BYTES:
        raise ValueError("HTTP response exceeds 64 KiB")
    return json.loads(response.stdout)


def _post_json(url: str, api_key: str, payload: dict[str, Any]) -> Any:
    escaped = api_key.replace("\\", "\\\\").replace('"', '\\"')
    try:
        response = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "--connect-timeout",
                "5",
                "--max-time",
                "15",
                "--max-filesize",
                str(MAX_RESPONSE_BYTES),
                "--request",
                "POST",
                "--header",
                "content-type: application/json",
                "--data",
                json.dumps(payload, separators=(",", ":")),
                "--config",
                "-",
                url,
            ],
            input=f'header = "x-api-key: {escaped}"\n'.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("HTTP request failed") from error
    if response.returncode == 63:
        raise ValueError("HTTP response exceeds 64 KiB")
    if response.returncode != 0:
        raise RuntimeError("HTTP request failed")
    return json.loads(response.stdout) if response.stdout else None


def _request_json_with_retry(
    url: str,
    api_key: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Any:
    deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
    while monotonic() < deadline:
        try:
            return _request_json(url, api_key)
        except RuntimeError:
            sleep(5)
    raise TimeoutError("Dokploy API request did not recover")


def submit_application_deploy(
    *,
    dokploy_url: str,
    api_key: str,
    application_id: str,
    baseline_image: str,
    candidate_image: str,
    candidate_revision: str,
    submission_id: str,
    submission_title: str,
    read_json: Callable[[str, str | None], Any] = _request_json,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    application_url = _dokploy_url(
        dokploy_url, "application.one", applicationId=application_id
    )
    baseline = read_json(application_url, api_key)
    if not isinstance(baseline, dict) or baseline.get("dockerImage") != baseline_image:
        raise RuntimeError("Dokploy baseline metadata does not match the service")
    baseline_labels = baseline.get("labelsSwarm")
    if not isinstance(baseline_labels, dict):
        raise RuntimeError("Dokploy baseline observability metadata is invalid")
    candidate_labels = _observability_labels(candidate_revision)
    update_url = f"{dokploy_url.rstrip('/')}/api/application.update"

    def wait_for_metadata(image: str, labels: dict[str, str]) -> None:
        deadline = monotonic() + 60
        while monotonic() < deadline:
            try:
                application = read_json(application_url, api_key)
                if (
                    isinstance(application, dict)
                    and application.get("dockerImage") == image
                    and application.get("labelsSwarm") == labels
                ):
                    return
            except (RuntimeError, ValueError):
                pass
            sleep(2)
        raise TimeoutError("Dokploy application metadata did not converge")

    def restore_baseline() -> None:
        deadline = monotonic() + 60
        while monotonic() < deadline:
            current = read_json(application_url, api_key)
            if not isinstance(current, dict):
                raise RuntimeError("Dokploy application metadata is invalid")
            if (
                current.get("dockerImage") == baseline_image
                and current.get("labelsSwarm") == baseline_labels
            ):
                return
            if (
                current.get("dockerImage") != candidate_image
                or current.get("labelsSwarm") != candidate_labels
            ):
                raise RuntimeError("Dokploy application metadata changed concurrently")
            try:
                _post_json(
                    update_url,
                    api_key,
                    {
                        "applicationId": application_id,
                        "expectedDockerImage": candidate_image,
                        "expectedLabelsSwarm": candidate_labels,
                        "dockerImage": baseline_image,
                        "labelsSwarm": baseline_labels,
                    },
                )
            except (RuntimeError, ValueError):
                pass
            sleep(2)
        raise TimeoutError("Dokploy baseline metadata did not converge")

    try:
        _post_json(
            update_url,
            api_key,
            {
                "applicationId": application_id,
                "expectedDockerImage": baseline_image,
                "expectedLabelsSwarm": baseline_labels,
                "dockerImage": candidate_image,
                "labelsSwarm": candidate_labels,
            },
        )
    except (RuntimeError, ValueError):
        pass
    try:
        wait_for_metadata(candidate_image, candidate_labels)
    except TimeoutError:
        restore_baseline()
        raise

    deployment_id = f"application-{application_id}-{submission_id}"
    deployment_url = f"{dokploy_url.rstrip('/')}/api/application.deploy"
    payload = {
        "applicationId": application_id,
        "title": submission_title,
        "idempotencyKey": submission_id,
        "expectedDockerImage": candidate_image,
        "expectedLabelsSwarm": candidate_labels,
    }
    try:
        _post_json(deployment_url, api_key, payload)
    except (RuntimeError, ValueError):
        pass
    return deployment_id


def _probe_health(
    url: str,
    count: int,
    read_json: Callable[[str, str | None], Any],
) -> list[Any]:
    def probe(_: int) -> Any:
        try:
            return read_json(f"{url}?rollout={uuid.uuid4()}", None)
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, count)) as executor:
        return list(executor.map(probe, range(count)))


def _dokploy_url(base_url: str, operation: str, **params: str) -> str:
    return f"{base_url.rstrip('/')}/api/{operation}?{urllib.parse.urlencode(params)}"


def verify_native_route(
    *, dokploy_url: str, api_key: str, application_id: str, app_name: str
) -> None:
    route = _request_json(
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
        raise ValueError(
            "Dokploy Swarm VIP transport must disable idle connection reuse"
        )
    if not routers:
        raise ValueError("Dokploy route has no routers")
    rules = []
    for router in routers.values():
        if not isinstance(router, dict) or not isinstance(router.get("service"), str):
            raise ValueError("Dokploy router has no service target")
        if not isinstance(router.get("rule"), str):
            raise ValueError("Dokploy router has no host rule")
        rules.append(router["rule"])
        service = services.get(router["service"])
        load_balancer = (
            service.get("loadBalancer") if isinstance(service, dict) else None
        )
        servers = (
            load_balancer.get("servers") if isinstance(load_balancer, dict) else None
        )
        urls = {
            server.get("url") for server in servers or [] if isinstance(server, dict)
        }
        if urls != {expected_url}:
            raise ValueError("Dokploy router does not target the Crawl4AI Swarm VIP")
        if load_balancer.get("serversTransport") != transport_name:
            raise ValueError("Dokploy router does not use the Swarm VIP transport")
    if any(not any(f"`{host}`" in rule for rule in rules) for host in CRAWL_HOSTS):
        raise ValueError("Dokploy route does not own both Crawl4AI hostnames")


def _public_health_sample(url: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        health = _request_json(f"{url}?rollout={uuid.uuid4()}")
        if not _is_exact_health_shape(health):
            raise ValueError("public health response is malformed")
        return {
            "ok": True,
            "url": url,
            "timestamp": time.time(),
            "latencySeconds": time.monotonic() - started,
            "instance": health["instance"],
            "revision": health["revision"],
        }
    except Exception as error:
        return {
            "ok": False,
            "url": url,
            "timestamp": time.time(),
            "latencySeconds": time.monotonic() - started,
            "error": type(error).__name__,
            "errorDetail": str(error)[:160],
        }


def _is_exact_health_shape(health: Any) -> bool:
    return (
        isinstance(health, dict)
        and bool(health.get("instance"))
        and bool(health.get("revision"))
        and health.get("status") == "ok"
    )


def _swarm_tasks(app_name: str, tracked_task_ids: set[str]) -> list[dict[str, Any]]:
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
    tasks = [
        task
        for line in result.stdout.splitlines()
        if line
        for task in (json.loads(line),)
        if _task_id(task) in tracked_task_ids
        or str(task.get("DesiredState", "")).lower() == "running"
    ]
    task_ids = [_task_id(task) for task in tasks]
    if task_ids:
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .ID}}\t{{json .Slot}}\t"
                + TASK_CONTAINER_ID_FORMAT
                + "\t{{json .Status.Timestamp}}\t{{json .Meta.UpdatedAt}}",
                *task_ids,
            ],
            capture_output=True,
            text=True,
        )
        error_lines = [line for line in inspected.stderr.splitlines() if line.strip()]
        missing_task_ids = {
            match.group(1)
            for line in error_lines
            if (
                match := re.fullmatch(
                    r"(?:Error: No such object|error: no such object): (\S+)", line
                )
            )
        }
        if inspected.returncode and (
            not error_lines
            or len(missing_task_ids) != len(error_lines)
            or not missing_task_ids <= set(task_ids)
        ):
            try:
                inspected.check_returncode()
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"docker inspect failed for {task_ids}: "
                    f"{inspected.stderr.strip()}"
                ) from error
        containers = {}
        for task_id, slot, container_id, status_timestamp, updated_at in (
            line.split("\t", 4)
            for line in inspected.stdout.splitlines()
            if line
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
    with (
        evidence_path.open("w") as evidence,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=len(health_urls) * 6
        ) as executor,
    ):
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
                "tasks": [task for task in tasks if _task_id(task) in tracked_task_ids],
            }
            evidence.write(json.dumps(sample, separators=(",", ":")) + "\n")
            evidence.flush()
            if not sample["ok"]:
                failed = [probe for probe in health if not probe["ok"]]
                raise RuntimeError(
                    f"public health monitor observed failed requests: {failed}"
                )
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
    final_revision: str,
    health_urls: frozenset[str],
) -> None:
    if evidence_path is None:
        return
    with evidence_path.open() as evidence:
        samples = [json.loads(line) for line in evidence if line.endswith("\n")]
    if not samples or any(not sample.get("ok") for sample in samples):
        raise RuntimeError("public health monitor did not remain successful")
    for url in health_urls:
        observed_instances = {
            probe.get("instance")
            for sample in samples
            for probe in sample["health"]
            if probe.get("url") == url and probe.get("revision") == final_revision
        }
        if not final_instances <= observed_instances:
            raise RuntimeError(
                f"public monitor did not observe the final task set through {url}"
            )
    baseline_instances: set[str] = set()
    for sample in samples:
        baseline_instances.update(probe["instance"] for probe in sample["health"])
        if len(baseline_instances) == len(final_instances):
            break
    if len(baseline_instances) != len(final_instances):
        raise RuntimeError(
            "public monitor did not capture the complete predecessor set"
        )
    if not baseline_instances.isdisjoint(final_instances):
        raise RuntimeError("public monitor did not prove full predecessor withdrawal")

    # A task without a container has not started and cannot be a predecessor
    # whose withdrawal this proof correlates.
    first_tasks_by_slot = {
        task.get("Slot"): _task_id(task)
        for task in samples[0]["tasks"]
        if str(task.get("DesiredState", "")).lower() == "running"
        and task.get("ContainerID")
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
        if "running" not in lifecycle:
            raise RuntimeError("monitor did not prove candidate admission")
        if len(slots) != 1:
            raise RuntimeError("monitor did not prove one stable slot for a candidate")
        slot = slots.pop()
        predecessor = first_tasks_by_slot.get(slot)
        if not predecessor:
            raise RuntimeError(
                "monitor did not correlate a candidate with its predecessor"
            )
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
            raise RuntimeError(
                "monitor did not prove per-slot admission before shutdown"
            )

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
            raise RuntimeError(
                "monitor did not observe predecessor withdrawal before exit"
            )
        withdrawn_task = next(
            task
            for task in samples[withdrawal]["tasks"]
            if task.get("ContainerID") == predecessor
        )
        if not str(withdrawn_task.get("CurrentState", "")).startswith("Running "):
            raise RuntimeError(
                "monitor did not observe a withdrawn predecessor still running"
            )
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
            TASK_CONTAINER_ID_FORMAT
            + "\t{{json .Spec.ContainerSpec.Labels}}\t"
            + "{{json .NetworksAttachments}}\t{{json .Spec.ContainerSpec.Image}}",
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
    return (
        json.loads(container)[:12],
        addresses,
        json.loads(labels) or {},
        json.loads(image),
    )


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
    expected_image: str,
    baseline_image: str,
    deployment_id: str = "deployment",
    health_urls: tuple[str, ...],
    read_json: Callable[[str, str | None], Any] = _request_json,
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
    if not health_urls:
        raise ValueError("at least one public health URL is required")
    if probe_health is None:

        def probe_health(url: str, count: int) -> list[Any]:
            return _probe_health(url, count, read_json)

    if baseline_force_update is not None:
        generation_deadline = monotonic() + 600
        while (
            _service_force_update(os.environ["APPLICATION_APP_NAME"])
            <= baseline_force_update
        ):
            if monotonic() >= generation_deadline:
                raise TimeoutError("Dokploy did not create a new service generation")
            sleep(5)
    deployment_deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
    while monotonic() < deployment_deadline:
        deployments = read_json(
            _dokploy_url(dokploy_url, "deployment.all", applicationId=application_id),
            api_key,
        )
        deployment = next(
            (
                deployment
                for deployment in deployments
                if deployment.get("deploymentId") == deployment_id
            ),
            None,
        )
        status = deployment["status"] if deployment else "none"
        if status == "done":
            break
        if status in {"error", "cancelled"}:
            raise RuntimeError(f"Dokploy reported a {status} deployment")
        sleep(15)
    else:
        raise TimeoutError(
            "Dokploy deployment did not finish within the rollout timeout"
        )

    expected_labels = _observability_labels(revision)
    network = _dokploy_network_id()

    proof_deadline = monotonic() + ROLLOUT_PROOF_TIMEOUT_SECONDS
    candidate: tuple[frozenset[str], frozenset[str], frozenset[str]] | None = None
    stable_rounds = 0
    while monotonic() < proof_deadline:
        application = read_json(
            _dokploy_url(dokploy_url, "application.one", applicationId=application_id),
            api_key,
        )
        _verify_release_configuration(application, revision, expected_image)
        app_name = application["appName"]
        update_state = _service_update_state(app_name)
        if update_state != "completed":
            if update_state in {"paused", "rollback_paused", "rollback_completed"}:
                raise RuntimeError(
                    f"Swarm update stopped before final convergence: {update_state}"
                )
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue
        redis_application = read_json(
            _dokploy_url(
                dokploy_url, "application.one", applicationId=redis_application_id
            ),
            api_key,
        )
        _verify_redis_configuration(redis_application)
        _verify_redis_service_spec(_service_spec(redis_application["appName"]))
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
        redis_desired = frozenset(
            task["containerId"]
            for task in redis_tasks
            if task.get("state") == "running"
        )
        if len(redis_running) != 1 or redis_running != redis_desired:
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue
        replicas = int(application["replicas"])
        if not 1 <= replicas <= MAX_REPLICAS:
            raise ValueError(
                f"configured replicas must be between 1 and {MAX_REPLICAS}"
            )
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
        node_inventory = _crawl_node_inventory()
        if node_inventory.keys() != CRAWL_ELIGIBLE_NODES or any(
            state != ("Ready", "Active") for state in node_inventory.values()
        ):
            raise ValueError("crawl4ai-eligible node inventory changed during rollout")
        running_nodes = frozenset(
            task["node"]
            for task in tasks
            if task["containerId"] in actual_running_tasks
        )
        if len(running_nodes) != replicas or not running_nodes <= CRAWL_ELIGIBLE_NODES:
            raise ValueError(
                "running Crawl4AI tasks must occupy distinct admitted nodes"
            )

        def inspect_task(task_id: str) -> tuple[str, frozenset[str]]:
            instance, addresses, labels, image = inspect_task_runtime(task_id, network)
            if not isinstance(labels, dict) or any(
                labels.get(key) != value for key, value in expected_labels.items()
            ):
                raise ValueError("container observability labels do not match release")
            if not addresses:
                raise ValueError("Crawl4AI task has no overlay address")
            if image != application["dockerImage"]:
                raise ValueError(
                    "running task image does not match the immutable release"
                )
            return instance, addresses

        with concurrent.futures.ThreadPoolExecutor(max_workers=replicas) as executor:
            inspected_tasks = list(executor.map(inspect_task, actual_running_tasks))
        authoritative_instances = frozenset(instance for instance, _ in inspected_tasks)
        authoritative_addresses = frozenset(
            address for _, addresses in inspected_tasks for address in addresses
        )

        public_instances_by_url = {}
        complete = True
        for url in health_urls:
            responses = probe_health(url, max(4, replicas * 4))
            matches = [_is_exact_health(health, revision) for health in responses]
            public_instances_by_url[url] = frozenset(
                health["instance"]
                for health, matches_revision in zip(responses, matches)
                if matches_revision
            )
            complete = complete and all(matches)
        complete = complete and all(
            instances == authoritative_instances
            for instances in public_instances_by_url.values()
        )
        public_instances = frozenset().union(*public_instances_by_url.values())
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
            _verify_current_service_spec(
                _service_spec(app_name), application["dockerImage"]
            )
            verify_native_route(
                dokploy_url=dokploy_url,
                api_key=api_key,
                application_id=application_id,
                app_name=app_name,
            )
            verify_monitor_evidence(
                monitor_evidence_path,
                authoritative_instances,
                frozenset(task[:12] for task in actual_running_tasks),
                revision,
                frozenset(health_urls),
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
    if arguments == ["submit"]:
        print(
            submit_application_deploy(
                dokploy_url=os.environ["DOKPLOY_URL"],
                api_key=os.environ["DOKPLOY_API_KEY"],
                application_id=os.environ["APPLICATION_ID"],
                baseline_image=os.environ["BASELINE_IMAGE"],
                candidate_image=f"{os.environ['IMAGE']}@{os.environ['IMAGE_DIGEST']}",
                candidate_revision=os.environ["GITHUB_SHA"],
                submission_id=os.environ["SUBMISSION_ID"],
                submission_title=os.environ["SUBMISSION_TITLE"],
            )
        )
        return
    if arguments == ["rollback"]:
        app_name = os.environ["APPLICATION_APP_NAME"]
        baseline_image = os.environ["BASELINE_IMAGE"]
        current_spec = _service_spec_with_retry(
            app_name, sleep=time.sleep, monotonic=time.monotonic
        )
        container = current_spec.get("TaskTemplate", {}).get("ContainerSpec", {})
        current_image = container.get("Image")
        if current_image != baseline_image:
            candidate_image = f"{os.environ['IMAGE']}@{os.environ['IMAGE_DIGEST']}"

            def owns_candidate(spec: dict[str, Any]) -> bool:
                candidate_container = spec.get("TaskTemplate", {}).get(
                    "ContainerSpec", {}
                )
                labels = (
                    candidate_container.get("Labels")
                    if isinstance(candidate_container, dict)
                    else None
                )
                return (
                    candidate_container.get("Image") == candidate_image
                    and isinstance(labels, dict)
                    and labels.get("dokploy.deployment.id")
                    == os.environ["DEPLOYMENT_ID"]
                    and all(
                        labels.get(key) == value
                        for key, value in _observability_labels(
                            os.environ["GITHUB_SHA"]
                        ).items()
                    )
                )

            if not owns_candidate(current_spec):
                raise RuntimeError("live service no longer matches this deployment")
            deployments = _request_json_with_retry(
                _dokploy_url(
                    os.environ["DOKPLOY_URL"],
                    "deployment.all",
                    applicationId=os.environ["APPLICATION_ID"],
                ),
                os.environ["DOKPLOY_API_KEY"],
                sleep=time.sleep,
                monotonic=time.monotonic,
            )
            deployment = next(
                (
                    deployment
                    for deployment in deployments
                    if deployment.get("deploymentId") == os.environ["DEPLOYMENT_ID"]
                ),
                None,
            )
            rollback_id = deployment.get("rollbackId") if deployment else None
            if not rollback_id:
                raise RuntimeError("Dokploy deployment has no rollback record")
            rollback_deadline = time.monotonic() + 60
            while True:
                try:
                    _post_json(
                        f"{os.environ['DOKPLOY_URL'].rstrip('/')}/api/rollback.rollback",
                        os.environ["DOKPLOY_API_KEY"],
                        {"rollbackId": rollback_id},
                    )
                    break
                except (RuntimeError, ValueError):
                    time.sleep(5)
                    observed = _service_spec_with_retry(
                        app_name, sleep=time.sleep, monotonic=time.monotonic
                    )
                    observed_image = (
                        observed.get("TaskTemplate", {})
                        .get("ContainerSpec", {})
                        .get("Image")
                    )
                    if observed_image == baseline_image:
                        break
                    if not owns_candidate(observed):
                        raise RuntimeError(
                            "live service no longer matches this deployment"
                        )
                    if _service_update_state(app_name) != "completed":
                        break
                    if time.monotonic() >= rollback_deadline:
                        raise TimeoutError(
                            "Dokploy rollback submission did not recover"
                        )
        _wait_for_verified_rollback(
            app_name=app_name,
            baseline_image=baseline_image,
            dokploy_url=os.environ["DOKPLOY_URL"],
            api_key=os.environ["DOKPLOY_API_KEY"],
            application_id=os.environ["APPLICATION_ID"],
            health_urls=tuple(os.environ["HEALTH_URLS"].split(",")),
            candidate_image=f"{os.environ['IMAGE']}@{os.environ['IMAGE_DIGEST']}",
            candidate_labels=_observability_labels(os.environ["GITHUB_SHA"]),
            read_json=_request_json,
            probe_health=lambda url, count: _probe_health(url, count, _request_json),
            inspect_task_runtime=_inspect_task_runtime,
            sleep=time.sleep,
            monotonic=time.monotonic,
        )
        print(json.dumps({"rollback": "verified", "image": baseline_image}))
        return
    if arguments == ["preflight"]:
        verify_rollout_preflight(
            dokploy_url=os.environ["DOKPLOY_URL"],
            api_key=os.environ["DOKPLOY_API_KEY"],
            application_id=os.environ["APPLICATION_ID"],
            redis_application_id=os.environ["REDIS_APPLICATION_ID"],
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
        expected_image=f"{os.environ['IMAGE']}@{os.environ['IMAGE_DIGEST']}",
        baseline_image=os.environ["BASELINE_IMAGE"],
        deployment_id=os.environ["DEPLOYMENT_ID"],
        health_urls=tuple(
            os.environ.get("HEALTH_URLS", "https://crawl4ai.haiku.host/health").split(
                ","
            )
        ),
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
