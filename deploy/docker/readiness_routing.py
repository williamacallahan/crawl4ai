"""Route Crawl4AI's public domains through its readiness-aware ingress."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 65_536
INGRESS_REPLICAS = 1
INGRESS_NODE_CONSTRAINT = "node.hostname==haiku-0"
MIN_CRAWLER_BACKENDS = 2
INGRESS_HEALTHCHECK = [
    "CMD",
    "haproxy",
    "-c",
    "-f",
    "/usr/local/etc/haproxy/haproxy.cfg",
]
MONITOR_INTERVAL_SECONDS = 0.5


def patch_traefik_config(
    raw_config: str, app_name: str, ingress_app_name: str
) -> str:
    source_url = f"http://{app_name}:11235"
    ingress_url = f"http://{ingress_app_name}:11235"
    if source_url not in raw_config and ingress_url not in raw_config:
        raise ValueError("No Crawl4AI Traefik service was found")
    return raw_config.replace(source_url, ingress_url)


def verify_traefik_config(
    raw_config: str, app_name: str, ingress_app_name: str
) -> None:
    if f"http://{ingress_app_name}:11235" not in raw_config:
        raise ValueError("Traefik does not route through Crawl4AI ingress")
    if f"http://{app_name}:11235" in raw_config:
        raise ValueError("Traefik still routes directly through the Crawl4AI VIP")


def _dokploy_request(
    base_url: str,
    api_key: str,
    operation: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    **params: str,
) -> Any:
    url = f"{base_url.rstrip('/')}/api/{operation}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"x-api-key": api_key, "content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Dokploy response exceeds 64 KiB")
    return json.loads(raw) if raw else None


def configure_readiness_routing(
    *,
    base_url: str,
    api_key: str,
    application_id: str,
    app_name: str,
    ingress_app_name: str,
) -> None:
    raw_config = _dokploy_request(
        base_url,
        api_key,
        "application.readTraefikConfig",
        applicationId=application_id,
    )
    updated_config = patch_traefik_config(raw_config, app_name, ingress_app_name)
    if updated_config == raw_config:
        verify_traefik_config(raw_config, app_name, ingress_app_name)
        return
    _dokploy_request(
        base_url,
        api_key,
        "application.updateTraefikConfig",
        method="POST",
        payload={"applicationId": application_id, "traefikConfig": updated_config},
    )
    verify_readiness_routing(
        base_url=base_url,
        api_key=api_key,
        application_id=application_id,
        app_name=app_name,
        ingress_app_name=ingress_app_name,
    )


def verify_readiness_routing(
    *,
    base_url: str,
    api_key: str,
    application_id: str,
    app_name: str,
    ingress_app_name: str,
) -> None:
    raw_config = _dokploy_request(
        base_url,
        api_key,
        "application.readTraefikConfig",
        applicationId=application_id,
    )
    verify_traefik_config(raw_config, app_name, ingress_app_name)


def _public_health_sample(url: str) -> dict[str, Any]:
    started = time.monotonic()
    request_url = f"{url}?rollout={uuid.uuid4()}"
    try:
        with urllib.request.urlopen(request_url, timeout=8) as response:
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if status != 200:
            raise RuntimeError(f"unexpected HTTP status {status}")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("HTTP response exceeds 64 KiB")
        health = json.loads(raw)
        if (
            not isinstance(health, dict)
            or health.get("status") != "ok"
            or not health.get("instance")
            or not health.get("revision")
        ):
            raise ValueError("public health response is malformed")
        return {
            "ok": True,
            "timestamp": time.time(),
            "latencySeconds": time.monotonic() - started,
            "status": status,
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


def ingress_backends(stats_url: str) -> list[str]:
    with urllib.request.urlopen(stats_url, timeout=5) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("HAProxy stats response exceeds 64 KiB")
    lines = raw.decode().splitlines()
    if not lines:
        raise ValueError("HAProxy stats response is empty")
    lines[0] = lines[0].removeprefix("# ")
    backends = sorted(
        row["addr"].rsplit(":", 1)[0]
        for row in csv.DictReader(lines)
        if row.get("pxname") == "crawl4ai"
        and row.get("svname", "").startswith("crawler")
        and row.get("status") == "UP"
        and row.get("addr")
    )
    if not backends:
        raise ValueError("HAProxy has no admitted Crawl4AI backends")
    return backends


def monitor_public_health(
    *,
    health_url: str,
    stats_url: str,
    expected_replicas: int,
    evidence_path: Path,
    armed_path: Path,
    stop_path: Path,
) -> None:
    baseline_instances: set[str] = set()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("w") as evidence:
        while not stop_path.exists():
            sample = _public_health_sample(health_url)
            try:
                sample["backends"] = ingress_backends(stats_url)
            except Exception as error:
                sample.update(ok=False, ingressError=type(error).__name__)
            evidence.write(json.dumps(sample, separators=(",", ":")) + "\n")
            evidence.flush()
            os.fsync(evidence.fileno())
            if not sample["ok"]:
                raise RuntimeError("public health monitor observed a failed request")
            baseline_instances.add(sample["instance"])
            if len(baseline_instances) >= expected_replicas and not armed_path.exists():
                armed_path.touch()
            time.sleep(MONITOR_INTERVAL_SECONDS)


def verify_monitor_evidence(
    evidence_path: Path | None,
    final_instances: frozenset[str],
    final_backends: frozenset[str],
) -> None:
    if evidence_path is None:
        return
    samples = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    if not samples or any(not sample.get("ok") for sample in samples):
        raise RuntimeError("public health monitor did not remain successful")
    observed_instances = {sample.get("instance") for sample in samples}
    if not final_instances <= observed_instances:
        raise RuntimeError("public monitor did not observe every replacement task")
    baseline_instances: set[str] = set()
    for sample in samples:
        if sample.get("instance"):
            baseline_instances.add(sample["instance"])
        if len(baseline_instances) == len(final_instances):
            break
    if len(baseline_instances) != len(final_instances):
        raise RuntimeError("public monitor did not capture the complete predecessor set")
    if not baseline_instances.isdisjoint(final_instances):
        raise RuntimeError("public monitor did not prove full predecessor withdrawal")
    if frozenset(samples[-1].get("backends", [])) != final_backends:
        raise RuntimeError("public monitor ended on the wrong admitted backend set")


def _verify_ingress_configuration(application: Any, image: str, revision: str) -> None:
    if not isinstance(application, dict):
        raise ValueError("ingress configuration response is invalid")
    if application.get("dockerImage") != f"{image}:{revision}":
        raise ValueError("ingress image does not match the release")
    if application.get("replicas") != INGRESS_REPLICAS:
        raise ValueError("ingress must keep one replica")
    healthcheck = application.get("healthCheckSwarm")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != INGRESS_HEALTHCHECK:
        raise ValueError("ingress healthcheck does not match the release")
    placement = application.get("placementSwarm")
    constraints = placement.get("Constraints") if isinstance(placement, dict) else None
    if (
        not isinstance(constraints, list)
        or INGRESS_NODE_CONSTRAINT not in constraints
        or placement.get("MaxReplicas") != 1
    ):
        raise ValueError("ingress must stay in the Traefik failure domain")
    for field in ("updateConfigSwarm", "rollbackConfigSwarm"):
        config = application.get(field)
        if not isinstance(config, dict) or config.get("Order") != "stop-first":
            raise ValueError(f"ingress {field} must use stop-first order")
        if config.get("Parallelism") != 1 or config.get("MaxFailureRatio") != 0:
            raise ValueError(f"ingress {field} must replace one task at a time")


def _task_runtime(task_id: str, network_id: str) -> tuple[str, str]:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            task_id,
            "--format",
            "{{json .Spec.ContainerSpec.Image}}\t{{json .NetworksAttachments}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    image, attachments = result.stdout.rstrip("\n").split("\t", 1)
    addresses = [
        address.split("/", 1)[0]
        for attachment in json.loads(attachments) or []
        if attachment.get("Network", {}).get("ID") == network_id
        for address in attachment.get("Addresses", [])
    ]
    if len(addresses) != 1:
        raise ValueError("ingress task does not have one dokploy-network address")
    return json.loads(image).split("@", 1)[0], addresses[0]


def _dokploy_network_id() -> str:
    result = subprocess.run(
        ["docker", "network", "inspect", "dokploy-network", "--format", "{{.ID}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    network_id = result.stdout.strip()
    if not network_id:
        raise ValueError("dokploy-network could not be resolved")
    return network_id


def verify_ingress(
    *,
    base_url: str,
    api_key: str,
    application_id: str,
    app_name: str,
    image: str,
    revision: str,
    stats_url: str,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    deadline = monotonic() + 600
    candidate: tuple[
        frozenset[str], tuple[tuple[str, tuple[str, ...]], ...]
    ] | None = None
    stable_rounds = 0
    while monotonic() < deadline:
        deployments = _dokploy_request(
            base_url,
            api_key,
            "deployment.all",
            applicationId=application_id,
        )
        status = deployments[0]["status"] if deployments else "none"
        if status == "error":
            raise RuntimeError("Dokploy reported a failed ingress deployment")
        if status != "done":
            sleep(5)
            continue
        application = _dokploy_request(
            base_url,
            api_key,
            "application.one",
            applicationId=application_id,
        )
        _verify_ingress_configuration(application, image, revision)
        tasks = _dokploy_request(
            base_url,
            api_key,
            "docker.getServiceContainersByAppName",
            appName=app_name,
        )
        actual_running = frozenset(
            task["containerId"]
            for task in tasks
            if str(task.get("currentState", "")).startswith("Running ")
            and task.get("node")
            and not str(task.get("error", "")).removeprefix("Error:").strip()
        )
        desired_running = frozenset(
            task["containerId"] for task in tasks if task.get("state") == "running"
        )
        if (
            len(actual_running) != INGRESS_REPLICAS
            or actual_running != desired_running
        ):
            candidate = None
            stable_rounds = 0
            sleep(5)
            continue
        expected_image = f"{image}:{revision}"
        network_id = _dokploy_network_id()
        task_runtime = [_task_runtime(task, network_id) for task in actual_running]
        direct_admission = tuple(
            sorted(
                (
                    address,
                    tuple(ingress_backends(f"http://{address}:8404/stats;csv")),
                )
                for task_image, address in task_runtime
                if task_image == expected_image
            )
        )
        vip_admission = ingress_backends(stats_url)
        snapshot = (actual_running, direct_admission)
        complete = (
            len(direct_admission) == INGRESS_REPLICAS
            and all(
                len(backends) >= MIN_CRAWLER_BACKENDS
                for _, backends in direct_admission
            )
            and len(vip_admission) >= MIN_CRAWLER_BACKENDS
        )
        if complete:
            stable_rounds = stable_rounds + 1 if snapshot == candidate else 1
            candidate = snapshot
        else:
            candidate = None
            stable_rounds = 0
        if stable_rounds >= 2:
            print(
                json.dumps(
                    {
                        "revision": revision,
                        "ingressTasks": sorted(actual_running),
                        "directAdmission": direct_admission,
                        "vipAdmission": vip_admission,
                    },
                    separators=(",", ":"),
                )
            )
            return
        sleep(5)
    raise TimeoutError("readiness ingress did not stabilize within 10 minutes")


def main(arguments: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments == ["monitor"]:
        monitor_public_health(
            health_url=os.environ.get(
                "HEALTH_URL", "https://crawl4ai.haiku.host/health"
            ),
            stats_url=os.environ["INGRESS_STATS_URL"],
            expected_replicas=int(os.environ.get("EXPECTED_REPLICAS", "3")),
            evidence_path=Path(os.environ["ROLLOUT_MONITOR_PATH"]),
            armed_path=Path(os.environ["ROLLOUT_MONITOR_ARMED_PATH"]),
            stop_path=Path(os.environ["ROLLOUT_MONITOR_STOP_PATH"]),
        )
        return
    if arguments == ["ingress"]:
        verify_ingress(
            base_url=os.environ["DOKPLOY_URL"],
            api_key=os.environ["DOKPLOY_API_KEY"],
            application_id=os.environ["INGRESS_APPLICATION_ID"],
            app_name=os.environ["INGRESS_APPLICATION_APP_NAME"],
            image=os.environ["INGRESS_IMAGE"],
            revision=os.environ["INGRESS_TAG"],
            stats_url=os.environ["INGRESS_STATS_URL"],
        )
        return
    routing = {
        "base_url": os.environ["DOKPLOY_URL"],
        "api_key": os.environ["DOKPLOY_API_KEY"],
        "application_id": os.environ["APPLICATION_ID"],
        "app_name": os.environ["APPLICATION_APP_NAME"],
        "ingress_app_name": os.environ["INGRESS_APPLICATION_APP_NAME"],
    }
    if not arguments:
        configure_readiness_routing(**routing)
    elif arguments == ["verify"]:
        verify_readiness_routing(**routing)
    else:
        raise ValueError(f"Unsupported readiness routing arguments: {arguments}")


if __name__ == "__main__":
    main()
