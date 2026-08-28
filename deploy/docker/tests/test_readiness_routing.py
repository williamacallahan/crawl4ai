from __future__ import annotations

import os
import re
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import readiness_routing


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = Path(__file__).resolve().parents[1]
INGRESS_DIR = REPO_ROOT / "deploy" / "ingress"

TRAEFIK_CONFIG = """
http:
  routers:
    crawl4ai-yq1svi-router-1:
      rule: Host(`crawl.example`)
      service: crawl4ai-yq1svi-service-1
  services:
    crawl4ai-yq1svi-service-1:
      loadBalancer:
        servers:
          - url: http://crawl4ai-yq1svi:11235
"""


def test_traefik_routes_through_the_readiness_ingress():
    updated = readiness_routing.patch_traefik_config(
        TRAEFIK_CONFIG,
        "crawl4ai-yq1svi",
        "crawl4ai-ingress",
    )

    readiness_routing.verify_traefik_config(
        updated, "crawl4ai-yq1svi", "crawl4ai-ingress"
    )
    assert "url: http://crawl4ai-ingress:11235" in updated
    assert "url: http://crawl4ai-yq1svi:11235" not in updated


def test_traefik_verifier_rejects_the_raw_swarm_vip():
    with pytest.raises(ValueError, match="ingress"):
        readiness_routing.verify_traefik_config(
            TRAEFIK_CONFIG, "crawl4ai-yq1svi", "crawl4ai-ingress"
        )


def test_configure_persists_and_reads_back_the_ingress_route(monkeypatch):
    written_config = None

    def request(_base_url, _api_key, operation, *, payload=None, **_params):
        nonlocal written_config
        if operation == "application.updateTraefikConfig":
            written_config = payload["traefikConfig"]
            return True
        return written_config or TRAEFIK_CONFIG

    monkeypatch.setattr(readiness_routing, "_dokploy_request", request)
    readiness_routing.configure_readiness_routing(
        base_url="https://dokploy.example",
        api_key="secret",
        application_id="app",
        app_name="crawl4ai-yq1svi",
        ingress_app_name="crawl4ai-ingress",
    )

    assert written_config is not None
    readiness_routing.verify_traefik_config(
        written_config, "crawl4ai-yq1svi", "crawl4ai-ingress"
    )


def test_monitor_records_ingress_and_public_failure_after_arming(monkeypatch, tmp_path):
    samples = iter(
        [
            {"ok": True, "status": 200, "instance": "old", "revision": "old"},
            {"ok": False, "error": "TimeoutError"},
        ]
    )
    monkeypatch.setattr(readiness_routing, "_public_health_sample", lambda _url: next(samples))
    monkeypatch.setattr(
        readiness_routing, "ingress_backends", lambda _url: ["10.0.1.11"]
    )
    monkeypatch.setattr(readiness_routing.time, "sleep", lambda _seconds: None)
    evidence = tmp_path / "monitor.jsonl"
    armed = tmp_path / "armed"

    with pytest.raises(RuntimeError, match="failed request"):
        readiness_routing.monitor_public_health(
            health_url="https://crawl.example/health",
            stats_url="http://crawl4ai-ingress:8404/stats;csv",
            expected_replicas=1,
            evidence_path=evidence,
            armed_path=armed,
            stop_path=tmp_path / "stop",
        )

    assert armed.exists()
    assert [json.loads(line)["ok"] for line in evidence.read_text().splitlines()] == [
        True,
        False,
    ]
    with pytest.raises(RuntimeError, match="did not remain successful"):
        readiness_routing.verify_monitor_evidence(
            evidence, frozenset({"replacement"}), frozenset({"10.0.1.11"})
        )


def test_monitor_evidence_proves_predecessor_withdrawal_and_final_admission(tmp_path):
    evidence = tmp_path / "monitor.jsonl"
    samples = [
        {"ok": True, "instance": "old-a", "backends": ["10.0.1.11"]},
        {"ok": True, "instance": "old-b", "backends": ["10.0.1.11"]},
        {"ok": True, "instance": "new-a", "backends": ["10.0.1.21"]},
        {
            "ok": True,
            "instance": "new-b",
            "backends": ["10.0.1.21", "10.0.1.22"],
        },
    ]
    evidence.write_text("".join(json.dumps(sample) + "\n" for sample in samples))

    readiness_routing.verify_monitor_evidence(
        evidence,
        frozenset({"new-a", "new-b"}),
        frozenset({"10.0.1.21", "10.0.1.22"}),
    )


def test_haproxy_stats_returns_only_up_crawler_addresses(monkeypatch):
    body = (
        b"# pxname,svname,status,addr\n"
        b"crawl4ai,crawler1,UP,10.0.1.11:11235\n"
        b"crawl4ai,crawler2,DOWN,10.0.1.12:11235\n"
        b"stats,FRONTEND,OPEN,\n"
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return body

    monkeypatch.setattr(
        readiness_routing.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )

    assert readiness_routing.ingress_backends("http://ingress:8404/stats;csv") == [
        "10.0.1.11"
    ]


def test_ingress_verifier_requires_two_stable_tasks_and_exact_image(monkeypatch):
    application = {
        "dockerImage": "registry.example/crawl4ai-ingress:target",
        "replicas": 1,
        "healthCheckSwarm": {"Test": readiness_routing.INGRESS_HEALTHCHECK},
        "placementSwarm": {
            "Constraints": [readiness_routing.INGRESS_NODE_CONSTRAINT],
            "MaxReplicas": 1,
        },
        "updateConfigSwarm": {
            "Order": "stop-first",
            "Parallelism": 1,
            "MaxFailureRatio": 0,
        },
        "rollbackConfigSwarm": {
            "Order": "stop-first",
            "Parallelism": 1,
            "MaxFailureRatio": 0,
        },
    }
    tasks = [
        {
            "containerId": "task-a",
            "state": "running",
            "currentState": "Running 10s",
            "node": "haiku-0",
            "error": "",
        },
    ]

    def request(_base_url, _api_key, operation, **_params):
        return {
            "deployment.all": [{"status": "done"}],
            "application.one": application,
            "docker.getServiceContainersByAppName": tasks,
        }[operation]

    monkeypatch.setattr(readiness_routing, "_dokploy_request", request)
    stats_urls = []

    def backends(url):
        stats_urls.append(url)
        return ["10.0.1.11", "10.0.1.12", "10.0.1.13"]

    monkeypatch.setattr(readiness_routing, "ingress_backends", backends)
    monkeypatch.setattr(
        readiness_routing,
        "_dokploy_network_id",
        lambda: "network",
    )
    monkeypatch.setattr(
        readiness_routing,
        "_task_runtime",
        lambda task, _network: (
            "registry.example/crawl4ai-ingress:target",
            "10.0.2.11",
        ),
    )

    readiness_routing.verify_ingress(
        base_url="https://dokploy.example",
        api_key="secret",
        application_id="ingress",
        app_name="crawl4ai-ingress",
        image="registry.example/crawl4ai-ingress",
        revision="target",
        stats_url="http://ingress:8404/stats;csv",
        sleep=lambda _seconds: None,
    )
    assert "http://10.0.2.11:8404/stats;csv" in stats_urls
    assert "http://ingress:8404/stats;csv" in stats_urls


def test_haproxy_discovers_and_admits_only_healthy_swarm_tasks():
    config = (INGRESS_DIR / "haproxy.cfg").read_text()

    assert "bind :11235" in config
    assert "frontend ingress" in config
    assert "nameserver docker 127.0.0.11:53" in config
    assert "tasks.crawl4ai-yq1svi:11235" in config
    assert "backend crawl4ai" in config
    assert "server-template crawler 8" in config
    assert "http-check send meth GET uri /health" in config
    assert "http-check expect status 200" in config
    assert re.search(
        r"default-server check inter 500ms fastinter 250ms downinter 250ms "
        r"fall 1 rise 1 .* init-addr none init-state fully-down",
        config,
    )


def test_haproxy_exposes_internal_backend_admission_stats():
    config = (INGRESS_DIR / "haproxy.cfg").read_text()

    assert "listen stats" in config
    assert "bind :8404" in config
    assert "stats enable" in config
    assert "stats uri /stats" in config


def test_haproxy_preserves_long_requests_websockets_and_streams():
    config = (INGRESS_DIR / "haproxy.cfg").read_text()

    assert "mode http" in config
    assert "timeout client 5m" in config
    assert "timeout server 5m" in config
    assert "timeout tunnel 5m" in config


def test_deploy_workflow_bootstraps_ingress_then_monitors_the_crawler_rollout():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-deploy.yml").read_text()
    ingress_update = workflow.index('"sourceType":"docker"')
    ingress_deploy = workflow.index(
        '"applicationId":"%s"}', ingress_update
    )
    ingress_proof = workflow.index("readiness_routing.py ingress", ingress_deploy)
    monitor = workflow.index("test -n \"$MONITOR_PID\" || start_monitor", ingress_proof)
    route_switch = workflow.index("python3 deploy/docker/readiness_routing.py", monitor)
    crawler_update = workflow.index('"dockerImage":"%s:%s"', monitor)
    crawler_deploy = workflow.index('"applicationId":"%s"}', crawler_update)
    final_proof = workflow.index("python3 deploy/docker/verify_rollout.py\n", crawler_deploy)

    assert ingress_update < ingress_deploy < ingress_proof < monitor < route_switch
    assert route_switch < crawler_update < crawler_deploy < final_proof
    assert "deploy/docker/tests/test_readiness_routing.py" in workflow
    assert '"placementSwarm":{"MaxReplicas":1}' in workflow
    assert "http://localhost:11235/health/route" not in workflow


def test_pid1_withdrawal_delay_outlives_haproxy_health_withdrawal():
    config = (INGRESS_DIR / "haproxy.cfg").read_text()
    entrypoint = (DOCKER_DIR / "entrypoint.sh").read_text()

    interval_ms = int(re.search(r"\binter (\d+)ms", config).group(1))
    timeout_ms = int(re.search(r"timeout check (\d+)ms", config).group(1))
    delay_seconds = int(
        re.search(r"CRAWL4AI_DRAIN_DELAY_SECONDS:-([0-9]+)", entrypoint).group(1)
    )

    assert delay_seconds * 1000 > interval_ms + timeout_ms


@pytest.mark.asyncio
async def test_public_health_admits_only_ready_nondraining_tasks(
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

            server_module.app.state.readiness_checks_active = True
            ready = await client.get("/health")
            assert ready.status_code == 200
            assert ready.json()["components"] == {
                "api": "ready",
                "redis": "ready",
            }

            drain_path.touch()
            assert (await client.get("/health")).status_code == 503
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
