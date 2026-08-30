import copy
import json
import subprocess
from pathlib import Path

import pytest

import verify_rollout as rollout

BASELINE = "registry.example/crawl4ai@sha256:baseline"
CANDIDATE = "registry.example/crawl4ai@sha256:candidate"
REVISION = "a" * 40


def application(image=BASELINE, revision="baseline"):
    return {
        "applicationId": "app",
        "appName": "crawl4ai",
        "sourceType": "docker",
        "dockerImage": image,
        "labelsSwarm": rollout._labels(revision),
        "env": (
            f"LLM_PROVIDER={rollout.LLM_PROVIDER}\n"
            f"LLM_BASE_URL={rollout.LLM_BASE_URL}\n"
            "LLM_API_KEY=secret"
        ),
        "replicas": 3,
        "healthCheckSwarm": copy.deepcopy(rollout.HEALTHCHECK),
        "placementSwarm": {
            "Constraints": [rollout.NODE_CONSTRAINT],
            "MaxReplicas": 1,
        },
        "endpointSpecSwarm": {"Mode": "vip", "Ports": []},
        "updateConfigSwarm": {
            "Parallelism": 1,
            "Delay": rollout.DELAY_NS,
            "FailureAction": "rollback",
            "Monitor": rollout.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "rollbackConfigSwarm": {
            "Parallelism": 1,
            "Delay": rollout.DELAY_NS,
            "FailureAction": "pause",
            "Monitor": rollout.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "stopGracePeriodSwarm": rollout.STOP_GRACE_NS,
        "cpuReservation": "500000000",
        "cpuLimit": "2000000000",
        "memoryReservation": "1073741824",
        "memoryLimit": "4294967296",
    }


def service_spec(image=BASELINE, revision="baseline"):
    return {
        "TaskTemplate": {
            "ContainerSpec": {
                "Image": image,
                "Labels": rollout._labels(revision),
                "Env": [
                    f"LLM_PROVIDER={rollout.LLM_PROVIDER}",
                    f"LLM_BASE_URL={rollout.LLM_BASE_URL}",
                    "LLM_API_KEY=secret",
                ],
                "Healthcheck": copy.deepcopy(rollout.HEALTHCHECK),
                "StopGracePeriod": rollout.STOP_GRACE_NS,
            },
            "Placement": {
                "Constraints": [rollout.NODE_CONSTRAINT],
                "MaxReplicas": 1,
            },
            "Resources": copy.deepcopy(rollout.RESOURCES),
        },
        "Mode": {"Replicated": {"Replicas": 3}},
        "UpdateConfig": {
            "Parallelism": 1,
            "Delay": rollout.DELAY_NS,
            "FailureAction": "rollback",
            "Monitor": rollout.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "RollbackConfig": {
            "Parallelism": 1,
            "Delay": rollout.DELAY_NS,
            "FailureAction": "pause",
            "Monitor": rollout.DELAY_NS,
            "MaxFailureRatio": 0,
            "Order": "start-first",
        },
        "EndpointSpec": {"Mode": "vip"},
    }


def health(instance="container123", revision=REVISION):
    return {
        "instance": instance,
        "revision": revision,
        "status": "ok",
        "components": {"api": "ready", "redis": "ready"},
    }


def route(
    custom_definition=False,
    custom_reference=False,
    wrong_url=False,
    wrong_router=False,
):
    transport = {"serversTransport": "custom"} if custom_reference else {}
    config = {
        "http": {
            "routers": {
                "one": {
                    "rule": "Host(`crawl4ai.haiku.host`)",
                    "service": "missing@internal" if wrong_router else "one",
                },
                "two": {
                    "rule": "Host(`crawl4ai.popos-sf0.com`)",
                    "service": "two",
                },
            },
            "services": {
                key: {
                    "loadBalancer": {
                        "servers": [
                            {
                                "url": (
                                    "http://wrong:11235"
                                    if wrong_url
                                    else "http://crawl4ai:11235"
                                )
                            }
                        ],
                        **transport,
                    }
                }
                for key in ("one", "two")
            },
        }
    }
    if custom_definition:
        config["http"]["serversTransports"] = {"custom": {}}
    return yaml_dump(config)


def yaml_dump(value):
    import yaml

    return yaml.safe_dump(value)


def test_policy_accepts_only_stock_docker_image_configuration():
    rollout._policy(application())
    for field, value in (
        ("sourceType", "github"),
        ("replicas", 2),
        ("healthCheckSwarm", {}),
        ("placementSwarm", {}),
        ("endpointSpecSwarm", {}),
        ("updateConfigSwarm", {}),
        ("rollbackConfigSwarm", {}),
        ("stopGracePeriodSwarm", 1),
        ("stopGracePeriodSwarm", rollout.STOP_GRACE_NS + 1),
        ("cpuReservation", "1"),
        ("cpuLimit", "1"),
        ("memoryReservation", "1"),
        ("memoryLimit", "1"),
    ):
        changed = application()
        changed[field] = value
        with pytest.raises(ValueError):
            rollout._policy(changed)

    changed = application()
    changed["env"] = changed["env"].replace(
        rollout.LLM_BASE_URL, "https://attacker.invalid/v1"
    )
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        rollout._policy(changed)

    changed = application()
    changed["env"] += f"\nLLM_BASE_URL={rollout.LLM_BASE_URL}"
    with pytest.raises(ValueError, match="duplicate LLM_BASE_URL"):
        rollout._policy(changed)


def test_running_spec_rejects_artifact_drift():
    rollout._running_spec(service_spec(), BASELINE, rollout._labels("baseline"))
    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["Image"] = CANDIDATE
    with pytest.raises(ValueError, match="artifact"):
        rollout._running_spec(changed, BASELINE, rollout._labels("baseline"))

    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["Env"][1] = (
        "LLM_BASE_URL=https://attacker.invalid/v1"
    )
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        rollout._running_spec(changed, BASELINE, rollout._labels("baseline"))

    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["Env"].append(
        f"LLM_BASE_URL={rollout.LLM_BASE_URL}"
    )
    with pytest.raises(ValueError, match="duplicate LLM_BASE_URL"):
        rollout._running_spec(changed, BASELINE, rollout._labels("baseline"))

    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["StopGracePeriod"] = (
        rollout.STOP_GRACE_NS + 1
    )
    with pytest.raises(ValueError, match="stop grace"):
        rollout._running_spec(changed, BASELINE, rollout._labels("baseline"))


@pytest.mark.parametrize(
    ("custom_definition", "custom_reference", "wrong_url", "wrong_router"),
    [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_route_rejects_non_stock_routing(
    monkeypatch, custom_definition, custom_reference, wrong_url, wrong_router
):
    monkeypatch.setattr(
        rollout,
        "_request_json",
        lambda *_args: route(
            custom_definition, custom_reference, wrong_url, wrong_router
        ),
    )
    with pytest.raises(ValueError):
        rollout.verify_route("https://dokploy", "key", "app", "crawl4ai")


def test_route_accepts_stock_dokploy_file(monkeypatch):
    monkeypatch.setattr(rollout, "_request_json", lambda *_args: route())
    rollout.verify_route("https://dokploy", "key", "app", "crawl4ai")


def test_post_keeps_api_key_out_of_argv(monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(rollout.subprocess, "run", run)
    rollout._post_json("https://dokploy/api/application.update", "secret", {"x": 1})
    assert "secret" not in " ".join(observed["command"])
    assert 'header = "x-api-key: secret"' in observed["input"]


def test_wait_deployment_accepts_one_exact_new_row(monkeypatch):
    monkeypatch.setattr(
        rollout,
        "_deployments",
        lambda *_args: [
            {
                "deploymentId": "new",
                "title": "title",
                "description": "description",
                "status": "done",
            },
            {"deploymentId": "old", "status": "done"},
        ],
    )
    result = rollout._wait_deployment(
        "https://dokploy", "key", "app", {"old"}, "title", "description"
    )
    assert result["deploymentId"] == "new"


def test_wait_deployment_fails_closed_on_foreign_row(monkeypatch):
    monkeypatch.setattr(
        rollout,
        "_deployments",
        lambda *_args: [{"deploymentId": "foreign", "title": "other"}],
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        rollout._wait_deployment(
            "https://dokploy", "key", "app", set(), "title", "description"
        )


def test_task_runtime_parses_container_identity_and_overlay(monkeypatch):
    output = (
        '"container123456789"\t'
        + json.dumps(rollout._labels(REVISION))
        + '\t"registry.example/crawl4ai@sha256:candidate"\t'
        + '[{"Network":{"ID":"network"},"Addresses":["10.0.1.23/24"]}]\n'
    )
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, output, ""),
    )
    runtime = rollout._task_runtime("task", "network")
    assert runtime == {
        "container": "container123",
        "labels": rollout._labels(REVISION),
        "image": CANDIDATE,
        "addresses": {"10.0.1.23"},
    }


def test_task_runtime_guards_the_container_id_swarm_omits(monkeypatch):
    # Swarm omits Status.ContainerStatus until a task reaches its container, and
    # docker inspect exits 1 on an unguarded read of that missing key. That aborted
    # three consecutive production rollouts (runs 33248150206, 33250958425,
    # 33265831563) in which every health probe had succeeded.
    output = (
        '""\t'
        + json.dumps(rollout._labels(REVISION))
        + '\t"registry.example/crawl4ai@sha256:candidate"\t'
        + "null\n"
    )

    def run(command, **_kwargs):
        assert "{{if .Status.ContainerStatus}}" in command[command.index("--format") + 1]
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(rollout.subprocess, "run", run)

    runtime = rollout._task_runtime("task", "network")

    assert runtime["container"] == ""
    assert runtime["addresses"] == set()


def test_verify_tasks_proves_each_overlay_backend(monkeypatch):
    rows = [
        {
            "ID": f"task{index}",
            "Name": f"crawl4ai.{index}",
            "Node": node,
            "DesiredState": "Running",
            "CurrentState": "Running 1m",
        }
        for index, node in enumerate(("haiku-4", "haiku-5", "haiku-9"), 1)
    ]
    rows += [
        {
            "ID": f"old{index}",
            "Name": f"crawl4ai.{index}",
            "Node": node,
            "DesiredState": "Shutdown",
            "CurrentState": "Shutdown 1m",
        }
        for index, node in enumerate(("haiku-5", "haiku-9", "haiku-18"), 1)
    ]
    runtimes = {
        f"task{index}": {
            "container": f"container{index}",
            "labels": rollout._labels(REVISION),
            "image": CANDIDATE,
            "addresses": {f"10.0.1.{index}"},
        }
        for index in range(1, 4)
    }
    def run(command, **_kwargs):
        if command[1:3] == ["service", "ps"]:
            return subprocess.CompletedProcess(
                command, 0, "\n".join(json.dumps(row) for row in rows), ""
            )
        return subprocess.CompletedProcess(command, 0, "network\n", "")

    monkeypatch.setattr(rollout.subprocess, "run", run)
    monkeypatch.setattr(
        rollout,
        "_task_state",
        lambda task: (
            ("running", "running")
            if task.startswith("task")
            else ("shutdown", "shutdown")
        ),
    )
    monkeypatch.setattr(
        rollout, "_task_runtime", lambda task, _network: runtimes[task]
    )
    monkeypatch.setattr(
        rollout,
        "_request_json",
        lambda url, *_args: health(
            next(runtime["container"] for runtime in runtimes.values() if next(iter(runtime["addresses"])) in url)
        ),
    )
    proof = rollout._verify_tasks("crawl4ai", CANDIDATE, REVISION)
    assert proof["nodes"] == ["haiku-4", "haiku-5", "haiku-9"]
    assert len(proof["instances"]) == 3


def test_deploy_uses_only_stock_update_and_deploy(monkeypatch):
    state = application()
    posts = []
    deployments = [{"deploymentId": "old", "status": "done"}]

    def post(url, _key, payload):
        posts.append((url, payload))
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]

    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setenv("IMAGE", "registry.example/crawl4ai")
    monkeypatch.setenv("IMAGE_DIGEST", "sha256:candidate")
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setattr(rollout, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout,
        "_deployments",
        lambda *_args: copy.deepcopy(deployments),
    )
    monkeypatch.setattr(rollout, "_post_json", post)
    monkeypatch.setattr(rollout, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout, "_eligible_nodes", lambda: rollout.ELIGIBLE_NODES)
    monkeypatch.setattr(rollout, "_service_spec", lambda _name: service_spec(state["dockerImage"], state["labelsSwarm"]["otel.service.version"]))
    monkeypatch.setattr(rollout, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout,
        "_wait_deployment",
        lambda *_args: {"deploymentId": "new", "status": "done"},
    )
    monkeypatch.setattr(
        rollout,
        "_verify_tasks",
        lambda *_args: {"tasks": ["1", "2", "3"], "nodes": ["haiku-4", "haiku-5", "haiku-9"], "instances": ["a", "b", "c"]},
    )
    monkeypatch.setattr(
        rollout,
        "_request_json",
        lambda url, *_args: health(revision="baseline" if "baseline=" in url else REVISION),
    )
    rollout.deploy()
    assert [url.rsplit("/", 1)[-1] for url, _ in posts] == [
        "application.update",
        "application.deploy",
    ]
    assert "idempotencyKey" not in posts[1][1]
    assert "expectedDockerImage" not in posts[0][1]

    state = application()
    posts.clear()
    deployments[:] = [{"deploymentId": "running", "status": "running"}]
    with pytest.raises(RuntimeError, match="nonterminal Dokploy deployment"):
        rollout.deploy()
    assert not posts

    state = application()
    posts.clear()
    deployments[:] = [{"deploymentId": "old", "status": "done"}]
    update_states = iter(["completed", "rollback_paused", "completed", "completed"])
    monkeypatch.setattr(rollout, "_update_state", lambda _name: next(update_states))
    with pytest.raises(RuntimeError, match="rollback_paused"):
        rollout.deploy()


@pytest.mark.parametrize("fail_on", [1, 2])
def test_deploy_does_not_compensate_for_ambiguous_write(monkeypatch, fail_on):
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setenv("IMAGE", "registry.example/crawl4ai")
    monkeypatch.setenv("IMAGE_DIGEST", "sha256:candidate")
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    state = application()
    monkeypatch.setattr(rollout, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(rollout, "_deployments", lambda *_args: [])
    monkeypatch.setattr(rollout, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout, "_eligible_nodes", lambda: rollout.ELIGIBLE_NODES)
    monkeypatch.setattr(rollout, "_service_spec", lambda _name: service_spec())
    monkeypatch.setattr(rollout, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout, "_verify_tasks", lambda *_args: {})
    monkeypatch.setattr(
        rollout,
        "_request_json",
        lambda _url, *_args: health(revision="baseline"),
    )
    calls = []

    def fail(url, _key, payload):
        calls.append(url)
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]
        if len(calls) == fail_on:
            raise RuntimeError("HTTP request failed; state is ambiguous")

    monkeypatch.setattr(rollout, "_post_json", fail)
    with pytest.raises(RuntimeError, match="ambiguous"):
        rollout.deploy()
    assert len(calls) == fail_on


def test_evidence_requires_candidate_on_both_domains(monkeypatch, tmp_path):
    path = tmp_path / "evidence.jsonl"
    rows = [
        {"ok": True, "url": url, "revision": REVISION, "instance": "one"}
        for url in rollout.HEALTH_URLS
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    rollout.evidence()
    rows[0]["revision"] = "baseline"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="each public domain"):
        rollout.evidence()


def test_monitor_arms_only_after_both_domains_are_ready(monkeypatch, tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    armed_path = tmp_path / "armed"
    stop_path = tmp_path / "stop"
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(evidence_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_ARMED_PATH", str(armed_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_STOP_PATH", str(stop_path))
    calls = []

    def ready(url, *_args):
        assert not armed_path.exists()
        calls.append(url)
        if len(calls) == len(rollout.HEALTH_URLS):
            stop_path.touch()
        return health()

    monkeypatch.setattr(rollout, "_request_json", ready)
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    rollout.monitor()
    assert armed_path.exists()
    assert len(evidence_path.read_text().splitlines()) == len(rollout.HEALTH_URLS)
