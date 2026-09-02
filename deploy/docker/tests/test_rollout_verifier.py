import copy
import http.client
import json
import subprocess
from dataclasses import fields
from threading import Thread

import pytest

import swarm_observer as observer
import swarm_observer_server as observer_server
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
        "placementSwarm": copy.deepcopy(rollout.PLACEMENT),
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
            "Placement": copy.deepcopy(rollout.PLACEMENT),
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


def redis_spec():
    return {
        "TaskTemplate": {
            "ContainerSpec": {
                "Image": "redis@sha256:redis",
                "Command": list(rollout.REDIS_COMMAND),
                "Mounts": copy.deepcopy(rollout.REDIS_MOUNTS),
                "Healthcheck": {
                    "Test": list(rollout.REDIS_HEALTHCHECK_TEST),
                    **copy.deepcopy(rollout.REDIS_HEALTHCHECK_FLOORS),
                },
            },
            "Placement": {
                "Constraints": [f"node.hostname=={rollout.REDIS_NODE}"],
                "MaxReplicas": 1,
            },
        },
        "Mode": {"Replicated": {"Replicas": 1}},
    }


def redis_tasks(state="Running 19 hours ago", node=None):
    return [
        {
            "ID": "redis1",
            "Name": "crawl4ai-redis.1",
            "Node": node or rollout.REDIS_NODE,
            "DesiredState": "Running",
            "CurrentState": state,
        },
        {
            "ID": "redis0",
            "Name": "crawl4ai-redis.1",
            "Node": rollout.REDIS_NODE,
            "DesiredState": "Shutdown",
            "CurrentState": "Shutdown 30 hours ago",
        },
    ]


def test_redis_pins_encode_the_2026_08_28_incident_invariants():
    """The drift check is only as good as what it pins, so pin the two values
    the incident was about rather than trusting the comparison alone."""
    assert "everysec" in rollout.REDIS_COMMAND and "always" not in rollout.REDIS_COMMAND
    assert rollout.REDIS_COMMAND[rollout.REDIS_COMMAND.index("--appendfsync") + 1] == "everysec"
    assert rollout.REDIS_COMMAND[rollout.REDIS_COMMAND.index("--appendonly") + 1] == "yes"
    assert rollout.REDIS_MOUNTS == [
        {"Type": "volume", "Source": "crawl4ai-redis-data", "Target": "/data"}
    ]
    assert rollout.REDIS_HEALTHCHECK_FLOORS["Retries"] >= 3
    assert rollout.REDIS_HEALTHCHECK_FLOORS["Interval"] >= 5_000_000_000


def test_redis_proof_rejects_durability_drift(monkeypatch):
    def check(spec=None, tasks=None):
        monkeypatch.setattr(rollout, "_service_spec", lambda _n: copy.deepcopy(spec or redis_spec()))
        monkeypatch.setattr(rollout, "_service_tasks", lambda _n: copy.deepcopy(tasks or redis_tasks()))
        rollout._verify_redis()

    check()

    # appendfsync=always was half of the 2026-08-28 fleet kill.
    drifted = redis_spec()
    command = drifted["TaskTemplate"]["ContainerSpec"]["Command"]
    command[command.index("everysec")] = "always"
    with pytest.raises(ValueError, match="command"):
        check(spec=drifted)

    drifted = redis_spec()
    drifted["TaskTemplate"]["ContainerSpec"]["Mounts"] = []
    with pytest.raises(ValueError, match="volume"):
        check(spec=drifted)

    drifted = redis_spec()
    drifted["TaskTemplate"]["ContainerSpec"]["Healthcheck"] = {}
    with pytest.raises(ValueError, match="healthcheck probe"):
        check(spec=drifted)

    drifted = redis_spec()
    drifted["TaskTemplate"]["ContainerSpec"]["Healthcheck"]["Retries"] = 1
    with pytest.raises(ValueError, match="Retries fell below"):
        check(spec=drifted)

    # Raising a floor is a safety improvement, not drift: the 2026-08-28 kill
    # came from a healthcheck that was too tight.
    relaxed = redis_spec()
    relaxed["TaskTemplate"]["ContainerSpec"]["Healthcheck"]["Retries"] = 5
    relaxed["TaskTemplate"]["ContainerSpec"]["Healthcheck"]["StartPeriod"] = 60_000_000_000
    check(spec=relaxed)

    drifted = redis_spec()
    drifted["TaskTemplate"]["Placement"]["Constraints"] = ["node.hostname==haiku-4"]
    with pytest.raises(ValueError, match="placement"):
        check(spec=drifted)

    drifted = redis_spec()
    drifted["Mode"] = {"Replicated": {"Replicas": 2}}
    with pytest.raises(ValueError, match="single-replica"):
        check(spec=drifted)

    with pytest.raises(RuntimeError, match="did not converge"):
        check(tasks=redis_tasks(state="Preparing 2 seconds ago"))

    with pytest.raises(RuntimeError, match="did not converge"):
        check(tasks=redis_tasks(node="haiku-4"))

    second = dict(redis_tasks()[0], ID="redis2", Name="crawl4ai-redis.2")
    with pytest.raises(RuntimeError, match="did not converge"):
        check(tasks=redis_tasks() + [second])


def test_stop_grace_covers_the_shipped_container_drain_budget(monkeypatch):
    """STOP_GRACE_NS was a bare literal; supervisord's stop wait and the
    entrypoint's drain delay could grow past it with nothing to notice."""
    shipped = application()
    assert rollout._drain_budget_ns(shipped["env"]) < rollout.STOP_GRACE_NS
    rollout._policy(shipped)

    # A drain delay raised in the Dokploy console must be what is measured,
    # not the fallback literal in entrypoint.sh.
    raised = application()
    raised["env"] += "\nCRAWL4AI_DRAIN_DELAY_SECONDS=600"
    assert rollout._drain_budget_ns(raised["env"]) > rollout.STOP_GRACE_NS
    with pytest.raises(ValueError, match="drain budget"):
        rollout._policy(raised)

    monkeypatch.setattr(rollout, "_drain_budget_ns", lambda _env: rollout.STOP_GRACE_NS)
    with pytest.raises(ValueError, match="drain budget"):
        rollout._policy(application())


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


def test_running_spec_requires_record_labels_beside_control_plane_labels():
    # Dokploy stamps dokploy.deployment.id on every rendered ContainerSpec since
    # its 2026-09-01 build; exact label equality then failed every rollout (runs
    # 33541789332 and 33544931205) although the record's own labels were intact.
    stamped = service_spec()
    stamped["TaskTemplate"]["ContainerSpec"]["Labels"]["dokploy.deployment.id"] = "row"
    rollout._running_spec(stamped, BASELINE, rollout._labels("baseline"))

    for missing_or_changed in ("otel.service.version", "otel.logs.enabled"):
        changed = service_spec()
        changed["TaskTemplate"]["ContainerSpec"]["Labels"]["dokploy.deployment.id"] = "row"
        changed["TaskTemplate"]["ContainerSpec"]["Labels"][missing_or_changed] = "other"
        with pytest.raises(ValueError, match="labels"):
            rollout._running_spec(changed, BASELINE, rollout._labels("baseline"))
        del changed["TaskTemplate"]["ContainerSpec"]["Labels"][missing_or_changed]
        with pytest.raises(ValueError, match="labels"):
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
    # Dokploy's own stamp beside the record's labels must not read as drift.
    runtimes["task1"]["labels"]["dokploy.deployment.id"] = "row"
    network = [{
        "Id": "network",
        "Services": {
            "crawl4ai": {
                "Tasks": [
                    {
                        "Name": f"{row['Name']}.{row['ID']}",
                        "EndpointIP": next(iter(runtimes[row["ID"]]["addresses"])),
                    }
                    for row in rows[:3]
                ]
            }
        },
    }]
    def run(command, **_kwargs):
        if command[1:3] == ["service", "ps"]:
            return subprocess.CompletedProcess(
                command, 0, "\n".join(json.dumps(row) for row in rows), ""
            )
        return subprocess.CompletedProcess(command, 0, json.dumps(network), "")

    monkeypatch.setattr(rollout.subprocess, "run", run)
    monkeypatch.setattr(rollout, "_verify_ingress_host", lambda: None)
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
    proof = rollout._verify_tasks("crawl4ai", CANDIDATE, REVISION, rollout.ELIGIBLE_NODES)
    assert proof["nodes"] == ["haiku-4", "haiku-5", "haiku-9"]
    assert len(proof["instances"]) == 3


@pytest.mark.parametrize("drift", ["missing", "extra"])
def test_verify_tasks_rejects_vip_membership_drift(monkeypatch, drift):
    rows, runtimes = _healed_baseline_rows()
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    original_run = rollout.subprocess.run

    def run(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[1:4] == ["network", "inspect", "--verbose"]:
            network = json.loads(result.stdout)
            tasks = network[0]["Services"]["crawl4ai"]["Tasks"]
            if drift == "missing":
                tasks.pop()
            else:
                tasks.append({"Name": "crawl4ai.4.foreign", "EndpointIP": "10.0.1.99"})
            return subprocess.CompletedProcess(command, 0, json.dumps(network), "")
        return result

    monkeypatch.setattr(rollout.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="VIP membership"):
        rollout._verify_tasks(
            "crawl4ai",
            BASELINE,
            "baseline",
            frozenset({"haiku-5", "haiku-9", "haiku-18"}),
            False,
        )


@pytest.mark.parametrize("drift", ["missing", "changed"])
def test_verify_tasks_rejects_record_label_drift_beside_control_plane_labels(
    monkeypatch, drift
):
    rows, runtimes = _healed_baseline_rows()
    labels = runtimes["task2"]["labels"]
    labels["dokploy.deployment.id"] = "row"
    if drift == "missing":
        del labels["otel.logs.enabled"]
    else:
        labels["otel.service.version"] = "other"
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    with pytest.raises(RuntimeError, match="identity drifted"):
        rollout._verify_tasks(
            "crawl4ai",
            BASELINE,
            "baseline",
            frozenset({"haiku-5", "haiku-9", "haiku-18"}),
            False,
        )


@pytest.mark.parametrize("containers", ["", "one\ntwo\n"])
def test_ingress_host_requires_one_running_dokploy_traefik(monkeypatch, containers):
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, containers, ""
        ),
    )
    with pytest.raises(RuntimeError, match="public ingress host"):
        rollout._verify_ingress_host()


def test_ingress_host_uses_fixed_read_only_docker_projections(monkeypatch):
    invocation = []

    def run(command, **kwargs):
        invocation.append((command, kwargs))
        stdout = "container-id\n" if command[1] == "ps" else "4321\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(rollout.subprocess, "run", run)

    assert rollout._verify_ingress_host() == 4321
    assert invocation == [
        (
            [
                "docker",
                "ps",
                "--no-trunc",
                "--filter",
                "name=^/dokploy-traefik$",
                "--filter",
                "status=running",
                "--format",
                "{{.ID}}",
            ],
            {"check": True, "capture_output": True, "text": True},
        ),
        (
            [
                "docker",
                "inspect",
                "container-id",
                "--format",
                "{{.State.Pid}}",
            ],
            {"check": True, "capture_output": True, "text": True},
        ),
    ]


def test_public_proof_observes_every_authoritative_instance_in_any_order(monkeypatch):
    instances = ["one", "two", "three"]
    responses = {
        rollout.HEALTH_URLS[0]: iter(["three", "one", "two"]),
        rollout.HEALTH_URLS[1]: iter(["two", "three", "one"]),
    }

    def request(url, *_args):
        route_url = next(route for route in rollout.HEALTH_URLS if url.startswith(route))
        return health(next(responses[route_url]))

    monkeypatch.setattr(rollout, "_request_json", request)
    assert rollout._verify_public(REVISION, instances) == {
        url: sorted(instances) for url in rollout.HEALTH_URLS
    }


def test_public_proof_rejects_a_missing_healthy_backend(monkeypatch):
    monkeypatch.setattr(rollout, "PUBLIC_PROOF_ATTEMPTS", 3)
    monkeypatch.setattr(rollout, "_request_json", lambda *_args: health("one"))
    with pytest.raises(RuntimeError, match="did not observe every authoritative task"):
        rollout._verify_public(REVISION, ["one", "two", "three"])


def test_public_proof_rejects_an_unknown_instance(monkeypatch):
    monkeypatch.setattr(rollout, "_request_json", lambda *_args: health("unknown"))
    with pytest.raises(RuntimeError, match="unknown task instance"):
        rollout._verify_public(REVISION, ["one", "two", "three"])


@pytest.mark.parametrize(
    "response",
    [health("one", revision="wrong"), {**health("one"), "status": "degraded"}],
)
def test_public_proof_rejects_wrong_revision_or_unhealthy_response(monkeypatch, response):
    monkeypatch.setattr(rollout, "_request_json", lambda *_args: response)
    with pytest.raises(RuntimeError, match="unhealthy or wrong-revision"):
        rollout._verify_public(REVISION, ["one", "two", "three"])


def test_deploy_uses_only_stock_update_and_deploy(monkeypatch, tmp_path, capsys):
    state = application()
    posts = []
    events = []
    deployments = [{"deploymentId": "old", "status": "done"}]

    def post(url, _key, payload):
        posts.append((url, payload))
        events.append(url.rsplit("/", 1)[-1])
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]
            state["placementSwarm"] = payload["placementSwarm"]

    _deploy_env(monkeypatch)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
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
    monkeypatch.setattr(rollout, "_verify_redis", lambda: events.append("verify_redis"))
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
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout.HEALTH_URLS},
    )
    monkeypatch.setattr(
        rollout,
        "_request_json",
        lambda url, *_args: health(revision="baseline" if "baseline=" in url else REVISION),
    )
    rollout.deploy()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["publicInstances"] == {
        url: ["a", "b", "c"] for url in rollout.HEALTH_URLS
    }
    # The Redis proof must precede the first write, not merely happen.
    assert events == ["verify_redis", "application.update", "application.deploy"]
    assert [url.rsplit("/", 1)[-1] for url, _ in posts] == [
        "application.update",
        "application.deploy",
    ]
    assert "idempotencyKey" not in posts[1][1]
    assert posts[0][1]["placementSwarm"] == rollout.PLACEMENT
    assert "expectedDockerImage" not in posts[0][1]
    proof = json.loads(rollout._task_proof_path().read_text())
    assert proof["instances"] == ["a", "b", "c"]
    assert set(proof["publicInstances"]) == set(rollout.HEALTH_URLS)

    state = application()
    posts.clear()
    events.clear()
    deployments[:] = [{"deploymentId": "running", "status": "running"}]
    with pytest.raises(RuntimeError, match="nonterminal Dokploy deployment"):
        rollout.deploy()
    assert not posts

    state = application()
    posts.clear()
    events.clear()
    deployments[:] = [{"deploymentId": "old", "status": "done"}]
    proofs = iter([
        {"tasks": ["old1", "old2", "old3"], "nodes": ["haiku-4", "haiku-5", "haiku-9"], "instances": ["old-a", "old-b", "old-c"]},
        {"tasks": ["1", "2", "3"], "nodes": ["haiku-4", "haiku-5", "haiku-9"], "instances": ["a", "b", "c"]},
        {"tasks": ["1", "2", "4"], "nodes": ["haiku-4", "haiku-5", "haiku-18"], "instances": ["a", "b", "d"]},
    ])
    monkeypatch.setattr(rollout, "_verify_tasks", lambda *_args: next(proofs))
    monkeypatch.setattr(rollout, "_update_state", lambda _name: "completed")
    with pytest.raises(RuntimeError, match="task census changed"):
        rollout.deploy()

    state = application()
    posts.clear()
    events.clear()
    monkeypatch.setattr(rollout, "_verify_tasks", lambda *_args: {})
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
    monkeypatch.setattr(rollout, "_verify_redis", lambda: None)
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
            state["placementSwarm"] = payload["placementSwarm"]
        if len(calls) == fail_on:
            raise RuntimeError("HTTP request failed; state is ambiguous")

    monkeypatch.setattr(rollout, "_post_json", fail)
    with pytest.raises(RuntimeError, match="ambiguous"):
        rollout.deploy()
    assert len(calls) == fail_on


def _wire_final_evidence(monkeypatch, tasks):
    monkeypatch.setattr(rollout, "_eligible_nodes", lambda: rollout.ELIGIBLE_NODES)
    monkeypatch.setattr(rollout, "_service_spec", lambda _name: service_spec(CANDIDATE, REVISION))
    monkeypatch.setattr(rollout, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout,
        "_verify_public",
        lambda _revision, instances: {
            url: sorted(instances) for url in rollout.HEALTH_URLS
        },
    )
    monkeypatch.setattr(rollout, "_verify_tasks", lambda *_args: tasks)


def test_evidence_requires_candidate_on_both_domains(monkeypatch, tmp_path, capsys):
    path = tmp_path / "evidence.jsonl"
    rows = [
        {"ok": True, "url": url, "revision": REVISION, "instance": "one"}
        for url in rollout.HEALTH_URLS
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    rollout._task_proof_path().write_text(json.dumps({
        "revision": REVISION,
        "image": CANDIDATE,
        "baselineRevision": "baseline",
        "tasks": ["task1", "task2", "task3"],
        "nodes": ["haiku-4", "haiku-5", "haiku-9"],
        "instances": ["one", "two", "three"],
        "publicInstances": {
            url: ["three", "one", "two"] for url in rollout.HEALTH_URLS
        },
    }))
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setattr(
        rollout, "_application", lambda *_args: application(CANDIDATE, REVISION)
    )
    _wire_final_evidence(
        monkeypatch,
        {
            "tasks": ["task1", "task2", "task3"],
            "nodes": ["haiku-4", "haiku-5", "haiku-9"],
            "instances": ["one", "two", "three"],
        },
    )
    rollout.evidence()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["publicInstances"] == {
        url: ["one", "three", "two"] for url in rollout.HEALTH_URLS
    }
    rows[0]["revision"] = "baseline"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="each public domain"):
        rollout.evidence()
    rows[0]["revision"] = "foreign"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="foreign revision"):
        rollout.evidence()
    rows[0]["revision"] = REVISION
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(
        rollout,
        "verify_route",
        lambda *_args: (_ for _ in ()).throw(ValueError("route changed")),
    )
    with pytest.raises(ValueError, match="route changed"):
        rollout.evidence()


@pytest.mark.parametrize("failure", ["missing", "unknown-proof", "unknown-monitor"])
def test_evidence_rejects_public_instance_set_mismatch(monkeypatch, tmp_path, failure):
    path = tmp_path / "evidence.jsonl"
    rows = [
        {"ok": True, "url": url, "revision": REVISION, "instance": "one"}
        for url in rollout.HEALTH_URLS
    ]
    coverage = {url: ["one", "two", "three"] for url in rollout.HEALTH_URLS}
    if failure == "missing":
        coverage[rollout.HEALTH_URLS[0]] = ["one", "two"]
    elif failure == "unknown-proof":
        coverage[rollout.HEALTH_URLS[0]] = ["one", "two", "unknown"]
    else:
        rows[0]["instance"] = "unknown"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    rollout._task_proof_path().write_text(json.dumps({
        "revision": REVISION,
        "image": CANDIDATE,
        "baselineRevision": "baseline",
        "tasks": ["task1", "task2", "task3"],
        "nodes": ["haiku-4", "haiku-5", "haiku-9"],
        "instances": ["one", "two", "three"],
        "publicInstances": coverage,
    }))
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setattr(
        rollout, "_application", lambda *_args: application(CANDIDATE, REVISION)
    )
    _wire_final_evidence(
        monkeypatch,
        {
            "tasks": ["task1", "task2", "task3"],
            "nodes": ["haiku-4", "haiku-5", "haiku-9"],
            "instances": ["one", "two", "three"],
        },
    )
    message = (
        "unknown candidate instance"
        if failure == "unknown-monitor"
        else "every authoritative task"
    )
    with pytest.raises(RuntimeError, match=message):
        rollout.evidence()


def test_evidence_rejects_a_task_replaced_after_deploy(monkeypatch, tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {"ok": True, "url": url, "revision": REVISION, "instance": "one"}
            )
            for url in rollout.HEALTH_URLS
        )
        + "\n"
    )
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(path))
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    rollout._task_proof_path().write_text(json.dumps({
        "revision": REVISION,
        "image": CANDIDATE,
        "baselineRevision": "baseline",
        "tasks": ["task1", "task2", "task3"],
        "nodes": ["haiku-4", "haiku-5", "haiku-9"],
        "instances": ["one", "two", "three"],
        "publicInstances": {
            url: ["one", "two", "three"] for url in rollout.HEALTH_URLS
        },
    }))
    monkeypatch.setattr(
        rollout, "_application", lambda *_args: application(CANDIDATE, REVISION)
    )
    _wire_final_evidence(
        monkeypatch,
        {
            "tasks": ["task1", "task2", "task4"],
            "nodes": ["haiku-4", "haiku-5", "haiku-18"],
            "instances": ["one", "two", "four"],
        },
    )
    with pytest.raises(RuntimeError, match="task census changed before final evidence"):
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


def test_request_json_failure_carries_curl_diagnosis(monkeypatch):
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 56, "", "curl: (56) Recv failure: Connection reset by peer"
        ),
    )
    with pytest.raises(rollout.CurlError) as error:
        rollout._request_json("https://example.test/health")
    assert error.value.curl_exit == 56
    assert "curl exit 56" in str(error.value)
    assert "Connection reset" in str(error.value)


def test_request_json_uses_namespace_argv_and_existing_bounds(monkeypatch):
    invocation = []
    url = "http://10.0.1.38:11235/health"

    def run(command, **kwargs):
        invocation.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(health("one")), "")

    monkeypatch.setattr(rollout.subprocess, "run", run)

    assert rollout._request_json(url, network_namespace_pid=4321) == health("one")
    assert invocation == [
        (
            [
                "nsenter",
                "--net=/proc/4321/ns/net",
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "--connect-timeout",
                "5",
                "--max-time",
                "15",
                "--max-filesize",
                "65536",
                "--config",
                "-",
                url,
            ],
            {"input": "", "text": True, "capture_output": True, "timeout": 20},
        )
    ]


@pytest.mark.parametrize("network_namespace_pid", [True, 1.5, 0])
def test_request_json_rejects_an_invalid_network_namespace_pid(
    monkeypatch, network_namespace_pid
):
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid PID must not execute a command"),
    )

    with pytest.raises(ValueError, match="PID must be a positive integer"):
        rollout._request_json(
            "http://10.0.1.38:11235/health",
            network_namespace_pid=network_namespace_pid,
        )


def test_monitor_failure_sample_is_attributable(monkeypatch, tmp_path):
    """A failed probe must record what failed — the 2026-08-31 run went red
    on a single sample that said only {"error": "RuntimeError"}."""
    evidence_path = tmp_path / "evidence.jsonl"
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(evidence_path))
    monkeypatch.setenv("ROLLOUT_MONITOR_ARMED_PATH", str(tmp_path / "armed"))
    monkeypatch.setenv("ROLLOUT_MONITOR_STOP_PATH", str(tmp_path / "stop"))
    calls = []

    def probe(url, *_args):
        calls.append(url)
        if len(calls) <= len(rollout.HEALTH_URLS):
            if len(calls) == len(rollout.HEALTH_URLS):
                pass  # first round healthy: arms the monitor
            return health()
        (tmp_path / "stop").touch()
        raise rollout.CurlError("HTTP request failed: curl exit 56: reset", 56)

    monkeypatch.setattr(rollout, "_request_json", probe)
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="recorded"):
        rollout.monitor()
    samples = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    failed = [s for s in samples if not s["ok"]]
    assert failed
    assert failed[0]["curl_exit"] == 56
    assert failed[0]["error"].startswith("CurlError: HTTP request failed: curl exit 56")


def _node_commands(nodes):
    """Stub docker node ls/inspect for a fleet of (hostname, labeled, state, availability)."""
    def run(cmd, **_kwargs):
        if cmd[1] == "node" and cmd[2] == "ls":
            return subprocess.CompletedProcess(cmd, 0, "".join(f"id{i}\n" for i in range(len(nodes))), "")
        lines = []
        for hostname, labeled, state, availability in nodes:
            labels = {"crawl4ai-eligible": "true"} if labeled else {}
            lines.append("\t".join([
                json.dumps(hostname), json.dumps(labels),
                json.dumps(state), json.dumps(availability),
            ]))
        return subprocess.CompletedProcess(cmd, 0, "".join(line + "\n" for line in lines), "")
    return run


def test_eligible_nodes_excludes_a_down_member_without_failing(monkeypatch):
    # A down node is a capacity event: it leaves the result, not raises.
    monkeypatch.setattr(rollout.subprocess, "run", _node_commands([
        ("haiku-0", False, "ready", "active"),
        ("haiku-4", True, "down", "active"),
        ("haiku-5", True, "ready", "active"),
        ("haiku-9", True, "ready", "active"),
        ("haiku-18", True, "ready", "active"),
    ]))
    assert rollout._eligible_nodes() == frozenset({"haiku-5", "haiku-9", "haiku-18"})


def test_eligible_nodes_flags_membership_drift_even_when_ready(monkeypatch):
    monkeypatch.setattr(rollout.subprocess, "run", _node_commands([
        ("haiku-4", True, "ready", "active"),
        ("haiku-5", True, "ready", "active"),
        ("haiku-9", True, "ready", "active"),
    ]))
    with pytest.raises(RuntimeError, match="inventory drifted"):
        rollout._eligible_nodes()


def test_policy_accepts_legacy_capped_placement_during_transition():
    app = application()
    app["placementSwarm"] = copy.deepcopy(rollout._LEGACY_PLACEMENT)
    rollout._policy(app)


def test_policy_rejects_foreign_placement():
    app = application()
    app["placementSwarm"] = {"Constraints": [rollout.NODE_CONSTRAINT], "MaxReplicas": 2}
    with pytest.raises(ValueError, match="placement drifted"):
        rollout._policy(app)


def test_running_spec_accepts_legacy_capped_placement():
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout._LEGACY_PLACEMENT)
    rollout._running_spec(spec, BASELINE, rollout._labels("baseline"))


def test_running_spec_strict_mode_rejects_legacy_placement():
    # The post-deploy readback must prove the cap is actually gone from the
    # rendered service — legacy tolerance is for the pre-write baseline only.
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout._LEGACY_PLACEMENT)
    with pytest.raises(ValueError, match="placement drifted"):
        rollout._running_spec(
            spec, BASELINE, rollout._labels("baseline"), placements=(rollout.PLACEMENT,)
        )


def _healed_baseline_rows():
    # haiku-4 died holding slot 1; Swarm healed the replica onto haiku-18,
    # and the ghost task on the dead node can never confirm its shutdown.
    rows = [
        {"ID": "task1", "Name": "crawl4ai.1", "Node": "haiku-18",
         "DesiredState": "Running", "CurrentState": "Running 1m"},
        {"ID": "ghost1", "Name": "crawl4ai.1", "Node": "haiku-4",
         "DesiredState": "Shutdown", "CurrentState": "Running 12 hours ago"},
        {"ID": "task2", "Name": "crawl4ai.2", "Node": "haiku-5",
         "DesiredState": "Running", "CurrentState": "Running 1h"},
        {"ID": "old2", "Name": "crawl4ai.2", "Node": "haiku-9",
         "DesiredState": "Shutdown", "CurrentState": "Shutdown 1h"},
        {"ID": "task3", "Name": "crawl4ai.3", "Node": "haiku-9",
         "DesiredState": "Running", "CurrentState": "Running 1h"},
        {"ID": "old3", "Name": "crawl4ai.3", "Node": "haiku-5",
         "DesiredState": "Shutdown", "CurrentState": "Shutdown 1h"},
    ]
    runtimes = {
        f"task{index}": {
            "container": f"container{index}",
            "labels": rollout._labels("baseline"),
            "image": BASELINE,
            "addresses": {f"10.0.1.{index}"},
        }
        for index in range(1, 4)
    }
    return rows, runtimes


def _wire_verify_tasks(monkeypatch, rows, runtimes, revision="baseline"):
    current = [
        row for row in rows if str(row.get("DesiredState", "")).lower() == "running"
    ]
    network = [{
        "Id": "network",
        "Services": {
            "crawl4ai": {
                "Tasks": [
                    {
                        "Name": f"{row['Name']}.{row['ID']}",
                        "EndpointIP": next(iter(runtimes[row["ID"]]["addresses"])),
                    }
                    for row in current
                ]
            }
        },
    }]

    def run(command, **_kwargs):
        if command[1:3] == ["service", "ps"]:
            return subprocess.CompletedProcess(
                command, 0, "\n".join(json.dumps(row) for row in rows), ""
            )
        return subprocess.CompletedProcess(command, 0, json.dumps(network), "")

    monkeypatch.setattr(rollout.subprocess, "run", run)
    monkeypatch.setattr(rollout, "_verify_ingress_host", lambda: None)
    monkeypatch.setattr(
        rollout,
        "_task_state",
        lambda task: (
            ("running", "running") if task.startswith("task")
            else ("shutdown", "running") if task.startswith("ghost")
            else ("shutdown", "rejected") if task.startswith("reject")
            else ("shutdown", "shutdown")
        ),
    )
    monkeypatch.setattr(rollout, "_task_runtime", lambda task, _network: runtimes[task])
    monkeypatch.setattr(
        rollout,
        "_request_json",
        lambda url, *_args: health(
            next(r["container"] for r in runtimes.values() if next(iter(r["addresses"])) in url),
            revision=revision,
        ),
    )


def test_baseline_tolerates_a_ghost_predecessor_on_a_down_node(monkeypatch):
    rows, runtimes = _healed_baseline_rows()
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    proof = rollout._verify_tasks("crawl4ai", BASELINE, "baseline", ready, False)
    assert sorted(proof["nodes"]) == ["haiku-18", "haiku-5", "haiku-9"]
    # The same state must still fail a converged proof: the ghost's shutdown
    # was never confirmed, so this deploy's own withdrawals stay strict.
    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout._verify_tasks("crawl4ai", BASELINE, "baseline", ready)


def test_baseline_tolerates_healed_colocation_but_converged_does_not(monkeypatch):
    rows, runtimes = _healed_baseline_rows()
    for row in rows:
        if row["ID"] == "task1":
            row["Node"] = "haiku-5"  # healed replica doubled up
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    proof = rollout._verify_tasks("crawl4ai", BASELINE, "baseline", ready, False)
    assert sorted(proof["nodes"]) == ["haiku-5", "haiku-9"]
    with pytest.raises(RuntimeError, match="not on distinct eligible nodes"):
        rollout._verify_tasks("crawl4ai", BASELINE, "baseline", ready)


def _deploy_env(monkeypatch):
    monkeypatch.setenv("DOKPLOY_URL", "https://dokploy")
    monkeypatch.setenv("DOKPLOY_API_KEY", "key")
    monkeypatch.setenv("APPLICATION_ID", "app")
    monkeypatch.setenv("IMAGE", "registry.example/crawl4ai")
    monkeypatch.setenv("IMAGE_DIGEST", "sha256:candidate")
    monkeypatch.setenv("GITHUB_SHA", REVISION)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")


def test_deploy_requires_ready_capacity_not_full_fleet(monkeypatch):
    _deploy_env(monkeypatch)
    monkeypatch.setattr(rollout, "_application", lambda *_args: application())
    monkeypatch.setattr(rollout, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout, "_service_spec", lambda _name: service_spec())
    monkeypatch.setattr(rollout, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(rollout, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout, "verify_route", lambda *_args: None)

    # Three ready of four labeled nodes is enough capacity: the gate passes
    # and evaluation reaches the baseline task census.
    monkeypatch.setattr(rollout, "_eligible_nodes", lambda: frozenset({"haiku-5", "haiku-9", "haiku-18"}))
    monkeypatch.setattr(
        rollout,
        "_verify_tasks",
        lambda *_args: (_ for _ in ()).throw(SystemExit("capacity gate passed")),
    )
    with pytest.raises(SystemExit, match="capacity gate passed"):
        rollout.deploy()

    # Two ready nodes cannot place three replicas: fail before the census.
    monkeypatch.setattr(rollout, "_verify_tasks", lambda *_args: pytest.fail("census must not run"))
    monkeypatch.setattr(rollout, "_eligible_nodes", lambda: frozenset({"haiku-9", "haiku-18"}))
    with pytest.raises(RuntimeError, match="not enough Ready eligible nodes"):
        rollout.deploy()


def test_deploy_rejects_a_reintroduced_cap_once_the_record_converged(monkeypatch):
    # Record already migrated to the capless shape; a capped LIVE spec is
    # drift, not transition residue.
    _deploy_env(monkeypatch)
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout._LEGACY_PLACEMENT)
    monkeypatch.setattr(rollout, "_application", lambda *_args: application())
    monkeypatch.setattr(rollout, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout, "_service_spec", lambda _name: spec)
    monkeypatch.setattr(rollout, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    with pytest.raises(ValueError, match="running placement drifted"):
        rollout.deploy()


def test_deploy_tolerates_legacy_live_spec_only_while_record_is_legacy(monkeypatch):
    _deploy_env(monkeypatch)
    state = application()
    state["placementSwarm"] = copy.deepcopy(rollout._LEGACY_PLACEMENT)
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout._LEGACY_PLACEMENT)
    monkeypatch.setattr(rollout, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(rollout, "_update_state", lambda _name: "completed")
    monkeypatch.setattr(rollout, "_service_spec", lambda _name: spec)
    monkeypatch.setattr(rollout, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout, "_verify_redis", lambda: (_ for _ in ()).throw(SystemExit("baseline spec accepted"))
    )
    with pytest.raises(SystemExit, match="baseline spec accepted"):
        rollout.deploy()


def test_baseline_keeps_unassigned_attempts_strict(monkeypatch):
    # A rejected scheduling attempt has no node; it is not a stranded ghost
    # and must not excuse the withdrawal proof.
    rows, runtimes = _healed_baseline_rows()
    rows.insert(1, {"ID": "reject1", "Name": "crawl4ai.1", "Node": "",
                    "DesiredState": "Shutdown", "CurrentState": "Rejected 1h"})
    ready = frozenset({"haiku-5", "haiku-9", "haiku-18"})
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    with pytest.raises(RuntimeError, match="contradicts the start-first rollout"):
        rollout._verify_tasks("crawl4ai", BASELINE, "baseline", ready, False)
OBSERVER_INCOMPLETE = observer.CoverageSnapshot(3, 2, 1)
OBSERVER_COMPLETE = observer.CoverageSnapshot(3, 3, 3)
OBSERVER_MISMATCH = observer.NetworkDbFdbComparison.DESTINATION_MISMATCH


def test_initial_incomplete_is_reported_once_until_full_recovery(tmp_path):
    latch = observer.CoverageEpisodeLatch(tmp_path / "coverage-state")

    assert latch.observe(
        OBSERVER_INCOMPLETE, OBSERVER_MISMATCH
    ) == observer.CoverageDiagnostic(3, 2, 1, OBSERVER_MISMATCH)
    latch.acknowledge()
    assert (
        latch.observe(OBSERVER_INCOMPLETE, observer.NetworkDbFdbComparison.MISSING)
        is None
    )
    assert (
        latch.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )
    assert latch.observe(
        OBSERVER_INCOMPLETE, observer.NetworkDbFdbComparison.MISSING
    ) == observer.CoverageDiagnostic(3, 2, 1, observer.NetworkDbFdbComparison.MISSING)


def test_full_to_incomplete_is_reported_once_until_recovery(tmp_path):
    latch = observer.CoverageEpisodeLatch(tmp_path / "coverage-state")

    assert (
        latch.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )
    assert latch.observe(
        OBSERVER_INCOMPLETE, OBSERVER_MISMATCH
    ) == observer.CoverageDiagnostic(3, 2, 1, OBSERVER_MISMATCH)
    latch.acknowledge()
    assert (
        latch.observe(OBSERVER_INCOMPLETE, observer.NetworkDbFdbComparison.INCONCLUSIVE)
        is None
    )
    assert (
        latch.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )


def test_latch_state_survives_a_restart_and_resets_after_recovery(tmp_path):
    state_path = tmp_path / "coverage-state"
    first = observer.CoverageEpisodeLatch(state_path)

    assert first.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is not None
    assert state_path.read_bytes() == b"pending\n"
    restarted = observer.CoverageEpisodeLatch(state_path)
    assert restarted.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is not None
    restarted.acknowledge()
    assert state_path.read_bytes() == b"open\n"
    emitted = observer.CoverageEpisodeLatch(state_path)
    assert emitted.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is None
    assert (
        emitted.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )
    assert state_path.read_bytes() == b"closed\n"
    recovered = observer.CoverageEpisodeLatch(state_path)
    assert recovered.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is not None


def test_coverage_metrics_are_fixed_unlabeled_and_zero_is_incomplete():
    snapshot = observer.CoverageSnapshot(3, 2, 1)

    assert snapshot.complete == 0
    assert snapshot.metrics_text() == (
        "crawl4ai_authoritative_task_count 3\n"
        "crawl4ai_direct_healthy_task_count 2\n"
        "crawl4ai_public_covered_task_count 1\n"
        "crawl4ai_coverage_complete 0\n"
    )
    empty = observer.CoverageSnapshot(0, 0, 0)
    assert empty.complete == 0
    assert empty.metrics_text().endswith("crawl4ai_coverage_complete 0\n")
    two_surviving_tasks = observer.CoverageSnapshot(2, 2, 2)
    assert two_surviving_tasks.complete == 0
    with pytest.raises(ValueError, match="exceeds direct_healthy_count"):
        observer.CoverageSnapshot(3, 1, 2)


def test_public_coverage_values_are_redacted_aggregate_counts(tmp_path):
    diagnostic = observer.CoverageEpisodeLatch(tmp_path / "coverage-state").observe(
        OBSERVER_INCOMPLETE, OBSERVER_MISMATCH
    )

    assert diagnostic is not None
    assert [field.name for field in fields(observer.CoverageSnapshot)] == [
        "authoritative_task_count",
        "direct_healthy_count",
        "public_covered_count",
    ]
    assert [field.name for field in fields(observer.CoverageDiagnostic)] == [
        "authoritative_task_count",
        "direct_healthy_count",
        "public_covered_count",
        "networkdb_fdb_comparison",
    ]
    assert diagnostic.text() == (
        "crawl4ai coverage incomplete: "
        "authoritative_tasks=3 direct_healthy=2 public_covered=1 "
        "networkdb_fdb=destination_mismatch"
    )
    with pytest.raises(TypeError, match="integer"):
        observer.CoverageSnapshot("task-id", 0, 0)
    with pytest.raises(TypeError, match="CoverageSnapshot"):
        observer.CoverageEpisodeLatch(tmp_path / "other-state").observe(
            object(), OBSERVER_MISMATCH
        )
    with pytest.raises(TypeError, match="NetworkDbFdbComparison"):
        observer.CoverageEpisodeLatch(tmp_path / "other-state").observe(
            OBSERVER_INCOMPLETE, "10.0.1.38"
        )


def test_sample_reuses_rollout_task_and_health_owners(monkeypatch):
    rows = [
        {
            "ID": f"task-{index}",
            "Node": f"node-{index}",
            "DesiredState": "Running",
            "CurrentState": "Running 1m",
        }
        for index in range(3)
    ]
    runtimes = {
        f"task-{index}": {
            "container": f"container-{index}",
            "addresses": {f"10.0.1.{index + 10}"},
            "labels": {"otel.service.version": REVISION},
        }
        for index in range(3)
    }
    monkeypatch.setattr(observer.rollout, "_verify_ingress_host", lambda: 4321)
    monkeypatch.setattr(observer.rollout, "_service_tasks", lambda _service: rows)
    monkeypatch.setattr(
        observer,
        "_network_details",
        lambda: {
            "Id": "network",
            "Services": {
                "crawl4ai": {
                    "Tasks": [
                        {
                            "Name": task_id,
                            "Info": {"Host IP": f"100.0.0.{index + 10}"},
                        }
                        for index, task_id in enumerate(runtimes)
                    ]
                }
            },
        },
    )
    monkeypatch.setattr(
        observer.rollout,
        "_task_runtime",
        lambda task_id, _network: runtimes[task_id],
    )
    direct_probe = []

    def direct_health(url, *, network_namespace_pid):
        direct_probe.append((url, network_namespace_pid))
        return health(
            next(
                runtime["container"]
                for runtime in runtimes.values()
                if next(iter(runtime["addresses"])) in url
            ),
            revision=REVISION,
        )

    monkeypatch.setattr(observer.rollout, "_request_json", direct_health)
    monkeypatch.setattr(observer, "_public_coverage", lambda _direct, _deadline: 2)
    monkeypatch.setattr(
        observer,
        "_fdb_comparison",
        lambda *_args: observer.NetworkDbFdbComparison.CONSISTENT,
    )

    snapshot, comparison = observer.sample("crawl4ai")

    assert snapshot == observer.CoverageSnapshot(3, 3, 2)
    assert comparison is observer.NetworkDbFdbComparison.CONSISTENT
    assert len(direct_probe) == 3
    assert {namespace_pid for _url, namespace_pid in direct_probe} == {4321}


def test_sample_rejects_a_task_census_change(monkeypatch):
    before = [
        {
            "ID": "task-before",
            "Node": "node-1",
            "DesiredState": "Running",
            "CurrentState": "Pending 1s",
        }
    ]
    after = [{**before[0], "ID": "task-after"}]
    calls = iter((before, after))
    monkeypatch.setattr(observer.rollout, "_verify_ingress_host", lambda: None)
    monkeypatch.setattr(
        observer.rollout, "_service_tasks", lambda _service: next(calls)
    )
    monkeypatch.setattr(observer, "_network_details", lambda: {"Id": "network"})

    with pytest.raises(RuntimeError, match="task census changed"):
        observer.sample("crawl4ai")


def test_public_coverage_accepts_each_task_revision_and_intersects_routes(monkeypatch):
    direct = {"one": "revision-one", "two": "revision-two"}
    route_call = {}

    def public_health(url):
        route = url.split("/health", 1)[0]
        route_call[route] = route_call.get(route, 0) + 1
        instance = "one" if route_call[route] == 1 else "two"
        return health(instance, revision=direct[instance])

    monkeypatch.setattr(observer.rollout, "_request_json", public_health)
    monkeypatch.setattr(observer.rollout, "PUBLIC_PROOF_ATTEMPTS", 2)

    assert observer._public_coverage(direct, observer.time.monotonic() + 10) == 2


def test_public_coverage_stops_at_the_sample_deadline(monkeypatch):
    monkeypatch.setattr(
        observer.rollout,
        "_request_json",
        lambda _url: pytest.fail("deadline must stop the public request"),
    )

    with pytest.raises(TimeoutError, match="deadline"):
        observer._public_coverage({"one": REVISION}, observer.time.monotonic() - 1)


def test_fdb_comparison_distinguishes_missing_and_wrong_destinations(monkeypatch):
    task = [{"ID": "task-1", "Node": "node-1"}]
    runtime = {"task-1": {"addresses": {"10.0.1.38"}}}
    data_path = {"task-1": "100.0.0.9"}

    def bridge(destination):
        return subprocess.CompletedProcess(
            [],
            0,
            ""
            if destination is None
            else f"02:42:0a:00:01:26 dst {destination} self permanent\n",
            "",
        )

    monkeypatch.setattr(
        observer.subprocess, "run", lambda *_args, **_kwargs: bridge(None)
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.MISSING
    )
    monkeypatch.setattr(
        observer.subprocess, "run", lambda *_args, **_kwargs: bridge("100.0.0.8")
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.DESTINATION_MISMATCH
    )
    monkeypatch.setattr(
        observer.subprocess, "run", lambda *_args, **_kwargs: bridge("100.0.0.9")
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.CONSISTENT
    )
    monkeypatch.setattr(
        observer.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "02:42:0a:00:01:26 malformed\n", ""
        ),
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.INCONCLUSIVE
    )


def test_sampling_failure_replaces_a_stale_healthy_sample(monkeypatch, tmp_path):
    state_path = tmp_path / "state"
    metrics_path = tmp_path / "metrics"
    monkeypatch.setenv("CRAWL4AI_SWARM_SERVICE", "crawl4ai")
    monkeypatch.setenv("CRAWL4AI_OBSERVER_STATE_PATH", str(state_path))
    monkeypatch.setenv("CRAWL4AI_OBSERVER_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        observer,
        "sample",
        lambda _service: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    observer.write_sample()

    assert metrics_path.read_text().endswith("crawl4ai_coverage_complete 0\n")
    assert state_path.read_bytes() == b"open\n"


def test_metrics_endpoint_exposes_only_the_fixed_contract(tmp_path):
    path = tmp_path / "metrics"
    path.write_text(OBSERVER_COMPLETE.metrics_text())
    metrics = observer_server.MetricsFile(path, 60)
    server = observer_server.http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), observer_server.handler(metrics)
    )
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read().decode() == OBSERVER_COMPLETE.metrics_text()
        connection.request("GET", "/tasks")
        assert connection.getresponse().status == 404
        connection.request("POST", "/metrics")
        assert connection.getresponse().status == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_metrics_endpoint_fails_closed_when_the_sampler_is_stale(monkeypatch, tmp_path):
    path = tmp_path / "metrics"
    path.write_text(OBSERVER_COMPLETE.metrics_text())
    monkeypatch.setattr(observer_server.time, "time", lambda: path.stat().st_mtime + 61)

    with pytest.raises(TimeoutError, match="stale"):
        observer_server.MetricsFile(path, 60).read()
