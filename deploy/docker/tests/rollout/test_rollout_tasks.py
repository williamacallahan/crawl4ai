import json
import subprocess

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import verify_rollout as rollout
from conftest import BASELINE, CANDIDATE, ELIGIBLE_NODES, REVISION, _healed_baseline_rows, _wire_verify_tasks, health


@pytest.mark.parametrize("status", [
    None,
    {"State": "rollback_paused", "StartedAt": "2026-10-22T10:00:00.123456789Z"},
])
def test_update_state_reads_state_and_start_from_one_service_snapshot(monkeypatch, status):
    def run(command, **_kwargs):
        assert command == [
            "docker", "service", "inspect", "crawl4ai", "--format", "{{json .UpdateStatus}}",
        ]
        return subprocess.CompletedProcess(command, 0, json.dumps(status) + "\n", "")

    monkeypatch.setattr(rollout_tasks.subprocess, "run", run)

    expected = (status["State"], status["StartedAt"]) if status else ("", "")
    assert rollout_tasks._update_state("crawl4ai") == expected


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


def yaml_dump(value):
    import yaml

    return yaml.safe_dump(value)


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
                    "rule": "Host(`crawl4ai.iocloudhost.net`)",
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
        rollout_http,
        "_request_json",
        lambda *_args: route(
            custom_definition, custom_reference, wrong_url, wrong_router
        ),
    )
    with pytest.raises(ValueError):
        rollout_tasks.verify_route("https://dokploy", "key", "app", "crawl4ai")


def test_route_accepts_stock_dokploy_file(monkeypatch):
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: route())
    rollout_tasks.verify_route("https://dokploy", "key", "app", "crawl4ai")


def test_task_runtime_parses_container_identity_and_overlay(monkeypatch):
    output = (
        '"container123456789"\t'
        + json.dumps(rollout_policy._labels(REVISION))
        + '\t"registry.example/crawl4ai@sha256:candidate"\t'
        + '[{"Network":{"ID":"network"},"Addresses":["10.0.1.23/24"]}]\n'
    )
    monkeypatch.setattr(
        rollout.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, output, ""),
    )
    runtime = rollout_tasks._task_runtime("task", "network")
    assert runtime == {
        "container": "container123",
        "labels": rollout_policy._labels(REVISION),
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
        + json.dumps(rollout_policy._labels(REVISION))
        + '\t"registry.example/crawl4ai@sha256:candidate"\t'
        + "null\n"
    )

    def run(command, **_kwargs):
        assert "{{if .Status.ContainerStatus}}" in command[command.index("--format") + 1]
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(rollout.subprocess, "run", run)

    runtime = rollout_tasks._task_runtime("task", "network")

    assert runtime["container"] == ""
    assert runtime["addresses"] == set()


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
        rollout_tasks._verify_ingress_host()


def test_ingress_host_uses_fixed_read_only_docker_projections(monkeypatch):
    invocation = []

    def run(command, **kwargs):
        invocation.append((command, kwargs))
        stdout = "container-id\n" if command[1] == "ps" else "4321\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(rollout.subprocess, "run", run)

    assert rollout_tasks._verify_ingress_host() == 4321
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
        rollout_policy.HEALTH_URLS[0]: iter(["three", "one", "two"]),
        rollout_policy.HEALTH_URLS[1]: iter(["two", "three", "one"]),
    }

    def request(url, *_args):
        route_url = next(route for route in rollout_policy.HEALTH_URLS if url.startswith(route))
        return health(next(responses[route_url]))

    monkeypatch.setattr(rollout_http, "_request_json", request)
    assert rollout_tasks._verify_public(REVISION, instances) == {
        url: sorted(instances) for url in rollout_policy.HEALTH_URLS
    }


def test_public_proof_rejects_a_missing_healthy_backend(monkeypatch):
    monkeypatch.setattr(rollout_policy, "PUBLIC_PROOF_ATTEMPTS", 3)
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health("one"))
    with pytest.raises(RuntimeError, match="did not observe every authoritative task"):
        rollout_tasks._verify_public(REVISION, ["one", "two", "three"])


def test_public_proof_rejects_an_unknown_instance(monkeypatch):
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health("unknown"))
    with pytest.raises(RuntimeError, match="unknown task instance"):
        rollout_tasks._verify_public(REVISION, ["one", "two", "three"])


@pytest.mark.parametrize(
    "response",
    [health("one", revision="wrong"), {**health("one"), "status": "degraded"}],
)
def test_public_proof_rejects_wrong_revision_or_unhealthy_response(monkeypatch, response):
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: response)
    with pytest.raises(RuntimeError, match="unhealthy or wrong-revision"):
        rollout_tasks._verify_public(REVISION, ["one", "two", "three"])


def test_eligible_nodes_excludes_a_down_member_without_failing(monkeypatch):
    # Down nodes retain membership but cannot supply rollout capacity.
    monkeypatch.setattr(rollout.subprocess, "run", _node_commands([
        ("haiku-0", False, "ready", "active"),
        ("haiku-18", True, "down", "drain"),
        ("haiku-5", True, "ready", "active"),
        ("haiku-6", True, "ready", "active"),
        ("haiku-9", True, "ready", "active"),
    ]))
    assert rollout_tasks._eligible_nodes() == (
        ELIGIBLE_NODES, frozenset({"haiku-5", "haiku-6", "haiku-9"})
    )


def test_eligible_nodes_tracks_live_label_membership(monkeypatch):
    monkeypatch.setattr(rollout.subprocess, "run", _node_commands([
        ("haiku-4", True, "ready", "active"),
        ("haiku-5", True, "ready", "active"),
        ("haiku-9", True, "ready", "active"),
    ]))
    labeled = frozenset({"haiku-4", "haiku-5", "haiku-9"})
    assert rollout_tasks._eligible_nodes() == (labeled, labeled)


def test_task_proof_rejects_a_node_without_the_live_eligibility_label(monkeypatch):
    rows, runtimes = _healed_baseline_rows()
    _wire_verify_tasks(monkeypatch, rows, runtimes)
    labeled = ELIGIBLE_NODES - {"haiku-18"}
    with pytest.raises(RuntimeError, match="not on eligible nodes"):
        rollout_tasks._verify_tasks("crawl4ai", BASELINE, "baseline", (labeled, labeled), False)
