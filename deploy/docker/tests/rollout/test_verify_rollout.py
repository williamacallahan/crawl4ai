import copy
import json

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import verify_rollout as rollout
from conftest import BASELINE, CANDIDATE, ELIGIBLE_NODES, REVISION, _deploy_env, application, health, service_spec


PRIOR_UPDATE_STARTED = "2026-10-21T10:00:00.123456789Z"
CURRENT_UPDATE_STARTED = "2026-10-21T10:00:01.123456789Z"


@pytest.mark.parametrize("placement", rollout_policy.PLACEMENTS)
def test_deploy_uses_only_stock_update_and_deploy(monkeypatch, tmp_path, capsys, placement):
    state = application(placement=placement)
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
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy,
        "_deployments",
        lambda *_args: copy.deepcopy(deployments),
    )
    monkeypatch.setattr(rollout_http, "_post_json", post)
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", CURRENT_UPDATE_STARTED if state["dockerImage"] == CANDIDATE else PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(state["dockerImage"], state["labelsSwarm"]["otel.service.version"], placement))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: events.append("verify_redis"))
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_dokploy,
        "_wait_deployment",
        lambda *_args: {"deploymentId": "new", "status": "done"},
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_tasks",
        lambda *_args: {"tasks": ["1", "2", "3"], "nodes": ["haiku-5", "haiku-6", "haiku-9"], "instances": ["a", "b", "c"]},
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(
        rollout_http,
        "_request_json",
        lambda url, *_args: health(revision="baseline" if "baseline=" in url else REVISION),
    )
    rollout.deploy()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["publicInstances"] == {
        url: ["a", "b", "c"] for url in rollout_policy.HEALTH_URLS
    }
    # The Redis proof must precede the first write, not merely happen.
    assert events == ["verify_redis", "application.update", "application.deploy"]
    assert [url.rsplit("/", 1)[-1] for url, _ in posts] == [
        "application.update",
        "application.deploy",
    ]
    assert "idempotencyKey" not in posts[1][1]
    assert posts[0][1]["placementSwarm"] == placement
    assert "expectedDockerImage" not in posts[0][1]
    proof = json.loads(rollout_tasks._task_proof_path().read_text())
    assert proof["instances"] == ["a", "b", "c"]
    assert set(proof["publicInstances"]) == set(rollout_policy.HEALTH_URLS)
    assert proof["placement"] == placement

    state = application(placement=placement)
    posts.clear()
    events.clear()
    deployments[:] = [{"deploymentId": "running", "status": "running"}]
    with pytest.raises(RuntimeError, match="nonterminal Dokploy deployment"):
        rollout.deploy()
    assert not posts

    state = application(placement=placement)
    posts.clear()
    events.clear()
    deployments[:] = [{"deploymentId": "old", "status": "done"}]
    proofs = iter([
        {"tasks": ["old1", "old2", "old3"], "nodes": ["haiku-5", "haiku-6", "haiku-9"], "instances": ["old-a", "old-b", "old-c"]},
        {"tasks": ["old1", "old2", "old3"], "nodes": ["haiku-5", "haiku-6", "haiku-9"], "instances": ["old-a", "old-b", "old-c"]},
        {"tasks": ["1", "2", "3"], "nodes": ["haiku-5", "haiku-6", "haiku-9"], "instances": ["a", "b", "c"]},
        {"tasks": ["1", "2", "4"], "nodes": ["haiku-5", "haiku-6", "haiku-18"], "instances": ["a", "b", "d"]},
    ])
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", lambda *_args: next(proofs))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", CURRENT_UPDATE_STARTED if state["dockerImage"] == CANDIDATE else PRIOR_UPDATE_STARTED))
    with pytest.raises(RuntimeError, match="task census changed"):
        rollout.deploy()


def test_deploy_proves_an_already_deployed_candidate_without_resubmitting(
    monkeypatch, tmp_path, capsys
):
    # Dokploy makes no spec change for an image the record already carries, so
    # it opens no deployment row: a rerun that resubmits waits out the whole
    # admission timeout on a row that never appears.
    candidate = "registry.example/crawl4ai@sha256:candidate"
    state = application(image=candidate, revision=REVISION)
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_dokploy, "_wait_deployment", lambda *_args: pytest.fail("no deployment may be submitted")
    )
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_tasks",
        lambda *_args: {
            "tasks": ["1", "2", "3"],
            "nodes": ["haiku-5", "haiku-6", "haiku-9"],
            "instances": ["a", "b", "c"],
        },
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health(revision=REVISION))

    rollout.deploy()

    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] is None
    assert receipt["image"] == candidate
    assert receipt["revision"] == REVISION
    assert receipt["publicInstances"] == {
        url: ["a", "b", "c"] for url in rollout_policy.HEALTH_URLS
    }


@pytest.mark.parametrize("down_node", ["haiku-9", "haiku-6"])
def test_deploy_proves_an_already_deployed_candidate_with_a_non_redis_spare_down(
    monkeypatch, tmp_path, capsys, down_node
):
    # A rerun of a revision that is already live performs no start-first
    # rollout: it submits no update, starts no replacement, and retires no
    # predecessor, so it needs no spare overlap slot. With exactly REPLICAS
    # Ready eligible nodes and the absent node a non-Redis spare (so Redis on
    # haiku-18 still passes), the no-op rerun must prove it in place rather
    # than abort under the start-first spare gate. Regression for the no-op
    # rerun path added in f43ef89, whose spare gate ran above the branch.
    candidate = "registry.example/crawl4ai@sha256:candidate"
    state = application(image=candidate, revision=REVISION)
    ready = frozenset(ELIGIBLE_NODES - {down_node})
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_dokploy, "_wait_deployment", lambda *_args: pytest.fail("no deployment may be submitted")
    )
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ready))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(candidate, REVISION))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_tasks",
        lambda *_args: {
            "tasks": ["1", "2", "3"],
            "nodes": sorted(ready),
            "instances": ["a", "b", "c"],
        },
    )
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(rollout_http, "_request_json", lambda *_args: health(revision=REVISION))

    rollout.deploy()

    assert len(ready) == rollout_policy.REPLICAS
    assert rollout_policy.REDIS_NODE in ready
    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] is None
    assert receipt["image"] == candidate
    assert receipt["revision"] == REVISION
    assert receipt["publicInstances"] == {
        url: ["a", "b", "c"] for url in rollout_policy.HEALTH_URLS
    }


def test_deploy_real_rollout_passes_strict_converged_to_the_candidate_proof(
    monkeypatch, tmp_path
):
    # A real start-first rollout performs the withdrawals the converged proof
    # exists to check, so its candidate proof must stay strict
    # (converged=True) while only the baseline proof stays lenient.
    calls = []

    def spy(_app, image, _revision, _ready, converged=True, placement=rollout_policy.PLACEMENT):
        calls.append((image, converged))
        return {
            "tasks": ["1", "2", "3"],
            "nodes": ["haiku-5", "haiku-6", "haiku-9"],
            "instances": ["a", "b", "c"],
        }

    state = application()  # baseline running; candidate not yet deployed
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )

    def post(url, _key, payload):
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]
            state["placementSwarm"] = payload["placementSwarm"]

    monkeypatch.setattr(rollout_http, "_post_json", post)
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", CURRENT_UPDATE_STARTED if state["dockerImage"] == CANDIDATE else PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(
        rollout_tasks, "_service_spec",
        lambda _name: service_spec(state["dockerImage"], state["labelsSwarm"]["otel.service.version"]),
    )
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_dokploy, "_wait_deployment", lambda *_args: {"deploymentId": "new", "status": "done"})
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", spy)
    monkeypatch.setattr(
        rollout_tasks,
        "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )
    monkeypatch.setattr(
        rollout_http, "_request_json",
        lambda url, *_args: health(revision="baseline" if "baseline=" in url else REVISION),
    )

    rollout.deploy()

    # The baseline proof (image == BASELINE) stays lenient; every candidate
    # proof (image == CANDIDATE) the rollout takes for itself stays strict.
    assert calls[0] == (BASELINE, False), calls
    assert all(converged is (image == CANDIDATE) for image, converged in calls), calls


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
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(rollout_dokploy, "_deployments", lambda *_args: [])
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec())
    monkeypatch.setattr(rollout_dokploy, "_ensure_rollback_source", lambda _image: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", lambda *_args: {})
    monkeypatch.setattr(
        rollout_http,
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

    monkeypatch.setattr(rollout_http, "_post_json", fail)
    with pytest.raises(RuntimeError, match="ambiguous"):
        rollout.deploy()
    assert len(calls) == fail_on


@pytest.mark.parametrize("placement", rollout_policy.PLACEMENTS)
@pytest.mark.parametrize("ready_count", [2, 3, 4])
def test_deploy_capacity_follows_placement(monkeypatch, capsys, placement, ready_count):
    _deploy_env(monkeypatch)
    ready = frozenset(sorted(ELIGIBLE_NODES)[:ready_count])
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: application(placement=placement))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec(placement=placement))
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ready))
    monkeypatch.setattr(
        rollout_tasks, "_verify_tasks",
        lambda *_args: (_ for _ in ()).throw(SystemExit("capacity gate passed")),
    )
    if ready_count < 3:
        monkeypatch.setattr(rollout.sys, "argv", ["verify_rollout.py", "deploy"])
        with pytest.raises(RuntimeError, match="not enough Ready eligible nodes"):
            rollout.main()
        # The message names nodes, so the public line carries only its class.
        assert capsys.readouterr().err == (
            "rollout failure: command=deploy check=_deploy error=RuntimeError\n"
        )
    elif ready_count == 3 and placement.get("MaxReplicas") == 1:
        with pytest.raises(RuntimeError, match="start-first needs a spare eligible node"):
            rollout.deploy()
    else:
        with pytest.raises(SystemExit, match="capacity gate passed"):
            rollout.deploy()


def test_deploy_treats_an_absent_update_status_as_terminal(monkeypatch):
    # Docker reports no UpdateStatus for a service that was never updated or
    # was recreated. Refusing there strands every later deploy.
    _deploy_env(monkeypatch)
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: application())
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec())
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_tasks, "_verify_redis", lambda: (_ for _ in ()).throw(SystemExit("gate passed"))
    )

    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("", ""))
    with pytest.raises(SystemExit, match="gate passed"):
        rollout.deploy()


@pytest.mark.parametrize("active_state", ["updating", "rollback_started"])
def test_deploy_waits_for_a_prior_native_update_before_reading_baseline(monkeypatch, active_state):
    _deploy_env(monkeypatch)
    states = iter([(state, PRIOR_UPDATE_STARTED) for state in [active_state, active_state, "completed"]])
    snapshots = iter([{"appName": application()["appName"]}, application()])
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: next(snapshots))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: service_spec())
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: next(states))
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        rollout_tasks, "_verify_redis", lambda: (_ for _ in ()).throw(SystemExit("baseline accepted"))
    )
    with pytest.raises(SystemExit, match="baseline accepted"):
        rollout.deploy()


@pytest.mark.parametrize("active_state", ["updating", "rollback_started"])
def test_deploy_times_out_a_prior_native_update_without_writing(monkeypatch, capsys, active_state):
    _deploy_env(monkeypatch)
    clock = iter([0, 0, 11])
    monkeypatch.setattr(rollout_policy, "TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: application())
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: (active_state, PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(rollout.sys, "argv", ["verify_rollout.py", "deploy"])
    with pytest.raises(TimeoutError, match="prior Crawl4AI Swarm update did not finish"):
        rollout.main()
    assert capsys.readouterr().err == (
        "rollout failure: command=deploy check=_deploy error=TimeoutError "
        'reason="prior Crawl4AI Swarm update did not finish"\n'
    )


def test_main_keeps_the_original_failure_when_its_summary_fails(monkeypatch, capsys):
    _deploy_env(monkeypatch)

    def application_failure(*_args):
        raise ValueError("application lookup failed")

    monkeypatch.setattr(rollout_dokploy, "_application", application_failure)
    monkeypatch.setattr(
        rollout, "_public_failure", lambda *_args: (_ for _ in ()).throw(OSError("unreadable"))
    )
    monkeypatch.setattr(rollout.sys, "argv", ["verify_rollout.py", "deploy"])
    with pytest.raises(ValueError, match="application lookup failed"):
        rollout.main()
    assert capsys.readouterr().err == "rollout failure: command=deploy error=ValueError\n"


def _wire_update_outcome(monkeypatch, tmp_path, prior_state="completed"):
    state = application()
    events = []
    submitted = {"value": False}
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(
        rollout_dokploy, "_deployments", lambda *_args: [{"deploymentId": "old", "status": "done"}]
    )

    def post(url, _key, payload):
        events.append(url.rsplit("/", 1)[-1])
        if url.endswith("application.update"):
            for field in ("dockerImage", "labelsSwarm", "placementSwarm"):
                state[field] = payload[field]
        elif url.endswith("application.deploy"):
            submitted["value"] = True

    monkeypatch.setattr(rollout_http, "_post_json", post)
    monkeypatch.setattr(
        rollout_tasks, "_update_state", lambda _name: (prior_state, PRIOR_UPDATE_STARTED)
    )
    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(
        rollout_tasks, "_service_spec",
        lambda _name: service_spec(state["dockerImage"], state["labelsSwarm"]["otel.service.version"]),
    )
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(
        rollout_dokploy, "_wait_deployment", lambda *_args: {"deploymentId": "new", "status": "done"}
    )

    def verify_tasks(_app, image, *_args):
        events.append(("tasks", image))
        return {"tasks": ["1", "2", "3"], "nodes": sorted(ELIGIBLE_NODES)[:rollout_policy.REPLICAS], "instances": ["a", "b", "c"]}

    monkeypatch.setattr(rollout_tasks, "_verify_tasks", verify_tasks)
    monkeypatch.setattr(
        rollout_tasks, "_verify_public",
        lambda _revision, instances: {url: instances for url in rollout_policy.HEALTH_URLS},
    )

    def request_json(url, *_args):
        events.append(("health", url.split("?", 1)[0]))
        return health(revision="baseline" if "baseline=" in url else REVISION)

    monkeypatch.setattr(rollout_http, "_request_json", request_json)
    return submitted, events


@pytest.mark.parametrize("state", ["paused", "rollback_paused", "rollback_completed"])
def test_deploy_proves_baseline_after_a_terminal_prior_update(monkeypatch, tmp_path, state):
    _submitted, events = _wire_update_outcome(monkeypatch, tmp_path, state)

    def baseline_proved(*_args):
        raise SystemExit("baseline proved")

    monkeypatch.setattr(rollout_dokploy, "_deployments", baseline_proved)
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))

    with pytest.raises(SystemExit, match="baseline proved"):
        rollout.deploy()

    assert ("tasks", BASELINE) in events
    assert all(("health", url) in events for url in rollout_policy.HEALTH_URLS)


@pytest.mark.parametrize("state", ["paused", "rollback_paused", "rollback_completed"])
def test_deploy_rejects_a_new_failed_native_update(monkeypatch, tmp_path, capsys, state):
    submitted, events = _wire_update_outcome(monkeypatch, tmp_path)
    # This offset sorts before the old UTC string but denotes a newer instant.
    newer_started = CURRENT_UPDATE_STARTED
    monkeypatch.setattr(
        rollout_tasks, "_update_state",
        lambda _name: (state, newer_started) if submitted["value"] else ("completed", PRIOR_UPDATE_STARTED),
    )
    monkeypatch.setattr(rollout.sys, "argv", ["verify_rollout.py", "deploy"])

    with pytest.raises(rollout.withdrawals.StateError, match=f"Swarm update ended in {state}") as failure:
        rollout.main()

    assert failure.value.state == state
    assert "application.deploy" in events
    assert ("tasks", CANDIDATE) not in events
    assert capsys.readouterr().err == (
        f"rollout failure: command=deploy check=_deploy error=StateError state={state}\n"
    )


@pytest.mark.parametrize("stale_state", ["completed", "paused", "rollback_paused", "rollback_completed"])
def test_deploy_waits_past_stale_terminal_until_a_new_completed_update(
    monkeypatch, tmp_path, capsys, stale_state
):
    submitted, events = _wire_update_outcome(monkeypatch, tmp_path)
    progressed = {"value": False}

    def update_state(_name):
        if not submitted["value"]:
            return "completed", PRIOR_UPDATE_STARTED
        if not progressed["value"]:
            return stale_state, PRIOR_UPDATE_STARTED
        return "completed", CURRENT_UPDATE_STARTED

    def advance_update(_seconds):
        events.append("waited for current update")
        progressed["value"] = True

    monkeypatch.setattr(rollout_tasks, "_update_state", update_state)
    monkeypatch.setattr(rollout.time, "sleep", advance_update)

    rollout.deploy()

    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] == "new"
    assert receipt["image"] == CANDIDATE
    assert receipt["publicInstances"] == {
        url: ["a", "b", "c"] for url in rollout_policy.HEALTH_URLS
    }
    assert events.index("application.deploy") < events.index("waited for current update")
    assert events.index("waited for current update") < events.index(("tasks", CANDIDATE))


def test_deploy_accepts_a_new_completed_update_when_prior_status_was_absent(
    monkeypatch, tmp_path, capsys
):
    submitted, events = _wire_update_outcome(monkeypatch, tmp_path)
    monkeypatch.setattr(
        rollout_tasks, "_update_state",
        lambda _name: ("completed", CURRENT_UPDATE_STARTED) if submitted["value"] else ("", ""),
    )

    rollout.deploy()

    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["deploymentId"] == "new"
    assert receipt["image"] == CANDIDATE
    assert ("tasks", CANDIDATE) in events
    assert receipt["publicInstances"] == {
        url: ["a", "b", "c"] for url in rollout_policy.HEALTH_URLS
    }


@pytest.mark.parametrize("stale_state", ["completed", "paused", "rollback_paused", "rollback_completed"])
def test_deploy_times_out_when_only_stale_terminal_update_status_remains(
    monkeypatch, tmp_path, stale_state
):
    submitted, events = _wire_update_outcome(monkeypatch, tmp_path)
    clock = {"seconds": 0}
    monkeypatch.setattr(rollout_policy, "TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock["seconds"])

    def advance_clock(seconds):
        clock["seconds"] += seconds

    monkeypatch.setattr(rollout.time, "sleep", advance_clock)
    monkeypatch.setattr(
        rollout_tasks, "_update_state",
        lambda _name: (stale_state, PRIOR_UPDATE_STARTED) if submitted["value"] else ("completed", PRIOR_UPDATE_STARTED),
    )

    with pytest.raises(TimeoutError, match="Swarm update did not reach completed"):
        rollout.deploy()

    assert "application.deploy" in events
    assert ("tasks", CANDIDATE) not in events
    assert not rollout_tasks._task_proof_path().exists()



def test_deploy_rejects_a_reintroduced_cap_once_the_record_converged(monkeypatch):
    # Record already migrated to the capless shape; a capped LIVE spec is
    # drift, not transition residue.
    _deploy_env(monkeypatch)
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: application())
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: spec)
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    with pytest.raises(ValueError, match="running placement drifted"):
        rollout.deploy()


def test_deploy_accepts_capped_live_spec_when_record_matches(monkeypatch):
    _deploy_env(monkeypatch)
    state = application()
    state["placementSwarm"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: ("completed", PRIOR_UPDATE_STARTED))
    monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _name: spec)
    monkeypatch.setattr(rollout_http, "_post_json", lambda *_args: pytest.fail("no write may happen"))
    monkeypatch.setattr(
        rollout_tasks, "_verify_redis", lambda: (_ for _ in ()).throw(SystemExit("baseline spec accepted"))
    )
    with pytest.raises(SystemExit, match="baseline spec accepted"):
        rollout.deploy()
