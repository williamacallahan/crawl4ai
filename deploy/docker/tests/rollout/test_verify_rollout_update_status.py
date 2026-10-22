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
