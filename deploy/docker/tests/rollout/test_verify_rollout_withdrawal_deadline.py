import copy

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import verify_rollout as rollout
from conftest import BASELINE, ELIGIBLE_NODES, REVISION, _deploy_env, application, health, service_spec


def test_deploy_refreshes_and_checks_withdrawal_deadline_at_each_phase(
    monkeypatch, tmp_path
):
    # Regression for the 2e49c6e7 stale-deadline bug: a phase within its own
    # fresh TIMEOUT_SECONDS bound must not fail a healthy collector for the
    # time the prior phases consumed. The caller now refreshes the runner
    # deadline at every per-phase boundary (queue admission, processing reset,
    # Swarm-update wait, post-update proofs, finish barrier) and checks inside
    # the wait loops so a collector death surfaces in the phase that owns it.
    calls = {"refresh": 0, "check": 0, "init_args": None}

    class Collector:
        nodes = {}

        def __init__(self, *_args, **_kwargs):
            calls["init_args"] = (_args, _kwargs)

        def start(self):
            return []

        def finish(self, _candidate):
            return None

        def refresh(self):
            calls["refresh"] += 1

        def check(self):
            calls["check"] += 1

    state = application()  # baseline running; candidate not yet deployed
    _deploy_env(monkeypatch)
    monkeypatch.setenv("ROLLOUT_MONITOR_PATH", str(tmp_path / "monitor.jsonl"))

    def post(url, _key, payload):
        if url.endswith("application.update"):
            state["dockerImage"] = payload["dockerImage"]
            state["labelsSwarm"] = payload["labelsSwarm"]
            state["placementSwarm"] = payload["placementSwarm"]

    monkeypatch.setattr(rollout.withdrawals, "Withdrawals", Collector)
    monkeypatch.setattr(rollout.withdrawals, "census", lambda _app: [])
    monkeypatch.setattr(rollout.withdrawals, "ready_nodes", lambda: {})
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_args: copy.deepcopy(state))
    # _deploy builds title from GITHUB_RUN_ID/GITHUB_RUN_ATTEMPT/REVISION
    # (_deploy_env sets the first two to "1"); description from candidate and
    # baseline. _wait_deployment's matching row must carry the same strings.
    matching = {
        "deploymentId": "new",
        "title": f"crawl4ai-1-1-{REVISION}",
        "description": (
            f"candidate=registry.example/crawl4ai@sha256:candidate;baseline={BASELINE}"
        ),
    }
    # _deploy's prior-row check sees only terminal rows; _wait_deployment's
    # poll then sees the matching row progress running -> done.
    deployments = iter(
        [
            [{"deploymentId": "old", "status": "done"}],
            [{**matching, "status": "running"}],
            [{**matching, "status": "done"}],
        ]
    )
    monkeypatch.setattr(rollout_dokploy, "_deployments", lambda *_args: next(deployments))
    monkeypatch.setattr(rollout_http, "_post_json", post)

    # The prior-update wait takes the first "completed" so it skips its loop.
    # The next call captures prior_started_at before the application.update POST.
    # The Swarm-update wait then takes "updating" (one iteration that calls
    # check()) and "completed" (loop exits); both use a started_at that differs
    # from the prior one so current_update is True.
    update_states = iter([
        ("completed", ""),
        ("", "2026-01-01T00:00:00Z"),
        ("updating", "2026-01-01T00:01:00Z"),
        ("completed", "2026-01-01T00:01:00Z"),
    ])
    monkeypatch.setattr(rollout_tasks, "_update_state", lambda _name: next(update_states))

    monkeypatch.setattr(rollout_tasks, "_eligible_nodes", lambda: (ELIGIBLE_NODES, ELIGIBLE_NODES))
    monkeypatch.setattr(
        rollout_tasks, "_service_spec",
        lambda _name: service_spec(state["dockerImage"], state["labelsSwarm"]["otel.service.version"]),
    )
    monkeypatch.setattr(rollout_tasks, "_verify_redis", lambda: None)
    monkeypatch.setattr(rollout_tasks, "verify_route", lambda *_args: None)
    monkeypatch.setattr(rollout_tasks, "_verify_tasks", lambda *_args: {
        "tasks": ["1", "2", "3"], "nodes": ["haiku-5", "haiku-6", "haiku-9"],
        "instances": ["a", "b", "c"]})
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
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)

    rollout.deploy()

    args, _kw = calls["init_args"]
    # The collector is constructed with the runner per-phase budget AND the
    # collect_seconds sized to outlive every phase it spans.
    assert args[-2:] == (rollout_policy.TIMEOUT_SECONDS,
                         rollout_policy.WITHDRAWAL_COLLECT_SECONDS), args
    # refresh() fires at every caller phase boundary — five lower bound:
    # queue admission (inside _wait_deployment), the processing reset on
    # admission (inside _wait_deployment), the Swarm-update wait (verify_rollout
    # _deploy), the post-update proofs phase, and the finish barrier.
    assert calls["refresh"] >= 5, calls
    # check() polls inside the wait loops: the queue/processing loop in
    # _wait_deployment and the Swarm-update wait loop in _deploy both call it,
    # so the failure surfaces in the phase that owns the slow wait rather than
    # at the finish barrier.
    assert calls["check"] >= 2, calls
