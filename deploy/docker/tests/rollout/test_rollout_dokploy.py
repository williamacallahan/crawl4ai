import subprocess

import pytest

import rollout_policy
import rollout_http
import rollout_tasks
import rollout_dokploy
import verify_rollout as rollout
from conftest import BASELINE, application, service_spec


# --- stranded-record healing (issue #83) -------------------------------------
# Dokploy deploys FROM the record, so a deploy must move the record to the
# candidate before the service can follow. A failure after that strands the
# record on an image the service never ran; the next run then dies in
# _running_spec instead of on the real cause. The failing run must not repair
# this (see test_deploy_does_not_compensate_for_ambiguous_write) - the next one
# does, at a point where nothing is in flight.


def _heal_wiring(monkeypatch, record, running_image, running_revision, deployments):
    posts = []
    healed = dict(record)

    def post(url, _key, body):
        posts.append(body)
        healed["dockerImage"] = body["dockerImage"]
        healed["labelsSwarm"] = body["labelsSwarm"]

    monkeypatch.setattr(
        rollout_tasks, "_service_spec",
        lambda _n: service_spec(image=running_image, revision=running_revision),
    )
    monkeypatch.setattr(rollout_dokploy, "_deployments", lambda *_a: deployments)
    monkeypatch.setattr(rollout_http, "_post_json", post)
    monkeypatch.setattr(rollout_dokploy, "_application", lambda *_a: healed)
    monkeypatch.setattr(rollout_policy, "_policy", lambda _a: None)
    return posts, healed


def test_wait_deployment_accepts_one_exact_new_row(monkeypatch):
    monkeypatch.setattr(
        rollout_dokploy,
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
    result = rollout_dokploy._wait_deployment(
        "https://dokploy", "key", "app", {"old"}, "title", "description"
    )
    assert result["deploymentId"] == "new"


def test_wait_deployment_refreshes_and_checks_the_collector(monkeypatch):
    # The runner deadline tracks the caller's per-phase resets so a slow queue
    # wait within its own fresh TIMEOUT_SECONDS bound never fails a healthy
    # collector whose runner deadline the earlier arming consumed (regression
    # for the 2e49c6e7 stale-deadline bug). The collector refresh hits at entry
    # (queue admission) and on admission itself (the processing reset), and
    # the check polls every polling iteration that does not return.
    matching = {
        "deploymentId": "new",
        "title": "title",
        "description": "description",
    }
    deployments = iter(
        [
            [{**matching, "status": "running"}],
            [{**matching, "status": "running"}],
            [{**matching, "status": "done"}],
        ]
    )
    clock = [0]
    refresh_calls = []
    check_calls = []

    def advance(seconds):
        clock[0] += seconds

    class Collector:
        def refresh(self):
            refresh_calls.append(clock[0])

        def check(self):
            check_calls.append(clock[0])

    monkeypatch.setattr(rollout_policy, "TIMEOUT_SECONDS", 20)
    monkeypatch.setattr(rollout_dokploy, "_deployments", lambda *_args: next(deployments))
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rollout.time, "sleep", advance)

    result = rollout_dokploy._wait_deployment(
        "https://dokploy", "key", "app", {"old"}, "title", "description", Collector()
    )
    assert result["status"] == "done"
    # Refresh at entry (clock=0, queue admission) and on admission (clock=0,
    # processing reset): both happen before the first time.sleep.
    assert refresh_calls == [0, 0], refresh_calls
    # Check polls every iteration that does not return early: iter 1 (running,
    # not done) at clock=0, iter 2 (running, not done) at clock=5; iter 3
    # returns done before reaching the check.
    assert check_calls == [0, 5], check_calls


def test_wait_deployment_checks_the_collector_without_one(monkeypatch):
    # Without a collector the wait still works: the collector parameter is
    # optional so the existing callers and tests that omit it keep their
    # behavior.
    monkeypatch.setattr(
        rollout_dokploy,
        "_deployments",
        lambda *_args: [{"deploymentId": "new", "status": "done",
                          "title": "title", "description": "description"}],
    )
    result = rollout_dokploy._wait_deployment(
        "https://dokploy", "key", "app", {"old"}, "title", "description", None
    )
    assert result["deploymentId"] == "new"


def test_wait_deployment_resets_processing_deadline_after_admission(monkeypatch, capsys):
    matching = {
        "deploymentId": "new",
        "title": "title",
        "description": "description",
    }
    deployments = iter(
        [
            [{"deploymentId": "old", "status": "done"}],
            [{**matching, "status": "running"}],
            [{**matching, "status": "running"}],
            [{**matching, "status": "running"}],
            [{**matching, "status": "done"}],
        ]
    )
    clock = [0]

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(rollout_policy, "TIMEOUT_SECONDS", 20)
    monkeypatch.setattr(rollout_dokploy, "_deployments", lambda *_args: next(deployments))
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rollout.time, "sleep", advance)

    result = rollout_dokploy._wait_deployment(
        "https://dokploy", "key", "app", {"old"}, "title", "description"
    )

    assert result["status"] == "done"
    assert clock[0] == 20
    assert capsys.readouterr().out == (
        "stock Dokploy deployment was admitted; waiting for processing\n"
    )


def test_wait_deployment_times_out_while_queued(monkeypatch):
    clock = [0]

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(rollout_policy, "TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(
        rollout_dokploy,
        "_deployments",
        lambda *_args: [{"deploymentId": "old", "status": "done"}],
    )
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rollout.time, "sleep", advance)

    with pytest.raises(TimeoutError, match="was not admitted from the queue"):
        rollout_dokploy._wait_deployment(
            "https://dokploy", "key", "app", {"old"}, "title", "description"
        )


def test_wait_deployment_survives_a_transient_control_plane_failure(monkeypatch):
    # A 502 while polling must not abandon a deployment that is already
    # running; the deadline is what bounds the wait.
    clock = [0]
    reads = [0]

    def advance(seconds):
        clock[0] += seconds

    def deployments(*_args):
        reads[0] += 1
        if reads[0] <= 3:
            raise rollout_http.CurlError("HTTP request failed: curl exit 22", 22)
        return [{"deploymentId": "new", "status": "done",
                 "title": "title", "description": "description"}]

    monkeypatch.setattr(rollout_policy, "TIMEOUT_SECONDS", 100)
    monkeypatch.setattr(rollout_dokploy, "_deployments", deployments)
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rollout.time, "sleep", advance)

    owned = rollout_dokploy._wait_deployment(
        "https://dokploy", "key", "app", {"old"}, "title", "description"
    )
    assert owned["deploymentId"] == "new"

    # Reads that never recover still end at the bounded deadline.
    reads[0] = 0
    clock[0] = 0
    monkeypatch.setattr(
        rollout_dokploy,
        "_deployments",
        lambda *_args: (_ for _ in ()).throw(rollout_http.CurlError("502", 22)),
    )
    with pytest.raises(TimeoutError, match="was not admitted from the queue"):
        rollout_dokploy._wait_deployment(
            "https://dokploy", "key", "app", {"old"}, "title", "description"
        )


def test_wait_deployment_times_out_after_single_admission_deadline(monkeypatch, capsys):
    clock = [0]

    def advance(seconds):
        clock[0] += seconds
        if clock[0] > 10:
            raise AssertionError("processing deadline was reset repeatedly")

    monkeypatch.setattr(rollout_policy, "TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(
        rollout_dokploy,
        "_deployments",
        lambda *_args: [
            {
                "deploymentId": "new",
                "title": "title",
                "description": "description",
                "status": "running",
            }
        ],
    )
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rollout.time, "sleep", advance)

    with pytest.raises(TimeoutError, match="did not finish after queue admission"):
        rollout_dokploy._wait_deployment(
            "https://dokploy", "key", "app", set(), "title", "description"
        )

    assert clock[0] == 10
    assert capsys.readouterr().out == (
        "stock Dokploy deployment was admitted; waiting for processing\n"
    )


def test_wait_deployment_fails_closed_on_foreign_row(monkeypatch):
    monkeypatch.setattr(
        rollout_dokploy,
        "_deployments",
        lambda *_args: [{"deploymentId": "foreign", "title": "other"}],
    )
    with pytest.raises(RuntimeError, match="ambiguous"):
        rollout_dokploy._wait_deployment(
            "https://dokploy", "key", "app", set(), "title", "description"
        )


def test_wait_deployment_fails_closed_on_conflicting_row_after_admission(monkeypatch):
    matching = {
        "deploymentId": "new",
        "title": "title",
        "description": "description",
        "status": "running",
    }
    deployments = iter(
        [
            [matching],
            [matching, {"deploymentId": "foreign", "title": "other"}],
        ]
    )
    monkeypatch.setattr(rollout_dokploy, "_deployments", lambda *_args: next(deployments))
    monkeypatch.setattr(rollout.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="ambiguous"):
        rollout_dokploy._wait_deployment(
            "https://dokploy", "key", "app", set(), "title", "description"
        )


def test_heal_rewrites_a_record_the_service_never_ran(monkeypatch):
    record = application()
    record["dockerImage"] = "registry.example/crawl4ai@sha256:strandedcandidate"
    posts, healed = _heal_wiring(monkeypatch, record, BASELINE, "baseline", [])
    out = rollout_dokploy._heal_stranded_record("https://dokploy", "key", "app", "svc", record)
    assert posts == [{
        "applicationId": "app",
        "dockerImage": BASELINE,
        "labelsSwarm": rollout_policy._labels("baseline"),
    }]
    assert out["dockerImage"] == BASELINE


def test_heal_is_a_noop_when_the_record_already_matches(monkeypatch):
    record = application()
    posts, _ = _heal_wiring(monkeypatch, record, record["dockerImage"], "baseline", [])
    assert rollout_dokploy._heal_stranded_record("https://dokploy", "key", "app", "svc", record) is record
    assert posts == [], "an aligned record must not be rewritten"


def test_heal_refuses_while_a_dokploy_deployment_is_running(monkeypatch):
    # A live deploy means the record is ahead on purpose, not stranded.
    record = application()
    record["dockerImage"] = "registry.example/crawl4ai@sha256:strandedcandidate"
    posts, _ = _heal_wiring(
        monkeypatch, record, BASELINE, "baseline", [{"status": "running"}]
    )
    with pytest.raises(RuntimeError, match="nonterminal Dokploy deployment"):
        rollout_dokploy._heal_stranded_record("https://dokploy", "key", "app", "svc", record)
    assert posts == [], "must not rewrite the record mid-deploy"


def test_heal_refuses_a_mutable_running_artifact(monkeypatch):
    record = application()
    record["dockerImage"] = "registry.example/crawl4ai@sha256:strandedcandidate"
    posts, _ = _heal_wiring(
        monkeypatch, record, "registry.example/crawl4ai:latest", "baseline", []
    )
    with pytest.raises(ValueError, match="no immutable artifact"):
        rollout_dokploy._heal_stranded_record("https://dokploy", "key", "app", "svc", record)
    assert posts == []


# --- rollback-registry source (issue #83) -----------------------------------
# Dokploy tags the currently-running service image on the Dokploy host before it
# touches the service. When that image is absent the tag fails, the deployment
# ends "error", and the Swarm service is never updated at all.


def _docker_image_calls(inspect_rc):
    calls = []

    def run(cmd, *a, **kw):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, inspect_rc, "", "")
        if cmd[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command {cmd}")

    return run, calls


def test_ensure_rollback_source_pulls_when_the_host_lacks_the_image(monkeypatch):
    run, calls = _docker_image_calls(inspect_rc=1)
    monkeypatch.setattr(rollout_dokploy.subprocess, "run", run)
    rollout_dokploy._ensure_rollback_source("registry.example/crawl4ai@sha256:deadbeef")
    assert ["docker", "pull", "registry.example/crawl4ai@sha256:deadbeef"] in calls


def test_ensure_rollback_source_does_not_pull_when_already_present(monkeypatch):
    run, calls = _docker_image_calls(inspect_rc=0)
    monkeypatch.setattr(rollout_dokploy.subprocess, "run", run)
    rollout_dokploy._ensure_rollback_source("registry.example/crawl4ai@sha256:deadbeef")
    assert not any(c[:2] == ["docker", "pull"] for c in calls)


def test_ensure_rollback_source_reports_why_the_pull_failed(monkeypatch):
    # A bare CalledProcessError cannot distinguish "not logged in" from
    # "manifest unknown"; the deploy job needs to know which.
    def run(cmd, *a, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "denied: authentication required\n")

    monkeypatch.setattr(rollout_dokploy.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="authentication required"):
        rollout_dokploy._ensure_rollback_source("registry.example/crawl4ai@sha256:deadbeef")
