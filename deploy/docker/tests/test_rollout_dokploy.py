import subprocess

import pytest

import rollout_dokploy


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
