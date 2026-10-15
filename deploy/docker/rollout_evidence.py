"""Public rollout monitor and final replica evidence for one deployment."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import rollout_withdrawals as withdrawals
import rollout_dokploy
import rollout_http
import rollout_policy
import rollout_tasks


def monitor() -> None:
    evidence = Path(os.environ["ROLLOUT_MONITOR_PATH"])
    armed = Path(os.environ["ROLLOUT_MONITOR_ARMED_PATH"])
    stop = Path(os.environ["ROLLOUT_MONITOR_STOP_PATH"])
    evidence.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    first_round = True
    with evidence.open("a") as output:
        while not stop.exists():
            for url in rollout_policy.HEALTH_URLS:
                probe_id = str(uuid.uuid4())
                started_at = time.time()
                started_monotonic = time.monotonic()
                probe: dict[str, Any] = {"probe_id": probe_id}
                try:
                    health = rollout_http._request_json(f"{url}?rollout={probe_id}", probe=probe)
                    if (
                        health.get("revision") == os.environ.get("GITHUB_SHA")
                        and not probe.get("correlation_verified")
                    ):
                        probe.setdefault("correlation_error", "candidate probe echo missing")
                    sample = {
                        "ok": (
                            rollout_tasks._exact_health(health)
                            and not probe.get("correlation_error")
                            and not probe.get("timing_error")
                        ),
                        "revision": health.get("revision"),
                        "instance": health.get("instance"),
                    }
                except subprocess.TimeoutExpired:
                    sample = {
                        "ok": False,
                        "error": "TimeoutExpired: curl wrapper exceeded 20 seconds",
                        "curl_exit": None,
                    }
                except Exception as error:
                    sample = {
                        "ok": False,
                        "error": (
                            re.sub(r"https?://\S+", "<url>", f"CurlError: {error}")[:300]
                            if isinstance(error, rollout_http.CurlError)
                            else type(error).__name__
                        ),
                        "curl_exit": getattr(error, "curl_exit", None),
                    }
                sample.update({
                    "url": url,
                    "started_at": started_at,
                    "timestamp": time.time(),
                    "elapsed_ms": round((time.monotonic() - started_monotonic) * 1_000, 3),
                    **probe,
                })
                failures += not sample["ok"]
                output.write(json.dumps(sample, separators=(",", ":")) + "\n")
                output.flush()
            if first_round:
                if failures:
                    raise RuntimeError("public rollout baseline is not ready")
                armed.touch()
                first_round = False
            time.sleep(0.5)
    if failures:
        raise RuntimeError(f"public rollout monitor recorded {failures} failures")


def evidence() -> None:
    rows = [
        json.loads(line)
        for line in Path(os.environ["ROLLOUT_MONITOR_PATH"])
        .read_text()
        .splitlines()
        if line
    ]
    if not rows or any(not row.get("ok") for row in rows):
        raise RuntimeError("public rollout evidence contains a failure")
    revision = os.environ["GITHUB_SHA"]
    task_proof = json.loads(rollout_tasks._task_proof_path().read_text())
    expected = set(task_proof.get("instances", []))
    public_instances = task_proof.get("publicInstances") or {}
    baseline_revision = task_proof.get("baselineRevision")
    if (
        task_proof.get("revision") != revision
        or not isinstance(baseline_revision, str)
        or not baseline_revision
        or len(expected) != rollout_policy.REPLICAS
    ):
        raise RuntimeError("authoritative task proof does not match the candidate")
    if set(public_instances) != set(rollout_policy.HEALTH_URLS) or any(
        set(public_instances.get(url, [])) != expected for url in rollout_policy.HEALTH_URLS
    ):
        raise RuntimeError("each public domain must observe every authoritative task")
    base = os.environ["DOKPLOY_URL"]
    api_key = os.environ["DOKPLOY_API_KEY"]
    application_id = os.environ["APPLICATION_ID"]
    application = rollout_dokploy._application(base, api_key, application_id)
    rollout_policy._policy(application)
    image = str(task_proof.get("image", ""))
    placement = task_proof.get("placement")
    if not rollout_dokploy._record_converged(application, image, revision, placement):
        raise RuntimeError("candidate metadata changed before final evidence")
    app_name = str(application["appName"])
    rollout_policy._running_spec(
        rollout_tasks._service_spec(app_name), image, rollout_policy._labels(revision), placements=(placement,)
    )
    rollout_tasks.verify_route(base, api_key, application_id, app_name)
    eligibility = rollout_tasks._eligible_nodes()
    recorded_task_proof = {
        key: task_proof.get(key) for key in ("tasks", "nodes", "instances")
    }
    # The proof file's baselineRevision records whether this run was a no-op
    # rerun (candidate already the baseline, so baselineRevision == revision)
    # or a real start-first rollout (a different baseline). Only a real rollout
    # owes the strict converged proof; a no-op re-proves the live SHA in place.
    converged = task_proof.get("rolloutSubmitted", baseline_revision != revision)
    if converged:
        journal = task_proof.get("withdrawals")
        if not isinstance(journal, dict):
            raise RuntimeError("real rollout has no exact withdrawal journal")
        current_census = withdrawals.census(app_name)
        withdrawals.validate(journal, app_name, task_proof.get("baselineTasks"), current_census)
        if (
            sorted(task["task"][:12] for task in current_census) != sorted(task_proof["tasks"])
            or sorted(task["container"][:12] for task in current_census) != sorted(task_proof["instances"])
            or set(withdrawals.ready_nodes()) != set(journal["nodes"])
        ):
            raise RuntimeError("saved withdrawal journal differs from the candidate census")
    if rollout_tasks._verify_tasks(
        app_name, image, revision, eligibility, converged, placement
    ) != recorded_task_proof:
        raise RuntimeError("Crawl4AI task census changed before final evidence")
    current_public_instances = rollout_tasks._verify_public(revision, sorted(expected))
    rollout_tasks.verify_route(base, api_key, application_id, app_name)
    if rollout_tasks._verify_tasks(
        app_name, image, revision, eligibility, converged, placement
    ) != recorded_task_proof:
        raise RuntimeError("Crawl4AI task census changed during final evidence")
    observed_urls = set()
    for row in rows:
        observed_revision = row.get("revision")
        if observed_revision not in {baseline_revision, revision}:
            raise RuntimeError("public rollout evidence contains a foreign revision")
        if observed_revision != revision:
            continue
        if row.get("instance") not in expected:
            raise RuntimeError("public rollout evidence contains an unknown candidate instance")
        observed_urls.add(row.get("url"))
    if not set(rollout_policy.HEALTH_URLS) <= observed_urls:
        raise RuntimeError("each public domain must observe the candidate revision")
    print(
        json.dumps(
            {
                "publicSuccesses": len(rows),
                "publicFailures": 0,
                "publicInstances": current_public_instances,
            },
            separators=(",", ":"),
        )
    )
